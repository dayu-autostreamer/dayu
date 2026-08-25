import abc
import concurrent.futures
import copy
import multiprocessing
import threading
import time
import weakref

from core.lib.common import (
    ClassFactory,
    ClassType,
    ConfigLoader,
    Context,
    GlobalInstanceManager,
    LOGGER,
    TaskConstant,
)
from core.lib.estimation import OverheadEstimator
from core.lib.scheduling import (
    END,
    START,
    materialize_offloading_plan,
    topological_order,
)

from .base_agent import BaseAgent
from .fragsplice import (
    FragSpliceLatencyModel,
    FragSpliceOptimizer,
    FragSpliceRandomInputOptimizer,
    FragSpliceRandomLatencyModel,
    FragSpliceStagewiseEFTOptimizer,
)

__all__ = ("FragSpliceAgent",)


_SYSTEM_INSTANCE_TOKENS = weakref.WeakKeyDictionary()
_SYSTEM_INSTANCE_TOKENS_LOCK = threading.RLock()


def _warm_rolling_worker():
    """Start the persistent planner process before its first real search."""

    return True


def _solve_rolling_plan(payload):
    """Solve one future decision in an isolated worker process.

    Only JSON/pickle-friendly snapshots cross the process boundary.  The
    request-serving process therefore never shares a mutable latency model or
    optimizer with the CPU-bound stochastic search.
    """

    if payload["distribution_profiler_enabled"]:
        planning_model = FragSpliceLatencyModel(
            profile=payload["latency_profile"],
            residual_half_life_tasks=payload["residual_half_life_tasks"],
        )
        optimizer_cls = (
            FragSpliceOptimizer
            if payload["plan_optimizer_enabled"]
            else FragSpliceStagewiseEFTOptimizer
        )
    else:
        planning_model = FragSpliceRandomLatencyModel(
            state=payload["random_latency_state"]
        )
        optimizer_cls = FragSpliceRandomInputOptimizer
    optimizer = optimizer_cls(planning_model, **payload["optimizer_parameters"])
    return optimizer.solve(
        payload["info"],
        payload["snapshot"],
        payload["deployment"],
        payload["cloud_device"],
        initial_plan=payload.get("initial_plan"),
    )


def _load_mapping(value, label):
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        loaded = ConfigLoader.load(Context.get_file_path(value))
        if isinstance(loaded, dict):
            return loaded
    raise TypeError(f"FragSplice {label} must be a mapping or mounted file path")


def _system_instance_token(system):
    """Return a stable per-System token for shared FragSplice state.

    ``id(system)`` can be reused after a short-lived System object is garbage
    collected, which may attach a later experiment to incompatible cached
    state. Keep this extension-owned identity outside the framework object.
    """

    with _SYSTEM_INSTANCE_TOKENS_LOCK:
        token = _SYSTEM_INSTANCE_TOKENS.get(system)
        if token is None:
            token = object()
            _SYSTEM_INSTANCE_TOKENS[system] = token
        return token


@ClassFactory.register(ClassType.SCH_AGENT, alias="fragsplice")
class FragSpliceAgent(BaseAgent, abc.ABC):
    """Commitment-aware full-DAG offloading under a fixed deployment."""

    DISTRIBUTION_PROFILER_ENABLED = True
    FUTURE_STATE_ESTIMATOR_ENABLED = True
    PLAN_OPTIMIZER_ENABLED = True
    OPTIMIZER_CLS = FragSpliceOptimizer

    def __init__(
        self,
        system,
        agent_id: int,
        configuration=None,
        latency_profile=None,
        latency_slo_s=2.5,
        scenario_count=32,
        max_scenarios=256,
        search_time_limit_s=0.0,
        random_seed=0,
        random_invocation_cost_min_s=0.0,
        random_invocation_cost_max_s=2.0,
        random_overhead_cost_min_s=0.0,
        random_overhead_cost_max_s=0.2,
        random_workload_token_min=0,
        random_workload_token_max=8,
        queue_state_max_age_s=1.5,
        residual_half_life_tasks=8.0,
        incumbent_neighborhood_size=4,
        screening_beam_width=16,
        rolling_planner_enabled=False,
        rolling_plan_max_age_s=0.75,
        rolling_plan_max_lag_tasks=4,
        rolling_reservation_wait_s=0.20,
        rolling_use_process=True,
        decision_log_interval=100,
    ):
        super().__init__(system, agent_id)
        self.agent_id = agent_id
        self.configuration = _load_mapping(configuration, "configuration")
        self.distribution_profiler_enabled = bool(
            self.DISTRIBUTION_PROFILER_ENABLED
        )
        revision = system.runtime_directory_revision()
        if self.distribution_profiler_enabled:
            profile = _load_mapping(latency_profile, "latency_profile")
            profile_context = FragSpliceLatencyModel.validate_profile_context(
                profile,
                self.configuration,
            )
            self.latency_model = GlobalInstanceManager.get_instance(
                FragSpliceLatencyModel,
                ("fragsplice", _system_instance_token(system), revision),
                profile=profile,
                residual_half_life_tasks=residual_half_life_tasks,
            )
            if profile_context is not None:
                self.latency_model.ensure_profile_context(
                    **profile_context,
                    require_complete=True,
                )
            else:
                self.latency_model.ensure_profile_context(
                    configuration=self.configuration,
                )
        else:
            if latency_profile is not None:
                raise ValueError(
                    "FragSplice no-distribution-profiler must not receive a "
                    "latency_profile"
                )
            self.latency_model = GlobalInstanceManager.get_instance(
                FragSpliceRandomLatencyModel,
                (
                    "fragsplice-random-input",
                    _system_instance_token(system),
                    revision,
                ),
                random_invocation_cost_min_s=(
                    random_invocation_cost_min_s
                ),
                random_invocation_cost_max_s=(
                    random_invocation_cost_max_s
                ),
                random_overhead_cost_min_s=random_overhead_cost_min_s,
                random_overhead_cost_max_s=random_overhead_cost_max_s,
                random_seed=random_seed,
            )
        self.plan_optimizer_enabled = bool(
            self.PLAN_OPTIMIZER_ENABLED
        )
        optimizer_parameters = {
            "default_slo_s": latency_slo_s,
            "scenario_count": scenario_count,
            "max_scenarios": max_scenarios,
            "search_time_limit_s": search_time_limit_s,
            "random_seed": random_seed,
            "queue_state_max_age_s": queue_state_max_age_s,
            "incumbent_neighborhood_size": incumbent_neighborhood_size,
            "screening_beam_width": screening_beam_width,
        }
        if not self.distribution_profiler_enabled:
            optimizer_parameters.update({
                "random_workload_token_min": random_workload_token_min,
                "random_workload_token_max": random_workload_token_max,
            })
        self.optimizer = self.OPTIMIZER_CLS(
            self.latency_model,
            **optimizer_parameters,
        )
        self._optimizer_parameters = copy.deepcopy(optimizer_parameters)
        self._residual_half_life_tasks = float(residual_half_life_tasks)
        self.overhead_estimator = OverheadEstimator(
            "FragSplice",
            "scheduler/fragsplice",
            agent_id=agent_id,
            write_file=False,
            log_each=False,
        )
        self.background_overhead_estimator = OverheadEstimator(
            "FragSpliceBackground",
            "scheduler/fragsplice",
            agent_id=agent_id,
            write_file=False,
            log_each=False,
        )
        self.future_state_estimator_enabled = bool(
            self.FUTURE_STATE_ESTIMATOR_ENABLED
        )
        self._lock = threading.RLock()
        self._optimizer_lock = threading.Lock()
        self.last_decision = None
        self.rolling_planner_enabled = bool(rolling_planner_enabled)
        self.rolling_plan_max_age_s = max(
            0.05, float(rolling_plan_max_age_s)
        )
        self.rolling_plan_max_lag_tasks = max(
            0, int(rolling_plan_max_lag_tasks)
        )
        self.rolling_reservation_wait_s = max(
            0.0, float(rolling_reservation_wait_s)
        )
        self.rolling_use_process = bool(rolling_use_process)
        self.decision_log_interval = max(1, int(decision_log_interval))
        self._rolling_condition = threading.Condition()
        self._rolling_pending = None
        self._rolling_cache = None
        self._rolling_worker = None
        self._rolling_executor = None
        self._rolling_generation = 0
        self._rolling_last_consumed_generation = 0
        self._decision_count = 0
        self._background_decision_count = 0
        self._feedback_count = 0
        # A compact request-process view of task-bound plans that have been
        # published but whose root task has not completed yet.  The exact
        # future-state engine remains the background authority.  This index is
        # intentionally much smaller than an execution calendar: it only lets
        # the millisecond-scale request path account for committed-but-not-
        # ready work while a new background BnB result is being computed.
        self._rolling_active_plans = {}
        # ``Scheduler.update_scheduler_resource`` already fan-outs every fresh
        # monitor report to the live agent.  Retain only the small queue-state
        # projection needed by the request path instead of deep-copying the
        # complete global resource table for every arriving task.
        self._rolling_resource_state = {}
        self._validated_deployment = None
        self._validated_revision = None
        if (
            self.distribution_profiler_enabled
            and not self.latency_model.has_samples()
        ):
            LOGGER.warning(
                "[FragSplice] No cold-start samples were loaded. Plan optimization "
                "will remain unavailable until every deployed pair has a valid sample."
            )
        elif not self.distribution_profiler_enabled:
            LOGGER.info(
                "[FragSplice] Distribution profiler disabled: timing inputs "
                "use Uniform[%.3f, %.3f]s, invocation tokens use "
                "UniformInt[%s, %s], and no live queue, commitment, profile, "
                "or completed-task history is exposed to planning.",
                random_invocation_cost_min_s,
                random_invocation_cost_max_s,
                random_workload_token_min,
                random_workload_token_max,
            )

    @staticmethod
    def _dag_signature(dag):
        return tuple(
            (
                str(name),
                tuple(sorted(str(item) for item in node.get("prev_nodes", []))),
                tuple(sorted(str(item) for item in node.get("next_nodes", []))),
            )
            for name, node in sorted((dag or {}).items())
        )

    @staticmethod
    def _snapshot_contains_root(snapshot, root_uuid):
        if not root_uuid:
            return True
        for field in ("reservations", "commitments"):
            for record in snapshot.get(field, []) or []:
                if str(record.get("root_uuid") or "") == root_uuid:
                    return True
        return False

    @staticmethod
    def _plan_supported(plan, deployment):
        if not isinstance(plan, dict) or not isinstance(deployment, dict):
            return False
        for service, device in plan.items():
            devices = deployment.get(service, [])
            if isinstance(devices, str):
                devices = [devices]
            if str(device) not in {str(item) for item in devices}:
                return False
        return bool(plan)

    def _snapshot_for_planning(self, snapshot):
        if not self.future_state_estimator_enabled:
            snapshot = copy.deepcopy(snapshot)
            snapshot["reservations"] = []
            snapshot["commitments"] = []
            snapshot["task_barriers"] = []
        return snapshot

    def _planning_snapshot(self):
        return self._snapshot_for_planning(
            self.system.get_scheduling_snapshot()
        )

    def _deployment_for_snapshot(self, snapshot):
        deployment = snapshot.get("deployment")
        if not isinstance(deployment, dict):
            raise ValueError("FragSplice scheduling snapshot has no deployment")
        return copy.deepcopy(deployment)

    def _validate_planning_context(self, dag, deployment):
        if self.distribution_profiler_enabled:
            self.latency_model.ensure_profile_context(
                configuration=self.configuration,
                deployment=deployment,
                dag=dag,
                require_complete=True,
            )
            self._validate_profile_coverage(dag, deployment)

    def _solve_now(self, decision_info, snapshot, deployment, initial_plan):
        self._validate_planning_context(decision_info["dag"], deployment)
        with self._optimizer_lock:
            return self.optimizer.solve(
                decision_info,
                snapshot,
                deployment,
                self.cloud_device,
                initial_plan=initial_plan,
            )

    def _ensure_rolling_worker(self):
        if not self.rolling_planner_enabled:
            return
        with self._rolling_condition:
            if self._rolling_worker is not None and self._rolling_worker.is_alive():
                return
            if self.rolling_use_process and self._rolling_executor is None:
                context = multiprocessing.get_context("spawn")
                self._rolling_executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                )
                # Process startup and module import cost roughly one source
                # interval on the current platform.  Warm it while the first
                # request performs its unavoidable synchronous bootstrap.
                self._rolling_executor.submit(_warm_rolling_worker)
            self._rolling_worker = threading.Thread(
                target=self._rolling_loop,
                name=f"fragsplice-rolling-{self.agent_id}",
                daemon=True,
            )
            self._rolling_worker.start()

    def _enqueue_rolling_plan(self, info, plan, after_root, revision):
        if not self.rolling_planner_enabled or not after_root:
            return
        self._ensure_rolling_worker()
        with self._rolling_condition:
            self._rolling_generation += 1
            generation = self._rolling_generation
            self._rolling_pending = {
                "generation": generation,
                "info": copy.deepcopy(info),
                "initial_plan": copy.deepcopy(plan),
                "after_root": str(after_root),
                "revision": int(revision or 0),
            }
            with self._lock:
                record = {
                    "revision": int(revision or 0),
                    "source_id": info.get("source_id"),
                    "dag_signature": self._dag_signature(info.get("dag")),
                    "plan": copy.deepcopy(plan),
                }
                self._rolling_active_plans[str(after_root)] = record
            self._rolling_condition.notify_all()

    def _wait_for_committed_snapshot(self, job):
        deadline = time.monotonic() + self.rolling_reservation_wait_s
        # The schedule response is reserved immediately after get_schedule_plan
        # returns.  Waiting in the background prevents a phantom plan from
        # being projected before that task-bound reservation exists.
        if self.rolling_reservation_wait_s > 0.0:
            time.sleep(min(0.01, self.rolling_reservation_wait_s))
        while True:
            # Validate that the plan was really committed against the raw
            # scheduler snapshot.  The current-state ablation deliberately
            # removes commitments from the optimizer input, but that must not
            # make the rolling worker wait for evidence it has just hidden.
            snapshot = self.system.get_scheduling_snapshot()
            if self._snapshot_contains_root(snapshot, job["after_root"]):
                return self._snapshot_for_planning(snapshot)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def _rolling_payload(self, job, snapshot, deployment):
        info = copy.deepcopy(job["info"])
        task_context = info.get("task_context")
        task_context = task_context if isinstance(task_context, dict) else {}
        next_task_id = task_context.get("task_id")
        try:
            next_task_id = int(next_task_id) + 1
        except (TypeError, ValueError):
            next_task_id = "next"
        info["task_context"] = {
            "source_id": info.get("source_id"),
            "task_id": next_task_id,
            "root_uuid": (
                f"__fragsplice_next__:{self.agent_id}:"
                f"{job['generation']}"
            ),
        }
        payload = {
            "residual_half_life_tasks": self._residual_half_life_tasks,
            "distribution_profiler_enabled": self.distribution_profiler_enabled,
            "plan_optimizer_enabled": self.plan_optimizer_enabled,
            "optimizer_parameters": copy.deepcopy(self._optimizer_parameters),
            "info": info,
            "snapshot": snapshot,
            "deployment": deployment,
            "cloud_device": self.cloud_device,
            "initial_plan": job.get("initial_plan"),
        }
        if self.distribution_profiler_enabled:
            payload["latency_profile"] = self.latency_model.to_profile(
                deployment=deployment
            )
        else:
            payload["random_latency_state"] = self.latency_model.to_state()
        return payload

    def _rolling_loop(self):
        while True:
            with self._rolling_condition:
                while self._rolling_pending is None:
                    self._rolling_condition.wait()
                job = self._rolling_pending
                self._rolling_pending = None
            try:
                snapshot = self._wait_for_committed_snapshot(job)
                if snapshot is None:
                    LOGGER.debug(
                        "[FragSpliceRolling] Skip root=%s because no task-bound "
                        "reservation became visible.",
                        job["after_root"],
                    )
                    continue
                revision = int(snapshot.get("runtime_directory_revision") or 0)
                if job["revision"] and revision != job["revision"]:
                    continue
                deployment = self._deployment_for_snapshot(snapshot)
                self._validate_planning_context(job["info"]["dag"], deployment)
                payload = self._rolling_payload(job, snapshot, deployment)
                started = time.monotonic()
                with self.background_overhead_estimator:
                    if self._rolling_executor is None:
                        result = _solve_rolling_plan(payload)
                    else:
                        result = self._rolling_executor.submit(
                            _solve_rolling_plan, payload
                        ).result()
                background_seconds = time.monotonic() - started
                cache = {
                    "generation": job["generation"],
                    "revision": revision,
                    "source_id": job["info"].get("source_id"),
                    "dag_signature": self._dag_signature(job["info"]["dag"]),
                    "deployment": deployment,
                    "result": copy.deepcopy(result),
                    "published_monotonic": time.monotonic(),
                    "background_seconds": background_seconds,
                }
                with self._lock:
                    # A bounded-lag consumer may use this result even when one
                    # or two arrivals occurred while the search was running.
                    # Generation and age guards in ``_cached_result`` prevent
                    # an old result from being reused indefinitely.
                    self._rolling_cache = cache
                    self._background_decision_count += 1
                    background_count = self._background_decision_count
                if (
                    background_count == 1
                    or background_count % self.decision_log_interval == 0
                    or background_seconds > 1.0
                ):
                    LOGGER.info(
                        "[FragSpliceRolling] count=%s after_root=%s "
                        "generation=%s background=%.4fs plan=%s",
                        background_count,
                        job["after_root"],
                        job["generation"],
                        background_seconds,
                        result.get("plan"),
                    )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                LOGGER.exception(
                    "[FragSpliceRolling] Background planning failed for root=%s: %s",
                    job.get("after_root"),
                    exc,
                )

    def _cached_result(self, info, revision):
        if not self.rolling_planner_enabled:
            return None, None, None
        with self._lock:
            # Published rolling results are immutable: the producer replaces
            # the whole cache instead of mutating it.  Validate the shared
            # reference while holding the lock, then copy only the result that
            # this request will actually consume.  Copying the whole cache in
            # the request path duplicated every candidate plan and sometimes
            # held the lock long enough to miss a micro-burst input slot.
            cache = self._rolling_cache
            if not isinstance(cache, dict):
                return None, None, None
            generation = int(cache.get("generation") or 0)
            if generation <= self._rolling_last_consumed_generation:
                return None, None, None
            lag_tasks = max(0, self._rolling_generation - generation)
            if lag_tasks > self.rolling_plan_max_lag_tasks:
                return None, None, None
            age = max(
                0.0, time.monotonic() - cache["published_monotonic"]
            )
            if age > self.rolling_plan_max_age_s:
                return None, None, None
            if int(cache.get("revision") or 0) != int(revision or 0):
                return None, None, None
            if cache.get("source_id") != info.get("source_id"):
                return None, None, None
            if cache.get("dag_signature") != self._dag_signature(
                info.get("dag")
            ):
                return None, None, None
            result = cache.get("result")
            deployment = cache.get("deployment")
            if not isinstance(result, dict) or not self._plan_supported(
                result.get("plan"), deployment
            ):
                return None, None, None
            # Consume each completed search at most once. Reusing a newly
            # computed plan for every arrival until the next search finishes
            # recreates the replica concentration that the rolling planner is
            # intended to avoid under bursty load.
            self._rolling_last_consumed_generation = generation
        result = copy.deepcopy(result)
        deployment = copy.deepcopy(deployment)
        result["planner_mode"] = "rolling_cache"
        result["rolling_cache_lag_tasks"] = lag_tasks
        result["background_search_seconds"] = cache.get(
            "background_seconds", 0.0
        )
        return result, deployment, age

    @staticmethod
    def _reuse_result(last_decision):
        result = copy.deepcopy(last_decision)
        result["evaluated"] = []
        result["expanded"] = 0
        result["optimality_proven"] = False
        result["budget_exhausted"] = True
        result["score_evaluated"] = False
        result["prediction_complete"] = False
        result["fallback_reason"] = "rolling_plan_not_ready"
        result["planner_mode"] = "rolling_fallback"
        result["background_search_seconds"] = 0.0
        result["search_seconds"] = 0.0
        result["state_build_seconds"] = 0.0
        result["budget_overrun_seconds"] = 0.0
        return result

    def _rolling_plan_pool(self, result, deployment):
        """Return a bounded set of complete plans and their base screen scores."""

        selected = result.get("plan") if isinstance(result, dict) else None
        if not isinstance(selected, dict):
            return []
        required = set(selected)
        raw = list(result.get("candidate_pool") or [])
        raw.append({"plan": selected, "screen_score": result.get("score")})
        for item in result.get("evaluated") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                raw.append({"plan": item[1], "screen_score": None})

        entries = []
        entry_by_key = {}

        def normalize_score(score):
            if not isinstance(score, (list, tuple)) or len(score) < 5:
                return None
            try:
                normalized = tuple(float(value) for value in score[:5])
            except (TypeError, ValueError):
                return None
            if not all(value < float("inf") for value in normalized):
                return None
            return normalized

        def add(plan, score=None):
            if not isinstance(plan, dict) or set(plan) != required:
                return
            if not self._plan_supported(plan, deployment):
                return
            key = tuple(sorted((str(k), str(v)) for k, v in plan.items()))
            normalized_score = normalize_score(score)
            previous = entry_by_key.get(key)
            if previous is not None:
                # A one-exchange escape is intentionally inserted before the
                # background beam.  If the beam contains the same plan, keep
                # its real calendar score instead of silently retaining the
                # earlier unscored placeholder.
                if (
                    previous.get("screen_score") is None
                    and normalized_score is not None
                ):
                    previous["screen_score"] = normalized_score
                return
            entry = {
                "plan": {str(k): str(v) for k, v in plan.items()},
                "screen_score": normalized_score,
            }
            entries.append(entry)
            entry_by_key[key] = entry

        # The first synchronous search may be interrupted before its calendar
        # beam is materialized.  Put every one-exchange alternative ahead of
        # background candidates so a late service is not silently omitted by
        # an alphabetically biased/capped beam.  Afterwards fill the remaining
        # slots from the background search.  This keeps the online rerank
        # bounded while preserving at least one direct escape from congestion
        # at every deployed service replica.  It is disabled for the
        # no-plan-optimizer ablation.
        if self.plan_optimizer_enabled:
            base = dict(selected)
            pool_limit = max(16, 2 * self.optimizer.screening_beam_width)
            add(base, result.get("score"))
            for service in sorted(required):
                devices = deployment.get(service, [])
                if isinstance(devices, str):
                    devices = [devices]
                for device in sorted(str(item) for item in devices):
                    if device == str(base.get(service)):
                        continue
                    neighbor = dict(base)
                    neighbor[service] = device
                    add(neighbor)
            for item in raw:
                if isinstance(item, dict) and isinstance(
                    item.get("plan"), dict
                ):
                    add(item["plan"], item.get("screen_score"))
                elif isinstance(item, dict):
                    add(item)
            entries = entries[:pool_limit]
            # Unscored direct alternatives are useful as bounded escapes from
            # commitments added after the background snapshot, but they must
            # never be interpreted as queue-free.  Until a later background
            # search evaluates them, use the selected plan's current-snapshot
            # score as a conservative common baseline.  The online delta then
            # only rewards a neighbor for commitments observed after that
            # snapshot.
            selected_score = normalize_score(result.get("score"))
            for entry in entries:
                if entry.get("screen_score") is None:
                    entry["screen_score"] = selected_score
            return entries

        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("plan"), dict):
                add(item["plan"], item.get("screen_score"))
            elif isinstance(item, dict):
                add(item)
        return entries

    def _rolling_active_work(
        self, info, revision, deployment, demand_cache=None
    ):
        """Return fast-path work from all locally active full-plan commitments.

        The detailed execution state is deliberately built only by the
        background planner.  Between two background publications, however,
        every newly scheduled root is still an immutable future commitment.
        Summing its profiled per-replica demand is a conservative, bounded
        approximation that prevents a stale plan from concentrating several
        services at once.  Live queue telemetry is folded in with ``max`` so
        observed work can correct missing local records without double
        counting tasks represented by both sources.
        """

        if not self.future_state_estimator_enabled:
            return {}, 0
        signature = self._dag_signature(info.get("dag"))
        with self._lock:
            records = [
                item
                for item in self._rolling_active_plans.values()
                if int(item.get("revision") or 0) == int(revision or 0)
                and item.get("source_id") == info.get("source_id")
                and item.get("dag_signature") == signature
            ]
            resource_states = dict(self._rolling_resource_state)

        source_id = info.get("source_id")
        work = {}
        for item in records:
            for service, device in item.get("plan", {}).items():
                demand = (
                    self._planning_task_demand(
                        source_id,
                        service,
                        device,
                        0.9,
                        demand_cache,
                    )
                    + self.latency_model.estimate_handoff(
                        service, device, 0.9
                    )
                )
                replica = (str(service), str(device))
                work[replica] = work.get(replica, 0.0) + max(0.0, demand)

        for service, raw_devices in (deployment or {}).items():
            devices = raw_devices if isinstance(raw_devices, list) else [raw_devices]
            for raw_device in devices:
                device = str(raw_device)
                state = resource_states.get((device, str(service)))
                if not isinstance(state, tuple) or len(state) != 3:
                    continue
                waiting, busy, elapsed = state
                if not waiting and not busy:
                    continue
                demand = (
                    self._planning_task_demand(
                        source_id,
                        service,
                        device,
                        0.9,
                        demand_cache,
                    )
                    + self.latency_model.estimate_handoff(
                        service, device, 0.9
                    )
                )
                observed = waiting * max(0.0, demand)
                if busy:
                    observed += max(0.1 * demand, demand - elapsed)
                replica = (str(service), device)
                work[replica] = max(work.get(replica, 0.0), observed)
        return work, len(records)

    def _rolling_repair_plans(
        self,
        result,
        info,
        deployment,
        queued_work,
        demand_cache=None,
    ):
        """Build bounded multi-service repairs around the latest full plan.

        A pool containing only one-replica exchanges cannot recover when two
        or more services become congested during the same microburst.  This
        small beam repeatedly completes partial repairs with the background
        plan and scores the resulting full DAG.  It therefore permits joint
        changes while keeping foreground work independent of the Cartesian
        plan-space size.
        """

        base = result.get("plan") if isinstance(result, dict) else None
        if not isinstance(base, dict):
            return []
        dag = info.get("dag") or {}
        source_id = info.get("source_id")
        services = [
            service for service in topological_order(dag)
            if service not in (START, END) and service in base
        ]

        def devices_for(service):
            raw = (deployment or {}).get(service, [])
            raw = raw if isinstance(raw, list) else [raw]
            return sorted({str(item) for item in raw if str(item or "")})

        def replica_cost(service, device):
            demand = (
                self._planning_task_demand(
                    source_id,
                    service,
                    device,
                    0.9,
                    demand_cache,
                )
                + self.latency_model.estimate_handoff(
                    service, device, 0.9
                )
            )
            return queued_work.get((str(service), str(device)), 0.0) + demand

        choices = [service for service in services if len(devices_for(service)) > 1]
        choices.sort(key=lambda service: (
            -max(
                0.0,
                replica_cost(service, str(base[service]))
                - min(replica_cost(service, device) for device in devices_for(service)),
            ),
            -replica_cost(service, str(base[service])),
            service,
        ))
        if not choices:
            return [dict(base)]

        # Foreground repair is on the offered-arrival path.  Four partial
        # plans are enough to preserve joint changes across all services; the
        # larger DP/BnB beam continues asynchronously in the rolling worker.
        beam_width = max(2, min(4, self.optimizer.screening_beam_width))

        def score(plan):
            projected, noqueue, max_work = self._rolling_intrinsic_latency(
                dag,
                plan,
                queued_work,
                source_id,
                demand_cache,
            )
            return (
                projected,
                max(0.0, projected - noqueue),
                max_work,
                sum(
                    1 for service in services
                    if str(plan[service]) != str(base[service])
                ),
                tuple(sorted(plan.items())),
            )

        beam = [(score(base), dict(base))]
        for service in choices:
            expanded = {}
            for _, partial in beam:
                for device in devices_for(service):
                    plan = dict(partial)
                    plan[service] = device
                    key = tuple(sorted(plan.items()))
                    if key not in expanded:
                        expanded[key] = (score(plan), plan)
            beam = sorted(expanded.values(), key=lambda item: item[0])[:beam_width]
        plans = [plan for _, plan in beam]
        # The critical-path score can legitimately regard a parallel service
        # as non-critical for the current root.  Still include one complete
        # minimum-pressure plan so repeated ties cannot concentrate that
        # service until it becomes the next bottleneck.
        balanced = dict(base)
        for service in choices:
            balanced[service] = min(
                devices_for(service),
                key=lambda device: (replica_cost(service, device), device),
            )
        balanced_key = tuple(sorted(balanced.items()))
        if balanced_key not in {
            tuple(sorted(plan.items())) for plan in plans
        }:
            plans.append(balanced)
        if tuple(sorted(base.items())) not in {
            tuple(sorted(plan.items())) for plan in plans
        }:
            plans.append(dict(base))
        return plans

    def _planning_task_demand(
        self,
        source_id,
        service,
        device,
        quantile,
        demand_cache=None,
    ):
        key = (
            str(source_id),
            str(service),
            str(device),
            float(quantile),
        )
        if demand_cache is not None and key in demand_cache:
            return demand_cache[key]
        value = self.latency_model.estimate_task(
            source_id, service, device, quantile
        )
        if demand_cache is not None:
            demand_cache[key] = value
        return value

    def _rolling_intrinsic_latency(
        self,
        dag,
        plan,
        queued_work=None,
        source_id=None,
        demand_cache=None,
    ):
        queued_work = queued_work or {}
        finish = {START: 0.0}
        noqueue_finish = {START: 0.0}
        max_replica_work = 0.0
        for service in topological_order(dag):
            if service == START:
                continue
            predecessors = dag[service].get("prev_nodes", [])
            ready = max(
                (finish.get(item, 0.0) for item in predecessors),
                default=0.0,
            )
            noqueue_ready = max(
                (noqueue_finish.get(item, 0.0) for item in predecessors),
                default=0.0,
            )
            if service == END:
                finish[service] = ready
                noqueue_finish[service] = noqueue_ready
                continue
            device = str(plan[service])
            duration = (
                self._planning_task_demand(
                    source_id,
                    service,
                    device,
                    0.9,
                    demand_cache,
                )
                + self.latency_model.estimate_handoff(
                    service, device, 0.9
                )
            )
            backlog = max(
                0.0, float(queued_work.get((str(service), device), 0.0))
            )
            finish[service] = ready + backlog + duration
            noqueue_finish[service] = noqueue_ready + duration
            max_replica_work = max(max_replica_work, backlog + duration)
        sinks = [
            name for name, node in dag.items()
            if not node.get("next_nodes")
        ]
        latency = finish.get(
            END,
            max((finish.get(name, 0.0) for name in sinks), default=0.0),
        )
        noqueue = noqueue_finish.get(
            END,
            max(
                (noqueue_finish.get(name, 0.0) for name in sinks),
                default=0.0,
            ),
        )
        return latency, noqueue, max_replica_work

    def _rerank_rolling_result(self, result, info, deployment, revision):
        """Re-rank a cached full-plan pool using only post-snapshot commitments."""

        if not (
            self.rolling_planner_enabled
            and self.future_state_estimator_enabled
            and self.plan_optimizer_enabled
        ):
            return result
        pool = self._rolling_plan_pool(result, deployment)
        # Every complete plan references the same small set of deployed
        # service-device pairs.  Computing a weighted residual quantile sorts
        # up to ``history_size`` values; memoize those pair estimates once per
        # request instead of repeating the sort for every plan in the pool.
        demand_cache = {}
        active_work, active_count = self._rolling_active_work(
            info, revision, deployment, demand_cache
        )
        repair_plans = self._rolling_repair_plans(
            result,
            info,
            deployment,
            active_work,
            demand_cache,
        )
        known = {
            tuple(sorted(item["plan"].items())) for item in pool
        }
        selected_score = tuple(result.get("score") or ())
        for plan in repair_plans:
            key = tuple(sorted(plan.items()))
            if key in known:
                continue
            known.add(key)
            pool.append({
                "plan": dict(plan),
                "screen_score": selected_score,
            })
        if not pool:
            return result
        if active_count == 0 and not active_work:
            result["candidate_pool"] = pool
            result["candidate_pool_size"] = len(pool)
            result["online_rerank_delta_tasks"] = 0
            result["online_active_commitment_tasks"] = 0
            result["online_rerank_changed"] = False
            return result
        metadata = info.get("meta_data")
        metadata = metadata if isinstance(metadata, dict) else {}
        try:
            slo = float(
                metadata.get(
                    "slo_seconds",
                    metadata.get(
                        "latency_slo_s", self.optimizer.default_slo_s
                    ),
                )
            )
        except (TypeError, ValueError):
            slo = self.optimizer.default_slo_s
        slo = max(1e-6, slo)

        ranked = []
        for rank, entry in enumerate(pool):
            plan = entry["plan"]
            projected, noqueue, max_work = self._rolling_intrinsic_latency(
                info["dag"],
                plan,
                active_work,
                info.get("source_id"),
                demand_cache,
            )
            queue_inflation = max(0.0, projected - noqueue)
            latency = projected
            # Sum the work already committed ahead of every selected replica.
            # This is the immediate externality of the full plan.  It remains
            # secondary to SLO feasibility/tardiness, but precedes shaving a
            # few milliseconds from a currently non-critical branch.  That
            # ordering is what keeps future service queues schedulable instead
            # of merely reacting after a replica has become the bottleneck.
            selected_pressure = sum(
                max(
                    0.0,
                    float(
                        active_work.get(
                            (str(service), str(device)), 0.0
                        )
                    ),
                )
                for service, device in plan.items()
            )
            score = (
                float(latency > slo),
                max(0.0, latency - slo) / slo,
                selected_pressure / slo,
                max_work / slo,
                queue_inflation / slo,
                latency,
                rank,
            )
            ranked.append((score, tuple(sorted(plan.items())), plan))
        ranked.sort(key=lambda item: (item[0], item[1]))
        chosen_score, _, chosen = ranked[0]
        previous = dict(result.get("plan") or {})
        result["plan"] = dict(chosen)
        result["candidate_pool"] = pool
        result["candidate_pool_size"] = len(pool)
        result["online_rerank_delta_tasks"] = active_count
        result["online_active_commitment_tasks"] = active_count
        result["online_repair_plan_count"] = len(repair_plans)
        result["online_rerank_changed"] = dict(chosen) != previous
        result["online_rerank_score"] = tuple(chosen_score[:6])
        if active_count or active_work:
            # Preserve the optimizer's five-field public score contract while
            # exposing the complete rolling score separately above.
            result["score"] = (
                chosen_score[0],
                chosen_score[1],
                chosen_score[5],
                chosen_score[4],
                chosen_score[3],
            )
            result["predicted_miss_probability"] = float(chosen_score[0])
            result["score_evaluated"] = False
            result["prediction_complete"] = False
            result["optimality_proven"] = False
            result["fallback_reason"] = (
                "rolling_plan_not_ready_commitment_rerank"
                if result.get("planner_mode") == "rolling_fallback"
                else "rolling_cache_commitment_rerank"
            )
        return result

    def _validate_profile_coverage(self, dag, deployment):
        missing = []
        for service in dag:
            if service in (TaskConstant.START.value, TaskConstant.END.value):
                continue
            devices = deployment.get(service, []) if isinstance(deployment, dict) else []
            if isinstance(devices, str):
                devices = [devices]
            for device in devices:
                if self.latency_model.sample_count(service, device) == 0:
                    missing.append(f"{service}@{device}")
        if missing:
            raise ValueError(
                "FragSplice latency profile does not cover the active fixed deployment: "
                + ", ".join(sorted(missing))
            )

    def get_schedule_plan(self, info):
        decision_started = time.monotonic()
        with self.overhead_estimator:
            dag = copy.deepcopy(info["dag"])
            decision_info = dict(info)
            decision_info["dag"] = dag
            revision = int(self.system.runtime_directory_revision() or 0)
            with self._lock:
                # Decisions are immutable after publication.  Defer copying
                # until the fallback path actually needs the prior result;
                # cache hits should not duplicate an unused candidate pool.
                previous_decision = self.last_decision
                validated_deployment = self._validated_deployment
                validated_revision = self._validated_revision

            result, deployment, cache_age = self._cached_result(
                decision_info,
                revision,
            )
            if result is None:
                previous_plan = (
                    previous_decision.get("plan")
                    if isinstance(previous_decision, dict) else None
                )
                can_reuse = (
                    self.rolling_planner_enabled
                    and isinstance(previous_decision, dict)
                    and validated_revision == revision
                    and self._plan_supported(
                        previous_plan, validated_deployment
                    )
                )
                if can_reuse:
                    result = self._reuse_result(previous_decision)
                    deployment = validated_deployment
                    cache_age = None
                else:
                    snapshot = self._planning_snapshot()
                    deployment = self._deployment_for_snapshot(snapshot)
                    result = self._solve_now(
                        decision_info,
                        snapshot,
                        deployment,
                        initial_plan=previous_plan,
                    )
                    result["planner_mode"] = "synchronous_bootstrap"
                    result["background_search_seconds"] = 0.0
                    cache_age = None

            result = self._rerank_rolling_result(
                result,
                decision_info,
                deployment,
                revision,
            )

            policy = materialize_offloading_plan(
                self.configuration,
                dag,
                result["plan"],
                info.get("source_device"),
                self.cloud_device,
            )
            task_context = info.get("task_context")
            task_context = task_context if isinstance(task_context, dict) else {}
            root_uuid = str(task_context.get("root_uuid") or "")
            result["online_decision_seconds"] = max(
                0.0, time.monotonic() - decision_started
            )
            with self._lock:
                # ``result`` is request-local and is not mutated after this
                # point, so publishing the reference avoids another complete
                # candidate-pool copy on every task arrival.
                self.last_decision = result
                self._validated_deployment = copy.deepcopy(deployment)
                self._validated_revision = revision
                self._decision_count += 1
                decision_count = self._decision_count
            self._enqueue_rolling_plan(
                decision_info,
                result["plan"],
                root_uuid,
                revision,
            )
            should_log_decision = (
                decision_count == 1
                or decision_count % self.decision_log_interval == 0
                or result["online_decision_seconds"] > 0.05
            )
            if should_log_decision:
                LOGGER.info(
                    "[FragSplice] count=%s source=%s plans=%s screened=%s "
                    "scenarios=%s/%s evaluated=%s optimal=%s "
                    "unschedulable=%s intrinsic_slo_infeasible=%s "
                    "budget_exhausted=%s refinement_exhausted=%s "
                    "score_evaluated=%s fallback=%s "
                    "future_commitments=%s predicted_miss=%s cost_domain=%s "
                    "score=%s "
                    "lower_bound=%s plan=%s mode=%s cache_age=%s "
                    "state_build=%.4fs search=%.4fs online=%.4fs "
                    "background=%.4fs overrun=%.4fs",
                    decision_count,
                    info.get("source_id"),
                    result["candidate_count"],
                    result["screened"],
                    result.get("selected_outcome_scenarios", 0),
                    result["scenario_count"],
                    result.get("evaluated_count", len(result["evaluated"])),
                    result["optimality_proven"],
                    result["unschedulable"],
                    result["intrinsic_slo_infeasible"],
                    result["budget_exhausted"],
                    result.get("scenario_refinement_exhausted", False),
                    result.get("score_evaluated", True),
                    result.get("fallback_reason") or "-",
                    self.future_state_estimator_enabled,
                    (
                        "-"
                        if result.get("predicted_miss_probability") is None
                        else f"{result['predicted_miss_probability']:.3f}"
                    ),
                    result.get("planning_cost_domain", "temporal"),
                    tuple(round(item, 6) for item in result["score"]),
                    tuple(
                        round(item, 6)
                        for item in result["best_open_lower_bound"]
                    ),
                    result["plan"],
                    result.get("planner_mode", "synchronous"),
                    (
                        "-" if cache_age is None
                        else f"{cache_age:.4f}s"
                    ),
                    result.get("state_build_seconds", 0.0),
                    result["search_seconds"],
                    result.get("online_decision_seconds", 0.0),
                    result.get("background_search_seconds", 0.0),
                    result.get("budget_overrun_seconds", 0.0),
                )
            return policy

    def update_task(self, task):
        root_uuid = str(task.get_root_uuid() or "")
        if root_uuid:
            with self._lock:
                self._rolling_active_plans.pop(root_uuid, None)
        updated = False
        if self.distribution_profiler_enabled:
            deployment = task.get_deployment()
            if not isinstance(deployment, dict) or not deployment:
                raise ValueError(
                    "FragSplice completed task has no fixed deployment context"
                )
            # Online feedback updates the shared in-memory distribution only.
            # The cold profile remains the immutable experiment starting point.
            self.latency_model.ensure_profile_context(
                configuration=self.configuration,
                deployment=deployment,
                dag=task.get_dag(),
                require_complete=True,
            )
            updated = self.latency_model.update_task(task)
        if updated:
            with self._lock:
                self._feedback_count += 1
                feedback_count = self._feedback_count
            if feedback_count % self.decision_log_interval == 0:
                LOGGER.info(
                    "[FragSplice] Online latency feedback count=%s "
                    "source=%s latest_task=%s",
                    feedback_count,
                    task.get_source_id(),
                    task.get_task_id(),
                )

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        queue_states = (
            resource.get("queue_state", {})
            if isinstance(resource, dict)
            else {}
        )
        if not isinstance(queue_states, dict):
            queue_states = {}
        normalized = {}
        for service, state in queue_states.items():
            if not isinstance(state, dict):
                continue
            try:
                waiting = max(0, int(state.get("waiting_count") or 0))
            except (TypeError, ValueError):
                waiting = 0
            try:
                elapsed = max(
                    0.0, float(state.get("running_elapsed_s") or 0.0)
                )
            except (TypeError, ValueError):
                elapsed = 0.0
            normalized[(str(device), str(service))] = (
                waiting,
                bool(state.get("busy")),
                elapsed,
            )
        with self._lock:
            stale = [
                key for key in self._rolling_resource_state
                if key[0] == str(device)
            ]
            for key in stale:
                self._rolling_resource_state.pop(key, None)
            self._rolling_resource_state.update(normalized)

    def update_policy(self, policy):
        pass

    def run(self):
        self._ensure_rolling_worker()

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
