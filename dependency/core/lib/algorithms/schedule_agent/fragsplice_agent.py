import abc
import copy
import threading

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

from .base_agent import BaseAgent
from .fragsplice import FragSpliceLatencyModel, FragSpliceOptimizer

__all__ = ("FragSpliceAgent",)


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


@ClassFactory.register(ClassType.SCH_AGENT, alias="fragsplice")
class FragSpliceAgent(BaseAgent, abc.ABC):
    """Commitment-aware full-DAG offloading under a fixed deployment."""

    def __init__(
        self,
        system,
        agent_id: int,
        configuration=None,
        latency_profile=None,
        latency_slo_s=3.0,
        scenario_count=32,
        max_scenarios=256,
        search_time_limit_s=0.0,
        random_seed=0,
        queue_state_max_age_s=1.5,
        residual_half_life_tasks=8.0,
        incumbent_neighborhood_size=4,
        screening_beam_width=16,
        use_future_commitments=True,
    ):
        super().__init__(system, agent_id)
        self.system = system
        self.agent_id = agent_id
        self.configuration = _load_mapping(configuration, "configuration")
        self.latency_profile_path = (
            Context.get_file_path(latency_profile)
            if isinstance(latency_profile, str) else None
        )
        profile = _load_mapping(latency_profile, "latency_profile")
        profile_context = FragSpliceLatencyModel.validate_profile_context(
            profile,
            self.configuration,
        )
        revision_getter = getattr(system, "runtime_directory_revision", None)
        revision = revision_getter() if callable(revision_getter) else 0
        self.latency_model = GlobalInstanceManager.get_instance(
            FragSpliceLatencyModel,
            ("fragsplice", id(system), revision),
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
        self.optimizer = FragSpliceOptimizer(
            self.latency_model,
            default_slo_s=latency_slo_s,
            scenario_count=scenario_count,
            max_scenarios=max_scenarios,
            search_time_limit_s=search_time_limit_s,
            random_seed=random_seed,
            queue_state_max_age_s=queue_state_max_age_s,
            incumbent_neighborhood_size=incumbent_neighborhood_size,
            screening_beam_width=screening_beam_width,
        )
        self.overhead_estimator = OverheadEstimator(
            "FragSplice", "scheduler/fragsplice", agent_id=agent_id
        )
        self.use_future_commitments = bool(use_future_commitments)
        self._lock = threading.RLock()
        self.last_decision = None
        if not self.latency_model.has_samples():
            LOGGER.warning(
                "[FragSplice] No cold-start samples were loaded. Full-plan scheduling "
                "will remain unavailable until every deployed pair has a valid sample."
            )

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
        with self.overhead_estimator, self._lock:
            dag = copy.deepcopy(info["dag"])
            decision_info = dict(info)
            decision_info["dag"] = dag
            snapshot = self.system.get_scheduling_snapshot()
            if not self.use_future_commitments:
                # Current-state ablation: preserve the exact live replica
                # telemetry and the same optimizer/search budget, but remove
                # work that is known only through already committed full-DAG
                # plans.  This isolates the value of future-state inference
                # without changing service-time prediction or planner cost.
                snapshot = copy.deepcopy(snapshot)
                snapshot["reservations"] = []
                snapshot["commitments"] = []
                snapshot["task_barriers"] = []
            deployment = snapshot.get("deployment")
            if not isinstance(deployment, dict):
                deployment = self.system.runtime_service_nodes()
            self.latency_model.ensure_profile_context(
                configuration=self.configuration,
                deployment=deployment,
                dag=dag,
                require_complete=True,
            )
            self._validate_profile_coverage(dag, deployment)
            previous_plan = (
                self.last_decision.get("plan")
                if isinstance(self.last_decision, dict) else None
            )
            result = self.optimizer.solve(
                decision_info,
                snapshot,
                deployment,
                self.cloud_device,
                initial_plan=previous_plan,
            )
            for service, device in result["plan"].items():
                dag[service]["service"]["execute_device"] = device
            if TaskConstant.START.value in dag:
                dag[TaskConstant.START.value]["service"]["execute_device"] = str(
                    info.get("source_device") or ""
                )
            if TaskConstant.END.value in dag:
                dag[TaskConstant.END.value]["service"]["execute_device"] = self.cloud_device
            policy = copy.deepcopy(self.configuration)
            policy["dag"] = dag
            self.last_decision = copy.deepcopy(result)
            LOGGER.info(
                "[FragSplice] source=%s plans=%s screened=%s scenarios=%s evaluated=%s "
                "optimal=%s unschedulable=%s intrinsic_slo_infeasible=%s "
                "budget_exhausted=%s future_commitments=%s predicted_miss=%.3f "
                "score=%s lower_bound=%s plan=%s overhead=%.4fs",
                info.get("source_id"),
                result["candidate_count"],
                result["screened"],
                result["scenario_count"],
                len(result["evaluated"]),
                result["optimality_proven"],
                result["unschedulable"],
                result["intrinsic_slo_infeasible"],
                result["budget_exhausted"],
                self.use_future_commitments,
                result["predicted_miss_probability"],
                tuple(round(item, 6) for item in result["score"]),
                tuple(round(item, 6) for item in result["best_open_lower_bound"]),
                result["plan"],
                result["search_seconds"],
            )
            return policy

    def update_task(self, task):
        with self._lock:
            deployment_getter = getattr(task, "get_deployment", None)
            deployment = deployment_getter() if callable(deployment_getter) else None
            if not isinstance(deployment, dict) or not deployment:
                deployment = self.system.runtime_service_nodes()
            self.latency_model.ensure_profile_context(
                configuration=self.configuration,
                deployment=deployment,
                dag=task.get_dag(),
                require_complete=True,
            )
            updated = self.latency_model.update_task(task)
            if updated and self.latency_profile_path:
                self.latency_model.save(
                    self.latency_profile_path,
                    deployment=deployment,
                )
        if updated:
            LOGGER.debug(
                "[FragSplice] Updated service-time distribution from source=%s task=%s",
                task.get_source_id(),
                task.get_task_id(),
            )

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        # Decisions read one atomic Scheduler snapshot.  Keeping a second copy
        # in every source agent would create avoidable consistency bugs.
        pass

    def update_policy(self, policy):
        pass

    def run(self):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
