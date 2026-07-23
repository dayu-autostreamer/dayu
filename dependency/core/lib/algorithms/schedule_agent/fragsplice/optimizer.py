import heapq
import math
import time

from .execution_state import (
    END,
    START,
    FragSpliceExecutionState,
    service_names,
    topological_order,
)


def _cvar(values, level=0.95):
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    start = min(len(ordered) - 1, int(math.floor(level * len(ordered))))
    tail = ordered[start:]
    return sum(tail) / len(tail)


class FragSpliceOptimizer:
    """Scenario evaluator plus DP-guided anytime branch-and-bound."""

    def __init__(
        self,
        latency_model,
        default_slo_s=3.0,
        scenario_count=32,
        max_scenarios=256,
        search_time_limit_s=0.0,
        random_seed=0,
        queue_state_max_age_s=1.5,
        incumbent_neighborhood_size=4,
        screening_beam_width=16,
    ):
        self.latency_model = latency_model
        self.default_slo_s = max(1e-6, float(default_slo_s))
        self.scenario_count = max(8, int(scenario_count))
        self.max_scenarios = max(self.scenario_count, int(max_scenarios))
        self.search_time_limit_s = max(0.0, float(search_time_limit_s))
        self.random_seed = int(random_seed)
        self.queue_state_max_age_s = max(0.0, float(queue_state_max_age_s))
        self.incumbent_neighborhood_size = max(
            0, int(incumbent_neighborhood_size)
        )
        self.screening_beam_width = max(1, int(screening_beam_width))

    def _candidate_devices(self, dag, deployment, source_device, cloud_device):
        result = {}
        for service in service_names(dag):
            raw = deployment.get(service, []) if isinstance(deployment, dict) else []
            if isinstance(raw, str):
                raw = [raw]
            devices = sorted({str(item) for item in raw if str(item or "")})
            if not devices:
                current = str(dag[service].get("service", {}).get("execute_device") or "")
                devices = [current] if current else []
            if not devices and cloud_device:
                devices = [str(cloud_device)]
            if not devices:
                raise ValueError(f"FragSplice has no deployed replica for service {service}")
            result[service] = devices
        return result

    def _optimistic_latency(self, dag, candidates, partial):
        finish = {START: 0.0}
        for service in topological_order(dag):
            if service == START:
                continue
            ready = max(
                (finish.get(item, 0.0) for item in dag[service].get("prev_nodes", [])),
                default=0.0,
            )
            if service == END:
                finish[service] = ready
                continue
            devices = [partial[service]] if service in partial else candidates[service]
            demand = min(
                self.latency_model.lower_bound(service, device)
                + self.latency_model.lower_bound_handoff(service, device)
                for device in devices
            )
            finish[service] = ready + demand
        if END in finish:
            return finish[END]
        sinks = [name for name in dag if not dag[name].get("next_nodes")]
        return max((finish.get(name, 0.0) for name in sinks), default=0.0)

    def _lower_score(self, dag, candidates, partial, slo, elapsed=0.0):
        latency = max(0.0, float(elapsed)) + self._optimistic_latency(
            dag, candidates, partial
        )
        miss = 1.0 if latency > slo else 0.0
        tardiness = max(0.0, latency - slo) / slo
        # The candidate's optimistic no-contention latency is also a lower
        # bound on system incremental latency: delaying existing tasks can
        # only add non-negative externality.  Queue inflation and committed
        # replica work have zero as their admissible lower bounds.
        return (miss, tardiness, latency, 0.0, 0.0)

    def _incumbent_neighborhood(self, services, candidates, preferred_plan):
        """Return the most consequential one-replica exchanges first.

        A short anytime budget can otherwise spend all of its full-plan
        evaluations inside the branch containing the previous decision.  The
        neighborhood is only a stronger primal warm start: branch-and-bound
        remains responsible for exact search whenever time permits.
        """
        if self.incumbent_neighborhood_size <= 0:
            return []
        ranked_services = sorted(
            (
                service for service in services
                if any(
                    device != preferred_plan[service]
                    for device in candidates[service]
                )
            ),
            key=lambda service: (
                -max(
                    self.latency_model.estimate(service, device, 0.9)
                    + self.latency_model.estimate_handoff(service, device, 0.9)
                    for device in candidates[service]
                ),
                service,
            ),
        )
        plans = []
        for service in ranked_services:
            alternatives = sorted(
                (
                    device for device in candidates[service]
                    if device != preferred_plan[service]
                ),
                key=lambda device: (
                    self.latency_model.estimate(service, device, 0.5)
                    + self.latency_model.estimate_handoff(service, device, 0.5),
                    device,
                ),
            )
            for device in alternatives:
                plan = dict(preferred_plan)
                plan[service] = device
                plans.append(plan)
                if len(plans) >= self.incumbent_neighborhood_size:
                    return plans
        return plans

    @staticmethod
    def _candidate_payload(
        dag,
        plan,
        source_id,
        candidate_root,
        candidate_created_at,
        slo,
    ):
        return {
            "root": str(
                candidate_root or f"__fragsplice_pending__:{source_id}"
            ),
            "source": source_id,
            "dag": dag,
            "plan": plan,
            "created_at": candidate_created_at,
            "slo": slo,
        }

    @staticmethod
    def _screen_score(projection, slo):
        latency = float(projection["latency"])
        return (
            float(latency > slo),
            max(0.0, latency - slo) / slo,
            latency,
            float(projection["queue_inflation"]),
            float(projection["max_replica_work"]),
        )

    def _screening_plans(
        self,
        state,
        dag,
        services,
        candidates,
        preferred_plan,
        source_id,
        candidate_root,
        candidate_created_at,
        slo,
        seed,
        baseline,
    ):
        """Generate promising complete plans with a cheap calendar beam.

        The calendar holds one sampled execution of already committed work.
        It is used only to select primal warm starts and never to prune the
        exact branch-and-bound tree. Therefore the small-space optimum and the
        anytime lower-bound semantics are unchanged.
        """

        beam = [({}, None)]
        for service in services:
            expanded = []
            for partial, _ in beam:
                for device in candidates[service]:
                    child = dict(partial)
                    child[service] = device
                    complete = dict(preferred_plan)
                    complete.update(child)
                    candidate = self._candidate_payload(
                        dag,
                        complete,
                        source_id,
                        candidate_root,
                        candidate_created_at,
                        slo,
                    )
                    projection = state.screen_candidate(
                        candidate, seed, baseline
                    )
                    expanded.append((
                        child,
                        self._screen_score(projection, slo),
                    ))
            expanded.sort(key=lambda item: (
                item[1], tuple(sorted(item[0].items()))
            ))
            beam = expanded[:self.screening_beam_width]

        ranked = sorted(
            ((score, plan) for plan, score in beam),
            key=lambda item: (item[0], tuple(sorted(item[1].items()))),
        )
        plans = [dict(plan) for _, plan in ranked]
        preferred_key = tuple(sorted(preferred_plan.items()))
        if all(tuple(sorted(plan.items())) != preferred_key for plan in plans):
            plans.append(dict(preferred_plan))
        return plans

    @staticmethod
    def _score_plan(
        state,
        dag,
        plan,
        source_id,
        candidate_root,
        candidate_created_at,
        slo,
        seeds,
        baseline_cache,
        outcome_cache,
    ):
        candidate = FragSpliceOptimizer._candidate_payload(
            dag,
            plan,
            source_id,
            candidate_root,
            candidate_created_at,
            slo,
        )
        root = candidate["root"]
        miss_deltas = []
        tardiness_deltas = []
        latency_impacts = []
        queue_inflation = []
        concentration = []
        plan_key = tuple(sorted(plan.items()))
        for seed in seeds:
            if seed not in baseline_cache:
                baseline_cache[seed] = state.simulate(
                    None, seed, include_calendar=True
                )
            baseline = baseline_cache[seed]
            cache_key = (plan_key, seed)
            if cache_key not in outcome_cache:
                outcome_cache[cache_key] = state.simulate(candidate, seed)
            outcome = outcome_cache[cache_key]

            def losses(result):
                misses = 0
                tardiness = 0.0
                for task_root, latency in result["latency"].items():
                    task_slo = max(1e-9, result["deadlines"].get(task_root, slo))
                    misses += int(latency > task_slo)
                    tardiness += max(0.0, latency - task_slo) / task_slo
                return misses, tardiness

            baseline_miss, baseline_tardy = losses(baseline)
            current_miss, current_tardy = losses(outcome)
            miss_deltas.append(max(0.0, current_miss - baseline_miss))
            tardiness_deltas.append(max(0.0, current_tardy - baseline_tardy))
            latency = outcome["latency"].get(root, float("inf"))
            old_task_externality = sum(
                max(
                    0.0,
                    outcome["latency"].get(task_root, baseline_latency)
                    - baseline_latency,
                )
                for task_root, baseline_latency in baseline["latency"].items()
            )
            latency_impacts.append(latency + old_task_externality)
            inflation = latency - outcome["candidate_noqueue"]
            queue_inflation.append(max(0.0, inflation) if inflation > 1e-9 else 0.0)
            work = list(outcome["replica_work"].values())
            # Per-replica committed work is a direct inverse-headroom signal.
            # A normalized concentration ratio would perversely prefer a
            # uniformly slow plan over a faster plan with small heterogeneity.
            concentration.append(max(work) if work else 0.0)
        count = len(seeds)
        return tuple(round(value, 12) for value in (
            sum(miss_deltas) / count,
            _cvar(tardiness_deltas, 0.95),
            sum(latency_impacts) / count,
            sum(queue_inflation) / count,
            sum(concentration) / count,
        ))

    def _search(
        self,
        state,
        dag,
        candidates,
        source_id,
        candidate_root,
        candidate_created_at,
        slo,
        seeds,
        deadline,
        baseline_cache,
        outcome_cache,
        initial_plan=None,
    ):
        services = [name for name in topological_order(dag) if name not in (START, END)]
        preferred_plan = {
            service: min(
                candidates[service],
                key=lambda device: (
                    self.latency_model.estimate(service, device, 0.5),
                    device,
                ),
            )
            for service in services
        }
        if isinstance(initial_plan, dict) and all(
            initial_plan.get(service) in candidates[service]
            for service in services
        ):
            preferred_plan = {
                service: initial_plan[service] for service in services
            }

        screen_seed = seeds[0]
        if screen_seed not in baseline_cache:
            baseline_cache[screen_seed] = state.simulate(
                None, screen_seed, include_calendar=True
            )
        screened_plans = self._screening_plans(
            state,
            dag,
            services,
            candidates,
            preferred_plan,
            source_id,
            candidate_root,
            candidate_created_at,
            slo,
            screen_seed,
            baseline_cache[screen_seed],
        )
        warm_limit = max(1, self.incumbent_neighborhood_size)
        warm_plans = screened_plans[:warm_limit]
        preferred_key = tuple(sorted(preferred_plan.items()))
        if all(
            tuple(sorted(plan.items())) != preferred_key
            for plan in warm_plans
        ):
            warm_plans.append(preferred_plan)

        incumbent_plan = warm_plans[0]
        incumbent_score = self._score_plan(
            state,
            dag,
            incumbent_plan,
            source_id,
            candidate_root,
            candidate_created_at,
            slo,
            seeds,
            baseline_cache,
            outcome_cache,
        )
        evaluated_by_plan = {
            tuple(sorted(incumbent_plan.items())): (
                incumbent_score, incumbent_plan
            )
        }
        expanded = 0

        # Spend the first full-simulation evaluations on plans selected by the
        # commitment calendar. This can change several service assignments at
        # once, unlike a one-exchange warm start, while exact scores still use
        # all configured common-random-number scenarios.
        for plan in warm_plans[1:]:
            if deadline is not None and time.monotonic() >= deadline:
                break
            plan_key = tuple(sorted(plan.items()))
            if plan_key in evaluated_by_plan:
                continue
            score = self._score_plan(
                state,
                dag,
                plan,
                source_id,
                candidate_root,
                candidate_created_at,
                slo,
                seeds,
                baseline_cache,
                outcome_cache,
            )
            evaluated_by_plan[plan_key] = (score, plan)
            if score < incumbent_score:
                incumbent_score = score
                incumbent_plan = plan

        branch_order = {}
        for service in services:
            branch_order[service] = sorted(
                candidates[service],
                key=lambda device: (
                    0 if device == incumbent_plan[service] else 1,
                    self.latency_model.estimate(service, device, 0.5),
                    device,
                ),
            )
        heap = []
        serial = 0
        # A candidate task has only reserved an identity. Its application SLO
        # clock starts after source data is materialized and immediately before
        # lease admission, so scheduler/search time is not task latency.
        elapsed = 0.0
        root_bound = self._lower_score(
            dag, candidates, {}, slo, elapsed=elapsed
        )
        heapq.heappush(heap, (root_bound, serial, 0, {}))

        while heap:
            if deadline is not None and time.monotonic() >= deadline:
                break
            lower, _, index, partial = heapq.heappop(heap)
            if lower >= incumbent_score:
                continue
            if index == len(services):
                plan = dict(partial)
                plan_key = tuple(sorted(plan.items()))
                previous = evaluated_by_plan.get(plan_key)
                if previous is None:
                    score = self._score_plan(
                        state,
                        dag,
                        plan,
                        source_id,
                        candidate_root,
                        candidate_created_at,
                        slo,
                        seeds,
                        baseline_cache,
                        outcome_cache,
                    )
                    evaluated_by_plan[plan_key] = (score, plan)
                else:
                    score = previous[0]
                if score < incumbent_score:
                    incumbent_score = score
                    incumbent_plan = plan
                continue
            service = services[index]
            expanded += 1
            for device in branch_order[service]:
                child = dict(partial)
                child[service] = device
                bound = self._lower_score(
                    dag, candidates, child, slo, elapsed=elapsed
                )
                if bound >= incumbent_score:
                    continue
                serial += 1
                heapq.heappush(heap, (bound, serial, index + 1, child))

        evaluated = sorted(evaluated_by_plan.values(), key=lambda item: item[0])
        best_open = heap[0][0] if heap else incumbent_score
        return {
            "plan": incumbent_plan,
            "score": incumbent_score,
            "evaluated": evaluated,
            "expanded": expanded,
            "screened": len(screened_plans),
            "optimality_proven": not heap,
            "best_open_lower_bound": best_open,
        }

    @staticmethod
    def _ranking_is_stable(result, scenario_count):
        evaluated = result["evaluated"]
        if len(evaluated) < 2:
            return False
        first, second = evaluated[0][0], evaluated[1][0]
        if second[0] - first[0] > 2.0 / scenario_count:
            return True
        if second[1] - first[1] > 0.02:
            return True
        return second[2] - first[2] > max(1e-3, 0.05 * max(first[2], 1e-6))

    def solve(
        self,
        info,
        snapshot,
        deployment,
        cloud_device,
        initial_plan=None,
    ):
        dag = info["dag"]
        source_id = info.get("source_id", "")
        task_context = info.get("task_context")
        task_context = task_context if isinstance(task_context, dict) else {}
        candidate_root = task_context.get("root_uuid")
        source_device = info.get("source_device", "")
        if START in dag:
            dag[START].setdefault("service", {})["execute_device"] = str(
                source_device or ""
            )
        if END in dag:
            dag[END].setdefault("service", {})["execute_device"] = str(
                cloud_device or ""
            )
        metadata = info.get("meta_data")
        metadata = metadata if isinstance(metadata, dict) else {}
        try:
            slo = float(
                metadata.get("slo_seconds", metadata.get("latency_slo_s", self.default_slo_s))
            )
        except (TypeError, ValueError):
            slo = self.default_slo_s
        slo = max(1e-6, slo)
        candidates = self._candidate_devices(
            dag, deployment, source_device, cloud_device
        )
        state = FragSpliceExecutionState(
            snapshot,
            self.latency_model,
            slo,
            queue_state_max_age_s=self.queue_state_max_age_s,
        )
        candidate_created_at = state.now
        started = time.monotonic()
        deadline = (
            started + self.search_time_limit_s
            if self.search_time_limit_s > 0.0 else None
        )
        counts = []
        count = self.scenario_count
        while True:
            counts.append(count)
            if count >= self.max_scenarios:
                break
            count = min(self.max_scenarios, count * 2)

        result = None
        used = 0
        baseline_cache = {}
        outcome_cache = {}
        incumbent_plan = initial_plan if isinstance(initial_plan, dict) else None
        for count in counts:
            if deadline is not None and time.monotonic() >= deadline and result is not None:
                break
            seeds = [
                self.random_seed + 1_000_003 * index
                for index in range(count)
            ]
            current = self._search(
                state,
                dag,
                candidates,
                source_id,
                candidate_root,
                candidate_created_at,
                slo,
                seeds,
                deadline,
                baseline_cache,
                outcome_cache,
                initial_plan=incumbent_plan,
            )
            result = current
            incumbent_plan = current["plan"]
            used = count
            if self._ranking_is_stable(current, count):
                break
        elapsed_before_execution = 0.0
        intrinsic_slo_infeasible = self._lower_score(
            dag,
            candidates,
            {},
            slo,
            elapsed=elapsed_before_execution,
        )[0] >= 1.0
        selected_key = tuple(sorted(result["plan"].items()))
        selected_root = str(
            candidate_root or f"__fragsplice_pending__:{source_id}"
        )
        selected_misses = []
        for seed in seeds:
            outcome = outcome_cache.get((selected_key, seed))
            latency = (
                outcome.get("latency", {}).get(selected_root, float("inf"))
                if isinstance(outcome, dict) else float("inf")
            )
            selected_misses.append(float(latency > slo))
        predicted_miss_probability = (
            sum(selected_misses) / len(selected_misses)
            if selected_misses else 1.0
        )
        result.update({
            "scenario_count": used,
            "search_seconds": time.monotonic() - started,
            # Only the no-queue model lower bound supports an infeasibility
            # claim. Expected added misses also include externalities on old
            # tasks and must not be relabeled as candidate unschedulability.
            "unschedulable": bool(intrinsic_slo_infeasible),
            "predicted_miss_probability": predicted_miss_probability,
            # A single task's no-queue lower bound cannot prove sustained
            # capacity overload. Keep that diagnosis separate and explicit.
            "structural_overload": False,
            "intrinsic_slo_infeasible": intrinsic_slo_infeasible,
            "budget_exhausted": bool(
                deadline is not None
                and time.monotonic() >= deadline
                and not result["optimality_proven"]
            ),
            "candidate_count": math.prod(len(candidates[name]) for name in candidates),
        })
        return result


__all__ = ("FragSpliceOptimizer",)
