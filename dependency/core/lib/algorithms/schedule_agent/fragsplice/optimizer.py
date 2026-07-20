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
    ):
        self.latency_model = latency_model
        self.default_slo_s = max(1e-6, float(default_slo_s))
        self.scenario_count = max(8, int(scenario_count))
        self.max_scenarios = max(self.scenario_count, int(max_scenarios))
        self.search_time_limit_s = max(0.0, float(search_time_limit_s))
        self.random_seed = int(random_seed)
        self.queue_state_max_age_s = max(0.0, float(queue_state_max_age_s))

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
        return (miss, tardiness, 0.0, 0.0, latency)

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
        root = str(candidate_root or f"__fragsplice_pending__:{source_id}")
        candidate = {
            "root": root,
            "source": source_id,
            "dag": dag,
            "plan": plan,
            "created_at": candidate_created_at,
            "slo": slo,
        }
        miss_deltas = []
        tardiness_deltas = []
        queue_inflation = []
        concentration = []
        candidate_latency = []
        plan_key = tuple(sorted(plan.items()))
        for seed in seeds:
            if seed not in baseline_cache:
                baseline_cache[seed] = state.simulate(None, seed)
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
            candidate_latency.append(latency)
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
            sum(queue_inflation) / count,
            sum(concentration) / count,
            sum(candidate_latency) / count,
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
        branch_order = {
            service: sorted(
                candidates[service],
                key=lambda device: (
                    self.latency_model.estimate(service, device, 0.5),
                    device,
                ),
            )
            for service in services
        }
        heap = []
        serial = 0
        elapsed = max(0.0, state.now - candidate_created_at)
        root_bound = self._lower_score(
            dag, candidates, {}, slo, elapsed=elapsed
        )
        heapq.heappush(heap, (root_bound, serial, 0, {}))
        preferred_plan = {
            service: branch_order[service][0] for service in services
        }
        if isinstance(initial_plan, dict) and all(
            initial_plan.get(service) in candidates[service]
            for service in services
        ):
            preferred_plan = {
                service: initial_plan[service] for service in services
            }
        incumbent_plan = preferred_plan
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
        return second[4] - first[4] > max(1e-3, 0.05 * max(first[4], 1e-6))

    def solve(self, info, snapshot, deployment, cloud_device):
        dag = info["dag"]
        source_id = info.get("source_id", "")
        task_context = info.get("task_context")
        task_context = task_context if isinstance(task_context, dict) else {}
        candidate_root = task_context.get("root_uuid")
        source_device = info.get("source_device", "")
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
        try:
            candidate_created_at = float(task_context.get("created_at"))
        except (TypeError, ValueError):
            candidate_created_at = state.now
        if not math.isfinite(candidate_created_at) or candidate_created_at <= 0.0:
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
        incumbent_plan = None
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
        elapsed_before_execution = max(0.0, state.now - candidate_created_at)
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
