import abc
import threading

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent
from .full_plan_support import (
    START,
    apply_full_plan,
    deployment_from_snapshot,
    load_profiled_latency_model,
    service_names,
    topological_order,
    validate_profile_coverage,
    visible_replica_loads,
)

__all__ = ("IBDASHAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias="ibdash")
class IBDASHAgent(BaseAgent, abc.ABC):
    """Profile-based DAG earliest-finish baseline adapted from IBDASH.

    IBDASH's failure replication, model placement, and unmanaged-device
    availability are intentionally outside Dayu's fixed-deployment action
    space.  This hook retains its defining scheduling mechanism: measured
    service costs plus DAG dependencies are used to greedily minimize each
    stage's projected finish time.
    """

    def __init__(
        self,
        system,
        agent_id: int,
        configuration=None,
        latency_profile=None,
        profile_quantile=0.5,
        queue_state_max_age_s=1.5,
    ):
        super().__init__(system, agent_id)
        self.system = system
        self.agent_id = agent_id
        self.configuration, self.latency_model = load_profiled_latency_model(
            configuration,
            latency_profile,
        )
        self.profile_quantile = min(1.0, max(0.0, float(profile_quantile)))
        self.queue_state_max_age_s = max(0.0, float(queue_state_max_age_s))
        self.overhead_estimator = OverheadEstimator(
            "IBDASH",
            "scheduler/fragsplice",
            agent_id=agent_id,
        )
        self._lock = threading.RLock()
        self.last_decision = None

    def _solve(self, info, snapshot, dag, deployment):
        loads = visible_replica_loads(
            snapshot,
            self.latency_model,
            dag,
            deployment,
            self.profile_quantile,
            self.queue_state_max_age_s,
        )
        source_device = str(info.get("source_device") or "")
        finish = {START: 0.0}
        devices = {START: source_device}
        plan = {}
        schedulable_services = set(service_names(dag))

        for service in topological_order(dag):
            if service not in schedulable_services:
                continue
            candidates = deployment[service]
            best = None
            for device in candidates:
                predecessor_finish = []
                for predecessor in dag[service].get("prev_nodes", []):
                    ready = float(finish.get(predecessor, 0.0))
                    predecessor_device = str(devices.get(predecessor) or "")
                    if predecessor_device and predecessor_device != device:
                        ready += self.latency_model.estimate_transfer(
                            service,
                            device,
                            self.profile_quantile,
                        )
                    predecessor_finish.append(ready)
                ready = max(predecessor_finish, default=0.0)
                ready += self.latency_model.estimate_control(
                    service,
                    device,
                    self.profile_quantile,
                )
                ready += self.latency_model.estimate_dispatch(
                    service,
                    device,
                    self.profile_quantile,
                )
                replica = loads[(service, device)]
                start = max(ready, float(replica["workload"]))
                completion = start + float(replica["demand"])
                score = (completion, start, str(device))
                if best is None or score < best[0]:
                    best = (score, str(device), completion)
            _, device, completion = best
            plan[service] = device
            devices[service] = device
            finish[service] = completion

        return {
            "plan": plan,
            "projected_finish_s": max(
                (finish.get(service, 0.0) for service in plan),
                default=0.0,
            ),
        }

    def get_schedule_plan(self, info):
        with self.overhead_estimator, self._lock:
            dag = info["dag"]
            snapshot = self.system.get_scheduling_snapshot()
            deployment = deployment_from_snapshot(self.system, snapshot)
            validate_profile_coverage(
                self.latency_model,
                self.configuration,
                dag,
                deployment,
                "IBDASH",
            )
            result = self._solve(info, snapshot, dag, deployment)
            self.last_decision = result
            LOGGER.info(
                "[IBDASH] source=%s projected_finish=%.4fs plan=%s",
                info.get("source_id"),
                result["projected_finish_s"],
                result["plan"],
            )
            return apply_full_plan(
                self.configuration,
                dag,
                result["plan"],
                info.get("source_device"),
                self.cloud_device,
            )

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        # Keep the profile frozen so this remains an offline model-driven
        # baseline rather than acquiring FragSplice's feedback adaptation.
        pass

    def run(self):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
