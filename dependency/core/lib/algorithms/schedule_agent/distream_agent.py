import abc
import threading

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent
from .full_plan_support import (
    apply_full_plan,
    deployment_from_snapshot,
    load_profiled_latency_model,
    service_names,
    validate_profile_coverage,
    visible_replica_loads,
)

__all__ = ("DistreamAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias="distream")
class DistreamAgent(BaseAgent, abc.ABC):
    """Reactive workload-balancing baseline adapted from Distream.

    Dayu cannot migrate an already committed invocation or repartition a
    pipeline.  The compatible Distream mechanism is therefore applied at task
    admission: each service is assigned to the replica with the smallest
    currently visible projected workload, and the resulting full DAG mapping
    is committed once.
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
            "Distream",
            "scheduler/fragsplice",
            agent_id=agent_id,
        )
        self._lock = threading.RLock()
        self.last_decision = None

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
                "Distream",
            )
            loads = visible_replica_loads(
                snapshot,
                self.latency_model,
                dag,
                deployment,
                self.profile_quantile,
                self.queue_state_max_age_s,
            )
            plan = {}
            projected = {}
            for service in service_names(dag):
                device = min(
                    deployment[service],
                    key=lambda candidate: (
                        loads[(service, candidate)]["workload"]
                        + loads[(service, candidate)]["demand"],
                        loads[(service, candidate)]["waiting_count"],
                        str(candidate),
                    ),
                )
                plan[service] = str(device)
                projected[service] = (
                    loads[(service, device)]["workload"]
                    + loads[(service, device)]["demand"]
                )
            self.last_decision = {
                "plan": plan,
                "projected_workload_s": projected,
            }
            LOGGER.info(
                "[Distream] source=%s projected_workload=%s plan=%s",
                info.get("source_id"),
                {key: round(value, 4) for key, value in projected.items()},
                plan,
            )
            return apply_full_plan(
                self.configuration,
                dag,
                plan,
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
        pass

    def run(self):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
