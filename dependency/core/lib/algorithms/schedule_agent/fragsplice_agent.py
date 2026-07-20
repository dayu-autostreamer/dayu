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
        revision_getter = getattr(system, "runtime_directory_revision", None)
        revision = revision_getter() if callable(revision_getter) else 0
        self.latency_model = GlobalInstanceManager.get_instance(
            FragSpliceLatencyModel,
            ("fragsplice", id(system), revision),
            profile=profile,
        )
        self.optimizer = FragSpliceOptimizer(
            self.latency_model,
            default_slo_s=latency_slo_s,
            scenario_count=scenario_count,
            max_scenarios=max_scenarios,
            search_time_limit_s=search_time_limit_s,
            random_seed=random_seed,
            queue_state_max_age_s=queue_state_max_age_s,
        )
        self.overhead_estimator = OverheadEstimator(
            "FragSplice", "scheduler/fragsplice", agent_id=agent_id
        )
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
            deployment = snapshot.get("deployment")
            if not isinstance(deployment, dict):
                deployment = self.system.runtime_service_nodes()
            self._validate_profile_coverage(dag, deployment)
            result = self.optimizer.solve(
                decision_info,
                snapshot,
                deployment,
                self.cloud_device,
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
                "[FragSplice] source=%s plans=%s scenarios=%s evaluated=%s "
                "optimal=%s unschedulable=%s intrinsic_slo_infeasible=%s "
                "budget_exhausted=%s score=%s overhead=%.4fs",
                info.get("source_id"),
                result["candidate_count"],
                result["scenario_count"],
                len(result["evaluated"]),
                result["optimality_proven"],
                result["unschedulable"],
                result["intrinsic_slo_infeasible"],
                result["budget_exhausted"],
                tuple(round(item, 6) for item in result["score"]),
                result["search_seconds"],
            )
            return policy

    def update_task(self, task):
        with self._lock:
            updated = self.latency_model.update_task(task)
            if updated and self.latency_profile_path:
                self.latency_model.save(
                    self.latency_profile_path,
                    deployment=self.system.runtime_service_nodes(),
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
