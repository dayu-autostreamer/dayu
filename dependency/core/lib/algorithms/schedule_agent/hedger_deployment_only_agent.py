import abc

from core.lib.common import ClassFactory, ClassType, KubeConfig, LOGGER, TaskConstant
from core.lib.content import Task
from core.lib.algorithms.shared.hedger_ablation import HedgerDeploymentOnly

from .hedger_ablation_support import HedgerAblationAgentBase

__all__ = ("HedgerDeploymentOnlyAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hedger-deployment-only')
class HedgerDeploymentOnlyAgent(HedgerAblationAgentBase, abc.ABC):
    """Deployment-only ablation: Hedger deployment PPO plus heuristic offloading."""

    controller_cls = HedgerDeploymentOnly
    controller_alias = "hedger_deployment_only"

    def get_schedule_plan(self, info):
        source_id = info['source_id']
        source_edge_device = info['source_device']
        all_edge_devices = info['all_edge_devices']
        cloud_device = self.cloud_device
        all_devices = [*all_edge_devices, cloud_device]
        dag = info['dag']

        self.hedger.register_logical_topology(Task.extract_dag_from_dag_deployment(dag))
        self.hedger.register_physical_topology(info['all_edge_devices'], source_edge_device)
        self.hedger.register_state_buffer()

        configuration = self._normalize_mapping(self.default_configuration)
        should_force_default = bool(getattr(self.hedger, "should_force_default_decisions", lambda: False)())
        if should_force_default:
            offloading = self._normalize_mapping(self.default_offloading)
            used_default_offloading = True
        else:
            offloading = self.hedger.get_heuristic_offloading_plan(default_offloading=self.default_offloading)
            used_default_offloading = False
            if not offloading:
                offloading = self._normalize_mapping(self.default_offloading)
                used_default_offloading = True

        deployment_version = self.hedger.get_active_deployment_version()
        policy = {}
        policy.update(configuration)
        service_info = KubeConfig.get_service_nodes_dict()
        for service_name in dag:
            if service_name in service_info and service_name in offloading and offloading[service_name] in all_devices:
                dag[service_name]['service']['execute_device'] = offloading[service_name]
            elif service_name == TaskConstant.START.value:
                dag[service_name]['service']['execute_device'] = source_edge_device
            else:
                dag[service_name]['service']['execute_device'] = cloud_device
        policy.update({'dag': dag, 'deployment_version': deployment_version})

        service_names = [name for name in dag if name not in (TaskConstant.START.value, TaskConstant.END.value)]
        cloud_count = sum(
            1 for service_name in service_names
            if dag[service_name]['service'].get('execute_device') == cloud_device
        )
        LOGGER.info(
            f"[HedgerDeploymentOnly][Schedule] source={source_id}, services={len(service_names)}, "
            f"deployment_version={deployment_version}, cloud={cloud_count}/{len(service_names) if service_names else 0}, "
            f"used_default_offloading={used_default_offloading}"
        )
        return policy
