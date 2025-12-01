from core.lib.network import NodeInfo, PortInfo, NetworkAPIPath, NetworkAPIMethod, http_request, merge_address
from core.lib.common import SystemConstant


class HealthChecker:
    @staticmethod
    def check_processors_health():
        """check if all processors is healthy"""
        nodes = NodeInfo.get_edge_nodes() + [NodeInfo.get_cloud_node()]
        for node in nodes:
            node_processor_health_address = merge_address(
                NodeInfo.hostname2ip(node),
                port=PortInfo.get_component_port(SystemConstant.CONTROLLER.value),
                path=NetworkAPIPath.CONTROLLER_CHECK)
            response = http_request(url=node_processor_health_address,
                                    method=NetworkAPIMethod.CONTROLLER_CHECK)
            if not response or response.get('status', '') != 'ok':
                return False
        return True
