from core.lib.network import NodeInfo, PortInfo, NetworkAPIPath, NetworkAPIMethod, http_request, merge_address
from core.lib.common import SystemConstant, LOGGER


class HealthChecker:
    @staticmethod
    def check_processors_health():
        """check if all processors is healthy"""
        nodes = NodeInfo.get_edge_nodes() + [NodeInfo.get_cloud_node()]
        LOGGER.debug(f'[HEALTH CHECK] health checking nodes: {nodes}')
        for node in nodes:
            node_processor_health_address = merge_address(
                NodeInfo.hostname2ip(node),
                port=PortInfo.get_component_port(SystemConstant.CONTROLLER.value),
                path=NetworkAPIPath.CONTROLLER_CHECK)
            response = http_request(url=node_processor_health_address,
                                    method=NetworkAPIMethod.CONTROLLER_CHECK)
            if not response or response.get('status', '') != 'ok':
                LOGGER.debug(f'[HEALTH CHECK] node {node} processor health check failed.')
                return False
            LOGGER.debug(f'[HEALTH CHECK] node {node} processor health check succeed.')

        return True
