import abc

from core.lib.common import LOGGER
from core.lib.network import NodeInfo


class BaseSelectionPolicy(metaclass=abc.ABCMeta):
    def __init__(self, system=None, agent_id=None, scope='node_set'):
        self.system = system
        self.agent_id = agent_id
        self.scope = scope or 'node_set'

    def get_candidate_node_set(self, info):
        node_set = list(dict.fromkeys(info.get('node_set') or []))

        if self.scope in ('node_set', 'selected_edge_nodes', 'source_bound'):
            return node_set

        if self.scope in ('all_edge_nodes', 'cluster_edge_nodes', 'cluster'):
            all_edge_nodes = info.get('all_edge_nodes')
            if all_edge_nodes is None:
                all_edge_nodes = NodeInfo.get_all_edge_nodes()
            return list(dict.fromkeys(all_edge_nodes or []))

        LOGGER.warning(f'Unknown selection scope "{self.scope}", falling back to source node set.')
        return node_set

    def __call__(self, info):
        raise NotImplementedError
