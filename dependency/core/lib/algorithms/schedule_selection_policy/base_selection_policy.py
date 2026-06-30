import abc

from core.lib.common import LOGGER
from core.lib.network import NodeInfo


class BaseSelectionPolicy(metaclass=abc.ABCMeta):
    VALID_SCOPES = ('selected_edge_nodes', 'all_edge_nodes')

    def __init__(self, system=None, agent_id=None, scope='selected_edge_nodes'):
        self.system = system
        self.agent_id = agent_id
        self.scope = self.normalize_scope(scope)

    @classmethod
    def normalize_scope(cls, scope):
        normalized_scope = scope or 'selected_edge_nodes'
        if normalized_scope not in cls.VALID_SCOPES:
            LOGGER.warning(
                f'Unsupported selection scope "{normalized_scope}". '
                f'Falling back to "selected_edge_nodes". '
                f'Supported scopes: {", ".join(cls.VALID_SCOPES)}.'
            )
            return 'selected_edge_nodes'
        return normalized_scope

    def get_candidate_node_set(self, info):
        node_set = list(dict.fromkeys(info.get('node_set') or []))

        if self.scope == 'selected_edge_nodes':
            return node_set

        if self.scope == 'all_edge_nodes':
            all_edge_nodes = info.get('all_edge_nodes')
            if all_edge_nodes is None:
                all_edge_nodes = NodeInfo.get_all_edge_nodes()
            return list(dict.fromkeys(all_edge_nodes or []))

        LOGGER.warning(
            f'Unsupported selection scope "{self.scope}" at runtime. '
            'Falling back to "selected_edge_nodes".'
        )
        return node_set

    def __call__(self, info):
        raise NotImplementedError
