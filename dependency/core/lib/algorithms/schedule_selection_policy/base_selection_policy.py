import abc

from core.lib.scheduling.source_selection import (
    VALID_SOURCE_SELECTION_SCOPES,
    normalize_source_selection_scope,
    source_selection_candidates,
)


class BaseSelectionPolicy(metaclass=abc.ABCMeta):
    VALID_SCOPES = tuple(sorted(VALID_SOURCE_SELECTION_SCOPES))

    def __init__(self, system=None, agent_id=None, scope='selected_edge_nodes'):
        self.system = system
        self.agent_id = agent_id
        self.scope = self.normalize_scope(scope)

    @classmethod
    def normalize_scope(cls, scope):
        return normalize_source_selection_scope(scope)

    def get_candidate_node_set(self, info):
        return source_selection_candidates(info, self.scope)

    def __call__(self, info):
        raise NotImplementedError
