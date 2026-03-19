import abc
import random

from .base_selection_policy import BaseSelectionPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('RandomSelectionPolicy',)


@ClassFactory.register(ClassType.SCH_SELECTION_POLICY, alias='random')
class RandomSelectionPolicy(BaseSelectionPolicy, abc.ABC):
    def __init__(self, system, agent_id, scope='node_set'):
        super().__init__(system=system, agent_id=agent_id, scope=scope)

    def __call__(self, info):
        node_set = self.get_candidate_node_set(info)
        source_id = info['source']['id']
        if not node_set:
            LOGGER.warning(f"[Source Node Selection] (source {source_id}) Node set is empty.")
            return None

        selected_node = random.choice(node_set)
        LOGGER.info(f'[Source Node Selection] (source {source_id}) Select node {selected_node} '
                    f'from candidate node set {node_set} (scope: {self.scope}).')
        return selected_node
