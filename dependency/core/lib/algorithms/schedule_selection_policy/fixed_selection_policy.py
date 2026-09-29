import abc

from .base_selection_policy import BaseSelectionPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('FixedSelectionPolicy',)


@ClassFactory.register(ClassType.SCH_SELECTION_POLICY, alias='fixed')
class FixedSelectionPolicy(BaseSelectionPolicy, abc.ABC):
    def __init__(self, system, agent_id, fixed_value=0, fixed_type="position", scope='selected_edge_nodes'):
        super().__init__(system=system, agent_id=agent_id, scope=scope)
        self.fixed_value = fixed_value
        self.fixed_type = fixed_type

        if self.fixed_type == "position":
            if not isinstance(self.fixed_value, int) or self.fixed_value < 0:
                raise ValueError("fixed source position must be a non-negative integer")
        elif self.fixed_type == "hostname":
            if not isinstance(self.fixed_value, str) or not self.fixed_value.strip():
                raise ValueError("fixed source hostname must be a non-empty string")
            self.fixed_value = self.fixed_value.strip()
        else:
            raise ValueError("fixed source type must be 'position' or 'hostname'")

    def __call__(self, info):
        node_set = self.get_candidate_node_set(info)
        source_id = info['source']['id']
        if not node_set:
            raise ValueError(f"source {source_id!r} has no permitted source candidate nodes")

        if self.fixed_type == "position":
            if self.fixed_value < len(node_set):
                LOGGER.info(f'[Source Node Selection] (source {source_id}) Select node {self.fixed_value} from '
                            f'candidate node set {node_set} (position:{self.fixed_value}, scope:{self.scope})).')
                return node_set[self.fixed_value]
            else:
                raise ValueError(
                    f"fixed source position {self.fixed_value} is outside the permitted "
                    f"candidate range for source {source_id!r}: {node_set}"
                )
        elif self.fixed_type == "hostname":
            if self.fixed_value in node_set:
                LOGGER.info(f'[Source Node Selection] (source {source_id}) Select node {self.fixed_value} from '
                            f'candidate node set {node_set} (hostname:{self.fixed_value}, scope:{self.scope})).')
                return self.fixed_value
            else:
                raise ValueError(
                    f"fixed source node {self.fixed_value!r} is not a permitted candidate "
                    f"for source {source_id!r}: {node_set}"
                )
