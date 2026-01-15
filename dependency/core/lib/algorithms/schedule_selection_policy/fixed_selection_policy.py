import abc

from .base_selection_policy import BaseSelectionPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('FixedSelectionPolicy',)


@ClassFactory.register(ClassType.SCH_SELECTION_POLICY, alias='fixed')
class FixedSelectionPolicy(BaseSelectionPolicy, abc.ABC):
    def __init__(self, system, agent_id, fixed_value=0, fixed_type="position"):
        self.fixed_value = fixed_value
        self.fixed_type = fixed_type

        if self.fixed_type == "position":
            if not isinstance(self.fixed_value, int) or self.fixed_value < 0:
                LOGGER.warning(f'Position value {self.fixed_value} is illegal, using 0 as default.')
                self.fixed_value = 0
        elif self.fixed_type == "hostname":
            if not isinstance(self.fixed_value, str):
                LOGGER.warning(f'Hostname value {self.fixed_value} is illegal, using empty string as default.')
                self.fixed_value = ''
        else:
            LOGGER.warning(f'Type of fixed selection policy "{self.fixed_type}"is not supported, '
                           f'only "position" and "hostname" are supported."')
            LOGGER.warning('Using "position" as default type.')
            self.fixed_type = 'position'

    def __call__(self, info):
        node_set = info['node_set']
        source_id = info['source']['id']
        if not node_set:
            LOGGER.warning(f"[Source Node Selection] (source {source_id}) Node set is empty.")
            return None

        if self.fixed_type == "position":
            if self.fixed_value < len(node_set):
                LOGGER.info(f'[Source Node Selection] (source {source_id}) Select node {self.fixed_value} from '
                            f'node set {node_set} (position:{self.fixed_value})).')
                return node_set[self.fixed_value]
            else:
                LOGGER.warning(f'[Source Node Selection] (source {source_id}) Position value {self.fixed_value} '
                               f'is out of node_set range ({node_set}), select the first node {node_set[0]}.')
                return node_set[0]
        elif self.fixed_type == "hostname":
            if self.fixed_value in node_set:
                LOGGER.info(f'[Source Node Selection] (source {source_id}) Select node {self.fixed_value} from '
                            f'node set {node_set} (hostname:{self.fixed_value})).')
                return self.fixed_value
            else:
                LOGGER.warning(f'[Source Node Selection] (source {source_id}) Hostname value {self.fixed_value} '
                               f'is not in node_set ({node_set}), select the first node {node_set[0]}.')
                return node_set[0]
