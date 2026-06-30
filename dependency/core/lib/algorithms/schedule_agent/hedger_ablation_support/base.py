import abc
import copy

from core.lib.common import GlobalInstanceManager

from ..hedger_agent import HedgerAgent

__all__ = ("HedgerAblationAgentBase",)


class HedgerAblationAgentBase(HedgerAgent, abc.ABC):
    controller_cls = None
    controller_alias = "hedger_ablation"

    def register_hedger(self, hedger_id='hedger'):
        if self.hedger is None:
            hedger_config = copy.deepcopy(self.system.hedger_config)
            hedger_config.setdefault("agent_id", self.agent_id)
            self.hedger = GlobalInstanceManager.get_instance(
                self.controller_cls,
                f"{self.controller_alias}_{self.agent_id}",
                config=hedger_config,
            )
