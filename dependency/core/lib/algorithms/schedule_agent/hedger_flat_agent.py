import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerFlat

from .hedger_ablation_support import HedgerAblationAgentBase

__all__ = ("HedgerFlatAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hedger-flat')
class HedgerFlatAgent(HedgerAblationAgentBase, abc.ABC):
    controller_cls = HedgerFlat
    controller_alias = "hedger_flat"
