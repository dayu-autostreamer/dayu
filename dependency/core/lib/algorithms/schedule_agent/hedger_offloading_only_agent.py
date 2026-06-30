import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerOffloadingOnly

from .hedger_ablation_support import HedgerAblationAgentBase

__all__ = ("HedgerOffloadingOnlyAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hedger-offloading-only')
class HedgerOffloadingOnlyAgent(HedgerAblationAgentBase, abc.ABC):
    """Offloading-only ablation: heuristic deployment plus Hedger offloading PPO."""

    controller_cls = HedgerOffloadingOnly
    controller_alias = "hedger_offloading_only"
