import abc

from core.lib.common import ClassFactory, ClassType

from .fragsplice import FragSpliceStagewiseEFTOptimizer
from .fragsplice_agent import FragSpliceAgent

__all__ = ("FragSpliceNoPlanOptimizerAgent",)


@ClassFactory.register(
    ClassType.SCH_AGENT,
    alias="fragsplice_no_plan_optimizer",
)
class FragSpliceNoPlanOptimizerAgent(FragSpliceAgent, abc.ABC):
    """FragSplice with stage-wise EFT instead of the Plan Optimizer."""

    PLAN_OPTIMIZER_ENABLED = False
    OPTIMIZER_CLS = FragSpliceStagewiseEFTOptimizer
