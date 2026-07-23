import abc

from core.lib.common import ClassFactory, ClassType

from .fragsplice import FragSpliceStagewiseEFTOptimizer
from .fragsplice_agent import FragSpliceAgent

__all__ = ("FragSpliceNoFullPlanOptimizerAgent",)


@ClassFactory.register(
    ClassType.SCH_AGENT,
    alias="fragsplice_no_full_plan_optimizer",
)
class FragSpliceNoFullPlanOptimizerAgent(FragSpliceAgent, abc.ABC):
    """FragSplice with stage-wise EFT instead of joint full-plan search."""

    FULL_PLAN_OPTIMIZER_ENABLED = False
    OPTIMIZER_CLS = FragSpliceStagewiseEFTOptimizer
