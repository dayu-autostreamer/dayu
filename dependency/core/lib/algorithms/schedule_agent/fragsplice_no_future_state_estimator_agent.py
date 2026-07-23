import abc

from core.lib.common import ClassFactory, ClassType

from .fragsplice_agent import FragSpliceAgent

__all__ = ("FragSpliceNoFutureStateEstimatorAgent",)


@ClassFactory.register(
    ClassType.SCH_AGENT,
    alias="fragsplice_no_future_state_estimator",
)
class FragSpliceNoFutureStateEstimatorAgent(FragSpliceAgent, abc.ABC):
    """FragSplice using only live queue state, without future commitments."""

    FUTURE_STATE_ESTIMATOR_ENABLED = False
