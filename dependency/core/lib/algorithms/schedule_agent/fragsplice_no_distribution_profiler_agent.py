import abc

from core.lib.common import ClassFactory, ClassType

from .fragsplice_agent import FragSpliceAgent

__all__ = ("FragSpliceNoDistributionProfilerAgent",)


@ClassFactory.register(
    ClassType.SCH_AGENT,
    alias="fragsplice_no_distribution_profiler",
)
class FragSpliceNoDistributionProfilerAgent(FragSpliceAgent, abc.ABC):
    """FragSplice with a fixed P50 cold profile and no online adaptation."""

    DISTRIBUTION_PROFILER_ENABLED = False
