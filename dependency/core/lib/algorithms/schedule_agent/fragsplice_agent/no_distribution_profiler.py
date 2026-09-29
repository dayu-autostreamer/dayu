import abc

from core.lib.common import ClassFactory, ClassType

from .agent import FragSpliceAgent
from .optimizer import FragSpliceRandomInputOptimizer

__all__ = ("FragSpliceNoDistributionProfilerAgent",)


@ClassFactory.register(
    ClassType.SCH_AGENT,
    alias="fragsplice_no_distribution_profiler",
)
class FragSpliceNoDistributionProfilerAgent(FragSpliceAgent, abc.ABC):
    """FragSplice whose downstream modules receive only random inputs."""

    DISTRIBUTION_PROFILER_ENABLED = False
    OPTIMIZER_CLS = FragSpliceRandomInputOptimizer

    def _rerank_rolling_result(self, result, info, deployment, revision):
        """Keep the background result without reading foreground telemetry.

        The normal foreground repair path consumes measured queue state and
        exact active commitments.  Using it here would silently restore the
        invocation-token signal that this ablation removes, so cached plans
        are consumed exactly as produced from their synthetic random state.
        """

        del info, deployment, revision
        result["online_rerank_changed"] = False
        result["online_active_commitment_tasks"] = None
        result["online_rerank_delta_tasks"] = None
        result["actual_state_consumed"] = False
        result["prediction_is_synthetic"] = True
        result["planning_cost_domain"] = "random_uninformed"
        return result
