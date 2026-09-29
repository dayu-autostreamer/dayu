"""Register the STEADY ablations without duplicating the runtime adapter."""

from core.lib.common import ClassFactory, ClassType

from ..steady_agent.hook import SteadyAgent

__all__ = ("UnsteadyAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias="unsteady")
class UnsteadyAgent(SteadyAgent):
    def __init__(self, system, agent_id: int, sch_param: dict, unsteady_param: dict):
        mode = unsteady_param.get("schedule_type", "macro")
        if mode not in ("macro", "micro"):
            raise ValueError("unsteady schedule_type must be 'macro' or 'micro'")
        self.scheduler_mode = mode
        super().__init__(system, agent_id, sch_param, unsteady_param)
