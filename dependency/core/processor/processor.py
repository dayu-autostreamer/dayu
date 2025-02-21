from core.lib.content import Task
from core.lib.common import Context


class Processor:
    def __init__(self):
        self.scenario_extractors_text = Context.get_parameter('SCENARIOS_EXTRACTORS', direct=False)

        self.scenario_extractors = []
        for scenario_extractor_text in self.scenario_extractors_text:
            self.scenario_extractors.append(
                Context.get_algorithm('PRO_SCENARIO', scenario_extractor_text)
            )

    def __call__(self, task: Task):
        raise NotImplementedError

    def get_scenario(self, result, task):
        scenarios = {}

        for scenario_extractor_text, scenario_extractor in zip(self.scenario_extractors_text, self.scenario_extractors):
            scenarios.update({scenario_extractor_text: scenario_extractor(result, task)})

        task.add_scenario(scenarios)

        return task
