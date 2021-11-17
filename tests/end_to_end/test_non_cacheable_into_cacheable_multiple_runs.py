import random

from tango import Step
from tango.common.testing import TangoTestCase


@Step.register("give_me_a_number")
class GiveMeANumber(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self, what_number: int) -> int:  # type: ignore
        return what_number


@Step.register("random_int")
class RandomInt(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, lower_bound: int, upper_bound: int) -> int:  # type: ignore
        return random.randint(lower_bound, upper_bound)


class TestExperiment(TangoTestCase):
    def test_experiment(self, caplog):
        config = {
            "steps": {
                "a_number": {
                    "type": "give_me_a_number",
                    "what_number": 3,
                },
                "final_number": {
                    "type": "random_int",
                    "lower_bound": 0,
                    "upper_bound": {"type": "ref", "ref": "a_number"},
                },
            }
        }

        self.run(config)
        self.run(config, overrides={"steps.final_number.lower_bound": 1})
