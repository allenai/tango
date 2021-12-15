from tango import Step
from tango.common.testing import TangoTestCase


@Step.register("make_number")
class MakeNumber(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, what_number: int) -> int:  # type: ignore
        return what_number


stored_number = None


@Step.register("store_number_globally")
class StoreNumberGlobally(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self, number: int) -> None:  # type: ignore
        global stored_number
        stored_number = number


class TestExperiment(TangoTestCase):
    def test_experiment(self, caplog):
        config = {
            "steps": {
                "a_number": {
                    "type": "make_number",
                    "what_number": 3,
                },
                "store_number": {
                    "type": "store_number_globally",
                    "number": {"type": "ref", "ref": "a_number"},
                },
            }
        }

        global stored_number
        assert stored_number is None
        self.run(config)
        assert stored_number == 3
