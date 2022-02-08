from tango import Format, JsonFormat, Step
from tango.common.testing import TangoTestCase


@Step.register("return_a_number")
class ReturnANumber(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()

    def run(self, what_number: int) -> int:  # type: ignore
        return what_number


class TestExperiment(TangoTestCase):
    def test_experiment_updates_latest_run_output(self, caplog):
        config = {
            "steps": {
                "a_number": {
                    "type": "return_a_number",
                    "what_number": 3,
                },
            }
        }

        self.run(config)
        assert (self.TEST_DIR / "workspace" / "latest" / "a_number").exists()

        fmt: Format = JsonFormat()
        data = fmt.read(self.TEST_DIR / "workspace" / "latest" / "a_number")
        assert data == 3

        config = {
            "steps": {
                "a_number": {
                    "type": "return_a_number",
                    "what_number": 5,
                },
            }
        }

        self.run(config)
        data = fmt.read(self.TEST_DIR / "workspace" / "latest" / "a_number")
        assert data == 5
