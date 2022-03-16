from tango import Step
from tango.common.testing import TangoTestCase, run_experiment
from test_fixtures.package.steps import MakeNumber  # noqa:F401

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


class TestExperimentMulticore(TangoTestCase):
    def test_experiment(self, caplog):
        file_name = self.TEST_DIR / "number_file.txt"
        assert not file_name.exists()
        with run_experiment(
            {
                "steps": {
                    "a_number": {
                        "type": "make_number",
                        "what_number": 3,
                    },
                    "store_number": {
                        "type": "store_number_in_file",
                        "number": {"type": "ref", "ref": "a_number"},
                        "file_name": str(file_name),
                    },
                }
            },
            multicore=True,
            include_package=["test_fixtures.package.steps"],
        ):
            with open(file_name) as file_ref:
                number = file_ref.read()

            assert int(number) == 3
