import pytest

from tango.common.exceptions import CliRunError
from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase


class TestExperiment(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        initialize_logging()
        self.config = {
            "steps": {
                "step1": {
                    "type": "sleep-print-maybe-fail",
                    "string": "string_to_pass_down",
                    "seconds": 1,
                    "fail": True,
                },
                "step2": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step1"},
                    "seconds": 1,
                    "fail": False,
                },
                "step3": {
                    "type": "sleep-print-maybe-fail",
                    "string": "This may or may not fail!",
                    "seconds": 3,
                    "fail": False,
                },
            }
        }

    def teardown_method(self):
        super().teardown_method()
        teardown_logging()

    def test_experiment(self, caplog):
        with pytest.raises(CliRunError):
            self.run(
                self.config,
                include_package=["test_fixtures.package.steps"],
                multicore=True,
                parallelism=2,
            )
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        num_executed = 0
        for out in latest_outputs.iterdir():
            if (out / "execution-metadata.json").exists():
                num_executed += 1
        assert num_executed == 1

    def test_experiment_with_overrides(self, caplog):
        import json

        self.run(
            self.config,
            include_package=["test_fixtures.package.steps"],
            multicore=True,
            parallelism=2,
            overrides=json.dumps({"steps.step1.fail": False}),
        )
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        num_executed = 0
        for out in latest_outputs.iterdir():
            if (out / "execution-metadata.json").exists():
                num_executed += 1
        assert num_executed == 3
