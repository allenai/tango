from tango.common.testing import TangoTestCase


class TestExperiment(TangoTestCase):
    def setup_method(self):
        super().setup_method()
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

    def test_experiment(self, caplog):
        num_other_files = 2
        self.run(
            self.config,
            include_package=["test_fixtures.package.steps"],
            multicore=True,
            parallelism=2,
        )
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        assert len(list(latest_outputs.iterdir())) == num_other_files + 1

    # def test_experiment_with_overrides(self, caplog):
    #     # TODO: uncomment once run_name PR is merged, and changes to use a single run name are added.
    #     import json
    #
    #     num_other_files = 2
    #     self.run(
    #         self.config,
    #         include_package=["test_fixtures.package.steps"],
    #         multicore=True,
    #         parallelism=2,
    #         overrides=json.dumps({"steps.step1.fail": False}),
    #     )
    #     latest_outputs = self.TEST_DIR / "workspace" / "latest"
    #     print(list(latest_outputs.iterdir()))
    #     assert len(list(latest_outputs.iterdir())) == num_other_files + 3
