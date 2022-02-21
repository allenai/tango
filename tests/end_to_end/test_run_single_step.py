from tango.common.testing import TangoTestCase


class TestRunSingleStep(TangoTestCase):
    def test_run_single_step(self):
        config = {
            "steps": {
                "strA": {"type": "string", "result": "Hello, "},
                "strB": {"type": "string", "result": "World"},
                "concatenated": {
                    "type": "concat_strings",
                    "string1": {"type": "ref", "ref": "strA"},
                    "string2": {"type": "ref", "ref": "strB"},
                },
            }
        }

        num_other_files = 2  # out.log and stepinfo.json

        # Regular run contains all step outputs.
        self.run(config, include_package=["test_fixtures.package.steps"])
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        assert len(list(latest_outputs.iterdir())) == num_other_files + 3

        # Running a single step with no dependencies should have a single output.
        self.run(config, step_name="strB", include_package=["test_fixtures.package.steps"])
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        assert len(list(latest_outputs.iterdir())) == num_other_files + 1

        # Running a single step with one or more dependencies should still have a single output.
        # Note: this is really up to us. The cache would still contain the outputs for intermediate steps.
        # The logic here is that for this particular "run", we only care about the specified step's output.
        # However, this can be changed if we feel that all intermediate steps should also be part of the run's
        # output.
        self.run(config, step_name="concatenated", include_package=["test_fixtures.package.steps"])
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        assert len(list(latest_outputs.iterdir())) == num_other_files + 1
