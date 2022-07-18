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
        self.run(config)
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        assert len(list(latest_outputs.iterdir())) == num_other_files + 3

        # Running a single step with no dependencies should have a single output.
        self.run(config, step_name="strB")
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        assert len(list(latest_outputs.iterdir())) == num_other_files + 1

        # Running a single step with one or more dependencies will also run the step's dependencies.
        self.run(config, step_name="concatenated")
        latest_outputs = self.TEST_DIR / "workspace" / "latest"
        assert len(list(latest_outputs.iterdir())) == num_other_files + 3
