from tango.common.testing import TangoTestCase
from tango.workspaces import LocalWorkspace


class TestStepIndexing(TangoTestCase):
    def test_step_indexing(self):
        run_name = "run1"
        config = {
            "steps": {
                "list": {"type": "range_step", "start": 0, "end": 3},
                "added": {
                    "type": "add_numbers",
                    "a_number": 2,
                    "b_number": {"type": "ref", "ref": "list", "key": 1},
                },
            }
        }
        self.run(config, name=run_name)
        workspace = LocalWorkspace(self.TEST_DIR / "workspace")
        result = workspace.step_result_for_run(run_name, "added")
        assert result == 3
