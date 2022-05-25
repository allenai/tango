from shutil import copytree

import pytest

from tango import Step
from tango.common.testing import TangoTestCase
from tango.step_info import StepState
from tango.workspaces import LocalWorkspace


class AdditionStep(Step):
    def run(self, a: int, b: int) -> int:  # type: ignore
        return a + b


class TestLocalWorkspace(TangoTestCase):
    def test_local_workspace_one_step(self):
        workspace = LocalWorkspace(self.TEST_DIR)
        step = AdditionStep(a=1, b=2)

        with pytest.raises(KeyError):
            # This can't possibly work because the workspace has never seen that step before.
            step_info = workspace.step_info(step.unique_id)
            assert step_info.state == StepState.INCOMPLETE
        step_info = workspace.step_info(step)
        assert step_info.state == StepState.INCOMPLETE

        result = step.result(workspace)
        assert result == 3

        step_info = workspace.step_info(step.unique_id)
        assert step_info.state == StepState.COMPLETED
        step_info = workspace.step_info(step)
        assert step_info.state == StepState.COMPLETED

    def test_local_workspace_two_steps(self):
        workspace = LocalWorkspace(self.TEST_DIR)
        step1 = AdditionStep(a=1, b=2)
        step2 = AdditionStep(a=step1, b=3)

        step_info = workspace.step_info(step2)
        assert step_info.state == StepState.INCOMPLETE
        step_info = workspace.step_info(step2.unique_id)
        assert step_info.state == StepState.INCOMPLETE
        assert step1.unique_id in step_info.dependencies
        step_info = workspace.step_info(step1.unique_id)
        assert step_info.state == StepState.INCOMPLETE
        step_info = workspace.step_info(step1)
        assert step_info.state == StepState.INCOMPLETE

        result = step2.result(workspace)
        assert result == 6

        for step in [step1, step2]:
            step_info = workspace.step_info(step.unique_id)
            assert step_info.state == StepState.COMPLETED
            step_info = workspace.step_info(step)
            assert step_info.state == StepState.COMPLETED

    def test_local_workspace_upgrade_v1_to_v2(self):
        workspace_dir = self.TEST_DIR / "workspace"
        copytree(
            self.FIXTURES_ROOT / "v1_local_workspace",
            workspace_dir,
            symlinks=True,
        )
        workspace = LocalWorkspace(workspace_dir)
        step_info = workspace.step_info("SubtractionStep-YCdedqjmmd9GUFi96VzPXD5tAVho3CTz")
        assert step_info.state == StepState.COMPLETED
        dependencies = list(step_info.dependencies)

        # Make sure all the dependencies are there.
        while len(dependencies) > 0:
            step_info = workspace.step_info(dependencies.pop())
            dependencies.extend(step_info.dependencies)
