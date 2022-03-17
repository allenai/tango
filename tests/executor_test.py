from pathlib import Path

from tango.common.params import Params
from tango.common.testing import TangoTestCase
from tango.executor import Executor
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspace import StepExecutionMetadata
from tango.workspaces import LocalWorkspace
from test_fixtures.package.steps import SleepPrintMaybeFail  # noqa:F401


@Step.register("sum_numbers")
class AdditionStep(Step):

    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a: int, b: int) -> int:  # type: ignore
        return a + b


class TestMetadata(TangoTestCase):
    def test_metadata(self):
        metadata = StepExecutionMetadata("some_step")
        metadata.save(self.TEST_DIR)

        if (Path.cwd() / ".git").exists():
            assert metadata.git is not None
            assert metadata.git.commit is not None
            assert metadata.git.remote is not None
            assert "allenai/tango" in metadata.git.remote

        metadata2 = StepExecutionMetadata.from_params(
            Params.from_file(self.TEST_DIR / "execution-metadata.json")
        )
        assert metadata == metadata2


class TestExecutor(TangoTestCase):
    def test_executor(self):
        workspace = LocalWorkspace(self.TEST_DIR)
        step = AdditionStep(a=1, b=2)
        step_graph = StepGraph.from_params({"sum": {"type": "sum_numbers", "a": 1, "b": 2}})
        executor = Executor(workspace)
        assert len(executor.workspace.step_cache) == 0
        output = executor.execute_step_graph(step_graph)
        assert "sum" in output.successful
        assert len(executor.workspace.step_cache) == 1
        assert executor.workspace.step_cache[step] == 3

    def test_executor_with_failing_steps(self):
        workspace = LocalWorkspace(self.TEST_DIR)
        step_graph = StepGraph.from_params(
            {
                "successful_step": {
                    "type": "sleep-print-maybe-fail",
                    "string": "This ran perfectly.",
                    "seconds": 0,
                    "fail": False,
                },
                "failing_step": {
                    "type": "sleep-print-maybe-fail",
                    "string": "This should fail.",
                    "seconds": 0,
                    "fail": True,
                },
                "dependent_step": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "failing_step"},
                    "seconds": 0,
                    "fail": False,
                },
            }
        )
        executor = Executor(workspace)
        assert len(executor.workspace.step_cache) == 0
        output = executor.execute_step_graph(step_graph)
        assert "successful_step" in output.successful
        assert "failing_step" in output.failed
        assert "dependent_step" in output.not_run
        assert len(executor.workspace.step_cache) == 1
