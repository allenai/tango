from pathlib import Path

from tango.common.params import Params
from tango.common.testing import TangoTestCase
from tango.executors import Executor
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspace import StepExecutionMetadata
from tango.workspaces import LocalWorkspace


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
        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 1
        assert executor.workspace.step_cache[step] == 3
