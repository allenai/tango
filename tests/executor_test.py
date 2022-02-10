from pathlib import Path

from tango.common.params import Params
from tango.common.testing import TangoTestCase
from tango.executor import Executor
from tango.local_workspace import ExecutorMetadata, LocalWorkspace
from tango.step import Step


class AdditionStep(Step):

    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a: int, b: int) -> int:  # type: ignore
        return a + b


class TestMetadata(TangoTestCase):
    def test_metadata(self):
        metadata = ExecutorMetadata("some_step")
        metadata.save(self.TEST_DIR)

        if (Path.cwd() / ".git").exists():
            assert metadata.git is not None
            assert metadata.git.commit is not None
            assert metadata.git.remote is not None
            assert "allenai/tango" in metadata.git.remote

        metadata2 = ExecutorMetadata.from_params(
            Params.from_file(self.TEST_DIR / "executor-metadata.json")
        )
        assert metadata == metadata2


class TestExecutor(TangoTestCase):
    def test_executor(self):
        workspace = LocalWorkspace(self.TEST_DIR)
        step = AdditionStep(a=1, b=2)
        step_graph = {"sum": step}
        executor = Executor(workspace)
        assert len(executor.workspace.step_cache) == 0
        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 1
        assert executor.workspace.step_cache[step] == 3
