from tango.common.testing import TangoTestCase
from tango.executor import Executor
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspaces import LocalWorkspace
from test_fixtures.package.steps import SleepPrintMaybeFail  # noqa:F401


@Step.register("sum_numbers")
class AdditionStep(Step):

    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a: int, b: int) -> int:  # type: ignore
        return a + b


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
