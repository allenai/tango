import time

from tango.workspaces import LocalWorkspace
from tango.common.testing import TangoTestCase
from tango.executors.multicore_executor import MulticoreExecutor
from tango.step import Step

from tango.step_graph import StepGraph


class PrintSleepStep(Step):
    def run(self, input: str, seconds: int = 5) -> str:
        time.sleep(seconds)
        print(input)
        return input


class TestMulticoreExecutor(TangoTestCase):
    def test_execution(self):
        step_graph = StepGraph({"step1": PrintSleepStep(input="hello"), "step2": PrintSleepStep(input="hi")})

        executor = MulticoreExecutor(workspace=LocalWorkspace(self.TEST_DIR), parallelism=2)
        start_time = time.time()
        executor.execute_step_graph(step_graph)
        end_time = time.time()
        print(end_time - start_time)
        assert False
