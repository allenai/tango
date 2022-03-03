import time

from tango.common.testing import TangoTestCase
from tango.executors.multicore_executor import MulticoreExecutor
from tango.step_graph import StepGraph
from tango.workspaces import LocalWorkspace
from test_fixtures.package.steps import SleepPrintMaybeFail


class TestMulticoreExecutor(TangoTestCase):
    def test_ready_to_run_parallel(self):
        step_graph = StepGraph(
            {
                "step1": SleepPrintMaybeFail(string="hello", seconds=5, fail=False),
                "step2": SleepPrintMaybeFail(string="hi", seconds=5, fail=False),
            }
        )

        executor = MulticoreExecutor(workspace=LocalWorkspace(self.TEST_DIR), parallelism=2)

        step_states = executor._sync_step_states(step_graph)
        assert executor._get_steps_to_run(step_graph, step_states) == set(["step1", "step2"])

        executor.execute_step(step_graph["step1"])
        step_states = executor._sync_step_states(step_graph)
        assert executor._get_steps_to_run(step_graph, step_states) == set(["step2"])

    def test_ready_to_run_chain(self):
        step_graph = StepGraph.from_params(
            {
                "step1": {
                    "type": "sleep-print-maybe-fail",
                    "string": "string_to_pass_down",
                    "seconds": 5,
                    "fail": False,
                },
                "step2": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step1"},
                    "seconds": 5,
                    "fail": False,
                },
            }
        )

        executor = MulticoreExecutor(workspace=LocalWorkspace(self.TEST_DIR), parallelism=2)

        step_states = executor._sync_step_states(step_graph)
        assert executor._get_steps_to_run(step_graph, step_states) == set(["step1"])

        executor.execute_step(step_graph["step1"])
        step_states = executor._sync_step_states(step_graph)
        assert executor._get_steps_to_run(step_graph, step_states) == set(["step2"])

    def test_simple_execution_in_parallel(self):
        step_graph = StepGraph(
            {
                "step1": SleepPrintMaybeFail(string="hello", seconds=5, fail=False),
                "step2": SleepPrintMaybeFail(string="hi", seconds=5, fail=False),
            }
        )

        executor = MulticoreExecutor(workspace=LocalWorkspace(self.TEST_DIR), parallelism=2)

        start_time = time.time()
        executor.execute_step_graph(step_graph)
        end_time = time.time()
        time_taken = end_time - start_time
        assert time_taken < 10  # TODO: will this be flaky?

        assert len(executor.workspace.step_cache) == 2

    def test_more_processes_ready_than_parallelism(self):
        step_graph = StepGraph(
            {
                "step1": SleepPrintMaybeFail(string="hello", seconds=5, fail=False),
                "step2": SleepPrintMaybeFail(string="hi", seconds=5, fail=False),
                "step3": SleepPrintMaybeFail(string="howdy", seconds=5, fail=False),
            }
        )

        executor = MulticoreExecutor(workspace=LocalWorkspace(self.TEST_DIR), parallelism=2)
        start_time = time.time()
        executor.execute_step_graph(step_graph)
        end_time = time.time()
        time_taken = end_time - start_time
        assert 10 < time_taken < 20  # TODO: will this be flaky?

        assert len(executor.workspace.step_cache) == 3

    # def test_failing_step(self):
    #     step_graph = StepGraph.from_params(
    #         {
    #             "step1": {
    #                 "type": "sleep-print-maybe-fail",
    #                 "string": "string_to_pass_down",
    #                 "seconds": 5,
    #                 "fail": False,
    #             },
    #             "step2": {
    #                 "type": "sleep-print-maybe-fail",
    #                 "string": {"type": "ref", "ref": "step1"},
    #                 "seconds": 5,
    #                 "fail": False,
    #             },
    #             "step3": {
    #                 "type": "sleep-print-maybe-fail",
    #                 "string": "This is going to fail!",
    #                 "seconds": 3,
    #                 "fail": True,
    #             },
    #         }
    #     )
    #
    #     executor = MulticoreExecutor(workspace=LocalWorkspace(self.TEST_DIR), parallelism=1)
    #     # step_states = executor._sync_step_states(step_graph)
    #     # assert executor._get_steps_to_run(step_graph, step_states) == set(["step1", "step3"])
    #     #
    #     # executor.execute_step(step_graph["step1"])
    #     # step_states = executor._sync_step_states(step_graph)
    #     # assert executor._get_steps_to_run(step_graph, step_states) == set(["step2", "step3"])
    #     #
    #     # try:
    #     #     executor.execute_step(step_graph["step3"])
    #     # except RuntimeError:
    #     #     pass
    #     # step_states = executor._sync_step_states(step_graph)
    #     # print(step_states)
    #     # assert executor._get_steps_to_run(step_graph, step_states) == set(["step2"])
    #
    #     executor.execute_step_graph(step_graph)
    #
    #     assert len(executor.workspace.step_cache) == 2
