import time

import pytest

from tango.common.logging import initialize_logging
from tango.common.testing import TangoTestCase
from tango.executors.multicore_executor import MulticoreExecutor
from tango.step_graph import StepGraph
from tango.workspaces import LocalWorkspace
from test_fixtures.package.steps import SleepPrintMaybeFail


class TestMulticoreExecutor(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        initialize_logging()

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

    @pytest.mark.parametrize("parallelism", [1, 2, 3])
    def test_failing_step_no_downstream_task(self, parallelism):
        step_graph = StepGraph.from_params(
            {
                "step1": {
                    "type": "sleep-print-maybe-fail",
                    "string": "string_to_pass_down",
                    "seconds": 0,
                    "fail": False,
                },
                "step2": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step1"},
                    "seconds": 0,
                    "fail": False,
                },
                "step3": {
                    "type": "sleep-print-maybe-fail",
                    "string": "This is going to fail!",
                    "seconds": 0,
                    "fail": True,
                },
            }
        )

        executor = MulticoreExecutor(
            workspace=LocalWorkspace(self.TEST_DIR),
            parallelism=parallelism,
            include_package=["test_fixtures.package.steps"],
        )

        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 2

    @pytest.mark.parametrize("parallelism", [1, 2, 3])
    def test_failing_step_with_downstream_task(self, parallelism):
        step_graph = StepGraph.from_params(
            {
                "step1": {
                    "type": "sleep-print-maybe-fail",
                    "string": "string_to_pass_down",
                    "seconds": 0,
                    "fail": True,
                },
                "step2": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step1"},
                    "seconds": 0,
                    "fail": False,
                },
                "step3": {
                    "type": "sleep-print-maybe-fail",
                    "string": "This is going to fail!",
                    "seconds": 0,
                    "fail": False,
                },
            }
        )

        executor = MulticoreExecutor(
            workspace=LocalWorkspace(self.TEST_DIR),
            parallelism=parallelism,
            include_package=["test_fixtures.package.steps"],
        )

        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 1

    @pytest.mark.parametrize("parallelism", [1, 2, 3])
    def test_failing_step_with_further_downstream_task(self, parallelism):
        step_graph = StepGraph.from_params(
            {
                "step1": {
                    "type": "sleep-print-maybe-fail",
                    "string": "string_to_pass_down",
                    "seconds": 0,
                    "fail": True,
                },
                "step2": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step1"},
                    "seconds": 0,
                    "fail": False,
                },
                "step3": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step2"},
                    "seconds": 0,
                    "fail": False,
                },
            }
        )

        executor = MulticoreExecutor(
            workspace=LocalWorkspace(self.TEST_DIR),
            parallelism=parallelism,
            include_package=["test_fixtures.package.steps"],
        )

        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 0

    def test_uncacheable_failing_step_no_downstream_task(self):
        step_graph = StepGraph.from_params(
            {
                "step1": {
                    "type": "sleep-print-maybe-fail",
                    "string": "string_to_pass_down",
                    "seconds": 0,
                    "fail": False,
                },
                "step2": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step1"},
                    "seconds": 0,
                    "fail": False,
                },
                "step3": {
                    "type": "sleep-print-maybe-fail",
                    "string": "This is going to fail!",
                    "seconds": 0,
                    "fail": True,
                    "cache_results": False,
                },
            }
        )

        executor = MulticoreExecutor(
            workspace=LocalWorkspace(self.TEST_DIR),
            parallelism=2,
            include_package=["test_fixtures.package.steps"],
        )

        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 2

    def test_uncacheable_failing_step_with_downstream_task(self):
        step_graph = StepGraph.from_params(
            {
                "step1": {
                    "type": "sleep-print-maybe-fail",
                    "string": "string_to_pass_down",
                    "seconds": 0,
                    "fail": True,
                    "cache_results": False,
                },
                "step2": {
                    "type": "sleep-print-maybe-fail",
                    "string": {"type": "ref", "ref": "step1"},
                    "seconds": 0,
                    "fail": False,
                },
                "step3": {
                    "type": "sleep-print-maybe-fail",
                    "string": "This is going to fail!",
                    "seconds": 0,
                    "fail": False,
                },
            }
        )

        executor = MulticoreExecutor(
            workspace=LocalWorkspace(self.TEST_DIR),
            parallelism=2,
            include_package=["test_fixtures.package.steps"],
        )

        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 1

    @pytest.mark.parametrize("parallelism", [1, 2, 3])
    def test_steps_with_their_own_multiprocessing(self, parallelism):
        step_graph = StepGraph.from_params(
            {
                "step1": {"type": "multiprocessing_step", "num_proc": 2},
                "step2": {"type": "multiprocessing_step", "num_proc": 3},
                "step3": {"type": "multiprocessing_step", "num_proc": 1},
            }
        )

        executor = MulticoreExecutor(
            workspace=LocalWorkspace(self.TEST_DIR),
            parallelism=parallelism,
            include_package=["test_fixtures.package.steps"],
        )

        executor.execute_step_graph(step_graph)
        assert len(executor.workspace.step_cache) == 3
