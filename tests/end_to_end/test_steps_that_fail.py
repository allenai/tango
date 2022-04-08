from collections import Counter
from typing import MutableMapping

import pytest

from tango import Step
from tango.common.exceptions import CliRunError
from tango.common.testing import TangoTestCase

step_execution_count: MutableMapping[str, int] = Counter()


@Step.register("step_a")
class StepA(Step):
    def run(self, what_number: int) -> int:  # type: ignore
        global step_execution_count
        step_execution_count["a"] += 1
        return what_number


@Step.register("step_b")
class StepB(Step):
    def run(self, what_number: int) -> int:  # type: ignore
        global step_execution_count
        step_execution_count["b"] += 1
        return what_number


step_should_fail: bool = True


@Step.register("step_fail")
class StepFail(Step):
    def run(self, what_number: int) -> int:  # type: ignore
        global step_execution_count
        step_execution_count["fail"] += 1
        global step_should_fail
        if step_should_fail:
            raise RuntimeError("Step should fail")
        else:
            return what_number


class TestExperiment(TangoTestCase):
    def test_experiment(self, caplog):
        global step_should_fail
        config = {
            "steps": {
                "a_number": {
                    "type": "step_a",
                    "what_number": 3,
                },
                "fail_number": {
                    "type": "step_fail",
                    "what_number": {"type": "ref", "ref": "a_number"},
                },
                "b_number": {
                    "type": "step_b",
                    "what_number": {"type": "ref", "ref": "fail_number"},
                },
            }
        }

        global step_should_fail
        global step_execution_count

        step_should_fail = True
        with pytest.raises(CliRunError):
            self.run(config)

        assert step_execution_count["a"] == 1
        assert step_execution_count["fail"] == 1
        assert step_execution_count["b"] == 0

        step_should_fail = False
        self.run(config)

        assert step_execution_count["a"] == 1
        assert step_execution_count["fail"] == 2
        assert step_execution_count["b"] == 1
