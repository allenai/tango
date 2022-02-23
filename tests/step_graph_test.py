import re

import pytest

from tango.common.exceptions import ConfigurationError
from tango.common.testing import TangoTestCase
from tango.step_graph import StepGraph
from test_fixtures.package.steps import (  # noqa: F401
    AddNumbersStep,
    ConcatStringsStep,
    StringStep,
)


class TestStepGraph(TangoTestCase):
    def test_ordered_steps(self):
        step_graph = StepGraph.from_params(
            {
                "stepB": {
                    "type": "add_numbers",
                    "a_number": 2,
                    "b_number": 3,
                },
                "stepC": {
                    "type": "add_numbers",
                    "a_number": {"type": "ref", "ref": "stepB"},
                    "b_number": 5,
                },
                "stepA": {
                    "type": "add_numbers",
                    "a_number": 3,
                    "b_number": 1,
                },
            }
        )

        result = StepGraph.ordered_steps(step_graph.parsed_steps)
        assert [res.name for res in result] == ["stepB", "stepC", "stepA"]

    def test_from_file(self):
        step_graph = StepGraph.from_file(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet")
        assert "hello" in step_graph
        assert "hello_world" in step_graph

    def test_missing_type(self):
        with pytest.raises(ConfigurationError, match=re.escape('key "type" is required')):
            StepGraph.from_params(
                {
                    "step3": {
                        "a_number": 3,
                        "b_number": 1,
                    },
                }
            )

    def test_direct_construction(self):
        step_a = AddNumbersStep(a_number=3, b_number=2, step_name="stepA")
        step_b = AddNumbersStep(a_number=step_a, b_number=2, step_name="stepB")
        step_graph = StepGraph({"stepA": step_a, "stepB": step_b})
        assert list(step_graph.parsed_steps.keys()) == ["stepA", "stepB"]

    def test_direct_construction_missing_dependency(self):
        step_a = AddNumbersStep(a_number=3, b_number=2, step_name="stepA")
        step_b = AddNumbersStep(a_number=step_a, b_number=2, step_name="stepB")
        with pytest.raises(ConfigurationError, match="Or a missing dependency"):
            StepGraph({"stepB": step_b})
