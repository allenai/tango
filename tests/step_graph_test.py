import re

import pytest

import test_fixtures.package.steps  # noqa: F401
from tango.common.exceptions import ConfigurationError
from tango.common.testing import TangoTestCase
from tango.step_graph import StepGraph


class TestStepGraph(TangoTestCase):
    def test_ordered_steps(self):
        step_graph = StepGraph(
            {
                "step1": {
                    "type": "add_numbers",
                    "a_number": 2,
                    "b_number": 3,
                },
                "step2": {
                    "type": "add_numbers",
                    "a_number": {"type": "ref", "ref": "step1"},
                    "b_number": 5,
                },
                "step3": {
                    "type": "add_numbers",
                    "a_number": 3,
                    "b_number": 1,
                },
            }
        )

        result = step_graph.ordered_steps()
        assert [res.name for res in result] == ["step1", "step2", "step3"]

    def test_from_file(self):
        step_graph = StepGraph.from_file(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet")
        assert "hello" in step_graph
        assert "hello_world" in step_graph

    def test_missing_type(self):
        with pytest.raises(ConfigurationError, match=re.escape('key "type" is required')):
            StepGraph(
                {
                    "step3": {
                        "a_number": 3,
                        "b_number": 1,
                    },
                }
            )
