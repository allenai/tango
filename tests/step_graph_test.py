import re
from copy import deepcopy
from tempfile import NamedTemporaryFile

import pytest

from tango.common.exceptions import ConfigurationError
from tango.common.testing import TangoTestCase
from tango.common.testing.steps import (  # noqa: F401
    AddNumbersStep,
    ConcatStringsStep,
    StringStep,
)
from tango.step_graph import StepGraph


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

    def test_to_file(self):
        step_graph = StepGraph.from_file(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet")

        with NamedTemporaryFile(
            prefix="test-step-graph-to-file-", suffix=".jsonnet", dir=self.TEST_DIR
        ) as file_ref:
            step_graph.to_file(file_ref.name)

            new_step_graph = StepGraph.from_file(file_ref.name)
            assert step_graph == new_step_graph

    def test_to_file_without_config(self):
        from tango.format import JsonFormat

        step_a = AddNumbersStep(a_number=3, b_number=2, step_name="stepA", cache_results=False)
        step_b = AddNumbersStep(
            a_number=step_a, b_number=2, step_name="stepB", step_format=JsonFormat("gz")
        )
        step_graph = StepGraph({"stepA": step_a, "stepB": step_b})

        with NamedTemporaryFile(
            prefix="test-step-graph-to-file-without-config", suffix=".jsonnet", dir=self.TEST_DIR
        ) as file_ref:
            step_graph.to_file(file_ref.name)
            new_step_graph = StepGraph.from_file(file_ref.name)
            assert step_graph == new_step_graph

    def test_with_step_indexer(self):
        config = {
            "list": {"type": "range_step", "start": 0, "end": 3},
            "added": {
                "type": "add_numbers",
                "a_number": 2,
                "b_number": {"type": "ref", "ref": "list", "key": 1},
            },
        }
        step_graph = StepGraph.from_params(deepcopy(config))
        assert [s.name for s in step_graph["added"].dependencies] == ["list"]
        assert step_graph.to_config() == config

    def test_with_forced_dependencies(self):
        config = {
            "some_string": {
                "type": "string",
                "result": "I should run second",
                "step_extra_dependencies": [{"type": "ref", "ref": "other_string"}],
            },
            "other_string": {"type": "string", "result": "I should run first"},
            "added": {
                "type": "concat_strings",
                "string1": "Some string:",
                "string2": {"type": "ref", "ref": "some_string"},
            },
        }
        step_graph = StepGraph.from_params(deepcopy(config))  # type: ignore[arg-type]
        assert step_graph["some_string"].dependencies == {step_graph["other_string"]}
        assert step_graph["added"].recursive_dependencies == {
            step_graph["other_string"],
            step_graph["some_string"],
        }
