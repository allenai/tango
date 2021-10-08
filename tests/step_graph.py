import pytest

from tango.step_graph import StepGraph
from tango.common.exceptions import ConfigurationError


@pytest.fixture
def basic_steps():
    return {
        "raw_data": {
            "type": "download",
        },
        "preprocessed_data": {
            "type": "preprocess",
            "raw_data": {"type": "ref", "ref": "raw_data"},
        },
        "model_a": {
            "type": "train_a",
            "data": {"type": "preprocessed_data", "ref": "preprocessed_data"},
        },
        "model_b": {
            "type": "train_b",
            "data": {"type": "preprocessed_data", "ref": "preprocessed_data"},
        },
        "combined": {
            "type": "combine",
            "models": [{"type": "ref", "ref": "model_a"}, {"type": "ref", "ref": "model_b"}],
        },
    }


def test_parse_direct_dependencies(basic_steps):
    assert StepGraph._parse_direct_step_dependencies(basic_steps["model_a"]) == {
        "preprocessed_data"
    }
    assert StepGraph._parse_direct_step_dependencies(basic_steps["combined"]) == {
        "model_a",
        "model_b",
    }


def test_step_graph(basic_steps):
    step_graph = StepGraph(basic_steps)
    assert len(step_graph) == 5
    assert step_graph[0].name == "raw_data"
    assert step_graph[-1].name == "combined"
    assert step_graph["combined"].dependencies == {"model_a", "model_b"}


def test_bad_step_graph():
    with pytest.raises(ConfigurationError):
        StepGraph({"a": {"type": "a", "b": {"type": "ref", "ref": "c"}}})
