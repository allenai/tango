from typing import Tuple, List

import pytest

from tango import Step
from tango.common.exceptions import ConfigurationError
from tango.step_graph import StepGraph


@Step.register("download")
class DownloadStep(Step[str]):
    def run(self) -> str:
        return "data"


@Step.register("preprocess")
class PreprocessStep(Step[str]):
    def run(self, raw_data: str) -> str:
        assert isinstance(raw_data, str)
        return raw_data + raw_data


@Step.register("train_a")
class TrainAStep(Step[float]):
    def run(self, data: str) -> float:
        assert isinstance(data, str)
        return 1 / len(data)


@Step.register("train_b")
class TrainBStep(Step[float]):
    def run(self, data: str) -> float:
        assert isinstance(data, str)
        return len(data)


@Step.register("combine")
class CombineStep(Step[Tuple[float]]):
    def run(self, models: List[float]) -> Tuple[float]:
        for model in models:
            assert isinstance(model, float)
        return tuple(1 / x for x in models)


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
            "data": {"type": "ref", "ref": "preprocessed_data"},
        },
        "model_b": {
            "type": "train_b",
            "data": {"type": "ref", "ref": "preprocessed_data"},
        },
        "combined": {
            "type": "combine",
            "models": [{"type": "ref", "ref": "model_a"}, {"type": "ref", "ref": "model_b"}],
        },
    }


def test_find_direct_dependencies(basic_steps):
    assert StepGraph._find_step_dependencies(basic_steps["model_a"]) == {
        "preprocessed_data"
    }
    assert StepGraph._find_step_dependencies(basic_steps["combined"]) == {
        "model_a",
        "model_b",
    }


def test_parse_step_graph(basic_steps):
    step_graph = StepGraph(basic_steps)
    assert len(step_graph) == 5
    assert step_graph["combined"].dependencies == {step_graph["model_a"], step_graph["model_b"]}


def test_run_step_graph(basic_steps):
    step_graph = StepGraph(basic_steps)
    assert step_graph["combined"].result() == (8, 1/8)


def test_bad_step_graph():
    with pytest.raises(ConfigurationError):
        StepGraph({"a": {"type": "a", "b": {"type": "ref", "ref": "c"}}})


def test_circular_reference():
    raise NotImplementedError()  # TODO