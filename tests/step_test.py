from typing import Dict

from flaky import flaky
import pytest

from tango.step import Step, step_graph_from_params
from tango.common.exceptions import ConfigurationError
from tango.common.params import Params
from tango.common.testing import TangoTestCase


@Step.register("float")
class FloatStep(Step):
    def run(self, result: float) -> float:  # type: ignore
        return result


@Step.register("string")
class StringStep(Step):
    def run(self, result: str) -> str:  # type: ignore
        return result


@Step.register("concat_strings")
class ConcatStringsStep(Step):
    def run(self, string1: str, string2: str, join_with: str = " ") -> str:  # type: ignore
        return join_with.join([string1, string2])


def teardown_module():
    del Step._registry[Step]["float"]
    del Step._registry[Step]["string"]
    del Step._registry[Step]["concat_strings"]


class TestStep(TangoTestCase):
    def test_from_params(self):
        step = Step.from_params({"type": "float", "result": 3})
        result = step.result()
        assert result == 3

    def test_from_params_wrong_type(self):
        with pytest.raises(TypeError):
            Step.from_params({"type": "float", "result": "not a float"})

    def test_nested_steps(self):
        step = Step.from_params(
            {
                "type": "concat_strings",
                "string1": {"type": "string", "result": "Hello,"},
                "string2": {"type": "string", "result": "World!"},
            }
        )
        assert step.result() == "Hello, World!"

    def test_nested_steps_wrong_type(self):
        with pytest.raises(TypeError):
            Step.from_params(
                {
                    "type": "concat_strings",
                    "string1": {"type": "float", "result": 1.0},
                    "string2": {"type": "string", "result": "World!"},
                }
            )

    @pytest.mark.parametrize("ordered_ascending", [True, False])
    def test_make_step_graph(self, ordered_ascending: bool):
        params = {
            "hello": {"type": "string", "result": "Hello"},
            "hello_world": {
                "type": "concat_strings",
                "string1": {"type": "ref", "ref": "hello"},
                "string2": "World!",
                "join_with": ", ",
            },
        }
        params = dict(sorted(params.items(), reverse=ordered_ascending))
        step_graph = step_graph_from_params(Params(params))
        assert len(step_graph) == 2
        assert isinstance(step_graph["hello"], StringStep)
        assert isinstance(step_graph["hello_world"], ConcatStringsStep)
        assert step_graph["hello_world"].kwargs["string1"] == step_graph["hello"]

    def test_make_step_graph_chained_refs(self):
        params = {
            "string0": {"type": "string", "result": "Hello"},
            "string1": {"type": "string", "result": {"ref": "string0"}},
            "string2": {"type": "string", "result": "World!"},
            "hello_world": {
                "type": "concat_strings",
                "string1": {"type": "ref", "ref": "string1"},
                "string2": {"type": "ref", "ref": "string2"},
                "join_with": ", ",
            },
        }
        step_graph = step_graph_from_params(params)  # type: ignore[arg-type]
        assert step_graph["hello_world"].kwargs["string1"] == step_graph["string1"]
        assert step_graph["hello_world"].kwargs["string1"].kwargs["result"] == step_graph["string0"]
        assert step_graph["hello_world"].kwargs["string2"] == step_graph["string2"]

    def test_make_step_graph_missing_step(self):
        params = {
            "hello_world": {
                "type": "concat_strings",
                "string1": {"type": "ref", "ref": "hello"},
                "string2": "World!",
                "join_with": ", ",
            },
        }
        with pytest.raises(ConfigurationError):
            step_graph_from_params(params)

    #  def test_ref_step_within_object(self):
    #      from tango.common.from_params import FromParams
    #      class Baz:
    #          def __init__(self, a: str):
    #              assert isinstance(a, str), a
    #              self.a = a

    #      @Step.register("baz")
    #      class BazStep(Step):
    #          def run(self, a: str) -> Baz:  # type: ignore[override]
    #              assert isinstance(a, str), a
    #              return Baz(a)

    #      class Bar(FromParams):
    #          def __init__(self, baz: Baz):
    #              assert isinstance(baz, Baz), baz
    #              self.baz = baz

    #      @Step.register("foo")
    #      class Foo(Step):
    #          def run(self, bar: Bar) -> str:  # type: ignore[override]
    #              return bar.baz.a

    #      params = {
    #          "baz": {"type": "baz", "a": "hello"},
    #          "foo": {"type": "foo", "bar": {"baz": {"type": "ref", "ref": "baz"}}},
    #      }
    #      step_graph_from_params(params)  # type: ignore[arg-type]

    @pytest.mark.parametrize("deterministic", [True, False])
    @flaky
    def test_random_seeds_are_initialized(self, deterministic: bool):
        class RandomNumberStep(Step[Dict[str, int]]):
            DETERMINISTIC = deterministic
            CACHEABLE = False

            def run(self) -> Dict[str, int]:  # type: ignore
                import random

                out = {"random": random.randint(0, 2 ** 32)}

                try:
                    import numpy

                    out["numpy"] = numpy.random.randint(0, 2 ** 32)
                except ImportError:
                    pass

                try:
                    import torch

                    out["torch"] = torch.randint(2 ** 32, [1])[0].item()
                except ImportError:
                    pass

                return out

        step1_result = RandomNumberStep().result()
        step2_result = RandomNumberStep().result()

        if deterministic:
            assert step1_result == step2_result
        else:
            assert step1_result != step2_result
