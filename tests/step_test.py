import collections
from typing import Any, Dict, Mapping, MutableMapping

import pytest

import test_fixtures.package.steps  # noqa: F401
from tango.common.from_params import FromParams
from tango.common.testing import TangoTestCase
from tango.step import Step


class TestStep(TangoTestCase):
    def test_from_params(self):
        step = Step.from_params({"type": "float", "result": 3})
        result = step.result()
        assert result == 3

    def test_from_params_wrong_type(self):
        with pytest.raises(TypeError):
            Step.from_params({"type": "float", "result": "not a float"})

    def test_step_with_from_params_input(self):
        class Bar(FromParams):
            def __init__(self, x: int):
                self.x = x

        @Step.register("foo")
        class FooStep(Step):
            def run(self, bar: Bar) -> Bar:  # type: ignore
                return bar

        step = Step.from_params({"type": "foo", "bar": {"x": 1}})
        assert step.result().x == 1

    def test_no_hash_arguments(self):
        @Step.register("no_hash_step")
        class SkipArgStep(Step):
            SKIP_ID_ARGUMENTS = {"arg"}

            def run(self, arg: str) -> int:  # type: ignore
                return 5

        step1 = SkipArgStep(arg="foo")
        step2 = SkipArgStep(arg="bar")
        assert step1.unique_id == step2.unique_id

    def test_massage_kwargs(self):
        class CountLettersStep(Step):
            @classmethod
            def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                kwargs = kwargs.copy()
                kwargs["text"] = kwargs["text"].lower()
                return kwargs

            def run(self, text: str) -> Mapping[str, int]:  # type: ignore
                text = text.lower()
                counter: MutableMapping[str, int] = collections.Counter()
                for c in text:
                    counter[c] += 1
                return counter

        upper = CountLettersStep(text="FOO")
        lower = CountLettersStep(text="foo")
        assert upper.unique_id == lower.unique_id
        assert upper.result() == lower.result()

    def test_default_args(self):
        class DefaultArgStep(Step[int]):
            def run(self, left: int, right: int = 0) -> int:  # type: ignore
                return left + right

        explicit = DefaultArgStep(left=1, right=0)
        implicit = DefaultArgStep(left=1)

        assert explicit.unique_id == implicit.unique_id
        assert explicit.result() == implicit.result()
