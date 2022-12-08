import collections
from typing import Any, Dict, Mapping, MutableMapping

import pytest

from tango import StepGraph
from tango.common import Params, Registrable
from tango.common.from_params import FromParams
from tango.common.testing import TangoTestCase
from tango.step import FunctionalStep, Step, step


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

        @Step.register("foo", exist_ok=True)
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

    def test_skip_default_arguments(self):
        class SkipArgStep(Step):
            def run(self) -> int:  # type: ignore
                return 5

        old_hash = SkipArgStep().unique_id

        class SkipArgStep(Step):
            SKIP_DEFAULT_ARGUMENTS = {"arg": 5}

            def run(self, arg: int = 5) -> int:  # type: ignore
                return arg

        assert SkipArgStep().unique_id == old_hash
        assert SkipArgStep(arg=5).unique_id == old_hash
        assert SkipArgStep(arg=6).unique_id != old_hash

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

    def test_steps_in_params(self):
        class Widget(Registrable):
            def __init__(self, x: int):
                self.x = x

        @Widget.register("gizmo")
        class GizmoWidget(Widget):
            def __init__(self, x: int):
                super().__init__(x * x)

        @Step.register("consumer")
        class WidgetConsumerStep(Step):
            def run(self, widget: Widget):  # type: ignore
                return widget.x

        @Step.register("producer")
        class WidgetProducerStep(Step):
            def run(self, x: int) -> Widget:  # type: ignore
                return GizmoWidget(x)

        config = {
            "widget_producer": Params({"type": "producer", "x": 4}),
            "widget_consumer": Params(
                {"type": "consumer", "widget": {"type": "ref", "ref": "widget_producer"}}
            ),
        }

        sg = StepGraph.from_params(config)
        assert len(sg["widget_consumer"].dependencies) > 0

        class WidgetHolder(Registrable):
            def __init__(self, widget: Widget):
                self.widget = widget

        @WidgetHolder.register("gizmo")
        class GizmoWidgetHolder(WidgetHolder):
            def __init__(self, gizmo: GizmoWidget):
                super().__init__(gizmo)

        @Step.register("holder_consumer")
        class WidgetHolderConsumerStep(Step):
            def run(self, widget_holder: WidgetHolder) -> int:  # type: ignore
                return widget_holder.widget.x

        config = {
            "widget_producer": Params({"type": "producer", "x": 4}),
            "holder_consumer": Params(
                {
                    "type": "holder_consumer",
                    "widget_holder": {
                        "type": "gizmo",
                        "gizmo": {"type": "ref", "ref": "widget_producer"},
                    },
                }
            ),
        }
        sg = StepGraph.from_params(config)
        assert len(sg["holder_consumer"].dependencies) > 0

    def test_functional_step(self):
        class Bar(FromParams):
            def __init__(self, x: int):
                self.x = x

        @step(exist_ok=True)
        def foo(bar: Bar) -> int:
            return bar.x

        assert issubclass(foo, FunctionalStep)
        assert foo().run(Bar(x=1)) == 1

        foo_step = Step.from_params({"type": "foo", "bar": {"x": 1}})
        assert isinstance(foo_step, FunctionalStep)
        assert isinstance(foo_step.kwargs["bar"], Bar)
