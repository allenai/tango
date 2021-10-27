import pytest

from tango.common.from_params import FromParams
from tango.common.testing import TangoTestCase
from tango.step import Step


class TestStep(TangoTestCase):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        @Step.register("float")
        class FloatStep(Step):
            def run(self, result: float) -> float:  # type: ignore
                return result

    def test_from_params(self):
        step = Step.from_params({"type": "float", "result": 3})
        result = step.run_with_work_dir(self.TEST_DIR / "work")
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
        assert step.run_with_work_dir(self.TEST_DIR / "work").x == 1
