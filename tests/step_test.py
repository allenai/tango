import pytest

from tango.step import Step
from tango.step_cache import DirectoryStepCache
from tango.common.testing import TangoTestCase


class TestStep(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        self.step_cache = DirectoryStepCache(self.TEST_DIR / "step_cache")

    def test_from_params(self):
        @Step.register("float")
        class FloatStep(Step):
            def run(self, result: float) -> float:  # type: ignore
                return result

        step = Step.from_params({"type": "float", "result": 3})
        result = step.result(self.step_cache)
        assert result == 3

    def test_from_params_wrong_type(self):
        @Step.register("float")
        class FloatStep(Step):
            def run(self, result: float) -> float:  # type: ignore
                return result

        with pytest.raises(TypeError):
            Step.from_params({"type": "float", "result": "not a float"})
