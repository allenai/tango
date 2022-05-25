import pickle
import sys

import pytest

from tango.common.testing import TangoTestCase
from tango.step import Step
from tango.step_caches.local_step_cache import LocalStepCache


class DummyStep(Step):
    def run(self, x: int) -> int:  # type: ignore[override]
        return x


class TestLocalStepCache(TangoTestCase):
    @pytest.mark.parametrize(
        "protocol",
        [pytest.param(protocol, id=f"protocol={protocol}") for protocol in range(4)]
        + [
            pytest.param(
                5,
                id="protocol=5",
                marks=pytest.mark.skipif(
                    sys.version_info < (3, 8), reason="Protocol 5 requires Python 3.8 or newer"
                ),
            ),
        ],
    )
    def test_pickling(self, protocol: int):
        step = DummyStep(step_name="dummy", x=1)
        step_cache = LocalStepCache(self.TEST_DIR)
        step_cache[step] = 1
        assert step in step_cache
        assert step.unique_id in step_cache.strong_cache
        pickled_step_cache = pickle.dumps(step_cache, protocol=protocol)
        unpickled_step_cache = pickle.loads(pickled_step_cache)
        assert step.unique_id not in unpickled_step_cache.strong_cache
        assert step in unpickled_step_cache
