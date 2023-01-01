import os

from tango.common.testing import TangoTestCase
from tango.common.testing.steps import FloatStep
from tango.integrations.gs.common import empty_bucket
from tango.integrations.gs.step_cache import GSStepCache

GS_BUCKET_NAME = os.environ.get("GS_BUCKET_NAME", "allennlp-tango-bucket")


class TestGSStepCache(TangoTestCase):
    def setup_method(self):
        super().setup_method()

    def teardown_method(self):
        super().teardown_method()
        empty_bucket(GS_BUCKET_NAME)

    def test_step_cache(self):
        cache = GSStepCache(bucket_name=GS_BUCKET_NAME)
        step = FloatStep(result=1.0)
        cache[step] = 1.0
        assert step in cache
        assert len(cache) == 1
        assert FloatStep(result=2.0) not in cache
        assert cache[step] == 1.0
