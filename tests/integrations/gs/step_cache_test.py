import os

import pytest

from tango.common.testing import TangoTestCase
from tango.common.testing.steps import FloatStep
from tango.integrations.gs.common import empty_bucket_folder
from tango.integrations.gs.step_cache import GSStepCache

GS_BUCKET_NAME = os.environ.get("GS_BUCKET_NAME", "allennlp-tango-bucket")
GS_SUBFOLDER = f"{GS_BUCKET_NAME}/my-workspaces/workspace1"


class TestGSStepCache(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        empty_bucket_folder(GS_BUCKET_NAME)
        empty_bucket_folder(GS_SUBFOLDER)

    def teardown_method(self):
        super().teardown_method()

    @pytest.mark.parametrize("gs_path", [GS_BUCKET_NAME, GS_SUBFOLDER])
    def test_step_cache(self, gs_path):
        cache = GSStepCache(folder_name=gs_path)
        step = FloatStep(result=1.0)
        cache[step] = 1.0
        assert step in cache
        assert len(cache) == 1
        assert FloatStep(result=2.0) not in cache
        assert cache[step] == 1.0
