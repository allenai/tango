from tango.common.testing.steps import FloatStep
from tango.integrations.gs.step_cache import GSStepCache


def test_step_cache(bucket_name: str):
    cache = GSStepCache(bucket_name=bucket_name)
    step = FloatStep(result=1.0)
    cache[step] = 1.0
    assert step in cache
    assert len(cache) == 1
    assert FloatStep(result=2.0) not in cache
    assert cache[step] == 1.0
