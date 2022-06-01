from tango.integrations.beaker.step_cache import BeakerStepCache
from test_fixtures.package.steps import FloatStep


def test_step_cache(beaker_workspace: str):
    cache = BeakerStepCache(workspace=beaker_workspace)

    step = FloatStep(result=1.0)
    cache[step] = 1.0
    assert step in cache
    assert len(cache) == 1
    assert FloatStep(result=2.0) not in cache
    assert cache[step] == 1.0
