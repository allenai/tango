import os
import pickle
import sys

import pytest

from tango import Step
from tango.integrations.wandb import WandbStepCache

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "allennlp")
WANDB_PROJECT = "tango-workspace-testing"


class SomeFakeStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self) -> int:  # type: ignore
        return 1


def test_step_cache_artifact_not_found():
    step = SomeFakeStep(step_name="hi there")
    step_cache = WandbStepCache(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    assert step not in step_cache


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
def test_pickling(protocol: int):
    step_cache = WandbStepCache(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    pickle.loads(pickle.dumps(step_cache, protocol=protocol))
