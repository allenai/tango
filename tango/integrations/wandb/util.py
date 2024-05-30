import os
import re
import warnings
from enum import Enum

from wandb.errors import Error as WandbError

_API_KEY_WARNING_ISSUED = False
_SILENCE_WARNING_ISSUED = False


def is_missing_artifact_error(err: WandbError):
    """
    Check if a specific W&B error is caused by a 404 on the artifact we're looking for.
    """
    # This is brittle, but at least we have a test for it.

    # This is a workaround for a bug in the wandb API
    if err.message == "'NoneType' object has no attribute 'get'":
        return True

    if re.search(r"^artifact '.*' not found in '.*'$", err.message):
        return True

    return ("does not contain artifact" in err.message) or (
        "Unable to fetch artifact with name" in err.message
    )


def check_environment():
    global _API_KEY_WARNING_ISSUED, _SILENCE_WARNING_ISSUED
    if "WANDB_API_KEY" not in os.environ and not _API_KEY_WARNING_ISSUED:
        warnings.warn(
            "Missing environment variable 'WANDB_API_KEY' required to authenticate to Weights & Biases.",
            UserWarning,
        )
        _API_KEY_WARNING_ISSUED = True
    if "WANDB_SILENT" not in os.environ and not _SILENCE_WARNING_ISSUED:
        warnings.warn(
            "The Weights & Biases client may produce a lot of log messages. "
            "You can silence these by setting the environment variable 'WANDB_SILENT=true'",
            UserWarning,
        )
        _SILENCE_WARNING_ISSUED = True


class RunKind(Enum):
    STEP = "step"
    TANGO_RUN = "tango_run"


class ArtifactKind(Enum):
    STEP_RESULT = "step_result"
