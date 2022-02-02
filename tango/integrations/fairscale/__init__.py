"""
.. important::
    To use this integration you should install ``tango`` with the "fairscale" extra
    (e.g. ``pip install tango[fairscale]``) or just install FairScale after the fact.

    This integration also depends on `PyTorch <https://pytorch.org/>`_, so make sure you
    install the correct version of torch *first* given your operating system and supported
    CUDA version. Check `pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_
    for more details.

Components for Tango integration with `FairScale <https://github.com/facebookresearch/fairscale>`_.
"""

__all__ = [
    "FairScaleTrainingEngine",
    "FSDPConfig",
    "with_wrapped_modules",
]

from .fsdp_config import FSDPConfig
from .module_wrapper import with_wrapped_modules
from .training_engine import FairScaleTrainingEngine
