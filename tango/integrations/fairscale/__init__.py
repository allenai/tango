"""
.. important::
    To use this integration you should install ``tango`` with the "fairscale" extra
    (e.g. ``pip install tango[fairscale]``) or just install FairScale after the fact.

    This integration also depends on `PyTorch <https://pytorch.org/>`_, so make sure you
    install the correct version of torch *first* given your operating system and supported
    CUDA version. Check `pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_
    for more details.

Components for Tango integration with `FairScale <https://github.com/facebookresearch/fairscale>`_.

Overview
--------

FairScale is a PyTorch library for large scale training. Among other things, it implements
the main memory-savings techniques for distributed data-parallel training (DDP) that came from the paper
`ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
<https://api.semanticscholar.org/CorpusID:203736482>`_.

The main part of this Tango integration is the :class:`FairScaleTrainingEngine`.
This is a :class:`~tango.integrations.torch.TrainingEngine` implementation that utilizes
FairScale's :class:`~fairscale.nn.FullyShardedDataParallel` (FSDP) for substantial memory savings
during distributed training.

For the best performance you should also use :func:`with_wrapped_modules()` to wrap the inner modules
of your :class:`~tango.integrations.torch.Model`. When used with FSDP this will dramatically reduce
the memory required to load your model.

"""

__all__ = [
    "FairScaleTrainingEngine",
    "FSDPConfig",
    "with_wrapped_modules",
]

from .fsdp_config import FSDPConfig
from .module_wrapper import with_wrapped_modules
from .training_engine import FairScaleTrainingEngine
