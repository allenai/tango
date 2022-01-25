# -*- coding: UTF-8 -*-
"""
.. important::
    To use this integration you should install ``tango`` with the "deepspeed" extra
    (e.g. ``pip install tango[deepspeed]``) or just install DeepSpeed after the fact.

    This integration also depends on `PyTorch <https://pytorch.org/>`_, so make sure you
    install the correct version of torch *first* given your operating system and supported
    CUDA version. Check `pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_
    for more details.

Components for Tango integration with `DeepSpeed <https://www.deepspeed.ai/>`_.
"""

__all__ = [
    "DeepSpeedAccelerator",
]

from .accelerator import DeepSpeedAccelerator
