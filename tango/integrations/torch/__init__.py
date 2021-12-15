# -*- coding: UTF-8 -*-
"""
.. important::
    To use this integration you should install ``tango`` with the "torch" extra
    (e.g. ``pip install tango[torch]``) or just install PyTorch after the fact.

    Make sure you install the correct version of torch given your operating system
    and supported CUDA version. Check
    `pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_
    for more details.

Components for Tango integration with `PyTorch <https://pytorch.org/>`_.

These include a basic training loop :class:`~tango.step.Step` and registrable versions
of many ``torch`` classes, such :class:`torch.optim.Optimizer` and :class:`torch.utils.data.DataLoader`.

Example: training a model
-------------------------

Let's look a simple example of training a model.

We'll make a very basic regression model and generate some fake data to train on.
First, the setup:

.. testcode::

    import torch
    import torch.nn as nn

    from tango.common.dataset_dict import DatasetDict
    from tango.step import Step
    from tango.integrations.torch import Model

Now let's build and register our model:

.. testcode::

    @Model.register("basic_regression")
    class BasicRegression(Model):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            self.sigmoid = nn.Sigmoid()
            self.mse = nn.MSELoss()

        def forward(self, x, y=None):
            pred = self.sigmoid(self.linear(x))
            out = {"pred": pred}
            if y is not None:
                out["loss"] = self.mse(pred, y)
            return out

        def _to_params(self):
            return {}

Lastly, we'll need a step to generate data:

.. testcode::

    @Step.register("generate_data")
    class GenerateData(Step):
        DETERMINISTIC = True
        CACHEABLE = False

        def run(self) -> DatasetDict:
            torch.manual_seed(1)
            return DatasetDict(
                {
                    "train": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(64)],
                    "validation": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(32)],
                }
            )

You could then run this experiment with a config that looks like this:

.. literalinclude:: ../../../../test_fixtures/integrations/torch/train.jsonnet

.. testcode::
    :hide:

    from tango.common.testing import run_experiment
    from tango.common.registrable import Registrable

    # Pickling the model fails because the class is defined ad hoc, not in a module.
    # So we put in this hack to pickle a 0 instead of the Model.
    def _return_zero(self):
        return (int, (0,))
    BasicRegression.__reduce__ = _return_zero

    with run_experiment(
        "test_fixtures/integrations/torch/train.jsonnet"
    ) as run_dir:
        assert (run_dir / "train").is_dir(), "Output for the 'train' step was not produced."
    # Restore state of registry.
    del Registrable._registry[Step]["generate_data"]
    del Registrable._registry[Model]["basic_regression"]

For example,

.. code-block::

    tango run train.jsonnet -i my_package -d /tmp/train

would produce the following output:

.. testoutput::
    :options: +ELLIPSIS

    Starting new run ...
    ● Starting step "data" (needed by "train") ...
    ✓ Finished step "data"
    ● Starting step "train" ...
    Loading best weights from state_worker0_best.pt
    ✓ Finished step "train"
    ✓ The output for "train" is in ...


Tips
----

Debugging
~~~~~~~~~

When debugging a training loop that's causing errors on a GPU, you should set the environment variable
``CUDA_LAUNCH_BLOCKING=1``. This will ensure that the stack traces shows where the error actually happened.

You could also use a custom :class:`TrainCallback` to log each batch before they are passed into the model
so that you can see the exact inputs that are causing the issue.

Stopping early
~~~~~~~~~~~~~~

You can stop the "torch::train" step early using a custom :class:`TrainCallback`. Your callback just
needs to raise the :class:`StopEarly` exception.

"""

__all__ = [
    "TorchFormat",
    "TorchTrainStep",
    "Optimizer",
    "LRScheduler",
    "Model",
    "DataLoader",
    "DataCollator",
    "Sampler",
    "ConcatTensorDictsCollator",
    "TrainCallback",
    "TrainConfig",
    "StopEarlyCallback",
    "StopEarly",
]

from .data import ConcatTensorDictsCollator, DataCollator, DataLoader, Sampler
from .exceptions import StopEarly
from .format import TorchFormat
from .model import Model
from .optim import LRScheduler, Optimizer
from .train import TorchTrainStep
from .train_callback import StopEarlyCallback, TrainCallback
from .train_config import TrainConfig
