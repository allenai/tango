"""
Components for Tango integration with `PyTorch <https://pytorch.org/>`_.

.. important::
    To use this integration you should install ``tango`` with the "torch" extra
    (e.g. ``pip install tango[torch]``) or just install PyTorch after the fact.

    Make sure you install the correct version of torch given your operating system
    and supported CUDA version. Check
    `pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_
    for more details.

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

.. testcode::

    config = {
        "steps": {
            "data": {
                "type": "generate_data",
            },
            "train": {
                "type": "torch::train",
                "model": {
                    "type": "basic_regression",
                },
                "dataset_dict": {
                    "type": "ref",
                    "ref": "data",
                },
                "train_dataloader": {
                    "batch_size": 8,
                    "shuffle": True,
                },
                "optimizer": {
                    "type": "Adam",
                },
                "validation_split": "validation",
                "validation_dataloader": {
                    "batch_size": 8,
                    "shuffle": False,
                },
                "train_steps": 100,
                "validate_every": 10,
                "checkpoint_every": 10,
                "log_every": 1,
            }
        }
    }

.. testcode::
    :hide:

    import os
    from tango.common.testing import run_experiment
    from tango.common.registrable import Registrable

    # Don't cache results, otherwise we'll have a pickling error.
    config["steps"]["train"]["cache_results"] = False
    with run_experiment(config) as run_dir:
        assert (run_dir / "step_cache").is_dir()
    # Restore state of registry.
    del Registrable._registry[Step]["generate_data"]
    del Registrable._registry[Model]["basic_regression"]

For example,

.. code-block::

    tango run config.jsonnet -i my_package -d /tmp/train

would produce the following output:

.. testoutput::

    ● Starting run for "data"
    ✓ Finished run for "data"
    ● Starting run for "train"
    Loading best weights from state_worker0_step100.pt
    ✓ Finished run for "train"

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
]

from .data import DataLoader, Sampler, DataCollator, ConcatTensorDictsCollator
from .format import TorchFormat
from .model import Model
from .optim import Optimizer, LRScheduler
from .train import TorchTrainStep
from .train_callback import TrainCallback
