PyTorch
=======

.. automodule:: tango.integrations.torch

Reference
---------

Train step
~~~~~~~~~~

.. autoclass:: tango.integrations.torch.TorchTrainStep
   :members:

.. autoclass:: tango.integrations.torch.TrainConfig
   :members:

Eval step
~~~~~~~~~

.. autoclass:: tango.integrations.torch.TorchEvalStep
   :members:

Torch format
~~~~~~~~~~~~

.. autoclass:: tango.integrations.torch.TorchFormat

Model
~~~~~

.. autoclass:: tango.integrations.torch.Model
   :members:

TrainEngine
~~~~~~~~~~~

.. autoclass:: tango.integrations.torch.TrainEngine
   :members:

.. autoclass:: tango.integrations.torch.TorchTrainEngine

Optim
~~~~~

.. autoclass:: tango.integrations.torch.Optimizer
   :members:

.. autoclass:: tango.integrations.torch.LRScheduler
   :members:

Data
~~~~

.. autoclass:: tango.integrations.torch.DataLoader
   :members:

.. autoclass:: tango.integrations.torch.Sampler
   :members:

.. autoclass:: tango.integrations.torch.DataCollator
   :members:
   :special-members: __call__

.. autoclass:: tango.integrations.torch.ConcatTensorDictsCollator
   :members:

Callbacks
~~~~~~~~~

.. autoclass:: tango.integrations.torch.TrainCallback
   :members:

.. autoclass:: tango.integrations.torch.EvalCallback
   :members:

.. autoclass:: tango.integrations.torch.StopEarlyCallback

.. autoclass:: tango.integrations.torch.StopEarly
   :members:
