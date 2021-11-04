PyTorch
=======

.. automodule:: tango.integrations.torch

Reference
---------

Train step
~~~~~~~~~~

.. autoclass:: tango.integrations.torch.TorchTrainStep
   :members:

Torch format
~~~~~~~~~~~~

.. autoclass:: tango.integrations.torch.TorchFormat

Model
~~~~~

.. autoclass:: tango.integrations.torch.Model
   :members:

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

.. autoclass:: tango.integrations.torch.StopEarlyCallback
   :members:

.. autoclass:: tango.integrations.torch.StopEarly
   :members:
