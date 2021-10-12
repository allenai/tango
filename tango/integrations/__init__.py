"""
In :mod:`tango.integrations` we provide many ready-to-use `component <../components/index.html>`_
implementations for leveraging the functionality from popular libraries.

.. tip::
    All registered components will be registered under a name that starts with the name of the integration module,
    possibly followed by a double colon ("::") and another identifier if there are multiple registered
    components of a given type.

    For example, the :class:`~tango.integrations.datasets.LoadDataset` step in the `🤗 Datasets <datasets.html>`_
    integration is registered under the name "datasets::load", and the
    :class:`~tango.integrations.torch.TorchFormat` format in the `PyTorch <torch.html>`_ integration
    is registered under the name "torch".

"""
