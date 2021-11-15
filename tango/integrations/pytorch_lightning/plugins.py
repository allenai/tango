from typing import Union

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments.cluster_environment import (
    ClusterEnvironment,
)
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.plugins.training_type.training_type_plugin import (
    TrainingTypePlugin,
)

from tango.common.registrable import Registrable


class LightningTrainingTypePlugin(TrainingTypePlugin, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning
    :class:`~pytorch_lightning.plugins.training_type.training_type_plugin.TrainingTypePlugin`
    class.
    """


class LightningPrecisionPlugin(PrecisionPlugin, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning
    :class:`~pytorch_lightning.plugins.precision.precision_plugin.PrecisionPlugin`
    class.
    """


class LightningClusterEnvironmentPlugin(ClusterEnvironment, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning
    :class:`~pytorch_lightning.plugins.environments.cluster_environment.ClusterEnvironment`
    class.
    """


class LightningCheckpointPlugin(CheckpointIO, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning
    :class:`~pytorch_lightning.plugins.o.checkpoint_plugin.CheckpointIO`
    class.
    """


# Register all plugins.
for name, cls in pl.plugins.training_type.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, TrainingTypePlugin)
        and not cls == TrainingTypePlugin
    ):
        LightningTrainingTypePlugin.register("pytorch_lightning::" + name)(cls)

for name, cls in pl.plugins.precision.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, PrecisionPlugin) and not cls == PrecisionPlugin:
        LightningPrecisionPlugin.register("pytorch_lightning::" + name)(cls)

for name, cls in pl.plugins.environments.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, ClusterEnvironment)
        and not cls == ClusterEnvironment
    ):
        LightningClusterEnvironmentPlugin.register("pytorch_lightning::" + name)(cls)

for name, cls in pl.plugins.io.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, CheckpointIO) and not cls == CheckpointIO:
        LightningCheckpointPlugin.register("pytorch_lightning::" + name)(cls)


ALL_PLUGIN_TYPES = Union[
    LightningTrainingTypePlugin,
    LightningPrecisionPlugin,
    LightningClusterEnvironmentPlugin,
    LightningCheckpointPlugin,
]
