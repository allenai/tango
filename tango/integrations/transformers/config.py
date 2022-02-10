from transformers import AutoConfig, PretrainedConfig

from tango.common import Registrable


class Config(PretrainedConfig, Registrable):
    """
    A :class:`~tango.common.Registrable` version of transformers'
    :class:`~transformers.PretrainedConfig`.
    """

    default_implementation = "auto"
    """
    The default registered implementation just calls
    :meth:`transformers.AutoConfig.from_pretrained()`.
    """


Config.register("auto", constructor="from_pretrained")(AutoConfig)
