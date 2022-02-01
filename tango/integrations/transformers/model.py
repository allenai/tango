from transformers import AutoModel

from tango.integrations.torch.model import Model

from .config import Config


@Model.register("transformers::from_pretrained", constructor="from_pretrained")
@Model.register("transformers::from_config", constructor="from_config")
class TransformerModel(AutoModel, Model):
    """
    A :class:`~tango.common.Registrable` version of transformers'
    :class:`~transformers.AutoModel`.

    .. tip::

        Registered as a :class:`~tango.integrations.torch.Model` under the names
        "transformers::from_pretrained" and "transformers::from_config", which use the
        :meth:`~transformers.AutoModel.from_pretrained()` and :meth:`~transformers.AutoModel.from_config()`
        methods respectively.

    Examples
    --------

    Instantiate a pretrained transformer model from params:

    .. testcode::

        from tango.integrations.torch import Model

        model = Model.from_params({
            "type": "transformers::from_pretrained",
            "pretrained_model_name_or_path": "epwalsh/bert-xsmall-dummy",
        })

    Instantiate a transformer model from params without loading pretrained weights:

    .. testcode::

        from tango.integrations.torch import Model

        model = Model.from_params({
            "type": "transformers::from_config",
            "config": {"pretrained_model_name_or_path": "epwalsh/bert-xsmall-dummy"},
        })

    """

    @classmethod
    def from_config(cls, config: Config, **kwargs) -> "TransformerModel":
        return super().from_config(config, **kwargs)
