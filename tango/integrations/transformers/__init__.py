"""
.. important::
    To use this integration you should install ``tango`` with the "transformers" extra
    (e.g. ``pip install tango[transformers]``) or just install the ``transformers`` library after the fact
    (e.g. ``pip install transformers``).

Components for Tango integration with `🤗 Transformers <https://huggingface.co/docs/transformers/>`_.

This integration provides some useful steps and also registers PyTorch components from the transformers
library under the corresponding class from the `torch <torch.html>`_ integration, such as:

- :class:`~tango.integrations.torch.Model`: All transformers "auto" model classes are registered
  according to their class names (e.g. "transformers::AutoModelForCausalLM::from_pretrained"
  or "transformers::AutoModelForCausalLM::from_config").

  For example, to instantiate a pretrained transformer model from params:

  .. testcode::

      from tango.integrations.torch import Model

      model = Model.from_params({
          "type": "transformers::AutoModel::from_pretrained",
          "pretrained_model_name_or_path": "epwalsh/bert-xsmall-dummy",
      })

  Or to instantiate a transformer model from params without loading pretrained weights:

  .. testcode::

      from tango.integrations.torch import Model

      model = Model.from_params({
          "type": "transformers::AutoModel::from_config",
          "config": {"pretrained_model_name_or_path": "epwalsh/bert-xsmall-dummy"},
      })

  .. tip::

        You can see a list of all of the available auto model constructors from transformers by running:

        .. testcode::

            from tango.integrations.torch import Model
            from tango.integrations.transformers import *

            for name in sorted(Model.list_available()):
                if name.startswith("transformers::AutoModel"):
                    print(name)

        .. testoutput::
            :options: +ELLIPSIS

            transformers::AutoModel::from_config
            transformers::AutoModel::from_pretrained
            transformers::AutoModelForAudioClassification::from_config
            transformers::AutoModelForAudioClassification::from_pretrained
            ...

- :class:`~tango.integrations.torch.Optimizer`: All optimizers from transformers are registered according
  to their class names (e.g. "transformers::AdaFactor").

  .. tip::

        You can see a list of all of the available optimizers from transformers by running:

        .. testcode::

            from tango.integrations.torch import Optimizer
            from tango.integrations.transformers import *

            for name in sorted(Optimizer.list_available()):
                if name.startswith("transformers::"):
                    print(name)

        .. testoutput::
            :options: +ELLIPSIS

            transformers::Adafactor
            transformers::AdamW

- :class:`~tango.integrations.torch.LRScheduler`: All learning rate scheduler function from transformers
  are registered according to their type name (e.g. "transformers::linear").

  .. tip::

        You can see a list of all of the available scheduler functions from transformers by running:

        .. testcode::

            from tango.integrations.torch import LRScheduler
            from tango.integrations.transformers import *

            for name in sorted(LRScheduler.list_available()):
                if name.startswith("transformers::"):
                    print(name)

        .. testoutput::
            :options: +ELLIPSIS

            transformers::constant
            transformers::constant_with_warmup
            transformers::cosine
            ...

- :class:`~tango.integrations.torch.DataCollator`: All data collators from transformers
  are registered according to their class name (e.g. "transformers::DefaultDataCollator").

  You can instantiate any of these from a config / params like so:

  .. testcode::

      from tango.integrations.torch import DataCollator

      collator = DataCollator.from_params({
          "type": "transformers::DataCollatorWithPadding",
          "tokenizer": {
              "pretrained_model_name_or_path": "epwalsh/bert-xsmall-dummy",
          },
      })

  .. tip::

        You can see a list of all of the available data collators from transformers by running:

        .. testcode::

            from tango.integrations.torch import DataCollator
            from tango.integrations.transformers import *

            for name in sorted(DataCollator.list_available()):
                if name.startswith("transformers::"):
                    print(name)

        .. testoutput::
            :options: +ELLIPSIS

            transformers::DataCollatorForLanguageModeling
            transformers::DataCollatorForPermutationLanguageModeling
            transformers::DataCollatorForSOP
            ...

"""

__all__ = ["RunGeneration", "RunGenerationDataset", "Tokenizer", "Config"]

from .config import Config
from .data import *  # noqa: F403
from .model import *  # noqa: F403
from .optim import *  # noqa: F403
from .run_generation import RunGeneration, RunGenerationDataset
from .tokenizer import Tokenizer
