from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tango.common import Registrable


class Tokenizer(PreTrainedTokenizerBase, Registrable):
    """
    A :class:`~tango.common.Registrable` version of transformers'
    :class:`~transformers.PreTrainedTokenizerBase`.
    """

    default_implementation = "auto"
    """
    The default registered implementation just calls
    :meth:`transformers.AutoTokenizer.from_pretrained()`.
    """


Tokenizer.register("auto", constructor="from_pretrained")(AutoTokenizer)
