from dataclasses import fields, is_dataclass
from typing import Callable

from transformers.data import data_collator as transformers_data_collator

from tango.integrations.torch.data import DataCollator

from .tokenizer import Tokenizer


# Some data collators take a tokenizer, so in order to instantiate those collators from params,
# we need to use a factory function that takes our registrable version of a tokenizer as
# an argument.
def data_collator_with_tokenizer_factory(cls) -> Callable[..., DataCollator]:
    def factory(tokenizer: Tokenizer, **kwargs) -> DataCollator:
        return cls(tokenizer=tokenizer, **kwargs)

    return factory


for name, cls in transformers_data_collator.__dict__.items():
    if (
        isinstance(cls, type)
        and is_dataclass(cls)
        and "DataCollator" in name
        and hasattr(cls, "__call__")
    ):
        for field in fields(cls):
            if field.name == "tokenizer":
                factory_func = data_collator_with_tokenizer_factory(cls)
                DataCollator.register("transformers::" + name)(factory_func)  # type: ignore
                break
        else:
            DataCollator.register("transformers::" + name)(cls)
