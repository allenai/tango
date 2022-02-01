from transformers.data import data_collator as transformers_data_collator

from tango.integrations.torch.data import DataCollator

for name, cls in transformers_data_collator.__dict__.items():
    if isinstance(cls, type) and "DataCollator" in name and hasattr(cls, "__call__"):
        DataCollator.register("transformers::" + name)(cls)
