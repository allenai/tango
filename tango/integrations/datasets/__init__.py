"""
Components for Tango integration with `🤗 Datasets <https://huggingface.co/docs/datasets/>`_.

.. important::
    To use this integration you should install ``tango`` with the "datasets" extra
    (e.g. ``pip install tango[datasets]``) or just install the ``datasets`` library after the fact
    (e.g. ``pip install datasets``).

"""


from typing import Union

import datasets

from tango.step import Step
from tango.common.dataset_dict import DatasetDict


__all__ = ["LoadDataset", "convert_to_tango_dataset_dict"]


def convert_to_tango_dataset_dict(hf_dataset_dict: datasets.DatasetDict) -> DatasetDict:
    """
    A helper function that can be used to convert a HuggingFace :class:`~datasets.DatasetDict`
    into a native Tango :class:`~tango.common.dataset_dict.DatasetDict`.

    This is important to do when your dataset dict is input to another step for caching
    reasons.
    """
    fingerprint = ""
    for key, dataset in sorted(hf_dataset_dict.items(), key=lambda x: x[0]):
        fingerprint += f"{key}-{dataset._fingerprint}-"
    return DatasetDict(splits=hf_dataset_dict, fingerprint=fingerprint)


@Step.register("datasets::load")
class LoadDataset(Step):
    """
    This step loads a `HuggingFace dataset <https://huggingface.co/datasets>`_.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::load".

    Examples
    --------

    .. testsetup::

        from tango import Step

    .. testcode::

        load_step = Step.from_params({
            "type": "datasets::load",
            "path": "lhoestq/test",
        })

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # These are already cached by huggingface.

    def run(  # type: ignore
        self, path: str, **kwargs
    ) -> Union[
        datasets.DatasetDict,
        datasets.Dataset,
        datasets.IterableDatasetDict,
        datasets.IterableDataset,
    ]:
        """
        Loads a HuggingFace dataset.

        ``path`` is the canonical name or path to the dataset. Additional key word arguments
        are passed as-is to :func:`datasets.load_dataset()`.
        """
        return datasets.load_dataset(path, **kwargs)
