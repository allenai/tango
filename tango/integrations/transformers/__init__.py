"""
.. important::
    To use this integration you should install ``tango`` with the "transformers" extra
    (e.g. ``pip install tango[transformers]``) or just install the ``transformers`` library after the fact
    (e.g. ``pip install transformers``).

Components for Tango integration with `🤗 Transformers <https://huggingface.co/docs/transformers/>`_.
"""

from .run_generation import RunGeneration, RunGenerationDataset
