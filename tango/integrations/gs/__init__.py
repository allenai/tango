"""
.. important::
    To use this integration you should install ``tango`` with the "gs" extra
    (e.g. ``pip install tango[gs]``) or just install the `gcsfs <https://gcsfs.readthedocs.io/>`_
    library after the fact (e.g. ``pip install gcsfs``).

Components for Tango integration with `GS <https://cloud.google.com/storage/>`_.
"""

from tango.common.exceptions import IntegrationMissingError

try:
    from google.cloud import datastore, storage
except (ModuleNotFoundError, ImportError):
    raise IntegrationMissingError("gs", dependencies={"google-cloud-storage"})

from .step_cache import GSStepCache
from .workspace import GSWorkspace

__all__ = [
    "GSStepCache",
    "GSWorkspace",
]
