"""
.. important::
    To use this integration you should install ``tango`` with the "gcs" extra
    (e.g. ``pip install tango[gcs]``) or just install the `gcsfs <https://gcsfs.readthedocs.io/>`_
    library after the fact (e.g. ``pip install gcsfs``).

Components for Tango integration with `GCS <https://cloud.google.com/storage/>`_.
"""

from tango.common.exceptions import IntegrationMissingError

try:
    import gcsfs
except (ModuleNotFoundError, ImportError):
    raise IntegrationMissingError("gcs", dependencies={"gcsfs"})

from .step_cache import GCSStepCache
from .workspace import GCSWorkspace

__all__ = [
    "GCSStepCache",
    "GCSWorkspace",
]
