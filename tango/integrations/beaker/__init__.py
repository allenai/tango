"""
.. important::
    To use this integration you should install ``tango`` with the "beaker" extra
    (e.g. ``pip install tango[beaker]``) or just install the `beaker-py <https://beaker-py.readthedocs.io>`_
    library after the fact (e.g. ``pip install beaker-py``).

Components for Tango integration with `Beaker <https://beaker.org/>`_.
"""

from tango.common.exceptions import IntegrationMissingError

try:
    import beaker
except ModuleNotFoundError:
    raise IntegrationMissingError("beaker", dependencies={"beaker-py"})

from .executor import BeakerExecutor
from .step_cache import BeakerStepCache
from .workspace import BeakerWorkspace

__all__ = ["BeakerStepCache", "BeakerWorkspace", "BeakerExecutor"]
