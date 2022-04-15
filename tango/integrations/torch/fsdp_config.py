from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch

from tango.common import FromParams


@dataclass
class TorchFSDPConfig(FromParams):
    """
    Base class for FullySharedDataParallel configurations.
    """  # noqa: E501

    def as_kwargs(self) -> Dict[str, Any]:
        """
        Convert to the appropriate ``kwargs``.
        """
        return asdict(self)

    def wrap(self, module: torch.nn.Module):
        """
        A convenience method for wrapping a module in FullySharedDataParallel
        with all of the options defined in this class.

        .. seealso::
            Internally this is what :func:`with_wrapped_modules()` calls.

        """
        raise NotImplementedError()
