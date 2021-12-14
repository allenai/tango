from pathlib import Path
from typing import Generic, TypeVar

import dill
import torch

from tango.common.aliases import PathOrStr
from tango.format import Format

T = TypeVar("T")


@Format.register("torch")
class TorchFormat(Format[T], Generic[T]):
    """
    This format writes the artifact using ``torch.save()``.

    Unlike :class:`tango.format.DillFormat`, this has no special support for iterators.

    .. tip::

        Registered as a :class:`~tango.format.Format` under the name "torch".

    """

    VERSION = 2

    def write(self, artifact: T, dir: PathOrStr):
        filename = Path(dir) / "data.pt"
        with open(filename, "wb") as f:
            torch.save((self.VERSION, artifact), f, pickle_module=dill)

    def read(self, dir: PathOrStr) -> T:
        filename = Path(dir) / "data.pt"
        with open(filename, "rb") as f:
            version, artifact = torch.load(f, pickle_module=dill, map_location=torch.device("cpu"))
            if version > self.VERSION:
                raise ValueError(
                    f"File {filename} is too recent for this version of {self.__class__}."
                )
            return artifact
