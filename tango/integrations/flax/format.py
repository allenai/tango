from pathlib import Path
from typing import Generic, TypeVar

from flax.training import checkpoints

from tango.common.aliases import PathOrStr
from tango.format import Format

T = TypeVar("T")


@Format.register("flax")
class FlaxFormat(Format[T], Generic[T]):
    """
    This format writes the artifact.
    """

    VERSION = "002"

    def write(self, artifact: T, dir: PathOrStr) -> None:
        checkpoints.save_checkpoint(Path(dir), artifact, step=1)

    def read(self, dir: PathOrStr) -> T:
        # will return a dict
        return checkpoints.restore_checkpoint(dir, target=None)
