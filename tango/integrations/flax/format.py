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

    VERSION = 2

    def write(self, artifact: T, dir: PathOrStr) -> None:
        step = 100  # How to pass step?
        checkpoints.save_checkpoint(Path(dir), artifact, step)

    def read(self, dir: PathOrStr) -> T:
        return checkpoints.restore_checkpoint(dir, target=None)
