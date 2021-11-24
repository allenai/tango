import gzip
import pathlib
from os import PathLike
from typing import Union

import dill

from tango import Format
from tango.common import DatasetDict, PathOrStr, filename_is_safe
from tango.common.sqlite_sparse_sequence import SqliteSparseSequence


@Format.register("sqlite")
class SqliteDictFormat(Format[DatasetDict]):
    VERSION = 3

    def write(self, artifact: DatasetDict, dir: Union[str, PathLike]):
        dir = pathlib.Path(dir)
        with gzip.open(dir / "metadata.dill.gz", "wb") as f:
            dill.dump(artifact.metadata, f)
        for split_name, split in artifact.splits.items():
            filename = f"{split_name}.sqlite"
            if not filename_is_safe(filename):
                raise ValueError(f"{split_name} is not a valid name for a split.")
            (dir / filename).unlink(missing_ok=True)
            if isinstance(split, SqliteSparseSequence):
                split.copy_to(dir / filename)
            else:
                sqlite = SqliteSparseSequence(dir / filename)
                sqlite.extend(split)

    def read(self, dir: Union[str, PathLike]) -> DatasetDict:
        dir = pathlib.Path(dir)
        with gzip.open(dir / "metadata.dill.gz", "rb") as f:
            metadata = dill.load(f)
        splits = {
            filename.stem: SqliteSparseSequence(filename, read_only=True)
            for filename in dir.glob("*.sqlite")
        }
        return DatasetDict(metadata=metadata, splits=splits)

    def checksum(self, dir: PathOrStr) -> str:
        # This is not trivial to implement because sqlite files can be different even if they contain the same
        # data.
        raise NotImplementedError()
