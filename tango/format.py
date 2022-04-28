import bz2
import dataclasses
import gzip
import importlib
import json
import logging
import lzma
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import dill

from tango.common import DatasetDict, filename_is_safe
from tango.common.aliases import PathOrStr
from tango.common.exceptions import ConfigurationError
from tango.common.registrable import Registrable
from tango.common.sequences import SqliteSparseSequence

T = TypeVar("T")


class Format(Registrable, Generic[T]):
    """
    Formats write objects to directories and read them back out.

    In the context of Tango, the objects that are written by formats are usually
    the result of a :class:`~tango.step.Step`.
    """

    VERSION: int = NotImplemented
    """
    Formats can have versions. Versions are part of a step's unique signature, part of
    :attr:`~tango.step.Step.unique_id`, so when a step's format changes,
    that will cause the step to be recomputed.
    """

    default_implementation = "dill"

    @abstractmethod
    def write(self, artifact: T, dir: PathOrStr):
        """Writes the ``artifact`` to the directory at ``dir``."""
        raise NotImplementedError()

    @abstractmethod
    def read(self, dir: PathOrStr) -> T:
        """Reads an artifact from the directory at ``dir`` and returns it."""
        raise NotImplementedError()

    def _to_params(self) -> Dict[str, Any]:
        params_dict = super()._to_params()
        for key in ["logger", "__orig_class__"]:
            params_dict.pop(key, None)  # Removing unnecessary keys.
        params_dict["type"] = self.__module__ + "." + self.__class__.__qualname__
        return params_dict


_OPEN_FUNCTIONS: Dict[Optional[str], Callable[[PathLike, str], IO]] = {
    None: open,
    "None": open,
    "none": open,
    "null": open,
    "gz": gzip.open,  # type: ignore
    "gzip": gzip.open,  # type: ignore
    "bz": bz2.open,  # type: ignore
    "bz2": bz2.open,  # type: ignore
    "bzip": bz2.open,  # type: ignore
    "bzip2": bz2.open,  # type: ignore
    "lzma": lzma.open,
}

_SUFFIXES: Dict[Callable, str] = {
    open: "",
    gzip.open: ".gz",
    bz2.open: ".bz2",
    lzma.open: ".xz",
}


def _open_compressed(filename: PathOrStr, mode: str) -> IO:
    open_fn: Callable
    filename = str(filename)
    for open_fn, suffix in _SUFFIXES.items():
        if len(suffix) > 0 and filename.endswith(suffix):
            break
    else:
        open_fn = open
    return open_fn(filename, mode)


@Format.register("dill")
class DillFormat(Format[T], Generic[T]):
    """
    This format writes the artifact as a single file called "data.dill" using dill
    (a drop-in replacement for pickle). Optionally, it can compress the data.

    This is very flexible, but not always the fastest.

    .. tip::
        This format has special support for iterables. If you write an iterator, it will consume the
        iterator. If you read an iterator, it will read the iterator lazily.

    """

    VERSION = 1

    def __init__(self, compress: Optional[str] = None):
        if compress not in _OPEN_FUNCTIONS:
            raise ConfigurationError(f"The {compress} compression format does not exist.")
        self.compress = compress

    def write(self, artifact: T, dir: PathOrStr):
        filename = self._get_artifact_path(dir)
        open_method = _OPEN_FUNCTIONS[self.compress]
        with open_method(filename, "wb") as f:
            pickler = dill.Pickler(file=f)
            pickler.dump(self.VERSION)
            if hasattr(artifact, "__next__"):
                pickler.dump(True)
                for item in cast(Iterable, artifact):
                    pickler.dump(item)
            else:
                pickler.dump(False)
                pickler.dump(artifact)

    def read(self, dir: PathOrStr) -> T:
        filename = self._get_artifact_path(dir)
        open_method = _OPEN_FUNCTIONS[self.compress]
        with open_method(filename, "rb") as f:
            unpickler = dill.Unpickler(file=f)
            version = unpickler.load()
            if version > self.VERSION:
                raise ValueError(
                    f"File {filename} is too recent for this version of {self.__class__}."
                )
            iterator = unpickler.load()
            if iterator:
                return DillFormatIterator(filename)  # type: ignore
            else:
                return unpickler.load()

    def _get_artifact_path(self, dir: PathOrStr) -> Path:
        return Path(dir) / ("data.dill" + _SUFFIXES[_OPEN_FUNCTIONS[self.compress]])


class DillFormatIterator(Iterator[T], Generic[T]):
    """
    An ``Iterator`` class that is used to return an iterator from :meth:`tango.format.DillFormat.read`.
    """

    def __init__(self, filename: PathOrStr):
        self.f: Optional[IO[Any]] = _open_compressed(filename, "rb")
        self.unpickler = dill.Unpickler(self.f)
        version = self.unpickler.load()
        if version > DillFormat.VERSION:
            raise ValueError(f"File {filename} is too recent for this version of {self.__class__}.")
        iterator = self.unpickler.load()
        if not iterator:
            raise ValueError(
                f"Tried to open {filename} as an iterator, but it does not store an iterator."
            )

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.f is None:
            raise StopIteration()
        try:
            return self.unpickler.load()
        except EOFError:
            self.f.close()
            self.f = None
            raise StopIteration()


@Format.register("json")
class JsonFormat(Format[T], Generic[T]):
    """This format writes the artifact as a single file in json format.
    Optionally, it can compress the data. This is very flexible, but not always the fastest.

    .. tip::
        This format has special support for iterables. If you write an iterator, it will consume the
        iterator. If you read an iterator, it will read the iterator lazily.
    """

    VERSION = 2

    def __init__(self, compress: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        if compress not in _OPEN_FUNCTIONS:
            raise ConfigurationError(f"The {compress} compression format does not exist.")
        self.compress = compress

    @staticmethod
    def _encoding_fallback(unencodable: Any):
        try:
            import torch

            if isinstance(unencodable, torch.Tensor):
                if len(unencodable.shape) == 0:
                    return unencodable.item()
                else:
                    raise TypeError(
                        "Tensors must have 1 element and no dimensions to be JSON serializable."
                    )
        except ImportError:
            pass

        if dataclasses.is_dataclass(unencodable):
            result = dataclasses.asdict(unencodable)
            module = type(unencodable).__module__
            qualname = type(unencodable).__qualname__
            if module == "builtins":
                result["_dataclass"] = qualname
            else:
                result["_dataclass"] = [module, qualname]
            return result

        raise TypeError(f"Object of type {type(unencodable)} is not JSON serializable")

    @staticmethod
    def _decoding_fallback(o: Dict) -> Any:
        if "_dataclass" in o:
            classname: Union[str, List[str]] = o.pop("_dataclass")
            if isinstance(classname, list) and len(classname) == 2:
                module, classname = classname
                constructor: Callable = importlib.import_module(module)  # type: ignore
                for item in classname.split("."):
                    constructor = getattr(constructor, item)
            elif isinstance(classname, str):
                constructor = globals()[classname]
            else:
                raise RuntimeError(f"Could not parse {classname} as the name of a dataclass.")
            return constructor(**o)
        return o

    def write(self, artifact: T, dir: PathOrStr):
        open_method = _OPEN_FUNCTIONS[self.compress]
        if hasattr(artifact, "__next__"):
            filename = self._get_artifact_path(dir, iterator=True)
            with open_method(filename, "wt") as f:
                for item in cast(Iterable, artifact):
                    json.dump(item, f, default=self._encoding_fallback)
                    f.write("\n")
        else:
            filename = self._get_artifact_path(dir, iterator=False)
            with open_method(filename, "wt") as f:
                json.dump(artifact, f, default=self._encoding_fallback)

    def read(self, dir: PathOrStr) -> T:
        iterator_filename = self._get_artifact_path(dir, iterator=True)
        iterator_exists = iterator_filename.exists()
        non_iterator_filename = self._get_artifact_path(dir, iterator=False)
        non_iterator_exists = non_iterator_filename.exists()

        if iterator_exists and non_iterator_exists:
            self.logger.warning(
                "Both %s and %s exist. Ignoring %s.",
                iterator_filename,
                non_iterator_filename,
                iterator_filename,
            )
            iterator_exists = False

        if not iterator_exists and not non_iterator_exists:
            raise IOError("Attempting to read non-existing data from %s", dir)
        if iterator_exists and not non_iterator_exists:
            return JsonFormatIterator(iterator_filename)  # type: ignore
        elif not iterator_exists and non_iterator_exists:
            open_method = _OPEN_FUNCTIONS[self.compress]
            with open_method(non_iterator_filename, "rt") as f:
                return json.load(f, object_hook=self._decoding_fallback)
        else:
            raise RuntimeError("This should be impossible.")

    def _get_artifact_path(self, dir: PathOrStr, iterator: bool = False) -> Path:
        return Path(dir) / (
            ("data.jsonl" if iterator else "data.json") + _SUFFIXES[_OPEN_FUNCTIONS[self.compress]]
        )


class JsonFormatIterator(Iterator[T], Generic[T]):
    """
    An ``Iterator`` class that is used to return an iterator from :meth:`tango.format.JsonFormat.read`.
    """

    def __init__(self, filename: PathOrStr):
        self.f: Optional[IO[Any]] = _open_compressed(filename, "rt")

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.f is None:
            raise StopIteration()
        try:
            line = self.f.readline()
            if len(line) <= 0:
                raise EOFError()
            return json.loads(line, object_hook=JsonFormat._decoding_fallback)
        except EOFError:
            self.f.close()
            self.f = None
            raise StopIteration()


@Format.register("text")
class TextFormat(Format[Union[str, Iterable[str]]]):
    """This format writes the artifact as a single file in text format.
    Optionally, it can compress the data. This is very flexible, but not always the fastest.

    This format can only write strings, or iterable of strings.

    .. tip::
        This format has special support for iterables. If you write an iterator, it will consume the
        iterator. If you read an iterator, it will read the iterator lazily.

        Be aware that if your strings contain newlines, you will read out more strings than you wrote.
        For this reason, it's often advisable to use :class:`JsonFormat` instead. With :class:`JsonFormat`,
        all special characters are escaped, strings are quoted, but it's all still human-readable.
    """

    VERSION = 1

    def __init__(self, compress: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        if compress not in _OPEN_FUNCTIONS:
            raise ConfigurationError(f"The {compress} compression format does not exist.")
        self.compress = compress

    def write(self, artifact: Union[str, Iterable[str]], dir: PathOrStr):
        open_method = _OPEN_FUNCTIONS[self.compress]
        if hasattr(artifact, "__next__"):
            filename = self._get_artifact_path(dir, iterator=True)
            with open_method(filename, "wt") as f:
                for item in cast(Iterable, artifact):
                    f.write(str(item))
                    f.write("\n")
        else:
            filename = self._get_artifact_path(dir, iterator=False)
            with open_method(filename, "wt") as f:
                f.write(str(artifact))

    def read(self, dir: PathOrStr) -> Union[str, Iterable[str]]:
        iterator_filename = self._get_artifact_path(dir, iterator=True)
        iterator_exists = iterator_filename.exists()
        non_iterator_filename = self._get_artifact_path(dir, iterator=False)
        non_iterator_exists = non_iterator_filename.exists()

        if iterator_exists and non_iterator_exists:
            self.logger.warning(
                "Both %s and %s exist. Ignoring %s.",
                iterator_filename,
                non_iterator_filename,
                iterator_filename,
            )
            iterator_exists = False

        if not iterator_exists and not non_iterator_exists:
            raise IOError("Attempting to read non-existing data from %s", dir)
        if iterator_exists and not non_iterator_exists:
            return TextFormatIterator(iterator_filename)  # type: ignore
        elif not iterator_exists and non_iterator_exists:
            open_method = _OPEN_FUNCTIONS[self.compress]
            with open_method(non_iterator_filename, "rt") as f:
                return f.read()
        else:
            raise RuntimeError("This should be impossible.")

    def _get_artifact_path(self, dir: PathOrStr, iterator: bool = False) -> Path:
        return Path(dir) / (
            ("texts.txt" if iterator else "text.txt") + _SUFFIXES[_OPEN_FUNCTIONS[self.compress]]
        )


class TextFormatIterator(Iterator[str]):
    """
    An ``Iterator`` class that is used to return an iterator from :meth:`tango.format.TextFormat.read`.
    """

    def __init__(self, filename: PathOrStr):
        self.f: Optional[IO[Any]] = _open_compressed(filename, "rt")

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self.f is None:
            raise StopIteration()
        try:
            line = self.f.readline()
            if len(line) <= 0:
                raise EOFError()
            return line
        except EOFError:
            self.f.close()
            self.f = None
            raise StopIteration()


@Format.register("sqlite_sequence")
class SqliteSequenceFormat(Format[Sequence[T]]):
    VERSION = 3

    FILENAME = "data.sqlite"

    def write(self, artifact: Sequence[T], dir: Union[str, PathLike]):
        dir = Path(dir)
        try:
            (dir / self.FILENAME).unlink()
        except FileNotFoundError:
            pass
        if isinstance(artifact, SqliteSparseSequence):
            artifact.copy_to(dir / self.FILENAME)
        else:
            sqlite = SqliteSparseSequence(dir / self.FILENAME)
            sqlite.extend(artifact)

    def read(self, dir: Union[str, PathLike]) -> Sequence[T]:
        dir = Path(dir)
        return SqliteSparseSequence(dir / self.FILENAME, read_only=True)


@Format.register("sqlite")
class SqliteDictFormat(Format[DatasetDict]):
    """This format works specifically on results of type :class:`~tango.common.DatasetDict`. It writes those
    datasets into Sqlite databases.

    During reading, the advantage is that the dataset can be read lazily. Reading a result that is stored
    in :class:`SqliteDictFormat` takes milliseconds. No actual reading takes place until you access individual
    instances.

    During writing, you have to take some care to take advantage of the same trick. Recall that
    :class:`~tango.DatasetDict` is basically a map, mapping split names to lists of instances. If you ensure
    that those lists of instances are of type :class:`~tango.common.sequences.SqliteSparseSequence`, then writing
    the results in :class:`SqliteDictFormat` can in many cases be instantaneous.

    Here is an example of the pattern to use to make writing fast:

    .. code-block:: Python

        @Step.register("my_step")
        class MyStep(Step[DatasetDict]):

            FORMAT: Format = SqliteDictFormat()
            VERSION = "001"

            def run(self, ...) -> DatasetDict:
                result: Dict[str, Sequence] = {}
                for split_name in my_list_of_splits:
                    output_split = SqliteSparseSequence(self.work_dir / f"{split_name}.sqlite")
                    for instance in instances:
                        output_split.append(instance)
                    result[split_name] = output_split

                metadata = {}
                return DatasetDict(result, metadata)

    Observe how for each split, we create a :class:`~tango.common.sequences.SqliteSparseSequence` in the step's
    work directory (accessible with :meth:`~tango.step.Step.work_dir`). This has the added advantage that if the
    step fails and you have to re-run it, the previous results that were already written to the
    :class:`~tango.common.sequences.SqliteSparseSequence` are still there. You could replace the inner ``for``
    loop like this to take advantage:

    .. code-block:: Python

        output_split = SqliteSparseSequence(self.work_dir / f"{split_name}.sqlite")
        for instance in instances[len(output_split):]:      # <-- here is the difference
            output_split.append(instance)
        result[split_name] = output_split

    This works because when you re-run the step, the work directory will still be there, so ``output_split`` is
    not empty when you open it.
    """

    VERSION = 3

    def write(self, artifact: DatasetDict, dir: Union[str, PathLike]):
        dir = Path(dir)
        with gzip.open(dir / "metadata.dill.gz", "wb") as f:
            dill.dump(artifact.metadata, f)
        for split_name, split in artifact.splits.items():
            filename = f"{split_name}.sqlite"
            if not filename_is_safe(filename):
                raise ValueError(f"{split_name} is not a valid name for a split.")
            try:
                (dir / filename).unlink()
            except FileNotFoundError:
                pass
            if isinstance(split, SqliteSparseSequence):
                split.copy_to(dir / filename)
            else:
                sqlite = SqliteSparseSequence(dir / filename)
                sqlite.extend(split)

    def read(self, dir: Union[str, PathLike]) -> DatasetDict:
        dir = Path(dir)
        with gzip.open(dir / "metadata.dill.gz", "rb") as f:
            metadata = dill.load(f)
        splits = {
            filename.stem: SqliteSparseSequence(filename, read_only=True)
            for filename in dir.glob("*.sqlite")
        }
        return DatasetDict(metadata=metadata, splits=splits)
