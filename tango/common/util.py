import importlib
import pkgutil
import signal
import string
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Union

from .aliases import PathOrStr
from .exceptions import SigTermReceived


def _handle_sigterm(sig, frame):
    raise SigTermReceived


def install_sigterm_handler():
    signal.signal(signal.SIGTERM, _handle_sigterm)


@contextmanager
def push_python_path(path: PathOrStr):
    """
    Prepends the given path to `sys.path`.

    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
    """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


_extra_imported_modules: Set[str] = set()


def get_extra_imported_modules() -> Set[str]:
    return _extra_imported_modules


def import_extra_module(package_name: str) -> None:
    global _extra_imported_modules
    import_module_and_submodules(package_name)
    _extra_imported_modules.add(package_name)


def resolve_module_name(package_name: str) -> Tuple[str, Path]:
    base_path = Path(".")
    package_path = Path(package_name)
    if not package_path.exists():
        raise ValueError(f"'{package_path}' looks like a path, but the path does not exist")

    parent = package_path.parent
    while parent != parent.parent:
        if (parent / "__init__.py").is_file():
            parent = parent.parent
        else:
            base_path = parent
            break

    package_name = str(package_path.relative_to(base_path)).replace("/", ".")

    if package_path.is_file():
        if package_path.name == "__init__.py":
            # If `__init__.py` file, resolve to the parent module.
            package_name = package_name[: -len(".__init__.py")]
        elif package_name.endswith(".py"):
            package_name = package_name[:-3]

        if not package_name:
            raise ValueError(f"invalid package path '{package_path}'")

    return package_name, base_path


def import_module_and_submodules(package_name: str, exclude: Optional[Set[str]] = None) -> None:
    """
    Import all submodules under the given package.

    Primarily useful so that people using tango can specify their own custom packages
    and have their custom classes get loaded and registered.
    """
    # If `package_name` is in the form of a path, convert to the module format.
    if "/" in package_name or package_name.endswith(".py"):
        package_name, base_path = resolve_module_name(package_name)
    else:
        base_path = Path(".")

    if exclude and package_name in exclude:
        return

    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path(base_path):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:  # type: ignore[union-attr]
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage, exclude=exclude)


def _parse_bool(value: Union[bool, str]) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"1", "true", "True", "TRUE"}:
        return True
    return False


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is not None:
        return int(value)
    return None


def find_submodules(
    module: Optional[str] = None,
    match: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
    recursive: bool = True,
) -> Iterable[str]:
    """
    Find tango submodules.
    """
    from fnmatch import fnmatch

    root = Path(__file__).parent / ".."
    if module:
        if module.startswith("tango."):
            module = module.replace("tango.", "", 1)
        for m in module.split("."):
            root = root / m
        module = f"tango.{module}"
    else:
        module = "tango"
    for path in root.iterdir():
        if path.name[0] in {"_", "."}:
            continue
        submodule: str
        if path.is_dir():
            submodule = path.name
        elif path.suffix == ".py":
            submodule = path.name[:-3]
        else:
            continue
        submodule = f"{module}.{submodule}"
        if exclude and any((fnmatch(submodule, pat) for pat in exclude)):
            continue
        if match and not any((fnmatch(submodule, pat) for pat in match)):
            continue
        yield submodule
        if recursive and path.is_dir():
            yield from find_submodules(submodule, match=match, exclude=exclude)


def find_integrations() -> Iterable[str]:
    """
    Find all tango integration modules.
    """
    yield from find_submodules("tango.integrations", recursive=False)


SAFE_FILENAME_CHARS = frozenset("-_.%s%s" % (string.ascii_letters, string.digits))


def filename_is_safe(filename: str) -> bool:
    return all(c in SAFE_FILENAME_CHARS for c in filename)


def could_be_class_name(name: str) -> bool:
    if "." in name and not name.endswith("."):
        return all([_is_valid_python_name(part) for part in name.split(".")])
    else:
        return False


def _is_valid_python_name(name: str) -> bool:
    return bool(name and name[0].isalpha() and name.replace("_", "").isalnum())


def threaded_generator(g, queue_size: int = 16):
    """
    Puts the generating side of a generator into its own thread

    Let's say you have a generator that reads records from disk, and something that consumes the
    generator that spends most of its time in PyTorch. Wouldn't it be great if you could read more
    records while the PyTorch code runs? If you wrap your record-reading generator with
    ``threaded_generator(inner)``, that's exactly what happens. The reading code will run in a new thread,
    while the consuming code runs in the main thread as normal. ``threaded_generator()`` uses a queue
    to hand off items.

    :param queue_size: the maximum queue size for hand-offs between the main thread and the generator thread
    """
    from queue import Queue
    from threading import Thread

    q: Queue = Queue(maxsize=queue_size)

    sentinel = object()

    def fill_queue():
        try:
            for value in g:
                q.put(value)
        finally:
            q.put(sentinel)

    thread = Thread(name=repr(g), target=fill_queue, daemon=True)
    thread.start()

    yield from iter(q.get, sentinel)

    thread.join()


def exception_to_string(e: BaseException) -> str:
    """
    Generates a string that contains an exception plus stack frames based on an exception.

    This became trivial in Python 3.10, but we need to run on Pytohn 3.7 as well.
    """
    if sys.version_info >= (3, 10):
        formatted = traceback.format_exception(e)
    else:
        formatted = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
    return "".join(formatted)
