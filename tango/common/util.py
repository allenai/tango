import importlib
import pkgutil
import signal
import string
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Set, Union

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


def import_module_and_submodules(package_name: str, exclude: Optional[Set[str]] = None) -> None:
    """
    Import all submodules under the given package.

    Primarily useful so that people using tango can specify their own custom packages
    and have their custom classes get loaded and registered.
    """
    if exclude and package_name in exclude:
        return

    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path("."):
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
    return bool(name and name[0].isalpha() and name.isalnum())
