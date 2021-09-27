from contextlib import contextmanager
import importlib
from pathlib import Path
from os import PathLike
import pkgutil
import signal
import sys
from typing import Union, Optional, Set

from .exceptions import SigTermReceived

PathOrStr = Union[str, PathLike]


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
