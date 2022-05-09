from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import yaml

from .common.aliases import PathOrStr
from .common.from_params import FromParams
from .common.params import Params


@dataclass
class TangoGlobalSettings(FromParams):
    """
    Defines global settings for tango.
    """

    workspace: Optional[Dict[str, Any]] = None
    """
    Parameters to initialize a :class:`tango.workspace.Workspace` with.
    """

    include_package: Optional[List[str]] = None
    """
    An list of modules where custom registered steps or classes can be found.
    """

    log_level: Optional[str] = None
    """
    The log level to use. Options are "debug", "info", "warning", and "error".

    .. note::
        This does not affect the :data:`~tango.common.logging.cli_logger`
        or logs from :class:`~tango.common.Tqdm` progress bars.

    """

    file_friendly_logging: Optional[bool] = None
    """
    If this flag is set to ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
    down tqdm's output to only once every 10 seconds.
    """

    multiprocessing_start_method: str = "spawn"
    """
    The ``start_method`` to use when starting new multiprocessing workers. Can be "fork", "spawn",
    or "forkserver". Default is "spawn".

    See :func:`multiprocessing.set_start_method()` for more details.
    """

    environment: Optional[Dict[str, str]] = None
    """
    Environment variables that will be set each time ``tango`` is run.
    """

    _path: Optional[Path] = None

    _DEFAULT_LOCATION: ClassVar[Path] = Path.home() / ".config" / "tango.yml"

    @classmethod
    def default(cls) -> "TangoGlobalSettings":
        """
        Initialize the config from files by checking the default locations
        in order, or just return the default if none of the files can be found.
        """
        for directory in (Path("."), cls._DEFAULT_LOCATION.parent):
            for extension in ("yml", "yaml"):
                path = directory / f"tango.{extension}"
                if path.is_file():
                    return cls.from_file(path)
        return cls()

    @classmethod
    def find_or_default(cls, path: Optional[PathOrStr] = None) -> "TangoGlobalSettings":
        """
        Initialize the config from a given configuration file, or falls back to returning
        the default configuration if no file is given.
        """
        if path is not None:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            return cls.from_file(path)
        else:
            return cls.default()

    @property
    def path(self) -> Optional[Path]:
        """
        The path to the file the config was read from.
        """
        return self._path

    @classmethod
    def from_file(cls, path: PathOrStr) -> "TangoGlobalSettings":
        """
        Read settings from a file.
        """
        params = Params.from_file(path)
        params["_path"] = Path(path).resolve()
        return cls.from_params(params)

    def to_file(self, path: PathOrStr) -> None:
        """
        Save the settings to a file.
        """
        data = {
            k: v for k, v in self.to_params().as_dict(quiet=True).items() if not k.startswith("_")
        }
        with open(path, "w") as settings_file:
            yaml.safe_dump(data, settings_file)

    def save(self) -> None:
        """
        Save the settings to the file it was read from.

        :raises ValueError: If the settings was not read from a file.
        """
        if self.path is None:
            raise ValueError("No path given, use .to_file() instead")
        self.to_file(self.path)
