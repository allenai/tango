from enum import Enum, unique
from os import PathLike
from typing import Set, Union

PathOrStr = Union[str, PathLike]


@unique
class EnvVarNames(Enum):
    FILE_FRIENDLY_LOGGING = "FILE_FRIENDLY_LOGGING"
    LOG_LEVEL = "TANGO_LOG_LEVEL"
    CLI_LOGGER_ENABLED = "TANGO_CLI_LOGGER_ENABLED"
    LOGGING_HOST = "TANGO_LOGGING_HOST"
    LOGGING_PORT = "TANGO_LOGGING_PORT"
    LOGGING_PREFIX = "TANGO_LOGGING_PREFIX"
    CONSOLE_WIDTH = "TANGO_CONSOLE_WIDTH"

    @classmethod
    def values(cls) -> Set[str]:
        return set(e.value for e in cls)
