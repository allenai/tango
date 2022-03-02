from enum import Enum, unique
from os import PathLike
from typing import Set, Union

PathOrStr = Union[str, PathLike]


@unique
class EnvVarNames(Enum):
    FILE_FRIENDLY_LOGGING_ENV_VAR = "FILE_FRIENDLY_LOGGING"
    LOG_LEVEL_ENV_VAR = "TANGO_LOG_LEVEL"
    CLICK_LOGGER_ENABLED_ENV_VAR = "TANGO_CLICK_LOGGER_ENABLED"
    LOGGING_HOST_ENV_VAR = "TANGO_LOGGING_HOST"
    LOGGING_PORT_ENV_VAR = "TANGO_LOGGING_PORT"

    @classmethod
    def values(cls) -> Set[str]:
        return set(e.value for e in cls)
