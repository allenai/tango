from typing import Any, Tuple, Union


class TangoError(Exception):
    """
    Base class for Tango exceptions.
    """


class ConfigurationError(TangoError):
    """
    The exception raised when a Tango object fails to initialize from a config
    that's misconfigured (e.g. missing properties, invalid properties, unknown properties).
    """

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


class RegistryKeyError(ConfigurationError):
    """
    A configuration error that is raised when attempting to get a class by a registered name
    that doesn't exist in the registry.
    """


class SigTermReceived(TangoError):
    """
    Raised when a SIGTERM is caught.
    """
