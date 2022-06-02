from typing import TYPE_CHECKING, Any, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from tango.step import Step
    from tango.step_info import StepInfo, StepState


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


class CliRunError(TangoError):
    """
    Raised when `tango run` command fails.
    """


class IntegrationMissingError(TangoError):
    """
    Raised when an integration can't be used due to missing dependencies.
    """

    def __init__(self, integration: str, dependencies: Optional[Set[str]] = None):
        self.integration = integration
        self.dependencies = dependencies or {integration}
        msg = (
            f"'{self.integration}' integration can't be used due to "
            f"missing dependencies ({', '.join(self.dependencies)})"
        )
        super().__init__(msg)


class StepStateError(TangoError):
    """
    Raised when a step is in an unexpected state.
    """

    def __init__(
        self,
        step: Union["Step", "StepInfo"],
        step_state: "StepState",
        context: Optional[str] = None,
    ):
        self.step_state = step_state
        self.step_id = step.unique_id
        msg = f"Step '{self.step_id}' is in unexpected state '{self.step_state.value}'"
        if context is not None:
            msg = msg + " " + context
        super().__init__(msg)
