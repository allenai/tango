import logging
import random
import re
from abc import abstractmethod
from pathlib import Path
from typing import (
    Optional,
    Any,
    Dict,
    Type,
    Union,
    cast,
    TypeVar,
    Generic,
    Callable,
    TYPE_CHECKING,
)

try:
    from typing import get_origin, get_args  # type: ignore
except ImportError:

    def get_origin(tp):  # type: ignore
        return getattr(tp, "__origin__", None)

    def get_args(tp):  # type: ignore
        return getattr(tp, "__args__", ())


from tango.common.det_hash import det_hash
from tango.common.registrable import Registrable
from tango.common.params import Params
from tango.common.exceptions import ConfigurationError
from tango.common.from_params import (
    pop_and_construct_arg,
    infer_method_params,
    infer_constructor_params,
)
from tango.common.logging import TangoLogger
from tango.format import Format, DillFormat

if TYPE_CHECKING:
    from tango.executor import Executor

logger = logging.getLogger(__name__)

_version_re = re.compile("""^[a-zA-Z0-9]+$""")

T = TypeVar("T")


class Step(Registrable, Generic[T]):
    """
    This class defines one step in your experiment. To write your own step, just derive from this class
    and overwrite the :meth:`run()` method. The :meth:`run()` method must have parameters with type hints.

    ``Step.__init__()`` takes all the arguments we want to run the step with. They get passed
    to :meth:`run()` (almost) as they are. If the arguments are other instances of ``Step``, those
    will be replaced with the step's results before calling :meth:`run()`. Further, there are four special
    parameters:

    * ``step_name`` contains an optional human-readable name for the step. This name is used for
      error messages and the like, and has no consequence on the actual computation.
    * ``cache_results`` specifies whether the results of this step should be cached. If this is
      ``False``, the step is recomputed every time it is needed. If this is not set at all,
      and :attr:`CACHEABLE` is ``True``, we cache if the step is marked as :attr:`DETERMINISTIC`,
      and we don't cache otherwise.
    * ``step_format`` gives you a way to override the step's default format (which is given in :attr:`FORMAT`).
    * ``step_config`` is the original raw part of the experiment config corresponding to this step.
      This can be accessed via the :attr:`config` property within each step's :meth:`run()` method.
    * ``step_executor`` is the :class:`~tango.executor.Executor` being used to run the step.
      This can be accessed via the :attr:`executor` property within each step's :meth:`run()` method.
    """

    DETERMINISTIC: bool = False
    """This describes whether this step can be relied upon to produce the same results every time
    when given the same inputs. If this is ``False``, the step can't be cached, and neither can any
    step that depends on it."""

    CACHEABLE: Optional[bool] = None
    """This provides a direct way to turn off caching. For example, a step that reads a HuggingFace
    dataset doesn't need to be cached, because HuggingFace datasets already have their own caching
    mechanism. But it's still a deterministic step, and all following steps are allowed to cache.
    If it is ``None``, the step figures out by itself whether it should be cacheable or not."""

    VERSION: Optional[str] = None
    """This is optional, but recommended. Specifying a version gives you a way to tell Tango that
    a step has changed during development, and should now be recomputed. This doesn't invalidate
    the old results, so when you revert your code, the old cache entries will stick around and be
    picked up."""

    FORMAT: Format = DillFormat("gz")
    """This specifies the format the results of this step will be serialized in. See the documentation
    for :class:`~tango.format.Format` for details."""

    def __init__(
        self,
        step_name: Optional[str] = None,
        cache_results: Optional[bool] = None,
        step_format: Optional[Format] = None,
        step_config: Optional[Dict[str, Any]] = None,
        step_executor: Optional["Executor"] = None,
        **kwargs,
    ):
        self.logger = cast(TangoLogger, logging.getLogger(self.__class__.__name__))

        if self.VERSION is not None:
            assert _version_re.match(
                self.VERSION
            ), f"Invalid characters in version '{self.VERSION}'"
        self.kwargs = kwargs

        if step_format is None:
            self.format = self.FORMAT
            if isinstance(self.format, type):
                self.format = self.format()
        else:
            self.format = step_format

        self.unique_id_cache: Optional[str] = None
        if step_name is None:
            self.name = self.unique_id
        else:
            self.name = step_name

        if cache_results is True:
            if not self.CACHEABLE:
                raise ConfigurationError(
                    f"Step {self.name} is configured to use the cache, but it's not a cacheable step."
                )
            if not self.DETERMINISTIC:
                logger.warning(
                    f"Step {self.name} is going to be cached despite not being deterministic."
                )
            self.cache_results = True
        elif cache_results is False:
            self.cache_results = False
        elif cache_results is None:
            c = (self.DETERMINISTIC, self.CACHEABLE)
            if c == (False, None):
                self.cache_results = False
            elif c == (True, None):
                self.cache_results = True
            elif c == (False, False):
                self.cache_results = False
            elif c == (True, False):
                self.cache_results = False
            elif c == (False, True):
                logger.warning(
                    f"Step {self.name} is set to be cacheable despite not being deterministic."
                )
                self.cache_results = True
            elif c == (True, True):
                self.cache_results = True
            else:
                assert False, "Step.DETERMINISTIC or step.CACHEABLE are set to an invalid value."
        else:
            raise ConfigurationError(
                f"Step {self.name}'s cache_results parameter is set to an invalid value."
            )

        self.work_dir_for_run: Optional[
            Path
        ] = None  # This is set only while the run() method runs.

        self._config = step_config
        self._executor = step_executor

    @classmethod
    def from_params(  # type: ignore[override]
        cls: Type["Step"],
        params: Union[Params, dict, str],
        constructor_to_call: Callable[..., "Step"] = None,
        constructor_to_inspect: Union[Callable[..., "Step"], Callable[["Step"], None]] = None,
        step_name: Optional[str] = None,
        step_config: Optional[Dict[str, Any]] = None,
        step_executor: Optional["Executor"] = None,
        **extras,
    ) -> "Step":
        # Why do we need a custom from_params? Step classes have a run() method that takes all the
        # parameters necessary to perform the step. The __init__() method of the step takes those
        # same parameters, but each of them could be wrapped in another Step instead of being
        # supplied directly. from_params() doesn't know anything about these shenanigans, so
        # we have to supply the necessary logic here.

        if constructor_to_call is not None:
            raise ConfigurationError(
                f"{cls.__name__}.from_params cannot be called with a constructor_to_call."
            )
        if constructor_to_inspect is not None:
            raise ConfigurationError(
                f"{cls.__name__}.from_params cannot be called with a constructor_to_inspect."
            )

        if isinstance(params, str):
            params = Params({"type": params})

        if not isinstance(params, Params):
            if isinstance(params, dict):
                params = Params(params)
            else:
                raise ConfigurationError(
                    "from_params was passed a ``params`` object that was not a ``Params``. This probably "
                    "indicates malformed parameters in a configuration file, where something that "
                    "should have been a dictionary was actually a list, or something else. "
                    f"This happened when constructing an object of type {cls}."
                )

        as_registrable = cast(Type[Registrable], cls)
        if "type" in params and params["type"] not in as_registrable.list_available():
            as_registrable.search_modules(params["type"])
        choice = params.pop_choice(
            "type", choices=as_registrable.list_available(), default_to_first_choice=True
        )
        subclass, constructor_name = as_registrable.resolve_class_name(choice)
        if not issubclass(subclass, Step):
            # This can happen if `choice` is a fully qualified name.
            raise ConfigurationError(
                f"Tried to make a Step of type {choice}, but ended up with a {subclass}."
            )

        parameters = infer_method_params(subclass, subclass.run, infer_kwargs=False)
        del parameters["self"]
        init_parameters = infer_constructor_params(subclass)
        del init_parameters["self"]
        del init_parameters["kwargs"]
        parameter_overlap = parameters.keys() & init_parameters.keys()
        assert len(parameter_overlap) <= 0, (
            f"If this assert fails it means that you wrote a Step with a run() method that takes one of the "
            f"reserved parameters ({', '.join(init_parameters.keys())})"
        )
        parameters.update(init_parameters)

        kwargs: Dict[str, Any] = {}
        accepts_kwargs = False
        for param_name, param in parameters.items():
            if param.kind == param.VAR_KEYWORD:
                # When a class takes **kwargs we store the fact that the method allows extra keys; if
                # we get extra parameters, instead of crashing, we'll just pass them as-is to the
                # constructor, and hope that you know what you're doing.
                accepts_kwargs = True
                continue

            explicitly_set = param_name in params
            constructed_arg = pop_and_construct_arg(
                subclass.__name__,
                param_name,
                param.annotation,
                param.default,
                params,
                **extras,
            )

            # If the param wasn't explicitly set in `params` and we just ended up constructing
            # the default value for the parameter, we can just omit it.
            # Leaving it in can cause issues with **kwargs in some corner cases, where you might end up
            # with multiple values for a single parameter (e.g., the default value gives you lazy=False
            # for a dataset reader inside **kwargs, but a particular dataset reader actually hard-codes
            # lazy=True - the superclass sees both lazy=True and lazy=False in its constructor).
            if explicitly_set or constructed_arg is not param.default:
                kwargs[param_name] = constructed_arg

        if accepts_kwargs:
            kwargs.update(params)
        else:
            params.assert_empty(subclass.__name__)

        return subclass(
            step_name=step_name, step_config=step_config, step_executor=step_executor, **kwargs
        )

    @abstractmethod
    def run(self, **kwargs) -> T:
        """
        Execute the step's action.

        This method needs to be implemented when creating a ``Step`` subclass, but
        it shouldn't be called directly. Instead, call :meth:`run_with_work_dir()`.
        """
        raise NotImplementedError()

    def run_with_work_dir(self, work_dir_for_run: Path) -> T:
        """
        Run the step with a working directory.
        """
        if self.work_dir_for_run is not None:
            raise ValueError("You can only run a Step's run() method once at a time.")

        logger.info("Starting run for step %s of type %s", self.name, self.__class__.__name__)

        if self.DETERMINISTIC:
            random.seed(784507111)

        try:
            self.work_dir_for_run = work_dir_for_run
            self.work_dir_for_run.mkdir(exist_ok=True, parents=True)
            return self.run(**self.kwargs)
        finally:
            # No cleanup, as we want to keep the directory for restarts or serialization.
            self.work_dir_for_run = None

    @property
    def work_dir(self) -> Path:
        """
        The working directory that a step can use while its ``run()`` method runs.

        This directory stays around across restarts. You cannot assume that it is empty when your
        step runs, but you can use it to store information that helps you restart a step if it
        got killed half-way through the last time it ran."""
        if self.work_dir_for_run is None:
            raise RuntimeError(
                "You can only call this method while the step is running with a working directory. "
                "Did you call '.run()' directly? You should only run a step with 'run_with_work_dir()'."
            )
        return self.work_dir_for_run

    @property
    def executor(self) -> "Executor":
        """
        The :class:`~tango.executor.Executor` being used to run this step.
        """
        if self._executor is None:
            raise ValueError(f"No Executor has been assigned to this step! ('{self.name}')")
        else:
            return self._executor

    @property
    def config(self) -> Dict[str, Any]:
        """
        The raw configuration parameters for this step.
        """
        if self._config is None:
            raise ValueError(f"No config has been assigned to this step! ('{self.name}')")
        else:
            return self._config

    def det_hash_object(self) -> Any:
        return self.unique_id

    @property
    def unique_id(self) -> str:
        """Returns the unique ID for this step.

        Unique IDs are of the shape ``$class_name-$version-$hash``, where the hash is the hash of the
        inputs for deterministic steps, and a random string of characters for non-deterministic ones."""
        if self.unique_id_cache is None:
            self.unique_id_cache = self.__class__.__name__
            if self.VERSION is not None:
                self.unique_id_cache += "-"
                self.unique_id_cache += self.VERSION

            self.unique_id_cache += "-"
            if self.DETERMINISTIC:
                self.unique_id_cache += det_hash(
                    (
                        (self.format.__class__.__module__, self.format.__class__.__qualname__),
                        self.format.VERSION,
                        self.kwargs,
                    )
                )[:32]
            else:
                self.unique_id_cache += det_hash(random.getrandbits((58 ** 32).bit_length()))[:32]

        return self.unique_id_cache

    def __hash__(self):
        return hash(self.unique_id)

    def __eq__(self, other):
        if isinstance(other, Step):
            return self.unique_id == other.unique_id
        else:
            return False
