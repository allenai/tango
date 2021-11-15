import collections.abc
import inspect
import logging
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

from tango.common._det_hash import CustomDetHash
from tango.common.exceptions import ConfigurationError
from tango.common.lazy import Lazy
from tango.common.params import Params

try:
    # For PEP 604 support (python >= 3.10)
    from types import UnionType  # type: ignore[attr-defined]
except ImportError:

    class UnionType:  # type: ignore
        pass


logger = logging.getLogger(__name__)

T = TypeVar("T", bound="FromParams")

# If a function parameter has no default value specified,
# this is what the inspect module returns.
_NO_DEFAULT = inspect.Parameter.empty


def takes_arg(obj, arg: str) -> bool:
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return arg in signature.parameters


def takes_kwargs(obj) -> bool:
    """
    Checks whether a provided object takes in any positional arguments.
    Similar to takes_arg, we do this for both the __init__ function of
    the class or a function / method
    Otherwise, we raise an error
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD  # type: ignore
        for p in signature.parameters.values()
    )


def is_base_registrable(cls) -> bool:
    """
    Checks whether this is a class that directly inherits from Registrable, or is a subclass of such
    a class.
    """
    from tango.common.registrable import (
        Registrable,  # import here to avoid circular imports
    )

    if not issubclass(cls, Registrable):
        return False
    method_resolution_order = inspect.getmro(cls)[1:]
    for base_class in method_resolution_order:
        if issubclass(base_class, Registrable) and base_class is not Registrable:
            return False
    return True


def remove_optional(annotation: type):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin == Union:
        return Union[tuple([arg for arg in args if arg != type(None)])]  # noqa: E721
    else:
        return annotation


def infer_constructor_params(
    cls: Type[T], constructor: Union[Callable[..., T], Callable[[T], None]] = None
) -> Dict[str, inspect.Parameter]:
    if constructor is None:
        constructor = cls.__init__
    return infer_method_params(cls, constructor)


infer_params = infer_constructor_params  # Legacy name


def infer_method_params(
    cls: Type[T], method: Callable, infer_kwargs: bool = True
) -> Dict[str, inspect.Parameter]:
    signature = inspect.signature(method)
    parameters = dict(signature.parameters)

    has_kwargs = False
    var_positional_key = None
    for param_name in parameters.keys():
        param = parameters[param_name]
        if param.kind == param.VAR_KEYWORD:
            has_kwargs = True
        elif param.kind == param.VAR_POSITIONAL:
            var_positional_key = param.name
        if isinstance(param.annotation, str):
            # For Python < 3.10, if the module where this class was defined used
            # `from __future__ import annotation`, the annotation will be a str,
            # so we need to resolve it using `get_type_hints` from the typing module.
            # See https://www.python.org/dev/peps/pep-0563/ for more info.
            parameters[param_name] = param.replace(annotation=get_type_hints(method)[param_name])

    if var_positional_key:
        del parameters[var_positional_key]

    if not has_kwargs or not infer_kwargs:
        return parameters

    # "mro" is "method resolution order". The first one is the current class, the next is the
    # first superclass, and so on. We take the first superclass we find that inherits from
    # FromParams.
    super_class = None
    for super_class_candidate in cls.mro()[1:]:
        if issubclass(super_class_candidate, FromParams):
            super_class = super_class_candidate
            break
    if super_class:
        super_parameters = infer_params(super_class)
    else:
        super_parameters = {}

    return {**super_parameters, **parameters}  # Subclass parameters overwrite superclass ones


def create_kwargs(
    constructor: Callable[..., T], cls: Type[T], params: Params, **extras
) -> Dict[str, Any]:
    """
    Given some class, a ``Params`` object, and potentially other keyword arguments,
    create a dict of keyword args suitable for passing to the class's constructor.

    The function does this by finding the class's constructor, matching the constructor
    arguments to entries in the ``params`` object, and instantiating values for the parameters
    using the type annotation and possibly a from_params method.

    Any values that are provided in the ``extras`` will just be used as is.
    For instance, you might provide an existing ``Vocabulary`` this way.
    """
    # Get the signature of the constructor.

    kwargs: Dict[str, Any] = {}

    parameters = infer_params(cls, constructor)
    accepts_kwargs = False

    # Iterate over all the constructor parameters and their annotations.
    for param_name, param in parameters.items():
        # Skip "self". You're not *required* to call the first parameter "self",
        # so in theory this logic is fragile, but if you don't call the self parameter
        # "self" you kind of deserve what happens.
        if param_name == "self":
            continue

        if param.kind == param.VAR_KEYWORD:
            # When a class takes **kwargs, we do two things: first, we assume that the **kwargs are
            # getting passed to the super class, so we inspect super class constructors to get
            # allowed arguments (that happens in `infer_params` above).  Second, we store the fact
            # that the method allows extra keys; if we get extra parameters, instead of crashing,
            # we'll just pass them as-is to the constructor, and hope that you know what you're
            # doing.
            accepts_kwargs = True
            continue

        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        annotation = remove_optional(param.annotation)

        explicitly_set = param_name in params
        constructed_arg = pop_and_construct_arg(
            cls.__name__, param_name, annotation, param.default, params, **extras
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
        for key in list(params):
            kwargs[key] = params.pop(key, keep_as_dict=True)
        for key, value in extras.items():
            kwargs[key] = value
    params.assert_empty(cls.__name__)
    return kwargs


def create_extras(cls: Type[T], extras: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a dictionary of extra arguments, returns a dictionary of
    kwargs that actually are a part of the signature of the cls.from_params
    (or cls) method.
    """
    subextras: Dict[str, Any] = {}
    if hasattr(cls, "from_params"):
        from_params_method = cls.from_params  # type: ignore
    else:
        # In some rare cases, we get a registered subclass that does _not_ have a
        # from_params method (this happens with Activations, for instance, where we
        # register pytorch modules directly).  This is a bit of a hack to make those work,
        # instead of adding a `from_params` method for them somehow. Then the extras
        # in the class constructor are what we are looking for, to pass on.
        from_params_method = cls
    if takes_kwargs(from_params_method):
        # If annotation.params accepts **kwargs, we need to pass them all along.
        # For example, `BasicTextFieldEmbedder.from_params` requires a Vocabulary
        # object, but `TextFieldEmbedder.from_params` does not.
        subextras = extras
    else:
        # Otherwise, only supply the ones that are actual args; any additional ones
        # will cause a TypeError.
        subextras = {k: v for k, v in extras.items() if takes_arg(from_params_method, k)}
    return subextras


def pop_and_construct_arg(
    class_name: str, argument_name: str, annotation: Type, default: Any, params: Params, **extras
) -> Any:
    """
    Does the work of actually constructing an individual argument for
    [``create_kwargs``](./#create_kwargs).

    Here we're in the inner loop of iterating over the parameters to a particular constructor,
    trying to construct just one of them.  The information we get for that parameter is its name,
    its type annotation, and its default value; we also get the full set of ``Params`` for
    constructing the object (which we may mutate), and any ``extras`` that the constructor might
    need.

    We take the type annotation and default value here separately, instead of using an
    ``inspect.Parameter`` object directly, so that we can handle ``Union`` types using recursion on
    this method, trying the different annotation types in the union in turn.
    """
    # We used `argument_name` as the method argument to avoid conflicts with 'name' being a key in
    # `extras`, which isn't _that_ unlikely.  Now that we are inside the method, we can switch back
    # to using `name`.
    name = argument_name

    # Some constructors expect extra non-parameter items, e.g. vocab: Vocabulary.
    # We check the provided `extras` for these and just use them if they exist.
    if name in extras:
        if name not in params:
            return extras[name]
        else:
            logger.warning(
                f"Parameter {name} for class {class_name} was found in both "
                "**extras and in params. Using the specification found in params, "
                "but you probably put a key in a config file that you didn't need, "
                "and if it is different from what we get from **extras, you might "
                "get unexpected behavior."
            )

    popped_params = params.pop(name, default) if default != _NO_DEFAULT else params.pop(name)
    if popped_params is None:
        return None

    return construct_arg(class_name, name, popped_params, annotation, default, **extras)


def _params_contain_step(o: Any) -> bool:
    from tango.step import Step

    if isinstance(o, Step):
        return True
    elif isinstance(o, str):
        return False  # Confusingly, str is an Iterable of itself, resulting in infinite recursion.
    elif isinstance(o, dict) or isinstance(o, Params):
        return _params_contain_step(o.values())
    elif isinstance(o, Iterable):
        return any(_params_contain_step(p) for p in o)
    else:
        return False


def construct_arg(
    class_name: str,
    argument_name: str,
    popped_params: Params,
    annotation: Type,
    default: Any,
    could_be_step: bool = True,
    **extras,
) -> Any:
    """
    The first two parameters here are only used for logging if we encounter an error.
    """
    from tango.step import Step

    if could_be_step:
        # We try parsing as a step _first_. Parsing as a non-step always succeeds, because
        # it will fall back to returning a dict. So we can't try parsing as a non-step first.
        backup_params = deepcopy(popped_params)
        try:
            return construct_arg(
                class_name,
                argument_name,
                popped_params,
                Step[annotation],  # type: ignore
                default,
                could_be_step=False,
                **extras,
            )
        except (ValueError, TypeError, ConfigurationError, AttributeError):
            popped_params = backup_params

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", [])

    # The parameter is optional if its default value is not the "no default" sentinel.
    optional = default != _NO_DEFAULT

    if (inspect.isclass(annotation) and issubclass(annotation, FromParams)) or (
        inspect.isclass(origin) and issubclass(origin, FromParams)
    ):
        if popped_params is default:
            return default
        elif origin is None and isinstance(popped_params, annotation):
            return popped_params
        elif popped_params is not None:
            subextras = create_extras(annotation, extras)

            # In some cases we allow a string instead of a param dict, so
            # we need to handle that case separately.
            if isinstance(popped_params, str):
                if origin != Step:
                    # We don't allow single strings to be upgraded to steps.
                    # Since we try everything as a step first, upgrading strings to
                    # steps automatically would cause confusion every time a step
                    # name conflicts with any string anywhere in a config.
                    popped_params = Params({"type": popped_params})
            elif isinstance(popped_params, dict):
                popped_params = Params(popped_params)
            elif not isinstance(popped_params, (Params, Step)):
                raise TypeError(
                    f"Expected a `Params` object, found `{popped_params}` instead while constructing "
                    f"parameter '{argument_name}' for `{class_name}`"
                )

            from tango.step import WithUnresolvedSteps

            result: Union[FromParams, WithUnresolvedSteps]
            if isinstance(popped_params, Step):
                result = popped_params
            else:
                if origin != Step and _params_contain_step(popped_params):
                    result = WithUnresolvedSteps(annotation.from_params, popped_params)
                else:
                    result = annotation.from_params(popped_params, **subextras)

            if isinstance(result, Step):
                expected_return_type = args[0]
                return_type = inspect.signature(result.run).return_annotation
                if return_type == inspect.Signature.empty:
                    logger.warning(
                        "Step %s has no return type annotation. Those are really helpful when "
                        "debugging, so we recommend them highly.",
                        result.__class__.__name__,
                    )
                else:
                    try:
                        if not issubclass(return_type, expected_return_type):
                            raise ConfigurationError(
                                f"Step {result.name} returns {return_type}, but "
                                f"we expected {expected_return_type}."
                            )
                    except TypeError:
                        pass

            return result
        elif not optional:
            # Not optional and not supplied, that's an error!
            raise ConfigurationError(f"expected key {argument_name} for {class_name}")
        else:
            return default

    # If the parameter type is a Python primitive, just pop it off
    # using the correct casting pop_xyz operation.
    elif annotation in {int, bool}:
        if type(popped_params) in {int, bool}:
            return annotation(popped_params)
        else:
            raise TypeError(f"Expected {argument_name} to be a {annotation.__name__}.")
    elif annotation == str:
        # Strings are special because we allow casting from Path to str.
        if type(popped_params) == str or isinstance(popped_params, Path):
            return str(popped_params)  # type: ignore
        else:
            raise TypeError(
                f"Expected {argument_name} to be a string, found {popped_params} ({type(popped_params)})"
            )
    elif annotation == float:
        # Floats are special because in Python, you can put an int wherever you can put a float.
        # https://mypy.readthedocs.io/en/stable/duck_type_compatibility.html
        if type(popped_params) in {int, float}:
            return popped_params
        else:
            raise TypeError(f"Expected {argument_name} to be numeric.")
    elif annotation == Path:
        if isinstance(popped_params, (str, Path)):
            return Path(popped_params)
        else:
            raise TypeError(
                f"Expected {argument_name} to be a str or Path, found {popped_params} ({type(popped_params)})"
            )

    # This is special logic for handling types like Dict[str, TokenIndexer],
    # List[TokenIndexer], Tuple[TokenIndexer, Tokenizer], and Set[TokenIndexer],
    # which it creates by instantiating each value from_params and returning the resulting structure.
    elif origin in {collections.abc.Mapping, Mapping, Dict, dict} and len(args) == 2:
        value_cls = annotation.__args__[-1]
        value_dict = {}
        if not isinstance(popped_params, Mapping):
            raise TypeError(
                f"Expected {argument_name} to be a Mapping (probably a dict or a Params object)."
            )

        for key, value_params in popped_params.items():
            value_dict[key] = construct_arg(
                str(value_cls),
                argument_name + "." + key,
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )

        return value_dict

    elif origin in (Tuple, tuple):
        value_list = []

        for i, (value_cls, value_params) in enumerate(zip(annotation.__args__, popped_params)):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return tuple(value_list)

    elif origin in (Set, set) and len(args) == 1:
        value_cls = annotation.__args__[0]

        value_set = set()

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_set.add(value)

        return value_set

    elif origin == Union or isinstance(annotation, UnionType):
        # Storing this so we can recover it later if we need to.
        backup_params = deepcopy(popped_params)

        # We'll try each of the given types in the union sequentially, returning the first one that
        # succeeds.
        error_chain: Optional[Exception] = None
        for arg_annotation in args:
            try:
                return construct_arg(
                    str(arg_annotation),
                    argument_name,
                    popped_params,
                    arg_annotation,
                    default,
                    **extras,
                )
            except (ValueError, TypeError, ConfigurationError, AttributeError) as e:
                # Our attempt to construct the argument may have modified popped_params, so we
                # restore it here.
                popped_params = deepcopy(backup_params)
                e.args = (f"While constructing an argument of type {arg_annotation}",) + e.args
                e.__cause__ = error_chain
                error_chain = e

        # If none of them succeeded, we crash.
        config_error = ConfigurationError(
            f"Failed to construct argument {argument_name} with type {annotation}."
        )
        config_error.__cause__ = error_chain
        raise config_error
    elif origin == Lazy:
        if popped_params is default:
            return default

        value_cls = args[0]
        return Lazy(value_cls, params=deepcopy(popped_params))  # type: ignore

    # For any other kind of iterable, we will just assume that a list is good enough, and treat
    # it the same as List. This condition needs to be at the end, so we don't catch other kinds
    # of Iterables with this branch.
    elif origin in {collections.abc.Iterable, Iterable, List, list} and len(args) == 1:
        value_cls = annotation.__args__[0]

        value_list = []

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return value_list

    elif (inspect.isclass(annotation) or inspect.isclass(origin)) and isinstance(
        popped_params, Params
    ):
        # Constructing arbitrary classes from params
        arbitrary_class = origin or annotation
        subextras = create_extras(arbitrary_class, extras)
        constructor_to_inspect = arbitrary_class.__init__
        constructor_to_call = arbitrary_class
        params_contain_step = _params_contain_step(popped_params)
        kwargs = create_kwargs(constructor_to_inspect, arbitrary_class, popped_params, **subextras)
        from tango.step import WithUnresolvedSteps

        if origin != Step and params_contain_step:
            return WithUnresolvedSteps(constructor_to_call, *[], **kwargs)
        else:
            return constructor_to_call(**kwargs)  # type: ignore

    else:
        # Pass it on as is and hope for the best.   ¯\_(ツ)_/¯
        if isinstance(popped_params, Params):
            return popped_params.as_dict()
        return popped_params


class FromParams(CustomDetHash):
    """
    Mixin to give a :meth:`from_params` method to classes. We create a distinct base class for this
    because sometimes we want non :class:`~tango.common.Registrable`
    classes to be instantiatable ``from_params``.
    """

    @classmethod
    def from_params(
        cls: Type[T],
        params_: Union[Params, dict, str],
        constructor_to_call: Callable[..., T] = None,
        constructor_to_inspect: Union[Callable[..., T], Callable[[T], None]] = None,
        **extras,
    ) -> T:
        """
        This is the automatic implementation of ``from_params``. Any class that subclasses
        from ``FromParams`` (or :class:`~tango.common.Registrable`,
        which itself subclasses ``FromParams``) gets this
        implementation for free.  If you want your class to be instantiated from params in the
        "obvious" way -- pop off parameters and hand them to your constructor with the same names --
        this provides that functionality.

        If you need more complex logic in your from ``from_params`` method, you'll have to implement
        your own method that overrides this one.

        The ``constructor_to_call`` and ``constructor_to_inspect`` arguments deal with a bit of
        redirection that we do.  We allow you to register particular ``@classmethods`` on a class as
        the constructor to use for a registered name.  This lets you, e.g., have a single
        ``Vocabulary`` class that can be constructed in two different ways, with different names
        registered to each constructor.  In order to handle this, we need to know not just the class
        we're trying to construct (``cls``), but also what method we should inspect to find its
        arguments (``constructor_to_inspect``), and what method to call when we're done constructing
        arguments (``constructor_to_call``).  These two methods are the same when you've used a
        ``@classmethod`` as your constructor, but they are ``different`` when you use the default
        constructor (because you inspect ``__init__``, but call ``cls()``).
        """

        from tango.common.registrable import (
            Registrable,  # import here to avoid circular imports
        )

        params = params_

        logger.debug(
            f"instantiating class {cls} from params {getattr(params, 'params', params)} "
            f"and extras {set(extras.keys())}"
        )

        if params is None:
            return None

        if isinstance(params, str):
            params = Params({"type": params})

        if not isinstance(params, Params):
            if isinstance(params, dict):
                params = Params(params)
            else:
                raise ConfigurationError(
                    "from_params was passed a `params` object that was not a `Params`. This probably "
                    "indicates malformed parameters in a configuration file, where something that "
                    "should have been a dictionary was actually a list, or something else. "
                    f"This happened when constructing an object of type {cls}."
                )

        if issubclass(cls, Registrable) and not constructor_to_call:
            # We know `cls` inherits from Registrable, so we'll use a cast to make mypy happy.
            as_registrable = cast(Type[Registrable], cls)

            if "type" in params and params["type"] not in as_registrable.list_available():
                as_registrable.search_modules(params["type"])

            if is_base_registrable(cls) or "type" in params:
                default_to_first_choice = as_registrable.default_implementation is not None
                choice = params.pop_choice(
                    "type",
                    choices=as_registrable.list_available(),
                    default_to_first_choice=default_to_first_choice,
                )
                subclass, constructor_name = as_registrable.resolve_class_name(choice)
            else:
                # Must be trying to instantiate the given class directly.
                subclass = cls
                constructor_name = None

            # See the docstring for an explanation of what's going on here.
            if not constructor_name:
                constructor_to_inspect = subclass.__init__
                constructor_to_call = subclass  # type: ignore
            else:
                constructor_to_inspect = cast(Callable[..., T], getattr(subclass, constructor_name))
                constructor_to_call = constructor_to_inspect

            if hasattr(subclass, "from_params"):
                # We want to call subclass.from_params.
                extras = create_extras(subclass, extras)
                # mypy can't follow the typing redirection that we do, so we explicitly cast here.
                retyped_subclass = cast(Type[T], subclass)
                return retyped_subclass.from_params(
                    params,
                    constructor_to_call=constructor_to_call,
                    constructor_to_inspect=constructor_to_inspect,
                    **extras,
                )
            else:
                # In some rare cases, we get a registered subclass that does _not_ have a
                # from_params method (this happens with Activations, for instance, where we
                # register pytorch modules directly).  This is a bit of a hack to make those work,
                # instead of adding a `from_params` method for them somehow.  We just trust that
                # you've done the right thing in passing your parameters, and nothing else needs to
                # be recursively constructed.
                return constructor_to_call(**params, **create_extras(constructor_to_call, extras))  # type: ignore
        else:
            # This is not a base class, so convert our params and extras into a dict of kwargs.

            # See the docstring for an explanation of what's going on here.
            if not constructor_to_inspect:
                constructor_to_inspect = cls.__init__
            if not constructor_to_call:
                constructor_to_call = cls

            if constructor_to_inspect == object.__init__:
                # This class does not have an explicit constructor, so don't give it any kwargs.
                # Without this logic, create_kwargs will look at object.__init__ and see that
                # it takes *args and **kwargs and look for those.
                kwargs: Dict[str, Any] = {}
                params.assert_empty(cls.__name__)
            else:
                # This class has a constructor, so create kwargs for it.
                constructor_to_inspect = cast(Callable[..., T], constructor_to_inspect)
                kwargs = create_kwargs(constructor_to_inspect, cls, params, **extras)

            return constructor_to_call(**kwargs)  # type: ignore

    def to_params(self) -> Params:
        """
        Returns a ``Params`` object that can be used with ``.from_params()`` to recreate an
        object just like it.

        This relies on ``_to_params()``. If you need this in your custom ``FromParams`` class,
        override ``_to_params()``, not this method.
        """

        def replace_object_with_params(o: Any) -> Any:
            if isinstance(o, FromParams):
                return o.to_params().as_dict(quiet=True)
            elif isinstance(o, (list, tuple, set)):
                return [replace_object_with_params(i) for i in o]
            elif isinstance(o, dict):
                return {key: replace_object_with_params(value) for key, value in o.items()}
            elif isinstance(o, Path):
                return str(o)
            elif o is None or isinstance(o, (str, float, int, bool)):
                return o
            else:
                raise NotImplementedError(
                    f"Unexpected type encountered in to_params(): {o} ({type(o)})\n"
                    "You may need to implement a custom '_to_params()'."
                )

        return Params(replace_object_with_params(self._to_params()))

    def _to_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters that, when turned into a ``Params`` object and
        then fed to ``.from_params()``, will recreate this object.

        You don't need to implement this all the time. Tango will let you know if you
        need it.
        """
        try:
            return self.__dict__
        except AttributeError:
            raise NotImplementedError(
                f"{self.__class__.__name__}._to_params() needs to be implemented"
            )

    def det_hash_object(self) -> Any:
        return self.to_params()
