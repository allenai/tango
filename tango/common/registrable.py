"""
:class:`Registrable` is a "mixin" for endowing
any base class with a named registry for its subclasses and a decorator
for registering them.
"""

import importlib
import logging
from collections import defaultdict
from typing import (
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from .exceptions import ConfigurationError, RegistryKeyError
from .from_params import FromParams
from .util import (
    could_be_class_name,
    find_integrations,
    find_submodules,
    import_module_and_submodules,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_RegistrableT = TypeVar("_RegistrableT", bound="Registrable")

_SubclassRegistry = Dict[str, Tuple[type, Optional[str]]]


class Registrable(FromParams):
    """
    Any class that inherits from ``Registrable`` gains access to a named registry for its
    subclasses. To register them, just decorate them with the classmethod
    ``@BaseClass.register(name)``.

    After which you can call ``BaseClass.list_available()`` to get the keys for the
    registered subclasses, and ``BaseClass.by_name(name)`` to get the corresponding subclass.
    Note that the registry stores the subclasses themselves; not class instances.
    In most cases you would then call :meth:`~tango.common.from_params.FromParams.from_params()`
    on the returned subclass.

    You can specify a default by setting ``BaseClass.default_implementation``.
    If it is set, it will be the first element of :meth:`list_available()`.

    Note that if you use this class to implement a new ``Registrable`` abstract class,
    you must ensure that all subclasses of the abstract class are loaded when the module is
    loaded, because the subclasses register themselves in their respective files. You can
    achieve this by having the abstract class and all subclasses in the ``__init__.py`` of the
    module in which they reside (as this causes any import of either the abstract class or
    a subclass to load all other subclasses and the abstract class).
    """

    _registry: ClassVar[DefaultDict[type, _SubclassRegistry]] = defaultdict(dict)

    default_implementation: Optional[str] = None

    @classmethod
    def register(
        cls, name: str, constructor: Optional[str] = None, exist_ok: bool = False
    ) -> Callable[[Type[_T]], Type[_T]]:
        """
        Register a class under a particular name.

        Parameters
        ----------
        name :
            The name to register the class under.
        constructor :
            The name of the method to use on the class to construct the object.  If this is given,
            we will use this method (which must be a ``@classmethod``) instead of the default
            constructor.
        exist_ok :
            If True, overwrites any existing models registered under ``name``. Else,
            throws an error if a model is already registered under ``name``.

        Examples
        --------

        To use this class, you would typically have a base class that inherits from ``Registrable``::

            class Vocabulary(Registrable):
                ...

        Then, if you want to register a subclass, you decorate it like this::

            @Vocabulary.register("my-vocabulary")
            class MyVocabulary(Vocabulary):
                def __init__(self, param1: int, param2: str):
                    ...

        Registering a class like this will let you instantiate a class from a config file, where you
        give ``"type": "my-vocabulary"``, and keys corresponding to the parameters of the ``__init__``
        method (note that for this to work, those parameters must have type annotations).

        If you want to have the instantiation from a config file call a method other than the
        constructor, either because you have several different construction paths that could be
        taken for the same object (as we do in ``Vocabulary``) or because you have logic you want to
        happen before you get to the constructor (as we do in ``Embedding``), you can register a
        specific ``@classmethod`` as the constructor to use, like this::

            @Vocabulary.register("my-vocabulary-from-instances", constructor="from_instances")
            @Vocabulary.register("my-vocabulary-from-files", constructor="from_files")
            class MyVocabulary(Vocabulary):
                def __init__(self, some_params):
                    ...

                @classmethod
                def from_instances(cls, some_other_params) -> MyVocabulary:
                    ...  # construct some_params from instances
                    return cls(some_params)

                @classmethod
                def from_files(cls, still_other_params) -> MyVocabulary:
                    ...  # construct some_params from files
                    return cls(some_params)
        """
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[_T]) -> Type[_T]:
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                if exist_ok:
                    message = (
                        f"{name} has already been registered as {registry[name][0].__name__}, but "
                        f"exist_ok=True, so overwriting with {cls.__name__}"
                    )
                    logger.info(message)
                else:
                    message = (
                        f"Cannot register {name} as {cls.__name__}; "
                        f"name already in use for {registry[name][0].__name__}"
                    )
                    raise ConfigurationError(message)
            registry[name] = (subclass, constructor)
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[_RegistrableT], name: str) -> Callable[..., _RegistrableT]:
        """
        Returns a callable function that constructs an argument of the registered class.  Because
        you can register particular functions as constructors for specific names, this isn't
        necessarily the ``__init__`` method of some class.
        """
        logger.debug(f"instantiating registered subclass {name} of {cls}")
        subclass, constructor = cls.resolve_class_name(name)
        if not constructor:
            return cast(Type[_RegistrableT], subclass)
        else:
            return cast(Callable[..., _RegistrableT], getattr(subclass, constructor))

    @classmethod
    def search_modules(cls: Type[_RegistrableT], name: str):
        """
        Search for and import modules where ``name`` might be registered.
        """
        if could_be_class_name(name) or name in cls.list_available():
            return None

        def try_import(module):
            try:
                import_module_and_submodules(module)
            except (ModuleNotFoundError, ImportError):
                pass

        integrations = {m.split(".")[-1]: m for m in find_integrations()}
        integrations_imported: Set[str] = set()
        if name in integrations:
            try_import(integrations[name])
            integrations_imported.add(name)
            if name in cls.list_available():
                return None

        if "::" in name:
            maybe_integration = name.split("::")[0]
            if maybe_integration in integrations:
                try_import(integrations[maybe_integration])
                integrations_imported.add(maybe_integration)
                if name in cls.list_available():
                    return None

        for module in find_submodules(exclude={"tango.integrations*"}, recursive=False):
            try_import(module)
            if name in cls.list_available():
                return None

        # If we still haven't found the registered 'name', try importing all other integrations.
        for integration_name, module in integrations.items():
            if integration_name not in integrations_imported:
                try_import(module)
                integrations_imported.add(integration_name)
                if name in cls.list_available():
                    return None

    @classmethod
    def resolve_class_name(
        cls: Type[_RegistrableT],
        name: str,
        search_modules: bool = True,
    ) -> Tuple[Type[_RegistrableT], Optional[str]]:
        """
        Returns the subclass that corresponds to the given ``name``, along with the name of the
        method that was registered as a constructor for that ``name``, if any.

        This method also allows ``name`` to be a fully-specified module name, instead of a name that
        was already added to the ``Registry``.  In that case, you cannot use a separate function as
        a constructor (as you need to call ``cls.register()`` in order to tell us what separate
        function to use).

        If the ``name`` given is not in the registry and ``search_modules`` is ``True``,
        it will search for and import modules where the class might be defined according to
        :meth:`search_modules()`.
        """
        if name in Registrable._registry[cls]:
            subclass, constructor = Registrable._registry[cls][name]
            return subclass, constructor
        elif could_be_class_name(name):
            # This might be a fully qualified class name, so we'll try importing its "module"
            # and finding it there.
            parts = name.split(".")
            submodule = ".".join(parts[:-1])
            class_name = parts[-1]

            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError:
                raise ConfigurationError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to import module {submodule}"
                )

            try:
                subclass = getattr(module, class_name)
                constructor = None
                return subclass, constructor
            except AttributeError:
                raise ConfigurationError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to find class {class_name} in {submodule}"
                )
        else:
            # is not a qualified class name
            if search_modules:
                cls.search_modules(name)
                return cls.resolve_class_name(name, search_modules=False)

            available = cls.list_available()
            suggestion = _get_suggestion(name, available)
            raise RegistryKeyError(
                (
                    f"'{name}' is not a registered name for '{cls.__name__}'"
                    + (". " if not suggestion else f", did you mean '{suggestion}'? ")
                )
                + "If your registered class comes from custom code, you'll need to import "
                "the corresponding modules. If you're using Tango or AllenNLP from the command-line, "
                "this is done by using the '--include-package' flag, or by specifying your imports "
                "in a '.allennlp_plugins' file. "
                "Alternatively, you can specify your choices "
                """using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} """
                "in which case they will be automatically imported correctly."
            )

    @classmethod
    def list_available(cls) -> List[str]:
        """List default first if it exists"""
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            raise ConfigurationError(f"Default implementation {default} is not registered")
        else:
            return [default] + [k for k in keys if k != default]


def _get_suggestion(name: str, available: List[str]) -> Optional[str]:
    # Check for simple mistakes like using '-' instead of '_', or vice-versa.
    for ch, repl_ch in (("_", "-"), ("-", "_")):
        suggestion = name.replace(ch, repl_ch)
        if suggestion in available:
            return suggestion
    return None
