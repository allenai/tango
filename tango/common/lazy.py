import copy
import inspect
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, cast

from .det_hash import CustomDetHash, DetHashWithVersion
from .params import Params

T = TypeVar("T")


class Lazy(Generic[T], CustomDetHash):
    """
    This class is for use when constructing objects using :class:`~tango.common.FromParams`,
    when an argument to a constructor has a `sequential dependency` with another argument to the same
    constructor.

    For example, in a ``Trainer`` class you might want to take a ``Model`` and an ``Optimizer`` as arguments,
    but the ``Optimizer`` needs to be constructed using the parameters from the ``Model``. You can give
    the type annotation ``Lazy[Optimizer]`` to the optimizer argument, then inside the constructor
    call ``optimizer.construct(parameters=model.parameters)``.

    This is only recommended for use when you have registered a ``@classmethod`` as the constructor
    for your class, instead of using ``__init__``.  Having a ``Lazy[]`` type annotation on an argument
    to an ``__init__`` method makes your class completely dependent on being constructed using the
    ``FromParams`` pipeline, which is not a good idea.

    The actual implementation here is incredibly simple; the logic that handles the lazy
    construction is actually found in ``FromParams``, where we have a special case for a ``Lazy`` type
    annotation.

    Examples
    --------

    ::

        @classmethod
        def my_constructor(
            cls,
            some_object: Lazy[MyObject],
            optional_object: Lazy[MyObject] = None,
            # or:
            #  optional_object: Optional[Lazy[MyObject]] = None,
            optional_object_with_default: Optional[Lazy[MyObject]] = Lazy(MyObjectDefault),
            required_object_with_default: Lazy[MyObject] = Lazy(MyObjectDefault),
        ) -> MyClass:
            obj1 = some_object.construct()
            obj2 = None if optional_object is None else optional_object.construct()
            obj3 = None optional_object_with_default is None else optional_object_with_default.construct()
            obj4 = required_object_with_default.construct()

    """

    def __init__(
        self,
        constructor: Union[Type[T], Callable[..., T]],
        params: Optional[Params] = None,
        constructor_extras: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self._constructor = constructor
        self._params = params or Params({})
        self._constructor_extras = constructor_extras or {}
        self._constructor_extras.update(kwargs)

    @property
    def constructor(self) -> Callable[..., T]:
        from tango.common.from_params import FromParams

        if inspect.isclass(self._constructor) and issubclass(self._constructor, FromParams):

            def constructor_to_use(**kwargs):
                return self._constructor.from_params(  # type: ignore[union-attr]
                    copy.deepcopy(self._params),
                    **kwargs,
                )

            return constructor_to_use
        else:
            return self._constructor

    def construct(self, **kwargs) -> T:
        """
        Call the constructor to create an instance of ``T``.
        """
        # If there are duplicate keys between self._constructor_extras and kwargs,
        # this will overwrite the ones in self._constructor_extras with what's in kwargs.
        constructor_kwargs = {**self._constructor_extras, **kwargs}
        return self.constructor(**constructor_kwargs)

    def det_hash_object(self) -> Any:
        from tango.common.from_params import FromParams

        class_to_construct: Union[Type[T], Callable[..., T]] = self._constructor
        if isinstance(class_to_construct, type) and issubclass(class_to_construct, FromParams):
            params = copy.deepcopy(self._params)
            if params is None:
                params = Params({})
            elif isinstance(params, str):
                params = Params({"type": params})
            elif isinstance(params, dict):
                params = Params(params)
            elif not isinstance(params, Params):
                return None

            from tango.common import Registrable

            if issubclass(class_to_construct, Registrable):
                as_registrable = cast(Type[Registrable], class_to_construct)

                if "type" in params and params["type"] not in as_registrable.list_available():
                    as_registrable.search_modules(params["type"])

                # Resolve the subclass and constructor.
                from .from_params import is_base_registrable

                if is_base_registrable(class_to_construct) or "type" in params:
                    default_to_first_choice = as_registrable.default_implementation is not None
                    choice = params.pop_choice(
                        "type",
                        choices=as_registrable.list_available(),
                        default_to_first_choice=default_to_first_choice,
                    )
                    subclass_or_factory_func, _ = as_registrable.resolve_class_name(choice)
                    if inspect.isclass(subclass_or_factory_func):
                        class_to_construct = subclass_or_factory_func
                    else:
                        # We have a function that returns an instance of the class.
                        factory_func = cast(Callable[..., T], subclass_or_factory_func)
                        return_type = inspect.signature(factory_func).return_annotation
                        if return_type != inspect.Signature.empty:
                            class_to_construct = return_type

        if isinstance(class_to_construct, type) and issubclass(
            class_to_construct, DetHashWithVersion
        ):
            return class_to_construct.VERSION, self
        else:
            return self
