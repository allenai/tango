import sys
from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pytest

from tango.common.exceptions import ConfigurationError
from tango.common.from_params import (
    FromParams,
    create_kwargs,
    is_base_registrable,
    remove_optional,
    takes_arg,
)
from tango.common.lazy import Lazy
from tango.common.params import Params
from tango.common.registrable import Registrable
from tango.common.testing import TangoTestCase


class TestFromParams(TangoTestCase):
    def test_takes_arg(self):
        def bare_function(some_input: int) -> int:
            return some_input + 1

        assert takes_arg(bare_function, "some_input")
        assert not takes_arg(bare_function, "some_other_input")

        class SomeClass:
            total = 0

            def __init__(self, constructor_param: str) -> None:
                self.constructor_param = constructor_param

            def check_param(self, check: str) -> bool:
                return self.constructor_param == check

            @classmethod
            def set_total(cls, new_total: int) -> None:
                cls.total = new_total

        assert takes_arg(SomeClass, "self")
        assert takes_arg(SomeClass, "constructor_param")
        assert not takes_arg(SomeClass, "check")

        assert takes_arg(SomeClass.check_param, "check")
        assert not takes_arg(SomeClass.check_param, "other_check")

        assert takes_arg(SomeClass.set_total, "new_total")
        assert not takes_arg(SomeClass.set_total, "total")

    def test_remove_optional(self):
        optional_type = Optional[Dict[str, str]]
        bare_type = remove_optional(optional_type)  # type: ignore
        bare_bare_type = remove_optional(bare_type)

        assert bare_type == Dict[str, str]
        assert bare_bare_type == Dict[str, str]

        assert remove_optional(Optional[str]) == str  # type: ignore[arg-type]
        assert remove_optional(str) == str

    @pytest.mark.parametrize("input_type", [dict, Params])
    def test_from_params(self, input_type):
        params = {"my_int": 10}
        my_class = MyClass.from_params(input_type(params), my_bool=True)

        assert isinstance(my_class, MyClass)
        assert my_class.my_int == 10
        assert my_class.my_bool

    def test_create_kwargs(self):
        kwargs = create_kwargs(MyClass, MyClass, Params({"my_int": 5}), my_bool=True, my_float=4.4)

        # my_float should not be included because it's not a param of the MyClass constructor
        assert kwargs == {"my_int": 5, "my_bool": True}

    def test_extras(self):
        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int, name: str) -> None:
                self.size = size
                self.name = name

        @A.register("c")
        class C(A):
            def __init__(self, size: int, name: str) -> None:
                self.size = size
                self.name = name

            # custom from params
            @classmethod
            def from_params(cls, params: Params, size: int, **extras) -> "C":  # type: ignore
                name = params.pop("name")
                return cls(size=size, name=name)

        # Check that extras get passed, even though A doesn't need them.
        params = Params({"type": "b", "size": 10})
        b: B = A.from_params(params, name="extra")  # type: ignore[assignment]

        assert b.name == "extra"
        assert b.size == 10

        # Check that extra extras don't get passed.
        params = Params({"type": "b", "size": 10})
        b = A.from_params(params, name="extra", unwanted=True)  # type: ignore[assignment]

        assert b.name == "extra"  # type: ignore[attr-defined]
        assert b.size == 10  # type: ignore[attr-defined]

        # Now the same with a custom from_params.
        params = Params({"type": "c", "name": "extra_c"})
        c: C = A.from_params(params, size=20)  # type: ignore[assignment]
        assert c.name == "extra_c"
        assert c.size == 20

        # Check that extra extras don't get passed.
        params = Params({"type": "c", "name": "extra_c"})
        c = A.from_params(params, size=20, unwanted=True)  # type: ignore[assignment]

        assert c.name == "extra_c"  # type: ignore[attr-defined]
        assert c.size == 20  # type: ignore[attr-defined]

    def test_extras_for_custom_classes(self):
        class BaseClass(Registrable):
            pass

        class BaseClass2(Registrable):
            pass

        @BaseClass.register("A")
        class A(BaseClass):
            def __init__(self, a: int, b: int, val: str) -> None:
                self.a = a
                self.b = b
                self.val = val

            def __hash__(self):
                return self.b

            def __eq__(self, other):
                return self.b == other.b

            @classmethod
            def from_params(cls, params: Params, a: int, **extras) -> "A":  # type: ignore
                # A custom from params
                b = params.pop_int("b")
                assert b is not None
                val = params.pop("val", "C")
                params.assert_empty(cls.__name__)
                return cls(a=a, b=b, val=val)

        @BaseClass2.register("B")
        class B(BaseClass2):
            def __init__(self, c: int, b: int) -> None:
                self.c = c
                self.b = b

            @classmethod
            def from_params(cls, params: Params, c: int, **extras) -> "B":  # type: ignore
                b = params.pop_int("b")
                assert b is not None
                params.assert_empty(cls.__name__)
                return cls(c=c, b=b)

        @BaseClass.register("E")
        class E(BaseClass):
            def __init__(self, m: int, n: int) -> None:
                self.m = m
                self.n = n

            @classmethod
            def from_params(cls, params: Params, **extras2) -> "E":  # type: ignore
                m = params.pop_int("m")
                assert m is not None
                params.assert_empty(cls.__name__)
                n = extras2["n"]
                return cls(m=m, n=n)

        class C:
            pass

        @BaseClass.register("D")
        class D(BaseClass):
            def __init__(
                self,
                arg1: List[BaseClass],
                arg2: Tuple[BaseClass, BaseClass2],
                arg3: Dict[str, BaseClass],
                arg4: Set[BaseClass],
                arg5: List[BaseClass],
            ) -> None:
                self.arg1 = arg1
                self.arg2 = arg2
                self.arg3 = arg3
                self.arg4 = arg4
                self.arg5 = arg5

        vals = [1, 2, 3]
        params = Params(
            {
                "type": "D",
                "arg1": [
                    {"type": "A", "b": vals[0]},
                    {"type": "A", "b": vals[1]},
                    {"type": "A", "b": vals[2]},
                ],
                "arg2": [{"type": "A", "b": vals[0]}, {"type": "B", "b": vals[0]}],
                "arg3": {
                    "class_1": {"type": "A", "b": vals[0]},
                    "class_2": {"type": "A", "b": vals[1]},
                },
                "arg4": [
                    {"type": "A", "b": vals[0], "val": "M"},
                    {"type": "A", "b": vals[1], "val": "N"},
                    {"type": "A", "b": vals[1], "val": "N"},
                ],
                "arg5": [{"type": "E", "m": 9}],
            }
        )
        extra = C()
        tval1 = 5
        tval2 = 6
        d: D = BaseClass.from_params(  # type: ignore[assignment]
            params, extra=extra, a=tval1, c=tval2, n=10
        )

        # Tests for List # Parameters
        assert len(d.arg1) == len(vals)
        assert isinstance(d.arg1, list)
        assert isinstance(d.arg1[0], A)
        assert all(x.b == y for x, y in zip(d.arg1, vals))  # type: ignore[attr-defined]
        assert all(x.a == tval1 for x in d.arg1)  # type: ignore[attr-defined]

        # Tests for Tuple
        assert isinstance(d.arg2, tuple)
        assert isinstance(d.arg2[0], A)
        assert isinstance(d.arg2[1], B)
        assert d.arg2[0].a == tval1
        assert d.arg2[1].c == tval2
        assert d.arg2[0].b == d.arg2[1].b == vals[0]

        # Tests for Dict
        assert isinstance(d.arg3, dict)
        assert isinstance(d.arg3["class_1"], A)
        assert d.arg3["class_1"].a == d.arg3["class_2"].a == tval1  # type: ignore[attr-defined]
        assert d.arg3["class_1"].b == vals[0]
        assert d.arg3["class_2"].b == vals[1]  # type: ignore[attr-defined]

        # Tests for Set
        assert isinstance(d.arg4, set)
        assert len(d.arg4) == 2
        assert any(x.val == "M" for x in d.arg4)  # type: ignore[attr-defined]
        assert any(x.val == "N" for x in d.arg4)  # type: ignore[attr-defined]

        # Tests for custom extras parameters
        assert isinstance(d.arg5, list)
        assert isinstance(d.arg5[0], E)
        assert d.arg5[0].m == 9
        assert d.arg5[0].n == 10

    def test_union(self):
        class A(FromParams):
            def __init__(self, a: Union[int, List[int]]) -> None:
                self.a = a

        class B(FromParams):
            def __init__(self, b: Union[A, List[A]]) -> None:
                # Really you would want to be sure that `self.b` has a consistent type, but for
                # this test we'll ignore that.
                self.b = b

        params = Params({"a": 3})
        a = A.from_params(params)
        assert a.a == 3

        params = Params({"a": [3, 4, 5]})
        a = A.from_params(params)
        assert a.a == [3, 4, 5]

        params = Params({"b": {"a": 3}})
        b = B.from_params(params)
        assert isinstance(b.b, A)
        assert b.b.a == 3

        params = Params({"b": [{"a": 3}, {"a": [4, 5]}]})
        b = B.from_params(params)
        assert isinstance(b.b, list)
        assert b.b[0].a == 3
        assert b.b[1].a == [4, 5]

    def test_non_params_object_with_params(self):
        bar = Bar.from_params({"foo": Foo(a=1)})
        assert bar.foo.a == 1

    def test_crazy_nested_union(self):
        class A(FromParams):
            def __init__(self, a: Union[int, List[int]]) -> None:
                self.a = a

        class B(FromParams):
            def __init__(self, b: Union[A, List[A]]) -> None:
                # Really you would want to be sure that `self.b` has a consistent type, but for
                # this test we'll ignore that.
                self.b = b

        class C(FromParams):
            def __init__(self, c: Union[A, B, Dict[str, A]]) -> None:
                # Really you would want to be sure that `self.c` has a consistent type, but for
                # this test we'll ignore that.
                self.c = c

        # This is a contrived, ugly example (why would you want to duplicate names in a nested
        # structure like this??), but it demonstrates a potential bug when dealing with mutatable
        # parameters.  If you're not careful about keeping the parameters un-mutated in two
        # separate places, you'll end up with a B, or with a dict that's missing the 'b' key.
        params = Params({"c": {"a": {"a": 3}, "b": {"a": [4, 5]}}})
        c = C.from_params(params)
        assert isinstance(c.c, dict)
        assert c.c["a"].a == 3
        assert c.c["b"].a == [4, 5]

    def test_union_of_castable_types(self):
        class IntFloat(FromParams):
            def __init__(self, a: Union[int, float]) -> None:
                self.a = a

        class FloatInt(FromParams):
            def __init__(self, a: Union[float, int]) -> None:
                self.a = a

        float_param_str = '{"a": 1.0}'
        int_param_str = '{"a": 1}'
        import json

        for expected_type, param_str in [(int, int_param_str), (float, float_param_str)]:
            for cls in [IntFloat, FloatInt]:
                c = cls.from_params(Params(json.loads(param_str)))
                assert type(c.a) == expected_type  # type: ignore[attr-defined]

    def test_invalid_type_conversions(self):
        class A(FromParams):
            def __init__(self, a: int) -> None:
                self.a = a

        with pytest.raises(TypeError):
            A.from_params(Params({"a": "1"}))
        with pytest.raises(TypeError):
            A.from_params(Params({"a": 1.0}))

    def test_dict(self):
        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: Dict[str, A]) -> None:
                self.items = items

        params = Params(
            {
                "type": "d",
                "items": {"first": {"type": "b", "size": 1}, "second": {"type": "b", "size": 2}},
            }
        )
        d: D = C.from_params(params)  # type: ignore[assignment]

        assert isinstance(d.items, dict)
        assert len(d.items) == 2
        assert all(isinstance(key, str) for key in d.items.keys())
        assert all(isinstance(value, B) for value in d.items.values())
        assert d.items["first"].size == 1  # type: ignore[attr-defined]
        assert d.items["second"].size == 2  # type: ignore[attr-defined]

    def test_dict_not_params(self):
        class A(FromParams):
            def __init__(self, counts: Dict[str, int]) -> None:
                self.counts = counts

        params = Params({"counts": {"a": 10, "b": 20}})
        a = A.from_params(params)

        assert isinstance(a.counts, dict)
        assert not isinstance(a.counts, Params)

    def test_list(self):
        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: List[A]) -> None:
                self.items = items

        params = Params(
            {"type": "d", "items": [{"type": "b", "size": 1}, {"type": "b", "size": 2}]}
        )
        d: D = C.from_params(params)  # type: ignore[assignment]

        assert isinstance(d.items, list)
        assert len(d.items) == 2
        assert all(isinstance(item, B) for item in d.items)
        assert d.items[0].size == 1  # type: ignore[attr-defined]
        assert d.items[1].size == 2  # type: ignore[attr-defined]

    def test_tuple(self):
        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, name: str) -> None:
                self.name = name

        class E(Registrable):
            pass

        @E.register("f")
        class F(E):
            def __init__(self, items: Tuple[A, C]) -> None:
                self.items = items

        params = Params(
            {"type": "f", "items": [{"type": "b", "size": 1}, {"type": "d", "name": "item2"}]}
        )
        f: F = E.from_params(params)  # type: ignore[assignment]

        assert isinstance(f.items, tuple)
        assert len(f.items) == 2
        assert isinstance(f.items[0], B)
        assert isinstance(f.items[1], D)
        assert f.items[0].size == 1
        assert f.items[1].name == "item2"

    def test_set(self):
        class A(Registrable):
            def __init__(self, name: str) -> None:
                self.name = name

            def __eq__(self, other):
                return self.name == other.name

            def __hash__(self):
                return hash(self.name)

        @A.register("b")
        class B(A):
            pass

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: Set[A]) -> None:
                self.items = items

        params = Params(
            {
                "type": "d",
                "items": [
                    {"type": "b", "name": "item1"},
                    {"type": "b", "name": "item2"},
                    {"type": "b", "name": "item2"},
                ],
            }
        )
        d: D = C.from_params(params)  # type: ignore[assignment]

        assert isinstance(d.items, set)
        assert len(d.items) == 2
        assert all(isinstance(item, B) for item in d.items)
        assert any(item.name == "item1" for item in d.items)
        assert any(item.name == "item2" for item in d.items)

    def test_kwargs_with_multiple_inheritance(self):
        # Basic idea: have two identical classes, differing only in the order of their multiple
        # inheritance, and make sure that passing kwargs up to the super class works in both cases.
        class A(Registrable):
            def __init__(self, a: int):
                self.a = a

        @A.register("b1")  # type: ignore
        class B1(A, Number):
            def __init__(self, b: float, **kwargs):
                super().__init__(**kwargs)
                self.b = b

        @A.register("b2")  # type: ignore
        class B2(Number, A):
            def __init__(self, b: float, **kwargs):
                super().__init__(**kwargs)
                self.b = b

        b1 = B1.from_params(Params({"a": 4, "b": 5}))
        assert b1.b == 5
        assert b1.a == 4

        b2 = B2.from_params(Params({"a": 4, "b": 5}))
        assert b2.b == 5
        assert b2.a == 4

    def test_instantiating_with_multiple_inheritance(self):
        class A(Registrable):
            def __init__(self, a: int):
                self.a = a

        @A.register("b")  # type: ignore
        class B(A, Number):
            def __init__(self, b: float, **kwargs):
                super().__init__(**kwargs)
                self.b = b

        assert not is_base_registrable(B)

        @B.register("c")
        class C(B):
            def __init__(self, c: float, **kwargs):
                super().__init__(**kwargs)
                self.c = c

        # make sure we can instantiate B directly.
        b = B.from_params({"b": 1.0, "a": 1})
        assert isinstance(b, B)

        # and also make sure we can instantiate subclasses of B.
        c = B.from_params({"type": "c", "c": 2.0, "b": 1.0, "a": 1})
        assert isinstance(c, C)

    def test_only_infer_superclass_params_if_unknown(self):
        class BaseClass(Registrable):
            def __init__(self):
                self.x = None
                self.a = None
                self.rest = None

        @BaseClass.register("a")
        class A(BaseClass):
            def __init__(self, a: int, x: int, **kwargs):
                super().__init__()
                self.x = x
                self.a = a
                self.rest = kwargs

        @BaseClass.register("b")
        class B(A):
            def __init__(self, a: str, x: int = 42, **kwargs):
                super().__init__(x=x, a=-1, raw_a=a, **kwargs)

        params = Params({"type": "b", "a": "123"})
        # The param `x` should not be required as it has default value in `B`
        # The correct type of the param `a` should be inferred from `B` as well.
        instance = BaseClass.from_params(params)
        assert instance.x == 42
        assert instance.a == -1
        assert len(instance.rest) == 1  # type: ignore
        assert type(instance.rest["raw_a"]) == str  # type: ignore
        assert instance.rest["raw_a"] == "123"  # type: ignore

    def test_kwargs_are_passed_to_deeper_superclasses(self):
        class BaseClass(Registrable):
            def __init__(self):
                self.a = None
                self.b = None
                self.c = None

        @BaseClass.register("a")
        class A(BaseClass):
            def __init__(self, a: str):
                super().__init__()
                self.a = a

        @BaseClass.register("b")
        class B(A):
            def __init__(self, b: str, **kwargs):
                super().__init__(**kwargs)
                self.b = b

        @BaseClass.register("c")
        class C(B):
            def __init__(self, c, **kwargs):
                super().__init__(**kwargs)
                self.c = c

        params = Params({"type": "c", "a": "a_value", "b": "b_value", "c": "c_value"})

        instance = BaseClass.from_params(params)
        assert instance.a == "a_value"
        assert instance.b == "b_value"
        assert instance.c == "c_value"

    def test_lazy_construction_can_happen_multiple_times(self):
        test_string = "this is a test"
        extra_string = "extra string"

        class ConstructedObject(FromParams):
            def __init__(self, string: str, extra: str):
                self.string = string
                self.extra = extra

        class Testing(FromParams):
            def __init__(self, lazy_object: Lazy[ConstructedObject]):
                first_time = lazy_object.construct(extra=extra_string)
                second_time = lazy_object.construct(extra=extra_string)
                assert first_time.string == test_string
                assert first_time.extra == extra_string
                assert second_time.string == test_string
                assert second_time.extra == extra_string

        Testing.from_params(Params({"lazy_object": {"string": test_string}}))

    def test_lazy_and_from_params_can_be_pickled(self):

        import pickle

        baz = Baz.from_params(Params({"bar": {"foo": {"a": 2}}}))
        pickle.dumps(baz)

    def test_optional_vs_required_lazy_objects(self):
        class ConstructedObject(FromParams):
            def __init__(self, a: int):
                self.a = a

        class Testing(FromParams):
            def __init__(
                self,
                lazy1: Lazy[ConstructedObject],
                lazy2: Lazy[ConstructedObject] = Lazy(ConstructedObject),
                lazy3: Lazy[ConstructedObject] = None,
                lazy4: Optional[Lazy[ConstructedObject]] = Lazy(ConstructedObject),
            ) -> None:
                self.lazy1 = lazy1.construct()
                self.lazy2 = lazy2.construct(a=2)
                self.lazy3 = None if lazy3 is None else lazy3.construct()
                self.lazy4 = None if lazy4 is None else lazy4.construct(a=1)

        test1 = Testing.from_params(Params({"lazy1": {"a": 1}}))
        assert test1.lazy1.a == 1
        assert test1.lazy2.a == 2
        assert test1.lazy3 is None
        assert test1.lazy4 is not None

        test2 = Testing.from_params(Params({"lazy1": {"a": 1}, "lazy2": {"a": 3}}))
        assert test2.lazy1.a == 1
        assert test2.lazy2.a == 3
        assert test2.lazy3 is None
        assert test2.lazy4 is not None

        test3 = Testing.from_params(Params({"lazy1": {"a": 1}, "lazy3": {"a": 3}, "lazy4": None}))
        assert test3.lazy1.a == 1
        assert test3.lazy2.a == 2
        assert test3.lazy3 is not None
        assert test3.lazy3.a == 3
        assert test3.lazy4 is None

        with pytest.raises(ConfigurationError, match='key "lazy1" is required'):
            Testing.from_params(Params({}))

    def test_wrapper_kwargs_passed_down(self):
        class BaseObject:
            def __init__(self, x: int = 1):
                self.x = x

        class BaseWrapper(BaseObject, FromParams):
            def __init__(self, y: int = 2, **kwargs):
                super().__init__(**kwargs)
                self.y = y

        o = BaseWrapper.from_params(Params({"y": 3}), x=2)
        assert o.x == 2

    def test_iterable(self):
        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: Iterable[A]) -> None:
                self.items = items

        params = Params(
            {"type": "d", "items": [{"type": "b", "size": 1}, {"type": "b", "size": 2}]}
        )
        d: D = C.from_params(params)  # type: ignore[assignment]

        assert isinstance(d.items, Iterable)
        items = list(d.items)
        assert len(items) == 2
        assert all(isinstance(item, B) for item in items)
        assert items[0].size == 1  # type: ignore
        assert items[1].size == 2  # type: ignore

    def test_mapping(self):
        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: Mapping[str, A]) -> None:
                self.items = items

        params = Params(
            {
                "type": "d",
                "items": {"first": {"type": "b", "size": 1}, "second": {"type": "b", "size": 2}},
            }
        )
        d: D = C.from_params(params)  # type: ignore[assignment]

        assert isinstance(d.items, Mapping)
        assert len(d.items) == 2
        assert all(isinstance(key, str) for key in d.items.keys())
        assert all(isinstance(value, B) for value in d.items.values())
        assert d.items["first"].size == 1  # type: ignore
        assert d.items["second"].size == 2  # type: ignore

    def test_custom_abc_mapping(self):
        from collections import abc

        class CustomMapping(abc.Mapping):
            def __init__(self, data: Dict[str, int]):
                self.data = data

            def __getitem__(self, key):
                return self.data[key]

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        class ClassWithCustomMapping(FromParams):
            def __init__(self, mapping: CustomMapping):
                self.mapping = mapping

        o = ClassWithCustomMapping.from_params({"mapping": {"data": {"a": 1}}})
        assert isinstance(o.mapping, CustomMapping)
        assert o.mapping["a"] == 1

    def test_extra_parameters_are_not_allowed_when_there_is_no_constructor(self):
        class A(FromParams):
            pass

        with pytest.raises(ConfigurationError, match="Extra parameters"):
            A.from_params(Params({"some_spurious": "key", "value": "pairs"}))

    def test_explicit_kwargs_always_passed_to_constructor(self):
        class Base(FromParams):
            def __init__(self, lazy: bool = False, x: int = 0) -> None:
                self.lazy = lazy
                self.x = x

        class A(Base):
            def __init__(self, **kwargs) -> None:
                assert "lazy" in kwargs
                super().__init__(**kwargs)

        A.from_params(Params({"lazy": False}))

        class B(Base):
            def __init__(self, **kwargs) -> None:
                super().__init__(lazy=True, **kwargs)

        b = B.from_params(Params({}))
        assert b.lazy is True

    def test_raises_when_there_are_no_implementations(self):
        class A(Registrable):
            pass

        with pytest.raises(ConfigurationError, match="not in acceptable choices for type"):
            A.from_params("nonexistent_class")

        with pytest.raises(ConfigurationError, match='key "type" is required'):
            A.from_params(Params({"some_spurious": "key", "value": "pairs"}))

        with pytest.raises(ConfigurationError, match='key "type" is required'):
            A.from_params(Params({}))

        # Some paths through the code are different if there is a constructor here versus not.  We
        # don't actually go through this logic anymore, but it's here as a regression test.
        class B(Registrable):
            def __init__(self):
                pass

        with pytest.raises(ConfigurationError, match="not in acceptable choices for type"):
            B.from_params("nonexistent_class")

        with pytest.raises(ConfigurationError, match='key "type" is required'):
            B.from_params(Params({"some_spurious": "key", "value": "pairs"}))

        with pytest.raises(ConfigurationError, match='key "type" is required'):
            B.from_params(Params({}))

    def test_from_params_raises_error_on_wrong_parameter_name_in_optional_union(self):
        class NestedClass(FromParams):
            def __init__(self, varname: Optional[str] = None):
                self.varname = varname

        class WrapperClass(FromParams):
            def __init__(self, nested_class: Optional[Union[str, NestedClass]] = None):
                if isinstance(nested_class, str):
                    nested_class = NestedClass(varname=nested_class)
                self.nested_class = nested_class

        with pytest.raises(ConfigurationError):
            WrapperClass.from_params(Params({"nested_class": {"wrong_varname": "varstring"}}))

    def test_from_params_handles_base_class_kwargs(self):
        class Foo(FromParams):
            def __init__(self, a: int, b: str = None, **kwargs) -> None:
                self.a = a
                self.b = b
                for key, value in kwargs.items():
                    setattr(self, key, value)

        foo = Foo.from_params(Params({"a": 2, "b": "hi"}))
        assert foo.a == 2
        assert foo.b == "hi"

        foo = Foo.from_params(Params({"a": 2, "b": "hi", "c": {"2": "3"}}))
        assert foo.a == 2
        assert foo.b == "hi"
        assert foo.c == {"2": "3"}  # type: ignore[attr-defined]

        class Bar(Foo):
            def __init__(self, a: int, b: str, d: int, **kwargs) -> None:
                super().__init__(a, b=b, **kwargs)
                self.d = d

        bar = Bar.from_params(Params({"a": 2, "b": "hi", "c": {"2": "3"}, "d": 0}))
        assert bar.a == 2
        assert bar.b == "hi"
        assert bar.c == {"2": "3"}  # type: ignore[attr-defined]
        assert bar.d == 0

        class Baz(Foo):
            def __init__(self, a: int, b: Optional[str] = "a", **kwargs) -> None:
                super().__init__(a, b=b, **kwargs)

        baz = Baz.from_params(Params({"a": 2, "b": None}))
        assert baz.b is None

        baz = Baz.from_params(Params({"a": 2}))
        assert baz.b == "a"

    def test_from_params_base_class_kwargs_crashes_if_params_not_handled(self):
        class Bar(FromParams):
            def __init__(self, c: str = None) -> None:
                self.c = c

        class Foo(Bar):
            def __init__(self, a: int, b: str = None, **kwargs) -> None:
                super().__init__(**kwargs)
                self.a = a
                self.b = b

        foo = Foo.from_params(Params({"a": 2, "b": "hi", "c": "some value"}))
        assert foo.a == 2
        assert foo.b == "hi"
        assert foo.c == "some value"

        with pytest.raises(TypeError, match="invalid_key"):
            Foo.from_params(Params({"a": 2, "b": "hi", "invalid_key": "some value"}))

    def test_from_params_handles_kwargs_in_non_from_params_registered_class(self):
        class Bar(Registrable):
            pass

        class Baz:
            def __init__(self, a: int) -> None:
                self.a = a

        @Bar.register("foo")
        class Foo(Baz):
            def __init__(self, a: int, b: str = None, **kwargs) -> None:
                super().__init__(a)
                self.b = b
                for key, value in kwargs.items():
                    setattr(self, key, value)

        foo: Foo = Bar.from_params(Params({"type": "foo", "a": 2, "b": "hi"}))  # type: ignore[assignment]
        assert foo.a == 2
        assert foo.b == "hi"

        foo = Bar.from_params(  # type: ignore[assignment]
            Params({"type": "foo", "a": 2, "b": "hi", "c": {"2": "3"}})
        )
        assert foo.a == 2  # type: ignore[attr-defined]
        assert foo.b == "hi"  # type: ignore[attr-defined]
        assert foo.c == {"2": "3"}  # type: ignore[attr-defined]

    def test_from_params_passes_extras_to_non_from_params_registered_class(self):
        class Bar(Registrable):
            pass

        class Baz:
            def __init__(self, a: int, c: Dict[str, str] = None, extra: str = "idk") -> None:
                self.a = a
                self.c = c
                self.extra = extra

        @Bar.register("foo")
        class Foo(Baz):
            def __init__(self, a: int, b: str = None, **kwargs) -> None:
                super().__init__(a, **kwargs)
                self.b = b

        foo: Foo = Bar.from_params(Params({"type": "foo", "a": 2, "b": "hi"}))  # type: ignore[assignment]
        assert foo.a == 2
        assert foo.b == "hi"
        assert foo.c is None

        foo = Bar.from_params(  # type: ignore[assignment]
            Params({"type": "foo", "a": 2, "b": "hi", "c": {"2": "3"}}), extra="4"
        )
        assert foo.a == 2  # type: ignore[attr-defined]
        assert foo.b == "hi"  # type: ignore[attr-defined]
        assert foo.c == {"2": "3"}  # type: ignore[attr-defined]
        assert foo.extra == "4"  # type: ignore[attr-defined]

    def test_from_params_child_has_kwargs_base_implicit_constructor(self):
        class Foo(FromParams):
            pass

        class Bar(Foo):
            def __init__(self, a: int, **kwargs) -> None:
                self.a = a

        bar = Bar.from_params(Params({"a": 2}))
        assert bar.a == 2

    def test_from_params_has_args(self):
        class Foo(FromParams):
            def __init__(self, a: int, *args) -> None:
                self.a = a

        foo = Foo.from_params(Params({"a": 2}))
        assert foo.a == 2

    def test_from_params_with_dataclass(self):
        @dataclass
        class Foo(FromParams):
            x: int
            y: str

        assert Foo.from_params({"x": 1, "y": "2"}).x == 1
        with pytest.raises(TypeError):
            Foo.from_params({"x": 1, "y": 2})

    def test_to_params(self):
        @dataclass
        class Bar(FromParams):
            z: bool

        @dataclass
        class Foo(FromParams):
            x: int
            bar: Bar

        params_dict = {"x": 1, "bar": {"z": True}}
        foo = Foo.from_params(deepcopy(params_dict))
        assert foo.bar.z
        params = foo.to_params()
        assert params.as_dict() == params_dict

    def test_to_params_needs_custom_to_params(self):
        @dataclass
        class Bar:
            z: bool

        @dataclass
        class Foo(FromParams):
            x: int
            bar: Bar

        foo = Foo.from_params({"x": 1}, bar=Bar(z=True))
        with pytest.raises(NotImplementedError):
            foo.to_params()

    @pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python 3.9 or higher")
    def test_type_hinting_generics_from_std_collections(self):
        class Item(FromParams):
            def __init__(self, a: int) -> None:
                self.a = a

        class ClassWithStdGenerics(FromParams):
            def __init__(self, x: list[Item], y: dict[str, Item]) -> None:
                self.x = x
                self.y = y

        o = ClassWithStdGenerics.from_params({"x": [{"a": 1}], "y": {"b": {"a": 1}}})
        assert isinstance(o.x, list)
        assert isinstance(o.x[0], Item)
        assert isinstance(o.y["b"], Item)

    def test_with_non_from_params_generics(self):
        T = TypeVar("T")

        class Item(Generic[T]):
            def __init__(self, x: T):
                self.x = x

        class ClassWithGenerics(FromParams):
            def __init__(self, item: Item[T]):
                self.item = item

        o = ClassWithGenerics.from_params({"item": {"x": 1}})
        assert isinstance(o.item, Item)

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python 3.10 or higher")
    def test_with_union_pipe(self):
        class Item(FromParams):
            def __init__(self, a: int) -> None:
                self.a = a

        class ClassWithUnionType(FromParams):
            def __init__(self, x: Item | str):
                self.x = x

        o = ClassWithUnionType.from_params({"x": {"a": 1}})
        assert isinstance(o.x, Item)


class MyClass(FromParams):
    def __init__(self, my_int: int, my_bool: bool = False) -> None:
        self.my_int = my_int
        self.my_bool = my_bool


class Foo(FromParams):
    def __init__(self, a: int = 1) -> None:
        self.a = a


class Bar(FromParams):
    def __init__(self, foo: Foo) -> None:
        self.foo = foo


class Baz(FromParams):
    def __init__(self, bar: Lazy[Bar]) -> None:
        self._bar = bar

    @property
    def bar(self):
        return self._bar.construct()
