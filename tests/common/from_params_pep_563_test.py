"""
This tests `FromParams` functionality with https://www.python.org/dev/peps/pep-0563/.
"""

from __future__ import annotations

from tango.common.from_params import FromParams, infer_method_params
from tango.common.lazy import Lazy


class Foo(FromParams):
    def __init__(self, x: int):
        self.x = x


class Bar(FromParams):
    def __init__(self, foo: Lazy[Foo]):
        self.foo = foo.construct()


class Baz(FromParams):
    def __init__(self, bar: Lazy[Bar]):
        self.bar = bar.construct()


def test_infer_method_params():
    parameters = infer_method_params(Baz, Baz.__init__)
    assert not isinstance(parameters["bar"].annotation, str)


def test_from_params():
    baz = Baz.from_params({"bar": {"foo": {"x": 1}}})
    assert baz.bar.foo.x == 1
