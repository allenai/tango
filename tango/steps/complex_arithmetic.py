import cmath
from typing import Tuple, Union

from tango import Step

ComplexOrTuple = Union[complex, Tuple[float, float]]


def make_complex(x: ComplexOrTuple) -> complex:
    if isinstance(x, complex):
        return x
    elif isinstance(x, (int, float)):
        return complex(x)
    else:
        return complex(*x)


@Step.register("cadd")
class AdditionStep(Step):
    def run(self, a: ComplexOrTuple, b: ComplexOrTuple) -> complex:  # type: ignore
        return make_complex(a) + make_complex(b)


@Step.register("csub")
class SubtractionStep(Step):
    def run(self, a: ComplexOrTuple, b: ComplexOrTuple) -> complex:  # type: ignore
        return make_complex(a) - make_complex(b)


@Step.register("cexp")
class ExponentiateStep(Step):
    def run(self, x: ComplexOrTuple, base: ComplexOrTuple = cmath.e) -> complex:  # type: ignore
        return make_complex(base) ** make_complex(x)


@Step.register("cmul")
class MultiplyStep(Step):
    def run(self, a: ComplexOrTuple, b: ComplexOrTuple) -> complex:  # type: ignore
        return make_complex(a) * make_complex(b)


@Step.register("csin")
class SineStep(Step):
    def run(self, x: ComplexOrTuple) -> complex:  # type: ignore
        return cmath.sin(make_complex(x))


@Step.register("ccos")
class CosineStep(Step):
    def run(self, x: ComplexOrTuple) -> complex:  # type: ignore
        return cmath.cos(make_complex(x))
