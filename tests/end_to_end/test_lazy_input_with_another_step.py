from dataclasses import dataclass

from tango import Format, JsonFormat, Step
from tango.common import FromParams, Lazy
from tango.common.testing import run_experiment


@dataclass
class Foo(FromParams):
    number: float


@Step.register("generate_number")
class GenerateNumberStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()

    def run(self) -> float:  # type: ignore[override]
        return 1.0


@Step.register("lazy_input")
class StepWithLazyInput(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()

    def run(self, foo: Lazy[Foo]) -> float:  # type: ignore[override]
        foo = foo.construct()
        assert isinstance(foo, Foo)
        assert isinstance(foo.number, float)
        return foo.number


def test_experiment():
    with run_experiment(
        {
            "steps": {
                "gen_number": {
                    "type": "generate_number",
                },
                "get_number": {
                    "type": "lazy_input",
                    "foo": {
                        "number": {
                            "type": "ref",
                            "ref": "gen_number",
                        }
                    },
                },
            }
        }
    ) as run_dir:
        assert (run_dir / "get_number").is_dir()
        fmt: Format = JsonFormat()
        data = fmt.read(run_dir / "get_number")
        assert data == 1.0
