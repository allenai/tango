from string import ascii_letters
import random

from tango import Step


@Step.register("float")
class FloatStep(Step):
    CACHEABLE = True
    DETERMINISTIC = True

    def run(self, result: float) -> float:  # type: ignore
        return result


@Step.register("string")
class StringStep(Step):
    CACHEABLE = True
    DETERMINISTIC = True

    def run(self, result: str) -> str:  # type: ignore
        return result


@Step.register("concat_strings")
class ConcatStringsStep(Step):
    CACHEABLE = True
    DETERMINISTIC = True

    def run(self, string1: str, string2: str, join_with: str = " ") -> str:  # type: ignore
        return join_with.join([string1, string2])


@Step.register("random_string")
class RandomStringStep(Step):
    def run(self, length: int = 10) -> str:  # type: ignore
        return "".join([random.choice(ascii_letters) for _ in range(length)])
