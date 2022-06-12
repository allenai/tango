from typing import Union

from tango import Step

IntOrFloat = Union[int, float]

@Step.register("addition")
class AdditionStep(Step):
    def run(self, num1: int, num2: int) -> int:
        return num1 + num2

@Step.register("scale_up")
class ScaleUp(Step):
    def run(self, num1: int, factor: int) -> int:
        return num1 * factor

@Step.register("scale_down")
class ScaleDown(Step):
    def run(self, num1: int, factor: int) -> IntOrFloat:
        return num1 / factor

@Step.register("print")
class Print(Step):
    def run(self, num: IntOrFloat) -> None:
        print(num)