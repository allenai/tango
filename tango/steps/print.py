from typing import Any

from tango.step import Step


@Step.register("print")
class PrintStep(Step):
    """
    This step just prints out its input and also returns what it prints.
    """

    DETERMINISTIC = True
    CACHEABLE = False  # so fast it's not worth caching

    def run(self, input: Any) -> str:  # type: ignore[override]
        """
        Print out the input.
        """
        out = str(input)
        print(out)
        return out
