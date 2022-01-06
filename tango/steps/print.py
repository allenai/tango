import logging
from typing import Any

from tango.common.logging import click_logger
from tango.step import Step


@Step.register("print")
class PrintStep(Step):
    """
    This step just logs or prints its input and also returns what it prints.
    """

    DETERMINISTIC = True
    CACHEABLE = False  # so fast it's not worth caching

    def run(self, input: Any) -> str:  # type: ignore[override]
        """
        Print out the input.
        """
        out = str(input)
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(out)
        elif click_logger.isEnabledFor(logging.INFO):
            click_logger.info(out)
        else:
            print(out)
        return out
