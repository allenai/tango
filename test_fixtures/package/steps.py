import logging
import multiprocessing as mp
import random
import time
from string import ascii_letters

import tango.common.logging as common_logging
from tango import Step
from tango.common import Tqdm


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


@Step.register("noisy_step")
class NoisyStep(Step):
    CACHEABLE = True
    DETERMINISTIC = True

    def run(self, raise_error: bool = False) -> None:  # type: ignore
        self.logger.debug("debug message")
        common_logging.cli_logger.debug("debug message from cli_logger")

        self.logger.info("info message")
        common_logging.cli_logger.info("info message from cli_logger")

        self.logger.warning("warning message")
        common_logging.cli_logger.warning("warning message from cli_logger")

        self.logger.error("error message")
        common_logging.cli_logger.error("error message from cli_logger")

        if raise_error:
            raise ValueError("Oh no!")


@Step.register("random_string")
class RandomStringStep(Step):
    def run(self, length: int = 10) -> str:  # type: ignore
        return "".join([random.choice(ascii_letters) for _ in range(length)])


@Step.register("add_numbers")
class AddNumbersStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a_number: int, b_number: int) -> int:  # type: ignore
        return a_number + b_number


@Step.register("sleep-print-maybe-fail")
class SleepPrintMaybeFail(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, string: str, seconds: int = 5, fail: bool = False) -> str:  # type: ignore
        time.sleep(seconds)
        self.logger.info(f"Step {self.name} is awake.")
        print(string)
        if fail:
            raise RuntimeError("Step had to fail!")
        return string


@Step.register("logging-step")
class LoggingStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, string: str, num_log_lines: int = 50) -> str:  # type: ignore
        for i in Tqdm.tqdm(list(range(num_log_lines)), desc="log progress"):
            time.sleep(0.1)
            self.logger.info(f"{i} - {string}")
        return string


@Step.register("make_number")
class MakeNumber(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, what_number: int) -> int:  # type: ignore
        return what_number


@Step.register("store_number_in_file")
class StoreNumberInFile(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self, number: int, file_name: str) -> None:  # type: ignore
        # Note: this is only for testing if the uncacheable step
        # ran in the multicore setting. Normally, a step like this
        # would be marked as CACHEABLE.
        with open(file_name, "w") as file_ref:
            file_ref.write(str(number))


@Step.register("multiprocessing_step")
class MultiprocessingStep(Step):
    """
    Mainly used to test that logging works properly in child processes.
    """

    def run(self, num_proc: int = 2) -> bool:  # type: ignore
        for _ in Tqdm.tqdm(list(range(10)), desc="progress from main process"):
            time.sleep(0.1)

        workers = []
        for i in range(num_proc):
            worker = mp.Process(target=_worker_function, args=(i,))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        return True


def _worker_function(worker_id: int):
    common_logging.initialize_worker_logging(worker_id)
    logger = logging.getLogger(MultiprocessingStep.__name__)
    logger.info("Hello from worker %d!", worker_id)
    common_logging.cli_logger.info("Hello from the cli logger in worker %d!", worker_id)
    for _ in Tqdm.tqdm(list(range(10)), desc="progress from worker", disable=worker_id > 0):
        time.sleep(0.1)
