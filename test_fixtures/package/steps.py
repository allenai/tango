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


@Step.register("random_string")
class RandomStringStep(Step):
    def run(self, length: int = 10) -> str:  # type: ignore
        return "".join([random.choice(ascii_letters) for _ in range(length)])


@Step.register("multiprocessing_step")
class MultiprocessingStep(Step):
    def run(self, num_proc: int = 2) -> bool:  # type: ignore
        for _ in Tqdm.tqdm(list(range(10)), desc="progress from main process"):
            time.sleep(0.1)

        workers = []
        for i in range(num_proc):
            worker = mp.Process(
                target=_worker_function, args=(i, common_logging.get_logging_queue())
            )
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        return True


def _worker_function(worker_id: int, logging_queue: mp.Queue):
    common_logging.initialize_worker_logging(worker_id, logging_queue)
    logger = logging.getLogger(MultiprocessingStep.__name__)
    logger.info("Hello from worker %d!", worker_id)
    for _ in Tqdm.tqdm(list(range(10)), desc="progress from worker", disable=worker_id > 0):
        time.sleep(0.1)
