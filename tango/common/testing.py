import logging
import os
import pathlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from tango.common.aliases import EnvVarNames, PathOrStr
from tango.common.logging import initialize_logging, teardown_logging
from tango.common.params import Params

import logging
import multiprocessing as mp
import random
import time
from string import ascii_letters

import tango.common.logging as common_logging
from tango import Step
from tango.common import Tqdm


class TangoTestCase:
    """
    A custom testing class that

    * disables some of the more verbose logging,
    * creates and destroys a temp directory as a test fixture, and
    * restores the internal state of the `Registrable` class at the end of each test method.

    """

    PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
    """
    Root of the git repository.
    """

    # to run test suite with finished package, which does not contain
    # tests & fixtures, we must be able to look them up somewhere else
    PROJECT_ROOT_FALLBACK = (
        # users wanting to run test suite for installed package
        pathlib.Path(os.environ["TANGO_SRC_DIR"])
        if "TANGO_SRC_DIR" in os.environ
        else (
            # fallback for conda packaging
            pathlib.Path(os.environ["SRC_DIR"])
            if "CONDA_BUILD" in os.environ
            # stay in-tree
            else PROJECT_ROOT
        )
    )

    MODULE_ROOT = PROJECT_ROOT_FALLBACK / "tango"
    """
    Root of the tango module.
    """

    TESTS_ROOT = PROJECT_ROOT_FALLBACK / "tests"
    """
    Root of the tests directory.
    """

    FIXTURES_ROOT = PROJECT_ROOT_FALLBACK / "test_fixtures"
    """
    Root of the test fixtures directory.
    """

    def setup_method(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
        )

        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger("urllib3.connectionpool").disabled = True

        # Create a temporary scratch directory.
        self.TEST_DIR = Path(tempfile.mkdtemp(prefix="tango_tests"))
        os.makedirs(self.TEST_DIR, exist_ok=True)

        # Set an artificial console width so logs are not mangled.
        os.environ[EnvVarNames.CONSOLE_WIDTH.value] = str(300)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
        if EnvVarNames.CONSOLE_WIDTH.value in os.environ:
            del os.environ[EnvVarNames.CONSOLE_WIDTH.value]

    def run(
        self,
        config: Union[PathOrStr, Dict[str, Any], Params],
        overrides: Optional[Union[Dict[str, Any], str]] = None,
        include_package: Optional[List[str]] = None,
        workspace_url: Optional[str] = None,
        step_name: Optional[str] = None,
        parallelism: int = 1,
        multicore: bool = False,
        name: Optional[str] = None,
    ) -> Path:
        from tango.__main__ import TangoGlobalSettings, _run

        if isinstance(config, (dict, Params)):
            params = config if isinstance(config, Params) else Params(config)
            config = self.TEST_DIR / "config.json"
            params.to_file(cast(Path, config))

        if isinstance(overrides, dict):
            import json

            overrides = json.dumps(overrides)

        run_name = _run(
            TangoGlobalSettings(),
            str(config),
            workspace_url=workspace_url or "local://" + str(self.TEST_DIR / "workspace"),
            overrides=overrides,
            include_package=include_package,
            start_server=False,
            step_name=step_name,
            parallelism=parallelism,
            multicore=multicore,
            name=name,
        )

        return self.TEST_DIR / "workspace" / "runs" / run_name


@contextmanager
def run_experiment(
    config: Union[PathOrStr, Dict[str, Any], Params],
    overrides: Optional[Union[Dict[str, Any], str]] = None,
    file_friendly_logging: bool = True,
    include_package: Optional[List[str]] = None,
    parallelism: int = 1,
    multicore: bool = False,
    name: Optional[str] = None,
):
    """
    A context manager to make testing experiments easier. On ``__enter__`` it runs
    the experiment and returns the path to the run directory, a temporary directory that will be
    cleaned up on ``__exit__``.
    """
    initialize_logging(enable_cli_logs=True, file_friendly_logging=file_friendly_logging)
    test_case = TangoTestCase()
    try:
        test_case.setup_method()
        yield test_case.run(
            config,
            overrides=overrides,
            include_package=include_package,
            parallelism=parallelism,
            multicore=multicore,
            name=name,
        )
    finally:
        test_case.teardown_method()
        teardown_logging()


def requires_gpus(test_method):
    """
    Decorator to indicate that a test requires multiple GPU devices.
    """
    import pytest
    import torch

    return pytest.mark.gpu(
        pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 or more GPUs required.")(
            test_method
        )
    )


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
