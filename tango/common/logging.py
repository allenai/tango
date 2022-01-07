import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import threading
from contextlib import contextmanager
from typing import Optional

import click

from .aliases import PathOrStr
from .exceptions import SigTermReceived
from .util import _parse_bool

FILE_FRIENDLY_LOGGING: bool = _parse_bool(os.environ.get("FILE_FRIENDLY_LOGGING", False))
"""
If this flag is set to ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
down tqdm's output to only once every 10 seconds.

By default, it is set to ``False``.
"""

TANGO_LOG_LEVEL: Optional[str] = os.environ.get("TANGO_LOG_LEVEL", None)
"""
The log level.
"""

_LOGGING_QUEUE: Optional[mp.Queue] = None
"""
Used to send log records from worker processes back to the main logging thread.
"""

_LOGGING_THREAD: Optional[threading.Thread] = None
"""
Thread used for logging records from worker processes.
"""


class TangoLogger(logging.Logger):
    """
    A custom subclass of 'logging.Logger' that keeps a set of messages to
    implement {debug,info,etc.}_once() methods.
    """

    def __init__(self, name):
        super().__init__(name)
        self._seen_msgs = set()

    def log(self, level, msg, *args, **kwargs):
        msg = msg if not FILE_FRIENDLY_LOGGING else click.unstyle(msg)
        super().log(level, msg, *args, **kwargs)

    def debug_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.debug(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def info_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.info(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def warning_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.warning(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def error_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.error(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def critical_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.critical(msg, *args, **kwargs)
            self._seen_msgs.add(msg)


class TangoFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        out = super().format(record)
        if FILE_FRIENDLY_LOGGING:
            out = click.unstyle(out)
        return out


class WarningFilter(logging.Filter):
    """
    Filters out everything that is at the WARNING level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way WARNING and ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.WARNING


logging.setLoggerClass(TangoLogger)


click_logger = logging.getLogger("click")
click_logger.propagate = False


class ClickLoggerHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        click.echo(record.getMessage(), color=not FILE_FRIENDLY_LOGGING)


click_logger.addHandler(ClickLoggerHandler())
click_logger.disabled = (
    True  # This is disabled by default, in case nobody calls initialize_logging().
)


def get_formatter(prefix: Optional[str] = None) -> TangoFormatter:
    log_format = "[%(process)d %(asctime)s %(levelname)s %(name)s] %(message)s"
    if prefix is not None:
        log_format = prefix + " " + log_format
    return TangoFormatter(log_format)


def logger_thread(queue):
    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def get_logging_queue() -> Optional[mp.Queue]:
    return _LOGGING_QUEUE


def initialize_logging(
    *,
    log_level: Optional[str] = None,
    enable_click_logs: bool = False,
    file_friendly_logging: Optional[bool] = None,
    prefix: Optional[str] = None,
    queue: Optional[mp.Queue] = None,
):
    global FILE_FRIENDLY_LOGGING
    global TANGO_LOG_LEVEL
    global _LOGGING_THREAD
    global _LOGGING_QUEUE

    if mp.parent_process() is None and _LOGGING_THREAD is not None:
        raise RuntimeError("initialize_logging() can only be called once!")

    if log_level is None:
        log_level = TANGO_LOG_LEVEL
    if log_level is None:
        log_level = "error"
    if file_friendly_logging is None:
        file_friendly_logging = FILE_FRIENDLY_LOGGING

    level = logging._nameToLevel[log_level.upper()]

    logging.basicConfig(
        level=level,
    )
    TANGO_LOG_LEVEL = log_level
    os.environ["TANGO_LOG_LEVEL"] = log_level

    # filelock emits too many messages, so tell it to be quiet unless it has something
    # important to say.
    logging.getLogger("filelock").setLevel(max(level, logging.WARNING))

    # We always want to see all click messages if we're running from the command line, and none otherwise.
    click_logger.setLevel(logging.DEBUG)
    click_logger.disabled = not enable_click_logs

    if file_friendly_logging:
        FILE_FRIENDLY_LOGGING = True
        os.environ["FILE_FRIENDLY_LOGGING"] = "true"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if queue is not None:
        if mp.parent_process() is None:
            raise ValueError("'queue' can only be given to initialize_logging() in child processes")

        queue_handler = logging.handlers.QueueHandler(queue)
        queue_handler.setLevel(level)

        from .tqdm import logger as tqdm_logger

        for logger in (root_logger, click_logger, tqdm_logger):
            logger.handlers.clear()
            logger.addHandler(queue_handler)

    # Write uncaught exceptions to the logs.
    def excepthook(exctype, value, traceback):
        # For interruptions, call the original exception handler.
        if issubclass(exctype, (KeyboardInterrupt, SigTermReceived)):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook

    if mp.parent_process() is None:
        # Main process, set formatter and handlers, start logging thread.
        formatter = get_formatter(prefix)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.addFilter(WarningFilter())
        stdout_handler.setFormatter(formatter)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)

        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

        _LOGGING_QUEUE = mp.Queue()
        _LOGGING_THREAD = threading.Thread(
            target=logger_thread, args=(_LOGGING_QUEUE,), daemon=True
        )
        _LOGGING_THREAD.start()


def teardown_logging():
    global _LOGGING_QUEUE
    global _LOGGING_THREAD

    if _LOGGING_QUEUE is not None:
        _LOGGING_QUEUE.put(None)

    if _LOGGING_THREAD is not None:
        _LOGGING_THREAD.join()


def add_file_handler(filepath: PathOrStr) -> logging.FileHandler:
    root_logger = logging.getLogger()

    from .tqdm import logger as tqdm_logger

    handler = logging.FileHandler(str(filepath))
    formatter = get_formatter()
    handler.setFormatter(formatter)

    for logger in (root_logger, click_logger, tqdm_logger):
        logger.addHandler(handler)

    return handler


def remove_file_handler(handler: logging.FileHandler):
    root_logger = logging.getLogger()

    from .tqdm import logger as tqdm_logger

    for logger in (root_logger, click_logger, tqdm_logger):
        logger.removeHandler(handler)


@contextmanager
def file_handler(filepath: PathOrStr):
    handler = add_file_handler(filepath)
    try:
        yield handler
    finally:
        remove_file_handler(handler)
