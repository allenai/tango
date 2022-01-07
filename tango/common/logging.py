"""
Tools for configuring logging.

Configuring logging in your own CLI
-----------------------------------

If you're writing your own CLI that uses tango, you can utilize the :func:`initialize_logging()`
function to easily configure logging properly.

For example,

.. testcode::

    from tango.common.logging import initialize_logging, teardown_logging

    initialize_logging(log_level="info")

    logger = logging.getLogger()
    logger.info("Running script!")

    teardown_logging()

.. testoutput::
    :options: +ELLIPSIS

    [... INFO root] Running script!

If you want to have logs written to a file, you can use the :func:`file_handler` context manager.

Logging from workers processes or threads
-----------------------------------------

"""

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
If this flag is set to ``True``, we remove special styling characters from log messages,
add newlines to tqdm output even on an interactive terminal, and we slow
down tqdm's output to only once every 10 seconds.

By default, it is set to ``False``. It can be changed by setting the corresponding environment
variable (``FILE_FRIENDLY_LOGGING``) or field in a :class:`~tango.__main__.TangoGlobalSettings`
file (``file_friendly_logging``) to "true" or "false",
or from the command line with the ``--file-friendly-logging`` flag.
For example,

.. code-block::

    $ tango --file-friendly-logging run ...

"""

TANGO_LOG_LEVEL: Optional[str] = os.environ.get("TANGO_LOG_LEVEL", None)
"""
The log level to use globally. The value can be set from the corresponding environment variable
(``TANGO_LOG_LEVEL``) or field in a :class:`~tango.__main__.TangoGlobalSettings` file (``log_level``),
or from the command line with the ``--log-level`` option.
Possible values are "debug", "info", "warning", or "error" (not case sensitive).
For example,

.. code-block::

    $ tango --log-level info run ...

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
    A custom subclass of :class:`logging.Logger` that does some additional cleaning
    of messages when :attr:`FILE_FRIENDLY_LOGGING` is on.

    This is the default logger class used when :func:`initialize_logging()` is called.
    """

    def __init__(self, name):
        super().__init__(name)

    def log(self, level, msg, *args, **kwargs):
        msg = msg if not FILE_FRIENDLY_LOGGING else click.unstyle(msg)
        super().log(level, msg, *args, **kwargs)


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


class WorkerLogFilter(logging.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"[rank {self._rank}] {record.msg}"
        return True


logging.setLoggerClass(TangoLogger)


click_logger = logging.getLogger("click")
"""
A logger that logs messages through
`click <https://click.palletsprojects.com/>`_'s
``click.echo()`` function.

This is provides a convenient way for command-line apps to log pretty, styled messages.
"""

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
    """
    Receives log records from worker processes and handles them.
    """
    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def get_logging_queue() -> Optional[mp.Queue]:
    """
    Get the logging queue to pass to :func:`initialize_logging()` from worker processes.
    """
    return _LOGGING_QUEUE


def initialize_logging(
    *,
    log_level: Optional[str] = None,
    enable_click_logs: bool = False,
    file_friendly_logging: Optional[bool] = None,
    prefix: Optional[str] = None,
    queue: Optional[mp.Queue] = None,
    worker_rank: Optional[int] = None,
):
    """
    Initialize logging, which includes setting the global log level, format, and configuring
    handlers.

    .. tip::
        This should be called as early on in your script as possible.

    .. tip::
        You should also call :func:`teardown_logging()` as the end of your script.

    Parameters
    ----------
    log_level : :class:`str`
        Can be one of "debug", "info", "warning", "error". Defaults to the value
        of :data:`TANGO_LOG_LEVEL`.
    enable_click_logs : :class:`bool`
        Set to ``True`` to enable messages from the :data:`click_logger`.
    file_friendly_logging : :class:`bool`
        Enable or disable file friendly logging. Defaults to the value of :data:`FILE_FRIENDLY_LOGGING`.
    prefix : :class:`str`
        An optional prefix to prepend to log lines.
    queue : :class:`multiprocessing.Queue`
        This should only be used from worker threads/processes, and should be set to result
        of :func:`get_logging_queue()`.

        .. tip::
            For worker threads/processes, use :func:`initialize_worker_logging()` instead.

    """
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
        if worker_rank is not None:
            queue_handler.addFilter(WorkerLogFilter(worker_rank))

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


def initialize_worker_logging(queue: mp.Queue, worker_rank: int):
    return initialize_logging(queue=queue, worker_rank=worker_rank)


def teardown_logging():
    """
    Cleanup any logging fixtures created from :func:`initialize_logging()`. Should
    be called at the end of your script.
    """
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
    """
    A context manager that can be used to route logs to a file by adding a
    :class:`logging.FileHandler` to the root logger's handlers.

    For example,

    .. code-block::

        from tango.common.logging import initialize_logging, file_handler, teardown_logging

        initialize_logging(log_level="info")

        logger = logging.getLogger()
        logger.info("Hi!")

        with file_handler("log.out"):
            logger.info("This message should also go into 'log.out'")

        teardown_logging()

    """
    handler = add_file_handler(filepath)
    try:
        yield handler
    finally:
        remove_file_handler(handler)
