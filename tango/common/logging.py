"""
Tango makes heavy use of the :mod:`logging` module from the standard library to convey information to users.
When you're writing your own :class:`~tango.step.Step` implementations we encourage you to also use standard
Python logging as opposed to :func:`print` or other functions that write directly to ``stdout`` or ``stderr``.
This is easy enough since each :class:`~tango.step.Step` class already comes with its own logger:
:attr:`Step.logger <tango.step.Step.logger>`.

When using the `Tango CLI <./commands.html>`_ you can set the log level in several different ways:

1. Through a Tango `global settings <./commands.html#global-settings>`_ file.
2. With the environment variable ``TANGO_LOG_LEVEL``.
3. Or with the ``--log-level`` command-line option.

In some cases (like when running on `Beaker <https://beaker.org>`_) you may also want
to enable `"file friendly logging" <#tango.common.logging.FILE_FRIENDLY_LOGGING>`_.

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

Logging from worker processes or threads
----------------------------------------

If you have steps or other functions that spawn workers, and you want to enable logging within
those workers, you can call the :func:`initialize_worker_logging()` function to configure
logging within each worker. This assumes that you've called :func:`initialize_logging()` from the
main process (the tango CLI does this for you).

For example,

.. testcode::

    import logging
    import multiprocessing as mp

    from tango import Step
    from tango.common.logging import initialize_worker_logging

    @Step.register("multiprocessing_step")
    class MultiprocessingStep(Step):
        def run(self, num_proc: int = 2) -> bool:  # type: ignore
            workers = []
            for i in range(num_proc):
                worker = mp.Process(target=_worker_function, args=(i,))
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()
            return True


    def _worker_function(worker_id: int):
        initialize_worker_logging(worker_rank=worker_id)
        logger = logging.getLogger(MultiprocessingStep.__name__)
        logger.info("Hello from worker %d!", worker_id)

"""

import logging
import logging.handlers
import os
import pickle
import socketserver
import struct
import sys
import threading
from contextlib import contextmanager
from typing import Optional

import click

from .aliases import PathOrStr
from .exceptions import SigTermReceived
from .util import _parse_bool, _parse_optional_int

FILE_FRIENDLY_LOGGING: bool = _parse_bool(os.environ.get("FILE_FRIENDLY_LOGGING", False))
"""
If this flag is set to ``True``, we remove special styling characters from log messages,
add newlines to :class:`~tango.common.tqdm.Tqdm` output even on an interactive terminal, and we slow
down :class:`~tango.common.tqdm.Tqdm`'s output to only once every 10 seconds.

.. attention::
    Unfortunately this won't affect ``tqdm`` output from other libraries that don't use
    Tango's :class:`~tango.common.tqdm.Tqdm` wrapper.

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

.. note::
    This does not affect the :data:`~tango.common.logging.click_logger`
    or logs from :class:`~tango.common.Tqdm` progress bars.

"""

# Click logger disabled by default in case nobody calls initialize_logging().
TANGO_CLICK_LOGGER_ENABLED: bool = _parse_bool(os.environ.get("TANGO_CLICK_LOGGER_ENABLED", False))


class TangoLogger(logging.Logger):
    """
    A custom subclass of :class:`logging.Logger` that does some additional cleaning
    of messages when :attr:`FILE_FRIENDLY_LOGGING` is on.

    This is the default logger class used when :func:`initialize_logging()` is called.
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


class WorkerLogFilter(logging.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"[rank {self._rank}] {record.msg}"
        return True


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.

    Taken from
    `the logging cookbook <https://docs.python.org/3.7/howto/logging-cookbook.html>`_.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver.

    Taken from
    `the logging cookbook <https://docs.python.org/3.7/howto/logging-cookbook.html>`_.
    """

    allow_reuse_address = True

    def __init__(self, host: str, port: int = 0):
        super().__init__((host, port), LogRecordStreamHandler)
        self.abort = False
        self.timeout = 0.2

    def serve_until_stopped(self):
        import select

        while not self.abort:
            rd, _, _ = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()


_LOGGING_HOST: str = os.environ.get("TANGO_LOGGING_HOST", "localhost")
_LOGGING_PORT: Optional[int] = _parse_optional_int(os.environ.get("TANGO_LOGGING_PORT", None))
_LOGGING_SERVER: Optional[LogRecordSocketReceiver] = None
_LOGGING_SERVER_THREAD: Optional[threading.Thread] = None


logging.setLoggerClass(TangoLogger)


click_logger = logging.getLogger("click")
"""
A logger that logs messages through
`click <https://click.palletsprojects.com/>`_'s
``click.echo()`` function.

This provides a convenient way for command-line apps to log pretty, styled messages.
"""

click_logger.propagate = False


class ClickLoggerHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if FILE_FRIENDLY_LOGGING:
            click.echo(click.unstyle(record.getMessage()))
        else:
            click.echo(record.getMessage())


click_logger.addHandler(ClickLoggerHandler())
click_logger.disabled = TANGO_CLICK_LOGGER_ENABLED


def get_formatter() -> TangoFormatter:
    log_format = "[%(process)d %(asctime)s %(levelname)s %(name)s] %(message)s"
    return TangoFormatter(log_format)


def initialize_logging(
    *,
    log_level: Optional[str] = None,
    enable_click_logs: Optional[bool] = None,
    file_friendly_logging: Optional[bool] = None,
):
    """
    Initialize logging, which includes setting the global log level, format, and configuring
    handlers.

    .. tip::
        This should be called as early on in your script as possible.

    .. tip::
        You should also call :func:`teardown_logging()` as the end of your script.

    .. tip::
        For worker threads/processes, use :func:`initialize_worker_logging()` instead.

    Parameters
    ----------
    log_level : :class:`str`
        Can be one of "debug", "info", "warning", "error". Defaults to the value
        of :data:`TANGO_LOG_LEVEL`.
    enable_click_logs : :class:`bool`
        Set to ``True`` to enable messages from the :data:`click_logger`.
    file_friendly_logging : :class:`bool`
        Enable or disable file friendly logging. Defaults to the value of :data:`FILE_FRIENDLY_LOGGING`.

    """
    import multiprocessing as mp

    is_main_process: bool
    if hasattr(mp, "parent_process"):  # python 3.8 or greater
        is_main_process = mp.parent_process() is None  # type: ignore
    else:
        is_main_process = mp.current_process().name == "MainProcess"

    _initialize_logging(
        log_level=log_level,
        enable_click_logs=enable_click_logs,
        file_friendly_logging=file_friendly_logging,
        main_process=is_main_process,
    )


def initialize_worker_logging(worker_rank: Optional[int] = None):
    """
    Initialize logging in a worker thread/process.

    Parameters
    ----------
    worker_rank : :class:`int`
        The rank/ID of the worker.

    """
    return _initialize_logging(worker_rank=worker_rank, main_process=False)


def _initialize_logging(
    *,
    log_level: Optional[str] = None,
    enable_click_logs: Optional[bool] = None,
    file_friendly_logging: Optional[bool] = None,
    worker_rank: Optional[int] = None,
    main_process: bool = True,
):
    global FILE_FRIENDLY_LOGGING, TANGO_LOG_LEVEL, TANGO_CLICK_LOGGER_ENABLED
    global _LOGGING_HOST, _LOGGING_PORT, _LOGGING_SERVER, _LOGGING_SERVER_THREAD

    if log_level is None:
        log_level = TANGO_LOG_LEVEL
    if log_level is None:
        log_level = "error"
    if file_friendly_logging is None:
        file_friendly_logging = FILE_FRIENDLY_LOGGING
    if enable_click_logs is None:
        enable_click_logs = TANGO_CLICK_LOGGER_ENABLED

    level = logging._nameToLevel[log_level.upper()]

    # Update global flags and corresponding environment variables, if necessary,
    # so that child processes can read the environment variables to determine the right
    # settings.
    TANGO_LOG_LEVEL = log_level
    os.environ["TANGO_LOG_LEVEL"] = log_level
    if file_friendly_logging is not None:
        FILE_FRIENDLY_LOGGING = file_friendly_logging
        os.environ["FILE_FRIENDLY_LOGGING"] = str(file_friendly_logging).lower()
    if enable_click_logs is not None:
        TANGO_CLICK_LOGGER_ENABLED = enable_click_logs
        os.environ["TANGO_CLICK_LOGGER_ENABLED"] = str(enable_click_logs).lower()

    from .tqdm import logger as tqdm_logger

    # Handle special cases for specific loggers:
    # These loggers emit too many messages, so we tell them to be quiet unless they have something
    # important to say.
    for loud_logger in {"filelock", "sqlitedict"}:
        logging.getLogger(loud_logger).setLevel(max(level, logging.WARNING))
    # We always want to see all click messages if we're running from the command line, and none otherwise.
    click_logger.setLevel(logging.DEBUG)
    click_logger.disabled = not enable_click_logs
    # We also want to enable the tqdm logger so that the progress bar lines end up in the log file.
    tqdm_logger.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if main_process:
        formatter = get_formatter()

        # Create stdout and stderr handlers so that we can route DEBUG and INFO
        # messages to stdout, and WARNING and ERROR messages to stderr.
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.addFilter(WarningFilter())
        stdout_handler.setFormatter(formatter)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)

        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

        # Main process: set formatter and handlers, initialize logging socket and server.
        # Set up logging socket to emit log records from worker processes/threads.
        # Inspired by:
        # https://docs.python.org/3.7/howto/logging-cookbook.html#sending-and-receiving-logging-events-across-a-network
        _LOGGING_SERVER = LogRecordSocketReceiver(_LOGGING_HOST, 0)
        _LOGGING_PORT = _LOGGING_SERVER.server_address[1]
        os.environ["TANGO_LOGGING_PORT"] = str(_LOGGING_PORT)
        _LOGGING_SERVER_THREAD = threading.Thread(
            target=_LOGGING_SERVER.serve_until_stopped, daemon=True
        )
        _LOGGING_SERVER_THREAD.start()
    else:
        # Child process: set handler and level, no need to set formatting since only raw log records
        # will be sent to the logging socket.
        if _LOGGING_PORT is None:
            raise ValueError(
                "missing logging socket configuration, "
                "did you forget to call 'initialize_logging()' from the main process?"
            )
        socket_handler = logging.handlers.SocketHandler(_LOGGING_HOST, _LOGGING_PORT)
        if worker_rank is not None:
            socket_handler.addFilter(WorkerLogFilter(worker_rank))

        for logger in (root_logger, click_logger, tqdm_logger):
            logger.handlers.clear()
            logger.addHandler(socket_handler)

    # Write uncaught exceptions to the logs.
    def excepthook(exctype, value, traceback):
        # For interruptions, call the original exception handler.
        if issubclass(exctype, (KeyboardInterrupt, SigTermReceived)):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook

    # Ensure warnings issued by the 'warnings' module will be redirected to the logging system.
    logging.captureWarnings(True)


def teardown_logging():
    """
    Cleanup any logging fixtures created from :func:`initialize_logging()`. Should
    be called at the end of your script.
    """
    global _LOGGING_HOST, _LOGGING_PORT, _LOGGING_SERVER, _LOGGING_SERVER_THREAD

    if _LOGGING_SERVER is not None:
        _LOGGING_SERVER.abort = True

    if _LOGGING_SERVER_THREAD is not None:
        _LOGGING_SERVER_THREAD.join()
        _LOGGING_SERVER_THREAD = None

    if _LOGGING_SERVER is not None:
        _LOGGING_SERVER = None


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
