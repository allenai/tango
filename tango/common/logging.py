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

    [...] INFO     Running script! ...

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

    @Step.register("multiprocessing_step_example")
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
from typing import ContextManager, Generator, List, Optional

import rich
from rich.console import Console
from rich.logging import RichHandler as _RichHandler
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table

from .aliases import EnvVarNames, PathOrStr
from .exceptions import CliRunError, SigTermReceived
from .util import _parse_bool, _parse_optional_int

FILE_FRIENDLY_LOGGING: bool = _parse_bool(
    os.environ.get(EnvVarNames.FILE_FRIENDLY_LOGGING.value, False)
)
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

TANGO_LOG_LEVEL: Optional[str] = os.environ.get(EnvVarNames.LOG_LEVEL.value, None)
"""
The log level to use globally. The value can be set from the corresponding environment variable
(``TANGO_LOG_LEVEL``) or field in a :class:`~tango.__main__.TangoGlobalSettings` file (``log_level``),
or from the command line with the ``--log-level`` option.
Possible values are "debug", "info", "warning", or "error" (not case sensitive).
For example,

.. code-block::

    $ tango --log-level info run ...

.. note::
    This does not affect the :data:`~tango.common.logging.cli_logger`
    or logs from :class:`~tango.common.Tqdm` progress bars.

"""

TANGO_CONSOLE_WIDTH: Optional[int] = _parse_optional_int(
    os.environ.get(EnvVarNames.CONSOLE_WIDTH.value, None)
)

# Click logger disabled by default in case nobody calls initialize_logging().
TANGO_CLI_LOGGER_ENABLED: bool = _parse_bool(
    os.environ.get(EnvVarNames.CLI_LOGGER_ENABLED.value, False)
)

# Keep track of exceptions logged so we don't log duplicates from our custom excepthook.
_EXCEPTIONS_LOGGED: List[BaseException] = []


class LevelFilter(logging.Filter):
    """
    Filters out everything that is above `max_level` or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way WARNING or ERROR
    messages aren't duplicated.
    """

    def __init__(self, max_level: int, min_level: Optional[int] = None, name=""):
        self.max_level = max_level
        self.min_level = min_level
        super().__init__(name)

    def filter(self, record):
        if self.min_level is not None:
            return self.min_level <= record.levelno <= self.max_level
        else:
            return record.levelno <= self.max_level


class CliFilter(logging.Filter):
    def __init__(self, filter_out: bool):
        self.filter_out = filter_out

    def filter(self, record):
        if self.filter_out:
            return record.name != "tango.__main__"
        else:
            return record.name == "tango.__main__"


class WorkerLogFilter(logging.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"[rank {self._rank}] {record.msg}"
        return True


class PrefixLogFilter(logging.Filter):
    def __init__(self, prefix):
        super().__init__()
        self._prefix = prefix

    def filter(self, record):
        if record.name == "tango.__main__":
            from rich.markup import escape

            record.msg = escape(f"[{self._prefix}] ") + record.msg
        else:
            record.msg = f"[{self._prefix}] {record.msg}"
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


_LOGGING_PREFIX: str = os.environ.get(EnvVarNames.LOGGING_PREFIX.value, "")
_LOGGING_HOST: str = os.environ.get(EnvVarNames.LOGGING_HOST.value, "localhost")
_LOGGING_PORT: Optional[int] = _parse_optional_int(
    os.environ.get(EnvVarNames.LOGGING_PORT.value, None)
)
_LOGGING_SERVER: Optional[LogRecordSocketReceiver] = None
_LOGGING_SERVER_THREAD: Optional[threading.Thread] = None


class RichHandler(_RichHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if isinstance(record.msg, Table):
            if record.msg.title is not None:
                attrdict = {k: v for k, v in record.__dict__.items() if k != "msg"}
                attrdict["msg"] = "[italic]" + record.msg.title + "[/]"
                self.emit(logging.makeLogRecord(attrdict))
            record.msg.title = None
            self.console.print(Padding(record.msg, (1, 0, 1, 1)))
        elif isinstance(record.msg, Syntax):
            self.console.print(Padding(record.msg, (1, 0, 1, 1)))
        elif hasattr(record.msg, "__rich__") or hasattr(record.msg, "__rich_console__"):
            self.console.print(record.msg)
        else:
            super().emit(record)


def get_handler(
    level: int,
    stderr: bool = False,
    enable_markup: bool = False,
    show_time: bool = True,
    show_level: bool = True,
    show_path: bool = True,
) -> logging.Handler:
    import click

    console = Console(
        color_system="auto" if not FILE_FRIENDLY_LOGGING else None,
        stderr=stderr,
        width=TANGO_CONSOLE_WIDTH,
    )
    if TANGO_CONSOLE_WIDTH is None and not console.is_terminal:
        console.width = 160
    handler = RichHandler(
        level=level,
        console=console,
        rich_tracebacks=False,
        tracebacks_show_locals=False,
        tracebacks_suppress=[click],
        markup=enable_markup,
        show_time=show_time,
        show_level=show_level,
        show_path=show_path,
        omit_repeated_times=False,
        highlighter=rich.highlighter.NullHighlighter(),
    )
    return handler


cli_logger = logging.getLogger("tango.__main__")
"""
A logger that emits messages directly to stdout/stderr using
`rich <https://github.com/Textualize/rich>`_'s
:class:`~rich.console.Console` class.

This provides a convenient way for command-line apps to log pretty, styled messages
uses the `markup style <https://rich.readthedocs.io/en/latest/markup.html>`_ provided by `rich`.
"""

cli_logger.propagate = False
cli_logger.disabled = TANGO_CLI_LOGGER_ENABLED


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    global _EXCEPTIONS_LOGGED
    # Ignore `CliRunError` because we don't need a traceback for those.
    if issubclass(exctype, (CliRunError,)):
        return
    # For interruptions, call the original exception handler.
    if issubclass(
        exctype,
        (
            KeyboardInterrupt,
            SigTermReceived,
        ),
    ):
        sys.__excepthook__(exctype, value, traceback)
        return
    if value not in _EXCEPTIONS_LOGGED:
        _EXCEPTIONS_LOGGED.append(value)
        root_logger = logging.getLogger()
        root_logger.error(
            "Uncaught exception",
            exc_info=(exctype, value, traceback),
            extra={"highlighter": rich.highlighter.ReprHighlighter()},
        )


def initialize_logging(
    *,
    log_level: Optional[str] = None,
    enable_cli_logs: Optional[bool] = None,
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

    :param log_level:
        Can be one of "debug", "info", "warning", "error". Defaults to the value
        of :data:`TANGO_LOG_LEVEL`, if set, or "error".
    :param enable_cli_logs:
        Set to ``True`` to enable messages from the :data:`cli_logger`.
    :param file_friendly_logging:
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
        enable_cli_logs=enable_cli_logs,
        file_friendly_logging=file_friendly_logging,
        main_process=is_main_process,
    )


def initialize_worker_logging(worker_rank: Optional[int] = None):
    """
    Initialize logging in a worker thread/process.

    :param worker_rank:
        The rank/ID of the worker.

    """
    if worker_rank is not None:
        if worker_rank != -1:
            prefix = f"rank {worker_rank}"
        else:
            prefix = None
    else:
        prefix = None
    return initialize_prefix_logging(prefix=prefix, main_process=False)


def initialize_prefix_logging(prefix: Optional[str] = None, main_process: bool = False):
    """
    Initialize logging with a prefix.

    :param prefix:
        The string prefix to add to the log message.
    :param main_process:
        Whether it is for the main/worker process.
    """
    return _initialize_logging(prefix=prefix, main_process=main_process)


def _initialize_logging(
    *,
    log_level: Optional[str] = None,
    enable_cli_logs: Optional[bool] = None,
    file_friendly_logging: Optional[bool] = None,
    prefix: Optional[str] = None,
    main_process: bool = True,
):
    global FILE_FRIENDLY_LOGGING, TANGO_LOG_LEVEL, TANGO_CLI_LOGGER_ENABLED
    global _LOGGING_HOST, _LOGGING_PORT, _LOGGING_SERVER, _LOGGING_SERVER_THREAD, _LOGGING_PREFIX

    if log_level is None:
        log_level = TANGO_LOG_LEVEL
    if log_level is None:
        log_level = "error"
    if file_friendly_logging is None:
        file_friendly_logging = FILE_FRIENDLY_LOGGING
    if enable_cli_logs is None:
        enable_cli_logs = TANGO_CLI_LOGGER_ENABLED

    level = logging._nameToLevel[log_level.upper()]

    # Update global flags and corresponding environment variables, if necessary,
    # so that child processes can read the environment variables to determine the right
    # settings.
    TANGO_LOG_LEVEL = log_level
    os.environ[EnvVarNames.LOG_LEVEL.value] = log_level
    if file_friendly_logging is not None:
        FILE_FRIENDLY_LOGGING = file_friendly_logging
        os.environ[EnvVarNames.FILE_FRIENDLY_LOGGING.value] = str(file_friendly_logging).lower()
    if enable_cli_logs is not None:
        TANGO_CLI_LOGGER_ENABLED = enable_cli_logs
        os.environ[EnvVarNames.CLI_LOGGER_ENABLED.value] = str(enable_cli_logs).lower()

    from .tqdm import logger as tqdm_logger

    # Handle special cases for specific loggers:
    # These loggers emit too many messages, so we tell them to be quiet unless they have something
    # important to say.
    for loud_logger in {"filelock", "sqlitedict"}:
        logging.getLogger(loud_logger).setLevel(max(level, logging.WARNING))
    # We always want to see all CLI messages if we're running from the command line, and none otherwise.
    cli_logger.setLevel(logging.DEBUG)
    cli_logger.disabled = not enable_cli_logs
    # We also want to enable the tqdm logger so that the progress bar lines end up in the log file.
    tqdm_logger.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if main_process:
        # Create stdout and stderr handlers so that we can route DEBUG and INFO
        # messages to stdout, and WARNING and ERROR messages to stderr.
        stdout_handler = get_handler(level)
        stdout_handler.addFilter(LevelFilter(logging.INFO))
        stderr_handler = get_handler(max(level, logging.WARNING), stderr=True)
        stderr_handler.addFilter(LevelFilter(logging.CRITICAL, min_level=logging.WARNING))
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

        # Configure cli_logger so that if log level <= INFO, it will behave
        # like a regular logger, otherwise it prints directly to stdout.
        cli_logger.handlers.clear()
        if enable_cli_logs:
            for handler_level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
                cli_handler = get_handler(
                    handler_level,
                    stderr=handler_level >= logging.WARNING,
                    enable_markup=True,
                    show_time=level <= handler_level,
                    show_level=(level <= handler_level) or handler_level >= logging.WARNING,
                    show_path=level <= handler_level,
                )
                cli_handler.addFilter(LevelFilter(handler_level))
                cli_logger.addHandler(cli_handler)

        # Main process: set formatter and handlers, initialize logging socket and server.
        # Set up logging socket to emit log records from worker processes/threads.
        # Inspired by:
        # https://docs.python.org/3.7/howto/logging-cookbook.html#sending-and-receiving-logging-events-across-a-network
        _LOGGING_SERVER = LogRecordSocketReceiver(_LOGGING_HOST, 0)
        _LOGGING_PORT = _LOGGING_SERVER.server_address[1]
        os.environ[EnvVarNames.LOGGING_PORT.value] = str(_LOGGING_PORT)
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
        if prefix:
            prefix = _LOGGING_PREFIX + " " + prefix if _LOGGING_PREFIX else prefix
        else:
            prefix = _LOGGING_PREFIX
        if prefix:
            socket_handler.addFilter(PrefixLogFilter(prefix))

        for logger in (root_logger, cli_logger, tqdm_logger):
            logger.handlers.clear()
            logger.addHandler(socket_handler)

    # Write uncaught exceptions to the logs.
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

    sys.excepthook = sys.__excepthook__  # type: ignore[assignment]


@contextmanager
def insert_handlers(*handlers: logging.Handler) -> Generator[None, None, None]:
    """
    A context manager that can be used to route logs to a specific handler temporarily.
    """
    global _EXCEPTIONS_LOGGED

    root_logger = logging.getLogger()

    from .tqdm import logger as tqdm_logger

    for logger in (root_logger, cli_logger, tqdm_logger):
        for handler in handlers:
            logger.addHandler(handler)

    try:
        yield None
    except BaseException as e:
        # We don't log `CliRunError` because we don't need a traceback for those.
        if not isinstance(e, CliRunError):
            root_logger.exception(e)
            _EXCEPTIONS_LOGGED.append(e)
        raise
    finally:
        for logger in (root_logger, cli_logger, tqdm_logger):
            for handler in handlers:
                logger.removeHandler(handler)


def file_handler(filepath: PathOrStr) -> ContextManager[None]:
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
    import click

    log_file = open(filepath, "w")
    handlers: List[logging.Handler] = []
    console = Console(
        color_system=None,
        file=log_file,
        force_terminal=False,
        width=TANGO_CONSOLE_WIDTH or 160,
    )
    for is_cli_handler in (True, False):
        handler = RichHandler(
            console=console,
            tracebacks_suppress=[click],
            markup=is_cli_handler,
            highlighter=rich.highlighter.NullHighlighter(),
            omit_repeated_times=False,
        )
        handler.addFilter(CliFilter(filter_out=not is_cli_handler))
        handlers.append(handler)
    return insert_handlers(*handlers)
