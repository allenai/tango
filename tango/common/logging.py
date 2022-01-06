import logging
import os
import sys
from typing import Optional

import click

from .aliases import PathOrStr
from .exceptions import SigTermReceived
from .util import _parse_bool


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


logging.setLoggerClass(TangoLogger)


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
    log_format = "[%(asctime)s %(levelname)s %(name)s] %(message)s"
    if prefix is not None:
        log_format = prefix + " " + log_format
    return TangoFormatter(log_format)


def initialize_logging(
    *,
    log_level: Optional[str] = None,
    enable_click_logs: bool = False,
    file_friendly_logging: Optional[bool] = None,
    prefix: Optional[str] = None,
):
    global FILE_FRIENDLY_LOGGING
    global TANGO_LOG_LEVEL

    if log_level is None:
        log_level = TANGO_LOG_LEVEL
    if log_level is None:
        log_level = "error"
    if file_friendly_logging is None:
        file_friendly_logging = FILE_FRIENDLY_LOGGING

    level = logging._nameToLevel[log_level.upper()]
    formatter = get_formatter(prefix)
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

    # Write uncaught exceptions to the logs.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    def excepthook(exctype, value, traceback):
        # For interruptions, call the original exception handler.
        if issubclass(exctype, (KeyboardInterrupt, SigTermReceived)):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook


def add_file_handler(filepath: PathOrStr):
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(str(filepath))
    formatter = get_formatter()
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    click_logger.addHandler(file_handler)
