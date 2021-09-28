import os
import logging

from .util import _parse_bool


class TangoLogger(logging.Logger):
    """
    A custom subclass of 'logging.Logger' that keeps a set of messages to
    implement {debug,info,etc.}_once() methods.
    """

    def __init__(self, name):
        super().__init__(name)
        self._seen_msgs = set()

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


logging.setLoggerClass(TangoLogger)
logger = logging.getLogger(__name__)


FILE_FRIENDLY_LOGGING: bool = _parse_bool(os.environ.get("FILE_FRIENDLY_LOGGING", False))
"""
If this flag is set to ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
down tqdm's output to only once every 10 seconds.

By default, it is set to ``False``.
"""
