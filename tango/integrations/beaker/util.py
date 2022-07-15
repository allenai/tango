import logging

import jsonpickle


class JsonHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        attrs = record.__dict__
        print(jsonpickle.dumps(attrs))


def do_json_logging(prefix: str):
    from tango.common.logging import cli_logger
    from tango.common.tqdm import logger as tqdm_logger

    root_logger = logging.getLogger()
    for logger in (root_logger, cli_logger, tqdm_logger):
        logger.handlers.clear()
        handler = JsonHandler()
        logger.addHandler(handler)
