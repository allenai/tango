import logging
from datetime import datetime

from pythonjsonlogger import jsonlogger


class JsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # Add timestamp.
        if not log_record.get("timestamp"):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            log_record["timestamp"] = now

        # Add log level.
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname

        # Add logger name.
        log_record["logger"] = record.name


def do_json_logging(prefix: str):
    from tango.common.logging import cli_logger
    from tango.common.tqdm import logger as tqdm_logger

    formatter = JsonFormatter()

    root_logger = logging.getLogger()
    for logger in (root_logger, cli_logger, tqdm_logger):
        logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
