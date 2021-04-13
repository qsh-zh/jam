import os
import sys
from loguru import logger

__all__ = ["get_logger"]

logger.remove()

logger_sink = {}


def get_logger(file_name=None, **kwargs):
    if file_name is None:
        file_name = sys.stderr
        if "level" not in kwargs:
            kwargs["level"] = "INFO"
    global logger_sink
    if file_name in logger_sink.values():
        logger.debug(f"{file_name} already registered")
    else:
        # if "level" not in kwargs:
        # kwargs["level"] = "DEBUG" if jam_is_debug() else "INFO"
        sink_id = logger.add(file_name, **kwargs)
        logger_sink[sink_id] = file_name
    return logger
