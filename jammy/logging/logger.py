import sys
from loguru import logger

__all__ = ["get_logger"]

logger.remove()

logger_sink = []

def get_logger(file_name=None, **kwargs):
    if file_name is None:
        file_name = sys.stderr
        kwargs["level"] = "INFO"
    global logger_sink
    if file_name in logger_sink:
        logger.debug(f"{file_name} already registered")
    else:
        logger.add(file_name,**kwargs)
        logger_sink.append(file_name)
    return logger
