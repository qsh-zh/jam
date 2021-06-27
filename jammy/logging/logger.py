import sys

from loguru import logger

__all__ = ["get_logger"]

logger.remove()

logger_sink = {}
LOG_FORMAT = (
    "<green>{time:MM-DD HH:mm:ss.SSS}</green> |"
    "<red>[{process.name}]</red>|"
    "<level>{level: <8}</level> |"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def get_logger(file_name=None, clear=False, **kwargs):
    global logger_sink  # pylint: disable=global-statement
    if clear:
        logger.remove()
        logger_sink.clear()
    if file_name is None:
        file_name = sys.stderr
        if "level" not in kwargs:
            kwargs["level"] = "INFO"
    if "format" not in kwargs:
        kwargs["format"] = LOG_FORMAT
    if file_name in logger_sink.values():
        # FIXME:
        # logger.debug("already registered")
        logger.debug(f"{str(file_name)} already registered")
    else:
        # if "level" not in kwargs:
        # kwargs["level"] = "DEBUG" if jam_is_debug() else "INFO"
        sink_id = logger.add(file_name, **kwargs)
        logger_sink[sink_id] = file_name
        logger.debug(f"sink_id: {sink_id:02d} ---> {str(file_name)}")
    return logger
