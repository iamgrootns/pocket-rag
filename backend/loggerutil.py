"""
Logger helper. Same pattern as DFAPI loggerutil.
Every class gets its own named logger from here.
"""

import logging


class logger_util:
    """Simple logger factory - one clean line per event."""

    def __init__(self, name: str = "pocket-rag"):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def info(self, msg: str):
        self._logger.info(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def exception(self, msg: str):
        self._logger.error(msg, exc_info=True)
