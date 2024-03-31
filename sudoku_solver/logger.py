from __future__ import annotations

import logging
import os
from datetime import datetime

import colorlog


class Logger:
    LOGGER_DIRECTORY = "sudoku_solver_logs"

    def __init__(self, log_file: str | None = None, debug_mode:bool=False):
        os.makedirs(Logger.LOGGER_DIRECTORY, exist_ok=True)

        if log_file is None:
            today = datetime.today()
            log_file = today.strftime("%Y_%m_%d") + "_sudoku_solver.log"

        self.logger_name = Logger.LOGGER_DIRECTORY + os.sep + log_file
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG) if debug_mode else self.logger.setLevel(logging.INFO)

        # Create a ColorFormatter
        self.color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # Format for date and time
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Create a StreamHandler for console output
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.DEBUG)
        self.stream_handler.setFormatter(self.color_formatter)

        # Create a FileHandler for file output
        self.file_handler = logging.FileHandler(self.logger_name)
        self.file_handler.setLevel(logging.DEBUG)

        # Create a Formatter for the file handler
        self.file_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # Format for date and time
        )
        self.file_handler.setFormatter(self.file_formatter)

        # Add the handlers to the logger
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)



if __name__ == "__main__":
    logger = Logger(debug_mode=True)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
