import logging
import os.path
import sys


def custom_logger(logger_name, log_file_path, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = (
        "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
        "%(lineno)d — %(message)s"
    )
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    # file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler = logging.FileHandler(os.path.join(log_file_path, logger_name))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
