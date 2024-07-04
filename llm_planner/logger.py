import logging
import sys


class Logger:

    def __init__(self, tag: str):
        # Configure the logger
        logger = logging.getLogger(tag)
        logger.setLevel(logging.DEBUG)  # Set the logging level

        # Create a StreamHandler to log to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(
            logging.DEBUG)  # Set the logging level for the handler

        # Create a formatter and set it for the handler
        #formatter = logging.Formatter(
        #    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('[%(name)s] %(message)s')
        stdout_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(stdout_handler)

        self.logger = logger

    def info(self, input: str):
        self.logger.info(input)
