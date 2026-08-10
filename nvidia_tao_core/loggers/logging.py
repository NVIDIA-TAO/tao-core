# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original source taken from https://github.com/NVIDIA/NeMo

"""Logger class"""

from abc import abstractmethod
from datetime import datetime
import json
import logging as _logging
import os


def _status_callback(data_string):
    """Forward status to the control plane when its optional stack is present.

    File status logging is used inside model containers that intentionally do
    not install the TAO API server's full cloud, database, and Kubernetes
    dependency graph.  Importing that graph at module import time made even
    ``StatusLogger`` unusable in those containers.
    """

    try:
        from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import status_callback
    except ImportError as exc:
        logger.debug("TAO control-plane status callback unavailable: %s", exc)
        return
    status_callback(data_string)


class MessageFormatter(_logging.Formatter):
    """Formatter that supports colored logs."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        _logging.DEBUG: grey + fmt + reset,
        _logging.INFO: grey + fmt + reset,
        _logging.WARNING: yellow + fmt + reset,
        _logging.ERROR: red + fmt + reset,
        _logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        """Format the log message."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = _logging.Formatter(log_fmt)
        return formatter.format(record)


logger = _logging.getLogger('TAO Toolkit')
logger.setLevel(_logging.DEBUG)
ch = _logging.StreamHandler()
ch.setLevel(_logging.DEBUG)
ch.setFormatter(MessageFormatter())
logger.addHandler(ch)
logging = logger


class Verbosity():
    """Verbosity levels."""

    DISABLE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# Defining a log level to name dictionary.
log_level_to_name = {
    Verbosity.DISABLE: "DISABLE",
    Verbosity.DEBUG: 'DEBUG',
    Verbosity.INFO: 'INFO',
    Verbosity.WARNING: 'WARNING',
    Verbosity.ERROR: 'ERROR',
    Verbosity.CRITICAL: 'CRITICAL'
}


class Status():
    """Status levels."""

    SUCCESS = 0
    FAILURE = 1
    STARTED = 2
    RUNNING = 3
    SKIPPED = 4


status_level_to_name = {
    Status.SUCCESS: 'SUCCESS',
    Status.FAILURE: 'FAILURE',
    Status.STARTED: 'STARTED',
    Status.RUNNING: 'RUNNING',
    Status.SKIPPED: 'SKIPPED'
}


class BaseLogger(object):
    """File logger class."""

    def __init__(self, is_master=False, verbosity=Verbosity.INFO):
        """Base logger class."""
        self.is_master = is_master
        self.verbosity = verbosity
        self.categorical = {}
        self.graphical = {}
        self.kpi = {}

    @property
    def date(self):
        """Get date from the status."""
        date_time = datetime.now()
        date_object = date_time.date()
        return f"{date_object.month}/{date_object.day}/{date_object.year}"

    @property
    def time(self):
        """Get date from the status."""
        date_time = datetime.now()
        time_object = date_time.time()
        return f"{time_object.hour}:{time_object.minute}:{time_object.second}"

    @property
    def categorical(self):
        """Categorical data to be logged."""
        return self._categorical

    @categorical.setter
    def categorical(self, value: dict):
        """Set categorical data to be logged."""
        self._categorical = value

    @property
    def graphical(self):
        """Graphical data to be logged."""
        return self._graphical

    @graphical.setter
    def graphical(self, value: dict):
        """Set graphical data to be logged."""
        self._graphical = value

    @property
    def kpi(self):
        """Set KPI data."""
        return self._kpi

    @kpi.setter
    def kpi(self, value: dict):
        """Set KPI data."""
        self._kpi = value

    def flush(self):
        """Flush the logger."""
        pass

    def format_data(self, data: dict):
        """Format the data."""
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary and not type {type(data)}.")
        data_string = json.dumps(data)
        return data_string

    def log(self, level, string):
        """Log the data string."""
        if level >= self.verbosity:
            logger.log(level, string)

    @abstractmethod
    def write(self, data=None,
              status_level=Status.RUNNING,
              verbosity_level=Verbosity.INFO,
              message=None):
        """Write data out to the log file."""
        if self.verbosity > Verbosity.DISABLE:
            if not data:
                data = {}
            # Define generic data.
            data["date"] = self.date
            data["time"] = self.time
            data["status"] = status_level_to_name.get(status_level, "RUNNING")
            data["verbosity"] = log_level_to_name.get(verbosity_level, "INFO")

            if message:
                data["message"] = message
            logger.log(verbosity_level, message)

            if self.categorical:
                data["categorical"] = self.categorical

            if self.graphical:
                data["graphical"] = self.graphical

            if self.kpi:
                data["kpi"] = self.kpi

            data_string = self.format_data(data)
            if self.is_master:
                self.log(verbosity_level, data_string)
                self.flush()
                _status_callback(data_string)


class StatusLogger(BaseLogger):
    """Simple logger to save the status file."""

    def __init__(self, filename=None,
                 is_master=False,
                 verbosity=Verbosity.INFO,
                 append=True):
        """Logger to write out the status."""
        super().__init__(is_master=is_master, verbosity=verbosity)
        self.log_path = os.path.realpath(filename)
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.append = append
        self.is_master = is_master
        if os.path.exists(self.log_path):
            logger.info(f"Log file already exists at {self.log_path}")
        if is_master:
            with open(self.log_path, "a" if append else "w", encoding="utf-8") as _:
                pass

    def log(self, level, string):
        """Log the data string."""
        if level >= self.verbosity:
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write(string + "\n")

    @staticmethod
    def format_data(data):
        """Format the dictionary data."""
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary and not type {type(data)}.")
        data_string = json.dumps(data)
        return data_string


# Define the logger here so it's static.
_STATUS_LOGGER = BaseLogger()


def set_status_logger(status_logger):
    """Set the status logger.

    Args:
        status_logger: An instance of the logger class.
    """
    global _STATUS_LOGGER  # pylint: disable=W0603 # noqa: F824
    _STATUS_LOGGER = status_logger


def get_status_logger():
    """Get the status logger."""
    global _STATUS_LOGGER  # pylint: disable=W0602,W0603 # noqa: F824
    return _STATUS_LOGGER
