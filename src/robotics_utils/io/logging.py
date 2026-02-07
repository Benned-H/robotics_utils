"""Define a unified approach to logging with rich formatting and ROS integration."""

from __future__ import annotations

import logging
import traceback
from typing import Callable, ClassVar

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

try:
    import rospy

    ROS_PRESENT = True
except ModuleNotFoundError:
    ROS_PRESENT = False


# TODO: Replace with unified approach below


logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


def log_info(message: str) -> None:
    """Log the given string to standard output."""
    if ROS_PRESENT:
        rospy.loginfo(message)
    else:
        logger.info(message)


def log_exception(exc: Exception, message: str | None = None) -> None:
    """Log an exception and optional message to standard error.

    :param exc: Exception to be logged
    :param message: Optional message describing the exception context (default: None)
    """
    formatted_exc = traceback.format_exc(exc)
    if ROS_PRESENT:
        if message:
            rospy.logerr(message)
        rospy.logerr(formatted_exc)
    else:
        if message:
            error_console.print(message)
        error_console.print(formatted_exc)


# TODO: Replace with unified approach below


def get_logger(name: str) -> logging.Logger:
    """Create and return a logger for the named file.

    :param name: Name of the calling file (i.e., its `__name__`)
    :return: Logger configured for the current runtime environment
    """
    logger = logging.getLogger(name)

    # Attach handlers for ROS or rich formatting (only once per logger)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        if ROS_PRESENT:  # Route to ROS logging if available
            logger.addHandler(_ROSHandler())
        else:  # Use Rich formatting for non-ROS logging
            handler = RichHandler(rich_tracebacks=True, show_path=False)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

    return logger


class _ROSHandler(logging.Handler):
    """A handler that routes Python logging to ROS."""

    _ROS_LOG_MAP: ClassVar[dict[int, Callable | None]] = {
        logging.DEBUG: rospy.logdebug if ROS_PRESENT else None,
        logging.INFO: rospy.loginfo if ROS_PRESENT else None,
        logging.WARNING: rospy.logwarn if ROS_PRESENT else None,
        logging.ERROR: rospy.logerr if ROS_PRESENT else None,
        logging.CRITICAL: rospy.logfatal if ROS_PRESENT else None,
    }

    def emit(self, record: logging.LogRecord) -> None:
        """Log the specified logging record to ROS, stripping any rich markup."""
        log_fn = self._ROS_LOG_MAP.get(record.levelno, rospy.loginfo)
        if log_fn:
            formatted = self.format(record)
            plain_text = Text.from_markup(formatted).plain
            log_fn(plain_text)
