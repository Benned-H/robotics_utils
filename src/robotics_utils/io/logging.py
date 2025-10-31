"""Define utility functions to simplify logging to the CLI."""

from rich.console import Console

try:
    import rospy

    ROS_PRESENT = True
except ModuleNotFoundError:
    ROS_PRESENT = False

import logging

logger = logging.getLogger(__name__)
console = Console()


def log_info(message: str) -> None:
    """Log the given string to standard output."""
    if ROS_PRESENT:
        rospy.loginfo(message)
    else:
        logger.info(message)
