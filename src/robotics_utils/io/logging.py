"""Define utility functions to simplify logging to the CLI."""

try:
    import rospy

    ROS_PRESENT = True
except ModuleNotFoundError:
    ROS_PRESENT = False

import logging

logger = logging.getLogger(__name__)


def log_info(message: str) -> None:
    """Log the given string to standard output."""
    if ROS_PRESENT:
        rospy.loginfo(message)
    else:
        logger.info(message)
