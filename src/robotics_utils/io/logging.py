"""Define utility functions to simplify logging to the CLI."""

try:
    import rospy

    ROS_PRESENT = True
except ModuleNotFoundError:
    ROS_PRESENT = False

import logging

logger = logging.getLogger(__name__)


def log_info(message: str) -> None:
    """Log the given information."""
    if ROS_PRESENT:
        rospy.loginfo(message)
    else:
        logger.info(message)


def log_warn(message: str) -> None:
    """Log the given warning."""
    if ROS_PRESENT:
        rospy.logwarn(message)
    else:
        logger.warning(message)


def log_err(message: str) -> None:
    """Log the given error."""
    if ROS_PRESENT:
        rospy.logerr(message)
    else:
        logger.error(message)
