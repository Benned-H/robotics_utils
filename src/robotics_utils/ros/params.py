"""Define utility functions to support loading from ROS parameters."""

from __future__ import annotations

from typing import TypeVar

import rospy

ParamT = TypeVar("ParamT")


def get_ros_param(name: str, param_type: type[ParamT]) -> ParamT:
    """Retrieve the parameter with the given name and type from the ROS parameter server.

    :param name: Name of the retrieved ROS parameter
    :param param_type: Type of the retrieved parameter
    :return: Value retrieved from the ROS parameter server
    """
    param_value = rospy.get_param(name)
    return param_type(param_value)
