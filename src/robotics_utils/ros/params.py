"""Define utility functions to support loading from ROS parameters."""

from __future__ import annotations

from typing import TypeVar

import rospy

ParamT = TypeVar("ParamT")


def get_ros_param(name: str, param_t: type[ParamT], default_value: ParamT | None = None) -> ParamT:
    """Retrieve the parameter with the given name and type from the ROS parameter server.

    :param name: Name of the retrieved ROS parameter
    :param param_t: Type of the retrieved parameter
    :param default_value: Default value used if the ROS parameter doesn't exist (defaults to None)
    :return: Value retrieved from the ROS parameter server
    """
    if default_value is None:
        param_value = rospy.get_param(name)
    else:
        param_value = rospy.get_param(name, default=default_value)

    return param_t(param_value)
