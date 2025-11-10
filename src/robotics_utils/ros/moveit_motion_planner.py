"""Define a class to solve motion planning queries using MoveIt."""

from typing import Tuple

from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

MoveItResult = Tuple[bool, RobotTrajectory, float, MoveItErrorCodes]
"""Boolean success, trajectory message, planning time (s), and error codes.

Reference: https://tinyurl.com/moveit-noetic-plan
"""
