"""Define a class to solve motion planning queries using MoveIt."""

from typing import Tuple

from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

MoveItResult = Tuple[bool, RobotTrajectory, float, MoveItErrorCodes]
"""Boolean success, trajectory message, planning time (s), and error codes.

Reference: https://docs.ros.org/en/noetic/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html#a79a475263bffd96978c87488a2bf7c98
"""
