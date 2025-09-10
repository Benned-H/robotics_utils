"""Define a class providing a ROS-based interface for robot grippers."""

from dataclasses import dataclass

import rospy
from actionlib.simple_action_client import SimpleActionClient
from control_msgs.msg import GripperCommandAction, GripperCommandGoal


@dataclass(frozen=True)
class GripperConfig:
    """Specifies joint limits (in radians) for a robot gripper."""

    open_rad: float
    """Angle (radians) at which the gripper is fully open."""

    closed_rad: float
    """Angle (radians) at which the gripper is fully closed."""


class Gripper:
    """A ROS-based interface for a robot gripper."""

    def __init__(self, action_name: str, config: GripperConfig) -> None:
        """Initialize a ROS action client to control a robot gripper.

        :param action_name: Name of the ROS action used to control the gripper
        """
        self._action_name = action_name
        self._client = SimpleActionClient(self._action_name, GripperCommandAction)
        if not self._client.wait_for_server(timeout=rospy.Duration.from_sec(30)):
            error_msg = f"Couldn't find ROS action server '{self._action_name}' in time!"
            rospy.logerr(error_msg)
            raise RuntimeError(error_msg)

        self.config = config

    def _move_to_angle_rad(self, target_rad: float, timeout_s: float = 10.0) -> None:
        """Move the gripper to a target angle (radians).

        :param target_rad: Target angle (radians) for the gripper
        :param timeout_s: Duration (seconds) after which the action is preempted (defaults to 10)
        """
        goal_msg = GripperCommandGoal()
        goal_msg.command.position = target_rad
        timeout_ros = rospy.Duration.from_sec(timeout_s)

        self._client.send_goal_and_wait(goal_msg, execute_timeout=timeout_ros)

    def open(self) -> None:
        """Open the gripper by calling its internal ROS action."""
        self._move_to_angle_rad(self.config.open_rad)

    def close(self) -> None:
        """Close the gripper by calling its internal ROS action."""
        self._move_to_angle_rad(self.config.closed_rad)
