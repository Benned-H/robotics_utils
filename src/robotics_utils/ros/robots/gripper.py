"""Define a class providing a ROS-based interface for robot grippers."""

from dataclasses import dataclass

import rospy
from actionlib.simple_action_client import SimpleActionClient
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from moveit_commander import RobotCommander


@dataclass(frozen=True)
class GripperJointLimits:
    """Specifies joint limits (in radians) for a robot gripper."""

    open_rad: float
    """Angle (radians) at which the gripper is fully open."""

    closed_rad: float
    """Angle (radians) at which the gripper is fully closed."""


class Gripper:
    """A ROS-based interface for a robot gripper."""

    def __init__(self, action_name: str, limits: GripperJointLimits, grasping_group: str) -> None:
        """Initialize a ROS action client to control a robot gripper.

        :param action_name: Name of the ROS action used to control the gripper
        :param limits: Joint limits (radians) for the gripper
        :param grasping_group: Name of the move group containing links used for grasping
        """
        self._action_name = action_name
        self._client = SimpleActionClient(self._action_name, GripperCommandAction)
        if not self._client.wait_for_server(timeout=rospy.Duration.from_sec(30)):
            error_msg = f"Couldn't find ROS action server '{self._action_name}' in time!"
            rospy.logerr(error_msg)
            raise RuntimeError(error_msg)

        self.joint_limits = limits

        self._robot_commander = RobotCommander()
        self.links = self._robot_commander.get_link_names(group=grasping_group)

    def move_to_angle_rad(self, target_rad: float, timeout_s: float = 10.0) -> None:
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
        self.move_to_angle_rad(self.joint_limits.open_rad)

    def close(self) -> None:
        """Close the gripper by calling its internal ROS action."""
        self.move_to_angle_rad(self.joint_limits.closed_rad)
