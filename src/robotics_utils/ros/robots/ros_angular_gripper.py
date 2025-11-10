"""Define a class providing a ROS-based interface for angular robot grippers."""

from __future__ import annotations

import rospy
from actionlib.simple_action_client import SimpleActionClient
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from moveit_commander import RobotCommander

from robotics_utils.robots import AngularGripper, GripperAngleLimits


class ROSAngularGripper(AngularGripper):
    """A ROS-based interface for an angular robot gripper."""

    def __init__(self, limits: GripperAngleLimits, grasping_group: str, action_name: str) -> None:
        """Initialize a ROS action client to control a robot gripper.

        :param limits: Joint limits (radians) for the gripper
        :param grasping_group: Name of the move group containing links used for grasping
        :param action_name: Name of the ROS action used to control the gripper
        """
        super().__init__(limits)

        self.grasping_group = grasping_group
        self._action_name = action_name
        self._client = SimpleActionClient(self._action_name, GripperCommandAction)
        if not self._client.wait_for_server(timeout=rospy.Duration.from_sec(30)):
            error_msg = f"Couldn't find ROS action server '{self._action_name}' in time!"
            rospy.logerr(error_msg)
            raise RuntimeError(error_msg)

        self._robot_commander = RobotCommander()

    @property
    def link_names(self) -> list[str]:
        """Retrieve the names of the links in the gripper."""
        return list(self._robot_commander.get_link_names(group=self.grasping_group))

    def move_to_angle_rad(self, target_rad: float, timeout_s: float = 10.0) -> None:
        """Move the gripper to a target angle (radians).

        :param target_rad: Target angle (radians) for the gripper
        :param timeout_s: Duration (seconds) after which the motion times out (defaults to 10 sec)
        """
        goal_msg = GripperCommandGoal()
        goal_msg.command.position = target_rad
        timeout_ros = rospy.Duration.from_sec(timeout_s)

        self._client.send_goal_and_wait(goal_msg, execute_timeout=timeout_ros)
