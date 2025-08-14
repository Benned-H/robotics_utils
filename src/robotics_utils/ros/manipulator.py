"""Define a class representing a MoveIt-controlled robot manipulator."""

from __future__ import annotations

import sys

import rospy
from actionlib.simple_action_client import SimpleActionClient
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from moveit_commander import MoveGroupCommander, RobotCommander
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory

from robotics_utils.kinematics import Pose3D
from robotics_utils.motion_planning.plan_skeletons import ManipulationPlanSkeleton
from robotics_utils.ros.msg_conversion import pose_to_stamped_msg
from robotics_utils.ros.transform_manager import TransformManager


class Manipulator:
    """A robot manipulator corresponding to a MoveIt move group."""

    def __init__(self, move_group: str, base_link: str, grasping_group: str) -> None:
        """Initialize the manipulator using its MoveIt move group name.

        :param move_group: Name of the move group corresponding to the manipulator
        :param base_link: Starting link of the manipulator's kinematic chain
        :param grasping_group: Name of the move group containing links used for grasping
        """
        self.move_group_name = move_group

        self._move_group = MoveGroupCommander(move_group)
        self._move_group.set_pose_reference_frame(base_link)

        self._robot_commander = RobotCommander()
        self.gripper_links = self._robot_commander.get_link_names(group=grasping_group)
        self.ee_link = self._move_group.get_end_effector_link()

        # Create action client to control the robot's gripper
        self._gripper_action_name = "gripper_controller/gripper_action"
        self._gripper_client = SimpleActionClient(self._gripper_action_name, GripperCommandAction)
        if not self._gripper_client.wait_for_server(timeout=rospy.Duration.from_sec(60)):
            rospy.logerr(f"Couldn't find ROS action server '{self._gripper_action_name}' in time!")
            sys.exit(1)

        self._GRIPPER_OPEN_RAD = -1.5707  # Fully open gripper angle (rad)
        self._GRIPPER_CLOSED_RAD = 0.0  # Fully closed gripper angle (rad)

    def execute_gripper_angle_rad(self, target_rad: float, timeout_s: float = 10.0) -> None:
        """Move the manipulator's gripper to the given target angle (radians).

        :param target_rad: Target angle (radians) for the gripper
        :param timeout_s: Duration (seconds) after which the action is preempted (defaults to 10)
        """
        gripper_goal_msg = GripperCommandGoal()
        gripper_goal_msg.command.position = target_rad

        self._gripper_client.send_goal_and_wait(
            gripper_goal_msg,
            execute_timeout=rospy.Duration.from_sec(timeout_s),
        )

    def open_gripper(self) -> None:
        """Open the robot's gripper by calling the corresponding ROS action."""
        self.execute_gripper_angle_rad(self._GRIPPER_OPEN_RAD)

    def close_gripper(self) -> None:
        """Close the robot's gripper by calling the corresponding ROS action."""
        self.execute_gripper_angle_rad(self._GRIPPER_CLOSED_RAD)

    def execute_motion_plan(self, target_pose_ee: Pose3D, body_frame: str) -> None:
        """Compute a motion plan from the manipulator's current configuration."""
        target_pose_b_ee = TransformManager.convert_to_frame(target_pose_ee, body_frame)

        # Broadcast planned/actual frames for debugging
        TransformManager.broadcast_transform("target_pose_ee", target_pose_ee)
        TransformManager.broadcast_transform("target_pose_b_ee", target_pose_b_ee)

        self._move_group.set_pose_target(pose_to_stamped_msg(target_pose_b_ee))
        success, robot_traj, planning_time_s, _ = self._move_group.plan()

        succeeded_or_failed = "succeeded" if success else "failed"
        rospy.loginfo(f"Motion planning {succeeded_or_failed} after {planning_time_s} seconds.")

        if not success:
            sys.exit(1)

        self._move_group.clear_pose_targets()
        self._move_group.execute(robot_traj.joint_trajectory, wait=True)  # Blocks until done
        self._move_group.stop()  # Ensure there's no residual movement
