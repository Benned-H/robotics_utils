"""Define a class to interface with robot manipulators using MoveIt."""

from __future__ import annotations

import sys

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize

from robotics_utils.kinematics import Configuration, Pose3D
from robotics_utils.ros.msg_conversion import pose_to_stamped_msg
from robotics_utils.ros.robots.gripper import Gripper
from robotics_utils.ros.transform_manager import TransformManager


class Manipulator:
    """A MoveIt-based interface for a robot manipulator."""

    def __init__(self, move_group: str, base_link: str, gripper: Gripper | None) -> None:
        """Initialize an interface for the manipulator's move group.

        :param move_group: Name of the move group corresponding to the manipulator
        :param base_link: Base link of the manipulator's kinematic chain (e.g., "body")
        :param gripper: End-effector of the manipulator (optional)
        """
        roscpp_initialize(sys.argv)
        self._move_group = MoveGroupCommander(move_group, wait_for_servers=30)
        self._base_frame = base_link
        self._move_group.set_pose_reference_frame(self._base_frame)

        self.gripper = gripper

        self.joint_names: tuple[str] = tuple(self._move_group.get_active_joints())

    def get_configuration(self) -> Configuration:
        """Retrieve the manipulator's current configuration."""
        joint_values = self._move_group.get_current_joint_values()
        assert len(self.joint_names) == len(joint_values)

        return dict(zip(self.joint_names, joint_values))

    def execute_motion_plan(self, target_ee_pose: Pose3D) -> bool:
        """Compute and execute a motion plan to bring the end-effector to the given pose.

        :param target_ee_pose: Target pose for the end-effector
        :return: True if the motion plan succeeded, otherwise False
        """
        target_pose_b_ee = TransformManager.convert_to_frame(target_ee_pose, self._base_frame)

        self._move_group.set_pose_target(pose_to_stamped_msg(target_pose_b_ee))
        success, robot_traj, planning_time_s, error_code = self._move_group.plan()

        outcome = "succeeded" if success else "failed"
        rospy.loginfo(f"Motion planning {outcome} after {planning_time_s} seconds.")
        if not success:
            rospy.loginfo(f"Motion planning error code: {error_code}.")
            return False

        self._move_group.clear_pose_targets()
        self._move_group.execute(robot_traj.joint_trajectory, wait=True)  # Blocks until done
        self._move_group.stop()  # Ensure there's no residual movement

        return True
