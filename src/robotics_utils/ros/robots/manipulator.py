"""Define a class to interface with robot manipulators using MoveIt."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize
from trac_ik_python.trac_ik import IK

from robotics_utils.kinematics import Configuration, Pose3D
from robotics_utils.ros.msg_conversion import pose_to_stamped_msg
from robotics_utils.ros.params import get_ros_param
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.skills.skill_structs import GraspPose

if TYPE_CHECKING:
    from robotics_utils.ros.robots.gripper import Gripper


class Manipulator:
    """A MoveIt-based interface for a robot manipulator."""

    def __init__(self, move_group: str, base_link: str, gripper: Gripper | None) -> None:
        """Initialize an interface for the manipulator's move group.

        :param move_group: Name of the move group corresponding to the manipulator
        :param base_link: Base link of the manipulator's kinematic chain (e.g., "body")
        :param gripper: End-effector of the manipulator (optional)
        """
        self.name = move_group

        roscpp_initialize(sys.argv)
        self._move_group = MoveGroupCommander(move_group, wait_for_servers=30)
        self._base_frame = base_link
        self._move_group.set_pose_reference_frame(self._base_frame)

        self.gripper = gripper

        self.ee_link: str = self._move_group.get_end_effector_link()

        # Read the robot's URDF from ROS params
        robot_urdf = get_ros_param("/robot_description", str)
        rospy.loginfo(f"[Manipulator {self.name}] Found robot description from ROS parameters.")

        self.ik_solver = IK(base_link, self.ee_link, urdf_string=robot_urdf)

    @property
    def joint_names(self) -> tuple[str]:
        """Get the names of the active joints in the manipulator.

        Verifies that MoveIt's move group and Trac-IK agree on the tuple of joint names.
        """
        moveit_joints = tuple(self._move_group.get_active_joints())
        trac_ik_joints = self.ik_solver.joint_names

        if moveit_joints != trac_ik_joints:
            rospy.loginfo(f"[Manipulator {self.name}] joints per MoveIt: {moveit_joints}")
            rospy.loginfo(f"[Manipulator {self.name}] joints per Trac-IK: {trac_ik_joints}")
            raise RuntimeError(f"MoveIt and Trac-IK disagree on manipulator {self.name}'s joints.")

        return moveit_joints

    def get_configuration(self) -> Configuration:
        """Retrieve the manipulator's current configuration."""
        joint_values = self._move_group.get_current_joint_values()

        assert len(self.joint_names) == len(joint_values)
        return dict(zip(self.joint_names, joint_values))

    def get_joint_values(self) -> list[float]:
        """Retrieve the current joint values of the manipulator."""
        return list(self.get_configuration().values())

    def convert_to_base_frame(self, pose: Pose3D) -> Pose3D:
        """Convert the given pose into the manipulator's base frame."""
        return TransformManager.convert_to_frame(pose, self._base_frame)

    def feasible_motion_plan_exists(self, ee_target: Pose3D | GraspPose) -> bool:
        """Check whether a feasible motion plan exists to reach a target end-effector pose.

        :param ee_target: Target pose of the end-effector
        :return: True if a motion plan exists, otherwise False
        """
        # If collisions should be ignored, feasibility only requires that an IK solution exists
        if isinstance(ee_target, GraspPose) and ee_target.ignore_collisions:
            ik_solution = self.compute_ik(ee_target.pose_o_g)
            return ik_solution is not None

        # Otherwise, attempt to compute a motion plan
        if isinstance(ee_target, Pose3D):
            target_pose = ee_target
        elif isinstance(ee_target, GraspPose):
            target_pose = ee_target.pose_o_g
        else:
            raise TypeError(f"Unrecognized end-effector target type: {type(ee_target)}")

        target_pose_b_ee = self.convert_to_base_frame(target_pose)

        self._move_group.set_pose_target(pose_to_stamped_msg(target_pose_b_ee))
        success, robot_traj, planning_time_s, error_code = self._move_group.plan()
        return success

    def execute_motion_plan(self, target_ee_pose: Pose3D) -> bool:
        """Compute and execute a motion plan to bring the end-effector to the given pose.

        :param target_ee_pose: Target pose for the end-effector
        :return: True if the motion plan succeeded, otherwise False
        """
        target_pose_b_ee = self.convert_to_base_frame(target_ee_pose)

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

    def compute_ik(self, ee_target: Pose3D) -> Configuration | None:
        """Compute an inverse kinematics solution to place the end-effector at the given pose.

        :param ee_target: Target pose of the end-effector
        :return: Manipulator configuration solving the IK problem (else None)
        """
        target_b_ee = self.convert_to_base_frame(ee_target)  # End-effector w.r.t. base frame

        ik_solution = self.ik_solver.get_ik(
            self.get_joint_values(),
            target_b_ee.position.x,
            target_b_ee.position.y,
            target_b_ee.position.z,
            target_b_ee.orientation.x,
            target_b_ee.orientation.y,
            target_b_ee.orientation.z,
            target_b_ee.orientation.w,
        )

        if ik_solution is None:
            return None

        return dict(zip(self.joint_names, list(ik_solution)))
