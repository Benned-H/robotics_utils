"""Define a class to interface with robot manipulators using MoveIt."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize
from trac_ik_python.trac_ik import IK

from robotics_utils.robots import Manipulator
from robotics_utils.ros import TransformManager, get_ros_param
from robotics_utils.ros.msg_conversion import trajectory_to_msg

if TYPE_CHECKING:
    from robotics_utils.kinematics import Configuration, Pose3D
    from robotics_utils.motion_planning import Trajectory
    from robotics_utils.robots.angular_gripper import AngularGripper


class MoveItManipulator(Manipulator):
    """A MoveIt-based interface for a robot manipulator."""

    def __init__(self, name: str, base_frame: str, gripper: AngularGripper) -> None:
        """Initialize the manipulator with its base frame and gripper.

        :param name: Name of the manipulator (used as its move group name)
        :param base_frame: Base frame of the manipulator's move group
        :param gripper: Interface for the end-effector of the manipulator
        """
        super().__init__(name, base_frame, gripper)
        move_group_name = self.name

        roscpp_initialize(sys.argv)
        self.move_group = MoveGroupCommander(move_group_name, wait_for_servers=30)
        self.move_group.set_pose_reference_frame(self.base_frame)

        self._ee_link = self.move_group.get_end_effector_link()

        # Read the robot's URDF from ROS params
        robot_urdf = get_ros_param("/robot_description", str)
        rospy.loginfo(f"[Manipulator {self.name}] Found robot description from ROS parameters.")

        self._ik_solver = IK(self.base_frame, self._ee_link, urdf_string=robot_urdf)

    @property
    def end_effector_link_name(self) -> str:
        """Retrieve the name of the manipulator's end-effector link."""
        return self._ee_link

    @property
    def joint_names(self) -> tuple[str]:
        """Get the names of the active joints in the manipulator.

        Verifies that MoveIt's move group and Trac-IK agree on the tuple of joint names.
        """
        moveit_joints = tuple(self.move_group.get_active_joints())
        trac_ik_joints = self._ik_solver.joint_names

        if moveit_joints != trac_ik_joints:
            rospy.logwarn(f"[Manipulator {self.name}] joints per MoveIt: {moveit_joints}")
            rospy.logwarn(f"[Manipulator {self.name}] joints per Trac-IK: {trac_ik_joints}")
            raise RuntimeError(f"MoveIt and Trac-IK disagree on manipulator {self.name}'s joints.")

        return moveit_joints

    @property
    def configuration(self) -> Configuration:
        """Retrieve the manipulator's current configuration."""
        joint_values = self.move_group.get_current_joint_values()

        assert len(self.joint_names) == len(joint_values)
        return dict(zip(self.joint_names, joint_values))

    def convert_to_base_frame(self, pose: Pose3D) -> Pose3D:
        """Convert the given pose into the manipulator's base frame."""
        return TransformManager.convert_to_frame(pose, self.base_frame)

    def execute_motion_plan(self, trajectory: Trajectory) -> None:
        """Execute the given trajectory using the manipulator."""
        trajectory_msg = trajectory_to_msg(trajectory)

        self.move_group.clear_pose_targets()
        self.move_group.execute(trajectory_msg, wait=True)  # Blocks until done
        self.move_group.stop()  # Ensure there's no residual movement

    def compute_ik(self, ee_target: Pose3D) -> Configuration | None:
        """Compute an inverse kinematics solution to place the end-effector at the given pose.

        :param ee_target: Target pose of the end-effector
        :return: Manipulator configuration solving the IK problem (else None)
        """
        target_b_ee = self.convert_to_base_frame(ee_target)  # End-effector w.r.t. base frame

        ik_solution = self._ik_solver.get_ik(
            self.joint_values,
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
