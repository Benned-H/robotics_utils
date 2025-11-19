"""Define a class to interface with robot manipulators using MoveIt."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize
from trac_ik_python.trac_ik import IK

from robotics_utils.robots import Manipulator
from robotics_utils.ros import TransformManager, get_ros_param
from robotics_utils.ros.msg_conversion import trajectory_to_msg

if TYPE_CHECKING:
    from moveit_msgs.msg import RobotTrajectory

    from robotics_utils.kinematics import Configuration, Pose3D
    from robotics_utils.motion_planning import Trajectory
    from robotics_utils.robots.angular_gripper import AngularGripper


class MoveItManipulator(Manipulator):
    """A MoveIt-based interface for a robot manipulator."""

    def __init__(self, name: str, base_frame: str, gripper: AngularGripper | None) -> None:
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

        self._ee_link: str = self.move_group.get_end_effector_link()

        # Read the robot's URDF from ROS params
        robot_urdf = get_ros_param("/robot_description", str)
        rospy.loginfo(f"[Manipulator {self.name}] Found robot description from ROS parameters.")

        self._ik_solver = IK(self.base_frame, self._ee_link, urdf_string=robot_urdf)

    @property
    def ee_link_name(self) -> str:
        """Retrieve the name of the manipulator's end-effector link."""
        return self._ee_link

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Retrieve the names of the joints in the manipulator in their canonical order.

        Verifies that MoveIt's move group and Trac-IK agree on the tuple of joint names.
        """
        moveit_joints = tuple(self.move_group.get_active_joints())
        trac_ik_joints = self._ik_solver.joint_names

        if moveit_joints != trac_ik_joints:
            rospy.logwarn(f"[Manipulator {self.name}] joints per MoveIt: {moveit_joints}")
            rospy.logwarn(f"[Manipulator {self.name}] joints per Trac-IK: {trac_ik_joints}")
            raise RuntimeError(f"MoveIt and Trac-IK disagree on joints in '{self.name}'.")

        return moveit_joints

    @property
    def configuration(self) -> Configuration:
        """Retrieve the manipulator's current configuration."""
        joint_values = self.move_group.get_current_joint_values()

        assert len(self.joint_names) == len(joint_values)
        return dict(zip(self.joint_names, joint_values))

    def get_current_ee_pose(self, timeout_s: float = 3.0) -> Pose3D | None:
        """Find the current pose of the end-effector pose w.r.t. the base frame.

        :param timeout_s: Duration (seconds) after which the pose lookup times out
        :return: End-effector pose in the manipulator's base frame, or None if /tf lookup fails
        """
        pose_b_ee = None  # End-effector (ee) w.r.t. base frame (b)
        end_time_s = time.time() + timeout_s
        while pose_b_ee is None and time.time() < end_time_s:
            pose_b_ee = TransformManager.lookup_transform(
                source_frame=self.ee_link_name,
                target_frame=self.base_frame,
                timeout_s=0.2,
            )

        if pose_b_ee is None:
            rospy.logwarn(f"Unable to look up end-effector pose for manipulator '{self.name}'.")

        return pose_b_ee

    def execute_motion_plan(self, trajectory: Trajectory) -> bool:
        """Execute the given trajectory on the manipulator using MoveIt.

        :return: True if execution succeeded, False otherwise
        """
        trajectory_msg = trajectory_to_msg(trajectory)

        self.move_group.clear_pose_targets()
        success = self.move_group.execute(trajectory_msg, wait=True)
        self.move_group.stop()  # Ensure there's no residual movement
        return success

    def execute_trajectory_msg(self, traj_msg: RobotTrajectory, *, wait: bool = True) -> bool:
        """Execute the given moveit_msgs/RobotTrajectory message on the manipulator.

        :param traj_msg: RobotTrajectory message to execute
        :param wait: Whether to block until execution completes (defaults to True)
        :return: True if execution succeeded, False otherwise
        """
        self.move_group.clear_pose_targets()
        success = self.move_group.execute(traj_msg, wait=wait)
        self.move_group.stop()  # Ensure there's no residual movement
        return success

    def compute_ik(self, ee_target: Pose3D) -> Configuration | None:
        """Compute an inverse kinematics solution to place the end-effector at the given pose.

        Reference for TRAC-IK: https://bitbucket.org/traclabs/trac_ik/src/rolling/trac_ik_python/

        :param ee_target: Target pose of the end-effector
        :return: Manipulator configuration solving the IK problem (else None)
        """
        target_b_ee = TransformManager.convert_to_frame(ee_target, self.base_frame)

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
