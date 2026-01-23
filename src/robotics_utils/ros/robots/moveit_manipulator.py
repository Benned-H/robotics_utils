"""Define a class to interface with robot manipulators using MoveIt."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize
from trac_ik_python.trac_ik import IK

from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.robots import Manipulator
from robotics_utils.ros import PlanningSceneManager, TransformManager, get_ros_param
from robotics_utils.ros.moveit_motion_planner import MoveItMotionPlanner
from robotics_utils.ros.msg_conversion import trajectory_to_msg
from robotics_utils.skills import Outcome

if TYPE_CHECKING:
    from moveit_msgs.msg import RobotTrajectory

    from robotics_utils.kinematics import Configuration
    from robotics_utils.motion_planning import Trajectory
    from robotics_utils.robots.angular_gripper import AngularGripper
    from robotics_utils.spatial import Pose3D


class MoveItManipulator(Manipulator):
    """A MoveIt-based interface for a robot manipulator."""

    def __init__(
        self,
        name: str,
        robot_name: str,
        base_frame: str,
        planning_frame: str,
        gripper: AngularGripper | None,
    ) -> None:
        """Initialize the manipulator with names, frames, and (optionally) a gripper.

        :param name: Name of the manipulator (used as its move group name)
        :param robot_name: Name of the robot the manipulator belongs to
        :param base_frame: Base frame of the manipulator's move group
        :param planning_frame: Frame used when computing motion plans
        :param gripper: Interface for the end-effector of the manipulator
        """
        super().__init__(name, gripper)
        self.robot_name = robot_name
        self.base_frame = base_frame

        roscpp_initialize(sys.argv)
        self.move_group = MoveGroupCommander(self.name, wait_for_servers=30)
        self.move_group.set_pose_reference_frame(planning_frame)

        self.planner = MoveItMotionPlanner(self.move_group, planning_frame=planning_frame)
        self.planning_scene = PlanningSceneManager(planning_frame=planning_frame)

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
        """Find the current pose of the end-effector w.r.t. the base frame.

        :param timeout_s: Duration (seconds) after which the pose lookup times out
        :return: End-effector pose in the manipulator's base frame, or None if /tf lookup fails
        """
        pose_b_ee = None  # End-effector (ee) w.r.t. base frame (b)
        end_time_s = time.time() + timeout_s
        while pose_b_ee is None and time.time() < end_time_s:
            pose_b_ee = TransformManager.lookup_transform(
                child_frame=self.ee_link_name,
                parent_frame=self.base_frame,
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

    def go_to(self, target: Configuration | Pose3D, max_retries: int = 3) -> bool:
        """Plan and execute a motion to bring the manipulator to an end-effector target.

        :param target: Target joint configuration or end-effector pose
        :param max_retries: Maximum number of planning attempts (defaults to 3)
        :return: True if motion was successfully planned and executed, False otherwise
        """
        query = MotionPlanningQuery(ee_target=target)

        for attempt in range(max_retries):
            plan_msg = self.planner.compute_motion_plan(query)
            if plan_msg is not None:
                return self.execute_trajectory_msg(plan_msg)
            rospy.logwarn(f"Motion planning attempt {attempt + 1}/{max_retries} failed.")

        rospy.logerr(f"Failed to plan motion to configuration after {max_retries} attempts.")
        return False

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

    def grasp(self, object_name: str) -> Outcome:
        """Grasp the named object using the manipulator's gripper (and close the gripper).

        :return: Boolean success of the grasp and an outcome message
        """
        if self.gripper is None:
            return Outcome(False, f"Cannot grasp '{object_name}' because gripper is None.")

        if not self.gripper.close():
            return Outcome(False, f"Failed to close gripper when grasping '{object_name}'.")

        # Find the current pose of the grasped object w.r.t. the end-effector
        pose_ee_o = TransformManager.lookup_transform(object_name, self.ee_link_name)
        if pose_ee_o is None:
            return Outcome(False, f"Failed to look up pose when grasping '{object_name}'.")

        TransformManager.broadcast_transform(object_name, pose_ee_o)  # Update object's pose in TF

        success = self.planning_scene.attach_object(
            obj_name=object_name,
            robot_name=self.robot_name,
            ee_link_name=self.ee_link_name,
            touch_links=self.gripper.link_names,
        )  # Attach the object to the robot's end-effector in MoveIt's planning scene

        message = (
            f"Successfully grasped '{object_name}'."
            if success
            else f"Failed to grasp '{object_name}' because the planning scene was not updated."
        )
        return Outcome(success=success, message=message)

    def release(self, object_name: str) -> Outcome:
        """Release the named object using the manipulator's gripper.

        :return: Boolean success of the release and an outcome message
        """
        if self.gripper is None:
            return Outcome(False, f"Cannot release '{object_name}' because gripper is None.")

        if not self.gripper.open():
            return Outcome(False, f"Failed to open gripper when releasing '{object_name}'.")

        success = self.planning_scene.detach_object(
            obj_name=object_name,
            robot_name=self.robot_name,
            ee_link_name=self.ee_link_name,
        )

        message = (
            f"Successfully released '{object_name}'."
            if success
            else f"Failed to release '{object_name}' because the planning scene was not updated."
        )
        return Outcome(success=success, message=message)
