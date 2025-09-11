"""Define a class to replay relative trajectories loaded from file."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import rospy
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics import Configuration, Pose3D
from robotics_utils.math.distances import euclidean_distance_3d_m
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros.moveit_motion_planner import MoveItMotionPlanner
from robotics_utils.ros.msg_conversion import pose_to_msg
from robotics_utils.ros.planning_scene_manager import PlanningSceneManager
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.ros.transform_manager import TransformManager

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class IKSolution:
    """An IK solution specifies an arm configuration to reach a target end-effector pose."""

    q: Configuration
    pose: Pose3D


@dataclass(frozen=True)
class RelativeTrajectoryConfig:
    """Configures the trajectory replayer when loading a relative trajectory from file."""

    ee_frame: str
    body_frame: str
    move_group_name: str
    ee_step_resolution_m: float = 0.01  # Resolution (meters) between computed configurations
    pose_lookup_timeout_s: float = 3.0  # Duration (seconds) to wait for pose lookup retries
    required_fraction: float = 0.95  # Proportion of the trajectory Cartesian planning must follow


class TrajectoryReplayer:
    """Play back relative trajectories loaded from file."""

    def __init__(self, config: RelativeTrajectoryConfig) -> None:
        """Configure the trajectory replayer with the given parameters."""
        TransformManager.init_node()  # Ensure that TransformManager is listening to /tf

        self.config = config

        roscpp_initialize(sys.argv)  # Needed for MoveIt
        self.move_group = MoveGroupCommander(config.move_group_name, wait_for_servers=30)
        self.move_group.set_pose_reference_frame(config.body_frame)

        self.display_trajectory_pub = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=20,
        )

        self.robot = RobotCommander()
        self.spot_gripper = ROSAngularGripper(
            limits=GripperAngleLimits(open_rad=-1.5707, closed_rad=0.0),
            grasping_group="gripper",
            action_name="gripper_controller/gripper_action",
        )
        self.spot_arm = MoveItManipulator(name="arm", base_frame="body", gripper=self.spot_gripper)
        self.moveit_planner = MoveItMotionPlanner(self.spot_arm, PlanningSceneManager("body"))

    def get_end_effector_pose(self) -> Pose3D | None:
        """Find the current end-effector pose w.r.t. the body frame (None if /tf lookup fails)."""
        ee_frame = self.config.ee_frame
        body_frame = self.config.body_frame

        pose_b_ee = None  # End-effector w.r.t. body frame
        end_time_s = time.time() + self.config.pose_lookup_timeout_s
        while pose_b_ee is None and time.time() < end_time_s:
            pose_b_ee = TransformManager.lookup_transform(ee_frame, body_frame)

        if pose_b_ee is None:
            rospy.logwarn(f"Unable to look up transform from '{body_frame}' to '{ee_frame}'.")

        return pose_b_ee

    def load_relative_trajectory(self, yaml_path: Path) -> list[Pose3D]:
        """Load a relative trajectory from the given YAML file.

        :param yaml_path: Path to a YAML file to be imported
        :return: Imported relative trajectory as a list of 3D poses
        """
        poses_data: list[dict] = load_yaml_data(yaml_path, {"transforms"})["transforms"]

        # These poses track the end-effector frame relative to its initial pose
        relative_poses = [Pose3D.from_yaml_data(pose_dict) for pose_dict in poses_data]

        curr_pose_b_ee = self.get_end_effector_pose()
        if curr_pose_b_ee is None:
            raise RuntimeError("Unable to load relative trajectory due to failed frame lookup")

        # Adjust the relative poses so that they're relative to the current body frame
        return [curr_pose_b_ee @ rel_pose_ee for rel_pose_ee in relative_poses]

    def compute_cartesian_plan(self, poses: list[Pose3D]) -> RobotTrajectory | None:
        """Compute a Cartesian plan along the given list of end-effector target poses.

        :param poses: List of target end-effector poses
        :return: Resulting trajectory, or None if the waypoints could not be followed
        """
        waypoints = [pose_to_msg(p) for p in poses]

        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step=self.config.ee_step_resolution_m,
            avoid_collisions=False,
        )

        # Display the generated trajectory in RViz
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory = [plan]
        self.display_trajectory_pub.publish(display_trajectory)

        if fraction < self.config.required_fraction:
            rospy.logwarn(f"MoveIt's plan followed {fraction * 100.0:.2f}% of the trajectory")
            return None

        return plan

    def compute_ik_sequence(self, poses: list[Pose3D]) -> list[IKSolution] | None:
        """Compute a sequence of IK solutions to recreate the given poses."""
        threshold_distance_m = 0.03

        ik_solutions: list[IKSolution] = []
        skipped = 0
        failed = 0
        for i, pose_i in enumerate(poses):
            rospy.loginfo(f"Computing IK solution {i + 1}/{len(poses)}...")

            # Skip this pose if it's within 3 cm of the most recent pose
            if (
                ik_solutions
                and euclidean_distance_3d_m(ik_solutions[-1].pose, pose_i, change_frames=False)
                < threshold_distance_m
            ):
                rospy.loginfo(f"Skipping pose {i}; too close to last solved pose.")
                skipped += 1
                continue

            q_i = self.spot_arm.compute_ik(pose_i)
            if q_i is None:
                rospy.loginfo(f"Could not compute IK solution for pose {i}: {pose_i}.")
                failed += 1
            else:
                ik_solutions.append(IKSolution(q_i, pose_i))

        rospy.loginfo(
            f"Found {len(ik_solutions)} IK solutions, skipped {skipped}, "
            f"and failed {failed} out of {len(poses)} target poses.",
        )

        return ik_solutions

    def go_to(self, config: Configuration) -> None:
        """Bring Spot's arm to the given configuration."""
        query = MotionPlanningQuery(config)
        plan = self.moveit_planner.compute_motion_plan(query)
        if plan is not None:
            self.spot_arm.execute_motion_plan(plan)
