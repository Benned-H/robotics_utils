"""Define a class to replay relative trajectories loaded from file."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING

import rospy
from moveit_commander import RobotCommander
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics import Configuration, Pose3D
from robotics_utils.math.distances import angle_between_quaternions_deg, euclidean_distance_3d_m
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.ros.moveit_motion_planner import MoveItMotionPlanner
from robotics_utils.ros.planning_scene_manager import PlanningSceneManager
from robotics_utils.ros.transform_manager import TransformManager

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.ros.robots import MoveItManipulator


@dataclass(frozen=True)
class RelativeTrajectoryConfig:
    """Configures the trajectory replayer when loading a relative trajectory from file."""

    min_pose_diff_m: float = 0.02
    """Minimum distance (meters) between filtered consecutive poses."""

    min_pose_diff_deg: float = 5.0
    """Minimum absolute angle (degrees) between filtered consecutive poses."""

    plan_ee_step_m: float = 0.01
    """Maximum distance (meters) between consecutive configs in the Cartesian plan."""


class TrajectoryPlayback:
    """Play back relative trajectories loaded from file on the given manipulator."""

    def __init__(self, config: RelativeTrajectoryConfig, manipulator: MoveItManipulator) -> None:
        """Configure the trajectory replayer with the given parameters."""
        TransformManager.init_node("trajectory_playback")  # Ensure listening to /tf
        self.config = config

        self.robot = RobotCommander()
        self.manipulator = manipulator

        self.moveit_planner = MoveItMotionPlanner(
            self.manipulator,
            PlanningSceneManager(move_group_name=self.manipulator.name),
        )

        self.display_trajectory_pub = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=20,
        )

    def load_relative_trajectory(self, yaml_path: Path) -> list[Pose3D]:
        """Load a relative trajectory from the given YAML file.

        :param yaml_path: Path to a YAML file to be imported
        :return: Imported relative trajectory as a list of 3D poses
        """
        yaml_data = load_yaml_data(yaml_path, required_keys={"transforms"})
        poses_data: list[dict] = yaml_data["transforms"]

        # These poses track the end-effector frame relative to its initial pose
        relative_poses = [Pose3D.from_yaml_data(pose_dict) for pose_dict in poses_data]

        curr_pose_b_ee = self.manipulator.get_current_ee_pose()
        if curr_pose_b_ee is None:
            raise RuntimeError("Unable to load relative trajectory due to failed frame lookup.")

        # Adjust the relative poses so that they're relative to the current body frame
        return [curr_pose_b_ee @ rel_pose_ee for rel_pose_ee in relative_poses]

    def _filter_poses(self, poses: list[Pose3D]) -> list[Pose3D]:
        """Filter the waypoints in the given trajectory based on distance and angle.

        :param poses: Sequence of end-effector target poses to be filtered
        :return: Thinned (per consecutive distance and angle proximity) sequence of poses
        """
        kept_poses: list[Pose3D] = []
        for pose in poses:
            if not kept_poses:
                kept_poses.append(pose)
                continue

            if pose.ref_frame != kept_poses[-1].ref_frame:
                other_rf = kept_poses[-1].ref_frame
                raise ValueError(f"Reference frames differ: {pose.ref_frame} vs {other_rf}.")

            diff_m = euclidean_distance_3d_m(pose, kept_poses[-1], change_frames=False)
            diff_deg = angle_between_quaternions_deg(pose.orientation, kept_poses[-1].orientation)

            if diff_m >= self.config.min_pose_diff_m or diff_deg >= self.config.min_pose_diff_deg:
                kept_poses.append(pose)

        return kept_poses

    def _compute_cartesian_prefix(
        self,
        waypoints: list[Pose3D],
    ) -> tuple[RobotTrajectory | None, int]:
        """Compute a Cartesian plan prefix for the given target end-effector poses.

        :param waypoints: List of target end-effector poses
        :return: Tuple containing:
            Robot trajectory if planning succeeded, else None
            Number of poses in the given sequence covered by the found plan
        """
        if not waypoints:
            return None, 0

        plan, fraction = self.moveit_planner.compute_cartesian_plan(
            waypoints,
            ee_step_m=self.config.plan_ee_step_m,
            avoid_collisions=False,
        )

        if plan is not None:
            self.visualize_plan(plan)
        rospy.logwarn(f"MoveIt's plan followed {fraction * 100.0:.2f}% of the trajectory.")

        n = floor(fraction * len(waypoints))
        return (plan if n > 0 else None, n)

    def _bridge_ik(self, pose: Pose3D) -> bool:
        """Attempt to reach the given end-effector pose using an IK solution."""
        q = self.manipulator.compute_ik(pose)
        if q is None:
            return False

        # Plan to the IK solution
        query = MotionPlanningQuery(q)
        plan = self.moveit_planner.compute_motion_plan(query)
        if plan is None:
            return False

        # Execute the plan
        self.manipulator.execute_motion_plan(plan)
        return True

    def execute_hybrid_cartesian_sequence(self, poses: list[Pose3D]) -> bool:
        """Compute and execute a Cartesian trajectory through the given end-effector poses.

        Uses a hybrid approach that combines Cartesian planning with IK-based bridging to
        handle singularities and other planning failures.
        """
        filtered_poses = self._filter_poses(poses)

        next_idx = 0  # Index of the earliest unreached waypoint
        while next_idx < len(filtered_poses):
            waypoints_left = filtered_poses[next_idx:]
            plan, n = self._compute_cartesian_prefix(waypoints_left)

            if plan is not None:  # Visualize and execute the prefix if successful
                self.visualize_plan(plan)

                if not self.manipulator.execute_trajectory_msg(plan):
                    rospy.logwarn(f"Failed to execute trajectory starting at waypoint {next_idx}.")
                    return False

                next_idx += n
                continue  # Move to the next unreached waypoint

            # Otherwise, we failed to find a plan, so attempt to bridge using IK
            next_pose = filtered_poses[next_idx]
            if not self._bridge_ik(next_pose):
                rospy.logwarn(f"Failed to bridge using IK solution for pose {next_idx}.")
                return False
            next_idx += 1

        return True

    def visualize_plan(self, plan: RobotTrajectory) -> None:
        """Visualize the given trajectory in RViz."""
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory = [plan]
        self.display_trajectory_pub.publish(display_trajectory)

    def go_to(self, config: Configuration, max_retries: int = 3) -> bool:
        """Plan and execute a motion to bring the manipulator to the given configuration.

        :param config: Target joint configuration
        :param max_retries: Maximum number of planning attempts
        :return: True if motion was successfully planned and executed, False otherwise
        """
        query = MotionPlanningQuery(config)

        for attempt in range(max_retries):
            plan = self.moveit_planner.compute_motion_plan(query)
            if plan is not None:
                self.manipulator.execute_motion_plan(plan)
                return True
            rospy.logwarn(f"Motion planning attempt {attempt + 1}/{max_retries} failed.")

        rospy.logerr(f"Failed to plan motion to configuration after {max_retries} attempts.")
        return False
