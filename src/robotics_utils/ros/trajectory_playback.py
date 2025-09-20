"""Define a class to replay relative trajectories loaded from file."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING

import rospy
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics import Configuration, Pose3D
from robotics_utils.math.distances import angle_between_quaternions_deg, euclidean_distance_3d_m
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros.moveit_motion_planner import MoveItMotionPlanner, MoveItResult
from robotics_utils.ros.msg_conversion import pose_to_msg
from robotics_utils.ros.planning_scene_manager import PlanningSceneManager
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.ros.transform_manager import TransformManager

if TYPE_CHECKING:
    from pathlib import Path

    from geometry_msgs.msg import Pose as PoseMsg


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


class TrajectoryPlayback:
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
        self.moveit_planner = MoveItMotionPlanner(
            self.spot_arm,
            PlanningSceneManager(move_group_name=self.spot_arm.name),
        )

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

    def _filter_poses(self, poses: list[Pose3D], min_m: float, min_deg: float) -> list[Pose3D]:
        """Filter the waypoints in the given trajectory based on distance and angle.

        :param poses: Sequence of end-effector target poses to be filtered
        :param min_m: Minimum distance (meters) between filtered consecutive poses
        :param min_deg: Minimum angle (degrees) between filtered consecutive poses
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

            if diff_m >= min_m or diff_deg >= min_deg:
                kept_poses.append(pose)

        return kept_poses

    def _compute_cartesian_prefix(
        self,
        waypoint_msgs: list[PoseMsg],
        eef_step_m: float,
    ) -> tuple[RobotTrajectory | None, int]:
        """Compute a Cartesian plan prefix for the given target end-effector poses.

        :param waypoint_msgs: List of ROS messages specifying target end-effector poses
        :param eef_step_m: Maximum distance (meters) between consecutive configs in the plan
        :return: Tuple containing:
            Robot trajectory if planning succeeded, else None
            Number of poses in the given sequence covered by the found plan
        """
        if not waypoint_msgs:
            return None, 0

        self.move_group.set_start_state_to_current_state()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoint_msgs,
            eef_step=eef_step_m,
            avoid_collisions=False,
        )
        n = floor(fraction * len(waypoint_msgs))
        return (plan if n > 0 else None, n)

    def _bridge_ik(self, pose: Pose3D) -> bool:
        """Attempt to reach the given end-effector pose using an IK solution."""
        q = self.spot_arm.compute_ik(pose)
        if q is None:
            return False

        self.move_group.set_start_state_to_current_state()
        self.move_group.set_joint_value_target(q)

        result: MoveItResult = self.move_group.plan()
        success, robot_traj, planning_time_s, error_code = result
        if not success or not robot_traj.joint_trajectory.points:
            return False

        return self.move_group.execute(robot_traj, wait=True)

    def execute_hybrid_cartesian_sequence(self, poses: list[Pose3D]) -> bool:
        """Compute and execute a Cartesian trajectory through the given end-effector poses."""
        filtered_poses = self._filter_poses(poses, min_m=0.015, min_deg=5)
        waypoint_msgs = [pose_to_msg(p) for p in filtered_poses]

        next_idx = 0  # Index of the earliest unreached waypoint
        while next_idx < len(waypoint_msgs):
            waypoints_left = waypoint_msgs[next_idx:]
            plan, n = self._compute_cartesian_prefix(waypoints_left, eef_step_m=0.015)

            if plan is not None:  # Visualize the prefix if successful
                self.visualize_plan(plan)

                if not self.move_group.execute(plan, wait=True):
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

        self.visualize_plan(plan)

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
