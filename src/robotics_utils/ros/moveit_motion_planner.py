"""Define a class to solve motion planning queries using MoveIt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import rospy
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

from robotics_utils.kinematics import Pose3D
from robotics_utils.ros.msg_conversion import pose_to_msg, pose_to_stamped_msg
from robotics_utils.ros.transform_manager import TransformManager as TFManager

if TYPE_CHECKING:
    from geometry_msgs.msg import Pose as PoseMsg

    from robotics_utils.motion_planning import MotionPlanningQuery
    from robotics_utils.ros.planning_scene_manager import PlanningSceneManager
    from robotics_utils.ros.robots import MoveItManipulator

MoveItResult = Tuple[bool, RobotTrajectory, float, MoveItErrorCodes]
"""Boolean success, trajectory message, planning time (s), and error codes.

Reference: https://tinyurl.com/moveit-noetic-plan
"""


class MoveItMotionPlanner:
    """An interface for computing motion plans using MoveIt."""

    def __init__(self, manipulator: MoveItManipulator, psm: PlanningSceneManager) -> None:
        """Initialize the motion planner with the manipulator it will plan for."""
        self._manipulator = manipulator
        self._planning_scene = psm

    def compute_motion_plan(self, query: MotionPlanningQuery) -> RobotTrajectory | None:
        """Compute a motion plan (i.e., trajectory) for the given planning query.

        :param query: Specifies an end-effector target and (optionally) objects to ignore
        :return: Message containing the planned trajectory, or None if no plan is found
        """
        if not self._planning_scene.apply_query_ignores(query):
            error_msg = f"Unable to ignore collisions for motion planning: {query}"
            rospy.logerr(error_msg)
            raise RuntimeError(error_msg)

        if isinstance(query.ee_target, Pose3D):
            target_b_ee = TFManager.convert_to_frame(query.ee_target, self._manipulator.base_frame)
            ee_target_msg = pose_to_stamped_msg(target_b_ee)
            self._manipulator.move_group.set_pose_target(ee_target_msg)
        elif isinstance(query.ee_target, dict):  # Configuration maps joint names to their values
            self._manipulator.move_group.set_joint_value_target(query.ee_target)
        else:
            raise TypeError(f"Unrecognized end-effector target type: {type(query.ee_target)}.")

        self._manipulator.move_group.set_start_state_to_current_state()
        result: MoveItResult = self._manipulator.move_group.plan()
        success, robot_traj, planning_time_s, error_code = result

        outcome_desc = "succeeded" if success else "failed"
        rospy.loginfo(f"Motion planning {outcome_desc} after {planning_time_s:.4f} seconds.")

        if not success:
            rospy.logerr(f"Motion planning error code: {error_code}.")

        # Reset any modifications to the planning scene before exiting
        if not self._planning_scene.revert_query_ignores(query):
            error_msg = f"Unable to revert collision ignores for motion planning: {query}"
            rospy.logerr(error_msg)
            raise RuntimeError(error_msg)

        return robot_traj if success else None

    def compute_cartesian_plan(
        self,
        waypoints: list[Pose3D],
        ee_step_m: float = 0.01,
        *,
        avoid_collisions: bool = True,
    ) -> tuple[RobotTrajectory | None, float]:
        """Compute a Cartesian path through the given waypoints.

        :param waypoints: List of target end-effector poses to follow
        :param ee_step_m: Maximum distance (meters) between consecutive planned configurations
        :param avoid_collisions: Whether to check for collisions during planning (defaults to True)
        :return: Tuple containing:
            Robot trajectory if planning succeeded, else None
            Fraction of the path successfully computed (0.0 to 1.0)
        """
        if not waypoints:
            return None, 0.0

        # Convert waypoints to base frame and then to messages
        waypoint_msgs: list[PoseMsg] = []
        for wp in waypoints:
            pose_b_ee = TFManager.convert_to_frame(wp, self._manipulator.base_frame)
            waypoint_msgs.append(pose_to_msg(pose_b_ee))

        self._manipulator.move_group.set_start_state_to_current_state()
        plan, fraction = self._manipulator.move_group.compute_cartesian_path(
            waypoint_msgs,
            eef_step=ee_step_m,
            avoid_collisions=avoid_collisions,
        )

        return (plan if fraction > 0.0 else None, fraction)
