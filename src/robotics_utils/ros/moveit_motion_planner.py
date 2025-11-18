"""Define a class to solve motion planning queries using MoveIt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import rospy
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

from robotics_utils.kinematics import Pose3D
from robotics_utils.ros.msg_conversion import pose_to_stamped_msg, trajectory_from_msg
from robotics_utils.ros.transform_manager import TransformManager as TFManager

if TYPE_CHECKING:
    from robotics_utils.motion_planning import MotionPlanningQuery, Trajectory
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

    def compute_motion_plan(self, query: MotionPlanningQuery) -> Trajectory | None:
        """Compute a motion plan (i.e., trajectory) for the given planning query.

        :param query: Specifies an end-effector target and (optionally) objects to ignore
        :return: Planned trajectory, or None if no plan is found
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

        return trajectory_from_msg(robot_traj.joint_trajectory) if success else None
