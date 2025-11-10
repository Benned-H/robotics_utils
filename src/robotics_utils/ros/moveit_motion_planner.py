"""Define a class to solve motion planning queries using MoveIt."""

from typing import Tuple

from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

MoveItResult = Tuple[bool, RobotTrajectory, float, MoveItErrorCodes]
"""Boolean success, trajectory message, planning time (s), and error codes.

Reference: https://docs.ros.org/en/noetic/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html#a79a475263bffd96978c87488a2bf7c98
"""


# class MoveItMotionPlanner:
#     """An interface to compute motion plans using MoveIt."""

#     def __init__(
#         self,
#         manipulator: MoveItManipulator,
#         planning_scene: PlanningSceneManager,
#     ) -> None:
#         """Initialize the motion planner with the manipulator it will plan for."""
#         self._manipulator = manipulator
#         self._planning_scene = planning_scene

#     def compute_motion_plan(self, query: MotionPlanningQuery) -> Trajectory | None:
#         """Compute a motion plan (i.e., trajectory) for the given planning query.

#         :param query: Specifies an end-effector target and (optionally) objects to ignore
#         :return: Planned trajectory, or None upon failure
#         """
#         if query.ignore_all_collisions:
#             self._planning_scene.hide_all_objects()
#         elif query.ignored_objects:
#             for object_to_ignore in query.ignored_objects:
#                 if not self._planning_scene.hide_object(object_to_ignore):
#                     rospy.loginfo(f"Failed to hide object '{object_to_ignore}' in planning scene.")

#         if isinstance(query.ee_target, Pose3D):
#             target_b_ee = TFManager.convert_to_frame(query.ee_target, self._manipulator.base_frame)
#             ee_target_msg = pose_to_stamped_msg(target_b_ee)
#             self._manipulator.move_group.set_pose_target(ee_target_msg)
#         elif isinstance(query.ee_target, dict):  # Configuration maps joint names to their positions
#             self._manipulator.move_group.set_joint_value_target(query.ee_target)
#         else:
#             raise TypeError(f"Unrecognized end-effector target type: {type(query.ee_target)}.")

#         result: MoveItResult = self._manipulator.move_group.plan()
#         success, robot_traj, planning_time_s, error_code = result

#         outcome_desc = "succeeded" if success else "failed"
#         rospy.loginfo(f"Motion planning {outcome_desc} after {planning_time_s} seconds.")

#         if not success:
#             rospy.loginfo(f"Motion planning error code: {error_code}.")

#         # Reset any planning scene modifications before exiting
#         if query.ignore_all_collisions:
#             self._planning_scene.unhide_all_objects()
#         elif query.ignored_objects:
#             for obj_name in query.ignored_objects:
#                 if not self._planning_scene.unhide_object(obj_name):
#                     rospy.loginfo(f"Failed to add object '{obj_name}' back to the planning scene.")

#         return trajectory_from_msg(robot_traj.joint_trajectory) if success else None
