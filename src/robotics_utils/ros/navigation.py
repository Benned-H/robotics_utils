"""Define ROS-dependent utilities for managing robot navigation."""

from __future__ import annotations

import rospy
from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse

from robotics_utils.kinematics import Pose2D, Pose3D
from robotics_utils.math.polyline import Polyline2D
from robotics_utils.ros.msg_conversion import pose_from_msg, pose_to_stamped_msg
from robotics_utils.ros.services import ServiceCaller


def compute_navigation_plan(
    start: Pose2D,
    goal: Pose2D,
    tolerance_m: float = 0.25,
    planner_service: str = "/move_base/make_plan",
) -> Polyline2D | None:
    """Compute a 2D path from the given start pose to the given goal pose.

    :param start: Initial pose in the planning problem
    :param goal: Goal pose in the planning problem
    :param tolerance_m: Max. distance (m) to relax the (x,y) constraint, if goal is obstructed
    :param planner_service: Name of a ROS service with the service type `nav_msgs/GetPlan`
    :return: Polyline of 2D points in the found path, or None if no plan is found
    """
    # start = Pose2D(-0.016433, 2.3881905, 0.0, ref_frame="map")
    # goal = Pose2D(0.713996891, -1.31167166, 0.0, ref_frame="map")

    move_base_caller = ServiceCaller[GetPlanRequest, GetPlanResponse](planner_service, GetPlan)

    start_msg = pose_to_stamped_msg(start.to_3d())
    goal_msg = pose_to_stamped_msg(goal.to_3d())

    rospy.loginfo(f"Start message: {start_msg}")
    rospy.loginfo(f"Goal message: {goal_msg}")
    rospy.loginfo(f"Tolerance (m): {tolerance_m}")

    request = GetPlanRequest(start_msg, goal_msg, tolerance_m)
    response = move_base_caller(request)
    if response is None or response.plan.poses is None:
        rospy.logwarn("GetPlan service response was None; navigation failed.")
        return None

    plan: list[Pose3D] = [pose_from_msg(ps) for ps in response.plan.poses]

    return Polyline2D.from_poses(plan)
