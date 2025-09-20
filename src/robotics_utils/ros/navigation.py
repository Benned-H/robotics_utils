"""Define ROS-dependent utilities for managing robot navigation."""

from __future__ import annotations

import rospy
from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse

from robotics_utils.kinematics import Pose3D
from robotics_utils.math.polyline import Polyline2D
from robotics_utils.ros.msg_conversion import pose_from_msg, pose_to_stamped_msg
from robotics_utils.ros.services import ServiceCaller


def compute_navigation_plan(
    start: Pose3D,
    goal: Pose3D,
    tolerance_m: float = 0.1,
    planner_service: str = "/move_base/make_plan",
) -> Polyline2D | None:
    """Compute a 2D path from the given start pose to the given goal pose.

    :param start: Initial pose in the planning problem
    :param goal: Goal pose in the planning problem
    :param tolerance_m: Max. distance (m) to relax the (x,y) constraint, if goal is obstructed
    :param planner_service: Name of a ROS service with the service type `nav_msgs/GetPlan`
    :return: Polyline of 2D points in the found path, or None if no plan is found
    """
    move_base_caller = ServiceCaller[GetPlanRequest, GetPlanResponse](planner_service, GetPlan)

    request = GetPlanRequest(pose_to_stamped_msg(start), pose_to_stamped_msg(goal), tolerance_m)
    response = move_base_caller(request)
    if response is None:
        rospy.logwarn("GetPlan service response was None; navigation failed.")
        return None

    plan: list[Pose3D] = [pose_from_msg(ps) for ps in response.plan.poses]
    # TODO: How do we know when the service failed? Detect and exit here

    return Polyline2D.from_poses(plan)
