"""Define a class to publish robot joint states as ROS messages."""

from robotics_utils.kinematics import Configuration


class RobotModel:
    """A ROS interface for a robot's kinematic model."""


# from __future__ import annotations

# import threading
# import time
# from typing import TYPE_CHECKING

# import rospy
# from moveit_commander import RobotCommander
# from robotics_utils.motion_planning.plan_skeletons import NavigationPlanSkeleton
# from robotics_utils.ros.msg_conversion import config_to_joint_state_msg, pose_to_stamped_msg
# from robotics_utils.ros.services import ServiceCaller
# from robotics_utils.ros.transform_manager import TransformManager
# from sensor_msgs.msg import JointState

# from spot_skills.srv import NavigateToPose, NavigateToPoseRequest, NavigateToPoseResponse
# from tmp3.Robots.Robot import Robot

# if TYPE_CHECKING:
#     from robotics_utils.kinematics import Configuration
#     from robotics_utils.kinematics.poses import Pose3D
#     from robotics_utils.ros.robots.manipulator import Manipulator


# class ROSRobot(Robot):
#     """A ROS interface for a robot's kinematic model."""

#     def __init__(self, name: str, initial_config: Configuration, manipulator: Manipulator):
#         """Initialize the robot ROS interface using its name and kinematics details.

#         :param name: Human-readable name of the robot
#         :param initial_config: Initial configuration of the robot's joints
#         :param manipulator: Active manipulator of the robot
#         """
#         self._using_moveit = rospy.get_param("/tamp/use_moveit", default=True)

#         joint_states_topic = rospy.get_param("tamp_joint_states_topic")
#         self._joint_state_pub = rospy.Publisher(joint_states_topic, JointState, queue_size=10)

#         super().__init__(name, initial_config, manipulator)

#         if self._using_moveit:
#             self.robot_commander = RobotCommander()
#             robot_move_groups = self.robot_commander.get_group_names()
#             rospy.loginfo(f"Robot {name} has move groups: {robot_move_groups}")
#             assert manipulator.name in robot_move_groups, (
#                 f"Unknown move group name: {manipulator.name}."
#             )

#         self.active_manipulator = manipulator

#         navigation_active = rospy.get_param("/spot/navigation/active", default=False)
#         if navigation_active:
#             # Create a client for the navigation service
#             self._nav_to_pose_topic = "/spot/navigation/to_pose"
#             self._nav_to_pose_srv = ServiceCaller[NavigateToPoseRequest, NavigateToPoseResponse](
#                 self._nav_to_pose_topic,
#                 NavigateToPose,
#                 timeout_s=15,
#             )

#         self.publisher_thread = threading.Thread(target=self._publish_robot_state_loop)
#         self.publisher_thread.daemon = True  # Thread exits when main process does
#         self.publisher_thread.start()

#     def _update_configuration(self) -> None:
#         """Notify external listeners that the robot's configuration has been updated."""
#         msg = self.get_joint_state_msg()
#         self._joint_state_pub.publish(msg)
#         time.sleep(0.1)

#     @property
#     def body_frame_name(self) -> str:
#         """Get the name of the robot's body frame."""
#         return "body"  # f"{self.name}/body"

#     def compute_navigation_plan(
#         self,
#         initial_base_pose: Pose3D,
#         target_base_pose: Pose3D,
#     ) -> NavigationPlanSkeleton | None:
#         """Compute a navigation plan between the two given robot base poses.

#         TODO: We don't currently compute navigation plans!

#         :param initial_base_pose: Robot's initial base pose from which the plan begins
#         :param target_base_pose: Target base pose to be reached by the navigation plan
#         :returns: Skeleton for a navigation plan, if one exists, else None
#         """
#         return NavigationPlanSkeleton(self.name, target_base_pose=target_base_pose)

#     def execute_navigation_plan(
#         self,
#         nav_plan: NavigationPlanSkeleton,
#         timeout_s: float = 60.0,
#     ) -> bool:
#         """Execute the given navigation plan by navigating to its target base pose.

#         :param nav_plan: Skeleton for a navigation plan to reach some target base pose
#         :param timeout_s: Duration (seconds) after which the plan is abandoned (default = 60 sec)
#         :return: Boolean indicating if the plan was successfully completed (False if not)
#         """
#         request = NavigateToPoseRequest(pose_to_stamped_msg(nav_plan.target_base_pose))
#         response = self._nav_to_pose_srv(request)

#         if response is None:
#             rospy.logerr(f"Service '{self._nav_to_pose_topic}' responded with None.")
#             return False

#         rospy.loginfo(f"Service '{self._nav_to_pose_topic}' responded: {response.message}")
#         return response.success

#     def get_joint_state_msg(self) -> JointState:
#         """Convert the robot's configuration into a sensor_msgs/JointState message."""
#         return config_to_joint_state_msg(self._current_configuration)

#     def _publish_robot_state_loop(self) -> None:
#         """Publish the robot's configuration in a loop."""
#         try:
#             rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
#             while not rospy.is_shutdown():
#                 self._update_configuration()

#                 rate_hz.sleep()
#         except rospy.ROSInterruptException as ros_exc:
#             rospy.logwarn(f"[_publish_robot_state_loop] {ros_exc}")

#     def get_ik_solution(self, ee_target: Pose3D) -> Configuration | None:
#         """Find an inverse kinematics (IK) solution for the given end-effector target.

#         TODO: What would the `check_collisions` argument (bool) have done?

#         :param ee_target: Target pose of the end-effector
#         :return: Manipulator configuration solving the given IK problem (else None)
#         """
#         return self.active_manipulator.compute_ik(ee_target)
