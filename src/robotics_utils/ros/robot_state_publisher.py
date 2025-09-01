"""Define a class to publish robot joint states as ROS messages."""

import threading

import rospy
from sensor_msgs.msg import JointState

from robotics_utils.kinematics import Configuration
from robotics_utils.ros import TransformManager
from robotics_utils.ros.msg_conversion import config_to_joint_state_msg


class RobotStatePublisher:
    """A ROS interface to manage a robot's kinematic state."""

    def __init__(self, robot_name: str, config: Configuration, joint_state_topic: str) -> None:
        """Initialize the robot state with its initial configuration."""
        self.name = robot_name

        self._configuration = config
        """A configuration maps joint names to their positions (in radians or meters)."""

        self._joint_state_pub = rospy.Publisher(joint_state_topic, JointState, queue_size=10)

        self._publisher_thread = threading.Thread(target=self._publish_state_loop, daemon=True)
        self._publisher_thread.start()  # Daemon = Thread exits when main process does

    def set_configuration(self, config: Configuration) -> None:
        """Set the robot state to the given configuration."""
        self._configuration = config

    def _publish_state_loop(self) -> None:
        """Publish the stored robot state in a continual loop."""
        try:
            rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
            while not rospy.is_shutdown():
                joint_state_msg = config_to_joint_state_msg(self._configuration)
                self._joint_state_pub.publish(joint_state_msg)
                rate_hz.sleep()
        except rospy.ROSInterruptException as ros_exc:
            rospy.logwarn(f"[RobotState._publish_state_loop] {ros_exc}")
