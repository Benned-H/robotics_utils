"""Define functions to convert poses into RViz visualization markers."""

from __future__ import annotations

from dataclasses import dataclass

import rospy
from geometry_msgs.msg import Point as PointMsg
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker as MarkerMsg

from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.poses import Pose3D
from robotics_utils.perception.pose_tracker import PoseTracker
from robotics_utils.ros.msg_conversion import point_to_msg
from robotics_utils.ros.transform_manager import TransformManager


@dataclass(frozen=True)
class RGBAxesConfig:
    """Default settings for RGB axes used to visualize 3D poses."""

    alpha: float = 1.0  # Opacity value within [0.0, 1.0]
    axis_length_m: float = 0.06  # Length (meters) of each x/y/z-axis
    axis_width_m: float = 0.005  # Width (meters) of the axes
    lifetime_s: float = 10.0  # Lifetime (seconds) of published axes


AxesMsgs = tuple[list[PointMsg], list[ColorRGBA]]


class FrameVisualizer:
    """A wrapper used to visualize estimated reference frames in RViz."""

    def __init__(self, config: RGBAxesConfig) -> None:
        """Initialize a ROS publisher used to send marker messages to RViz.

        :param config: Settings for visual properties of the generated RViz markers
        """
        self.marker_pub = rospy.Publisher("visualization_marker", MarkerMsg, queue_size=10)
        self.config = config

        self._marker_id = 0  # Increments indefinitely

    def create_rgb_axes(self, pose: Pose3D) -> AxesMsgs:
        """Convert a 3D pose into points visualizing its frame as RGB axes.

        :param pose: Pose to be visualized using RGB axes
        :return: Tuple containing two lists of ROS messages: Six 3D points and six RGBA colors
        """
        axis_origin = Point3D.identity()

        # Position of x-axis, y-axis, and z-axis endpoints w.r.t. pose frame (p)
        position_p_x = Point3D(self.config.axis_length_m, 0, 0)
        position_p_y = Point3D(0, self.config.axis_length_m, 0)
        position_p_z = Point3D(0, 0, self.config.axis_length_m)

        x_axis = [pose @ axis_origin, pose @ position_p_x]
        y_axis = [pose @ axis_origin, pose @ position_p_y]
        z_axis = [pose @ axis_origin, pose @ position_p_z]
        all_points = x_axis + y_axis + z_axis
        point_msgs = [point_to_msg(p) for p in all_points]

        red_color = ColorRGBA(1, 0, 0, self.config.alpha)
        green_color = ColorRGBA(0, 1, 0, self.config.alpha)
        blue_color = ColorRGBA(0, 0, 1, self.config.alpha)
        color_msgs = [red_color, red_color, green_color, green_color, blue_color, blue_color]

        return point_msgs, color_msgs

    def estimates_to_msg(self, frame: str, estimates: list[Pose3D]) -> MarkerMsg:
        """Convert a list of pose estimates into a visualization_msgs/Marker message.

        :param frame: Name of the frame corresponding to the pose estimates
        :param estimates: Pose estimates for the frame
        :return: ROS message visualizing the pose estimates as RGB axes
        """
        msg = MarkerMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = DEFAULT_FRAME  # Visualize the poses w.r.t. the default frame

        msg.ns = frame
        msg.id = self._marker_id
        self._marker_id += 1
        msg.lifetime = rospy.Duration.from_sec(self.config.lifetime_s)

        msg.type = MarkerMsg.LINE_LIST
        msg.action = MarkerMsg.ADD
        msg.scale.x = self.config.axis_width_m

        for estimate in estimates:
            pose_w = TransformManager.convert_to_frame(estimate, DEFAULT_FRAME)
            point_msgs, color_msgs = self.create_rgb_axes(pose_w)
            msg.points += point_msgs
            msg.colors += color_msgs

        return msg

    def visualize_estimates(self, tracker: PoseTracker) -> None:
        """Visualize a collection of tracked poses by publishing markers to RViz."""
        for frame_name, estimates in tracker.all_estimates.items():
            marker_msg = self.estimates_to_msg(frame_name, estimates)
            self.marker_pub.publish(marker_msg)
