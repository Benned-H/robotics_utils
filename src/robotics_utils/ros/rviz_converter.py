"""Define a class to visualize data structures in RViz."""

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from robotics_utils.kinematics import Waypoints
from robotics_utils.motion_planning.navigation_query import NavigationQuery
from robotics_utils.motion_planning.rectangular_footprint import RectangularFootprint
from robotics_utils.ros.msg_conversion import pose_to_msg
from robotics_utils.spatial import Pose2D
from robotics_utils.vision import RGB

ROBOT_RGB: RGB = (237, 237, 36)


class RVizConverter:
    """A class that visualizes data structures in RViz."""

    def __init__(self) -> None:
        """Initialize the RViz marker converter."""
        self._footprint_ids: dict[str, int] = {}
        """A map from robot footprint names (e.g., "goal pose") to their integer marker IDs."""

        self._array_pub = rospy.Publisher("rviz_markers", MarkerArray, latch=True, queue_size=10)

    def footprint_to_msg(
        self,
        pose: Pose2D,
        footprint: RectangularFootprint,
        name: str,
        color: RGB = ROBOT_RGB,
    ) -> Marker:
        """Visualize the given robot footprint at the given base pose.

        :param pose: Base pose at which the footprint is visualized
        :param footprint: Rectangular robot footprint
        :param name: Identifier for the footprint, used to manage overwriting old versions
        :param color: RGB color used for the footprint
        :return: Constructed visualization_msgs/Marker message
        """
        msg = Marker()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = pose.ref_frame

        msg.ns = "footprints"
        if name not in self._footprint_ids:
            new_id = len(self._footprint_ids)
            self._footprint_ids[name] = new_id
        msg.id = self._footprint_ids[name]
        msg.type = Marker.LINE_STRIP  # Draws a line between every two consecutive points
        msg.action = Marker.ADD

        msg.pose = pose_to_msg(pose.to_3d())
        msg.scale.x = 0.02  # Width (m) of line segments
        msg.color.r = color[0] / 255.0
        msg.color.g = color[1] / 255.0
        msg.color.b = color[2] / 255.0
        msg.color.a = 0.9

        msg.lifetime = rospy.Duration.from_sec(10.0)

        corner_msgs = [Point(c.x, c.y, 0.0) for c in footprint.corners]
        msg.points = [*corner_msgs, corner_msgs[0]]

        return msg

    def visualize_nav_query(self, query: NavigationQuery) -> None:
        """Visualize the given navigation query in RViz."""
        msg = MarkerArray()

        start_marker = self.footprint_to_msg(
            pose=query.start_pose,
            footprint=query.robot_footprint,
            name="start",
            color=(0, 255, 0),
        )
        goal_marker = self.footprint_to_msg(
            pose=query.goal_pose,
            footprint=query.robot_footprint,
            name="goal",
            color=(255, 0, 0),
        )

        msg.markers = [start_marker, goal_marker]
        self._array_pub.publish(msg)

    def visualize_waypoints(
        self,
        waypoints: Waypoints,
        footprint: RectangularFootprint,
        color: RGB = (0, 255, 255),
    ) -> None:
        """Visualize the given collection of waypoints."""
        msg = MarkerArray()
        msg.markers = [
            self.footprint_to_msg(wp_pose, footprint, wp_name, color)
            for wp_name, wp_pose in waypoints.items()
        ]

        self._array_pub.publish(msg)
