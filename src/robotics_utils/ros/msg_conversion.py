"""Define functions to convert between kinematic data structures and ROS messages."""

from __future__ import annotations

from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped, Vector3
from geometry_msgs.msg import Quaternion as QuaternionMsg

from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.pose3d import Pose3D
from robotics_utils.kinematics.rotations import Quaternion


def point_to_msg(point: Point3D) -> Point:
    """Convert the given point into a geometry_msgs/Point message."""
    return Point(point.x, point.y, point.z)


def point_from_msg(point_msg: Point) -> Point3D:
    """Construct a Point3D from a geometry_msgs/Point message."""
    return Point3D(point_msg.x, point_msg.y, point_msg.z)


def point_to_vector3_msg(point: Point3D) -> Vector3:
    """Convert the given point into a geometry_msgs/Vector3 message."""
    return Vector3(point.x, point.y, point.z)


def point_from_vector3_msg(vector_msg: Vector3) -> Point3D:
    """Construct a Point3D from a geometry_msgs/Vector3 message."""
    return Point3D(vector_msg.x, vector_msg.y, vector_msg.z)


def quaternion_to_msg(q: Quaternion) -> QuaternionMsg:
    """Convert the given quaternion into a geometry_msgs/Quaternion message."""
    return QuaternionMsg(q.x, q.y, q.z, q.w)


def quaternion_from_msg(q_msg: QuaternionMsg) -> Quaternion:
    """Construct a Quaternion from a geometry_msgs/Quaternion message."""
    return Quaternion(q_msg.x, q_msg.y, q_msg.z, q_msg.w)


def pose_to_msg(pose: Pose3D) -> Pose:
    """Convert the given pose into a geometry_msgs/Pose message."""
    return Pose(point_to_msg(pose.position), quaternion_to_msg(pose.orientation))


def pose_to_stamped_msg(pose: Pose3D) -> PoseStamped:
    """Convert the given pose into a geometry_msgs/PoseStamped message."""
    msg = PoseStamped()
    msg.header.frame_id = pose.ref_frame
    msg.pose = pose_to_msg(pose)
    return msg


def pose_from_msg(pose_msg: Pose | PoseStamped) -> Pose3D:
    """Construct a Pose3D from a geometry_msgs/Pose or geometry_msgs/PoseStamped message.

    :param pose_msg: ROS message representing a pose or time-stamped pose
    :return: Constructed Pose3D instance
    :raises TypeError: If the given message is neither a geometry_msgs/Pose nor PoseStamped
    """
    if isinstance(pose_msg, Pose):
        frame_id = DEFAULT_FRAME
        pose = pose_msg
    elif isinstance(pose_msg, PoseStamped):
        frame_id = pose_msg.header.frame_id
        pose = pose_msg.pose  # Extract just the Pose from the PoseStamped
    else:
        raise TypeError(f"Received unexpected ROS message type: {type(pose_msg)}")

    return Pose3D(point_from_msg(pose.position), quaternion_from_msg(pose.orientation), frame_id)


def pose_to_tf_msg(pose: Pose3D) -> Transform:
    """Convert the given pose into a geometry_msgs/Transform message."""
    translation = point_to_vector3_msg(pose.position)
    rotation = quaternion_to_msg(pose.orientation)
    return Transform(translation, rotation)


def pose_from_tf_msg(tf_msg: Transform, ref_frame: str) -> Pose3D:
    """Construct a Pose3D from a geometry_msgs/Transform message."""
    position = point_from_vector3_msg(tf_msg.translation)
    orientation = quaternion_from_msg(tf_msg.rotation)
    return Pose3D(position, orientation, ref_frame)


def pose_to_tf_stamped_msg(pose: Pose3D, child_frame: str) -> TransformStamped:
    """Convert the given pose into a geometry_msgs/TransformStamped message."""
    tf_stamped_msg = TransformStamped()
    tf_stamped_msg.header.frame_id = pose.ref_frame
    tf_stamped_msg.child_frame_id = child_frame
    tf_stamped_msg.transform = pose_to_tf_msg(pose)
    return tf_stamped_msg


def pose_from_tf_stamped_msg(tf_stamped_msg: TransformStamped) -> Pose3D:
    """Construct a Pose3D from a geometry_msgs/TransformStamped message."""
    ref_frame = tf_stamped_msg.header.frame_id
    return pose_from_tf_msg(tf_stamped_msg.transform, ref_frame)


def point_msg_to_vector3_msg(point_msg: Point) -> Vector3:
    """Convert a geometry_msgs/Point message into a geometry_msgs/Vector3 message."""
    return Vector3(point_msg.x, point_msg.y, point_msg.z)


def vector3_msg_to_point_msg(vector_msg: Vector3) -> Point:
    """Convert a geometry_msgs/Vector3 message into a geometry_msgs/Point message."""
    return Point(vector_msg.x, vector_msg.y, vector_msg.z)


def pose_msg_to_tf_msg(pose_msg: Pose) -> Transform:
    """Convert a geometry_msgs/Pose message into a geometry_msgs/Transform message."""
    return Transform(point_msg_to_vector3_msg(pose_msg.position), pose_msg.orientation)


def tf_msg_to_pose_msg(tf_msg: Transform) -> Pose:
    """Convert a geometry_msgs/Transform message into a geometry_msgs/Pose message."""
    return Pose(vector3_msg_to_point_msg(tf_msg.translation), tf_msg.rotation)
