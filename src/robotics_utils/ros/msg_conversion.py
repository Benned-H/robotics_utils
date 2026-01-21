"""Define functions to convert between kinematic data structures and ROS messages."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped, Vector3
from geometry_msgs.msg import Quaternion as QuaternionMsg
from moveit_msgs.msg import CollisionObject
from nav_msgs.msg import MapMetaData, OccupancyGrid
from sensor_msgs import point_cloud2
from sensor_msgs.msg import JointState, PointCloud2, PointField
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robotics_utils.collision_models import Box, CollisionModel, Cylinder, PrimitiveShape, Sphere
from robotics_utils.geometry import Point3D
from robotics_utils.motion_planning import Trajectory, TrajectoryPoint
from robotics_utils.spatial import DEFAULT_FRAME, Pose3D, Quaternion

if TYPE_CHECKING:
    import trimesh

    from robotics_utils.kinematics import Configuration
    from robotics_utils.perception import OccupancyGrid2D
    from robotics_utils.reconstruction import PointCloud


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


def config_to_joint_state_msg(config: Configuration) -> JointState:
    """Convert the given configuration into a sensor_msgs/JointState message."""
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = list(config.keys())
    msg.position = list(config.values())
    msg.effort = [0] * len(config)
    msg.velocity = [0] * len(config)
    return msg


def trimesh_to_msg(mesh: trimesh.Trimesh) -> Mesh:
    """Convert a trimesh.Trimesh into a shape_msgs/Mesh message."""
    mesh_msg = Mesh()
    mesh_msg.triangles = [MeshTriangle(list(tri)) for tri in mesh.faces]
    mesh_msg.vertices = [Point(v[0], v[1], v[2]) for v in mesh.vertices]
    return mesh_msg


def primitive_shape_type_to_integer(shape: PrimitiveShape) -> int:
    """Get the shape_msgs/SolidPrimitive integer type for the given type of primitive shape."""
    if isinstance(shape, Box):
        return 1
    if isinstance(shape, Sphere):
        return 2
    if isinstance(shape, Cylinder):
        return 3

    raise ValueError(f"Unrecognized type of PrimitiveShape: {shape}")


def primitive_shape_to_msg(shape: PrimitiveShape) -> SolidPrimitive:
    """Convert a primitive geometric shape into a shape_msgs/SolidPrimitive message."""
    msg = SolidPrimitive()
    msg.type = primitive_shape_type_to_integer(shape)
    msg.dimensions = shape.to_dimensions()
    return msg


def trajectory_point_to_msg(point: TrajectoryPoint) -> JointTrajectoryPoint:
    """Convert a trajectory point into a trajectory_msgs/JointTrajectoryPoint message."""
    msg = JointTrajectoryPoint()
    msg.positions = [point.positions[j] for j in point.joint_names]
    msg.velocities = [point.velocities[j] for j in point.joint_names]
    msg.time_from_start = rospy.Duration.from_sec(point.time_s)
    return msg


def trajectory_to_msg(trajectory: Trajectory) -> JointTrajectory:
    """Convert a trajectory into a trajectory_msgs/JointTrajectory message."""
    msg = JointTrajectory()
    msg.joint_names = trajectory.joint_names
    msg.points = [trajectory_point_to_msg(p) for p in trajectory.points]
    return msg


def trajectory_point_from_msg(
    msg: JointTrajectoryPoint,
    joint_names: list[str],
) -> TrajectoryPoint:
    """Construct a TrajectoryPoint from a trajectory_msgs/JointTrajectoryPoint message."""
    return TrajectoryPoint(
        time_s=msg.time_from_start.to_sec(),
        positions=dict(zip(joint_names, msg.positions)),
        velocities=dict(zip(joint_names, msg.velocities)),
    )


def trajectory_from_msg(traj_msg: JointTrajectory) -> Trajectory:
    """Construct a Trajectory from a trajectory_msgs/JointTrajectory message."""
    joint_names = traj_msg.joint_names
    traj_msg.points = traj_msg.points or []
    return Trajectory([trajectory_point_from_msg(p_msg, joint_names) for p_msg in traj_msg.points])


def pointcloud_to_msg(cloud: PointCloud, frame_id: str = DEFAULT_FRAME) -> PointCloud2:
    """Convert a point cloud into a sensor_msgs/PointCloud2 message.

    :param cloud: PointCloud with XYZ points and optional RGB colors
    :param frame_id: Reference frame for the point cloud
    :return: PointCloud2 message containing the point data
    """
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    if cloud.colors is None:  # XYZ-only point cloud
        return point_cloud2.create_cloud_xyz32(header, cloud.points)

    # XYZ-RGB point cloud - pack RGB into a single float32
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    points_with_rgb = []
    for i in range(len(cloud)):
        r, g, b = cloud.colors[i]
        # Pack RGB as a single 32-bit integer, then reinterpret as float32
        rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
        rgb_packed = struct.unpack("f", struct.pack("I", rgb_int))[0]
        x, y, z = cloud.points[i]
        points_with_rgb.append([x, y, z, rgb_packed])

    return point_cloud2.create_cloud(header, fields, points_with_rgb)


def make_collision_object_msg(
    object_name: str,
    object_type: str,
    object_pose: Pose3D,
    collision_model: CollisionModel,
) -> CollisionObject:
    """Construct a moveit_msgs/CollisionObject message using the given data."""
    msg = CollisionObject()
    msg.header.frame_id = object_pose.ref_frame
    msg.pose = pose_to_msg(object_pose)

    msg.id = object_name
    msg.type.key = object_type  # Ignore 'db' field of message

    msg.meshes = [trimesh_to_msg(mesh) for mesh in collision_model.meshes]
    msg.mesh_poses = [Pose() for _ in collision_model.meshes]

    msg.primitives = [primitive_shape_to_msg(shape) for shape in collision_model.primitives]

    # TODO: Frames used to be shifted to bottom of objects
    msg.primitive_poses = [Pose() for _ in collision_model.primitives]

    msg.operation = CollisionObject.ADD
    return msg


def occupancy_grid_to_msg(grid: OccupancyGrid2D) -> OccupancyGrid:
    """Convert an OccupancyGrid2D into a nav_msgs/OccupancyGrid message.

    The log-odds values are converted into occupancy probabilities in the range [0, 100].
    Cells with log-odds of exactly 0.0 (never updated) are marked as unknown (-1).

    :param grid: OccupancyGrid2D with log-odds occupancy values
    :return: nav_msgs/OccupancyGrid message
    """
    msg = OccupancyGrid()

    # Header
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = grid.grid.origin.ref_frame

    # MapMetaData
    msg.info = MapMetaData()
    msg.info.map_load_time = msg.header.stamp
    msg.info.resolution = grid.grid.resolution_m
    msg.info.width = grid.grid.width_cells
    msg.info.height = grid.grid.height_cells
    msg.info.origin = pose_to_msg(grid.grid.origin.to_3d())

    # Reference: Equation (4.14) on pg. 95 of ProbRob
    # Use numerically stable sigmoid: 1 / (1 + exp(-x)) instead of 1 - 1 / (1 + exp(x))
    # to avoid overflow when log_odds has large positive values
    p_occupied = 1 / (1 + np.exp(-grid.log_odds))

    # Scale to [0, 100] as int8; mark unobserved cells (log_odds == 0) as unknown (-1)
    occupancy_values = (p_occupied * 100).astype(np.int8)
    unobserved_mask = grid.log_odds == 0.0
    occupancy_values[unobserved_mask] = -1

    # Data is row-major, starting with (0, 0) - flatten the array
    msg.data = occupancy_values.flatten().tolist()

    return msg
