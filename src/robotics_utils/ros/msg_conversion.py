"""Define functions to convert between kinematic data structures and ROS messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped, Vector3
from geometry_msgs.msg import Quaternion as QuaternionMsg
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robotics_utils.collision_models import Box, Cylinder, PrimitiveShape, Sphere
from robotics_utils.kinematics import DEFAULT_FRAME, Configuration, Point3D, Pose3D, Quaternion
from robotics_utils.motion_planning import Trajectory, TrajectoryPoint

if TYPE_CHECKING:
    import trimesh

    from robotics_utils.world_models.simulators import ObjectModel


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


def make_collision_object_msg(
    object_model: ObjectModel,
    object_type: str | None = None,
) -> CollisionObject:
    """Construct a moveit_msgs/CollisionObject message using the given data."""
    msg = CollisionObject()
    msg.header.frame_id = object_model.pose.ref_frame
    msg.pose = pose_to_msg(object_model.pose)

    msg.id = object_model.name
    if object_type is not None:
        msg.type.key = object_type  # Ignore 'db' field of message

    identity_pose_msg = pose_to_msg(Pose3D.identity(ref_frame=object_model.name))

    msg.meshes = [trimesh_to_msg(mesh) for mesh in object_model.collision_model.meshes]
    msg.mesh_poses = [identity_pose_msg] * len(msg.meshes)

    msg.primitives = [primitive_shape_to_msg(ps) for ps in object_model.collision_model.primitives]

    # Find the pose of each primitive shape w.r.t. the object frame
    ps_aabbs = [ps.aabb for ps in object_model.collision_model.primitives]
    ps_z_dims = [aabb.max_xyz.z - aabb.min_xyz.z for aabb in ps_aabbs]
    ps_poses = [Pose3D.from_xyz_rpy(z=h_m / 2, ref_frame=object_model.name) for h_m in ps_z_dims]
    msg.primitive_poses = [pose_to_msg(pose) for pose in ps_poses]

    msg.operation = CollisionObject.ADD
    return msg


def trajectory_point_to_msg(point: TrajectoryPoint) -> JointTrajectoryPoint:
    """Convert a trajectory point into a trajectory_msgs/JointTrajectoryPoint message."""
    msg = JointTrajectoryPoint()
    msg.positions = [point.position[j] for j in point.joint_names]
    msg.velocities = [point.velocities[j] for j in point.joint_names]
    msg.time_from_start = rospy.Duration.from_sec(point.time_s)
    return msg


def trajectory_to_msg(trajectory: Trajectory) -> JointTrajectory:
    """Convert a trajectory of points into a trajectory_msgs/JointTrajectory message."""
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
        position=dict(zip(joint_names, msg.positions)),
        velocities=dict(zip(joint_names, msg.velocities)),
    )


def trajectory_from_msg(traj_msg: JointTrajectory) -> Trajectory:
    """Construct a Trajectory from a trajectory_msgs/JointTrajectory message."""
    joint_names = traj_msg.joint_names
    traj_msg.points = traj_msg.points or []
    return Trajectory([trajectory_point_from_msg(p_msg, joint_names) for p_msg in traj_msg.points])
