"""Define a dataclass to model attachments between objects and end-effectors."""

from __future__ import annotations

from dataclasses import dataclass, field

from robotics_utils.spatial import Pose3D


@dataclass(frozen=True)
class GraspAttachment:
    """The kinematic details of an object-to-end-effector grasp attachment."""

    obj_name: str
    """Name of the grasped object."""

    robot_name: str
    """Name of the robot grasping the object."""

    ee_link_name: str
    """Name of the end-effector link used in the grasp.

    Once the object is grasped, this link becomes the parent frame of the object.
    """

    pose_ee_o: Pose3D
    """Pose of the grasped object relative to the end-effector (assumed authoritative)."""

    touching_link_names: set[str] = field(default_factory=set)
    """Optional set of additional link names permitted to touch or collide with the object."""
