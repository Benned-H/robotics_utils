"""Define dataclasses to structure manipulation skills."""

from dataclasses import dataclass

from robotics_utils.kinematics import Pose3D


@dataclass(frozen=True)
class GraspPose:
    """An end-effector grasp pose w.r.t. a grasped object."""

    pose_o_g: Pose3D
    """End-effector grasp pose (frame g) w.r.t. the object (frame o)."""

    ignore_collisions: bool
    """Should collisions between the end-effector and the object be ignored? Default: True."""


@dataclass(frozen=True)
class PickParameters:
    """Parameters defining a 'Pick' trajectory w.r.t. some object.

    Steps in the skill:
        1. Move the end-effector to the pre-grasp pose.
        2. Move the end-effector to the grasp pose.
        3. Close the end-effector.
        4. Move the end-effector to the post-grasp pose.
    """

    grasp_pose: GraspPose

    pre_grasp_offset_m: float
    """Backward offset (meters) of the pre-grasp pose w.r.t. the grasp pose."""

    post_grasp_lift_m: float
    """Vertical offset (meters) of the post-grasp pose w.r.t. the grasp pose (in global frame)."""
