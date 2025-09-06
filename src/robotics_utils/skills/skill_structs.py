"""Define dataclasses to structure manipulation skills."""

from dataclasses import dataclass


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
