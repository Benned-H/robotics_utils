"""Define dataclasses to structure manipulation skills."""

from dataclasses import dataclass

from robotics_utils.kinematics import Pose3D


@dataclass(frozen=True)
class PickTemplate:
    """A template defining a 'Pick' trajectory w.r.t. some object.

    Steps in the skill:
        1. Open the gripper.
        2. Move the end-effector to the pre-grasp pose.
        3. Move the end-effector to the grasp pose.
        4. Close the gripper.
        5. Move the end-effector to the post-grasp pose.
        6. Move the end-effector to the carry pose.
    """

    pose_o_g: Pose3D
    """Grasp pose of the end-effector w.r.t. the object to be picked."""

    pre_grasp_x_m: float
    """Offset (absolute meters) of the pre-grasp pose "back" (-x) from the grasp pose."""

    post_grasp_lift_m: float
    """Offset (meters) of the post-grasp pose "up" (+z) from the grasp pose in the world frame."""

    pose_b_carry: Pose3D
    """End-effector pose (w.r.t. body frame) used to carry the object."""

    stow_carry: bool
    """If True, stow the arm instead of carrying based on `pose_b_carry`."""


@dataclass(frozen=True)
class PlaceTemplate:
    """A template defining a 'Place' trajectory.

    Steps in the skill:
        1. Move the end-effector to the pre-place pose.
        2. Move the end-effector to the place pose.
        3. Open the gripper.
        4. Move the gripper to the post-place pose.
        5. Stow the arm.
    """

    pose_s_o: Pose3D
    """Place pose of the held object w.r.t. the placement surface."""

    pre_place_lift_m: float
    """Offset (meters) of the pre-place pose "up" (+z) from the place pose in the world frame."""

    post_place_x_m: float
    """Offset (absolute meters) of the post-place pose "back" (-x) from the place pose."""
