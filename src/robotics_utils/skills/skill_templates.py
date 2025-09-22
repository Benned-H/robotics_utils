"""Define dataclasses to structure manipulation skills."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics_utils.kinematics import Pose3D
from robotics_utils.math.sampling import RealRange


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
class PlacementSurface:
    """A model of a stable placement surface."""

    frame: str
    """Name of the object frame w.r.t. which this surface is defined."""

    height_m: float
    """Height (+z meters) of the surface w.r.t. the frame."""

    x_range_m: RealRange
    """Range of x-values (meters) of stable placement poses on the surface."""

    y_range_m: RealRange
    """Range of y-values (meters) of stable placement poses on the surface."""

    def sample_placement_pose(
        self,
        yaw_range_rad: RealRange | None = None,
        rng: np.random.Generator | None = None,
    ) -> Pose3D:
        """Sample a 6-DoF placement pose on this surface.

        :param yaw_range_rad: Optional yaw range [low, high] in radians; defaults to [-pi, pi]
        :param rng: Optional NumPy random number generator; defaults to np.random.default_rng()
        :return: Sampled Pose3D placement pose w.r.t. the surface frame
        """
        rng = np.random.default_rng() if rng is None else rng
        yaw_range_rad = RealRange(-np.pi, np.pi) if yaw_range_rad is None else yaw_range_rad

        return Pose3D.from_xyz_rpy(
            x=self.x_range_m.sample(rng),
            y=self.y_range_m.sample(rng),
            z=float(self.height_m),
            yaw_rad=yaw_range_rad.sample(rng),
            ref_frame=self.frame,
        )


@dataclass(frozen=True)
class PlaceTemplate:
    """A template defining a 'Place' trajectory."""

    pose_s_base: Pose3D
    """Surface-relative base pose during the skill."""

    place_pose_s_o: Pose3D
    """Placement pose of the held object w.r.t. the placement surface."""

    pre_place_lift_m: float
    """Offset (meters) of the pre-place pose "up" (+z world frame) from the place pose."""

    post_place_x_m: float
    """Offset (absolute meters) of the post-place pose "back" (-x) relative to the place pose."""
