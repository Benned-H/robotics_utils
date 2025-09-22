"""Define dataclasses to structure manipulation skills."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from robotics_utils.kinematics import DEFAULT_FRAME, Pose3D
from robotics_utils.ros import TransformManager


@dataclass(frozen=True)
class PickTemplate:
    """A template defining a 'Pick' skill trajectory.

    Steps in the skill:
        1. Open the gripper to the specified angle (open_gripper_angle_rad).
        2. Move the end-effector to the pre-grasp pose.
        3. Move the end-effector to the grasp pose (pose_o_g).
        4. Close the gripper.
        5. Move the end-effector to the post-grasp pose.
        6. Stow the arm, if requested.
    """

    object_name: str
    """Name of the object to be picked."""

    open_gripper_angle_rad: float
    """Gripper angle (radians) before picking the object."""

    pre_grasp_x_m: float
    """Offset (absolute meters) of the pre-grasp pose "back" (-x) from the grasp pose."""

    pose_o_g: Pose3D
    """Object-relative pose of the end-effector when the object is grasped."""

    post_grasp_lift_m: float
    """Offset (meters) of the post-grasp pose "up" (+z) from the grasp pose in the world frame."""

    stow_after: bool
    """If True, stow the arm after picking the object."""

    @property
    def pose_o_pregrasp(self) -> Pose3D:
        """Compute and return the object-relative pre-grasp pose."""
        pose_g_pregrasp = Pose3D.from_xyz_rpy(x=-abs(self.pre_grasp_x_m))
        return self.pose_o_g @ pose_g_pregrasp

    @property
    def pose_w_postgrasp(self) -> Pose3D:
        """Compute and return the world-frame post-grasp pose."""
        pose_w_g = TransformManager.convert_to_frame(self.pose_o_g, target_frame=DEFAULT_FRAME)
        pose_w_postgrasp = deepcopy(pose_w_g)
        pose_w_postgrasp.position.z += self.post_grasp_lift_m
        return pose_w_postgrasp


@dataclass(frozen=True)
class PlaceTemplate:
    """A template defining a 'Place' skill trajectory."""

    ee_link_name: str
    """Name of the end-effector link used to place an object."""

    held_object_name: str
    """Name of the held object to be placed."""

    pre_place_lift_m: float
    """Offset (meters) of the pre-place pose "up" (+z world frame) from the place pose."""

    place_pose_s_o: Pose3D
    """Surface-relative placement pose of the placed object."""

    post_place_x_m: float
    """Offset (absolute meters) of the post-place pose "back" (-x) from the place pose."""

    @property
    def preplace_pose_w_ee(self) -> Pose3D | None:
        """Compute and return the pre-place end-effector pose in the world frame."""
        place_pose_s_ee = self.place_pose_s_ee
        if place_pose_s_ee is None:
            return None

        place_pose_w_ee = TransformManager.convert_to_frame(place_pose_s_ee, DEFAULT_FRAME)
        preplace_pose_w_ee = deepcopy(place_pose_w_ee)
        preplace_pose_w_ee.position.z += self.pre_place_lift_m
        return preplace_pose_w_ee

    @property
    def place_pose_s_ee(self) -> Pose3D | None:
        """Compute and return the surface-relative end-effector pose when placing.

        :return: End-effector pose when placing, or None if transform lookup fails
        """
        pose_o_ee = TransformManager.lookup_transform(self.ee_link_name, self.held_object_name)

        return None if pose_o_ee is None else self.place_pose_s_o @ pose_o_ee

    @property
    def postplace_pose_s_ee(self) -> Pose3D | None:
        """Compute and return the surface-relative post-place end-effector pose."""
        place_pose_s_ee = self.place_pose_s_ee
        pose_place_postplace = Pose3D.from_xyz_rpy(x=-self.post_place_x_m)  # w.r.t. place pose
        return None if place_pose_s_ee is None else place_pose_s_ee @ pose_place_postplace


@dataclass(frozen=True)
class OpenDrawerTemplate:
    """A template defining an 'OpenDrawer' skill."""

    pregrasp_pose_ee: Pose3D
    """Target end-effector pose before the drawer-grasping pose."""

    grasp_drawer_pose_ee: Pose3D
    """End-effector pose used to grasp the drawer handle."""

    pull_drawer_pose_ee: Pose3D
    """Target end-effector pose after initially pulling the drawer open."""

    open_traj_path: Path
    """Path to a YAML file containing the trajectory used to finish opening the drawer."""


# @dataclass(frozen=True)
# class PlacementSurface:
#     """A model of a stable placement surface."""

#     frame: str
#     """Name of the object frame w.r.t. which this surface is defined."""

#     height_m: float
#     """Height (+z meters) of the surface w.r.t. the frame."""

#     x_range_m: RealRange
#     """Range of x-values (meters) of stable placement poses on the surface."""

#     y_range_m: RealRange
#     """Range of y-values (meters) of stable placement poses on the surface."""

#     def sample_placement_pose(
#         self,
#         yaw_range_rad: RealRange | None = None,
#         rng: np.random.Generator | None = None,
#     ) -> Pose3D:
#         """Sample a 6-DoF placement pose on this surface.

#         :param yaw_range_rad: Optional yaw range [low, high] in radians; defaults to [-pi, pi]
#         :param rng: Optional NumPy random number generator; defaults to np.random.default_rng()
#         :return: Sampled Pose3D placement pose w.r.t. the surface frame
#         """
#         rng = np.random.default_rng() if rng is None else rng
#         yaw_range_rad = RealRange(-np.pi, np.pi) if yaw_range_rad is None else yaw_range_rad

#         return Pose3D.from_xyz_rpy(
#             x=self.x_range_m.sample(rng),
#             y=self.y_range_m.sample(rng),
#             z=float(self.height_m),
#             yaw_rad=yaw_range_rad.sample(rng),
#             ref_frame=self.frame,
#         )
