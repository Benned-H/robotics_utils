"""Define a data structure to compute and validate placement poses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from robotics_utils.io import console
from robotics_utils.ros import TransformManager
from robotics_utils.spatial import DEFAULT_FRAME, Pose3D

if TYPE_CHECKING:
    from robotics_utils.robots import Manipulator


@dataclass(frozen=True)
class PlacePoses:
    """End-effector poses used to place an object onto a surface."""

    preplace_pose: Pose3D
    place_pose: Pose3D
    postplace_pose: Pose3D

    @staticmethod
    def compute_preplace_pose(pose_s_ee: Pose3D, lift_z_m: float, world_frame: str) -> Pose3D:
        """Compute a pre-place pose for the given end-effector place pose.

        :param pose_s_ee: End-effector (frame ee) place pose w.r.t. the surface (frame s)
        :param lift_z_m: Offset (m) of the pre-place pose up (+z) w.r.t. the world frame
        :param world_frame: Global reference frame used to define "up"
        :return: Pre-place pose equivalent to the place pose but "lifted" in the world frame
        """
        pose_w_ee = TransformManager.convert_to_frame(pose_s_ee, target_frame=world_frame)
        lifted_z = pose_w_ee.position.z + lift_z_m  # Pre-place z-coord. w.r.t. world frame
        lifted_xyz = replace(pose_w_ee.position, z=lifted_z)
        return replace(pose_w_ee, position=lifted_xyz)  # Pre-place w.r.t. world

    @staticmethod
    def compute_postplace_pose(pose_s_ee: Pose3D, back_x_m: float) -> Pose3D:
        """Compute a post-place pose for the given end-effector place pose.

        :param pose_s_ee: End-effector (frame ee) place pose w.r.t. the surface (frame s)
        :param back_x_m: Offset (abs. m) of the post-place pose "back" (-x) from the place pose
        :return: Post-place pose expressed in the surface frame
        """
        pose_place_postplace = Pose3D.from_xyz_rpy(x=-abs(back_x_m))  # post-place w.r.t. place
        return pose_s_ee @ pose_place_postplace  # post-place w.r.t. surface

    @classmethod
    def from_place_pose(
        cls,
        pose_s_ee: Pose3D,
        lift_z_m: float = 0.1,
        back_x_m: float = 0.05,
        world_frame: str = DEFAULT_FRAME,
    ) -> PlacePoses:
        """Compute all poses needed to place an object using the given end-effector place pose.

        :param pose_s_ee: Placement end-effector pose (frame ee) relative to the surface (frame s)
        :param lift_z_m: Offset (m) of the pre-place pose up (+z) w.r.t. the world frame
        :param back_x_m: Offset (abs. m) of the post-place pose "back" (-x) from the place pose
        :param world_frame: Global reference frame used to define "up"
        :return: Constructed PlacePoses instance containing the computed poses
        """
        pre_place = PlacePoses.compute_preplace_pose(pose_s_ee, lift_z_m, world_frame)
        post_place = PlacePoses.compute_postplace_pose(pose_s_ee=pose_s_ee, back_x_m=back_x_m)
        return PlacePoses(preplace_pose=pre_place, place_pose=pose_s_ee, postplace_pose=post_place)

    def validate_ik(self, manipulator: Manipulator) -> bool:
        """Check whether the placement poses have inverse kinematics (IK) solutions for an arm.

        :param manipulator: Robot arm used to compute IK solutions
        :return: True if all poses have an IK solution, else False
        """
        pre_ok = manipulator.compute_ik(self.preplace_pose) is not None
        if not pre_ok:
            console.print(f"[red]Invalid pre-place pose: {self.preplace_pose}[/]")
            TransformManager.broadcast_transform("failed_preplace", self.preplace_pose)

        place_ok = manipulator.compute_ik(self.place_pose) is not None
        if not place_ok:
            console.print(f"[red]Invalid place pose: {self.place_pose}[/]")
            TransformManager.broadcast_transform("failed_place", self.place_pose)

        post_ok = manipulator.compute_ik(self.postplace_pose) is not None
        if not post_ok:
            console.print(f"[red]Invalid post-place pose: {self.postplace_pose}[/]")
            TransformManager.broadcast_transform("failed_postplace", self.postplace_pose)

        return pre_ok and place_ok and post_ok
