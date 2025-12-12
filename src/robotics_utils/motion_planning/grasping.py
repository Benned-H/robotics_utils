"""Define functions and data structures to compute and convert grasp poses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from robotics_utils.io import console
from robotics_utils.kinematics import DEFAULT_FRAME, Pose3D
from robotics_utils.ros.transform_manager import TransformManager

if TYPE_CHECKING:
    from robotics_utils.robots import Manipulator


@dataclass(frozen=True)
class PickPoses:
    """End-effector poses used to pick up an object."""

    pregrasp_pose: Pose3D
    grasp_pose: Pose3D
    postgrasp_pose: Pose3D

    @staticmethod
    def compute_pre_grasp_pose(pose_o_g: Pose3D, pre_grasp_x_m: float) -> Pose3D:
        """Compute a pre-grasp pose for the given grasp pose.

        Assumes that +x is "forward" in the end-effector frame.

        :param pose_o_g: Grasp pose (frame g) w.r.t. an object (frame o)
        :param pre_grasp_x_m: Offset (abs. m) of the pre-grasp pose "back" (-x) from the grasp pose
        :return: Pre-grasp pose expressed in the object frame (i.e., pose_o_pregrasp)
        """
        pose_g_pregrasp = Pose3D.from_xyz_rpy(x=-abs(pre_grasp_x_m))  # pre-grasp w.r.t. grasp
        return pose_o_g @ pose_g_pregrasp  # pre-grasp w.r.t. object

    @staticmethod
    def compute_post_grasp_pose(pose_o_g: Pose3D, lift_z_m: float, world_frame: str) -> Pose3D:
        """Compute a post-grasp pose that lifts the end-effector in the world frame.

        :param pose_o_g: Grasp pose (frame g) w.r.t. an object (frame o)
        :param lift_z_m: Offset (m) of the post-grasp pose up (+z) w.r.t. the world frame
        :param world_frame: Global reference frame used to define up (defaults to "map")
        :return: Post-grasp pose equivalent to the grasp pose but "lifted" in the world frame
        """
        pose_w_g = TransformManager.convert_to_frame(pose_o_g, target_frame=world_frame)
        lifted_z = pose_w_g.position.z + lift_z_m  # Post-grasp z-coordinate w.r.t. world
        lifted_xyz = replace(pose_w_g.position, z=lifted_z)
        return replace(pose_w_g, position=lifted_xyz)  # Post-grasp w.r.t. world

    @classmethod
    def from_grasp_pose(
        cls,
        pose_o_g: Pose3D,
        pre_grasp_x_m: float,
        lift_z_m: float,
        world_frame: str = DEFAULT_FRAME,
    ) -> PickPoses:
        """Compute all poses needed for picking up an object based on the given grasp pose.

        :param pose_o_g: Grasp pose (frame g) w.r.t. an object (frame o)
        :param pre_grasp_x_m: Offset (abs. m) of the pre-grasp pose "back" (-x) from the grasp pose
        :param lift_z_m: Offset (m) of the post-grasp pose "up" (+z) w.r.t. the world frame
        :param world_frame: Global reference frame used to define "up"
        :return: Constructed PickPoses instances containing the computed poses
        """
        pre_grasp = PickPoses.compute_pre_grasp_pose(pose_o_g, pre_grasp_x_m=pre_grasp_x_m)
        post_grasp = PickPoses.compute_post_grasp_pose(pose_o_g, lift_z_m, world_frame)
        return PickPoses(pregrasp_pose=pre_grasp, grasp_pose=pose_o_g, postgrasp_pose=post_grasp)

    @classmethod
    def select_grasp_pose(
        cls,
        candidates: list[Pose3D],
        manipulator: Manipulator,
        pre_grasp_x_m: float,
        lift_z_m: float,
        world_frame: str = DEFAULT_FRAME,
    ) -> Pose3D | None:
        """Select a grasp pose based on its (and its derived poses') kinematic reachability.

        :param candidates: List of potential grasp poses (w.r.t. the to-be-picked object)
        :param manipulator: Robot arm used to compute IK solutions
        :param pre_grasp_x_m: Offset (abs. m) of the pre-grasp pose "back" (-x) from the grasp pose
        :param lift_z_m: Offset (m) of the post-grasp pose "up" (+z) w.r.t. the world frame
        :param world_frame: Global reference frame used to define "up"
        :return: First feasible grasp pose identified, or None if no candidates are valid
        """
        for grasp_pose in candidates:
            poses = PickPoses.from_grasp_pose(grasp_pose, pre_grasp_x_m, lift_z_m, world_frame)

            if poses.validate_ik(manipulator):
                return grasp_pose

        return None

    def validate_ik(self, manipulator: Manipulator) -> bool:
        """Validate whether the poses have inverse kinematics (IK) solutions for the given arm.

        :param manipulator: Robot arm used to compute IK solutions
        :return: True if all poses have an IK solution, else False
        """
        pre_ok = manipulator.compute_ik(self.pregrasp_pose) is not None
        if not pre_ok:
            console.print(f"[red]Invalid pre-grasp pose: {self.pregrasp_pose}[/]")

        grasp_ok = manipulator.compute_ik(self.grasp_pose) is not None
        if not grasp_ok:
            console.print(f"[red]Invalid grasp pose: {self.grasp_pose}[/]")

        post_ok = manipulator.compute_ik(self.postgrasp_pose) is not None
        if not post_ok:
            console.print(f"[red]Invalid post-grasp pose: {self.postgrasp_pose}[/]")

        return pre_ok and grasp_ok and post_ok
