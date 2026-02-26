"""Define a generator to sample random placement poses on a stable resting surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

from robotics_utils.motion_planning.ros import PlacePoses
from robotics_utils.spatial import Pose3D
from robotics_utils.tamp import Generator

if TYPE_CHECKING:
    from robotics_utils.robots import Manipulator
    from robotics_utils.states import ObjectKinematicState, PlacementSurface


@dataclass(frozen=True)
class PlaceObjectPoseArgs:
    """Input arguments for an object place pose generator."""

    surface: PlacementSurface
    placed_object: ObjectKinematicState
    """Kinematic state of the object to be placed, used for its collision geometry."""


class PlaceObjectPoseGenerator(Generator[PlaceObjectPoseArgs, Pose3D]):
    """A sampler for object placement poses on stable resting surfaces."""

    def _generate(self, inputs: PlaceObjectPoseArgs) -> Iterator[Pose3D]:
        """Generate an infinite sequence of object placement pose samples.

        :param inputs: Conditioning values for the generator
        :yield: Sequence of generated placement poses in the surface frame (pose_s_o)
        """
        obj_aabb = inputs.placed_object.collision_model.aabb
        obj_radius_m = max(obj_aabb.size_x_m, obj_aabb.size_y_m) / 2.0

        narrowed_x_range = inputs.surface.x_range.narrow_inward(obj_radius_m)
        narrowed_y_range = inputs.surface.y_range.narrow_inward(obj_radius_m)

        while True:
            yield Pose3D.from_xyz_rpy(
                x=narrowed_x_range.uniform_sample(self._rng),
                y=narrowed_y_range.uniform_sample(self._rng),
                z=inputs.surface.height_m,
                yaw_rad=self._rng.uniform(-np.pi, np.pi),
                ref_frame=inputs.surface.frame,
            )


@dataclass(frozen=True)
class PlacePosesArgs:
    """Input arguments for generating a trio of end-effector place poses."""

    surface: PlacementSurface
    placed_object: ObjectKinematicState
    pose_ee_o: Pose3D
    manipulator: Manipulator


class PlacePosesGenerator(Generator[PlacePosesArgs, PlacePoses]):
    """A sampler of feasible end-effector poses for placing on a surface (with pre/post poses)."""

    def _generate(self, inputs: PlacePosesArgs) -> Iterator[Pose3D]:
        """Generate trios of feasible end-effector pre-place, place, and post-place poses.

        :param inputs: Conditioning values for the generator
        :yield: Sequence of generated end-effector pre-place/place/post-place poses
        """
        object_name = inputs.placed_object.name
        pose_o_ee = inputs.pose_ee_o.inverse(pose_frame=object_name)

        obj_place_pose_args = PlaceObjectPoseArgs(inputs.surface, inputs.placed_object)

        for pose_s_o in PlaceObjectPoseGenerator(obj_place_pose_args, rng_seed=self._rng_seed):
            pose_s_ee = pose_s_o @ pose_o_ee

            place_poses = PlacePoses.from_place_pose(pose_s_ee=pose_s_ee)
            if place_poses.validate_ik(inputs.manipulator):
                yield place_poses
