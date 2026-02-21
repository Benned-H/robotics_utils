"""Define a generator to sample random placement poses on a stable resting surface."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from robotics_utils.spatial import Pose3D
from robotics_utils.states import ObjectKinematicState, PlacementSurface
from robotics_utils.tamp import Generator


@dataclass(frozen=True)
class PlaceObjectPoseArgs:
    """Input arguments for a place pose generator."""

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
                x=narrowed_x_range.uniform_sample(self.rng),
                y=narrowed_y_range.uniform_sample(self.rng),
                z=inputs.surface.height_m,
                yaw_rad=self.rng.uniform(-np.pi, np.pi),
                ref_frame=inputs.surface.frame,
            )
