"""Define a class representing the visual state of an object-centric environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Hashable, Iterable, Mapping

from robotics_utils.kinematics import Pose3D
from robotics_utils.perception.vision import RGBImage
from robotics_utils.skills import SkillInstance


@dataclass(frozen=True)
class HashablePose3D:
    """A hashable encoding of a Pose3D instance."""

    xyz_rpy: tuple[float, float, float, float, float, float]
    ref_frame: str

    @classmethod
    def from_pose(cls, pose: Pose3D) -> HashablePose3D:
        """Create a hashable encoding of the given 3D pose."""
        (x, y, z), (r, p, yaw) = pose.to_xyz_rpy()
        return HashablePose3D(xyz_rpy=(x, y, z, r, p, yaw), ref_frame=pose.ref_frame)

    def to_pose(self) -> Pose3D:
        """Construct a Pose3D from an equivalent hashable encoding."""
        return Pose3D.from_list(list(self.xyz_rpy), ref_frame=self.ref_frame)


@dataclass(frozen=True)
class FixedViewpoint:
    """A fixed-in-place camera viewpoint used to collect image observations."""

    camera_id: str


@dataclass(frozen=True)
class RelativeViewpoint:
    """An object-relative camera viewpoint used to collect image observations."""

    camera_id: str
    _pose: HashablePose3D
    _object_name: str

    def relative_to(self, obj_name: str) -> RelativeViewpoint:
        """Return a relative viewpoint modified to be relative to the named object."""
        new_pose = replace(self._pose, ref_frame=obj_name)
        return replace(self, _pose=new_pose, _object_name=obj_name)


Viewpoint = FixedViewpoint | RelativeViewpoint


@dataclass(frozen=True)
class ObservationSchema:
    """Maps object types to their observation schema."""

    _fixed_viewpoints: dict[type, frozenset[FixedViewpoint]]
    _relative_viewpoints: dict[type, frozenset[RelativeViewpoint]]

    def get_viewpoints(self, obj: object, obj_name: str) -> frozenset[Viewpoint]:
        """Retrieve instantiated viewpoints for the given object."""
        obj_viewpoints: set[Viewpoint] = set()

        for type_, fixed_viewpoints in self._fixed_viewpoints.items():
            if isinstance(obj, type_):
                obj_viewpoints.update(fixed_viewpoints)

        for type_, rel_viewpoints in self._relative_viewpoints.items():
            if isinstance(obj, type_):
                for rel_vp in rel_viewpoints:
                    obj_viewpoints.add(rel_vp.relative_to(obj_name))

        if not obj_viewpoints:
            raise ValueError(f"Schema specified no viewpoints for object '{obj_name}': {obj}.")

        return frozenset(obj_viewpoints)


ImageState = dict[Viewpoint, RGBImage]
"""An image state contains the current image for each viewpoint in the world."""


class ObjectCentricImageState:
    """An image-based state representation of an object-centric environment."""

    def __init__(self, objects: Mapping[str, object], schema: ObservationSchema) -> None:
        """Initialize the image-based state for the given objects.

        :param objects: Map from object names to object instances
        :param schema: Observation schema specifying viewpoint templates per object type
        """
        self._obj_viewpoints: dict[str, frozenset[Viewpoint]] = {
            obj_name: schema.get_viewpoints(obj, obj_name) for obj_name, obj in objects.items()
        }  # Instantiate fixed and relative viewpoints for each object

        all_viewpoints: set[Viewpoint] = set()
        for obj_viewpoints in self._obj_viewpoints.values():
            all_viewpoints.update(obj_viewpoints)

        self._all_viewpoints = frozenset(all_viewpoints)


# @dataclass(frozen=True)
# class ObjectImageState:
#     """The visual state of a single object."""

#     observations: dict[Viewpoint, RGBImage]


# 	def update_affected_views(self, skill: SkillInstance, schema: ObservationSchema) -> None:
# 		"""Update the current observations for all viewpoints potentially affected by a skill."""
# 		for obj in skill.arguments:
# 			viewpoints = schema.type_to_views[obj.type_]
# 			new_obs = collect_observations(viewpoints)
# 			self.curr_observations[obj.name].update(new_obs)


# Observations = dict[Viewpoint, Image] # Maps viewpoints to captured images

# def collect_observations(viewpoints: frozenset[Viewpoint]) -> Observations:
# 	"""Collect updated observations for the given viewpoints."""
# 	...robot handles this...

# @dataclass
# class ObjectCentricImageState:
# 	"""A collection of visual observations of an object-centric environment."""
# 	curr_observations: dict[str, Observations] # Map object names to their current observations

# 	@classmethod
# 	def from_objects(cls, objects: set[Object], schema: ObservationSchema) -> ObjectCentricImageState:
# 		"""Construct an ObjectCentricImageState for the given objects."""
# 		curr_obs = {}
# 		for obj in objects:
# 			viewpoints = schema.type_to_views[obj.type_]
# 			curr_obs[obj.name] = collect_observations(viewpoints)
# 		return cls(curr_observations=curr_obs)

# 	def update_affected_views(self, skill: SkillInstance, schema: ObservationSchema) -> None:
# 		"""Update the current observations for all viewpoints potentially affected by a skill."""
# 		for obj in skill.arguments:
# 			viewpoints = schema.type_to_views[obj.type_]
# 			new_obs = collect_observations(viewpoints)
# 			self.curr_observations[obj.name].update(new_obs)
