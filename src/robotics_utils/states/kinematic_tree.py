"""Represent the geometric state of an environment as a kinematic tree."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from robotics_utils.io import console

if TYPE_CHECKING:
    from robotics_utils.collision_models import CollisionModel
    from robotics_utils.spatial import Pose3D


class KinematicTree:
    """A tree of coordinate frames specifying relative poses between entities."""

    def __init__(self, root_frame: str) -> None:
        """Initialize the kinematic tree's member variables based on its root frame."""
        self.root_frame = root_frame

        self._frames: dict[str, Pose3D] = {}
        """A map from the name of each frame to its relative pose."""

        self._children: defaultdict[str, set[str]] = defaultdict(set)
        """A map from the name of each frame to the set of its child frames."""

        self._collision_models: dict[str, CollisionModel] = {}
        """A map from the name of each frame to its (optional) attached collision geometry."""

    @property
    def all_known_poses(self) -> dict[str, Pose3D]:
        """Create and return a dictionary mapping each frame name to its 3D pose (if known)."""
        return self._frames

    def valid_frame(self, frame_name: str) -> bool:
        """Evaluate whether the given frame name is valid in the kinematic tree."""
        return (frame_name in self._frames) or (frame_name == self.root_frame)

    def get_poses(self, frame_names: set[str]) -> dict[str, Pose3D]:
        """Retrieve the relative poses of the named frames (if known).

        :param frame_names: Names of reference frames to find poses for
        :return: Map from frame names to their 3D poses, for frames with known poses
        """
        return {frame: pose for frame, pose in self._frames.items() if frame in frame_names}

    def get_parent_frame(self, child_frame: str) -> str | None:
        """Retrieve the parent frame of the given child frame.

        :param child_frame: Frame whose parent frame is found
        :return: Name of the parent frame of the child frame (None if parent frame is unknown)
        """
        child_pose = self._frames.get(child_frame)
        return None if child_pose is None else child_pose.ref_frame

    def set_pose(self, frame_name: str, pose: Pose3D) -> None:
        """Update the named frame with the given relative pose.

        :param frame_name: Name of the reference frame added or updated
        :param pose: New relative pose of the frame
        """
        prev_parent_frame = self.get_parent_frame(frame_name)
        if prev_parent_frame is not None:  # Remove the frame from its previous parent's children
            self._children[prev_parent_frame].remove(frame_name)

        self._frames[frame_name] = pose
        self._children[pose.ref_frame].add(frame_name)  # Add the frame to its new parent's children

    def clear_pose(self, frame_name: str) -> dict[str, Pose3D | None]:
        """Clear the pose of the named frame and its descendants and return their poses, if known.

        :param frame_name: Name of the reference frame to be cleared
        :return: Map from names of cleared frames to their previous pose, if known (else None)
        """
        cleared_poses: dict[str, Pose3D | None] = {}

        # Clear any children frames first
        while self._children[frame_name]:
            child_frame = self._children[frame_name].pop()
            child_cleared_poses = self.clear_pose(child_frame)
            cleared_poses.update(child_cleared_poses)

        parent_frame = self.get_parent_frame(frame_name)
        if parent_frame is not None:
            self._children[parent_frame].discard(frame_name)

        cleared_poses[frame_name] = self._frames.pop(frame_name, None)
        return cleared_poses

    def set_collision_model(self, frame_name: str, collision_model: CollisionModel) -> None:
        """Set the collision geometry attached to the named frame.

        :param frame_name: Name of the frame whose collision model is updated
        :param collision_model: Rigid-body collision geometry (primitive shape(s) and/or mesh(es))
        """
        if not self.valid_frame(frame_name):
            console.print(f"[yellow]Invalid frame '{frame_name}' had its collision model set.[/]")

        self._collision_models[frame_name] = collision_model

    def get_collision_model(self, frame_name: str) -> CollisionModel | None:
        """Retrieve the collision model attached to the named frame.

        :param frame_name: Name of the frame of the returned collision geometry
        :return: Collision model for the frame, or None if the frame has no attached geometry
        """
        return self._collision_models.get(frame_name)

    def clear_collision_model(self, frame_name: str) -> CollisionModel | None:
        """Clear the collision model attached to the named frame.

        :param frame_name: Name of the reference frame cleared of its collision model
        :return: Previous collision model attached to the frame, if one existed, else None
        """
        if not self.valid_frame(frame_name):
            console.print(
                f"[yellow]Invalid frame '{frame_name}' had its collision model cleared.[/]",
            )
            return None

        return self._collision_models.pop(frame_name, None)
