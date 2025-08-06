"""Define a class to record transforms from /tf for future playback."""

from __future__ import annotations

from pathlib import Path

import rospy

from robotics_utils.io.yaml_utils import export_yaml_data
from robotics_utils.kinematics import Pose3D
from robotics_utils.ros.transform_manager import TransformManager


class TransformRecorder:
    """Listen to /tf transforms and record relative motion from an initial state."""

    def __init__(self, reference_frame: str, tracked_frame: str) -> None:
        """Initialize the transform recorder with reference and tracked frames.

        :param reference_frame: "Fixed" reference frame for the recording
        :param tracked_frame: Frame for which relative motion will be recorded
        """
        self.reference_frame = reference_frame
        self.tracked_frame = tracked_frame
        self.tracked_relative_poses: list[Pose3D] = []  # Tracked frame w.r.t. its original pose

        # Initial pose of the reference frame w.r.t. the tracked frame (yes, I meant to say that)
        self.initial_pose_t_r: Pose3D | None = None

    def get_relative_pose(self) -> Pose3D | None:
        """Look up the current pose of the tracked frame relative to its initial state.

        :return: Relative pose of tracked frame w.r.t. its original pose, or None if lookup fails
        """
        curr_pose_r_t = TransformManager.lookup_transform(self.tracked_frame, self.reference_frame)
        if curr_pose_r_t is None:
            return None

        # Record the initial pose if this is the first successful lookup
        if self.initial_pose_t_r is None:
            self.initial_pose_t_r = curr_pose_r_t.inverse(pose_frame=self.tracked_frame)
            rospy.loginfo(f"Captured an initial pose: {self.initial_pose_t_r}")
            return Pose3D.identity()  # Initial pose relative to itself = Identity transform

        # Compute the current relative pose of the tracked frame w.r.t. its initial pose
        return self.initial_pose_t_r @ curr_pose_r_t

    def update(self) -> None:
        """Update the tracked transforms."""
        relative_pose = self.get_relative_pose()
        if relative_pose is not None:
            self.tracked_relative_poses.append(relative_pose)

    def save_to_file(self, output_path: Path) -> None:
        """Save the recorded relative transforms to file."""
        yaml_dicts = [pose.to_yaml_dict() for pose in self.tracked_relative_poses]
        export_yaml_data(yaml_dicts, output_path)
        rospy.loginfo(f"Saved {len(self.tracked_relative_poses)} poses to {output_path}")
