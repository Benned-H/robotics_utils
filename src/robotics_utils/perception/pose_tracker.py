"""Define a class to represent and manage estimated reference frames."""

from __future__ import annotations

from robotics_utils.kinematics.poses import Pose3D
from robotics_utils.math.averages_3d import average_poses


class PoseTracker:
    """Tracks relative poses by averaging across noisy pose estimates."""

    def __init__(self, max_estimates: int) -> None:
        """Initialize the pose tracker.

        :param max_estimates: Maximum number of estimates to retain per frame
        """
        self.max_estimates = max_estimates
        self._estimates: dict[str, list[Pose3D]] = {}

    @property
    def all_estimates(self) -> dict[str, list[Pose3D]]:
        """Retrieve a map from each tracked frame name to its current pose estimate."""
        return self._estimates.copy()

    def reset(self) -> None:
        """Clear all stored pose estimates without resetting known frames."""
        for frame_name in self._estimates:
            self._estimates[frame_name] = []

    def update(self, frame_name: str, pose: Pose3D) -> None:
        """Update the pose estimate for the named frame.

        :param frame_name: Name of the frame approximated by the given pose
        :param pose: New pose estimate of the frame
        """
        self._estimates.setdefault(frame_name, []).append(pose)

        # Enforce the sliding window size if necessary
        while len(self._estimates[frame_name]) > self.max_estimates:
            self._estimates[frame_name].pop(0)

    def get_pose_estimate(self, frame_name: str) -> Pose3D | None:
        """Compute the current pose estimate for the requested frame.

        :param frame_name: Name of the frame for which an estimate is computed
        :return: None if the frame is unknown or doesn't have an estimate yet.
        """
        if frame_name not in self._estimates or not self._estimates[frame_name]:
            return None

        return average_poses(poses=self._estimates[frame_name])
