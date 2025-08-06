"""Define a class to aggregate and average estimated poses."""

from __future__ import annotations

from collections import defaultdict, deque

from robotics_utils.kinematics import Pose3D
from robotics_utils.math.averages_3d import average_poses


class PoseEstimateAverager:
    """Aggregate noisy pose estimates per frame using a fixed-size sliding window."""

    def __init__(self, window_size: int) -> None:
        """Initialize the pose averager with a maximum window size (per frame)."""
        self._window_size = window_size
        self._estimates: dict[str, deque[Pose3D]] = defaultdict(lambda: deque(maxlen=window_size))
        self._averages: dict[str, Pose3D | None] = {}

    @property
    def all_stored_estimates(self) -> dict[str, list[Pose3D]]:
        """Retrieve a read-only map of each frame name to its currently stored pose estimates."""
        return {frame: list(estimates) for frame, estimates in self._estimates.items()}

    @property
    def all_available_averages(self) -> dict[str, Pose3D]:
        """Retrieve a map of all available (frame name, averaged pose) pairs."""
        return {frame: opt_avg for frame, opt_avg in self._averages.items() if opt_avg is not None}

    def update(self, frame_name: str, pose: Pose3D) -> None:
        """Add a new pose estimate for the given frame.

        :param frame_name: Identifier of the relevant reference frame
        :param pose: New noisy pose estimate
        """
        self._estimates[frame_name].append(pose)
        self._averages.pop(frame_name, None)  # Clear the cached average for this frame

    def get(self, frame_name: str) -> Pose3D | None:
        """Retrieve, or compute and cache, the current averaged pose for the requested frame.

        :param frame_name: Reference frame identifier
        :return: Averaged Pose3D, or None if the frame has no estimates
        """
        if frame_name in self._averages:
            return self._averages[frame_name]  # Return the cached average if previously computed

        poses = self._estimates.get(frame_name)
        self._averages[frame_name] = average_poses(poses) if poses else None
        return self._averages[frame_name]

    def compute_all_averages(self) -> dict[str, Pose3D | None]:
        """Compute and return a map from each frame name to its averaged pose estimate."""
        return {frame: self.get(frame) for frame in self._estimates}

    def reset(self) -> None:
        """Clear all stored pose estimates and cached averages."""
        self._estimates.clear()
        self._averages.clear()
