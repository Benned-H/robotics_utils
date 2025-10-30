"""Define a class to aggregate and average estimated poses."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

from robotics_utils.math.averages_3d import average_poses

if TYPE_CHECKING:
    from robotics_utils.kinematics import Pose3D


class PoseEstimateAverager:
    """Aggregate noisy pose estimates per reference frame using a fixed-size sliding window."""

    def __init__(self, window_size: int) -> None:
        """Initialize the pose averager with a maximum sliding window size."""
        self._window_size = window_size
        self._data: dict[str, deque[Pose3D]] = defaultdict(lambda: deque(maxlen=window_size))
        """A map from each frame's name to a FIFO queue of estimated poses for that frame."""

        self._averages: dict[str, Pose3D | None] = {}
        """A map from each frame's name to its latest averaged pose estimate, if it's computed."""

    @property
    def all_stored_data(self) -> dict[str, list[Pose3D]]:
        """Retrieve a read-only map of each frame name to its currently stored pose estimates."""
        return {frame: list(estimates) for frame, estimates in self._data.items()}

    @property
    def all_available_averages(self) -> dict[str, Pose3D]:
        """Retrieve a map of all available (frame name, averaged pose) pairs."""
        return {frame: avg for frame, avg in self._averages.items() if avg is not None}

    def update(self, frame_name: str, pose: Pose3D) -> None:
        """Add a new pose estimate for the given frame.

        :param frame_name: Identifier of the relevant reference frame
        :param pose: New noisy pose estimate
        """
        curr_data = self._data[frame_name]
        if curr_data and curr_data[0].ref_frame != pose.ref_frame:
            raise ValueError(
                f"Pose estimates for frame '{frame_name}' have different parent "
                f"frames: {curr_data[0].ref_frame} and {pose.ref_frame}.",
            )

        self._data[frame_name].append(pose)
        self._averages.pop(frame_name, None)  # Clear the cached average for this frame

    def get(self, frame_name: str) -> Pose3D | None:
        """Retrieve, or compute and cache, the current averaged pose for the requested frame.

        :param frame_name: Reference frame identifier
        :return: Averaged Pose3D, or None if the frame has no stored pose estimates
        """
        if frame_name in self._averages:
            return self._averages[frame_name]  # Return the cached average if previously computed

        poses_data = self._data.get(frame_name)
        self._averages[frame_name] = average_poses(poses_data) if poses_data else None
        return self._averages[frame_name]

    def compute_all_averages(self) -> dict[str, Pose3D | None]:
        """Compute and return a map from each frame name to its averaged pose estimate."""
        return {frame: self.get(frame) for frame in self._data}

    def reset_frame(self, frame_name: str) -> Pose3D | None:
        """Clear any stored pose estimates and cached averages for the named frame.

        :param frame_name: Name of the frame to be cleared
        :return: Previously stored pose estimate for the frame, if one exists, else None
        """
        latest_average = self.get(frame_name)

        self._data[frame_name].clear()
        self._averages[frame_name] = None

        return latest_average

    def reset_all(self) -> None:
        """Clear all stored pose estimates and cached averages."""
        self._data.clear()
        self._averages.clear()
