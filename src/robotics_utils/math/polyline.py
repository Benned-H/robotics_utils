"""Define a class to represent 2D polylines."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from robotics_utils.kinematics import Point2D, Pose2D, Pose3D


class Polyline2D:
    """A 2D polyline stored as an (N, 2) NumPy array."""

    def __init__(self, points: Iterable[Point2D] | np.ndarray) -> None:
        """Initialize from an iterable of Point2Ds or an (N, 2) NumPy array."""
        if isinstance(points, np.ndarray):
            arr = np.asarray(points, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"Polyline2D expects array of shape (N, 2), got {arr.shape}.")
            self._xy = arr

        else:  # Initialize from an iterable of Point2Ds
            pts = [p.to_array() for p in points if isinstance(p, Point2D)]  # N arrays, shape (2,)
            if not pts:
                raise ValueError(f"Polyline2D requires at least one point, got: {points}.")
            self._xy = np.vstack(pts).astype(float)

        if self.n < 2:
            raise ValueError(f"Cannot construct Polyline2D with fewer than 2 points; got {self.n}")

    @classmethod
    def from_poses(cls, poses: Iterable[Pose2D | Pose3D]) -> Polyline2D:
        """Construct a Polyline2D from a sequence of 3D poses by using their (x, y)."""
        return cls([Point2D(p.position.x, p.position.y) for p in poses])

    @property
    def n(self) -> int:
        """Return the number of points in the polyline."""
        return int(self._xy.shape[0])

    @property
    def total_arclength_m(self) -> float:
        """Compute the total arclength (m) of the polyline."""
        if self.n <= 1:
            return 0.0

        return float(np.sum(self.arclength_per_segment_m()))

    def arclength_per_segment_m(self) -> np.ndarray:
        """Compute and return the arclength (m) of each segment in the polyline."""
        diffs_m = np.diff(self._xy, axis=0)  # (N-1, 2)
        return np.linalg.norm(diffs_m, axis=1)  # Compute vector norms along axis 1

    def cumulative_arclength_m(self) -> np.ndarray:
        """Compute the cumulative arclength (m) to each vertex from the start of the polyline."""
        if self.n == 1:
            return np.array([0.0], dtype=float)

        segment_lengths_m = self.arclength_per_segment_m()
        return np.concatenate(([0.0], np.cumsum(segment_lengths_m)))

    def interpolate_at_s(self, s_target: float) -> tuple[Point2D, int]:
        """Compute the point at the given arclength (m) along the polyline.

        :param s_target: Target arclength (m) along the polyline
        :return: ( interpolated point, segment index in [0..N-2] used for the interpolation )
        """
        s_m = self.cumulative_arclength_m()
        if s_target <= 0.0:
            return Point2D.from_array(self._xy[0]), 0

        s_target = min(s_target, self.total_arclength_m)

        # Find the segment whose upper vertex is the first with arclength >= target
        j = int(np.searchsorted(s_m, s_target, side="left"))

        # Compute segment index used for interpolation
        seg_idx = int(np.clip(j - 1, 0, self.n - 2))  # Ensure a valid segment index

        start_s, end_s = float(s_m[seg_idx]), float(s_m[seg_idx + 1])
        if end_s <= start_s:  # Handle degenerate segments by selecting their end point
            return Point2D.from_array(self._xy[seg_idx + 1]), seg_idx

        t = (s_target - start_s) / (end_s - start_s)
        t = float(np.clip(t, 0.0, 1.0))

        p_xy = (1.0 - t) * self._xy[seg_idx] + t * self._xy[seg_idx + 1]

        return Point2D.from_array(p_xy), seg_idx

    def tangent_yaw_at_segment(self, idx: int) -> float:
        """Return yaw (radians) of the direction of segment `idx`, between [-pi, pi]."""
        if not (0 <= idx < self.n - 1):
            raise IndexError(f"Segment index out of range: {idx} for N={self.n}.")

        dx, dy = (self._xy[idx + 1] - self._xy[idx]).tolist()
        return np.arctan2(dy, dx)  # Due to NumPy 1.17

    def project_point(self, point: Point2D) -> tuple[float, Point2D, int]:
        """Project a point onto the polyline.

        :param p: 2D point to be projected onto the line
        :return: Tuple containing (closest arclength, closest point, segment index)
        """
        if self.n == 1:  # One vertex, so it must be the closest
            return 0.0, Point2D.from_array(self._xy[0]), 0

        P = point.to_array()  # (2,)
        A = self._xy[:-1]  # Start of each segment; (M,2)
        B = self._xy[1:]  # End of each segment; (M,2)
        AB = B - A  # (M,2)
        AB2 = (AB * AB).sum(axis=1)  # Element-wise multiply and sum over axis 1 --> (M,)
        AP = P - A  # (M,2) due to broadcasting

        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(AB2 > 1e-18, (AP * AB).sum(axis=1) / AB2, 0.0)
        t = np.clip(t, 0.0, 1.0)  # Clamp into [0,1]; (M,)

        closest_points = A + t[:, None] * AB  # Closest point to p per segment; (M,2)
        distance_to_p = np.square(P - closest_points).sum(axis=1)  # (M,)

        best_i = int(np.argmin(distance_to_p))
        best_point = closest_points[best_i]

        seg_len_m = np.hypot(AB[:, 0], AB[:, 1])
        cumulative_arclengths = self.cumulative_arclength_m()
        best_arclength = float(cumulative_arclengths[best_i] + t[best_i] * seg_len_m[best_i])

        return best_arclength, Point2D.from_array(best_point), best_i
