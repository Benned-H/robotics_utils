"""Utility functions for computing weighted averages of 3D geometric primitives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from robotics_utils.kinematics import DEFAULT_FRAME, Point3D, Pose3D, Quaternion

OptionalWeights = Sequence[float] | None


def average_positions(positions: Sequence[Point3D], weights: OptionalWeights = None) -> Point3D:
    """Compute a weighted average of 3D positions.

    :param positions: Collection of (x,y,z) coordinates
    :param weights: Optional sequence of per-position weights (defaults to uniform weighting)
    :return: Point3D result of the weighted average
    """
    if not positions:
        raise ValueError(f"Cannot compute average of zero positions: {positions}.")

    if weights is None:
        weights = [1.0] * len(positions)
    if len(positions) != len(weights):
        lp = len(positions)
        lw = len(weights)
        raise ValueError(f"Positions and weights must have the same length, got {lp} and {lw}.")

    arr = np.vstack([p.to_array() for p in positions])  # (N, 3)
    avg = np.average(arr, axis=0, weights=weights)
    return Point3D.from_array(avg)


def average_quaternions(qs: Sequence[Quaternion], weights: OptionalWeights = None) -> Quaternion:
    """Compute a maximum-likelihood average of quaternions.

    Uses eigen-decomposition of the weighted sum of outer products.
        Reference: Method 2 from this answer: https://math.stackexchange.com/a/3435296/614782

    :param qs: Collection of unit quaternions representing 3D rotations
    :param weights: Optional sequence of per-quaternion weights (defaults to uniform weighting)
    :return: Quaternion result of the weighted average
    """
    if not qs:
        raise ValueError(f"Cannot compute average of zero quaternions: {qs}.")

    if weights is None:
        weights = [1.0] * len(qs)
    if len(qs) != len(weights):
        lq = len(qs)
        lw = len(weights)
        raise ValueError(f"Quaternions and weights must have the same length, got {lq} and {lw}.")

    # Accumulate weighted outer products
    matrix = np.zeros((4, 4))
    for q, w in zip(qs, weights, strict=True):
        v = q.to_array().reshape(4, 1)
        matrix += w * (v @ v.T)  # Sum of weighted outer products

    # Compute the principal eigenvector of the symmetric matrix
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest_idx = np.argmax(eigenvalues)
    principal = eigenvectors[:, largest_idx]
    return Quaternion.from_array(principal)


def average_poses(poses: Sequence[Pose3D], weights: OptionalWeights = None) -> Pose3D:
    """Compute the weighted average of 3D poses.

    :param poses: Collection of poses in 3D space (must be non-empty)
    :param weights: Optional sequence of per-pose weights (defaults to uniform weighting)
    :return: Pose3D result of the weighted average
    """
    if not poses:
        raise ValueError(f"Cannot compute average of zero poses: {poses}.")

    if weights is None:
        weights = [1.0] * len(poses)
    if len(poses) != len(weights):
        lp = len(poses)
        lw = len(weights)
        raise ValueError(f"Poses and weights must have the same length, got {lp} and {lw}.")

    avg_position = average_positions([pose.position for pose in poses], weights)
    avg_orientation = average_quaternions([pose.orientation for pose in poses], weights)

    # If the given poses have a common frame, replicate that frame in the output
    first_ref_frame = poses[0].ref_frame
    default_frame = DEFAULT_FRAME
    if all(p.ref_frame == first_ref_frame for p in poses):
        default_frame = first_ref_frame

    return Pose3D(avg_position, avg_orientation, ref_frame=default_frame)
