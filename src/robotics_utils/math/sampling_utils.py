"""Define utility functions for sampling deterministic or random values."""

from __future__ import annotations

import numpy as np


def sample_centered_angles(center_rad: float, num_angles: int = 8) -> list[float]:
    """Sample angles alternating symmetrically from a central starting angle (radians).

    :param center_rad: Angle (radians) starting the sequence of samples
    :param num_angles: Total number of angle samples generated
    :return: List of sampled angles, evenly spaced around the full circle
    """
    if num_angles < 1:
        raise ValueError(f"Cannot sample {num_angles} angles.")

    step_rad = 2 * np.pi / (num_angles + 1)  # Step (in radians) between neighbor samples
    steps_away = 1  # Steps away from the center angle
    angles = [center_rad]

    while len(angles) < num_angles:
        angles.append(center_rad + steps_away * step_rad)
        if len(angles) < num_angles:
            angles.append(center_rad - steps_away * step_rad)
        steps_away += 1

    return angles[:num_angles]
