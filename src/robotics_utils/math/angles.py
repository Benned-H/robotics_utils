"""Define utility functions for computations involving angles."""

import numpy as np


def normalize_angle(angle_rad: float) -> float:
    """Normalize the given angle (in radians) into the range [-pi, pi]."""
    while angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    while angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    return angle_rad


def angle_difference_rad(a_rad: float, b_rad: float) -> float:
    """Compute the absolute difference (in normalized radians) between two angles.

    :param a_rad: First angle (radians) in the difference
    :param b_rad: Second angle (radians) in the difference
    :return: Absolute angle difference (radians, between 0 and pi)
    """
    difference_rad = normalize_angle(a_rad - b_rad)
    return abs(difference_rad)
