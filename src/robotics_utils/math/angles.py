"""Define utility functions for computations involving angles."""

import numpy as np


def normalize_angle(angle_rad: float) -> float:
    """Normalize the given angle (in radians) into the range [-pi, pi]."""
    while angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    while angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    return angle_rad
