"""Define a class to represent positions in 3D space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Point3D:
    """An (x,y,z) position in 3D space."""

    x: float
    y: float
    z: float

    @classmethod
    def identity(cls) -> Point3D:
        """Construct a Point3D corresponding to the identity translation."""
        return Point3D(0, 0, 0)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Point3D:
        """Construct a Point3D from a NumPy array."""
        if arr.shape != (3,):
            raise ValueError(f"Cannot construct Point3D from an array of shape {arr.shape}")

        return cls(arr[0], arr[1], arr[2])

    def to_array(self) -> NDArray[np.float64]:
        """Convert the 3D point to a NumPy array."""
        return np.array([self.x, self.y, self.z])

    def to_homogeneous_coordinate(self) -> NDArray[np.float64]:
        """Convert the 3D point into a homogeneous coordinate."""
        return np.array([self.x, self.y, self.z, 1.0])

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> Point3D:
        """Construct a Point3D instance from a sequence (e.g., list or tuple) of values."""
        if len(values) != 3:
            raise ValueError(f"Point3D expects 3 values, got {len(values)}")
        return Point3D(values[0], values[1], values[2])

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert the Point3D into a tuple of (x, y, z) coordinates."""
        return (self.x, self.y, self.z)

    def approx_equal(self, other: Point3D, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """Evaluate whether another Point3D is approximately equal to this one."""
        return np.allclose(self.to_array(), other.to_array(), rtol=rtol, atol=atol)
