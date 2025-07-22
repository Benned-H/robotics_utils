"""Define a class to represent positions in 3D space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    def to_array(self) -> np.ndarray:
        """Convert the 3D point to a NumPy array."""
        return np.array([self.x, self.y, self.z])

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert the Point3D into a tuple of (x, y, z) coordinates."""
        return (self.x, self.y, self.z)

    def approx_equal(self, other: Point3D, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """Evaluate whether another Point3D is approximately equal to this one."""
        result = np.allclose(self.to_array(), other.to_array(), rtol=rtol, atol=atol)

        if not result:
            print(f"Self P: {self}")
            print(f"Other P: {other}")

        return result
