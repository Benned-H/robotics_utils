"""Define a class to represent positions in 3D space."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


@dataclass
class Point3D:
    """An (x,y,z) position in 3D space."""

    x: float
    y: float
    z: float

    def __iter__(self) -> Iterator[float]:
        """Provide an iterator over the point's (x,y,z) coordinates."""
        yield from astuple(self)

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

    @classmethod
    def from_homogeneous_coordinate(cls, coord: NDArray) -> Point3D:
        """Construct a Point3D instance from a homogeneous coordinate given as a NumPy array."""
        if coord.shape != (4,):
            raise ValueError(f"Homogeneous coordinate should have shape (4,), got {coord.shape}.")
        return Point3D.from_array(coord[:3])

    def to_homogeneous_coordinate(self) -> NDArray[np.float64]:
        """Convert the 3D point into a homogeneous coordinate."""
        return np.array([self.x, self.y, self.z, 1.0])

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> Point3D:
        """Construct a Point3D instance from a sequence (e.g., list or tuple) of values."""
        if len(values) != 3:
            raise ValueError(f"Point3D expects 3 values, got {len(values)}")
        return Point3D(float(values[0]), float(values[1]), float(values[2]))

    def approx_equal(self, other: Point3D, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """Evaluate whether another Point3D is approximately equal to this one."""
        return np.allclose(self.to_array(), other.to_array(), rtol=rtol, atol=atol)
