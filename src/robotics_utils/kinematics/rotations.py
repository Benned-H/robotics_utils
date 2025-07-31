"""Define classes to represent 3D rotations and orientations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import astuple, dataclass

import numpy as np
from numpy.typing import NDArray
from trimesh.transformations import (
    euler_from_matrix,
    euler_from_quaternion,
    euler_matrix,
    quaternion_from_euler,
    quaternion_from_matrix,
    quaternion_matrix,
)


@dataclass
class EulerRPY:
    """A 3D rotation represented using three fixed-frame Euler angles."""

    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    def __iter__(self) -> Iterator[float]:
        """Provide an iterator over the roll, pitch, and yaw values."""
        yield from astuple(self)

    @classmethod
    def identity(cls) -> EulerRPY:
        """Construct EulerRPY angles corresponding to the identity rotation."""
        return EulerRPY(0, 0, 0)

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> EulerRPY:
        """Construct Euler angles from a sequence of angle values (in radians)."""
        if len(values) != 3:
            raise ValueError(f"EulerRPY expects 3 values, got {len(values)}")
        return EulerRPY(values[0], values[1], values[2])

    @classmethod
    def from_homogeneous_matrix(cls, matrix: NDArray[np.float64]) -> EulerRPY:
        """Construct Euler angles from a 4x4 homogeneous transformation matrix."""
        if matrix.shape != (4, 4):
            raise ValueError(f"EulerRPY expects a 4x4 homogeneous matrix, got {matrix.shape}")
        roll, pitch, yaw = euler_from_matrix(matrix, axes="sxyz")
        return EulerRPY(roll, pitch, yaw)

    def to_homogeneous_matrix(self) -> NDArray[np.float64]:
        """Convert the Euler RPY angles into an equivalent homogeneous transformation matrix."""
        return euler_matrix(self.roll_rad, self.pitch_rad, self.yaw_rad, axes="sxyz")

    def to_quaternion(self) -> Quaternion:
        """Convert the Euler angles into an equivalent unit quaternion."""
        w, x, y, z = quaternion_from_euler(self.roll_rad, self.pitch_rad, self.yaw_rad, axes="sxyz")
        return Quaternion(x=x, y=y, z=z, w=w)


@dataclass
class Quaternion:
    """A unit quaternion representing a 3D orientation."""

    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        """Normalize the quaternion after it is initialized."""
        self.normalize()

    def normalize(self) -> None:
        """Normalize the quaternion to ensure it is a unit quaternion."""
        norm = float(np.linalg.norm(self.to_array()))
        if norm == 0:
            raise ValueError(f"Cannot normalize a zero-valued quaternion: {self}")

        self.x /= norm
        self.y /= norm
        self.z /= norm
        self.w /= norm

    @classmethod
    def identity(cls) -> Quaternion:
        """Construct a Quaternion corresponding to the identity rotation."""
        return Quaternion(0, 0, 0, 1)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Quaternion:
        """Construct a quaternion from a NumPy array of the form [x,y,z,w]."""
        if arr.shape != (4,):
            raise ValueError(f"Quaternion expects a 4-vector, got {arr.shape}")

        return cls(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def to_array(self) -> NDArray[np.float64]:
        """Convert the quaternion to a NumPy array of the form [x,y,z,w]."""
        return np.array([self.x, self.y, self.z, self.w])

    def to_euler_rpy(self) -> EulerRPY:
        """Convert the quaternion into equivalent Euler roll, pitch, and yaw angles."""
        r, p, y = euler_from_quaternion(quaternion=[self.w, self.x, self.y, self.z], axes="sxyz")
        return EulerRPY(r, p, y)

    @classmethod
    def from_homogeneous_matrix(cls, matrix: NDArray[np.float64]) -> Quaternion:
        """Construct a quaternion from a 4x4 homogeneous transformation matrix."""
        if matrix.shape != (4, 4):
            raise ValueError(f"Quaternion expects a 4x4 homogeneous matrix, got {matrix.shape}")
        w, x, y, z = quaternion_from_matrix(matrix)  # Note: trimesh puts w (q's real value) first
        return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))

    def to_homogeneous_matrix(self) -> NDArray[np.float64]:
        """Convert the quaternion to a 4x4 homogeneous transformation matrix."""
        return quaternion_matrix(quaternion=[self.w, self.x, self.y, self.z])

    def approx_equal(self, other: Quaternion, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """Evaluate whether another Quaternion is approximately equal to this one.

        Note: A quaternion is considered equal to its negation, as they express the same rotation.
        """
        self_array = self.to_array()
        other_array = other.to_array()

        return np.allclose(self_array, other_array, rtol=rtol, atol=atol) or np.allclose(
            -self_array,
            other_array,
            rtol=rtol,
            atol=atol,
        )
