"""Define classes to represent 3D rotations and orientations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transforms3d.euler import euler2mat, euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import mat2quat, quat2mat


def normalize_angle(angle_rad: float) -> float:
    """Normalize the given angle (in radians) into the range [-pi, pi]."""
    while angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    while angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    return angle_rad


@dataclass
class EulerRPY:
    """A 3D rotation represented using three fixed-frame Euler angles."""

    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    @classmethod
    def identity(cls) -> EulerRPY:
        """Construct EulerRPY angles corresponding to the identity rotation."""
        return EulerRPY(0, 0, 0)

    @classmethod
    def from_rotation_matrix(cls, r_matrix: np.ndarray) -> EulerRPY:
        """Construct Euler angles from a 3x3 rotation matrix."""
        if r_matrix.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 rotation matrix; received {r_matrix.shape}")
        roll, pitch, yaw = mat2euler(r_matrix, axes="sxyz")
        return EulerRPY(roll, pitch, yaw)

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert the Euler RPY angles into an equivalent rotation matrix."""
        return euler2mat(self.roll_rad, self.pitch_rad, self.yaw_rad, axes="sxyz")

    def to_quaternion(self) -> Quaternion:
        """Convert the Euler angles into an equivalent unit quaternion."""
        w, x, y, z = euler2quat(self.roll_rad, self.pitch_rad, self.yaw_rad, axes="sxyz")
        return Quaternion(x=x, y=y, z=z, w=w)

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert the Euler angles into a tuple of (roll, pitch, yaw) angles (in radians)."""
        return (self.roll_rad, self.pitch_rad, self.yaw_rad)


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
            raise ValueError(f"Expected a four-element vector but received shape {arr.shape}")

        return cls(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def to_array(self) -> np.ndarray:
        """Convert the quaternion to a NumPy array of the form [x,y,z,w]."""
        return np.array([self.x, self.y, self.z, self.w])

    def to_euler_rpy(self) -> EulerRPY:
        """Convert the quaternion into equivalent Euler roll, pitch, and yaw angles."""
        roll, pitch, yaw = quat2euler(quaternion=[self.w, self.x, self.y, self.z], axes="sxyz")
        return EulerRPY(roll, pitch, yaw)

    @classmethod
    def from_rotation_matrix(cls, r_matrix: np.ndarray) -> Quaternion:
        """Construct a quaternion from a 3x3 rotation matrix."""
        if r_matrix.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 rotation matrix; received {r_matrix.shape}")
        w, x, y, z = mat2quat(r_matrix)  # Note: transforms3d quaternions have w first
        return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert the quaternion to a 3x3 rotation matrix."""
        return quat2mat(q=[self.w, self.x, self.y, self.z])

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
