"""Define classes to represent 3D planes and rectangles on them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Plane3D:
    """A plane in 3D space defined by a point and normal vector.

    Reference: https://mathworld.wolfram.com/Plane.html
    """

    point: NDArray[np.float64]
    """A point on the plane of shape (3,)."""

    normal: NDArray[np.float64]
    """Unit normal vector to the plane of shape (3,)."""

    @property
    def d(self) -> float:
        """Compute the plane equation constant: ax + by + cz = d."""
        return float(np.dot(self.normal, self.point))

    @property
    def equation_string(self) -> str:
        """Retrieve a string expressing the equation of the plane."""
        a_str = f"{self.normal[0]:.3f}"
        b_str = f"{self.normal[1]:.3f}"
        c_str = f"{self.normal[2]:.3f}"

        return f"{a_str}x + {b_str}y + {c_str}z = {self.d:.3f}"


@dataclass(frozen=True)
class Rectangle3D:
    """A rectangle in 3D space, defined by corner points on a plane."""

    corners: NDArray[np.float64]  # Shape (4, 3): corners in order (TL, TR, BR, BL)
    normal: NDArray[np.float64]  # Shape (3,): plane normal

    @property
    def center(self) -> NDArray[np.float64]:
        """Compute the center point of the rectangle."""
        return self.corners.mean(axis=0)

    @property
    def width(self) -> float:
        """Compute the width of the rectangle (horizontal extent)."""
        return float(np.linalg.norm(self.corners[1] - self.corners[0]))

    @property
    def height(self) -> float:
        """Compute the height of the rectangle (vertical extent)."""
        return float(np.linalg.norm(self.corners[3] - self.corners[0]))

    def find_basis_vectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Find the (u, v) basis vectors for the plane defined by this rectangle.

        :return: Tuple containing (u, v) vectors, each of shape (3,)
        """
        u = self.corners[1] - self.corners[0]  # Rectangle's top edge gives first basis
        u = u / np.linalg.norm(u)
        v = np.cross(self.normal, u)
        return u, v

    def find_local_coordinates(self) -> NDArray[np.float64]:
        """Project the rectangle corners onto the plane's 2D coordinate system.

        :return: Array of shape (4, 2) containing the projected 2D coordinates
        """
        u, v = self.find_basis_vectors()

        # Project the corners onto the plane's 2D coordinate system
        local_coords = np.zeros((4, 2))
        for i, corner in enumerate(self.corners):
            offset = corner - self.center
            local_coords[i, 0] = np.dot(offset, u)
            local_coords[i, 1] = np.dot(offset, v)
        return local_coords
