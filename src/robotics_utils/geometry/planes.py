"""Define classes to represent 3D planes and rectangles on them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PlaneBasis:
    """Orthogonal basis vectors (u, v) spanning a plane.

    Together with the plane's normal vector, these form a right-handed coordinate system.
    """

    u: NDArray[np.float64]
    """First basis vector in the plane of shape (3,)."""

    v: NDArray[np.float64]
    """Second basis vector in the plane of shape (3,)."""


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

    def find_basis(self) -> PlaneBasis:
        """Find orthogonal basis vectors (u, v) spanning this plane.

        The basis vectors form a right-handed coordinate system with the plane normal.

        :return: PlaneBasis containing orthogonal unit vectors u and v
        """
        arbitrary = (
            np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        )  # Start with an arbitrary vector not parallel to the plane's normal

        # Create a basis vector using the Gram-Schmidt process, then normalize
        # Reference: https://math.stackexchange.com/a/695879
        u = arbitrary - np.dot(arbitrary, self.normal) * self.normal
        u = u / np.linalg.norm(u)
        v = np.cross(self.normal, u)

        return PlaneBasis(u=u, v=v)


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

    def find_basis_vectors(self) -> PlaneBasis:
        """Find the (u, v) basis vectors for the plane defined by this rectangle.

        :return: PlaneBasis containing orthogonal unit vectors u and v, each of shape (3,)
        """
        u = self.corners[1] - self.corners[0]  # Rectangle's top edge gives first basis
        u = u / np.linalg.norm(u)
        v = np.cross(self.normal, u)
        return PlaneBasis(u=u, v=v)

    def find_local_coordinates(self) -> NDArray[np.float64]:
        """Project the rectangle corners onto the plane's 2D coordinate system.

        :return: Array of shape (4, 2) containing the projected 2D coordinates
        """
        basis = self.find_basis_vectors()

        # Project the corners onto the plane's 2D coordinate system
        local_coords = np.zeros((4, 2))
        for i, corner in enumerate(self.corners):
            offset = corner - self.center
            local_coords[i, 0] = np.dot(offset, basis.u)
            local_coords[i, 1] = np.dot(offset, basis.v)
        return local_coords
