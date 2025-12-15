"""Define a class to represent an (x,y) pixel coordinate in an image."""

from __future__ import annotations

import numpy as np


class PixelXY:
    """An (x,y) coordinate of a pixel in an image.

    - x increases from left to right, starting at 0.
    - y increases from top to bottom, starting at 0.
    """

    def __init__(self, xy: np.typing.NDArray | tuple[int, int]) -> None:
        """Initialize the pixel using the given (x,y) coordinate."""
        self.xy = np.asarray(xy, dtype=int)

        if self.xy.size != 2:
            raise ValueError(f"Cannot construct a PixelXY from an array of size {self.xy.size}.")

    def __add__(self, other: PixelXY) -> PixelXY:
        """Find the sum of this PixelXY and another."""
        return PixelXY(self.xy + other.xy)

    def __str__(self) -> str:
        """Return a readable string representation of the pixel."""
        return f"({self.x}, {self.y})"

    def __sub__(self, other: PixelXY) -> PixelXY:
        """Find the difference of this PixelXY and another."""
        return PixelXY(self.xy - other.xy)

    @classmethod
    def from_json(cls, json_data: dict) -> PixelXY:
        """Construct a PixelXY from JSON data."""
        return PixelXY((json_data["x"], json_data["y"]))

    def to_json(self) -> dict:
        """Convert the PixelXY to a dictionary of JSON data."""
        return {"x": self.x, "y": self.y}

    @property
    def x(self) -> int:
        """Retrieve the x-coordinate of this pixel."""
        return self.xy[0].item()

    @property
    def y(self) -> int:
        """Retrieve the y-coordinate of this pixel."""
        return self.xy[1].item()
