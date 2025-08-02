"""Utility functions for image processing and computer vision."""

from __future__ import annotations

import platform
from collections.abc import Iterator
from dataclasses import astuple, dataclass

import numpy as np
import torch
from numpy.typing import NDArray

RGB = tuple[int, int, int]
"""A tuple of (red, green, blue) integer values between 0 and 255."""


def determine_pytorch_device() -> torch.device:
    """Determine which PyTorch device to use."""
    if torch.cuda.is_available():  # Use CUDA on Linux if available
        return torch.device("cuda")
    if (
        platform.system() == "Darwin"
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")  # Use Metal on macOS if available

    return torch.device("cpu")  # Otherwise, fall back to CPU


@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrinsic parameters for a pinhole model camera.

    Definitions:
        - Principal axis - Line perpendicular to the image plane through the camera pinhole.
        - Principal point - Where the principal axis intersects with the image plane, relative
            to the origin of the film (i.e., the pinhole's location if projected onto the film).

    Reference: https://ksimek.github.io/2013/08/13/intrinsic/
    """

    fx: float  # Focal length (pixels) in x
    fy: float  # Focal length (pixels) in y
    x0: float  # Principal point offset in x
    y0: float  # Principal point offset in y

    def __iter__(self) -> Iterator[float]:
        """Provide an iterator over the camera intrinsics: [fx, fy, x0, y0]."""
        yield from astuple(self)

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert the camera intrinsic parameters into a 3x3 intrinsic matrix."""
        return np.array([[self.fx, 0, self.x0], [0, self.fy, self.y0], [0, 0, 1]])
