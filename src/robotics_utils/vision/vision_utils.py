"""Utility functions for image processing and computer vision."""

from __future__ import annotations

import platform
from pathlib import Path

import cv2
import numpy as np
import torch

RGB = tuple[int, int, int]
"""A tuple of (red, green, blue) integer values between 0 and 255."""


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an RGB image from the given path into a NumPy array."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot load image from nonexistent file: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image from path: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
