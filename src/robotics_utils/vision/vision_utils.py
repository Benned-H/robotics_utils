"""Utility functions for image processing and computer vision."""

from __future__ import annotations

import platform

import torch

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
