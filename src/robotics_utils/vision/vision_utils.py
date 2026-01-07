"""Utility functions for image processing and computer vision."""

from __future__ import annotations

import platform
from typing import Tuple

import distinctipy
import torch

RGB = Tuple[int, int, int]
"""A tuple of (red, green, blue) integer values between 0 and 255."""


def get_rgb_colors(n: int) -> list[RGB]:
    """Generate a list of N maximally visually distinct RGB colors.

    :param n: Number of colors to generate
    :return: List of RGB tuples representing the colors
    """
    return [distinctipy.get_rgb256(rgb) for rgb in distinctipy.get_colors(n)]


def assign_random_colors(strings: list[str]) -> dict[str, RGB]:
    """Assign random visually distinct colors to the given strings.

    :param strings: Strings to be assigned RGB colors
    :return: Dictionary mapping the strings to their assigned RGB colors
    """
    colors = get_rgb_colors(len(strings))
    return dict(zip(strings, colors))


def determine_pytorch_device() -> torch.device:
    """Determine which PyTorch device to use."""
    if torch.cuda.is_available():  # Use CUDA on Linux if available
        return torch.device("cuda:0")  # Specify device index for CUDA operations
    if (
        platform.system() == "Darwin"
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")  # Use Metal on macOS if available

    return torch.device("cpu")  # Otherwise, fall back to CPU
