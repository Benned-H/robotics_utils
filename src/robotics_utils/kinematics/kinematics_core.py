"""Implement core definitions for kinematics."""

from typing import Dict

DEFAULT_FRAME = "map"

Configuration = Dict[str, float]
"""A map from joint names to positions (rad or m)."""
