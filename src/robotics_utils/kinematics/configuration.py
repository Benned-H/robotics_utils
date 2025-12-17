"""Define a type alias to represent robot joint configurations."""

from typing import Dict

Configuration = Dict[str, float]
"""A map from joint names to positions (rad or m)."""
