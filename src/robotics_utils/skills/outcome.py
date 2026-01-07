"""Define a minimal dataclass to represent the outcome of an action or skill."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    """An outcome (and optional output value) from an action or skill execution."""

    success: bool
    message: str
    output: object | None = None
    """Optional output value resulting from the action or skill (defaults to None)."""
