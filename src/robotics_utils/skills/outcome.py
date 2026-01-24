"""Define a minimal dataclass to represent the outcome of an action or skill."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

OutputT = TypeVar("OutputT")
"""Type variable representing output data associated with an outcome."""


@dataclass(frozen=True)
class Outcome(Iterable, Generic[OutputT]):
    """An outcome (and optional output value) from an action or skill execution."""

    success: bool
    message: str
    output: OutputT | None = None
    """Optional output value resulting from the action or skill (defaults to None)."""

    def __iter__(self) -> Iterator:
        """Return an iterator over the values of the outcome (skips its output if it's None)."""
        if self.output is not None:
            return iter((self.success, self.message, self.output))

        return iter((self.success, self.message))
