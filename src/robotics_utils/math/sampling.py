"""Define classes to simplify sampling from common spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealRange:
    """A closed interval [low, high] on the real line."""

    low: float  # Inclusive
    high: float  # Inclusive

    def __post_init__(self) -> None:
        """Verify that any constructed RealRange defines a valid range."""
        if self.high < self.low:
            raise ValueError(f"Invalid RealRange: high ({self.high}) < low ({self.low}).")

    @classmethod
    def from_tuple(cls, t: tuple[float, float]) -> RealRange:
        """Construct a RealRange from a (low, high) tuple."""
        return RealRange(t[0], t[1])

    @property
    def length(self) -> float:
        """Retrieve the length of the interval."""
        return self.high - self.low

    def contains(self, x: float) -> bool:
        """Check whether the given value is inside the interval [low, high]."""
        return self.low <= x <= self.high

    def clamp(self, x: float) -> float:
        """Clamp the given value into the interval [low, high]."""
        return np.clip(x, a_min=self.low, a_max=self.high)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample a value uniformly from the interval [low, high].

        :param rng: Optional NumPy random number generator; defaults to np.random.default_rng()
        :return: Float drawn uniformly from [low, high]
        """
        rng = np.random.default_rng() if rng is None else rng

        if self.length == 0.0:
            return self.low
        return rng.uniform(self.low, self.high)
