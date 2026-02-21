"""Define a dataclass to represent a closed interval of real numbers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClosedInterval:
    """An interval of real numbers between two endpoints (both included in the interval)."""

    minimum: float
    """Lower bound of the interval (included in the interval)."""

    maximum: float
    """Upper bound of the interval (included in the interval)."""

    @property
    def midpoint(self) -> float:
        """Compute and return the midpoint of the closed interval."""
        return (self.minimum + self.maximum) / 2.0

    def narrow_inward(self, x: float) -> ClosedInterval:
        """Create a closed interval with the same center but both bounds narrowed inward.

        :param x: Length (unitless) by which each bound of the interval is narrowed
        :return: Resulting closed interval: [a + x, b - x]
        """
        return ClosedInterval(self.minimum + x, self.maximum - x)

    def uniform_sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample from the interval uniformly.

        :param rng: Optional random number generator (default: None)
        :return: Sampled value within the interval
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(low=self.minimum, high=self.maximum)
