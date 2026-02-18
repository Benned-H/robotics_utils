"""Define a class to generate enumerable sequences of samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

import numpy as np

InputT = TypeVar("InputT")
"""Represents an object tuple input to a generator."""

OutputT = TypeVar("OutputT")
"""Represents an object tuple output from a generator."""


class Generator(ABC, Generic[InputT, OutputT]):
    """A generator providing a lazy iterator over a sequence of sampled values.

    This class generalizes the "pose generators" described by Srivastava et al. (ICRA 2014).

    In the terminology of Garrett et al. (ICAPS 2020), this class implements both:
        1) A generator taking no inputs (when InputT = NoneType)
        2) A conditional generator (when InputT is a dataclass)

    References:
        Section IV.B.2 (pg. 643) of Srivastava et al. (ICRA 2014).
        Section 3.1 (pg. 2) of Garrett et al. (ICAPS 2020).

    """

    @abstractmethod
    def _generate(self, inputs: InputT) -> Iterator[OutputT]:
        """Generate a sequence of samples conditioned on the given inputs.

        :param inputs: Values on which the generator is conditioned
        :yield: Sequence of generated output values
        """

    def __init__(self, inputs: InputT, rng_seed: int | None = None) -> None:
        """Initialize the generator's internal state using the given inputs.

        :param inputs: Values on which the generator is conditioned
        :param rng_seed: Optional random number generator seed (default: None)
        """
        self.inputs = inputs
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(self._rng_seed)

        self._generator_state: Iterator[OutputT] = self._generate(self.inputs)
        self._call_count = 0
        """Number of times next() has been called on the generator."""

    def __iter__(self) -> Iterator[OutputT]:
        """Return the generator itself, providing an Iterator over its output values."""
        return self

    def __next__(self) -> OutputT:
        """Return the next generated value from the stored lazy iterator.

        Unlike the description of Garrett et al. (ICAPS 2020), this method raises
            a StopIteration, rather than returning None, when the sequence is exhausted.

        Reference: https://docs.python.org/3/library/stdtypes.html#iterator.__next__

        :return: Next generated value from the iterator
        :raises StopIteration: When the generator's sequence of samples has been exhausted
        """
        output = next(self._generator_state)  # Allow StopIteration to propagate
        self._call_count += 1  # Increment only when StopIteration wasn't raised
        return output

    @property
    def count(self) -> int:
        """Retrieve the current number of times the generator has been called."""
        return self._call_count

    def reset(self) -> None:
        """Reset the generator to its initial state."""
        self._rng = np.random.default_rng(self._rng_seed)
        self._generator_state = self._generate(self.inputs)
        self._call_count = 0

    def reseed(self, rng_seed: int) -> None:
        """Reinitialize the generator using the given RNG seed.

        :param rng_seed: Random number generator seed
        """
        self._rng_seed = rng_seed
        self.reset()
