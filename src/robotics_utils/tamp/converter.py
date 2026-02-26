"""Define an interface for classes that attempt to convert sampled values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
"""Represents an object or tuple of objects input into a converter."""

OutputT = TypeVar("OutputT")
"""Represents an object or tuple of objects output from a converter."""


class Converter(ABC, Generic[InputT, OutputT]):
    """A converter that attempts to convert input values into output values."""

    @classmethod
    @abstractmethod
    def convert(cls, inputs: InputT) -> OutputT | None:
        """Attempt to convert the given inputs into output values.

        :param inputs: Input values to be converted
        :return: Resulting output values, or None if conversion failed
        """
