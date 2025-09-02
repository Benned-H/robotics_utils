"""Unit tests for the Parameter class."""

from dataclasses import dataclass

from robotics_utils.predicates.parameters import Parameter


@dataclass(frozen=True)
class WeatherReport:
    """A weather report for an afternoon."""

    temperature: float
    cloudy: bool
    precipitation: str
    """Type of precipitation, if any, expected during the day."""


def test_parameter_tuple_from_dataclass() -> None:
    """Verify that a tuple of parameters can be constructed from a Python dataclass."""
    parameters = Parameter.tuple_from_dataclass(WeatherReport)

    assert len(parameters) == 3
