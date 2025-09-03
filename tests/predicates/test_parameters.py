"""Unit tests for the Parameter class."""

from dataclasses import dataclass, field

from robotics_utils.predicates import Parameter


@dataclass(frozen=True)
class WeatherReport:
    """A weather report for a city."""

    city_name: str = field(metadata={"doc": "Name of the city."})

    temperature: int

    rain_cm: float = field(metadata={"doc": "Centimeters of rain expected for the day."})


def test_parameter_tuple_from_dataclass() -> None:
    """Verify that a tuple of parameters can be constructed from a Python dataclass."""
    parameters = Parameter.tuple_from_dataclass(WeatherReport)

    assert len(parameters) == 3
    city_name, temperature, rain_cm = parameters

    assert city_name.name == "city_name"
    assert city_name.type_ is str
    assert city_name.semantics == "Name of the city."

    assert temperature.name == "temperature"
    assert temperature.type_ is int
    assert temperature.semantics is None

    assert rain_cm.name == "rain_cm"
    assert rain_cm.type_ is float
    assert rain_cm.semantics == "Centimeters of rain expected for the day."
