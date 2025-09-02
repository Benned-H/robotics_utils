"""Unit tests for the DataclassType class."""

from dataclasses import dataclass

from robotics_utils.predicates.dataclass_type import DataclassType


@dataclass(frozen=True)
class WeatherReport:
    """An example weather report for a city."""

    city_name: str
    """Name of the city."""

    temperature: int

    rain_cm: float
    """Centimeters of rain expected for the day."""


def test_dataclass_type_get_docstrings() -> None:
    """Verify that dataclass docstrings can be retrieved by the DataclassType class."""
    # Arrange - See the WeatherReport dataclass definition above

    # Act - Construct a DataclassType instance and extract its docstrings
    weather_report_type = DataclassType(WeatherReport)
    docstrings_map = weather_report_type.get_docstrings()

    # Assert - Expect that the docstring-annotated member variables have docstrings
    assert "city_name" in docstrings_map
    assert docstrings_map["city_name"] == "Name of the city."

    assert "temperature" in docstrings_map
    assert docstrings_map["temperature"] is None

    assert "rain_cm" in docstrings_map
    assert docstrings_map["rain_cm"] == "Centimeters of rain expected for the day."
