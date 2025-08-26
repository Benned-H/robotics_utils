"""Unit tests for classes representing visual fiducial markers."""

from pathlib import Path

import pytest

from robotics_utils.perception.sensors.visual_fiducials import FiducialSystem


@pytest.fixture
def markers_yaml() -> Path:
    """Specify a path to an example fiducial markers YAML file."""
    yaml_path = Path(__file__).parent.parent / "test_data/markers_example.yaml"
    assert yaml_path.exists(), f"Expected to find file: {yaml_path}"
    return yaml_path


def test_visual_fiducial_system_from_yaml(markers_yaml: Path) -> None:
    """Verify that a FiducialSystem can be loaded from an example YAML file."""
    system = FiducialSystem.from_yaml(markers_yaml)

    # Assert - Expect that the loaded data matches what's specified in the YAML file
    assert len(system.markers) == 2

    marker_1 = system.markers[1]
    assert marker_1.id == 1
    assert marker_1.size_cm == pytest.approx(4)
    assert len(marker_1.relative_frames) == 2
    assert "eraser" in marker_1.relative_frames
    assert "table" in marker_1.relative_frames

    marker_2 = system.markers[2]
    assert marker_2.id == 2
    assert marker_2.size_cm == pytest.approx(18.5)
    assert len(marker_2.relative_frames) == 1
    assert "sink" in marker_2.relative_frames

    assert len(system.cameras) == 3
    frontleft = system.cameras["frontleft"]
    assert len(frontleft.recognized_sizes_cm) == 1
    assert any(s == pytest.approx(5) for s in frontleft.recognized_sizes_cm)

    frontright = system.cameras["frontright"]
    assert len(frontright.recognized_sizes_cm) == 1
    assert any(s == pytest.approx(10) for s in frontright.recognized_sizes_cm)

    hand = system.cameras["hand"]
    assert len(hand.recognized_sizes_cm) == 2
    assert any(s == pytest.approx(4) for s in hand.recognized_sizes_cm)
    assert any(s == pytest.approx(18.5) for s in hand.recognized_sizes_cm)
