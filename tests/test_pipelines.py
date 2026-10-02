"""
Integration tests for Air Combat, Airport Runway, and ANPR vision pipelines.
"""

import pytest

from air_combat_intelligence.src.main import AirCombatSystem
from airport_runway_intelligence.src.main import AirportRunwaySystem
from common.utils import generate_synthetic_test_video
from license_plate_intelligence.src.main import LicensePlateSystem


@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "test_synth.mp4"
    generate_synthetic_test_video(fn, num_frames=15, width=640, height=360, scenario="combat")
    return str(fn)


def test_air_combat_pipeline(test_video, tmp_path):
    system = AirCombatSystem(source=test_video, output_dir=tmp_path / "combat")
    out_video = system.run(max_frames=10)
    assert out_video.exists()
    assert out_video.stat().st_size > 0


def test_airport_runway_pipeline(test_video, tmp_path):
    system = AirportRunwaySystem(source=test_video, output_dir=tmp_path / "runway")
    out_video = system.run(max_frames=10)
    assert out_video.exists()
    assert out_video.stat().st_size > 0


def test_license_plate_pipeline(test_video, tmp_path):
    system = LicensePlateSystem(source=test_video, output_dir=tmp_path / "anpr")
    out_video = system.run(max_frames=10)
    assert out_video.exists()
    assert out_video.stat().st_size > 0
