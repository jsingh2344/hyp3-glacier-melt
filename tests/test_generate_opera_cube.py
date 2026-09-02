"""Tests for OPERA datacube input validation."""

from pathlib import Path

import pytest

from hyp3_glacier_melt.hyp3_datacube.generate_opera_cube import (
    extract_acquisition_timestamp_from_filename,
    normalize_opera_burst_id,
    opera_timestamp_to_datetime64,
    polygon_from_bounds,
    select_latest_opera_files,
    source_copernicus_dem,
    verify_opera_burst_paths,
)


def opera_path(
    burst_id: str,
    acquisition: str,
    production: str = "20240408T211029Z",
) -> Path:
    """Return a representative OPERA RTC-S1 polarization filename."""
    return Path(
        f"OPERA_L2_RTC-S1_{burst_id}_{acquisition}_{production}_"
        "S1A_30_v1.0_VH.tif"
    )


def test_normalize_opera_burst_id_accepts_underscore_form() -> None:
    assert normalize_opera_burst_id("t014_028627_iw2") == "T014-028627-IW2"


def test_verify_opera_burst_paths_accepts_requested_burst() -> None:
    paths = [
        opera_path("T014-028627-IW2", "20240311T155650Z"),
        opera_path("T014-028627-IW2", "20240323T155651Z"),
    ]

    burst_id = verify_opera_burst_paths(paths, "T014_028627_IW2")

    assert burst_id == "T014-028627-IW2"


def test_verify_opera_burst_paths_rejects_mixed_bursts() -> None:
    paths = [
        opera_path("T014-028627-IW2", "20240311T155650Z"),
        opera_path("T015-030000-IW1", "20240312T155650Z"),
    ]

    with pytest.raises(ValueError, match="multiple bursts"):
        verify_opera_burst_paths(paths, "T014-028627-IW2")


def test_verify_opera_burst_paths_rejects_wrong_requested_burst() -> None:
    paths = [opera_path("T015-030000-IW1", "20240312T155650Z")]

    with pytest.raises(ValueError, match="Requested OPERA burst T014-028627-IW2"):
        verify_opera_burst_paths(paths, "T014-028627-IW2")


def test_select_latest_opera_file_for_each_acquisition() -> None:
    older = opera_path(
        "T014-028627-IW2",
        "20240311T155650Z",
        "20240312T224358Z",
    )
    newer = opera_path(
        "T014-028627-IW2",
        "20240311T155650Z",
        "20240408T211029Z",
    )
    other_acquisition = opera_path("T014-028627-IW2", "20240323T155651Z")

    selected = select_latest_opera_files([newer, other_acquisition, older])

    assert selected == [newer, other_acquisition]


def test_opera_acquisition_timestamp_is_preserved() -> None:
    path = opera_path("T014-028627-IW2", "20240311T155650Z")

    timestamp = extract_acquisition_timestamp_from_filename(path)

    assert timestamp == "20240311T155650Z"
    assert str(opera_timestamp_to_datetime64(timestamp)) == "2024-03-11T15:56:50"


def test_polygon_from_bounds_uses_the_requested_extent() -> None:
    polygon = polygon_from_bounds((-144.0, 61.0, -142.0, 63.0))

    assert polygon.GetEnvelope() == (-144.0, -142.0, 61.0, 63.0)


def test_source_copernicus_dem_uses_hyp3lib(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    def fake_prepare_dem_geotiff(**kwargs):
        calls.update(kwargs)
        return kwargs["output_name"]

    monkeypatch.setattr(
        "hyp3lib.dem.prepare_dem_geotiff",
        fake_prepare_dem_geotiff,
    )
    output_path = tmp_path / "dem.tif"

    result = source_copernicus_dem(
        output_path=output_path,
        output_bounds=(-144.0, 61.0, -142.0, 63.0),
        output_epsg_str="EPSG:4326",
    )

    assert result == output_path
    assert calls["epsg_code"] == 4326
    assert calls["pixel_size"] == 30.0
    assert calls["buffer_size_in_degrees"] == 0.01
    assert calls["height_above_ellipsoid"] is False
    assert calls["geometry"].GetEnvelope() == (-144.0, -142.0, 61.0, 63.0)
