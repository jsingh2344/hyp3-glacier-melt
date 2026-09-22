from pathlib import Path

from hyp3_glacier_melt.datacube import _glacier_time_series_path


def test_glacier_time_series_path_contains_full_rgi_id(tmp_path: Path) -> None:
    path = _glacier_time_series_path(tmp_path, '01.05589', 'all_years')

    assert path == tmp_path / 'melt_snowline_time_series_01.05589_all_years.csv'
