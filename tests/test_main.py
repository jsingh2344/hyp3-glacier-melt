"""Tests for the glacier-melt command-line interface."""

import sys

import pytest

from hyp3_glacier_melt.__main__ import main


def test_help_does_not_expose_rgi_paths(monkeypatch, capsys) -> None:
    """The container's bundled RGI paths are not user-configurable."""
    monkeypatch.setattr(sys, 'argv', ['hyp3_glacier_melt', '--help'])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert '--rgi-root' not in help_text
    assert '--rgi-shapefile' not in help_text


def test_single_burst_id_is_forwarded_with_existing_datacube(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    def fake_process_glacier_melt(**kwargs):
        calls.update(kwargs)
        return tmp_path / 'product.zip'

    monkeypatch.setattr(
        'hyp3_glacier_melt.__main__.process_glacier_melt',
        fake_process_glacier_melt,
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'hyp3_glacier_melt',
            '--datacube',
            'cube.nc',
            '--opera-burst-id',
            'T014_028627_IW2',
            '--output-root',
            str(tmp_path),
        ],
    )

    main()

    assert calls['datacube'] == 'cube.nc'
    assert calls['opera_burst_id'] == 'T014_028627_IW2'


def test_multiple_burst_ids_are_rejected_with_one_existing_datacube(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'hyp3_glacier_melt',
            '--datacube',
            'cube.nc',
            '--opera-burst-id',
            'T014_028627_IW2',
            '--opera-burst-id',
            'T014_028628_IW2',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
