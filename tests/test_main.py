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
