"""Tests for glacier-melt product packaging."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from hyp3_glacier_melt.product import MeltRunResult, package_product


def write_csv(path: Path) -> Path:
    """Write a minimal CSV fixture and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('date,melt_area_m2\n2024-06-01,10000\n')
    return path


def test_package_product_contains_only_explicit_files(tmp_path: Path) -> None:
    csv_a = write_csv(tmp_path / 'run' / 'glacier_1.csv')
    csv_b = write_csv(tmp_path / 'run' / 'glacier_2.csv')
    write_csv(tmp_path / 'run' / 'stale.csv')
    result = MeltRunResult(scenes=3, csv_files=[csv_b, csv_a], failed_glacnos={})

    product_path = package_product(
        result,
        product_name='HYP3_GLACIER_MELT_TEST',
        output_root=tmp_path / 'products',
        metadata={'source_id': 'TEST'},
    )

    with ZipFile(product_path) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [
            'HYP3_GLACIER_MELT_TEST/glacier_1.csv',
            'HYP3_GLACIER_MELT_TEST/glacier_2.csv',
            'HYP3_GLACIER_MELT_TEST/product_metadata.json',
            'HYP3_GLACIER_MELT_TEST/README.txt',
        ]
        metadata = json.loads(
            archive.read('HYP3_GLACIER_MELT_TEST/product_metadata.json')
        )

    assert metadata['status'] == 'complete'
    assert metadata['scene_count'] == 3
    assert metadata['successful_glacier_count'] == 2
    assert metadata['failed_glacier_count'] == 0
    assert metadata['files'] == ['glacier_1.csv', 'glacier_2.csv']


def test_package_product_records_partial_status(tmp_path: Path) -> None:
    csv_file = write_csv(tmp_path / 'glacier_1.csv')
    result = MeltRunResult(
        scenes=3,
        csv_files=[csv_file],
        failed_glacnos={42: 'ValueError: test failure'},
    )

    product_path = package_product(result, 'partial_product', tmp_path, {})

    with ZipFile(product_path) as archive:
        metadata = json.loads(archive.read('partial_product/product_metadata.json'))

    assert metadata['status'] == 'partial'
    assert metadata['failed_glacier_count'] == 1
    assert metadata['failed_glacnos'] == {'42': 'ValueError: test failure'}


def test_package_product_rejects_empty_result(tmp_path: Path) -> None:
    result = MeltRunResult(scenes=3, csv_files=[], failed_glacnos={})

    with pytest.raises(RuntimeError, match='produced no CSV'):
        package_product(result, 'empty_product', tmp_path, {})


def test_package_product_rejects_missing_csv(tmp_path: Path) -> None:
    result = MeltRunResult(
        scenes=3,
        csv_files=[tmp_path / 'missing.csv'],
        failed_glacnos={},
    )

    with pytest.raises(FileNotFoundError, match='does not exist'):
        package_product(result, 'missing_product', tmp_path, {})


def test_package_product_rejects_duplicate_filenames(tmp_path: Path) -> None:
    csv_a = write_csv(tmp_path / 'a' / 'same.csv')
    csv_b = write_csv(tmp_path / 'b' / 'same.csv')
    result = MeltRunResult(scenes=3, csv_files=[csv_a, csv_b], failed_glacnos={})

    with pytest.raises(ValueError, match='Duplicate CSV filenames'):
        package_product(result, 'duplicate_product', tmp_path, {})


def test_package_product_refuses_to_overwrite(tmp_path: Path) -> None:
    csv_file = write_csv(tmp_path / 'glacier.csv')
    result = MeltRunResult(scenes=3, csv_files=[csv_file], failed_glacnos={})
    package_product(result, 'existing_product', tmp_path, {})

    with pytest.raises(FileExistsError, match='already exists'):
        package_product(result, 'existing_product', tmp_path, {})
