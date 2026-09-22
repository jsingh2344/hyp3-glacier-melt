# Kennicott characterization baseline

This directory captures the output of the current onset-map pipeline before
correctness fixes. These files are characterization fixtures: they document
existing behavior but are not assertions that the scientific values are
correct.

## Source revision and environment

- Git revision: `198e7f4` (`develop`)
- Package version: `0.0.1.dev38+g198e7f426.d20260827`
- Python: `3.14.7`
- NumPy: `2.5.2`
- pandas: `3.0.5`
- xarray: `2026.7.0`
- Datacube: `Kennicott_32607_S1_cube_VH_014_387.nc`
- Datacube size: `5,940,344,443` bytes
- Datacube SHA-256:
  `595e069dc00e87c4274d73b43c04c9c429f8fe9e00f1b2ce79758dc1182a20f9`
- Configuration SHA-256:
  `98a1c0a24182587237da67fdcd9332f15e2614741ddc1c89474eb506d8c0d0e2`
- RGI shapefile SHA-256:
  `505c1cfbaf9858ed30ff704276d0055d7e43f9c42c17cf2ac83ac0df8166d8dc`
- RGI attributes SHA-256:
  `ba283a4d242a9050c44b5b27e15fd156581b593d500693efcd60f429aa8398d3`

The cube contract observed with `ncdump` is:

- Dimensions: `time=226`, `y=2225`, `x=2921`
- Variables: `images(time,y,x)`, `dem(y,x)`,
  `rgi_ind_glacier_mask(y,x)`, `rgi_aspect_arr(y,x)`
- Projection: `EPSG:32607`
- Pixel size: 100 m
- Time range begins: `2017-06-04`

The environment resolved Python 3.14 even though `pyproject.toml` currently
advertises support only through Python 3.13. Reproducibility work should pin a
supported Python version before this baseline is treated as portable.

## Baseline command

```bash
python -m hyp3_glacier_melt \
  --datacube "$PWD/datacubes/Kennicott_32607_S1_cube_VH_014_387.nc" \
  --output-root "$PWD/output" \
  --rgi-root "$PWD/Glaciers" \
  --rgi-shapefile "$PWD/Glaciers/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp"
```

The run produced 223 glacier CSV files, each containing 226 data rows.

## Selected fixtures

Glaciers 5589 and 5999 are retained because the project README uses them as
worked scientific examples.

| Glacier | Rows | SHA-256 |
| --- | ---: | --- |
| 5589 | 226 | `c876a65fc3ab17c07f22d357bce5dbbaecb184adf4726ef29bfcea1a9f59c4d1` |
| 5999 | 226 | `ff5f3f6bfd962e090bd0efb7c9cbcb91cd9f86efe569af2d19cf4dd0bb7eedc9` |

## Known defects represented by this baseline

- The summary reports every produced glacier as failed because
  `failed_glacnos.append(glacno)` is unconditional.
- Tile-boundary glaciers may be omitted.
- Melt area has a documented one-pixel undercount.
- No-detection values may be replaced with minimum glacier elevation.
- Several uncertainty indices are not bounds-checked.
- Thresholds and seasonal windows have not yet been reconciled with the README.

When a correction intentionally changes either CSV, update the fixture and
record the scientific or algorithmic reason in the corresponding test or
change note.
