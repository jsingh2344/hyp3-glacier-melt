WORKFLOW PARITY — 22 October 2026

The stitching workflow now follows the primary (DOY-producing, non-stitching)
container's metric preparation and calculation path:

- Automatic Copernicus preparation uses the existing primary helper (cubic
  interpolation to 30 m). No new DEM preparation algorithm was introduced.
- The primary and DOY workflows now share read_metric_dem, extracting the
  original GDAL Translate/Warp, averaging and int16 rounding procedure without
  changing the primary calculation. A supplied DEM retains its existing upstream
  history; the native source metric DEM is preferred to preserve that history.
- Primary exports carry the exact metric DEM, all-acquisition SAR validity and
  source acquisition dates. Stitching unions valid coverage and source calendars;
  the same validity is applied to onset counts and elevation distributions.
- Native acquisition dates are the default evaluation calendar. A regular DOY
  interval must be explicitly requested with --doy-step.
- Default processing rejects source products missing the required native inputs
  rather than silently substituting approximate validity or a daily calendar.
  Legacy approximate processing remains explicit through --doy-step.
- Glacier area eligibility and usable onset-year selection match the primary
  routine. The existing metric calculation and its thresholds remain unchanged.

Verification data
The original 028627 product's event maps, original SAR validity and reconstructed
historical cubic/average DEM were exported using the updated production writer.
The single product was also partitioned into overlapping left/right tiles with
identical observations and preserved metadata. No acquisition-calendar patch was
used. The local process used the repository RGI; container runs used its bundled
RGI. Every comparison joins glacier/year/date with the original archived CSVs.

A separate local image, hyp3-glacier-melt-doy:workflow-parity, uses the existing
historical image ghcr.io/jsingh2344/hyp3-glacier-melt:test as its base/runtime and
loads the updated production sources. This avoids changing numerical library
versions during comparison. It does not replace or publish existing image tags.
The normal downstream Dockerfile remains the standard full-build definition.

Four container scenarios run with network disabled and read-only input mounts:
1. Single source, embedded native DEM.
2. Two overlapping tiles, embedded native DEM.
3. Same tiles in reverse input order.
4. Single source, explicit 30 m DEM, exercising the shared primary GDAL routine.

All four scenarios match the original archived CSVs for 31 glaciers, 226 dates,
7,006 rows and eight metrics: 56,048 exact values per scenario. The packaged
scene count is 226, the calendar is source_acquisitions, and no glaciers fail.
Local single-source and stitched package runs match the same reference as well.
See verification.json and verify_outputs.py for checks.

39 tests pass. The scope is workflow equivalence when underlying observations
are identical. Independent bursts can legitimately differ after event fusion or
expanded valid coverage; this does not validate the scientific accuracy of the
existing correction thresholds or minimum-DOY fusion policy.

Source changes
src/hyp3_glacier_melt/hyp3_datacube/generate_opera_cube.py: shared original DEM reader.
src/hyp3_glacier_melt/doy_product.py: preserve acquisition calendar with native inputs.
src/hyp3_glacier_melt/doy_workflow/pipeline.py: use shared reader/native calendar and enforce provenance.
src/hyp3_glacier_melt/doy_workflow/process.py and __main__.py: calendar defaults and explicit override.
README.md: behavior and migration requirements.
