# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Continue processing observed years when a burst-year has no April–July acquisitions; record the skipped year and export nodata event maps.
- Package annual DOY maps even when no glacier CSVs are produced, preserving partial glaciers for downstream stitching.
- Exclude missing-season sources from downstream annual coverage without discarding a glacier's other usable years.

## [0.1.0]

### Added
- hyp3-glacier-melt plugin created with the [HyP3 Cookiecutter](https://github.com/ASFHyP3/hyp3-cookiecutter)
- Updated glacier melt processing workflow
