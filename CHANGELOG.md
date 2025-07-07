# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CDFSplineCalibrator` for multi-class probability calibration (based on Gupta et al. 2021 ICLR paper)
- Comprehensive Sphinx documentation
- Read the Docs configuration
- Examples for calibration usage
- Type hints throughout the codebase

### Changed
- Restructured package: moved estimators to `splinator.estimators` submodule
- Updated examples to use new API

## [0.2.0] - 2024-01-XX

### Changed
- Migrated from PDM to Hatchling as build backend
- Updated dependency version constraints

### Added
- Additional example notebooks
- Metrics module with calibration metrics

## [0.1.0] - Initial Release

### Added
- `LinearSplineLogisticRegression` estimator
- Basic scikit-learn compatibility
- Initial test suite
- Example notebooks 