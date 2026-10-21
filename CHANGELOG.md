# Changelog

## Scene calibration extension

- Add explicit scene datasets with stored T0, per-frame Ti, and left residuals.
- Support dynamic nominal jitter, fixed-T0 sampling, and per-frame-Ti sampling.
- Add configurable MID360 and RS32 training examples.
- Consolidate scene and KITTI evaluation under `python test.py`.
- Preserve upstream localization modules, licensing, and attribution.
- Document the scene data format and portable training and evaluation commands.
