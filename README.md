# CMRNext Scene Calibration Extension

This repository is an unofficial extension of [CMRNext](https://github.com/robot-learning-freiburg/CMRNext) for camera-LiDAR extrinsic calibration on explicitly selected scene directories. It adds MID360 and RS32 examples, fixed and per-frame extrinsic training data, online perturbation modes, deterministic evaluation, and auditable outputs.

The original CMRNext paper, models, localization code, and experimental claims belong to the upstream authors. Results produced by this fork should not be attributed to them.

## Changes from upstream CMRNext

| Area | Upstream CMRNext | This repository |
|---|---|---|
| Calibration training data | KITTI, Argoverse 1, and PandaSet loaders | Explicit scene directories with configurable camera and LiDAR geometry |
| Sensor examples | Public automotive datasets | Livox MID360 and RoboSense RS32 |
| Extrinsic targets | Dataset calibration with random perturbations | Stored `T0`/per-frame `Ti`, fixed `T0`, or perturbations sampled around each `Ti` |
| Geometry | Dataset-specific behavior | `delta = Ti @ inverse(T0)`, with `delta @ T0 = Ti` |
| Evaluation | Direct evaluator calls | `python test.py` for scenes, KITTI fixed cameras, and the original KITTI perturbation benchmark |
| Reproducibility | Dataset-dependent | Explicit scenes, seed, dry-run, logs, checkpoint-stage mapping, and `errors.json` |

Training now uses `python train_calibration.py` and the scene contract below. KITTI, Argoverse, and PandaSet code remains available for upstream inference and localization use, but the current training entry point does not accept those raw datasets.


## Installation

Docker is recommended because the project builds OpenCV with CUDA and the custom PnP extension.

~~~bash
docker build . -t cmrnext-scene
docker run --gpus=all --shm-size=2g -it \
  -v /path/to/data:/data \
  -v /path/to/checkpoints:/root/CMRNext/checkpoints \
  cmrnext-scene
~~~

The image uses CUDA 11.3, PyTorch 1.11, and OpenCV 4.7. RTX 50 series support is not included in the current Dockerfile. For manual setup, see [requirements.txt](requirements.txt), [Dockerfile](Dockerfile), and [visibility_pkg](visibility_pkg/).

## Scene data contract

Each scene is selected explicitly. Directory names, metadata split fields, manifests, and statistics do not change whether a scene is used for training, validation, or testing.

~~~text
scene_001/
├── image_2/                  # synchronized RGB images, e.g. 000000.png
├── velodyne/                 # matching Nx4 float32 .bin point clouds
├── calib.txt                 # camera projection matrix, P0 or P2
├── nominal_extrinsic.txt     # one 3x4 or 4x4 LiDAR-to-camera T0
├── poses.txt                 # one per-frame LiDAR-to-camera Ti
└── residuals.txt             # one per-frame Ti @ inverse(T0)
~~~

Directory names can be changed with `dataset.layout`, and file names with `dataset.files`. Matching image and point-cloud stems identify a frame. Translations are in meters, transforms use column vectors, and extrinsics map LiDAR points into the camera frame.

~~~text
network input: T0 @ p_lidar
target:        Ti @ p_lidar
supervision:   delta = Ti @ inverse(T0)
identity:      delta @ T0 = Ti
~~~

## Training data and perturbations

The same scene structure supports three modes:

1. **Stored multi-extrinsic with nominal jitter** (`dynamic_nominal_augmentation`): reads stored `T0`, per-frame `Ti`, and residuals. Each training access samples translation uniformly inside a ball and a random-axis rotation, applies `new_T0 = jitter @ stored_T0`, then recomputes `Ti @ inverse(new_T0)`.
2. **Fixed single extrinsic** (`fixed_t0_online_perturbation`): requires `Ti = T0` and identity stored residuals. It samples XYZ translation and XYZ Euler components independently from symmetric uniform ranges and constructs `T_input = inverse(delta) @ Ti`.
3. **Per-frame Ti online perturbation** (`per_ti_online_perturbation`): keeps each stored `Ti` as the target, ignores stored `T0` and residuals when generating the input, samples a new per-axis `delta`, and constructs `T_input = inverse(delta) @ Ti`.

Only one online perturbation section may be enabled. Stored multi-extrinsic evaluation never resamples its residual. Fixed-T0 evaluation derives deterministic perturbations from the seed and sample index.

## Example YAML files

| File | Purpose |
|---|---|
| `config/train_mid360_multi_extrinsic.yaml` | MID360 stored multi-extrinsic data with small online `T0` jitter |
| `config/train_mid360_multi_extrinsic_online.yaml` | MID360 per-frame `Ti` with online per-axis perturbations |
| `config/train_mid360_single_extrinsic.yaml` | MID360 fixed `T0` with online per-axis perturbations |
| `config/train_rs32_multi_extrinsic.yaml` | RS32 stored multi-extrinsic data with small online `T0` jitter |
| `config/train_rs32_single_extrinsic.yaml` | RS32 fixed `T0` with online per-axis perturbations |
| Training YAML files | Use the matching training YAML for evaluation; override scenes with `--scene-paths` |

Examples use `schema_version: cmrnext_scene_calibration_v1`. The loader also accepts the old `cmrnext_emd_training_v2` schema. Replace `dataset.root`, scene lists, camera intrinsics, and acquisition values before a real run.

## Training examples

Always run `--dry-run` first. It reports resolved scenes, geometry, seed, output directory, image sizes, and perturbation mode without loading samples or starting CUDA.

### MID360 per-frame Ti online perturbation

~~~bash
python train_calibration.py \
  --config config/train_mid360_multi_extrinsic_online.yaml \
  --level perturbed_t0 \
  --img_shape 576 768 \
  --savemodel checkpoints --run_name cmrnext_ti_mid360_stage2 \
  --epochs 300 --batch_size 4 --num_worker 4 \
  --seed 42 --gpu -1 --gpu_count 4 --val_every 0 \
  --dry-run
~~~

### RS32 fixed single extrinsic

~~~bash
python train_calibration.py \
  --config config/train_rs32_single_extrinsic.yaml \
  --level fixed_baseline \
  --img_shape 576 768 \
  --savemodel checkpoints --run_name rs32_fixed_t0 \
  --epochs 300 --batch_size 4 --num_worker 4 \
  --seed 42 --gpu 0 --gpu_count 1 \
  --dry-run
~~~

Remove `--dry-run` after reviewing the report. `--train-scene-paths`, `--val-scene-paths`, and `--test-scene-paths` replace the corresponding YAML lists. If validation is empty, set `--val_every 0`.

Outputs are written to `<savemodel>/<run_name>/` and include the training log, TensorBoard events, rolling checkpoints, and the last checkpoint. TensorBoard exposes each run through the project-root `tf-logs/<run_name>` link; override the service root with `--tb_mount_root` when needed. Use `--weights ... --resume` to restore optimizer and epoch state; add `--rebuild_scheduler true` when changing the total epoch budget.

## Evaluation examples

List checkpoints in cascade order. `--start-stage` is the stage number of the first checkpoint.

`scene` is the required subcommand for explicit scene evaluation; it selects the evaluator mode after `test.py`. Other modes use their own subcommand, such as `kitti` or `kitti-perturbed`.

### Explicit scene evaluation

~~~bash
python test.py scene \
  --config config/train_mid360_multi_extrinsic.yaml \
  --level perturbed_t0 \
  --weights checkpoints/stage1.tar checkpoints/stage2.tar \
  --start-stage 1 --num-workers 4 --seed 42 --gpu 1 \
  --errors-json evaluation_reports/mid360/ti_2stage_errors.json \
  --log-file evaluation_reports/mid360/ti_2stage_evaluation.log \
  --dry-run
~~~

Omit `--scene-paths` only when the YAML provides a non-empty `dataset.scenes.test` list.

### KITTI fixed left/right camera extrinsics

Camera `2` is the rectified left camera and camera `3` is the rectified right camera. The target camera supplies the image, intrinsics, and `Ti`; the other camera supplies the fixed physical `T0`. They must differ.

~~~bash
python test.py kitti \
  --config config/train_mid360_multi_extrinsic.yaml \
  --data-root /data/KITTI/sequences --sequences 00 \
  --target-cam 2 --nominal-cam 3 \
  --weights checkpoints/model/last_iter_checkpoint.tar \
  --start-stage 1 --num-workers 4 --seed 42 --gpu 0 \
  --errors-json evaluation_reports/kitti_fixed/errors.json \
  --log-file evaluation_reports/kitti_fixed/evaluation.log \
  --dry-run
~~~

Use `test.py kitti-perturbed` only to reproduce the original CMRNext perturbation benchmark. Paper-scale defaults are `--max-t 1.5`, `--max-r 20`, and `--seed 42`.

## Adapting to another LiDAR

Change these YAML fields to describe the point clouds that were actually generated:

| Field | Meaning |
|---|---|
| `rings` | Physical channels or virtual groups used during generation |
| `points_per_ring` | Points or emitted beams represented per ring/group |
| `vertical_fov_deg` | Minimum and maximum vertical beam angles |
| `repeated_pattern` | `true` for repeating rotary scans; `false` for non-repeating patterns such as MID360 |
| `scan_frequency_hz` | Physical scan frequency |
| `beams_per_scan` | Emitted beams in one scan period |
| `beam_rate_hz` | Must equal `beams_per_scan * scan_frequency_hz` |
| `accumulation_duration_s` | Time window already accumulated into each offline point cloud |

Also replace camera `width`, `height`, `fx`, `fy`, `cx`, `cy`, and `skew`. `preprocessing.resize_hw` controls online resize and intrinsics are scaled with the image. The loader audits acquisition values but does not accumulate point clouds online. Changing `accumulation_duration_s` therefore requires matching offline point clouds.

A rotary multi-line LiDAR normally uses `repeated_pattern: true` and one complete scan period. A non-repeating LiDAR uses `false`, its actual beam rate, and the offline accumulation window.

## Upstream, citation, and license

- Original CMRNext paper code: https://github.com/robot-learning-freiburg/CMRNext
- Project page: https://cmrnext.cs.uni-freiburg.de/
- Paper: *CMRNext: Camera to LiDAR Matching in the Wild for Localization and Extrinsic Calibration*
- arXiv: https://arxiv.org/abs/2402.00129
- IEEE T-RO: https://ieeexplore.ieee.org/document/10908048

This repository keeps the upstream [GNU GPLv3 license](LICENSE). Preserve upstream copyright and license notices when redistributing this fork. Replace `/path/to/dataset_root` and the example scenes in the YAML files with paths to your own data.
