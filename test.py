"""Command construction and orchestration for calibration evaluation.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from datasets.emd_extrinsic_calib import load_emd_training_config


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATOR = PROJECT_ROOT / "evaluate_flow_calibration.py"
# Default scene configuration for Python API callers.
DEFAULT_SCENE_CONFIG = str(
    PROJECT_ROOT / "config" / "train_mid360_multi_extrinsic.yaml")
# Portable KITTI mount used in Docker.
DEFAULT_KITTI_DATA_ROOT = "/data/KITTI/sequences"


def _require_weights(weights: Sequence[str]) -> list[str]:
    """Resolve checkpoint paths and reject an empty list or missing files.

    Args:
        weights: Checkpoint paths in cascade order.

    Returns:
        list[str].

    Notes:
        Raise ValueError for an empty list and FileNotFoundError when a checkpoint does not exist. This function does not write files.
    """
    resolved = [str(Path(weight).expanduser().resolve()) for weight in weights]
    if not resolved:
        raise ValueError("At least one checkpoint is required")
    missing = [weight for weight in resolved if not Path(weight).is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少 checkpoint：" + ", ".join(missing))
    return resolved


def build_emd_test_args(
        weights: Sequence[str],
        *,
        level: str,
        scene_paths: Sequence[str],
        errors_json: str,
        start_stage: int = 1,
        emd_config: str = DEFAULT_SCENE_CONFIG,
        per_scene_worst_k: Optional[int] = None,
        triptych_dir: Optional[str] = None,
        num_workers: int = 4,
        seed: int = 42,
        max_frames: Optional[int] = None,
) -> list[str]:
    """Build evaluator arguments for an explicitly selected scene cascade.

    Args:
        weights: Checkpoint paths in cascade order.
        level: Configured curriculum or perturbation level.
        scene_paths: Explicit scene directories in loading order.
        errors_json: Destination for per-frame errors and evaluation metadata.
        start_stage: One-based stage number assigned to the first checkpoint.
        emd_config: Scene, sensor, preprocessing, and perturbation configuration.
        per_scene_worst_k: Number of worst frames selected independently per scene.
        triptych_dir: Output directory for reprojection panels.
        num_workers: Number of DataLoader worker processes.
        seed: Explicit seed for reproducible random sampling.
        max_frames: Optional diagnostic frame limit; None evaluates all selected frames.

    Returns:
        list[str].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    if level not in {"large", "medium", "small", "fixed_baseline", "perturbed_t0"}:
        raise ValueError(f"不支持的 EMD level：{level}")
    if start_stage not in {1, 2, 3}:
        raise ValueError(f"不支持的级联起始阶段：{start_stage}")
    if per_scene_worst_k is not None and per_scene_worst_k <= 0:
        raise ValueError("每场景最差样本数必须为正数")
    if (per_scene_worst_k is None) != (triptych_dir is None):
        raise ValueError(
            "每场景最差样本数和三联图目录必须同时提供")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("诊断帧数上限必须为正数")
    if not scene_paths:
        raise ValueError("EMD 场景路径列表不能为空")

    arguments = [
        "--dataset", "emd",
        "--weights", *_require_weights(weights),
        "--emd_level", level,
        "--emd_start_iteration", str(start_stage),
        "--emd_config", str(emd_config),
        "--num_worker", str(num_workers),
        "--seed", str(seed),
        "--errors_json", str(errors_json),
    ]
    arguments.extend(["--emd_scene_paths", *[str(path) for path in scene_paths]])
    if max_frames is not None:
        arguments.extend(["--max_frames", str(max_frames)])
    if per_scene_worst_k is not None:
        arguments.extend([
            "--viz", "true",
            "--viz_strategy", "per_scene_worst_joint",
            "--viz_per_scene_worst_k", str(per_scene_worst_k),
            "--viz_dir", str(triptych_dir),
        ])
    return arguments


def build_kitti_fixed_nominal_test_args(
        weights: Sequence[str],
        *,
        sequences: Sequence[str],
        target_cam: str,
        nominal_cam: str,
        errors_json: str,
        data_root: str = DEFAULT_KITTI_DATA_ROOT,
        emd_config: str = DEFAULT_SCENE_CONFIG,
        start_stage: int = 1,
        num_workers: int = 4,
        seed: int = 42,
) -> list[str]:
    """Build KITTI arguments using target-camera Ti and the other camera calibration as fixed T0.

    Args:
        weights: Checkpoint paths in cascade order.
        sequences: Explicit KITTI sequence identifiers.
        target_cam: Rectified KITTI target camera: 2 for left, 3 for right.
        nominal_cam: Distinct KITTI camera supplying fixed nominal T0.
        errors_json: Destination for per-frame errors and evaluation metadata.
        data_root: Dataset root directory.
        emd_config: Scene, sensor, preprocessing, and perturbation configuration.
        start_stage: One-based stage number assigned to the first checkpoint.
        num_workers: Number of DataLoader worker processes.
        seed: Explicit seed for reproducible random sampling.

    Returns:
        list[str].

    Notes:
        Cameras 2 and 3 are the rectified left and right cameras. Reject identical target and nominal cameras. This mode does not sample max_t/max_r perturbations.

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    target_cam = str(target_cam)
    nominal_cam = str(nominal_cam)
    if target_cam not in {"2", "3"} or nominal_cam not in {"2", "3"}:
        raise ValueError("KITTI 相机必须为左相机 2 或右相机 3")
    if target_cam == nominal_cam:
        raise ValueError("KITTI 目标相机与标称相机不能相同")
    if not sequences:
        raise ValueError("At least one KITTI sequence is required")
    if start_stage not in {1, 2, 3}:
        raise ValueError(f"不支持的级联起始阶段：{start_stage}")

    return [
        "--dataset", "kitti",
        "--data_folder", str(data_root),
        "--kitti_test_sequences", *[str(sequence) for sequence in sequences],
        "--cam", target_cam,
        "--kitti_nominal_cam", nominal_cam,
        "--weights", *_require_weights(weights),
        "--emd_config", str(emd_config),
        "--emd_start_iteration", str(start_stage),
        "--num_worker", str(num_workers),
        "--seed", str(seed),
        "--errors_json", str(errors_json),
    ]


def build_kitti_sampled_perturbation_test_args(
        weights: Sequence[str],
        *,
        sequences: Sequence[str],
        target_cam: str,
        errors_json: str,
        data_root: str = DEFAULT_KITTI_DATA_ROOT,
        emd_config: str = DEFAULT_SCENE_CONFIG,
        start_stage: int = 1,
        max_t: float = 1.5,
        max_r: float = 20.0,
        num_workers: int = 4,
        seed: int = 42,
) -> list[str]:
    """Build arguments for the original CMRNext KITTI perturbation benchmark.

    Args:
        weights: Checkpoint paths in cascade order.
        sequences: Explicit KITTI sequence identifiers.
        target_cam: Rectified KITTI target camera: 2 for left, 3 for right.
        errors_json: Destination for per-frame errors and evaluation metadata.
        data_root: Dataset root directory.
        emd_config: Scene, sensor, preprocessing, and perturbation configuration.
        start_stage: One-based stage number assigned to the first checkpoint.
        max_t: Translation sampling limit in meters.
        max_r: Rotation sampling limit in degrees.
        num_workers: Number of DataLoader worker processes.
        seed: Explicit seed for reproducible random sampling.

    Returns:
        list[str].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    target_cam = str(target_cam)
    if target_cam not in {"2", "3"}:
        raise ValueError("KITTI 相机必须为左相机 2 或右相机 3")
    if not sequences:
        raise ValueError("At least one KITTI sequence is required")
    if start_stage not in {1, 2, 3}:
        raise ValueError(f"不支持的级联起始阶段：{start_stage}")
    if max_t <= 0 or max_r <= 0:
        raise ValueError("KITTI 采样参数 max_t 和 max_r 必须为正数")

    return [
        "--dataset", "kitti",
        "--data_folder", str(data_root),
        "--kitti_test_sequences", *[str(sequence) for sequence in sequences],
        "--cam", target_cam,
        "--kitti_sampled_perturbation",
        "--max_t", str(float(max_t)),
        "--max_r", str(float(max_r)),
        "--weights", *_require_weights(weights),
        "--emd_config", str(emd_config),
        "--emd_start_iteration", str(start_stage),
        "--num_worker", str(num_workers),
        "--seed", str(seed),
        "--errors_json", str(errors_json),
    ]


def build_gt_flow_pnp_diagnostic_args(
        *,
        source_errors_json: str,
        selection_manifest: str,
        output_dir: str,
        repetitions: int = 5,
) -> list[str]:
    """Build diagnostic arguments comparing predicted-flow and ground-truth-flow PnP.

    Args:
        source_errors_json: Source errors json: str.
        selection_manifest: Selection manifest: str.
        output_dir: Output dir: str.
        repetitions: Repetitions: int.

    Returns:
        list[str].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    source_path = Path(source_errors_json).expanduser().resolve()
    manifest_path = Path(selection_manifest).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"缺少源误差 JSON：{source_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"缺少诊断选择清单：{manifest_path}")
    if repetitions <= 0:
        raise ValueError("真实光流 PnP 重复次数必须为正数")
    return [
        "--gt_flow_pnp_source_errors_json", str(source_path),
        "--gt_flow_pnp_selection_manifest", str(manifest_path),
        "--gt_flow_pnp_output_dir", str(output_dir),
        "--gt_flow_pnp_repetitions", str(repetitions),
    ]
def build_recorded_reprojection_args(
        *,
        source_errors_json: str,
        output_dir: str,
        per_scene_k: int = 50,
) -> list[str]:
    """Build arguments for rendering worst frames from recorded cascade poses.

    Args:
        source_errors_json: Source errors json: str.
        output_dir: Output dir: str.
        per_scene_k: Per scene k: int.

    Returns:
        list[str].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    source_path = Path(source_errors_json).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"缺少源误差 JSON：{source_path}")
    if per_scene_k <= 0:
        raise ValueError("记录重投影的每场景样本数必须为正数")
    return [
        "--recorded_reprojection_errors_json", str(source_path),
        "--recorded_reprojection_output_dir", str(output_dir),
        "--recorded_reprojection_per_scene_k", str(per_scene_k),
    ]


def build_geometry_domain_gap_args(
        *, source_errors_json: str, selection_manifest: str,
        dataset_root: str, output_dir: str,
        train_reference_per_scene: int = 100,
        test_reference_per_scene: int = 200, seed: int = 42,
) -> list[str]:
    """Build arguments for geometry and domain-gap analysis of recorded evaluation frames.

    Args:
        source_errors_json: Source errors json: str.
        selection_manifest: Selection manifest: str.
        dataset_root: Root used to resolve relative scene paths.
        output_dir: Output dir: str.
        train_reference_per_scene: Train reference per scene: int.
        test_reference_per_scene: Test reference per scene: int.
        seed: Explicit seed for reproducible random sampling.

    Returns:
        list[str].
    """
    source_path = Path(source_errors_json).expanduser().resolve()
    manifest_path = Path(selection_manifest).expanduser().resolve()
    dataset_path = Path(dataset_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"缺少源误差 JSON：{source_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少选择清单：{manifest_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"缺少 EMD 数据根目录：{dataset_path}")
    if train_reference_per_scene <= 0 or test_reference_per_scene <= 0:
        raise ValueError("几何参考样本数必须为正数")
    return [
        "--geometry_source_errors_json", str(source_path),
        "--geometry_selection_manifest", str(manifest_path),
        "--geometry_dataset_root", str(dataset_path),
        "--geometry_output_dir", str(output_dir),
        "--geometry_train_reference_per_scene",
        str(train_reference_per_scene),
        "--geometry_test_reference_per_scene",
        str(test_reference_per_scene),
        "--geometry_seed", str(seed),
    ]



def _run_evaluator(
        arguments: Sequence[str],
        *,
        log_file: Optional[str],
        gpu: Optional[str],
        dry_run: bool,
        overwrite: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Run the evaluator from the project root, or print its command during a dry run.

    Args:
        arguments: Arguments: Sequence[str].
        log_file: Path for the complete run log.
        gpu: CUDA device selection for the subprocess.
        dry_run: Print the resolved configuration or command without running the task.
        overwrite: Explicit permission to replace existing output artifacts.

    Returns:
        Optional[subprocess.CompletedProcess].

    Notes:
        Reject existing log/result destinations unless overwrite is explicit. Executed subprocesses may use CUDA and write evaluation artifacts; failures propagate to the caller.
    """
    command = [sys.executable, str(EVALUATOR), *arguments]
    command_text = shlex.join(command)
    if dry_run:
        print(command_text)
        return None

    if "--errors_json" in arguments:
        errors_index = arguments.index("--errors_json") + 1
        errors_path = Path(arguments[errors_index]).expanduser().resolve()
        if errors_path.exists() and not overwrite:
            raise FileExistsError(
                f"评估结果已存在，使用 --overwrite 才能替换: {errors_path}")

    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if log_file is None:
        return subprocess.run(
            command, check=True, cwd=str(PROJECT_ROOT), env=environment)

    log_path = Path(log_file).expanduser().resolve()
    if log_path.exists() and not overwrite:
        raise FileExistsError(
            f"评估日志已存在，使用 --overwrite 才能替换: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation log: {log_path}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {command_text}\n")
        handle.flush()
        return subprocess.run(
            command,
            check=True,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _run_external_reprojection(
        *,
        errors_json: str,
        output_dir: str,
        tool_path: str,
        dry_run: bool,
) -> Optional[subprocess.CompletedProcess]:
    """Invoke the explicitly supplied external reprojection tool after evaluation.

    Args:
        errors_json: Destination for per-frame errors and evaluation metadata.
        output_dir: Output dir: str.
        tool_path: Tool path: str.
        dry_run: Print the resolved configuration or command without running the task.

    Returns:
        Optional[subprocess.CompletedProcess].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    # Validate the external entry point before evaluation.
    resolved_tool = Path(tool_path).expanduser().resolve()
    if not resolved_tool.is_file():
        raise FileNotFoundError("找不到独立点云重投影入口: {}".format(
            resolved_tool))

    # Check errors.json only before a real invocation.
    resolved_errors = Path(errors_json).expanduser().resolve()
    if not dry_run and not resolved_errors.is_file():
        raise FileNotFoundError("找不到评估输出 errors.json: {}".format(
            resolved_errors))
    resolved_output = Path(output_dir).expanduser().resolve()

    # Invoke the external project only through its CLI.
    command = [
        sys.executable,
        str(resolved_tool),
        "--input-type", "errors",
        "--input", str(resolved_errors),
        "--output", str(resolved_output),
    ]
    command_text = shlex.join(command)
    if dry_run:
        print(command_text)
        return None
    return subprocess.run(
        command, check=True, cwd=str(resolved_tool.parent))


def run_emd_test(
        weights: Sequence[str],
        *,
        level: str,
        scene_paths: Sequence[str],
        errors_json: str,
        log_file: Optional[str],
        start_stage: int = 1,
        emd_config: str = DEFAULT_SCENE_CONFIG,
        per_scene_worst_k: Optional[int] = None,
        triptych_dir: Optional[str] = None,
        num_workers: int = 4,
        seed: int = 42,
        gpu: Optional[str] = "0",
        dry_run: bool = False,
        overwrite: bool = False,
        max_frames: Optional[int] = None,
) -> Optional[subprocess.CompletedProcess]:
    """Run scene evaluation through the canonical evaluator entry point.

    Args:
        weights: Checkpoint paths in cascade order.
        level: Configured curriculum or perturbation level.
        scene_paths: Explicit scene directories in loading order.
        errors_json: Destination for per-frame errors and evaluation metadata.
        log_file: Path for the complete run log.
        start_stage: One-based stage number assigned to the first checkpoint.
        emd_config: Scene, sensor, preprocessing, and perturbation configuration.
        per_scene_worst_k: Number of worst frames selected independently per scene.
        triptych_dir: Output directory for reprojection panels.
        num_workers: Number of DataLoader worker processes.
        seed: Explicit seed for reproducible random sampling.
        gpu: CUDA device selection for the subprocess.
        dry_run: Print the resolved configuration or command without running the task.
        overwrite: Explicit permission to replace existing output artifacts.
        max_frames: Optional diagnostic frame limit; None evaluates all selected frames.

    Returns:
        Optional[subprocess.CompletedProcess].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    arguments = build_emd_test_args(
        weights,
        level=level,
        scene_paths=scene_paths,
        errors_json=errors_json,
        start_stage=start_stage,
        emd_config=emd_config,
        per_scene_worst_k=per_scene_worst_k,
        triptych_dir=triptych_dir,
        num_workers=num_workers,
        seed=seed,
        max_frames=max_frames,
    )
    return _run_evaluator(
        arguments, log_file=log_file, gpu=gpu, dry_run=dry_run,
        overwrite=overwrite)


def run_kitti_fixed_nominal_test(
        weights: Sequence[str],
        *,
        sequences: Sequence[str],
        target_cam: str,
        nominal_cam: str,
        errors_json: str,
        log_file: Optional[str],
        data_root: str = DEFAULT_KITTI_DATA_ROOT,
        emd_config: str = DEFAULT_SCENE_CONFIG,
        start_stage: int = 1,
        num_workers: int = 4,
        seed: int = 42,
        gpu: Optional[str] = "0",
        dry_run: bool = False,
        overwrite: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Run KITTI evaluation with distinct target and fixed nominal cameras.

    Args:
        weights: Checkpoint paths in cascade order.
        sequences: Explicit KITTI sequence identifiers.
        target_cam: Rectified KITTI target camera: 2 for left, 3 for right.
        nominal_cam: Distinct KITTI camera supplying fixed nominal T0.
        errors_json: Destination for per-frame errors and evaluation metadata.
        log_file: Path for the complete run log.
        data_root: Dataset root directory.
        emd_config: Scene, sensor, preprocessing, and perturbation configuration.
        start_stage: One-based stage number assigned to the first checkpoint.
        num_workers: Number of DataLoader worker processes.
        seed: Explicit seed for reproducible random sampling.
        gpu: CUDA device selection for the subprocess.
        dry_run: Print the resolved configuration or command without running the task.
        overwrite: Explicit permission to replace existing output artifacts.

    Returns:
        Optional[subprocess.CompletedProcess].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    arguments = build_kitti_fixed_nominal_test_args(
        weights,
        sequences=sequences,
        target_cam=target_cam,
        nominal_cam=nominal_cam,
        errors_json=errors_json,
        data_root=data_root,
        emd_config=emd_config,
        start_stage=start_stage,
        num_workers=num_workers,
        seed=seed,
    )
    return _run_evaluator(
        arguments, log_file=log_file, gpu=gpu, dry_run=dry_run,
        overwrite=overwrite)


def run_kitti_sampled_perturbation_test(
        weights: Sequence[str],
        *,
        sequences: Sequence[str],
        target_cam: str,
        errors_json: str,
        log_file: Optional[str],
        data_root: str = DEFAULT_KITTI_DATA_ROOT,
        emd_config: str = DEFAULT_SCENE_CONFIG,
        start_stage: int = 1,
        max_t: float = 1.5,
        max_r: float = 20.0,
        num_workers: int = 4,
        seed: int = 42,
        gpu: Optional[str] = "0",
        dry_run: bool = False,
        overwrite: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Run KITTI evaluation with the original sampled perturbations.

    Args:
        weights: Checkpoint paths in cascade order.
        sequences: Explicit KITTI sequence identifiers.
        target_cam: Rectified KITTI target camera: 2 for left, 3 for right.
        errors_json: Destination for per-frame errors and evaluation metadata.
        log_file: Path for the complete run log.
        data_root: Dataset root directory.
        emd_config: Scene, sensor, preprocessing, and perturbation configuration.
        start_stage: One-based stage number assigned to the first checkpoint.
        max_t: Translation sampling limit in meters.
        max_r: Rotation sampling limit in degrees.
        num_workers: Number of DataLoader worker processes.
        seed: Explicit seed for reproducible random sampling.
        gpu: CUDA device selection for the subprocess.
        dry_run: Print the resolved configuration or command without running the task.
        overwrite: Explicit permission to replace existing output artifacts.

    Returns:
        Optional[subprocess.CompletedProcess].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    arguments = build_kitti_sampled_perturbation_test_args(
        weights,
        sequences=sequences,
        target_cam=target_cam,
        errors_json=errors_json,
        data_root=data_root,
        emd_config=emd_config,
        start_stage=start_stage,
        max_t=max_t,
        max_r=max_r,
        num_workers=num_workers,
        seed=seed,
    )
    return _run_evaluator(
        arguments, log_file=log_file, gpu=gpu, dry_run=dry_run,
        overwrite=overwrite)


def run_recorded_reprojections(
        *,
        source_errors_json: str,
        output_dir: str,
        log_file: Optional[str],
        per_scene_k: int = 50,
        dry_run: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Render reprojections from recorded errors and cascade poses.

    Args:
        source_errors_json: Source errors json: str.
        output_dir: Output dir: str.
        log_file: Path for the complete run log.
        per_scene_k: Per scene k: int.
        dry_run: Print the resolved configuration or command without running the task.

    Returns:
        Optional[subprocess.CompletedProcess].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    arguments = build_recorded_reprojection_args(
        source_errors_json=source_errors_json,
        output_dir=output_dir,
        per_scene_k=per_scene_k,
    )
    return _run_evaluator(
        arguments, log_file=log_file, gpu=None, dry_run=dry_run)

def run_geometry_domain_gap(
        *, source_errors_json: str, selection_manifest: str,
        dataset_root: str, output_dir: str, log_file: Optional[str],
        train_reference_per_scene: int = 100,
        test_reference_per_scene: int = 200, seed: int = 42,
        dry_run: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Run geometry and domain-gap analysis on recorded evaluation results.

    Args:
        source_errors_json: Source errors json: str.
        selection_manifest: Selection manifest: str.
        dataset_root: Root used to resolve relative scene paths.
        output_dir: Output dir: str.
        log_file: Path for the complete run log.
        train_reference_per_scene: Train reference per scene: int.
        test_reference_per_scene: Test reference per scene: int.
        seed: Explicit seed for reproducible random sampling.
        dry_run: Print the resolved configuration or command without running the task.

    Returns:
        Optional[subprocess.CompletedProcess].
    """
    arguments = build_geometry_domain_gap_args(
        source_errors_json=source_errors_json,
        selection_manifest=selection_manifest, dataset_root=dataset_root,
        output_dir=output_dir,
        train_reference_per_scene=train_reference_per_scene,
        test_reference_per_scene=test_reference_per_scene, seed=seed,
    )
    return _run_evaluator(
        arguments, log_file=log_file, gpu=None, dry_run=dry_run)


def run_gt_flow_pnp_diagnostic(
        *,
        source_errors_json: str,
        selection_manifest: str,
        output_dir: str,
        log_file: Optional[str],
        repetitions: int = 5,
        gpu: Optional[str] = "0",
        dry_run: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Run the recorded-frame ground-truth-flow PnP comparison.

    Args:
        source_errors_json: Source errors json: str.
        selection_manifest: Selection manifest: str.
        output_dir: Output dir: str.
        log_file: Path for the complete run log.
        repetitions: Repetitions: int.
        gpu: CUDA device selection for the subprocess.
        dry_run: Print the resolved configuration or command without running the task.

    Returns:
        Optional[subprocess.CompletedProcess].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    arguments = build_gt_flow_pnp_diagnostic_args(
        source_errors_json=source_errors_json,
        selection_manifest=selection_manifest,
        output_dir=output_dir,
        repetitions=repetitions,
    )
    return _run_evaluator(
        arguments, log_file=log_file, gpu=gpu, dry_run=dry_run)



@dataclass(frozen=True)
class KITTIFixedNominalCase:
    """Describe a KITTI case with target-camera images and a distinct fixed nominal-camera calibration.


    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    sequences: tuple[str, ...]
    target_cam: str
    nominal_cam: str
    errors_json: str
    log_file: Optional[str]


@dataclass(frozen=True)
class KITTISampledPerturbationCase:
    """Describe a KITTI case using the original per-frame perturbation sampler.
    """
    sequences: tuple[str, ...]
    target_cam: str
    errors_json: str
    log_file: Optional[str]
    max_t: float = 1.5
    max_r: float = 20.0


def run_post_training_suite(
        weights: Sequence[str],
        *,
        emd_level: str,
        emd_scene_paths: Sequence[str],
        emd_errors_json: str,
        emd_log_file: Optional[str],
        emd_start_stage: int = 1,
        kitti_cases: Sequence[KITTIFixedNominalCase] = (),
        kitti_sampled_cases: Sequence[KITTISampledPerturbationCase] = (),
        emd_config: str = DEFAULT_SCENE_CONFIG,
        kitti_data_root: str = DEFAULT_KITTI_DATA_ROOT,
        num_workers: int = 4,
        seed: int = 42,
        gpu: Optional[str] = "0",
        dry_run: bool = False,
) -> None:
    """Evaluate explicit generated scenes before KITTI cases and then archive the training and evaluation artifacts.

    Args:
        weights: Checkpoint paths in cascade order.
        emd_level: Emd level: str.
        emd_scene_paths: Emd scene paths: Sequence[str].
        emd_errors_json: Emd errors json: str.
        emd_log_file: Emd log file: Optional[str].
        emd_start_stage: Emd start stage: int.
        kitti_cases: Kitti cases: Sequence[KITTIFixedNominalCase].
        kitti_sampled_cases: Kitti sampled cases: Sequence[KITTISampledPerturbationCase].
        emd_config: Scene, sensor, preprocessing, and perturbation configuration.
        kitti_data_root: Kitti data root: str.
        num_workers: Number of DataLoader worker processes.
        seed: Explicit seed for reproducible random sampling.
        gpu: CUDA device selection for the subprocess.
        dry_run: Print the resolved configuration or command without running the task.

    Returns:
        None.

    Notes:
        Execution may write logs, per-frame errors, visualizations, and archived training artifacts; dry_run only renders the commands.
    """
    run_emd_test(
        weights,
        level=emd_level,
        scene_paths=emd_scene_paths,
        errors_json=emd_errors_json,
        log_file=emd_log_file,
        start_stage=emd_start_stage,
        emd_config=emd_config,
        num_workers=num_workers,
        seed=seed,
        gpu=gpu,
        dry_run=dry_run,
    )
    for case in kitti_cases:
        run_kitti_fixed_nominal_test(
            weights,
            sequences=case.sequences,
            target_cam=case.target_cam,
            nominal_cam=case.nominal_cam,
            errors_json=case.errors_json,
            log_file=case.log_file,
            data_root=kitti_data_root,
            emd_config=emd_config,
            start_stage=emd_start_stage,
            num_workers=num_workers,
            seed=seed,
            gpu=gpu,
            dry_run=dry_run,
        )

    for case in kitti_sampled_cases:
        run_kitti_sampled_perturbation_test(
            weights,
            sequences=case.sequences,
            target_cam=case.target_cam,
            errors_json=case.errors_json,
            log_file=case.log_file,
            data_root=kitti_data_root,
            emd_config=emd_config,
            start_stage=emd_start_stage,
            max_t=case.max_t,
            max_r=case.max_r,
            num_workers=num_workers,
            seed=seed,
            gpu=gpu,
            dry_run=dry_run,
        )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared checkpoint, stage, seed, logging, and output arguments.

    Args:
        parser: Parser: argparse.ArgumentParser.

    Returns:
        None.
    """
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--errors-json", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument(
        "--config", dest="emd_config", metavar="PATH", required=True,
        help="场景、传感器、图像预处理和扰动方式 YAML")
    parser.add_argument("--start-stage", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="显式允许替换已有评估日志和 errors.json")
    parser.add_argument(
        "--reprojection", action="store_true",
        help="评估成功后调用独立项目生成全部点云重投影 PNG")
    parser.add_argument(
        "--reprojection-output", default=None,
        help="独立重投影项目的分类 PNG 输出根目录")
    parser.add_argument(
        "--reprojection-tool", default=None,
        help="使用 --reprojection 时必须显式提供的 reproject.py 入口")


def _resolve_emd_test_scene_paths(
    emd_config_path: str,
    command_scene_paths: Optional[Sequence[str]],
) -> list[str]:
    """Resolve explicit test scenes, falling back to the YAML test list only when no override is supplied.

    Args:
        emd_config_path: Path to the scene configuration YAML.
        command_scene_paths: CLI scene override; None uses the YAML test list.

    Returns:
        list[str].

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    if command_scene_paths is not None:
        resolved_paths = [
            str(Path(path).expanduser().resolve())
            for path in command_scene_paths]
    else:
        emd_config = load_emd_training_config(emd_config_path)
        resolved_paths = [
            str(path) for path in emd_config.test_scene_paths]
    if not resolved_paths:
        raise ValueError(
            "EMD 测试场景为空；请填写 YAML scenes.test 或传入 --scene-paths")
    return resolved_paths




def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the supported evaluation modes.


    Returns:
        argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="CMRNext 统一测试入口")
    subparsers = parser.add_subparsers(dest="test_mode", required=True)

    scene_parser = subparsers.add_parser(
        "scene", help="评估显式指定的相机-LiDAR 场景目录")
    _add_common_arguments(scene_parser)
    scene_parser.add_argument(
        "--level",
        choices=["large", "medium", "small", "fixed_baseline", "perturbed_t0"],
        required=True)
    scene_parser.add_argument("--per-scene-worst-k", type=int, default=None)
    scene_parser.add_argument("--triptych-dir", default=None)
    scene_parser.add_argument(
        "--max-frames", type=int, default=None,
        help="仅诊断时限制推理帧数；正式评估不应设置")
    scene_parser.add_argument(
        "--scene-paths", nargs="+", default=None,
        help="可选场景覆盖；省略时读取 YAML scenes.test")

    recorded_parser = subparsers.add_parser(
        "recorded-reprojections",
        help="根据记录的级联位姿渲染各场景误差最大的帧")
    recorded_parser.add_argument("--source-errors-json", required=True)
    recorded_parser.add_argument("--output-dir", required=True)
    recorded_parser.add_argument("--log-file", required=True)
    recorded_parser.add_argument("--per-scene-k", type=int, default=50)
    recorded_parser.add_argument("--dry-run", action="store_true")

    geometry_parser = subparsers.add_parser(
        "geometry-domain-gap",
        help="分析重力、深度、视差和几何域差异")
    geometry_parser.add_argument("--source-errors-json", required=True)
    geometry_parser.add_argument("--selection-manifest", required=True)
    geometry_parser.add_argument("--dataset-root", required=True)
    geometry_parser.add_argument("--output-dir", required=True)
    geometry_parser.add_argument("--log-file", required=True)
    geometry_parser.add_argument(
        "--train-reference-per-scene", type=int, default=100)
    geometry_parser.add_argument(
        "--test-reference-per-scene", type=int, default=200)
    geometry_parser.add_argument("--seed", type=int, default=42)
    geometry_parser.add_argument("--dry-run", action="store_true")


    kitti_parser = subparsers.add_parser(
        "kitti", help="使用固定左右相机标称外参评估 KITTI")
    _add_common_arguments(kitti_parser)
    kitti_parser.add_argument(
        "--data-root",
        default=DEFAULT_KITTI_DATA_ROOT)
    kitti_parser.add_argument("--sequences", nargs="+", required=True)
    kitti_parser.add_argument(
        "--target-cam", choices=["2", "3"], required=True)
    kitti_parser.add_argument(
        "--nominal-cam", choices=["2", "3"], required=True)

    perturbed_parser = subparsers.add_parser(
        "kitti-perturbed",
        help="使用原始 CMRNext 采样扰动评估 KITTI")
    _add_common_arguments(perturbed_parser)
    perturbed_parser.add_argument(
        "--data-root",
        default=DEFAULT_KITTI_DATA_ROOT)
    perturbed_parser.add_argument("--sequences", nargs="+", required=True)
    perturbed_parser.add_argument(
        "--target-cam", choices=["2", "3"], required=True)
    perturbed_parser.add_argument("--max-t", type=float, default=1.5)
    perturbed_parser.add_argument("--max-r", type=float, default=20.0)

    gt_flow_parser = subparsers.add_parser(
        "gt-flow-pnp",
        help="比较记录的预测光流 PnP 与精确真实光流 PnP")
    gt_flow_parser.add_argument("--source-errors-json", required=True)
    gt_flow_parser.add_argument("--selection-manifest", required=True)
    gt_flow_parser.add_argument("--output-dir", required=True)
    gt_flow_parser.add_argument("--log-file", required=True)
    gt_flow_parser.add_argument("--repetitions", type=int, default=5)
    gt_flow_parser.add_argument("--gpu", default="0")
    gt_flow_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    # Keep the parser to report missing reprojection arguments clearly.
    """Run test with the resolved command-line or runtime configuration.

    Args:
        argv: Optional command-line arguments; None uses the process arguments.

    Returns:
        None.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.test_mode in {"scene", "kitti", "kitti-perturbed"}:
        if args.reprojection and args.reprojection_output is None:
            parser.error("使用 --reprojection 时必须提供 --reprojection-output")
        if args.reprojection and args.reprojection_tool is None:
            parser.error("使用 --reprojection 时必须提供 --reprojection-tool")
    if args.test_mode == "recorded-reprojections":
        run_recorded_reprojections(
            source_errors_json=args.source_errors_json,
            output_dir=args.output_dir,
            log_file=args.log_file,
            per_scene_k=args.per_scene_k,
            dry_run=args.dry_run,
        )
        return
    if args.test_mode == "geometry-domain-gap":
        run_geometry_domain_gap(
            source_errors_json=args.source_errors_json,
            selection_manifest=args.selection_manifest,
            dataset_root=args.dataset_root, output_dir=args.output_dir,
            log_file=args.log_file,
            train_reference_per_scene=args.train_reference_per_scene,
            test_reference_per_scene=args.test_reference_per_scene,
            seed=args.seed, dry_run=args.dry_run,
        )
        return
    if args.test_mode == "gt-flow-pnp":
        run_gt_flow_pnp_diagnostic(
            source_errors_json=args.source_errors_json,
            selection_manifest=args.selection_manifest,
            output_dir=args.output_dir,
            log_file=args.log_file,
            repetitions=args.repetitions,
            gpu=args.gpu,
            dry_run=args.dry_run,
        )
        return

    common = {
        "weights": args.weights,
        "errors_json": args.errors_json,
        "log_file": args.log_file,
        "emd_config": args.emd_config,
        "start_stage": args.start_stage,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "gpu": args.gpu,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
    }
    if args.test_mode == "scene":
        try:
            args.scene_paths = _resolve_emd_test_scene_paths(
                args.emd_config, args.scene_paths)
        except ValueError as error:
            parser.error(str(error))

        run_emd_test(
            level=args.level,
            scene_paths=args.scene_paths,
            per_scene_worst_k=args.per_scene_worst_k,
            triptych_dir=args.triptych_dir,
            max_frames=args.max_frames,
            **common,
        )
    elif args.test_mode == "kitti":
        run_kitti_fixed_nominal_test(
            sequences=args.sequences,
            target_cam=args.target_cam,
            nominal_cam=args.nominal_cam,
            data_root=args.data_root,
            **common,
        )
    elif args.test_mode == "kitti-perturbed":
        run_kitti_sampled_perturbation_test(
            sequences=args.sequences,
            target_cam=args.target_cam,
            data_root=args.data_root,
            max_t=args.max_t,
            max_r=args.max_r,
            **common,
        )
    else:
        raise ValueError(f"不支持的测试模式：{args.test_mode}")

    # Render reprojections only after evaluation writes errors.json.
    if args.reprojection:
        _run_external_reprojection(
            errors_json=args.errors_json,
            output_dir=args.reprojection_output,
            tool_path=args.reprojection_tool,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
