"""统一评估命令解析与评估任务调度。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
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
    """完成 统一评估命令解析与评估任务调度 中的 `_require_weights` 操作。

    参数:
        weights: 按级联阶段顺序排列的 checkpoint 路径序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """构造 EMD 级联评估器参数。

    参数:
        weights: 按级联顺序排列且必须存在的 checkpoint 路径。
        level: 评估数据的 curriculum level。
        scene_paths: 用户显式选择的一个或多个场景目录。
        errors_json: 逐帧误差和审计信息输出路径。
        start_stage: 第一个 checkpoint 对应的阶段编号。
        emd_config: 传感器和评估扰动模式 YAML。
        per_scene_worst_k: 每场景最差样本渲染数量。
        triptych_dir: 最差样本渲染目录。
        num_workers: DataLoader 工作进程数。
        seed: 裁剪、在线扰动和评估随机源的显式种子。
        max_frames: 仅诊断时最多执行的帧数；None 表示完整划分。

    返回值:
        传给 evaluate_flow_calibration.py 的参数列表。

    异常和副作用:
        参数非法或 checkpoint 缺失时抛出异常；本函数不写文件。

    坐标系和外参约定:
        输入使用 T0 @ p_lidar，目标使用 Ti @ p_lidar，左乘残差为
        delta_Ti = Ti @ inverse(T0)。checkpoint 不按训练数据标签限制评估场景。
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
    """根据显式输入构建下游调用所需的参数、报告或数据结构。

    参数:
        weights: 按级联阶段顺序排列的 checkpoint 路径序列。
        sequences: 用户选择的数据序列编号列表。
        target_cam: 提供目标图像、内参和真实外参的相机编号。
        nominal_cam: 提供固定标称 T0 的另一相机编号。
        errors_json: 逐帧评估误差 JSON 的输出路径。
        data_root: 数据集根目录路径。
        emd_config: EMD 传感器、裁剪、课程级别和扰动配置 YAML 路径。
        start_stage: 当前 checkpoint 列表对应的首个级联阶段编号。
        num_workers: DataLoader 工作进程数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """根据显式输入构建下游调用所需的参数、报告或数据结构。

    参数:
        weights: 按级联阶段顺序排列的 checkpoint 路径序列。
        sequences: 用户选择的数据序列编号列表。
        target_cam: 提供目标图像、内参和真实外参的相机编号。
        errors_json: 逐帧评估误差 JSON 的输出路径。
        data_root: 数据集根目录路径。
        emd_config: EMD 传感器、裁剪、课程级别和扰动配置 YAML 路径。
        start_stage: 当前 checkpoint 列表对应的首个级联阶段编号。
        max_t: 逐轴平移扰动上限，单位为米。
        max_r: 逐轴欧拉角扰动上限，单位为度。
        num_workers: DataLoader 工作进程数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """根据显式输入构建下游调用所需的参数、报告或数据结构。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        selection_manifest: 明确记录诊断所选帧的清单文件路径。
        output_dir: 任务产物目录路径。
        repetitions: 当前诊断操作的重复次数。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """根据显式输入构建下游调用所需的参数、报告或数据结构。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        output_dir: 任务产物目录路径。
        per_scene_k: 每个场景选择并渲染的样本数量。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """根据显式输入构建下游调用所需的参数、报告或数据结构。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        selection_manifest: 明确记录诊断所选帧的清单文件路径。
        dataset_root: 待分析数据集的根目录路径。
        output_dir: 任务产物目录路径。
        train_reference_per_scene: 每个训练场景抽取的几何参考帧数量。
        test_reference_per_scene: 每个测试场景抽取的几何参考帧数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """执行共享评估器并保存完整审计日志。

    参数:
        arguments: 传给 evaluate_flow_calibration.py 的参数列表。
        log_file: 子进程标准输出和标准错误的日志路径；None 表示直接输出到终端。
        gpu: 写入 CUDA_VISIBLE_DEVICES 的设备编号；None 表示不覆盖环境变量。
        dry_run: 为 True 时只打印完整 Python 命令。
        overwrite: 为 True 时允许替换已有日志和 errors.json。

    返回值:
        实际执行时返回 CompletedProcess，dry-run 时返回 None。

    异常和副作用:
        默认遇到已有正式输出时抛出 FileExistsError；子进程非零退出时抛出
        CalledProcessError。真实执行会创建日志目录并写入日志和评估结果。

    坐标系和外参约定:
        本函数只负责进程与产物，不改变 T0、Ti 或左乘残差。
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
    """完成 统一评估命令解析与评估任务调度 中的 `_run_external_reprojection` 操作。

    参数:
        errors_json: 逐帧评估误差 JSON 的输出路径。
        output_dir: 任务产物目录路径。
        tool_path: 当前操作读取或写入的文件路径。
        dry_run: 为 True 时仅打印将执行的命令，不启动实际任务。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """通过规范 Python 入口执行一次 EMD 评估。

    参数:
        weights: 按阶段顺序排列的 checkpoint。
        level: 评估数据 level。
        scene_paths: 直接读取的场景目录列表。
        errors_json: 结构化结果路径。
        log_file: 完整子进程日志路径。
        start_stage: 第一个 checkpoint 的阶段编号。
        emd_config: 数据与传感器配置。
        per_scene_worst_k: 每场景可视化数量。
        triptych_dir: 可视化目录。
        num_workers: DataLoader 进程数。
        seed: 显式随机种子。
        gpu: CUDA_VISIBLE_DEVICES 使用的 GPU 编号。
        dry_run: 只打印命令而不执行。
        overwrite: 是否显式允许替换已有日志和 errors.json。
        max_frames: 诊断推理帧数上限。

    返回值:
        执行时返回 CompletedProcess，dry-run 时返回 None。

    异常和副作用:
        参数非法或子进程失败时抛出异常；真实执行写日志和 errors.json。
        T0、Ti 和左乘残差约定与 build_emd_test_args 保持一致。
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
    """组织并运行对应评估或诊断任务，同时保留日志和退出状态。

    参数:
        weights: 按级联阶段顺序排列的 checkpoint 路径序列。
        sequences: 用户选择的数据序列编号列表。
        target_cam: 提供目标图像、内参和真实外参的相机编号。
        nominal_cam: 提供固定标称 T0 的另一相机编号。
        errors_json: 逐帧评估误差 JSON 的输出路径。
        log_file: 完整标准输出和错误输出的日志文件路径。
        data_root: 数据集根目录路径。
        emd_config: EMD 传感器、裁剪、课程级别和扰动配置 YAML 路径。
        start_stage: 当前 checkpoint 列表对应的首个级联阶段编号。
        num_workers: DataLoader 工作进程数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。
        gpu: 用于运行任务的 CUDA 设备编号；单位为设备索引。
        dry_run: 为 True 时仅打印将执行的命令，不启动实际任务。
        overwrite: 是否允许覆盖目标正式输出；必须由调用方显式启用。

    返回值:
        实际执行时返回子进程结果或任务产物；dry-run 时可能返回 None。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """组织并运行对应评估或诊断任务，同时保留日志和退出状态。

    参数:
        weights: 按级联阶段顺序排列的 checkpoint 路径序列。
        sequences: 用户选择的数据序列编号列表。
        target_cam: 提供目标图像、内参和真实外参的相机编号。
        errors_json: 逐帧评估误差 JSON 的输出路径。
        log_file: 完整标准输出和错误输出的日志文件路径。
        data_root: 数据集根目录路径。
        emd_config: EMD 传感器、裁剪、课程级别和扰动配置 YAML 路径。
        start_stage: 当前 checkpoint 列表对应的首个级联阶段编号。
        max_t: 逐轴平移扰动上限，单位为米。
        max_r: 逐轴欧拉角扰动上限，单位为度。
        num_workers: DataLoader 工作进程数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。
        gpu: 用于运行任务的 CUDA 设备编号；单位为设备索引。
        dry_run: 为 True 时仅打印将执行的命令，不启动实际任务。
        overwrite: 是否允许覆盖目标正式输出；必须由调用方显式启用。

    返回值:
        实际执行时返回子进程结果或任务产物；dry-run 时可能返回 None。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """组织并运行对应评估或诊断任务，同时保留日志和退出状态。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        output_dir: 任务产物目录路径。
        log_file: 完整标准输出和错误输出的日志文件路径。
        per_scene_k: 每个场景选择并渲染的样本数量。
        dry_run: 为 True 时仅打印将执行的命令，不启动实际任务。

    返回值:
        实际执行时返回子进程结果或任务产物；dry-run 时可能返回 None。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """组织并运行对应评估或诊断任务，同时保留日志和退出状态。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        selection_manifest: 明确记录诊断所选帧的清单文件路径。
        dataset_root: 待分析数据集的根目录路径。
        output_dir: 任务产物目录路径。
        log_file: 完整标准输出和错误输出的日志文件路径。
        train_reference_per_scene: 每个训练场景抽取的几何参考帧数量。
        test_reference_per_scene: 每个测试场景抽取的几何参考帧数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。
        dry_run: 为 True 时仅打印将执行的命令，不启动实际任务。

    返回值:
        实际执行时返回子进程结果或任务产物；dry-run 时可能返回 None。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """组织并运行对应评估或诊断任务，同时保留日志和退出状态。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        selection_manifest: 明确记录诊断所选帧的清单文件路径。
        output_dir: 任务产物目录路径。
        log_file: 完整标准输出和错误输出的日志文件路径。
        repetitions: 当前诊断操作的重复次数。
        gpu: 用于运行任务的 CUDA 设备编号；单位为设备索引。
        dry_run: 为 True 时仅打印将执行的命令，不启动实际任务。

    返回值:
        实际执行时返回子进程结果或任务产物；dry-run 时可能返回 None。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """封装 统一评估命令解析与评估任务调度 中的 `KITTIFixedNominalCase` 状态和操作。

    主要职责:
        封装 统一评估命令解析与评估任务调度 中的 `KITTIFixedNominalCase` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
    """
    sequences: tuple[str, ...]
    target_cam: str
    nominal_cam: str
    errors_json: str
    log_file: Optional[str]


@dataclass(frozen=True)
class KITTISampledPerturbationCase:
    """封装 统一评估命令解析与评估任务调度 中的 `KITTISampledPerturbationCase` 状态和操作。

    主要职责:
        封装 统一评估命令解析与评估任务调度 中的 `KITTISampledPerturbationCase` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """按固定顺序运行 EMD 评估以及用户指定的 KITTI 评估并归档日志。

    参数:
        weights: 按级联阶段顺序排列的 checkpoint 路径序列。
        emd_level: 当前 EMD 课程级别或训练阶段名称。
        emd_scene_paths: 生成数据评估使用的显式场景目录列表。
        emd_errors_json: EMD 逐帧误差 JSON 输出路径。
        emd_log_file: 当前操作读取或写入的文件路径。
        emd_start_stage: EMD checkpoint 列表对应的首个级联阶段编号。
        kitti_cases: 需要运行的固定标称 KITTI 评估用例序列。
        kitti_sampled_cases: 需要运行的原版随机扰动 KITTI 评估用例序列。
        emd_config: EMD 传感器、裁剪、课程级别和扰动配置 YAML 路径。
        kitti_data_root: KITTI Odometry 数据根目录。
        num_workers: DataLoader 工作进程数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。
        gpu: 用于运行任务的 CUDA 设备编号；单位为设备索引。
        dry_run: 为 True 时仅打印将执行的命令，不启动实际任务。

    返回值:
        实际执行时返回子进程结果或任务产物；dry-run 时可能返回 None。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """完成 统一评估命令解析与评估任务调度 中的 `_add_common_arguments` 操作。

    参数:
        parser: 需要添加公共选项的 argparse 解析器。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """解析 EMD 测试场景，并让命令行整组覆盖 YAML 默认值。

    参数:
        emd_config_path: EMD v2 YAML 文件路径。
        command_scene_paths: --scene-paths 提供的路径；None 表示使用
            YAML dataset.scenes.test。

    返回值:
        可直接传给共享评估器的场景绝对路径列表。

    异常和副作用:
        命令行与 YAML 最终都没有测试场景时抛出 ValueError。函数只读取 YAML，
        不检查 split、metadata、manifest 或场景统计文件。

    坐标系和外参约定:
        场景选择不改变输入 T0、目标 Ti 或 delta_Ti = Ti @ inverse(T0)。
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
    """构造统一测试入口的命令行解析器及所有受支持子命令。

    参数:
        无。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
    """执行当前模块的主流程；在入口模块中负责解析或运行完整训练、评估任务。

    参数:
        argv: 可选命令行参数序列；为 None 时读取当前进程命令行。

    返回值:
        无显式返回值；成功或失败通过进程退出码、日志和输出文件体现。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。
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
