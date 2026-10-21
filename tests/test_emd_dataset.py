"""项目源码模块 tests/test_emd_dataset.py。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from datasets.emd_extrinsic_calib import (
    DatasetEMDExtrinsicCalib,
    load_emd_training_config,
)
from datasets.emd_dynamic_nominal import (
    DatasetEMDDynamicNominalExtrinsicCalib,
    DynamicNominalLevelConfig,
    EMDDynamicNominalConfig,
    load_emd_dynamic_nominal_config,
)
from datasets.emd_fixed_t0_online import (
    DatasetEMDFixedT0OnlineExtrinsicCalib,
    DatasetEMDPerTiOnlineExtrinsicCalib,
    load_fixed_t0_online_perturbation_config,
    load_per_ti_online_perturbation_config,
)
from evaluate_flow_calibration import (
    _camera_matrix_for_crop,
    _checkpoint_training_domain_record,
    _crop_evaluation_frame,
    _draw_reprojection_panels,
    _build_flow_epe_report,
    _distribution_summary,
    _estimate_from_left_residual,
    _mask_evaluation_flow_endpoints,
    _resolve_evaluation_image_shape,
    _residual_component_summary,
    _residual_components,
    _sample_emd_crop_offsets,
    _select_emd_worst_per_scene,
)
from utils import merge_inputs, quat2mat, tvector2mat


def _write_fixture(
    tmp_path: Path,
    metadata_overrides=None,
    *,
    level="large",
    split="train",
    target_network="cmrnext",
    fixed_identity=False,
):
    """把当前函数接收的数据编码并写入目标文件。

    参数:
        tmp_path: pytest 提供的独立临时目录。
        metadata_overrides: 测试夹具写入 metadata.json 的字段覆盖映射。
        level: EMD 课程或级联阶段名称。
        split: 仅用于记录来源的数据角色文本，不参与显式场景选择。
        target_network: 测试夹具声明的目标网络名称。
        fixed_identity: 是否构造 Ti=T0 且残差为单位阵的固定外参测试场景。

    返回值:
        无显式返回值；结果写入调用方指定的文件或输出流。

    异常和副作用:
        路径或编码失败时抛出异常；会创建或更新调用方指定的输出文件。
    """
    root = tmp_path / "emd"
    sequence = root / level / split / "scene_a"
    image_dir = sequence / "image_2"
    cloud_dir = sequence / "velodyne"
    image_dir.mkdir(parents=True)
    cloud_dir.mkdir()

    fingerprint = "task-fingerprint"
    (root / "task_manifest.json").write_text(json.dumps({
        "status": "complete",
        "target_network": target_network,
        "required_levels": (
            [level] if fixed_identity else ["large", "medium", "small"]),
        "task_config_fingerprint": fingerprint,
    }))

    angle = np.deg2rad(5.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    nominal = np.eye(4)
    nominal[:3, 3] = [0.3, -0.1, 0.2]
    if fixed_identity:
        residuals = np.repeat(np.eye(4)[None], 2, axis=0)
    else:
        residuals = []
        for translation in ([0.04, -0.02, 0.01], [-0.03, 0.01, 0.02]):
            residual = np.eye(4)
            residual[:3, :3] = rotation
            residual[:3, 3] = translation
            residuals.append(residual)
        residuals = np.stack(residuals)
    true_extrinsics = residuals @ nominal

    np.savetxt(sequence / "nominal_extrinsic.txt", nominal[:3])
    np.savetxt(sequence / "residuals.txt", residuals[:, :3].reshape(2, 12))
    np.savetxt(sequence / "poses.txt", true_extrinsics[:, :3].reshape(2, 12))
    (sequence / "calib.txt").write_text(
        "P2: 707.0912 0 601.8873 0 0 707.0912 183.1104 0 0 0 1 0\n")

    for index, count in enumerate((2, 3)):
        Image.new("RGB", (1226, 370), color=(index, 2, 3)).save(
            image_dir / f"{index:06d}.png")
        points = np.zeros((count, 4), dtype=np.float32)
        points[:, :3] = np.array([1.0, 0.1, 5.0], dtype=np.float32)
        points.tofile(cloud_dir / f"{index:06d}.bin")

    metadata = {
        "training_semantics": "nominal_to_true_residual_v1",
        "target_network": target_network,
        "curriculum_level": level,
        "split": split,
        "n_samples": 2,
        "task": {
            "config_fingerprints": {"task": fingerprint},
            "residual_sampler": {
                "change_frame": False,
                "positive_named_z_cap_m": 1.0,
            },
        },
        "level": {
            "effective_max_translation_m": 1.5,
            "residual_translation_normalization_m": 2.598076211353316,
            "max_translation_component_m": 1.5,
            "translation_limit_semantics": (
                "fixed_absolute_per_named_component_box"),
            "max_euler_component_deg": 20.0,
            "max_geodesic_rotation_deg": 36.4352,
        },
        "sensor_setup": {
            "profile": "kitti_odometry_04_12_rectified_hdl64e",
            "camera": {
                "width": 1226, "height": 370,
                "fx": 707.0912, "fy": 707.0912,
                "cx": 601.8873, "cy": 183.1104,
            },
            "lidar": {
                "num_rings": 64, "points_per_ring": 2083,
                "vertical_fov_deg": [-24.9, 2.0],
            },
        },
        "samples": [{"frame_id": "000000"}, {"frame_id": "000001"}],
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (sequence / "metadata.json").write_text(json.dumps(metadata))
    (sequence / "manifest.json").write_text(json.dumps({
        "split": split, "sample_count": 2, "audit": {"valid": True},
    }))

    config_path = tmp_path / "emd.yaml"
    config = {
        "schema_version": "cmrnext_emd_training_v2",
        "dataset": {
            "root": str(root),
            "training_semantics": "nominal_to_true_residual_v1",
            "target_network": target_network,
            "dummy_intensity": 0.0,
            "scenes": {
                "train": [f"{level}/{split}/scene_a"],
                "validation": [],
                "test": [],
            },
            "files": {
                "calibration": "calib.txt",
                "nominal_extrinsic": "nominal_extrinsic.txt",
                "true_extrinsics": "poses.txt",
                "residuals": "residuals.txt",
            },
            "sensor": {
                "profile": "kitti_odometry_04_12_rectified_hdl64e",
                "camera": {
                    "width": 1226, "height": 370,
                    "fx": 707.0912, "fy": 707.0912,
                    "cx": 601.8873, "cy": 183.1104,
                },
                "lidar": {
                    "rings": 64, "points_per_ring": 2083,
                    "acquisition": {
                        "repeated_pattern": True,
                        "scan_frequency_hz": None,
                        "beams_per_scan": None,
                        "beam_rate_hz": None,
                        "accumulation_duration_s": None,
                    },
                    "vertical_fov_deg": [-24.9, 2.0],
                },
            },
            "validation": {
                "transform_atol": 1e-5,
                "require_sequence_audit": True,
                "require_continuous_coverage": False,
            },
        },
        "augmentation": {"color_jitter": False, "geometric": False},
        "evaluation": {
            "success_translation_m": 0.1,
            "success_rotation_deg": 1.0,
            "residual_magnitude_bins": [0.0, 0.5, 1.0, 1.5, 2.0],
        },
        "curriculum": (
            {"iter1": level}
            if fixed_identity else
            {"iter1": "large", "iter2": "medium", "iter3": "small"}
        ),
    }
    if fixed_identity:
        config["dataset"]["source_mode"] = "calibnet_fixed_t0_v1"
        config["fixed_t0_online_perturbation"] = {
            "enabled": True,
            "sampler": "cmrnext_legacy_per_axis_box_v1",
            "max_translation_component_m": 0.6,
            "max_rotation_euler_component_deg": 10.0,
            "seed": 42,
            "train_resample_each_access": True,
            "deterministic_splits": ["val", "test"],
        }
    config_path.write_text(yaml.safe_dump(config))
    return root, config_path, nominal, residuals, true_extrinsics


def _write_recorded_pose_fixture(
    tmp_path: Path,
    *,
    fixed_t0: bool,
):
    """构造使用 image、mid360 和七列 pose.txt 的真实域测试夹具。

    参数:
        tmp_path: pytest 提供的独立临时目录。
        fixed_t0: 为 True 时令所有 Ti 等于 T0；为 False 时写入相对 T0
            具有平移和旋转变化的 Ti。

    返回值:
        配置路径、场景路径、四乘四 T0 和逐帧四乘四 Ti；平移单位均为米。

    异常和副作用:
        文件创建或图像编码失败时抛出异常；只在 pytest 临时目录写入夹具。

    坐标系和外参约定:
        pose.txt 采用 tx、ty、tz、qx、qy、qz、qw，矩阵满足
        p_camera = Ti @ p_lidar，左乘残差为 Ti @ inverse(T0)。
    """
    root = tmp_path / "recorded"
    scene = root / "test" / "scene_a"
    image_dir = scene / "image"
    point_cloud_dir = scene / "mid360"
    image_dir.mkdir(parents=True)
    point_cloud_dir.mkdir()

    nominal = np.eye(4, dtype=np.float64)
    nominal[:3, 3] = [0.1, -0.2, 0.3]
    if fixed_t0:
        pose_rows = np.array([
            [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        true_extrinsics = np.repeat(nominal[None], 2, axis=0)
    else:
        half_sqrt = math.sqrt(0.5)
        pose_rows = np.array([
            [0.2, -0.1, 0.4, 0.0, 0.0, half_sqrt, half_sqrt],
            [0.3, 0.0, 0.5, 0.0, 0.0, half_sqrt, half_sqrt],
        ], dtype=np.float64)
        true_extrinsics = np.repeat(np.eye(4)[None], 2, axis=0)
        true_extrinsics[:, :3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        true_extrinsics[:, :3, 3] = pose_rows[:, :3]

    np.savetxt(scene / "pose.txt", pose_rows, fmt="%.12f")
    (scene / "calib.txt").write_text(
        "P0: 4 0 3 0 0 4 2 0 0 0 1 0\n")
    for index in range(2):
        Image.new("RGB", (8, 6), color=(index, 2, 3)).save(
            image_dir / f"{index:06d}.png")
        points = np.array([
            [1.0, 0.1, 5.0, 0.0],
            [2.0, 0.2, 6.0, 0.0],
        ], dtype=np.float32)
        points.tofile(point_cloud_dir / f"{index:06d}.bin")

    config = {
        "schema_version": "cmrnext_emd_training_v2",
        "dataset": {
            "root": str(root),
            "source_mode": "recorded_pose_v1",
            "training_semantics": "nominal_to_true_residual_v1",
            "target_network": "calibnet",
            "dummy_intensity": 0.0,
            "scenes": {
                "train": [],
                "validation": [],
                "test": ["test/scene_a"],
            },
            "files": {
                "calibration": "calib.txt",
                "true_extrinsics": "pose.txt",
            },
            "layout": {"images": "image", "point_clouds": "mid360"},
            "recorded_pose": {
                "format": "translation_xyz_quaternion_xyzw",
                "nominal_extrinsic": nominal.tolist(),
            },
            "sensor": {
                "profile": "livox_mid360_recorded_test",
                "camera": {
                    "width": 8,
                    "height": 6,
                    "fx": 4.0,
                    "fy": 4.0,
                    "cx": 3.0,
                    "cy": 2.0,
                    "skew": 0.0,
                },
                "lidar": {
                    "rings": 4,
                    "points_per_ring": 5000,
                    "vertical_fov_deg": [-7.2123, 52.164],
                    "acquisition": {
                        "repeated_pattern": False,
                        "scan_frequency_hz": None,
                        "beams_per_scan": None,
                        "beam_rate_hz": None,
                        "accumulation_duration_s": None,
                    },
                },
            },
            "validation": {
                "transform_atol": 1.0e-5,
                "require_sequence_audit": False,
                "require_continuous_coverage": False,
            },
        },
        "preprocessing": {"resize_hw": [6, 8]},
        "augmentation": {"color_jitter": False, "geometric": False},
        "evaluation": {
            "success_translation_m": 0.1,
            "success_rotation_deg": 1.0,
            "residual_magnitude_bins": [0.0, 0.1, 0.2, 0.5],
        },
        "curriculum": {
            "iter1": "fixed_baseline" if fixed_t0 else "perturbed_t0"},
    }
    if fixed_t0:
        config["fixed_t0_online_perturbation"] = {
            "enabled": True,
            "sampler": "cmrnext_legacy_per_axis_box_v1",
            "max_translation_component_m": 0.2,
            "max_rotation_euler_component_deg": 10.0,
            "train_resample_each_access": True,
        }
    config_path = tmp_path / "recorded.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, scene, nominal, true_extrinsics


def test_recorded_pose_dataset_reads_custom_layout_and_xyzw(tmp_path):
    """验证真实域目录与 xyzw 位姿会生成正确的 Ti、T0 和左乘残差。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功。

    异常和副作用:
        断言或夹具写入失败时使测试失败；仅访问 pytest 临时目录。

    坐标系和外参约定:
        Ti 和 T0 均为 LiDAR-to-camera，验证 delta @ T0 == Ti。
    """
    config_path, scene, nominal, true_extrinsics = (
        _write_recorded_pose_fixture(tmp_path, fixed_t0=False))
    config = load_emd_training_config(config_path)
    dataset = DatasetEMDExtrinsicCalib(
        [scene], level="perturbed_t0", contract=config.contract, train=False)
    sample = dataset[0]

    assert config.contract.image_directory == "image"
    assert config.contract.point_cloud_directory == "mid360"
    assert torch.allclose(
        sample["cam2vel"], torch.tensor(true_extrinsics[0]).float())
    assert torch.allclose(
        sample["nominal_extrinsic"], torch.tensor(nominal).float())
    assert torch.allclose(
        sample["residual"] @ sample["nominal_extrinsic"],
        sample["cam2vel"], atol=1.0e-5)


def test_recorded_fixed_t0_online_is_deterministic_and_bounded(tmp_path):
    """验证真实固定 T0 场景按 seed 产生可复现且有界的初始扰动。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功。

    异常和副作用:
        断言或夹具写入失败时使测试失败；仅访问 pytest 临时目录。

    坐标系和外参约定:
        采样 delta 为目标相对输入的左乘残差，满足 delta @ T_input == Ti。
    """
    config_path, scene, _, _ = _write_recorded_pose_fixture(
        tmp_path, fixed_t0=True)
    config = load_emd_training_config(config_path)
    online = load_fixed_t0_online_perturbation_config(config_path)
    dataset = DatasetEMDFixedT0OnlineExtrinsicCalib(
        [scene], level="fixed_baseline", contract=config.contract,
        online_config=online, seed=42, train=False)
    first = dataset[0]
    repeated = dataset[0]

    assert torch.equal(first["sampled_residual"], repeated["sampled_residual"])
    assert torch.all(first["sampled_translation_xyz_m"].abs() <= 0.2)
    assert torch.all(first["sampled_euler_xyz_deg"].abs() <= 10.0)
    assert torch.allclose(
        first["residual"] @ first["nominal_extrinsic"],
        first["cam2vel"], atol=1.0e-5)


def test_recorded_pose_rejects_non_unit_quaternion(tmp_path):
    """验证真实域七列位姿会拒绝非单位 xyzw 四元数。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过预期异常表达成功。

    异常和副作用:
        未抛出预期 ValueError 时使测试失败；测试会改写临时 pose.txt。

    坐标系和外参约定:
        被测文件仍按 LiDAR-to-camera 解释，但四元数范数故意设为非法值。
    """
    config_path, scene, _, _ = _write_recorded_pose_fixture(
        tmp_path, fixed_t0=False)
    pose_rows = np.loadtxt(scene / "pose.txt")
    pose_rows[0, 3:] *= 2.0
    np.savetxt(scene / "pose.txt", pose_rows)
    config = load_emd_training_config(config_path)

    with pytest.raises(ValueError, match="四元数范数不是 1"):
        DatasetEMDExtrinsicCalib(
            [scene], level="perturbed_t0", contract=config.contract,
            train=False)


def test_recorded_pose_rejects_frame_count_mismatch(tmp_path):
    """验证真实域 pose.txt 行数必须与图像和点云帧数相同。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过预期异常表达成功。

    异常和副作用:
        未抛出预期 ValueError 时使测试失败；测试会缩短临时 pose.txt。

    坐标系和外参约定:
        保留的位姿仍为 LiDAR-to-camera，但故意缺少第二帧 Ti。
    """
    config_path, scene, _, _ = _write_recorded_pose_fixture(
        tmp_path, fixed_t0=False)
    pose_rows = np.loadtxt(scene / "pose.txt")
    np.savetxt(scene / "pose.txt", pose_rows[:1])
    config = load_emd_training_config(config_path)

    with pytest.raises(ValueError, match="图像和点云共有 2 帧"):
        DatasetEMDExtrinsicCalib(
            [scene], level="perturbed_t0", contract=config.contract,
            train=False)


def test_recorded_pose_reports_missing_configured_directory(tmp_path):
    """验证真实域 YAML 指定的图像或点云目录不存在时给出明确错误。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过预期异常表达成功。

    异常和副作用:
        未抛出预期 FileNotFoundError 时使测试失败；只改写临时 YAML。

    坐标系和外参约定:
        本用例在读取 Ti 前失败，不改变 LiDAR-to-camera 或左乘残差约定。
    """
    config_path, scene, _, _ = _write_recorded_pose_fixture(
        tmp_path, fixed_t0=False)
    raw = yaml.safe_load(config_path.read_text())
    raw["dataset"]["layout"]["point_clouds"] = "missing_mid360"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_emd_training_config(config_path)

    with pytest.raises(FileNotFoundError, match="missing_mid360/"):
        DatasetEMDExtrinsicCalib(
            [scene], level="perturbed_t0", contract=config.contract,
            train=False)


def test_training_yaml_strategies_are_parseable():
    """验证训练 YAML 可直接作为对应真实场景的评估配置。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功。

    异常和副作用:
        配置语义或路径解析变化时使测试失败；只读仓库 YAML，不读取真实帧。

    坐标系和外参约定:
        固定 T0 配置要求 Ti=T0；逐帧 Ti 配置读取每帧 Ti 与存储残差。
    """
    config_root = Path(__file__).resolve().parents[1] / "config"
    fixed = load_emd_training_config(
        config_root / "train_mid360_single_extrinsic.yaml")
    varying = load_emd_training_config(
        config_root / "train_mid360_multi_extrinsic_online.yaml")

    assert fixed.schema_version == "cmrnext_scene_calibration_v1"
    assert varying.schema_version == "cmrnext_scene_calibration_v1"
    assert fixed.contract.source_mode == "calibnet_fixed_t0_v1"
    assert varying.contract.source_mode == "calibnet_stored_residuals_v1"
    assert fixed.train_scene_paths
    assert varying.train_scene_paths


def test_emd_dataset_uses_t0_for_input_and_ti_for_target(tmp_path):
    """验证回归场景 `test_emd_dataset_uses_t0_for_input_and_ti_for_target`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, nominal, residuals, true_extrinsics = _write_fixture(tmp_path)
    config = load_emd_training_config(config_path)
    scene = root / "large" / "train" / "scene_a"
    dataset = DatasetEMDExtrinsicCalib(
        [scene], level="large", contract=config.contract, train=False)

    sample = dataset[0]
    internal = tvector2mat(sample["tr_error"]) @ quat2mat(sample["rot_error"])
    raw = np.fromfile(
        root / "large" / "train" / "scene_a" / "velodyne" / "000000.bin",
        dtype=np.float32).reshape(-1, 4)
    raw[:, 3] = 1.0
    raw = torch.from_numpy(raw).t()

    assert len(dataset) == 2
    assert torch.allclose(sample["cam2vel"], torch.tensor(true_extrinsics[0]).float())
    assert torch.allclose(sample["residual"], torch.tensor(residuals[0]).float())
    assert torch.allclose(internal, torch.linalg.inv(sample["residual"]), atol=1e-6)
    assert torch.allclose(
        internal @ sample["point_cloud"],
        torch.tensor(nominal).float() @ raw,
        atol=1e-5,
    )


def test_emd_dataset_direct_scene_ignores_split_and_manifest(tmp_path):
    """验证回归场景 `test_emd_dataset_direct_scene_ignores_split_and_manifest`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, _, _, _ = _write_fixture(tmp_path)
    scene = root / "large" / "train" / "scene_a"
    (root / "task_manifest.json").unlink()
    (scene / "manifest.json").unlink()
    (scene / "metadata.json").unlink()

    config = load_emd_training_config(config_path)
    dataset = DatasetEMDExtrinsicCalib(
        [scene],
        level="large",
        contract=config.contract,
        train=False,
    )

    assert len(dataset) == 2
    assert dataset.scene_paths == (str(scene.resolve()),)


def test_dynamic_nominal_preserves_true_extrinsic_and_stage_bounds(tmp_path):
    """验证回归场景 `test_dynamic_nominal_preserves_true_extrinsic_and_stage_bounds`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, nominal, residuals, true_extrinsics = _write_fixture(
        tmp_path)
    config = load_emd_training_config(config_path)
    dynamic_config = EMDDynamicNominalConfig(
        enabled=True,
        levels={
            "large": DynamicNominalLevelConfig(
                translation_alpha_t=0.05,
                translation_min_floor_m=0.001,
                max_rotation_deg=1.0,
            ),
        },
    )
    scene = root / "large" / "train" / "scene_a"
    dataset = DatasetEMDDynamicNominalExtrinsicCalib(
        [scene], level="large", contract=config.contract,
        dynamic_config=dynamic_config, train=True)

    np.random.seed(7)
    sample = dataset[0]
    internal = tvector2mat(sample["tr_error"]) @ quat2mat(sample["rot_error"])
    raw = np.fromfile(
        root / "large" / "train" / "scene_a" / "velodyne" / "000000.bin",
        dtype=np.float32).reshape(-1, 4)
    raw[:, 3] = 1.0
    raw = torch.from_numpy(raw).t()

    assert sample["dataset_mode"] == "emd_dynamic_nominal"
    assert not torch.allclose(
        sample["nominal_jitter"], torch.eye(4))
    assert not torch.allclose(
        sample["nominal_extrinsic"], torch.tensor(nominal).float())
    assert torch.allclose(
        sample["original_residual"], torch.tensor(residuals[0]).float())
    assert torch.allclose(
        sample["residual"] @ sample["nominal_extrinsic"],
        torch.tensor(true_extrinsics[0]).float(),
        atol=1e-5,
    )
    assert torch.allclose(
        internal @ sample["point_cloud"],
        sample["nominal_extrinsic"] @ raw,
        atol=1e-5,
    )


def test_per_ti_online_ignores_stored_t0_and_residual(tmp_path):
    """验证逐帧 Ti 在线扰动与场景存储 T0 和 residual 完全无关。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；固定相同随机种子并修改存储外参后，在线输入、监督和
        目标点云完全相同即通过。

    异常和副作用:
        断言不满足时使测试失败；测试只修改临时夹具加载后的内存张量。

    坐标系和外参约定:
        每帧以 Ti 为目标外参，在线采样 delta 后构造
        T_input = inverse(delta) @ Ti，并验证 delta @ T_input == Ti。
    """
    root, config_path, _, _, true_extrinsics = _write_fixture(
        tmp_path, level="perturbed_t0", target_network="calibnet")
    config_raw = yaml.safe_load(config_path.read_text())
    config_raw["dataset"]["source_mode"] = "calibnet_stored_residuals_v1"
    config_raw["curriculum"] = {"iter1": "perturbed_t0"}
    config_raw["per_ti_online_perturbation"] = {
        "enabled": True,
        "sampler": "cmrnext_legacy_per_axis_box_v1",
        "max_translation_component_m": 0.08,
        "max_rotation_euler_component_deg": 1.5,
        "train_resample_each_access": True,
    }
    config_path.write_text(yaml.safe_dump(config_raw))

    config = load_emd_training_config(config_path)
    online = load_per_ti_online_perturbation_config(config_path)
    scene = root / "perturbed_t0" / "train" / "scene_a"
    dataset = DatasetEMDPerTiOnlineExtrinsicCalib(
        [scene], level="perturbed_t0", contract=config.contract,
        online_config=online, seed=42, train=True)

    np.random.seed(7)
    original = dataset[0]
    resampled = dataset[0]
    assert not torch.equal(
        original["sampled_residual"], resampled["sampled_residual"])
    # Change stored T0 and residual after loading while preserving Ti.
    dataset.sequences[0].nominal_extrinsic.copy_(
        torch.tensor([
            [1.0, 0.0, 0.0, 9.0],
            [0.0, 1.0, 0.0, -8.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32))
    dataset.sequences[0].residuals.copy_(
        torch.eye(4).expand_as(dataset.sequences[0].residuals))
    np.random.seed(7)
    changed_stored_values = dataset[0]

    assert online.max_translation_component_m == 0.08
    assert online.max_rotation_euler_component_deg == 1.5
    assert original["dataset_mode"] == "emd_per_ti_online_per_axis"
    assert torch.all(original["sampled_translation_xyz_m"].abs() <= 0.08)
    assert torch.all(original["sampled_euler_xyz_deg"].abs() <= 1.5)
    assert "source_t0_extrinsic" not in original
    assert torch.equal(original["residual"], original["sampled_residual"])
    assert torch.equal(
        original["sampled_residual"], changed_stored_values["sampled_residual"])
    assert torch.equal(
        original["nominal_extrinsic"],
        changed_stored_values["nominal_extrinsic"])
    assert torch.equal(
        original["point_cloud"], changed_stored_values["point_cloud"])
    assert torch.allclose(
        original["residual"] @ original["nominal_extrinsic"],
        torch.tensor(true_extrinsics[0]).float(), atol=1e-5)
    internal = tvector2mat(original["tr_error"]) @ quat2mat(
        original["rot_error"])
    raw = np.fromfile(
        scene / "velodyne" / "000000.bin", dtype=np.float32).reshape(-1, 4)
    raw[:, 3] = 1.0
    raw = torch.from_numpy(raw).t()
    assert torch.allclose(
        internal @ original["point_cloud"],
        original["nominal_extrinsic"] @ raw,
        atol=1e-5)


def test_dynamic_nominal_is_disabled_in_original_config(tmp_path):
    """验证回归场景 `test_dynamic_nominal_is_disabled_in_original_config`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    _, config_path, _, _, _ = _write_fixture(tmp_path)
    dynamic_config = load_emd_dynamic_nominal_config(config_path)

    assert not dynamic_config.enabled
    assert dynamic_config.levels == {}


def test_emd_dataset_collates_variable_point_counts(tmp_path):
    """验证回归场景 `test_emd_dataset_collates_variable_point_counts`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, _, _, _ = _write_fixture(tmp_path)
    config = load_emd_training_config(config_path)
    scene = root / "large" / "train" / "scene_a"
    dataset = DatasetEMDExtrinsicCalib(
        [scene], level="large", contract=config.contract, train=False)

    batch = merge_inputs([dataset[0], dataset[1]])

    assert [cloud.shape[1] for cloud in batch["point_cloud"]] == [2, 3]
    assert batch["residual"].shape == (2, 4, 4)
    assert batch["nominal_extrinsic"].shape == (2, 4, 4)


def test_emd_dataset_ignores_metadata_semantics(tmp_path):
    """验证回归场景 `test_emd_dataset_ignores_metadata_semantics`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, _, _, _ = _write_fixture(
        tmp_path, {"training_semantics": "legacy_online_perturbation"})
    config = load_emd_training_config(config_path)

    scene = root / "large" / "train" / "scene_a"
    dataset = DatasetEMDExtrinsicCalib(
        [scene], level="large", contract=config.contract, train=False)

    assert len(dataset) == 2


def test_emd_dataset_ignores_unused_intensity(tmp_path):
    """验证回归场景 `test_emd_dataset_ignores_unused_intensity`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, _, _, _ = _write_fixture(tmp_path)
    cloud_path = root / "large" / "train" / "scene_a" / "velodyne" / "000000.bin"
    points = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
    points[0, 3] = 1.0
    points.tofile(cloud_path)
    config = load_emd_training_config(config_path)
    scene = root / "large" / "train" / "scene_a"
    dataset = DatasetEMDExtrinsicCalib(
        [scene], level="large", contract=config.contract, train=False)

    assert torch.all(dataset[0]["point_cloud"][3] == 1.0)


def test_fixed_t0_online_uses_per_axis_bounds_and_cmrnext_geometry(tmp_path):
    """验证回归场景 `test_fixed_t0_online_uses_per_axis_bounds_and_cmrnext_geometry`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, nominal, _, true_extrinsics = _write_fixture(
        tmp_path,
        level="fixed_baseline",
        target_network="calibnet",
        fixed_identity=True,
    )
    config = load_emd_training_config(config_path)
    online = load_fixed_t0_online_perturbation_config(config_path)
    scene = root / "fixed_baseline" / "train" / "scene_a"
    dataset = DatasetEMDFixedT0OnlineExtrinsicCalib(
        [scene], level="fixed_baseline", contract=config.contract,
        online_config=online, seed=42, train=True)

    np.random.seed(7)
    sample = dataset[0]
    internal = tvector2mat(sample["tr_error"]) @ quat2mat(sample["rot_error"])
    raw = np.fromfile(
        root / "fixed_baseline" / "train" / "scene_a"
        / "velodyne" / "000000.bin",
        dtype=np.float32,
    ).reshape(-1, 4)
    raw[:, 3] = 1.0
    raw = torch.from_numpy(raw).t()

    assert config.contract.source_mode == "calibnet_fixed_t0_v1"
    assert sample["dataset_mode"] == "emd_fixed_t0_online_per_axis"
    assert torch.all(sample["sampled_translation_xyz_m"].abs() <= 0.6)
    assert torch.all(sample["sampled_euler_xyz_deg"].abs() <= 10.0)
    assert torch.allclose(
        sample["source_t0_extrinsic"], torch.tensor(nominal).float())
    assert torch.allclose(
        sample["cam2vel"], torch.tensor(true_extrinsics[0]).float())
    assert torch.allclose(
        sample["residual"] @ sample["nominal_extrinsic"],
        sample["cam2vel"], atol=1e-5)
    assert torch.allclose(
        internal @ sample["point_cloud"],
        sample["nominal_extrinsic"] @ raw,
        atol=1e-5,
    )


def test_fixed_t0_train_resamples_and_validation_is_deterministic(tmp_path):
    """验证回归场景 `test_fixed_t0_train_resamples_and_validation_is_deterministic`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    train_root, train_config_path, _, _, _ = _write_fixture(
        tmp_path / "train_fixture",
        level="fixed_baseline",
        target_network="calibnet",
        fixed_identity=True,
    )
    train_config = load_emd_training_config(train_config_path)
    online = load_fixed_t0_online_perturbation_config(train_config_path)
    train_scene = train_root / "fixed_baseline" / "train" / "scene_a"
    train_dataset = DatasetEMDFixedT0OnlineExtrinsicCalib(
        [train_scene], level="fixed_baseline", contract=train_config.contract,
        online_config=online, seed=42, train=True)
    first_train = train_dataset[0]
    second_train = train_dataset[0]
    assert not torch.equal(
        first_train["sampled_residual"], second_train["sampled_residual"])

    val_root, val_config_path, _, _, _ = _write_fixture(
        tmp_path / "val_fixture",
        level="fixed_baseline",
        split="val",
        target_network="calibnet",
        fixed_identity=True,
    )
    val_config = load_emd_training_config(val_config_path)
    val_online = load_fixed_t0_online_perturbation_config(val_config_path)
    val_scene = val_root / "fixed_baseline" / "val" / "scene_a"
    val_dataset = DatasetEMDFixedT0OnlineExtrinsicCalib(
        [val_scene], level="fixed_baseline", contract=val_config.contract,
        online_config=val_online, seed=42, train=False)
    first_val = val_dataset[0]
    second_val = val_dataset[0]
    assert torch.equal(
        first_val["sampled_residual"], second_val["sampled_residual"])
    batch = merge_inputs([first_val, val_dataset[1]])
    assert batch["residual"].shape == (2, 4, 4)


def test_calibnet_source_requires_explicit_fixed_t0_mode(tmp_path):
    """验证回归场景 `test_calibnet_source_requires_explicit_fixed_t0_mode`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    _, config_path, _, _, _ = _write_fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["dataset"]["target_network"] = "calibnet"
    config_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="要求 target_network 为 cmrnext"):
        load_emd_training_config(config_path)


def test_fixed_t0_evaluation_distribution_reports_declared_percentiles():
    """验证回归场景 `test_fixed_t0_evaluation_distribution_reports_declared_percentiles`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    summary = _distribution_summary(range(1, 11))

    assert summary == {
        "count": 10,
        "mean": 5.5,
        "median": 5.0,
        "p80": 8.0,
        "p90": 9.0,
        "p95": 10.0,
        "p99": 10.0,
        "max": 10.0,
    }


def test_fixed_t0_residual_components_use_xyz_euler_convention():
    """验证回归场景 `test_fixed_t0_residual_components_use_xyz_euler_convention`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    roll, pitch, yaw = map(math.radians, (3.0, -4.0, 5.0))
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor([
        [cy * cz, -cy * sz, sy],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
    ])
    transform[:3, 3] = torch.tensor([0.12, -0.23, 0.34])

    components = _residual_components(transform)

    assert components["translation_xyz_m"] == pytest.approx(
        [0.12, -0.23, 0.34], abs=1e-6)
    assert components["rotation_euler_xyz_deg"] == pytest.approx(
        [3.0, -4.0, 5.0], abs=1e-5)
    assert components["translation_max_abs_component_m"] == pytest.approx(0.34)
    assert components["rotation_max_abs_euler_component_deg"] == pytest.approx(5.0)


def test_stage_range_p90_uses_each_frames_largest_absolute_component():
    """验证回归场景 `test_stage_range_p90_uses_each_frames_largest_absolute_component`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    records = []
    for index in range(10):
        translation = (0.01 * index, 0.20 if index == 0 else 0.0, 0.0)
        rotation = (0.1 * index, 2.0 if index == 0 else 0.0, 0.0)
        components = {
            "translation_xyz_m": translation,
            "rotation_euler_xyz_deg": rotation,
            "translation_max_abs_component_m": max(map(abs, translation)),
            "rotation_max_abs_euler_component_deg": max(map(abs, rotation)),
        }
        records.append({"iter1": {"residual_components": components}})

    summary = _residual_component_summary(records, "iter1")

    translation = summary["translation_m"]["max_abs_component_per_frame"]
    rotation = summary["rotation_euler_deg"]["max_abs_component_per_frame"]
    assert translation["p90"] == pytest.approx(0.09)
    assert rotation["p90"] == pytest.approx(0.9)
    assert translation["max"] == pytest.approx(0.20)
    assert rotation["max"] == pytest.approx(2.0)


def test_worst_residual_selection_is_ranked_independently_per_scene():
    """验证回归场景 `test_worst_residual_selection_is_ranked_independently_per_scene`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    records = []
    for scene in ("scene_a", "scene_b"):
        for index, (translation_cm, rotation_deg) in enumerate(
                ((10.0, 0.0), (20.0, 0.0), (0.0, 3.0))):
            records.append({
                "sequence": scene,
                "batch_idx": len(records),
                "effective_max_t_m": 1.0,
                "effective_max_r_deg": 10.0,
                "iter1": {
                    "t_cm": translation_cm,
                    "r_deg": rotation_deg,
                },
            })

    selected = _select_emd_worst_per_scene(records, "iter1", 2)

    assert [(item["scene"], item["rank"]) for item in selected] == [
        ("scene_a", 1), ("scene_a", 2),
        ("scene_b", 1), ("scene_b", 2),
    ]
    assert [item["record"]["iter1"]["r_deg"] for item in selected] == [
        3.0, 0.0, 3.0, 0.0,
    ]
    assert [item["score"] for item in selected] == pytest.approx([
        0.3, 0.2, 0.3, 0.2,
    ])


def test_absolute_estimate_is_recovered_from_the_same_left_residual():
    """验证回归场景 `test_absolute_estimate_is_recovered_from_the_same_left_residual`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    angle = torch.tensor(math.radians(31.0))
    true_extrinsic = torch.eye(4)
    true_extrinsic[:3, :3] = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    true_extrinsic[:3, 3] = torch.tensor([1.1, -0.7, 0.4])

    estimate = torch.eye(4)
    estimate[:3, 3] = torch.tensor([-0.2, 0.3, 1.4])
    left_residual = true_extrinsic @ torch.linalg.inv(estimate)

    recovered = _estimate_from_left_residual(
        true_extrinsic, left_residual)

    assert torch.allclose(recovered, estimate, atol=1e-6)
    assert torch.allclose(left_residual @ recovered, true_extrinsic, atol=1e-6)


def test_reprojection_triptych_is_vertical_and_annotated(tmp_path):
    """验证回归场景 `test_reprojection_triptych_is_vertical_and_annotated`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from PIL import Image

    rgb = torch.full((3, 40, 120), 0.25)
    depths = []
    for row in (8, 20, 32):
        depth = torch.zeros((1, 40, 120))
        depth[:, row:row + 2, 10:110] = torch.linspace(1.0, 30.0, 100)
        depths.append(depth)
    labels = ["Ti true", "T0 xyz error", "Stage-1 xyz error"]
    figure = _draw_reprojection_panels(
        plt, list(zip(labels, depths)), rgb, overlay_text=True)
    output = tmp_path / "triptych.png"
    figure.savefig(output, dpi=40)
    plt.close(figure)

    with Image.open(output) as image:
        assert image.height > image.width
        pixels = np.asarray(image.convert("RGB"))
    assert pixels.std() > 1.0
    assert [axis.texts[0].get_text() for axis in figure.axes] == labels


def test_fixed_t0_curriculum_accepts_contiguous_second_stage(tmp_path):
    """验证回归场景 `test_fixed_t0_curriculum_accepts_contiguous_second_stage`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    _, config_path, _, _, _ = _write_fixture(
        tmp_path,
        level="fixed_baseline",
        target_network="calibnet",
        fixed_identity=True,
    )
    raw = yaml.safe_load(config_path.read_text())
    raw["curriculum"] = {
        "iter1": "fixed_baseline",
        "iter2": "fixed_baseline",
    }
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_emd_training_config(config_path)

    assert config.curriculum == raw["curriculum"]

    raw["curriculum"] = {
        "iter1": "fixed_baseline",
        "iter3": "fixed_baseline",
    }
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="必须从 iter1 连续定义"):
        load_emd_training_config(config_path)


def test_calibnet_stored_residuals_support_custom_dynamic_level(tmp_path):
    """验证回归场景 `test_calibnet_stored_residuals_support_custom_dynamic_level`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, _, _, _ = _write_fixture(
        tmp_path, level="perturbed_t0", target_network="calibnet")

    config_raw = yaml.safe_load(config_path.read_text())
    config_raw["dataset"]["source_mode"] = "calibnet_stored_residuals_v1"
    config_raw["curriculum"] = {"iter1": "perturbed_t0"}
    config_raw["dynamic_nominal_augmentation"] = {
        "enabled": True,
        "levels": {
            "perturbed_t0": {
                "translation_alpha_t": 0.05,
                "translation_min_floor_m": 0.001,
                "max_rotation_deg": 1.0,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config_raw))

    task_manifest_path = root / "task_manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text())
    task_manifest["required_levels"] = ["perturbed_t0"]
    task_manifest_path.write_text(json.dumps(task_manifest))

    config = load_emd_training_config(config_path)
    dynamic = load_emd_dynamic_nominal_config(config_path)
    scene = root / "perturbed_t0" / "train" / "scene_a"
    dataset = DatasetEMDDynamicNominalExtrinsicCalib(
        [scene], level="perturbed_t0", contract=config.contract,
        dynamic_config=dynamic, train=True)

    assert config.contract.source_mode == "calibnet_stored_residuals_v1"
    assert set(dynamic.levels) == {"perturbed_t0"}
    assert dataset[0]["dataset_mode"] == "emd_dynamic_nominal"


def test_emd_dataset_preserves_camera_skew(tmp_path):
    """验证回归场景 `test_emd_dataset_preserves_camera_skew`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    root, config_path, _, _, _ = _write_fixture(tmp_path)
    skew = -0.085165

    config_raw = yaml.safe_load(config_path.read_text())
    config_raw["dataset"]["sensor"]["camera"]["skew"] = skew
    config_path.write_text(yaml.safe_dump(config_raw))

    sequence = root / "large" / "train" / "scene_a"
    metadata_path = sequence / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sensor_setup"]["camera"]["skew"] = skew
    metadata_path.write_text(json.dumps(metadata))
    (sequence / "calib.txt").write_text(
        f"P2: 707.0912 {skew} 601.8873 0 "
        "0 707.0912 183.1104 0 0 0 1 0\n")

    config = load_emd_training_config(config_path)
    scene = root / "large" / "train" / "scene_a"
    dataset = DatasetEMDExtrinsicCalib(
        [scene], level="large", contract=config.contract, train=False)

    assert config.contract.camera_skew == pytest.approx(skew)
    assert dataset[0]["calib"].tolist() == pytest.approx(
        [707.0912, 707.0912, 601.8873, 183.1104, skew])


def test_emd_config_loads_optional_crop_ranges(tmp_path):
    """验证回归场景 `test_emd_config_loads_optional_crop_ranges`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    _, config_path, _, _, _ = _write_fixture(tmp_path)
    config_raw = yaml.safe_load(config_path.read_text())
    config_raw["augmentation"]["crop_top"] = [921, 976]
    config_raw["augmentation"]["crop_left"] = [358, 640]
    config_path.write_text(yaml.safe_dump(config_raw))

    config = load_emd_training_config(config_path)

    assert config.contract.crop_top_range == (921, 976)
    assert config.contract.crop_left_range == (358, 640)


def test_flow_epe_report_maps_values_to_checkpoint_stage():
    """验证回归场景 `test_flow_epe_report_maps_values_to_checkpoint_stage`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    report = _build_flow_epe_report(
        [[1.0, 3.0], [0.5, 1.5]], [1, 2])

    assert report["units"] == "pixels"
    assert report["iterations"]["iter1"]["stage"] == 1
    assert report["iterations"]["iter1"]["mean"] == pytest.approx(2.0)
    assert report["iterations"]["iter2"]["stage"] == 2
    assert report["iterations"]["iter2"]["median"] == pytest.approx(0.5)


def test_emd_evaluation_uses_checkpoint_training_shape():
    """验证回归场景 `test_emd_evaluation_uses_checkpoint_training_shape`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    checkpoint = {"config": {"img_shape": [320, 960]}}

    assert _resolve_evaluation_image_shape(checkpoint, "emd") == (320, 960)


def test_emd_evaluation_prefers_fixed_checkpoint_crop_shape():
    """验证新 checkpoint 的固定评估尺寸覆盖训练随机 crop 尺寸。

    参数:
        无。

    返回值:
        无显式返回值；解析结果为固定评估 H×W 即通过。

    异常和副作用:
        断言失败时使测试失败；不读取真实数据或执行 GPU 推理。

    坐标系和外参约定:
        本测试只核对像素画布尺寸，不改变 Ti @ inverse(T0) 外参约定。
    """
    checkpoint = {"config": {
        "img_shape": [512, 640],
        "emd_evaluation_crop": {
            "top": 0, "left": 0, "height": 640, "width": 768,
        },
    }}

    assert _resolve_evaluation_image_shape(
        checkpoint, "emd") == (640, 768)


def test_legacy_checkpoint_training_domain_keeps_missing_emd_fields():
    """验证旧版兼容权重缺少 EMD 标签时仍可形成准确的审计记录。

    参数:
        无。

    返回值:
        无显式返回值；通过断言确认缺失训练域字段保持为 None。

    异常和副作用:
        断言失败时使测试失败；不加载真实 checkpoint，不写文件，也不执行 GPU
        推理。

    坐标系和外参约定:
        该记录只描述训练来源，不改变 EMD 的 T0、Ti 或
        delta_Ti = Ti @ inverse(T0) 几何关系。
    """
    checkpoint = {
        "config": {
            "img_shape": [320, 960],
            "max_depth": 160.0,
        },
        "state_dict": {},
    }

    record = _checkpoint_training_domain_record(
        checkpoint, "/checkpoints/legacy.tar", stage=1)

    assert record == {
        "stage": 1,
        "checkpoint": "/checkpoints/legacy.tar",
        "emd_only": None,
        "emd_level": None,
        "emd_dynamic_nominal": None,
        "emd_fixed_t0_online": None,
        "training_img_shape": [320, 960],
        "emd_source_image_shape": None,
        "emd_resize_shape": None,
        "emd_evaluation_crop": None,
        "flow_epe_units": "pixels",
        "fixed_t0_max_translation_component_m": None,
        "fixed_t0_max_rotation_euler_component_deg": None,
    }


def test_emd_checkpoint_training_domain_preserves_declared_values():
    """验证 EMD checkpoint 已声明的训练域字段原样进入评估审计记录。

    参数:
        无。

    返回值:
        无显式返回值；通过断言确认布尔值、阶段和扰动单位均未被改写。

    异常和副作用:
        断言失败时使测试失败；不加载真实 checkpoint，不写文件，也不执行 GPU
        推理。

    坐标系和外参约定:
        平移扰动单位为米，旋转扰动单位为度；记录过程不参与外参计算。
    """
    checkpoint = {"config": {
        "emd_only": True,
        "emd_level": "fixed_baseline",
        "emd_dynamic_nominal": False,
        "emd_fixed_t0_online": True,
        "img_shape": [512, 640],
        "emd_source_image_shape": [1296, 1600],
        "emd_resize_shape": [648, 800],
        "emd_evaluation_crop": {
            "top": 0, "left": 0, "height": 640, "width": 768,
        },
        "emd_fixed_t0_online_max_translation_component_m": 0.2,
        "emd_fixed_t0_online_max_rotation_euler_component_deg": 10.0,
    }}

    record = _checkpoint_training_domain_record(
        checkpoint, "/checkpoints/emd.tar", stage=2)

    assert record["stage"] == 2
    assert record["emd_only"] is True
    assert record["emd_dynamic_nominal"] is False
    assert record["emd_fixed_t0_online"] is True
    assert record["training_img_shape"] == [512, 640]
    assert record["emd_resize_shape"] == [648, 800]
    assert record["emd_evaluation_crop"]["width"] == 768
    assert record["flow_epe_units"] == "resized_pixels"
    assert record["fixed_t0_max_translation_component_m"] == 0.2
    assert record["fixed_t0_max_rotation_euler_component_deg"] == 10.0


def test_emd_online_resize_scales_rgb_and_intrinsics(tmp_path):
    """验证在线缩放 RGB 时针孔内参按相同横纵比例缩放。

    参数:
        tmp_path: pytest 提供的临时目录，用于生成最小 EMD 场景。

    返回值:
        无显式返回值；RGB、内参和审计尺寸满足半分辨率约定即通过。

    异常和副作用:
        在临时目录创建测试图片、点云和 YAML；断言失败时使测试失败。

    坐标系和外参约定:
        三维点保持目标相机坐标不变，fx、fy、cx、cy 和 skew 按像素倍率缩放。
    """
    root, config_path, _, _, _ = _write_fixture(tmp_path)
    config_raw = yaml.safe_load(config_path.read_text())
    config_raw["dataset"]["sensor"]["camera"]["skew"] = -0.08
    config_raw["preprocessing"] = {
        "resize_hw": [185, 613],
        "evaluation_crop": {
            "top": 0, "left": 0, "height": 128, "width": 512,
        },
    }
    config_path.write_text(yaml.safe_dump(config_raw))
    sequence = root / "large" / "train" / "scene_a"
    (sequence / "calib.txt").write_text(
        "P2: 707.0912 -0.08 601.8873 0 "
        "0 707.0912 183.1104 0 0 0 1 0\n")

    config = load_emd_training_config(config_path)
    dataset = DatasetEMDExtrinsicCalib(
        [sequence], level="large", contract=config.contract, train=False)
    sample = dataset[0]

    assert sample["rgb"].shape == (185, 613, 3)
    assert sample["calib"].tolist() == pytest.approx([
        353.5456, 353.5456, 300.94365, 91.5552, -0.04,
    ])
    assert sample["source_image_shape"].tolist() == [370, 1226]
    assert sample["resized_image_shape"].tolist() == [185, 613]
    assert sample["image_scale_xy"].tolist() == pytest.approx([0.5, 0.5])
    assert config.contract.evaluation_crop_height == 128
    assert config.contract.evaluation_crop_width == 512


def test_v2_config_resolves_scene_paths_and_mid360_audit(tmp_path):
    """验证 v2 场景相对路径与 MID360 累计参数进入审计记录。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；路径、累计扫描数和发射束数满足配置即通过。

    异常和副作用:
        在临时目录创建 YAML；断言失败时使测试失败。

    坐标系和外参约定:
        配置解析不读取外参内容；后续仍使用 delta_Ti = Ti @ inverse(T0)。
    """
    _, config_path, _, _, _ = _write_fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["dataset"]["sensor"]["lidar"]["acquisition"] = {
        "repeated_pattern": False,
        "scan_frequency_hz": 10.0,
        "beams_per_scan": 20000,
        "beam_rate_hz": 200000.0,
        "accumulation_duration_s": 0.5,
    }
    raw["dataset"]["scenes"] = {
        "train": ["large/train/scene_a"],
        "validation": [],
        "test": ["large/test/scene_b"],
    }
    config_path.write_text(yaml.safe_dump(raw))

    config = load_emd_training_config(config_path)
    audit = config.audit_record()

    assert config.train_scene_paths == (
        (tmp_path / "emd/large/train/scene_a").resolve(),)
    assert config.test_scene_paths == (
        (tmp_path / "emd/large/test/scene_b").resolve(),)
    assert config.contract.lidar_accumulated_scan_equivalents == 5.0
    assert config.contract.lidar_accumulated_beam_count == 100000
    assert audit["lidar_accumulation_duration_s"] == 0.5


def test_v2_configurable_extrinsic_filenames_are_used(tmp_path):
    """验证 v2 自定义标定、T0、Ti 和 residual 文件名真实参与加载。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；使用改名后的四个文件成功读取样本即通过。

    异常和副作用:
        在临时场景内重命名测试文件并更新 YAML；断言失败时使测试失败。

    坐标系和外参约定:
        改名不改变矩阵内容，仍要求 residual @ T0 == Ti。
    """
    root, config_path, _, _, _ = _write_fixture(tmp_path)
    scene = root / "large/train/scene_a"
    renamed_files = {
        "calibration": ("calib.txt", "camera_intrinsics.txt"),
        "nominal_extrinsic": ("nominal_extrinsic.txt", "fixed_t0.txt"),
        "true_extrinsics": ("poses.txt", "frame_ti.txt"),
        "residuals": ("residuals.txt", "frame_delta.txt"),
    }
    for source_name, target_name in renamed_files.values():
        (scene / source_name).rename(scene / target_name)

    raw = yaml.safe_load(config_path.read_text())
    raw["dataset"]["files"] = {
        key: target_name
        for key, (_, target_name) in renamed_files.items()
    }
    config_path.write_text(yaml.safe_dump(raw))

    config = load_emd_training_config(config_path)
    dataset = DatasetEMDExtrinsicCalib(
        [scene], level="large", contract=config.contract, train=False)

    assert len(dataset) == 2
    assert config.contract.nominal_extrinsic_file == "fixed_t0.txt"


def test_v1_config_is_rejected_after_v2_migration(tmp_path):
    """验证旧 v1 配置不会被静默按 v2 语义解释。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；加载器明确拒绝 v1 即通过。

    异常和副作用:
        在临时目录修改 YAML schema 字段；预期 ValueError 被测试捕获。

    坐标系和外参约定:
        schema 拒绝发生在任何 T0、Ti 或 residual 文件读取之前。
    """
    _, config_path, _, _, _ = _write_fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["schema_version"] = "cmrnext_emd_training_v1"
    config_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="cmrnext_emd_training_v2"):
        load_emd_training_config(config_path)

def test_cropped_flow_mask_excludes_targets_outside_view():
    """验证裁剪后目标终点越界的 LiDAR 光流监督被排除。

    参数:
        无。

    返回值:
        无显式返回值；越界点的 flow/mask 清零且内部点保留即通过。

    异常和副作用:
        断言失败时使测试失败；只创建内存张量。

    坐标系和外参约定:
        目标像素等于 source_uv + flow_gt，光流单位为裁剪图像像素。
    """
    flow = torch.zeros((4, 5, 2), dtype=torch.float32)
    mask = torch.zeros((4, 5), dtype=torch.int32)
    mask[1, 1] = 1
    mask[2, 2] = 1
    flow[1, 1, 0] = -2.0
    flow[2, 2] = torch.tensor([1.0, 1.0])

    filtered_flow, filtered_mask = _mask_evaluation_flow_endpoints(
        flow, mask)

    assert filtered_mask[1, 1].item() == 0
    assert filtered_flow[1, 1].tolist() == [0.0, 0.0]
    assert filtered_mask[2, 2].item() == 1
    assert filtered_flow[2, 2].tolist() == [1.0, 1.0]


def test_emd_crop_moves_uv_and_keeps_3d_correspondences_aligned():
    """验证回归场景 `test_emd_crop_moves_uv_and_keeps_3d_correspondences_aligned`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    rgb = torch.arange(3 * 6 * 8).reshape(3, 6, 8)
    depth = torch.arange(6 * 8).reshape(1, 6, 8)
    flow = torch.zeros((6, 8, 2))
    mask = torch.ones((6, 8), dtype=torch.int32)
    uv = torch.tensor([[2, 1], [4, 2], [6, 4], [7, 5]])
    points = torch.tensor([
        [10.0, 0.0, 1.0, 1.0],
        [20.0, 0.0, 1.0, 1.0],
        [30.0, 0.0, 1.0, 1.0],
        [40.0, 0.0, 1.0, 1.0],
    ])

    cropped = _crop_evaluation_frame(
        rgb, depth, flow, mask, uv, points, (3, 4), crop_top=2,
        crop_left=3)

    cropped_rgb, cropped_depth, cropped_flow, cropped_mask = cropped[:4]
    cropped_uv, cropped_points = cropped[4:]
    assert cropped_rgb.shape == (3, 3, 4)
    assert cropped_depth.shape == (1, 3, 4)
    assert cropped_flow.shape == (3, 4, 2)
    assert cropped_mask.shape == (3, 4)
    assert cropped_uv.tolist() == [[1, 0], [3, 2]]
    assert cropped_points[:, 0].tolist() == [20.0, 30.0]


def test_emd_crop_sampling_and_pnp_intrinsics_use_crop_offsets(monkeypatch):
    """验证回归场景 `test_emd_crop_sampling_and_pnp_intrinsics_use_crop_offsets`，确保相关数据、几何或入口行为保持预期。

    参数:
        monkeypatch: pytest 提供的属性、环境变量和函数替换工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    draws = iter([950, 500])
    monkeypatch.setattr(random, "randint", lambda lower, upper: next(draws))
    offsets = _sample_emd_crop_offsets(
        {
            "emd_crop_top_range": (921, 976),
            "emd_crop_left_range": (358, 640),
        },
        (1296, 1600),
        (320, 960),
    )

    class Camera:
        """封装 tests/test_emd_dataset.py 中的 `Camera` 状态和操作。

        主要职责:
            封装 tests/test_emd_dataset.py 中的 `Camera` 状态和操作，并集中维护该职责所需状态。

        返回值:
            类实例由构造方法创建；各公共方法的返回值见对应方法说明。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。
        """

        @staticmethod
        def get_matrix():
            """把当前相机内参组装为三乘三针孔相机矩阵。

            参数:
                无。

            返回值:
                NumPy 三乘三相机内参矩阵 K。

            异常和副作用:
                输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。
            """
            return np.array([
                [731.0, -0.08, 812.0],
                [0.0, 731.0, 653.0],
                [0.0, 0.0, 1.0],
            ])

    cropped_matrix = _camera_matrix_for_crop(Camera(), *offsets)
    assert offsets == (950, 500)
    assert cropped_matrix[0, 2] == pytest.approx(312.0)
    assert cropped_matrix[1, 2] == pytest.approx(-297.0)
