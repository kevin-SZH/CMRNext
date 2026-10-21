"""显式 EMD 场景、T0、Ti 和左乘残差加载。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import quaternion_from_matrix


@dataclass(frozen=True)
class EMDContract:
    """封装 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `EMDContract` 状态和操作。

    主要职责:
        封装 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `EMDContract` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    training_semantics: str
    target_network: str
    sensor_profile: str
    camera_width: int
    camera_height: int
    camera_fx: float
    camera_fy: float
    camera_cx: float
    camera_cy: float
    camera_skew: float
    resize_width: int
    resize_height: int
    evaluation_crop_top: int | None
    evaluation_crop_left: int | None
    evaluation_crop_width: int | None
    evaluation_crop_height: int | None
    lidar_rings: int
    lidar_points_per_ring: int
    lidar_vertical_fov_min_deg: float
    lidar_vertical_fov_max_deg: float
    lidar_repeated_pattern: bool
    lidar_scan_frequency_hz: float | None
    lidar_beams_per_scan: int | None
    lidar_accumulation_duration_s: float | None
    lidar_beam_rate_hz: float | None
    lidar_accumulated_scan_equivalents: float | None
    lidar_accumulated_beam_count: int | None
    calibration_file: str
    nominal_extrinsic_file: str | None
    true_extrinsics_file: str
    residuals_file: str | None
    image_directory: str
    point_cloud_directory: str
    recorded_pose_format: str | None
    recorded_nominal_extrinsic: tuple[float, ...] | None
    dummy_intensity: float
    transform_atol: float
    color_jitter: bool
    geometric_augmentation: bool
    crop_top_range: tuple[int, int] | None
    crop_left_range: tuple[int, int] | None
    require_sequence_audit: bool
    require_continuous_coverage: bool
    success_translation_m: float
    success_rotation_deg: float
    residual_magnitude_bins: tuple[float, ...]
    source_mode: str


@dataclass(frozen=True)
class EMDTrainingConfig:
    """封装 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `EMDTrainingConfig` 状态和操作。

    主要职责:
        封装 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `EMDTrainingConfig` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    schema_version: str
    path: Path
    dataset_root: Path
    train_scene_paths: tuple[Path, ...]
    validation_scene_paths: tuple[Path, ...]
    test_scene_paths: tuple[Path, ...]
    curriculum: Mapping[str, str]
    contract: EMDContract

    def scene_paths(self, role: str) -> tuple[Path, ...]:
        """返回 YAML 为指定数据角色声明的场景绝对路径。

        参数:
            role: 数据角色，只允许 ``train``、``validation`` 或 ``test``。

        返回值:
            按 YAML 原顺序排列的场景绝对路径元组；允许为空。

        异常和副作用:
            role 不受支持时抛出 ValueError；函数不访问场景文件且无副作用。

        坐标系和外参约定:
            本函数只选择路径，不读取或修改 T0、Ti 与左乘残差。
        """
        scene_groups = {
            "train": self.train_scene_paths,
            "validation": self.validation_scene_paths,
            "test": self.test_scene_paths,
        }
        try:
            return scene_groups[role]
        except KeyError as error:
            raise ValueError(f"不支持的 EMD 场景角色: {role}") from error

    def audit_record(self) -> dict:
        """构造可写入 dry-run、日志和 checkpoint 的 EMD v2 审计记录。

        参数:
            self: 已完成字段类型和几何语义校验的训练配置。

        返回值:
            仅包含 JSON 和 checkpoint 可序列化基础类型的配置字典。

        异常和副作用:
            无外部副作用；不读取数据文件，也不依据元数据筛选场景。

        坐标系和外参约定:
            外参文件遵循相机从 LiDAR 的列向量变换，残差满足
            delta_Ti = Ti @ inverse(T0)。
        """
        record = asdict(self.contract)
        record.update({
            "schema_version": self.schema_version,
            "config_path": str(self.path),
            "dataset_root": str(self.dataset_root),
            "scenes": {
                "train": [str(path) for path in self.train_scene_paths],
                "validation": [
                    str(path) for path in self.validation_scene_paths],
                "test": [str(path) for path in self.test_scene_paths],
            },
            "curriculum": dict(self.curriculum),
        })
        return record



@dataclass(frozen=True)
class _SequenceData:
    """封装 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `_SequenceData` 状态和操作。

    主要职责:
        封装 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `_SequenceData` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    directory: str
    image_paths: tuple[str, ...]
    point_cloud_paths: tuple[str, ...]
    intrinsics: torch.Tensor
    true_extrinsics: torch.Tensor
    residuals: torch.Tensor
    nominal_extrinsic: torch.Tensor
    effective_max_translation_m: float
    max_geodesic_rotation_deg: float


def _require_mapping(value, name: str) -> Mapping:
    """完成 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `_require_mapping` 操作。

    参数:
        value: 需要换算、比较或写入报告的数值。
        name: 用于错误信息定位配置字段、矩阵集合或数据项的名称。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是映射")
    return value


def _optional_int_range(value, name: str) -> tuple[int, int] | None:
    """完成 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `_optional_int_range` 操作。

    参数:
        value: 需要换算、比较或写入报告的数值。
        name: 用于错误信息定位配置字段、矩阵集合或数据项的名称。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} 必须包含 [最小值, 最大值]")
    minimum, maximum = value
    if (isinstance(minimum, bool) or isinstance(maximum, bool)
            or not isinstance(minimum, int) or not isinstance(maximum, int)
            or minimum < 0 or maximum < minimum):
        raise ValueError(
            f"{name} 必须包含非负且递增的整数")
    return minimum, maximum


def _read_calib_file(path: Path) -> dict[str, np.ndarray]:
    """读取并解析当前函数约定的文件格式。

    参数:
        path: 输入或输出文件路径。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    data = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.partition(":")
            if not separator:
                continue
            try:
                data[key] = np.asarray(
                    [float(item) for item in value.split()], dtype=np.float64)
            except ValueError:
                continue
    return data


def _resolve_scene_paths(values, *, name: str, dataset_root: Path) -> tuple[Path, ...]:
    """把 YAML 场景列表解析为以数据根目录为基准的绝对路径。

    参数:
        values: YAML 中的场景路径列表，元素可为绝对或相对路径字符串。
        name: 用于错误信息定位的字段名称。
        dataset_root: 相对场景路径的解析基准目录。

    返回值:
        保留用户声明顺序的绝对路径元组；空列表返回空元组。

    异常和副作用:
        非列表、空字符串或重复路径会抛出 ValueError；不检查路径是否存在，
        也不读取 split、metadata 或 manifest。

    坐标系和外参约定:
        本函数只解析路径，不读取相机内参或 T0、Ti、delta_Ti。
    """
    if not isinstance(values, list):
        raise ValueError(f"{name} 必须是路径列表")
    resolved_paths = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] 必须是非空路径字符串")
        configured_path = Path(value).expanduser()
        resolved_path = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (dataset_root / configured_path).resolve()
        )
        resolved_paths.append(resolved_path)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(f"{name} 不能包含重复场景")
    return tuple(resolved_paths)


def _optional_positive_float(value, name: str) -> float | None:
    """读取可选的正浮点传感器参数。

    参数:
        value: YAML 数值或 None。
        name: 用于错误信息定位的字段名称。

    返回值:
        None 或有限正浮点数。

    异常和副作用:
        布尔值、非数值、非有限值或非正值会抛出 ValueError；无副作用。

    坐标系和外参约定:
        本函数只校验标量；具体单位由调用字段名称声明。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正数或 null")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return parsed


def _optional_positive_int(value, name: str) -> int | None:
    """读取可选的正整数传感器参数。

    参数:
        value: YAML 整数或 None。
        name: 用于错误信息定位的字段名称。

    返回值:
        None 或正整数。

    异常和副作用:
        布尔值、非整数或非正值会抛出 ValueError；无副作用。

    坐标系和外参约定:
        本函数只校验计数，不读取点云或外参。
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数或 null")
    return value


def _safe_scene_filename(value, name: str) -> str:
    """校验场景内标定或外参文件名。

    参数:
        value: YAML 中的文件名字符串。
        name: 用于错误信息定位的字段名称。

    返回值:
        不含目录部分的场景内文件名。

    异常和副作用:
        空值、绝对路径或带目录分隔的名称会抛出 ValueError；无副作用。

    坐标系和外参约定:
        文件内容由加载器按相机从 LiDAR 的列向量变换解释。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空文件名")
    filename = Path(value)
    if filename.is_absolute() or filename.name != value or value in {".", ".."}:
        raise ValueError(f"{name} 只能是场景目录内的文件名")
    return value


def _configured_extrinsic_matrix(
    value,
    *,
    name: str,
    atol: float,
) -> tuple[float, ...]:
    """读取 YAML 中声明的四乘四 LiDAR-to-camera 外参矩阵。

    参数:
        value: YAML 列表形式的四行四列齐次矩阵。
        name: 用于错误信息定位的配置字段名称。
        atol: 旋转正交性、行列式和齐次底行校验使用的绝对容差。

    返回值:
        按行展开的 16 个有限浮点数元组，可安全保存到不可变数据合同中。

    异常和副作用:
        矩阵形状或刚体变换几何非法时抛出 ValueError；函数不访问文件且无副作用。

    坐标系和外参约定:
        输入矩阵必须满足 p_camera = T0 @ p_lidar，平移单位为米，采用列向量约定。
    """
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是四行四列数值矩阵") from error
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} 必须是四行四列数值矩阵")
    _validate_transforms(matrix[None], name, atol)
    return tuple(float(item) for item in matrix.reshape(-1))


def load_emd_training_config(
    config_path: str | os.PathLike,
    dataset_root_override: str | os.PathLike | None = None,
) -> EMDTrainingConfig:
    """从 YAML 读取 EMD 传感器、裁剪、课程级别和几何容差配置。

    参数:
        config_path: 当前操作读取或写入的文件路径。
        dataset_root_override: 可选数据根目录覆盖值；只替换 YAML 中的根路径。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = _require_mapping(raw, "EMD config")
    supported_schema_versions = {
        "cmrnext_scene_calibration_v1",
        "cmrnext_emd_training_v2",
    }
    if raw.get("schema_version") not in supported_schema_versions:
        raise ValueError(
            "schema_version 必须是 cmrnext_scene_calibration_v1；"
            "旧配置可继续使用 cmrnext_emd_training_v2")

    dataset = _require_mapping(raw.get("dataset"), "dataset")
    sensor = _require_mapping(dataset.get("sensor"), "dataset.sensor")
    camera = _require_mapping(sensor.get("camera"), "dataset.sensor.camera")
    lidar = _require_mapping(sensor.get("lidar"), "dataset.sensor.lidar")
    validation = _require_mapping(dataset.get("validation"), "dataset.validation")
    augmentation = _require_mapping(raw.get("augmentation"), "augmentation")
    acquisition = _require_mapping(
        lidar.get("acquisition"), "dataset.sensor.lidar.acquisition")
    files = _require_mapping(dataset.get("files"), "dataset.files")
    scenes = _require_mapping(dataset.get("scenes"), "dataset.scenes")

    evaluation = _require_mapping(raw.get("evaluation"), "evaluation")
    preprocessing_raw = raw.get("preprocessing")
    preprocessing = (
        {} if preprocessing_raw is None
        else _require_mapping(preprocessing_raw, "preprocessing"))
    curriculum = dict(_require_mapping(raw.get("curriculum"), "curriculum"))
    source_mode = str(dataset.get("source_mode", "cmrnext_stored_residuals_v1"))
    if source_mode == "cmrnext_stored_residuals_v1":
        if set(curriculum) != {"iter1", "iter2", "iter3"}:
            raise ValueError(
                "存储式 EMD 课程必须正好定义 iter1、iter2 和 iter3")
    elif source_mode in {"calibnet_stored_residuals_v1", "recorded_pose_v1"}:
        expected_keys = [
            f"iter{stage}" for stage in range(1, len(curriculum) + 1)]
        if not 1 <= len(curriculum) <= 3 or list(curriculum) != expected_keys:
            raise ValueError(
                "CalibNet 与真实域位姿课程必须从 iter1 开始连续声明，"
                "并且最多包含三个阶段")
        if source_mode == "recorded_pose_v1":
            unknown_levels = set(curriculum.values()) - {
                "fixed_baseline", "perturbed_t0"}
            if unknown_levels:
                raise ValueError(
                    "recorded_pose_v1 只支持 fixed_baseline 或 perturbed_t0")
    elif source_mode == "calibnet_fixed_t0_v1":
        expected_keys = [
            f"iter{stage}" for stage in range(1, len(curriculum) + 1)]
        if (
            not 1 <= len(curriculum) <= 3
            or list(curriculum) != expected_keys
            or any(level != "fixed_baseline" for level in curriculum.values())
        ):
            raise ValueError(
                "固定 T0 课程必须从 iter1 连续定义到最多 iter3，且都映射到 fixed_baseline")
        online = _require_mapping(
            raw.get("fixed_t0_online_perturbation"),
            "fixed_t0_online_perturbation",
        )
        if online.get("enabled") is not True:
            raise ValueError(
                "calibnet_fixed_t0_v1 要求启用固定 T0 在线扰动")
    else:
        raise ValueError(f"不支持的 dataset.source_mode：{source_mode}")

    configured_root = Path(str(dataset.get("root", ""))).expanduser()
    if dataset_root_override is not None:
        dataset_root = Path(dataset_root_override).expanduser().resolve()
    elif configured_root.is_absolute():
        dataset_root = configured_root.resolve()
    else:
        dataset_root = (path.parent / configured_root).resolve()
    # Resolve only user-configured scenes; ignore split metadata and manifests.
    train_scene_paths = _resolve_scene_paths(
        scenes.get("train"), name="dataset.scenes.train",
        dataset_root=dataset_root)
    validation_scene_paths = _resolve_scene_paths(
        scenes.get("validation"), name="dataset.scenes.validation",
        dataset_root=dataset_root)
    test_scene_paths = _resolve_scene_paths(
        scenes.get("test"), name="dataset.scenes.test",
        dataset_root=dataset_root)

    # Require calibration files inside each explicit scene.
    calibration_file = _safe_scene_filename(
        files.get("calibration"), "dataset.files.calibration")
    true_extrinsics_file = _safe_scene_filename(
        files.get("true_extrinsics"), "dataset.files.true_extrinsics")
    layout_raw = dataset.get("layout", {})
    layout = _require_mapping(layout_raw, "dataset.layout")
    image_directory = _safe_scene_filename(
        layout.get("images", "image_2"), "dataset.layout.images")
    point_cloud_directory = _safe_scene_filename(
        layout.get("point_clouds", "velodyne"),
        "dataset.layout.point_clouds")
    if image_directory == point_cloud_directory:
        raise ValueError("图像目录和点云目录必须不同")

    recorded_pose_format = None
    recorded_nominal_extrinsic = None
    if source_mode == "recorded_pose_v1":
        nominal_extrinsic_file = None
        residuals_file = None
        recorded_pose = _require_mapping(
            dataset.get("recorded_pose"), "dataset.recorded_pose")
        recorded_pose_format = str(recorded_pose.get("format", ""))
        if recorded_pose_format != "translation_xyz_quaternion_xyzw":
            raise ValueError(
                "dataset.recorded_pose.format 必须是 "
                "translation_xyz_quaternion_xyzw")
        recorded_nominal_extrinsic = _configured_extrinsic_matrix(
            recorded_pose.get("nominal_extrinsic"),
            name="dataset.recorded_pose.nominal_extrinsic",
            atol=float(validation["transform_atol"]),
        )
        if calibration_file == true_extrinsics_file:
            raise ValueError("真实域标定文件和位姿文件必须不同")
    else:
        nominal_extrinsic_file = _safe_scene_filename(
            files.get("nominal_extrinsic"),
            "dataset.files.nominal_extrinsic")
        residuals_file = _safe_scene_filename(
            files.get("residuals"), "dataset.files.residuals")
        if len({
                calibration_file, nominal_extrinsic_file,
                true_extrinsics_file, residuals_file}) != 4:
            raise ValueError("dataset.files 中的四个文件名必须互不相同")

    # Accumulation metadata describes offline clouds; training does not rebuild scans.
    repeated_pattern = acquisition.get("repeated_pattern")
    if not isinstance(repeated_pattern, bool):
        raise ValueError(
            "dataset.sensor.lidar.acquisition.repeated_pattern 必须是布尔值")
    scan_frequency_hz = _optional_positive_float(
        acquisition.get("scan_frequency_hz"),
        "dataset.sensor.lidar.acquisition.scan_frequency_hz")
    beams_per_scan = _optional_positive_int(
        acquisition.get("beams_per_scan"),
        "dataset.sensor.lidar.acquisition.beams_per_scan")
    accumulation_duration_s = _optional_positive_float(
        acquisition.get("accumulation_duration_s"),
        "dataset.sensor.lidar.acquisition.accumulation_duration_s")
    beam_rate_hz = _optional_positive_float(
        acquisition.get("beam_rate_hz"),
        "dataset.sensor.lidar.acquisition.beam_rate_hz")
    acquisition_values = (
        scan_frequency_hz, beams_per_scan,
        accumulation_duration_s, beam_rate_hz)
    if any(value is not None for value in acquisition_values):
        if any(value is None for value in acquisition_values):
            raise ValueError(
                "声明 LiDAR 累计参数时必须同时提供 scan_frequency_hz、"
                "beams_per_scan、accumulation_duration_s 和 beam_rate_hz")
        expected_beam_rate = float(beams_per_scan) * scan_frequency_hz
        if not np.isclose(beam_rate_hz, expected_beam_rate, rtol=1.0e-9):
            raise ValueError(
                "beam_rate_hz 必须等于 beams_per_scan * scan_frequency_hz")
        accumulated_scan_equivalents = (
            accumulation_duration_s * scan_frequency_hz)
        accumulated_beam_count = int(round(
            accumulation_duration_s * beam_rate_hz))
        if not np.isclose(
                accumulated_beam_count,
                accumulation_duration_s * beam_rate_hz,
                atol=1.0e-6):
            raise ValueError("累计时长必须对应整数个发射束")
    else:
        accumulated_scan_equivalents = None
        accumulated_beam_count = None


    vertical_fov = lidar.get("vertical_fov_deg")
    if not isinstance(vertical_fov, list) or len(vertical_fov) != 2:
        raise ValueError("dataset.sensor.lidar.vertical_fov_deg 必须包含两个值")
    # resize_hw uses HxW and defaults to the source resolution.
    resize_hw = preprocessing.get(
        "resize_hw", [int(camera["height"]), int(camera["width"])])
    if (not isinstance(resize_hw, list) or len(resize_hw) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in resize_hw)):
        raise ValueError(
            "preprocessing.resize_hw 必须包含正数 [高度, 宽度]")
    resize_height, resize_width = resize_hw

    # An evaluation crop must define its origin and HxW.
    evaluation_crop_raw = preprocessing.get("evaluation_crop")
    if evaluation_crop_raw is None:
        evaluation_crop = None
    else:
        evaluation_crop = _require_mapping(
            evaluation_crop_raw, "preprocessing.evaluation_crop")
        required_crop_keys = {"top", "left", "height", "width"}
        if set(evaluation_crop) != required_crop_keys:
            raise ValueError(
                "preprocessing.evaluation_crop 必须定义 top、left、height 和 width")
        if any(isinstance(evaluation_crop[key], bool)
               or not isinstance(evaluation_crop[key], int)
               or evaluation_crop[key] < 0
               for key in required_crop_keys):
            raise ValueError(
                "preprocessing.evaluation_crop 的值必须是非负整数")
        if evaluation_crop["height"] <= 0 or evaluation_crop["width"] <= 0:
            raise ValueError(
                "preprocessing.evaluation_crop 的高度和宽度必须为正数")
        if (evaluation_crop["top"] + evaluation_crop["height"] > resize_height
                or evaluation_crop["left"] + evaluation_crop["width"] > resize_width):
            raise ValueError(
                "preprocessing.evaluation_crop 超出 preprocessing.resize_hw")
    contract = EMDContract(
        training_semantics=str(dataset["training_semantics"]),
        target_network=str(dataset["target_network"]),
        sensor_profile=str(sensor["profile"]),
        camera_width=int(camera["width"]),
        camera_height=int(camera["height"]),
        camera_fx=float(camera["fx"]),
        camera_fy=float(camera["fy"]),
        camera_cx=float(camera["cx"]),
        camera_cy=float(camera["cy"]),
        resize_width=int(resize_width),
        resize_height=int(resize_height),
        evaluation_crop_top=(
            None if evaluation_crop is None else int(evaluation_crop["top"])),
        evaluation_crop_left=(
            None if evaluation_crop is None else int(evaluation_crop["left"])),
        evaluation_crop_width=(
            None if evaluation_crop is None else int(evaluation_crop["width"])),
        evaluation_crop_height=(
            None if evaluation_crop is None else int(evaluation_crop["height"])),
        lidar_rings=int(lidar["rings"]),
        lidar_points_per_ring=int(lidar["points_per_ring"]),
        lidar_vertical_fov_min_deg=float(vertical_fov[0]),
        camera_skew=float(camera.get("skew", 0.0)),
        lidar_vertical_fov_max_deg=float(vertical_fov[1]),
        lidar_repeated_pattern=repeated_pattern,
        lidar_scan_frequency_hz=scan_frequency_hz,
        lidar_beams_per_scan=beams_per_scan,
        lidar_accumulation_duration_s=accumulation_duration_s,
        lidar_beam_rate_hz=beam_rate_hz,
        lidar_accumulated_scan_equivalents=accumulated_scan_equivalents,
        lidar_accumulated_beam_count=accumulated_beam_count,
        calibration_file=calibration_file,
        nominal_extrinsic_file=nominal_extrinsic_file,
        true_extrinsics_file=true_extrinsics_file,
        residuals_file=residuals_file,
        image_directory=image_directory,
        point_cloud_directory=point_cloud_directory,
        recorded_pose_format=recorded_pose_format,
        recorded_nominal_extrinsic=recorded_nominal_extrinsic,
        dummy_intensity=float(dataset["dummy_intensity"]),
        transform_atol=float(validation["transform_atol"]),
        color_jitter=bool(augmentation["color_jitter"]),
        geometric_augmentation=bool(augmentation["geometric"]),
        crop_top_range=_optional_int_range(
            augmentation.get("crop_top"), "augmentation.crop_top"),
        crop_left_range=_optional_int_range(
            augmentation.get("crop_left"), "augmentation.crop_left"),
        require_sequence_audit=bool(validation["require_sequence_audit"]),
        require_continuous_coverage=bool(
            validation["require_continuous_coverage"]),
        success_translation_m=float(evaluation["success_translation_m"]),
        success_rotation_deg=float(evaluation["success_rotation_deg"]),
        residual_magnitude_bins=tuple(
            float(value) for value in evaluation["residual_magnitude_bins"]),
        source_mode=source_mode,
    )
    if contract.training_semantics != "nominal_to_true_residual_v1":
        raise ValueError("EMD training_semantics 必须为 nominal_to_true_residual_v1")
    expected_target = (
        "calibnet"
        if (contract.source_mode.startswith("calibnet_")
            or contract.source_mode == "recorded_pose_v1")
        else "cmrnext"
    )
    if contract.target_network != expected_target:
        raise ValueError(
            f"{contract.source_mode} 要求 target_network 为 {expected_target}")
    if contract.geometric_augmentation:
        raise ValueError(
            "EMD 暂不支持几何增强，因为必须对存储的左乘残差进行共轭变换；"
            "请保持 augmentation.geometric=false"
        )
    if contract.transform_atol <= 0.0:
        raise ValueError("dataset.validation.transform_atol 必须为正数")
    if contract.success_translation_m <= 0.0 or contract.success_rotation_deg <= 0.0:
        raise ValueError("评估成功阈值必须为正数")
    if (len(contract.residual_magnitude_bins) < 2
            or contract.residual_magnitude_bins[0] != 0.0
            or any(right <= left for left, right in zip(
                contract.residual_magnitude_bins,
                contract.residual_magnitude_bins[1:]))):
        raise ValueError("evaluation residual_magnitude_bins 必须从零开始递增")
    return EMDTrainingConfig(
        schema_version=str(raw["schema_version"]),
        path=path,
        dataset_root=dataset_root,
        train_scene_paths=train_scene_paths,
        validation_scene_paths=validation_scene_paths,
        test_scene_paths=test_scene_paths,
        curriculum=curriculum,
        contract=contract,
    )


def _homogeneous_rows(path: Path, expected_count: int | None = None) -> np.ndarray:
    """完成 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `_homogeneous_rows` 操作。

    参数:
        path: 输入或输出文件路径。
        expected_count: 对应元素或操作的数量。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape == (3, 4):
        rows = rows.reshape(1, 12)
    elif rows.shape == (4, 4):
        rows = rows.reshape(1, 16)
    if rows.ndim != 2 or rows.shape[1] not in (12, 16):
        raise ValueError(f"{path} 中每个变换必须包含 12 或 16 个值")
    if expected_count is not None and rows.shape[0] != expected_count:
        raise ValueError(
            f"{path} 包含 {rows.shape[0]} 个变换，预期 {expected_count} 个"
        )
    if rows.shape[1] == 12:
        matrices = np.zeros((rows.shape[0], 4, 4), dtype=np.float64)
        matrices[:, :3, :4] = rows.reshape(-1, 3, 4)
        matrices[:, 3, 3] = 1.0
    else:
        matrices = rows.reshape(-1, 4, 4)
    return matrices


def _recorded_pose_rows(
    path: Path,
    *,
    expected_count: int,
    atol: float,
) -> np.ndarray:
    """读取 xyz 与 xyzw 四元数形式的逐帧真实外参。

    参数:
        path: 每行依次保存 tx、ty、tz、qx、qy、qz、qw 的文本文件路径。
        expected_count: 图像和点云共同确定的期望帧数。
        atol: 四元数单位范数和刚体变换校验使用的绝对容差。

    返回值:
        形状为 N×4×4 的 LiDAR-to-camera Ti 数组，平移单位为米。

    异常和副作用:
        列数、帧数、有限性、四元数范数或生成旋转非法时抛出 ValueError；只读文件。

    坐标系和外参约定:
        四元数采用 qx、qy、qz、qw 顺序，生成满足
        p_camera = Ti @ p_lidar 的列向量左乘外参。
    """
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"{path} 每行必须包含 xyz 和 qx、qy、qz、qw 共 7 个数")
    if rows.shape[0] != expected_count:
        raise ValueError(
            f"{path} 包含 {rows.shape[0]} 行位姿，但图像和点云共有 "
            f"{expected_count} 帧")
    if not np.isfinite(rows).all():
        raise ValueError(f"{path} 包含非有限位姿数值")

    # Validate unit quaternions before building xyzw LiDAR-to-camera rotations.
    quaternions = rows[:, 3:]
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.allclose(norms, 1.0, atol=atol, rtol=0.0):
        invalid_index = int(np.argmax(np.abs(norms - 1.0)))
        raise ValueError(
            f"{path} 第 {invalid_index + 1} 行四元数范数不是 1: "
            f"{norms[invalid_index]:.12f}")
    quaternions = quaternions / norms[:, None]
    x, y, z, w = quaternions.T
    rotations = np.empty((rows.shape[0], 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    matrices = np.repeat(
        np.eye(4, dtype=np.float64)[None], rows.shape[0], axis=0)
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = rows[:, :3]
    _validate_transforms(matrices, str(path), atol)
    return matrices


def _validate_transforms(matrices: np.ndarray, name: str, atol: float) -> None:
    """完成 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `_validate_transforms` 操作。

    参数:
        matrices: 待校验的一组四乘四齐次刚体变换矩阵。
        name: 用于错误信息定位配置字段、矩阵集合或数据项的名称。
        atol: 数值一致性判断使用的绝对容差。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if not np.isfinite(matrices).all():
        raise ValueError(f"{name} contains non-finite values")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(matrices[:, 3, :], expected_bottom, atol=atol):
        raise ValueError(f"{name} contains invalid homogeneous bottom rows")
    rotations = matrices[:, :3, :3]
    identities = np.matmul(rotations, np.swapaxes(rotations, 1, 2))
    if not np.allclose(identities, np.eye(3), atol=atol):
        raise ValueError(f"{name} contains non-orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=atol):
        raise ValueError(f"{name} contains improper rotations")


def _check_close(actual, expected, name: str, atol: float) -> None:
    """完成 显式 EMD 场景、T0、Ti 和左乘残差加载 中的 `_check_close` 操作。

    参数:
        actual: 从数据文件读取或计算得到的实际数值数组。
        expected: 配置或几何关系要求的期望数值数组。
        name: 用于错误信息定位配置字段、矩阵集合或数据项的名称。
        atol: 数值一致性判断使用的绝对容差。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=atol):
        raise ValueError(f"{name} does not match the configured sensor profile")


@lru_cache(maxsize=256)
def _load_sequence(
    directory_text: str,
    contract: EMDContract,
) -> _SequenceData:
    """直接读取一个 EMD 场景目录。

    参数:
        directory_text: 场景目录路径。目录内直接包含图像、点云、内参和外参文件。
        contract: 配置文件给出的相机、激光雷达和外参容差约定。

    返回值:
        已完成文件归属、传感器参数和外参几何检查的不可变场景数据。

    异常和副作用:
        必需文件缺失、图像与点云编号不对应、相机内参不匹配或
        residual @ T0 != Ti 时抛出异常。函数只读实际训练文件，不读取
        metadata.json、manifest.json 或任何 train/val/test 标记。

    坐标系和外参约定:
        所有外参均为相机从激光雷达的 4x4 列向量变换。KITTI 风格来源直接
        读取 T0、Ti 和残差；真实记录来源读取逐帧 Ti，并用 YAML T0 派生残差。
    """
    directory = Path(directory_text)
    required_files = [contract.calibration_file, contract.true_extrinsics_file]
    if contract.source_mode != "recorded_pose_v1":
        required_files.extend([
            contract.nominal_extrinsic_file,
            contract.residuals_file,
        ])
    missing = [name for name in required_files if not (directory / name).is_file()]
    for folder in (contract.image_directory, contract.point_cloud_directory):
        if not (directory / folder).is_dir():
            missing.append(folder + "/")
    if missing:
        raise FileNotFoundError(f"{directory} 缺少: {', '.join(missing)}")

    image_by_index = {}
    for path in sorted((directory / contract.image_directory).iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            continue
        try:
            image_by_index[int(path.stem)] = str(path)
        except ValueError:
            continue
    point_by_index = {}
    for path in sorted(
            (directory / contract.point_cloud_directory).glob("*.bin")):
        try:
            point_by_index[int(path.stem)] = str(path)
        except ValueError:
            continue
    frame_indices = sorted(image_by_index)
    if not frame_indices or frame_indices != sorted(point_by_index):
        raise ValueError(f"{directory}: 图像与点云帧编号不一致")

    sample_count = len(frame_indices)
    if contract.source_mode == "recorded_pose_v1":
        true_extrinsics = _recorded_pose_rows(
            directory / contract.true_extrinsics_file,
            expected_count=sample_count,
            atol=contract.transform_atol,
        )
        nominal = np.asarray(
            contract.recorded_nominal_extrinsic,
            dtype=np.float64,
        ).reshape(1, 4, 4)
        # Geometry uniquely defines the complete Ti-from-T0 left residual.
        residuals = np.matmul(true_extrinsics, np.linalg.inv(nominal[0]))
    else:
        true_extrinsics = _homogeneous_rows(
            directory / contract.true_extrinsics_file)
        if true_extrinsics.shape[0] != sample_count:
            raise ValueError(
                f"{directory}: 文件帧数 {sample_count} 与 "
                f"{contract.true_extrinsics_file} "
                f"行数 {true_extrinsics.shape[0]} 不一致")
        residuals = _homogeneous_rows(
            directory / contract.residuals_file, sample_count)
        nominal = _homogeneous_rows(directory / contract.nominal_extrinsic_file)
        if nominal.shape[0] != 1:
            raise ValueError(
                f"{directory}: {contract.nominal_extrinsic_file} "
                "必须只包含一个外参")
    _validate_transforms(true_extrinsics, f"{directory} poses", contract.transform_atol)
    _validate_transforms(residuals, f"{directory} residuals", contract.transform_atol)
    _validate_transforms(nominal, f"{directory} nominal extrinsic", contract.transform_atol)
    reconstructed = np.matmul(residuals, nominal[0])
    if not np.allclose(reconstructed, true_extrinsics, atol=contract.transform_atol):
        max_error = float(np.max(np.abs(reconstructed - true_extrinsics)))
        raise ValueError(f"{directory}: delta @ T0 != Ti (max error {max_error:.3e})")

    calib_data = _read_calib_file(directory / contract.calibration_file)
    projection = calib_data.get("P2", calib_data.get("P0"))
    if projection is None or projection.size != 12:
        raise ValueError(
            f"{directory}: {contract.calibration_file} 必须包含 3x4 P2 或 P0")
    projection = projection.reshape(3, 4)
    if np.linalg.norm(projection[:, 3]) > contract.transform_atol:
        raise ValueError(f"{directory}：投影矩阵的平移项必须为零")
    intrinsics = np.array(
        [projection[0, 0], projection[1, 1], projection[0, 2],
         projection[1, 2], projection[0, 1]],
        dtype=np.float32,
    )
    _check_close(
        intrinsics,
        [contract.camera_fx, contract.camera_fy, contract.camera_cx,
         contract.camera_cy, contract.camera_skew],
        f"{directory} {contract.calibration_file}",
        contract.transform_atol,
    )

    # Derive normalization from scene residuals, not statistics files.
    translations = np.linalg.norm(residuals[:, :3, 3], axis=1)
    rotation_traces = np.trace(residuals[:, :3, :3], axis1=1, axis2=2)
    rotation_cosines = np.clip((rotation_traces - 1.0) * 0.5, -1.0, 1.0)
    effective_max_t = max(float(np.max(translations)), 1.0e-12)
    max_geodesic_r = max(
        float(np.degrees(np.max(np.arccos(rotation_cosines)))), 1.0e-12)

    return _SequenceData(
        directory=str(directory),
        image_paths=tuple(image_by_index[index] for index in frame_indices),
        point_cloud_paths=tuple(point_by_index[index] for index in frame_indices),
        intrinsics=torch.from_numpy(intrinsics),
        true_extrinsics=torch.from_numpy(true_extrinsics.astype(np.float32)),
        residuals=torch.from_numpy(residuals.astype(np.float32)),
        nominal_extrinsic=torch.from_numpy(nominal[0].astype(np.float32)),
        effective_max_translation_m=effective_max_t,
        max_geodesic_rotation_deg=max_geodesic_r,
    )


class DatasetEMDExtrinsicCalib(Dataset):
    """直接读取显式 EMD 场景，并返回满足 delta=Ti@inverse(T0) 的训练样本。

    主要职责:
        直接读取显式 EMD 场景，并返回满足 delta=Ti@inverse(T0) 的训练样本，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    def __init__(
        self,
        scene_paths: Sequence[str | os.PathLike],
        *,
        level: str,
        contract: EMDContract,
        train: bool = False,
        normalize_images: bool = True,
        use_reflectance: bool = False,
    ):
        """初始化 EMD 外参标定数据集。

        参数:
            scene_paths: 用户显式选择的场景目录列表，列表顺序即数据读取顺序。
            level: 当前 curriculum level。
            contract: 传感器、裁剪、阈值和 nominal_to_true_residual_v1 契约。
            train: 是否启用训练期图像增强；该标志不参与场景选择。
            normalize_images: 是否按 CMRNext 图像归一化约定返回 RGB。
            use_reflectance: EMD 强制为 False，因为导出强度是占位值。

        返回值:
            无显式返回值；实例保存经过验证的序列索引和数据契约。

        异常和副作用:
            传感器、帧归属、外参或残差不一致时抛出异常。初始化只读数据。

        坐标系和外参约定:
            原始点使用列向量，输入外参为 T0，目标外参为 Ti，左乘残差满足
            delta_Ti = Ti @ inverse(T0)，并逐序列验证 delta_Ti @ T0 == Ti。
        """
        if use_reflectance:
            raise ValueError("EMD 占位强度要求 use_reflectance=false")
        self.dataset = "emd"
        self.level = str(level)
        self.contract = contract
        self.train = bool(train)
        self.normalize_images = bool(normalize_images)
        self.use_reflectance = False
        self.color_transform = (
            transforms.ColorJitter(0.2, 0.2, 0.2)
            if self.train and contract.color_jitter else None
        )

        sequence_dirs = [Path(path).expanduser().resolve()
                         for path in scene_paths]
        if not sequence_dirs:
            raise ValueError("EMD 场景路径列表为空")
        self.sequences = tuple(
            _load_sequence(str(path), contract)
            for path in sequence_dirs
        )
        self.scene_paths = tuple(sequence.directory for sequence in self.sequences)

        self.all_files = []
        for sequence_index, sequence in enumerate(self.sequences):
            for frame_index in range(len(sequence.image_paths)):
                self.all_files.append((sequence_index, frame_index))

    def __len__(self) -> int:
        """返回当前数据集或容器可访问的样本数量。

        参数:
            self: 当前类实例。

        返回值:
            可访问样本或元素数量，类型为整数。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:
        """读取一帧图像、点云和外参监督。

        参数:
            idx: 数据集全局样本索引，无单位。

        返回值:
            CMRNext 样本字典，包含 RGB、相机系目标点云、内参、T0、Ti、
            左乘残差和用于训练的目标到输入误差。

        异常和副作用:
            点云为空、数值非有限或图像尺寸不匹配时抛出异常；只读场景文件。

        坐标系和外参约定:
            point_cloud 保存 Ti @ p_lidar，prepare_input 再使用 inverse(delta)
            还原到 T0 @ p_lidar；delta = Ti @ inverse(T0)。
        """
        sequence_index, frame_index = self.all_files[idx]
        sequence = self.sequences[sequence_index]
        image_path = sequence.image_paths[frame_index]
        point_cloud_path = sequence.point_cloud_paths[frame_index]

        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32)
        if point_cloud.size == 0 or point_cloud.size % 4 != 0:
            raise ValueError(f"{point_cloud_path}：应为非空 Nx4 float32 点云")
        point_cloud = point_cloud.reshape(-1, 4)
        if not np.isfinite(point_cloud).all():
            raise ValueError(f"{point_cloud_path}: point cloud contains non-finite values")
        # EMD intensity is unused; force homogeneous coordinates to one.
        point_cloud[:, 3] = 1.0

        true_extrinsic = sequence.true_extrinsics[frame_index]
        residual = sequence.residuals[frame_index]
        nominal_extrinsic = sequence.nominal_extrinsic
        point_cloud = torch.from_numpy(point_cloud).t()
        aligned_point_cloud = torch.mm(true_extrinsic, point_cloud)

        # The network uses target-to-input error; files store input-to-target residuals.
        input_from_target = torch.linalg.inv(residual)
        rot_error = quaternion_from_matrix(input_from_target).float()
        tr_error = input_from_target[:3, 3].clone().float()

        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            if image.size != (self.contract.camera_width, self.contract.camera_height):
                raise ValueError(
                    f"{image_path}: image size {image.size} does not match sensor profile"
                )
            # Resize RGB now and reproject LiDAR with scaled intrinsics later.
            if image.size != (
                    self.contract.resize_width, self.contract.resize_height):
                resampling = getattr(Image, "Resampling", Image)
                image = image.resize(
                    (self.contract.resize_width, self.contract.resize_height),
                    resample=resampling.BILINEAR)
            if self.color_transform is not None:
                image = self.color_transform(image)
            rgb = torch.from_numpy(np.asarray(image).copy()).float()

        # Scale pinhole intrinsics per axis; skew follows the horizontal scale.
        scale_x = self.contract.resize_width / self.contract.camera_width
        scale_y = self.contract.resize_height / self.contract.camera_height
        resized_intrinsics = sequence.intrinsics.clone()
        resized_intrinsics[0] *= scale_x
        resized_intrinsics[1] *= scale_y
        resized_intrinsics[2] *= scale_x
        resized_intrinsics[3] *= scale_y
        if resized_intrinsics.numel() > 4:
            resized_intrinsics[4] *= scale_x

        return {
            "rgb": rgb,
            "point_cloud": aligned_point_cloud,
            "calib": resized_intrinsics,
            "tr_error": tr_error,
            "rot_error": rot_error,
            "rgb_name": image_path,
            "idx": idx,
            "cam2vel": true_extrinsic.clone(),
            "nominal_extrinsic": nominal_extrinsic.clone(),
            "residual": residual.clone(),
            "effective_max_t": torch.tensor(
                sequence.effective_max_translation_m, dtype=torch.float32
            ),
            "effective_max_r_deg": torch.tensor(
                sequence.max_geodesic_rotation_deg, dtype=torch.float32
            ),
            "sequence": Path(sequence.directory).name,
            "curriculum_level": self.level,
            "data_role": "train" if self.train else "eval",
            "dataset_mode": "emd",
            "source_image_shape": torch.tensor(
                [self.contract.camera_height, self.contract.camera_width],
                dtype=torch.int32),
            "resized_image_shape": torch.tensor(
                [self.contract.resize_height, self.contract.resize_width],
                dtype=torch.int32),
            "image_scale_xy": torch.tensor(
                [scale_x, scale_y], dtype=torch.float32),
        }
