"""多 GT 外参场景的 T0 在线小扰动。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

from datasets.emd_extrinsic_calib import (
    DatasetEMDExtrinsicCalib,
    EMDContract,
)
from utils import quaternion_from_matrix


@dataclass(frozen=True)
class DynamicNominalLevelConfig:
    """封装 多 GT 外参场景的 T0 在线小扰动 中的 `DynamicNominalLevelConfig` 状态和操作。

    主要职责:
        封装 多 GT 外参场景的 T0 在线小扰动 中的 `DynamicNominalLevelConfig` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    translation_alpha_t: float
    translation_min_floor_m: float
    max_rotation_deg: float


@dataclass(frozen=True)
class EMDDynamicNominalConfig:
    """封装 多 GT 外参场景的 T0 在线小扰动 中的 `EMDDynamicNominalConfig` 状态和操作。

    主要职责:
        封装 多 GT 外参场景的 T0 在线小扰动 中的 `EMDDynamicNominalConfig` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    enabled: bool
    levels: Mapping[str, DynamicNominalLevelConfig]


def _require_mapping(value, name: str) -> Mapping:
    """完成 多 GT 外参场景的 T0 在线小扰动 中的 `_require_mapping` 操作。

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


def load_emd_dynamic_nominal_config(
    config_path: str | Path,
) -> EMDDynamicNominalConfig:
    """读取在存储 T0 基础上叠加小扰动的训练配置。

    参数:
        config_path: YAML 配置路径，旋转单位为度，平移下限单位为米。

    返回值:
        是否启用以及每个训练 level 的平移比例、平移下限和旋转上限。

    异常和副作用:
        配置缺字段、数值非正或 level 与 curriculum 不一致时抛出异常；只读文件。

    坐标系和外参约定:
        采样扰动左乘到存储 T0，随后重新计算 delta_Ti = Ti @ inverse(T0)。
    """
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = _require_mapping(raw, "EMD config")
    section = raw.get("dynamic_nominal_augmentation")
    if section is None:
        return EMDDynamicNominalConfig(False, {})
    section = _require_mapping(section, "dynamic_nominal_augmentation")
    enabled = bool(section.get("enabled", False))

    raw_levels = _require_mapping(
        section.get("levels", {}), "dynamic_nominal_augmentation.levels")
    if not enabled:
        return EMDDynamicNominalConfig(False, {})
    curriculum = _require_mapping(raw.get("curriculum"), "curriculum")
    expected_levels = set(str(level) for level in curriculum.values())
    if not expected_levels or set(raw_levels) != expected_levels:
        raise ValueError(
            "启用动态标称外参增强时，其 level 必须与配置的课程阶段完全一致"
        )

    levels = {}
    for level, value in raw_levels.items():
        value = _require_mapping(
            value, f"dynamic_nominal_augmentation.levels.{level}")
        level_config = DynamicNominalLevelConfig(
            translation_alpha_t=float(value["translation_alpha_t"]),
            translation_min_floor_m=float(
                value["translation_min_floor_m"]),
            max_rotation_deg=float(value["max_rotation_deg"]),
        )
        values = (
            level_config.translation_alpha_t,
            level_config.translation_min_floor_m,
            level_config.max_rotation_deg,
        )
        if not all(np.isfinite(item) and item > 0.0 for item in values):
            raise ValueError(
                f"{level} 的动态标称外参范围必须为有限正数")
        levels[str(level)] = level_config
    return EMDDynamicNominalConfig(True, levels)


def _rotation_angle_deg(transform: torch.Tensor) -> float:
    """从四乘四刚体变换计算旋转部分的测地角度，单位为度。

    参数:
        transform: 四乘四齐次刚体变换矩阵，平移单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    cosine = ((torch.trace(transform[:3, :3]) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _rigid_inverse(transform: torch.Tensor) -> torch.Tensor:
    """利用旋转矩阵正交性计算四乘四刚体变换的解析逆矩阵。

    参数:
        transform: 四乘四齐次刚体变换矩阵，平移单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    inverse = torch.eye(4, dtype=transform.dtype, device=transform.device)
    rotation_t = transform[:3, :3].transpose(0, 1)
    inverse[:3, :3] = rotation_t
    inverse[:3, 3] = -(rotation_t @ transform[:3, 3])
    return inverse


def _sample_uniform_ball(max_radius: float) -> np.ndarray:
    """在给定半径的三维球体内部按体积均匀采样平移向量。

    参数:
        max_radius: 三维球体采样半径上限，单位为米或调用处声明的无量纲尺度。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    direction = np.random.normal(size=3)
    norm = float(np.linalg.norm(direction))
    while norm == 0.0:
        direction = np.random.normal(size=3)
        norm = float(np.linalg.norm(direction))
    radius = max_radius * float(np.random.random()) ** (1.0 / 3.0)
    return direction * (radius / norm)


def _sample_left_perturbation(
    max_translation_m: float,
    max_rotation_deg: float,
) -> torch.Tensor:
    """采样旋转轴角和平移并组装左乘 T0 小扰动矩阵。

    参数:
        max_translation_m: 长度量或长度上限，单位为米。
        max_rotation_deg: 角度量或角度上限，单位为度。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    axis = _sample_uniform_ball(1.0)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(np.random.uniform(-max_rotation_deg, max_rotation_deg))
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
    perturbation = np.eye(4, dtype=np.float64)
    perturbation[:3, :3] = rotation
    perturbation[:3, 3] = _sample_uniform_ball(max_translation_m)
    return torch.from_numpy(perturbation)


class DatasetEMDDynamicNominalExtrinsicCalib(DatasetEMDExtrinsicCalib):
    """在每次训练访问时对已存 T0 左乘小扰动，同时保持逐帧 Ti 不变。

    主要职责:
        在每次训练访问时对已存 T0 左乘小扰动，同时保持逐帧 Ti 不变，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    def __init__(
        self,
        scene_paths: Sequence[str | Path],
        *,
        level: str,
        contract: EMDContract,
        dynamic_config: EMDDynamicNominalConfig,
        train: bool = True,
        normalize_images: bool = True,
        use_reflectance: bool = False,
    ):
        """初始化多 GT 外参加在线 T0 小扰动的数据集。

        参数:
            scene_paths: 直接读取的训练场景目录，不读取 split 或统计 JSON。
            level: 选择 dynamic_nominal_augmentation.levels 中的扰动尺度。
            contract: 相机、裁剪和外参容差配置。
            dynamic_config: T0 平移比例、米制下限和旋转角度上限。
            train: 必须为 True，因为该模式每次读取都会重新采样。
            normalize_images: 是否按 CMRNext 约定归一化图像。
            use_reflectance: 必须为 False。

        返回值:
            无显式返回值；实例按访问采样新的 T0 和对应左乘残差。

        异常和副作用:
            非训练用途、配置未启用或 level 缺失时抛出异常；读取样本会消耗
            当前进程的 NumPy 随机状态。

        坐标系和外参约定:
            new_T0 = jitter @ stored_T0，Ti 保持不变，
            new_delta = Ti @ inverse(new_T0)。
        """
        if not train:
            raise ValueError("dynamic nominal augmentation is train-only")
        if not dynamic_config.enabled:
            raise ValueError("dynamic nominal augmentation is not enabled")
        if level not in dynamic_config.levels:
            raise ValueError(f"missing dynamic nominal configuration for {level}")
        super().__init__(
            scene_paths=scene_paths,
            level=level,
            contract=contract,
            train=train,
            normalize_images=normalize_images,
            use_reflectance=use_reflectance,
        )
        self.dynamic_config = dynamic_config
        self.level_config = dynamic_config.levels[level]

    def __getitem__(self, idx: int) -> dict:
        """读取样本并在存储 T0 上叠加一次随机小扰动。

        参数:
            idx: 数据集全局样本索引，无单位。

        返回值:
            更新后的 CMRNext 样本字典，额外包含原始 T0、原始残差、采样扰动
            以及新残差的平移米制范数和旋转角度。

        异常和副作用:
            外参重建误差超过配置容差时抛出异常；调用会推进 NumPy 随机状态。

        坐标系和外参约定:
            new_T0 = jitter @ stored_T0，new_delta = Ti @ inverse(new_T0)。
        """
        sample = super().__getitem__(idx)
        true_extrinsic = sample["cam2vel"].double()
        original_nominal = sample["nominal_extrinsic"].double()
        original_residual = sample["residual"].double()
        true_translation_scale = float(
            torch.linalg.vector_norm(true_extrinsic[:3, 3]))
        max_jitter_translation = max(
            self.level_config.translation_alpha_t * true_translation_scale,
            self.level_config.translation_min_floor_m,
        )
        perturbation = _sample_left_perturbation(
            max_jitter_translation, self.level_config.max_rotation_deg)
        new_nominal = perturbation @ original_nominal
        new_residual = true_extrinsic @ _rigid_inverse(new_nominal)
        reconstruction_error = float(torch.max(torch.abs(
            new_residual @ new_nominal - true_extrinsic)))
        if reconstruction_error > self.contract.transform_atol:
            raise RuntimeError(
                f"T0 扰动后的外参重建误差过大: {reconstruction_error:.3e}")
        translation_norm = float(torch.linalg.vector_norm(new_residual[:3, 3]))
        rotation_deg = _rotation_angle_deg(new_residual)
        input_from_target = _rigid_inverse(new_residual).float()
        sample.update({
            "tr_error": input_from_target[:3, 3].clone(),
            "rot_error": quaternion_from_matrix(input_from_target).float(),
            "nominal_extrinsic": new_nominal.float(),
            "residual": new_residual.float(),
            "original_nominal_extrinsic": original_nominal.float(),
            "original_residual": original_residual.float(),
            "nominal_jitter": perturbation.float(),
            "nominal_jitter_max_translation_m": torch.tensor(
                max_jitter_translation, dtype=torch.float32),
            "residual_translation_norm_m": torch.tensor(
                translation_norm, dtype=torch.float32),
            "residual_rotation_deg": torch.tensor(
                rotation_deg, dtype=torch.float32),
            "dataset_mode": "emd_dynamic_nominal",
        })
        return sample
