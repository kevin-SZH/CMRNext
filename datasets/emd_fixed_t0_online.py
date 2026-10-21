"""固定外参或逐帧真实外参场景的原版逐轴在线扰动。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mathutils
import numpy as np
import torch
import yaml

from datasets.emd_extrinsic_calib import DatasetEMDExtrinsicCalib, EMDContract
from utils import quaternion_from_matrix


@dataclass(frozen=True)
class FixedT0OnlinePerturbationConfig:
    """封装 固定外参场景的原版逐轴在线扰动 中的 `FixedT0OnlinePerturbationConfig` 状态和操作。

    主要职责:
        封装 固定外参场景的原版逐轴在线扰动 中的 `FixedT0OnlinePerturbationConfig` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    enabled: bool
    max_translation_component_m: float
    max_rotation_euler_component_deg: float
    train_resample_each_access: bool


def _require_mapping(value, name: str) -> Mapping:
    """完成 固定外参场景的原版逐轴在线扰动 中的 `_require_mapping` 操作。

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


def load_fixed_t0_online_perturbation_config(
    config_path: str | Path,
) -> FixedT0OnlinePerturbationConfig:
    """读取固定 T0 数据的在线逐轴扰动配置。

    参数:
        config_path: YAML 配置路径；平移单位为米，旋转单位为度。

    返回值:
        是否启用、逐轴平移上限、逐轴欧拉角上限和训练期重采样开关。

    异常和副作用:
        采样器名称或数值范围不合法时抛出异常；只读配置文件。

    坐标系和外参约定:
        配置描述左乘残差 delta，输入外参满足 T_input = inverse(delta) @ Ti。
    """
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle), "EMD config")
    section = raw.get("fixed_t0_online_perturbation")
    if section is None:
        return FixedT0OnlinePerturbationConfig(
            False, 0.0, 0.0, False)
    section = _require_mapping(section, "fixed_t0_online_perturbation")
    enabled = bool(section.get("enabled", False))
    if not enabled:
        return FixedT0OnlinePerturbationConfig(
            False, 0.0, 0.0, False)

    sampler = str(section.get("sampler", ""))
    if sampler != "cmrnext_legacy_per_axis_box_v1":
        raise ValueError(
            "固定 T0 采样器必须为 cmrnext_legacy_per_axis_box_v1")
    max_translation = float(section["max_translation_component_m"])
    max_rotation = float(section["max_rotation_euler_component_deg"])
    if not np.isfinite(max_translation) or max_translation <= 0.0:
        raise ValueError("固定 T0 的平移分量上限必须为正数")
    if not np.isfinite(max_rotation) or not 0.0 < max_rotation < 90.0:
        raise ValueError("固定 T0 的欧拉角分量上限必须在 0 到 90 度之间")
    train_resample = bool(section.get("train_resample_each_access", True))
    return FixedT0OnlinePerturbationConfig(
        enabled=True,
        max_translation_component_m=max_translation,
        max_rotation_euler_component_deg=max_rotation,
        train_resample_each_access=train_resample,
    )


def load_per_ti_online_perturbation_config(
    config_path: str | Path,
) -> FixedT0OnlinePerturbationConfig:
    """读取以逐帧真实外参 Ti 为基准的在线逐轴扰动配置。

    参数:
        config_path: YAML 配置路径；逐轴平移上限单位为米，逐轴欧拉角上限
            单位为度。

    返回值:
        是否启用、逐轴平移上限、逐轴欧拉角上限和训练期重采样开关。

    异常和副作用:
        采样器名称、数据来源或数值范围不合法时抛出异常；本函数只读配置文件。

    坐标系和外参约定:
        配置直接采样左乘残差 delta，并对每帧 Ti 构造
        T_input = inverse(delta) @ Ti；存储 T0 和存储 residual 不参与该构造。
    """
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle), "EMD config")
    section = raw.get("per_ti_online_perturbation")
    if section is None:
        return FixedT0OnlinePerturbationConfig(False, 0.0, 0.0, False)
    section = _require_mapping(section, "per_ti_online_perturbation")
    enabled = bool(section.get("enabled", False))
    if not enabled:
        return FixedT0OnlinePerturbationConfig(False, 0.0, 0.0, False)

    dataset = _require_mapping(raw.get("dataset"), "dataset")
    source_mode = str(
        dataset.get("source_mode", "cmrnext_stored_residuals_v1"))
    if source_mode != "calibnet_stored_residuals_v1":
        raise ValueError(
            "per-Ti 在线扰动要求 dataset.source_mode 为 "
            "calibnet_stored_residuals_v1")
    sampler = str(section.get("sampler", ""))
    if sampler != "cmrnext_legacy_per_axis_box_v1":
        raise ValueError(
            "per-Ti 采样器必须为 cmrnext_legacy_per_axis_box_v1")
    max_translation = float(section["max_translation_component_m"])
    max_rotation = float(section["max_rotation_euler_component_deg"])
    if not np.isfinite(max_translation) or max_translation <= 0.0:
        raise ValueError("per-Ti 逐轴平移上限必须为正数")
    if not np.isfinite(max_rotation) or not 0.0 < max_rotation < 90.0:
        raise ValueError("per-Ti 逐轴欧拉角上限必须在 (0, 90) 度范围内")
    train_resample = bool(section.get("train_resample_each_access", True))
    return FixedT0OnlinePerturbationConfig(
        enabled=True,
        max_translation_component_m=max_translation,
        max_rotation_euler_component_deg=max_rotation,
        train_resample_each_access=train_resample,
    )


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


def _per_axis_residual(
    translation_xyz: np.ndarray,
    euler_xyz_deg: np.ndarray,
) -> torch.Tensor:
    """由逐轴 xyz 平移和 xyz 欧拉角构造原版扰动残差矩阵。

    参数:
        translation_xyz: 按 x、y、z 排列的逐轴平移残差，单位为米。
        euler_xyz_deg: 角度量或角度上限，单位为度。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    rotation = mathutils.Euler(
        tuple(math.radians(float(value)) for value in euler_xyz_deg),
        "XYZ",
    ).to_matrix()
    residual = np.eye(4, dtype=np.float64)
    residual[:3, :3] = np.asarray(rotation, dtype=np.float64)
    residual[:3, 3] = translation_xyz
    return torch.from_numpy(residual)


class DatasetEMDFixedT0OnlineExtrinsicCalib(DatasetEMDExtrinsicCalib):
    """从 Ti 等于固定 T0 的场景在线采样原版逐轴扰动输入。

    主要职责:
        从 Ti 等于固定 T0 的场景在线采样原版逐轴扰动输入，并集中维护该职责所需状态。

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
        online_config: FixedT0OnlinePerturbationConfig,
        seed: int,
        train: bool = True,
        normalize_images: bool = True,
        use_reflectance: bool = False,
    ):
        """初始化固定 T0 在线扰动数据集。

        参数:
            scene_paths: 直接读取的固定 Ti=T0 场景目录列表。
            level: 配置声明的 fixed_baseline level。
            contract: 传感器和 nominal_to_true_residual_v1 数据契约。
            online_config: 逐轴平移米制范围、欧拉角度范围和训练重采样方式。
            seed: 验证或测试时按样本生成确定性扰动的显式随机种子。
            train: 是否按训练方式每次访问重新采样；不参与场景选择。
            normalize_images: 是否执行 CMRNext 图像归一化。
            use_reflectance: 必须为 False。

        返回值:
            无显式返回值；实例按访问或样本索引生成输入外参。

        异常和副作用:
            固定外参、单位残差或扰动配置不合法时抛出异常。训练读取会消耗
            NumPy 随机状态；评估读取仅由 seed 和样本索引决定。

        坐标系和外参约定:
            采样左乘残差 delta 后构造 T_input = inverse(delta) @ Ti，保证
            delta @ T_input == Ti；点云和外参均采用列向量约定。
        """
        if contract.source_mode not in {
                "calibnet_fixed_t0_v1", "recorded_pose_v1"}:
            raise ValueError(
                "固定 T0 在线扰动只支持 calibnet_fixed_t0_v1 或 "
                "recorded_pose_v1 数据来源")
        if not online_config.enabled:
            raise ValueError("fixed-T0 online perturbation is not enabled")
        if train and not online_config.train_resample_each_access:
            raise ValueError(
                "固定 T0 训练要求 train_resample_each_access=true")
        super().__init__(
            scene_paths=scene_paths,
            level=level,
            contract=contract,
            train=train,
            normalize_images=normalize_images,
            use_reflectance=use_reflectance,
        )
        self.online_config = online_config
        self.seed = int(seed)
        identity = torch.eye(4, dtype=torch.float32)
        reference_t0 = self.sequences[0].nominal_extrinsic
        for sequence in self.sequences:
            if not torch.allclose(
                    sequence.nominal_extrinsic, reference_t0,
                    atol=contract.transform_atol):
                raise ValueError(
                    f"{sequence.directory}: nominal extrinsic differs across scenes")
            if not torch.allclose(
                    sequence.residuals, identity, atol=contract.transform_atol):
                raise ValueError(
                    f"{sequence.directory}: fixed-T0 source residuals are not identity")
            expected_true = sequence.nominal_extrinsic.expand_as(
                sequence.true_extrinsics)
            if not torch.allclose(
                    sequence.true_extrinsics, expected_true,
                    atol=contract.transform_atol):
                raise ValueError(f"{sequence.directory}: fixed-T0 source has Ti != T0")

    def _rng(self, idx: int):
        """返回当前样本使用的随机数生成器。

        参数:
            idx: 数据集全局样本索引，无单位。

        返回值:
            训练时返回进程级 NumPy 随机模块，评估时返回由 seed 和 idx 派生的生成器。

        异常和副作用:
            训练分支会推进进程级随机状态；评估分支无共享状态副作用。

        坐标系和外参约定:
            本函数只负责随机数，不改变外参坐标系。
        """
        if self.train and self.online_config.train_resample_each_access:
            return np.random
        seed = np.random.SeedSequence([
            self.seed,
            int(idx),
        ])
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        """读取固定外参样本并构造在线扰动后的输入外参。

        参数:
            idx: 数据集全局样本索引，无单位。

        返回值:
            更新后的 CMRNext 样本字典，包含采样平移米值、欧拉角度值、
            T_input、Ti 和满足重建关系的左乘残差。

        异常和副作用:
            residual @ T_input 无法在容差内重建 Ti 时抛出异常。训练访问推进
            NumPy 随机状态，评估访问由 seed 和 idx 确定。

        坐标系和外参约定:
            T_input = inverse(delta) @ Ti，且 delta @ T_input == Ti。
        """
        sample = super().__getitem__(idx)
        rng = self._rng(idx)
        translation_xyz = rng.uniform(
            -self.online_config.max_translation_component_m,
            self.online_config.max_translation_component_m,
            size=3,
        )
        euler_xyz_deg = rng.uniform(
            -self.online_config.max_rotation_euler_component_deg,
            self.online_config.max_rotation_euler_component_deg,
            size=3,
        )
        sampled_residual = _per_axis_residual(
            translation_xyz, euler_xyz_deg)

        true_extrinsic = sample["cam2vel"].double()
        source_t0 = sample["nominal_extrinsic"].double()
        input_extrinsic = _rigid_inverse(sampled_residual) @ true_extrinsic
        residual = true_extrinsic @ _rigid_inverse(input_extrinsic)
        reconstruction_error = float(torch.max(torch.abs(
            residual @ input_extrinsic - true_extrinsic)))
        if reconstruction_error > self.contract.transform_atol:
            raise RuntimeError(
                "固定 T0 在线残差重建失败："
                f"{reconstruction_error:.3e}")

        input_from_target = _rigid_inverse(residual).float()
        sample.update({
            "tr_error": input_from_target[:3, 3].clone(),
            "rot_error": quaternion_from_matrix(input_from_target).float(),
            "nominal_extrinsic": input_extrinsic.float(),
            "residual": residual.float(),
            "source_t0_extrinsic": source_t0.float(),
            "sampled_residual": sampled_residual.float(),
            "sampled_translation_xyz_m": torch.tensor(
                translation_xyz, dtype=torch.float32),
            "sampled_euler_xyz_deg": torch.tensor(
                euler_xyz_deg, dtype=torch.float32),
            "effective_max_t": torch.tensor(
                self.online_config.max_translation_component_m,
                dtype=torch.float32),
            "effective_max_r_deg": torch.tensor(
                self.online_config.max_rotation_euler_component_deg,
                dtype=torch.float32),
            "online_perturbation_seed": self.seed,
            "online_perturbation_deterministic": not self.train,
            "dataset_mode": "emd_fixed_t0_online_per_axis",
        })
        return sample


class DatasetEMDPerTiOnlineExtrinsicCalib(
    DatasetEMDExtrinsicCalib,
):
    """对多外参数据的每帧 Ti 独立采样在线残差。

    主要职责:
        保持真实外参 Ti 和目标点云不变，每次训练访问直接采样左乘残差，
        由 Ti 推导输入外参，不使用场景存储的 T0 或 residual。

    返回值:
        类实例由构造方法创建；每个样本包含在线生成的输入外参、监督残差、
        逐轴采样值和原始 Ti。

    异常和副作用:
        数据来源或配置不合法时抛出异常；训练读取会推进 NumPy 随机状态。

    坐标系和数据约定:
        点云目标为 Ti @ p_lidar；在线采样 delta 后，输入为
        T_input @ p_lidar，其中 T_input = inverse(delta) @ Ti，且
        delta @ T_input == Ti。平移单位为米，欧拉角单位为度。
    """

    def __init__(
        self,
        scene_paths: Sequence[str | Path],
        *,
        level: str,
        contract: EMDContract,
        online_config: FixedT0OnlinePerturbationConfig,
        seed: int,
        train: bool = True,
        normalize_images: bool = True,
        use_reflectance: bool = False,
    ):
        """初始化多 Ti 在线逐轴扰动数据集。

        参数:
            scene_paths: 用户显式指定的多外参场景目录列表。
            level: 当前 curriculum level，仅用于样本审计记录。
            contract: 相机、点云和 nominal_to_true_residual_v1 数据契约。
            online_config: 逐轴平移米制范围、欧拉角度范围和重采样方式。
            seed: 非训练读取时按样本派生确定性随机数的基础种子。
            train: 是否每次访问重新采样；正式训练必须为 True。
            normalize_images: 是否按 CMRNext 约定归一化图像。
            use_reflectance: 必须为 False，因为当前 EMD 强度为占位值。

        返回值:
            无显式返回值；实例建立场景索引并保存在线扰动配置。

        异常和副作用:
            非多 Ti CalibNet 数据、未启用配置或训练未开启重采样时抛出异常；
            初始化只读取场景文件，不采样扰动。

        坐标系和外参约定:
            每帧以 Ti 为唯一外参基准，输入外参由
            T_input = inverse(delta) @ Ti 构造，与存储 T0 无关。
        """
        if contract.source_mode != "calibnet_stored_residuals_v1":
            raise ValueError(
                "多 Ti 在线扰动只支持 calibnet_stored_residuals_v1 数据来源")
        if not online_config.enabled:
            raise ValueError("逐帧 Ti 在线扰动配置未启用")
        if train and not online_config.train_resample_each_access:
            raise ValueError(
                "多 Ti 在线训练要求 train_resample_each_access=true")
        DatasetEMDExtrinsicCalib.__init__(
            self,
            scene_paths=scene_paths,
            level=level,
            contract=contract,
            train=train,
            normalize_images=normalize_images,
            use_reflectance=use_reflectance,
        )
        self.online_config = online_config
        self.seed = int(seed)

    def _rng(self, idx: int):
        """返回当前逐帧 Ti 样本使用的随机数生成器。

        参数:
            idx: 数据集全局样本索引，无单位。

        返回值:
            训练时返回进程级 NumPy 随机模块；非训练时返回由 seed 和 idx
            派生的独立生成器。

        异常和副作用:
            训练分支会推进进程级随机状态；非训练分支无共享状态副作用。

        坐标系和外参约定:
            本函数只负责随机数，不读取或改变 Ti、T0 和点云坐标系。
        """
        if self.train and self.online_config.train_resample_each_access:
            return np.random
        seed = np.random.SeedSequence([self.seed, int(idx)])
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        """读取一帧并仅围绕该帧 Ti 在线生成扰动输入。

        参数:
            idx: 数据集全局样本索引，无单位。

        返回值:
            CMRNext 样本字典；nominal_extrinsic 是在线生成的 T_input，
            residual 与 sampled_residual 相同，均为在线采样的 delta。

        异常和副作用:
            delta @ T_input 不能在容差内重建 Ti 时抛出异常；训练读取会推进
            NumPy 随机状态。

        坐标系和外参约定:
            目标点云保持 Ti @ p_lidar；网络输入通过 inverse(delta) 变换为
            T_input @ p_lidar，其中 delta @ T_input == Ti。
        """
        sample = DatasetEMDExtrinsicCalib.__getitem__(self, idx)
        rng = self._rng(idx)
        # Sample each residual axis from this mode's limits.
        translation_xyz = rng.uniform(
            -self.online_config.max_translation_component_m,
            self.online_config.max_translation_component_m,
            size=3,
        )
        euler_xyz_deg = rng.uniform(
            -self.online_config.max_rotation_euler_component_deg,
            self.online_config.max_rotation_euler_component_deg,
            size=3,
        )
        sampled_residual = _per_axis_residual(
            translation_xyz, euler_xyz_deg)

        # Derive the input from Ti, ignoring stored T0 and residual.
        true_extrinsic = sample["cam2vel"].double()
        input_extrinsic = _rigid_inverse(sampled_residual) @ true_extrinsic
        reconstruction_error = float(torch.max(torch.abs(
            sampled_residual @ input_extrinsic - true_extrinsic)))
        if reconstruction_error > self.contract.transform_atol:
            raise RuntimeError(
                "逐帧 Ti 在线残差无法重建真实外参: "
                f"{reconstruction_error:.3e}")

        input_from_target = _rigid_inverse(sampled_residual).float()
        sample.update({
            "tr_error": input_from_target[:3, 3].clone(),
            "rot_error": quaternion_from_matrix(input_from_target).float(),
            "nominal_extrinsic": input_extrinsic.float(),
            "residual": sampled_residual.float(),
            "sampled_residual": sampled_residual.float(),
            "sampled_translation_xyz_m": torch.tensor(
                translation_xyz, dtype=torch.float32),
            "sampled_euler_xyz_deg": torch.tensor(
                euler_xyz_deg, dtype=torch.float32),
            "effective_max_t": torch.tensor(
                self.online_config.max_translation_component_m,
                dtype=torch.float32),
            "effective_max_r_deg": torch.tensor(
                self.online_config.max_rotation_euler_component_deg,
                dtype=torch.float32),
            "online_perturbation_seed": self.seed,
            "online_perturbation_deterministic": not self.train,
            "dataset_mode": "emd_per_ti_online_per_axis",
        })
        return sample
