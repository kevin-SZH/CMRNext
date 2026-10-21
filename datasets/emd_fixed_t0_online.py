"""Per-axis online residual sampling for fixed T0 and per-frame Ti datasets.
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
    """Store per-axis translation and Euler-angle bounds and sampling behavior.
    """

    enabled: bool
    max_translation_component_m: float
    max_rotation_euler_component_deg: float
    train_resample_each_access: bool


def _require_mapping(value, name: str) -> Mapping:
    """Validate a configuration section as a mapping.

    Args:
        value: Input value to parse or validate.
        name: Field name used in validation diagnostics.

    Returns:
        Mapping.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是映射")
    return value


def load_fixed_t0_online_perturbation_config(
    config_path: str | Path,
) -> FixedT0OnlinePerturbationConfig:
    """Load and validate fixed-T0 online perturbation settings.

    Args:
        config_path: Configuration YAML path.

    Returns:
        FixedT0OnlinePerturbationConfig.
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
    """Load and validate per-frame-Ti online perturbation settings.

    Args:
        config_path: Configuration YAML path.

    Returns:
        FixedT0OnlinePerturbationConfig.
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
    """Invert a homogeneous rigid transform using its rotation transpose and translated origin.

    Args:
        transform: Transform: torch.Tensor.

    Returns:
        torch.Tensor.
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
    """Sample independent XYZ translations and XYZ Euler angles to construct a left residual.

    Args:
        translation_xyz: Translation xyz: np.ndarray.
        euler_xyz_deg: Euler xyz deg: np.ndarray.

    Returns:
        torch.Tensor.

    Notes:
        Sample XYZ translation components in meters and XYZ Euler components in degrees independently from symmetric uniform intervals.

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
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
    """Sample online residuals from scenes with Ti equal to fixed T0 and identity stored residuals.


    Notes:
        Require Ti = stored_T0 and identity stored residuals. Construct T_input = inverse(delta) @ Ti. Training resamples on access; evaluation uses a deterministic seed and sample index.

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
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
        """Initialize DatasetEMDFixedT0OnlineExtrinsicCalib from the supplied settings.

        Args:
            scene_paths: Explicit scene directories in loading order.
            level: Configured curriculum or perturbation level.
            contract: Sensor, file layout, preprocessing, and extrinsic conventions.
            online_config: Per-axis online perturbation settings.
            seed: Explicit seed for reproducible random sampling.
            train: Train: bool.
            normalize_images: Normalize images: bool.
            use_reflectance: Use reflectance: bool.
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
        """Select the sample random generator, resampling during training and using seed/index determinism for evaluation.

        Args:
            idx: Idx: int.
        """
        if self.train and self.online_config.train_resample_each_access:
            return np.random
        seed = np.random.SeedSequence([
            self.seed,
            int(idx),
        ])
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        """Load and prepare one sample for DatasetEMDFixedT0OnlineExtrinsicCalib.

        Args:
            idx: Idx: int.

        Returns:
            dict.
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
    """Preserve each recorded Ti and sample a new input transform independently of stored T0 and residuals.


    Notes:
        Construct T_input = inverse(delta) @ Ti without using the stored nominal transform or stored residual to generate the input.

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
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
        """Initialize DatasetEMDPerTiOnlineExtrinsicCalib from the supplied settings.

        Args:
            scene_paths: Explicit scene directories in loading order.
            level: Configured curriculum or perturbation level.
            contract: Sensor, file layout, preprocessing, and extrinsic conventions.
            online_config: Per-axis online perturbation settings.
            seed: Explicit seed for reproducible random sampling.
            train: Train: bool.
            normalize_images: Normalize images: bool.
            use_reflectance: Use reflectance: bool.
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
        """Select the sample random generator, resampling during training and using seed/index determinism for evaluation.

        Args:
            idx: Idx: int.
        """
        if self.train and self.online_config.train_resample_each_access:
            return np.random
        seed = np.random.SeedSequence([self.seed, int(idx)])
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        """Load and prepare one sample for DatasetEMDPerTiOnlineExtrinsicCalib.

        Args:
            idx: Idx: int.

        Returns:
            dict.
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
