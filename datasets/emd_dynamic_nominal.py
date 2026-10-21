"""Online nominal-extrinsic augmentation around stored scene calibration.
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
    """Store scale-relative translation and rotation bounds for one nominal-augmentation level.


    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """

    translation_alpha_t: float
    translation_min_floor_m: float
    max_rotation_deg: float


@dataclass(frozen=True)
class EMDDynamicNominalConfig:
    """Store whether online nominal augmentation is enabled and its configured level bounds.


    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """

    enabled: bool
    levels: Mapping[str, DynamicNominalLevelConfig]


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


def load_emd_dynamic_nominal_config(
    config_path: str | Path,
) -> EMDDynamicNominalConfig:
    """Load and validate dynamic nominal perturbation settings and curriculum-level consistency.

    Args:
        config_path: Configuration YAML path.

    Returns:
        EMDDynamicNominalConfig.

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
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
    """Compute the magnitude of a rotation in degrees.

    Args:
        transform: Transform: torch.Tensor.

    Returns:
        float.
    """
    cosine = ((torch.trace(transform[:3, :3]) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


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


def _sample_uniform_ball(max_radius: float) -> np.ndarray:
    """Sample a translation uniformly within a three-dimensional ball.

    Args:
        max_radius: Max radius: float.

    Returns:
        np.ndarray.
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
    """Sample a bounded translation and random-axis rotation as a left perturbation.

    Args:
        max_translation_m: Max translation m: float.
        max_rotation_deg: Max rotation deg: float.

    Returns:
        torch.Tensor.

    Notes:
        The translation radius is max(alpha_t * ||t(Ti)||, min_floor_m). Sample translation uniformly in the ball and rotation about a random axis.
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
    """Jitter stored T0 online while preserving Ti and recomputing the left residual.


    Notes:
        Use T_input = jitter @ stored_T0 and recompute delta = Ti @ inverse(T_input). Ti remains unchanged; stored-residual evaluation does not resample this training jitter.

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
        dynamic_config: EMDDynamicNominalConfig,
        train: bool = True,
        normalize_images: bool = True,
        use_reflectance: bool = False,
    ):
        """Initialize DatasetEMDDynamicNominalExtrinsicCalib from the supplied settings.

        Args:
            scene_paths: Explicit scene directories in loading order.
            level: Configured curriculum or perturbation level.
            contract: Sensor, file layout, preprocessing, and extrinsic conventions.
            dynamic_config: Nominal-extrinsic online augmentation settings.
            train: Train: bool.
            normalize_images: Normalize images: bool.
            use_reflectance: Use reflectance: bool.
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
        """Load and prepare one sample for DatasetEMDDynamicNominalExtrinsicCalib.

        Args:
            idx: Idx: int.

        Returns:
            dict.
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
