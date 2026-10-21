"""Regression tests for kitti sampled perturbation.
"""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import datasets.DatasetExtrinsicCalib as dataset_module
from datasets.DatasetExtrinsicCalib import (
    DatasetGeneralExtrinsicCalib,
    compute_kitti_sampled_initial_state,
)
from utils import rotate_forward


def test_kitti_original_sampling_builds_actual_left_perturbed_t0(
        tmp_path, monkeypatch):
    """Verify that kitti original sampling builds actual left perturbed t0.

    Args:
        tmp_path: Tmp path.
        monkeypatch: Monkeypatch.
    """
    sequence = tmp_path / "kitti_odometry" / "sequences" / "00"
    image_dir = sequence / "image_2"
    cloud_dir = sequence / "velodyne"
    image_dir.mkdir(parents=True)
    cloud_dir.mkdir()

    Image.new("RGB", (8, 4), color=(10, 20, 30)).save(
        image_dir / "000000.png")
    points = np.array([
        [2.0, 0.5, 8.0, 0.25],
        [3.0, -0.5, 10.0, 0.75],
    ], dtype=np.float32)
    points.tofile(cloud_dir / "000000.bin")

    true_extrinsic = np.eye(4, dtype=np.float64)
    true_extrinsic[:3, 3] = [0.2, -0.1, 0.3]
    intrinsic = np.array([
        [707.0, 0.0, 601.0],
        [0.0, 707.0, 183.0],
        [0.0, 0.0, 1.0],
    ])
    fake_kitti = SimpleNamespace(calib=SimpleNamespace(
        T_cam2_velo=true_extrinsic,
        K_cam2=intrinsic,
    ))
    monkeypatch.setattr(
        dataset_module.pykitti, "odometry",
        lambda base_folder, sequence_name: fake_kitti)

    uniform_calls = []

    def midpoint_uniform(lower, upper):
        """Midpoint uniform.

        Args:
            lower: Lower.
            upper: Upper.
        """
        uniform_calls.append((float(lower), float(upper)))
        return (float(lower) + float(upper)) / 2.0

    monkeypatch.setattr(dataset_module.np.random, "uniform", midpoint_uniform)

    dataset = DatasetGeneralExtrinsicCalib(
        sequence,
        train=False,
        dataset="kitti",
        cam="2",
        max_t=1.5,
        max_r=20.0,
    )
    sample = dataset[0]
    initial_extrinsic, residual = compute_kitti_sampled_initial_state(
        sample["cam2vel"], sample["rot_error"], sample["tr_error"])

    raw_homogeneous = torch.from_numpy(points.copy()).t()
    raw_homogeneous[3] = 1.0
    misaligned_points = rotate_forward(sample["point_cloud"], residual)

    assert uniform_calls == [
        (-20.0, 20.0),
        (-20.0, 20.0),
        (-20.0, 20.0),
        (-1.5, 1.5),
        (-1.5, 1.5),
        (-1.5, 1.0),
    ]
    assert torch.allclose(
        residual @ initial_extrinsic, sample["cam2vel"], atol=1e-6)
    assert torch.allclose(
        misaligned_points,
        initial_extrinsic @ raw_homogeneous,
        atol=1e-5,
    )
