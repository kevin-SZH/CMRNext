"""项目源码模块 tests/test_kitti_sampled_perturbation.py。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
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
    """验证回归场景 `test_kitti_original_sampling_builds_actual_left_perturbed_t0`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。
        monkeypatch: pytest 提供的属性、环境变量和函数替换工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
        """完成 tests/test_kitti_sampled_perturbation.py 中的 `midpoint_uniform` 操作。

        参数:
            lower: 替身均匀采样器区间下界。
            upper: 替身均匀采样器区间上界。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。
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
