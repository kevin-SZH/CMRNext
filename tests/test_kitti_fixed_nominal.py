"""项目源码模块 tests/test_kitti_fixed_nominal.py。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import datasets.DatasetExtrinsicCalib as dataset_module
from datasets.DatasetExtrinsicCalib import DatasetGeneralExtrinsicCalib
from utils import quat2mat, tvector2mat


def test_kitti_fixed_nominal_uses_target_camera_without_random_perturbation(
        tmp_path, monkeypatch):
    """验证回归场景 `test_kitti_fixed_nominal_uses_target_camera_without_random_perturbation`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。
        monkeypatch: pytest 提供的属性、环境变量和函数替换工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    sequence = tmp_path / 'kitti_odometry' / 'sequences' / '00'
    image_dir = sequence / 'image_3'
    cloud_dir = sequence / 'velodyne'
    image_dir.mkdir(parents=True)
    cloud_dir.mkdir()

    Image.new('RGB', (8, 4), color=(10, 20, 30)).save(
        image_dir / '000000.png')
    points = np.array([
        [2.0, 0.5, 8.0, 0.25],
        [3.0, -0.5, 10.0, 0.75],
    ], dtype=np.float32)
    points.tofile(cloud_dir / '000000.bin')

    nominal = np.eye(4, dtype=np.float64)
    nominal[:3, 3] = [0.2, -0.1, 0.3]
    angle = np.deg2rad(5.0)
    residual = np.eye(4, dtype=np.float64)
    residual[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    residual[:3, 3] = [-0.53, 0.02, -0.01]
    true_extrinsic = residual @ nominal

    intrinsic_2 = np.array([
        [707.0, 0.0, 601.0],
        [0.0, 707.0, 183.0],
        [0.0, 0.0, 1.0],
    ])
    intrinsic_3 = intrinsic_2.copy()
    intrinsic_3[0, 2] = 602.0
    fake_kitti = SimpleNamespace(calib=SimpleNamespace(
        T_cam2_velo=nominal,
        T_cam3_velo=true_extrinsic,
        K_cam2=intrinsic_2,
        K_cam3=intrinsic_3,
    ))
    monkeypatch.setattr(
        dataset_module.pykitti, 'odometry',
        lambda base_folder, sequence_name: fake_kitti)

    dataset = DatasetGeneralExtrinsicCalib(
        sequence,
        train=False,
        dataset='kitti',
        cam='3',
        nominal_cam='2',
        max_t=99.0,
        max_r=99.0,
    )
    sample = dataset[0]
    input_from_target = (
        tvector2mat(sample['tr_error']) @ quat2mat(sample['rot_error']))
    raw_homogeneous = torch.from_numpy(points.copy()).t()
    raw_homogeneous[3] = 1.0

    assert sample['dataset_mode'] == 'kitti_fixed_nominal'
    assert torch.allclose(
        sample['residual'] @ sample['nominal_extrinsic'],
        sample['cam2vel'], atol=1e-6)
    assert torch.allclose(input_from_target, torch.linalg.inv(
        sample['residual']), atol=1e-6)
    assert torch.allclose(
        sample['point_cloud'],
        sample['cam2vel'] @ raw_homogeneous,
        atol=1e-5)
    assert torch.allclose(
        input_from_target @ sample['point_cloud'],
        sample['nominal_extrinsic'] @ raw_homogeneous,
        atol=1e-5)
