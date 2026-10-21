"""项目源码模块 tests/test_camera_model.py。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import numpy as np
import torch

from camera_model import CameraModel
from utils import get_flow_zforward


def test_camera_model_skew_projection_and_round_trip():
    """验证回归场景 `test_camera_model_skew_projection_and_round_trip`，确保相关数据、几何或入口行为保持预期。

    参数:
        无。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    parameters = torch.tensor(
        [731.392029, 731.015015, 812.549011, 653.577026, -0.085165])
    camera = CameraModel().set_intrinsics(parameters)
    points = torch.tensor([
        [1.0, -0.5],
        [0.25, 0.75],
        [5.0, 8.0],
        [1.0, 1.0],
    ])

    uv, depth, _, _ = camera.project_pytorch(points, [1296, 1600, 3])
    expected_u = (
        parameters[0] * points[0] / points[2]
        + parameters[4] * points[1] / points[2]
        + parameters[2]
    )
    assert torch.allclose(uv[0], expected_u)

    flow, reconstructed, valid = get_flow_zforward(
        uv.t(), depth, torch.eye(4), camera, [1296, 1600, 3],
        scale_flow=False, get_valid_indexes=True)
    assert torch.all(valid)
    assert torch.allclose(flow, torch.zeros_like(flow), atol=1e-5)
    assert torch.allclose(reconstructed[:, :3], points.t()[:, :3], atol=1e-5)

    matrix = camera.get_matrix()
    assert matrix[0, 1] == np.float64(parameters[4])
