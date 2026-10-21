"""四元数、旋转矩阵和旋转距离计算。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import math

import numpy as np
import torch


def quatmultiply(q, r, device='cpu'):
    """完成 四元数、旋转矩阵和旋转距离计算 中的 `quatmultiply` 操作。

    参数:
        q: 第一个四元数张量，分量顺序遵循当前模块实现。
        r: 第二个四元数张量，分量顺序遵循当前模块实现。
        device: PyTorch 计算设备，例如 CUDA 设备或 CPU。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if isinstance(q, torch.Tensor):
        t = torch.zeros(q.shape[0], 4, device=device)
    elif isinstance(q, np.ndarray):
        t = np.zeros(q.shape[0], 4)
    else:
        raise TypeError("Type not supported")
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    t[:, 2] = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    t[:, 3] = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    return t


def quatinv(q):
    """完成 四元数、旋转矩阵和旋转距离计算 中的 `quatinv` 操作。

    参数:
        q: 第一个四元数张量，分量顺序遵循当前模块实现。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if isinstance(q, torch.Tensor):
        t = q.clone()
    elif isinstance(q, np.ndarray):
        t = q.copy()
    else:
        raise TypeError("Type not supported")
    t *= -1
    t[:, 0] *= -1
    return t


def quaternion_loss(q, r, device):
    """完成 四元数、旋转矩阵和旋转距离计算 中的 `quaternion_loss` 操作。

    参数:
        q: 第一个四元数张量，分量顺序遵循当前模块实现。
        r: 第二个四元数张量，分量顺序遵循当前模块实现。
        device: PyTorch 计算设备，例如 CUDA 设备或 CPU。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    t = quatmultiply(q, quatinv(r), device)
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))


def rotation_from_quaternion(x):
    """完成 四元数、旋转矩阵和旋转距离计算 中的 `rotation_from_quaternion` 操作。

    参数:
        x: 当前层输入张量；通道和空间布局由所在网络模块定义。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    # See <http://en.wikipedia.org/w/index.php?title=
    # Quaternions_and_spatial_rotation&oldid=402924915>
    a, b, c, d = x

    r1 = [a ** 2 + b ** 2 - c ** 2 - d ** 2,
          2 * b * c - 2 * a * d,
          2 * b * d + 2 * a * c]
    r2 = [2 * b * c + 2 * a * d,
          a ** 2 - b ** 2 + c ** 2 - d ** 2,
          2 * c * d - 2 * a * b]
    r3 = [2 * b * d - 2 * a * c,
          2 * c * d + 2 * a * b,
          a ** 2 - b ** 2 - c ** 2 + d ** 2]

    return np.array([r1, r2, r3])


def axis_angle_from_rotation(R):
    """完成 四元数、旋转矩阵和旋转距离计算 中的 `axis_angle_from_rotation` 操作。

    参数:
        R: 旋转矩阵、四元数或旋转对象；表示形式由调用点决定。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    angle = np.arccos(np.clip(((R.trace() - 1) / 2), -1.0, 1.0))

    if angle == 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    else:
        v = np.array([R[2, 1] - R[1, 2],
                   R[0, 2] - R[2, 0],
                   R[1, 0] - R[0, 1]])

        computer_with_infinite_precision = False
        if computer_with_infinite_precision:
            axis = (1 / (2 * np.sin(angle))) * v
        else:
            # Use the stable form for small angles.
            axis = (v * np.sign(angle)) / np.linalg.norm(v)

        return axis, angle


def geodesic_distance_for_rotations(R1, R2):
    """完成 四元数、旋转矩阵和旋转距离计算 中的 `geodesic_distance_for_rotations` 操作。

    参数:
        R1: 第一组旋转矩阵，形状为 B×3×3。
        R2: 第二组旋转矩阵，形状为 B×3×3。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    R = np.dot(R1, R2.T)
    axis1, angle1 = axis_angle_from_rotation(R)  # Kept for API symmetry.
    return angle1
