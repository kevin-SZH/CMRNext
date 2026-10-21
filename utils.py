"""点云变换、外参表示、标定光流和训练公共工具。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import argparse
from functools import reduce

import cv2
import mathutils
import torch
import torch.nn.functional as F
import liegroups
import logging
import math
import numpy as np
import scipy
import visibility
from matplotlib import cm
from torch.utils.data.dataloader import default_collate
from torch_scatter import scatter_mean
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid

try:
    import geomstats.geometry.lie_group as lie_group
except ImportError:
    import geomstats.lie_group as lie_group


def init_logger(path, resume=False, save_to_file=False):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `init_logger` 操作。

    参数:
        path: 输入或输出文件路径。
        resume: 是否以追加方式继续写入已有日志文件。
        save_to_file: 当前操作读取或写入的文件路径。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    # Configure console logging.
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Configure file logging when requested.
    if save_to_file:
        mode = 'w'
        if resume:
            mode = 'a'
        file_handler = logging.FileHandler(path, mode=mode)
        file_formatter = logging.Formatter(fmt="%(levelname).1s - %(asctime)s - %(name)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('argoverse').setLevel(logging.ERROR)
    return logger


def rotate_points(PC, R, T=None, inverse=True):
    """使用 mathutils 表示的位姿对齐次点云执行正向或逆向刚体变换。

    参数:
        PC: 齐次点云，形状为 4×N 或 N×4，平移单位为米。
        R: 旋转矩阵、四元数或旋转对象；表示形式由调用点决定。
        T: 三维平移向量或齐次平移矩阵，单位为米。
        inverse: 是否应用输入刚体变换的逆变换。

    返回值:
        施加指定正向或逆向刚体变换后的齐次点云，布局与输入兼容。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        try:
            RT = T @ R
        except:
            RT = T * R
    else:
        RT = R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=PC.dtype)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    """使用 PyTorch 四元数、平移或四乘四矩阵变换齐次点云。

    参数:
        PC: 齐次点云，形状为 4×N 或 N×4，平移单位为米。
        R: 旋转矩阵、四元数或旋转对象；表示形式由调用点决定。
        T: 三维平移向量或齐次平移矩阵，单位为米。
        inverse: 是否应用输入刚体变换的逆变换。

    返回值:
        施加指定正向或逆向刚体变换后的 PyTorch 齐次点云。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_forward(PC, R, T=None):
    """把点云变换为指定新位姿下观察到的坐标；历史命名中的 forward 实际调用逆变换。

    参数:
        PC: 齐次点云，形状为 4×N 或 N×4，平移单位为米。
        R: 旋转矩阵、四元数或旋转对象；表示形式由调用点决定。
        T: 三维平移向量或齐次平移矩阵，单位为米。

    返回值:
        指定新位姿下观察到的齐次点云。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """撤销 rotate_forward 对点云施加的位姿变化。

    参数:
        PC_ROTATED: 已经施加位姿变化的齐次点云，形状为 4×N 或 N×4。
        R: 旋转矩阵、四元数或旋转对象；表示形式由调用点决定。
        T: 三维平移向量或齐次平移矩阵，单位为米。

    返回值:
        撤销位姿变化后的齐次点云。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `invert_pose` 操作。

    参数:
        R: 旋转矩阵、四元数或旋转对象；表示形式由调用点决定。
        T: 三维平移向量或齐次平移矩阵，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    try:
        RT = T @ R
    except:
        RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def merge_inputs(queries):
    """将可变点数点云样本合并为批次，同时对定长字段执行默认堆叠。

    参数:
        queries: 待合并的单样本字典序列。

    返回值:
        批次字典；可变点数点云和图像保留为列表，其余字段按默认规则堆叠。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    point_clouds = []
    imgs = []
    reflectances = []
    car_masks = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb' and key != 'reflectance' and key != 'car_mask'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
        if 'car_mask' in input:
            car_masks.append(input['car_mask'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    if len(car_masks) > 0:
        returns['car_mask'] = car_masks
    return returns


def quaternion_from_matrix(matrix):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `quaternion_from_matrix` 操作。

    参数:
        matrix: 三乘三或四乘四旋转矩阵，用于转换为单位四元数。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def euler2mat(z, y, x):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `euler2mat` 操作。

    参数:
        z: 绕 z 轴的欧拉旋转角，单位为弧度。
        y: 绕 y 轴的欧拉旋转角，单位为弧度。
        x: 当前层输入张量；通道和空间布局由所在网络模块定义。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def euler2quat(roll, pitch, yaw):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `euler2quat` 操作。

    参数:
        roll: 绕 x 轴的滚转角，单位为弧度。
        pitch: 绕 y 轴的俯仰角，单位为弧度。
        yaw: 绕 z 轴的偏航角，单位为弧度。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    q = np.zeros(4)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr
    return q


def quatmultiply(q, r):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `quatmultiply` 操作。

    参数:
        q: 第一个四元数张量，分量顺序遵循当前模块实现。
        r: 第二个四元数张量，分量顺序遵循当前模块实现。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `quat2mat` 操作。

    参数:
        q: 第一个四元数张量，分量顺序遵循当前模块实现。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `tvector2mat` 操作。

    参数:
        t: 三维平移向量，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `mat2xyzrpy` 操作。

    参数:
        rotmatrix: 待分解的四乘四齐次外参矩阵。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `to_rotation_matrix` 操作。

    参数:
        R: 旋转矩阵、四元数或旋转对象；表示形式由调用点决定。
        T: 三维平移向量或齐次平移矩阵，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def overlay_imgs(rgb, lidar, idx=0, pooling=3, max_depth=160., close_thr=25.):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `overlay_imgs` 操作。

    参数:
        rgb: RGB 图像张量或数组；通道顺序和数值范围遵循调用方约定。
        lidar: 待叠加到 RGB 图像的单通道深度或 LiDAR 投影图。
        idx: 样本、批次或可视化项索引，从零开始。
        pooling: 可视化投影点向周围扩展的像素半径。
        max_depth: 保留投影点的最大 Z 深度，单位为米。
        close_thr: 近距离点高亮或过滤阈值，单位与深度图一致。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()

    if pooling > 0:
        far_points = lidar.clone()
        far_points[far_points<close_thr] = 0
        close_points = lidar.clone()
        close_points[close_points>=close_thr] = 0.
        close_points[lidar == 0] = 1000.
        close_points = -close_points
        close_points = F.max_pool2d(close_points, pooling, 1, 1)
        close_points = -close_points
        close_points[close_points == 1000.] = 0.
        lidar = close_points+far_points

    lidar = lidar[0][0] / max_depth
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    return blended_img


def get_flow(uv, depth, RT, cam_model, image_shape, scale_flow=True, get_valid_indexes=False, reverse=True):
    """在历史轴向约定下由深度点和刚体变换生成投影像素位移。

    参数:
        uv: 图像像素坐标，形状为 N×2，顺序为 u、v，单位为像素。
        depth: 与投影像素对应的 Z 向深度，单位为米。
        RT: 四乘四齐次刚体变换；方向由函数的坐标约定说明。
        cam_model: 已经设置内参的 CameraModel 实例。
        image_shape: 图像或网络输入尺寸，按高度、宽度排列，单位为像素。
        scale_flow: 是否按旧 PWC-Net 约定将光流数值除以 20。
        get_valid_indexes: 是否在返回结果中附带目标投影仍位于图像内的有效索引。
        reverse: 是否返回目标到源的反向像素位移，而非源到目标的正向位移。

    返回值:
        二维像素位移和变换前的三维点；请求有效索引时额外返回布尔索引。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    points_3D = torch.zeros((uv.shape[0], 4),
                            device=uv.device, dtype=torch.float)

    # Back-project depth in an X-forward frame.
    normalized_v = (uv[:, 1] - cam_model.principal_point[1]) / cam_model.focal_length[1]
    points_3D[:, 1] = (
        uv[:, 0] - cam_model.principal_point[0] - cam_model.skew * normalized_v
    ) * depth / cam_model.focal_length[0]  # Y
    points_3D[:, 2] = normalized_v * depth  # Z
    points_3D[:, 0] = depth
    points_3D[:, 3] = 1.

    pc_rotated = rotate_forward(points_3D.clone(), RT)

    uv_rgb = torch.zeros((uv.shape[0], 2), device=uv.device, dtype=torch.float)
    uv_rgb[:, 0] = (cam_model.focal_length[0] * pc_rotated[:, 1] / pc_rotated[:, 0]
                    + cam_model.skew * pc_rotated[:, 2] / pc_rotated[:, 0]
                    + cam_model.principal_point[0])
    uv_rgb[:, 1] = cam_model.focal_length[1] * pc_rotated[:, 2] / pc_rotated[:, 0] + cam_model.principal_point[1]

    if reverse:
        flow = uv - uv_rgb
    else:
        flow = uv_rgb - uv
    flow[uv_rgb < 0] = 0.
    flow[uv_rgb[:, 0] >= image_shape[1]] = 0.
    flow[uv_rgb[:, 1] >= image_shape[0]] = 0.
    
    valid_index = torch.ones_like(flow[:, 0]).bool()
    valid_index = valid_index & (uv_rgb[:, 0] >= 0)
    valid_index = valid_index & (uv_rgb[:, 1] >= 0)
    valid_index = valid_index & (uv_rgb[:, 0] < image_shape[1])
    valid_index = valid_index & (uv_rgb[:, 1] < image_shape[0])
    
    
    # Remove tiny flow.
    flow[abs(flow)<0.5] = 0

    # Apply PWC-Net's 1/20 flow scale.
    if scale_flow:
        flow /= 20.

    if get_valid_indexes:
        return flow.clone(), points_3D.clone(), valid_index
    else:
        return flow.clone(), points_3D.clone()


def get_flow_zforward(uv, depth, RT, cam_model, image_shape, scale_flow=True, get_valid_indexes=False, reverse=False):
    """在 Z 轴朝前的相机坐标系中反投影深度并生成标定光流监督。

    参数:
        uv: 图像像素坐标，形状为 N×2，顺序为 u、v，单位为像素。
        depth: 与投影像素对应的 Z 向深度，单位为米。
        RT: 四乘四齐次刚体变换；方向由函数的坐标约定说明。
        cam_model: 已经设置内参的 CameraModel 实例。
        image_shape: 图像或网络输入尺寸，按高度、宽度排列，单位为像素。
        scale_flow: 是否按旧 PWC-Net 约定将光流数值除以 20。
        get_valid_indexes: 是否在返回结果中附带目标投影仍位于图像内的有效索引。
        reverse: 是否返回目标到源的反向像素位移，而非源到目标的正向位移。

    返回值:
        二维像素位移和反投影三维点；请求有效索引时额外返回布尔索引。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    points_3D = torch.zeros((uv.shape[0], 4),
                            device=uv.device, dtype=torch.float)

    # Back-project depth in a Z-forward frame.
    normalized_v = (uv[:, 1] - cam_model.principal_point[1]) / cam_model.focal_length[1]
    points_3D[:, 0] = (
        uv[:, 0] - cam_model.principal_point[0] - cam_model.skew * normalized_v
    ) * depth / cam_model.focal_length[0]  # X
    points_3D[:, 1] = normalized_v * depth  # Y
    points_3D[:, 2] = depth
    points_3D[:, 3] = 1.

    pc_rotated = rotate_forward(points_3D.clone(), RT)

    uv_rgb = torch.zeros((uv.shape[0], 2), device=uv.device, dtype=torch.float)
    uv_rgb[:, 0] = (cam_model.focal_length[0] * pc_rotated[:, 0] / pc_rotated[:, 2]
                    + cam_model.skew * pc_rotated[:, 1] / pc_rotated[:, 2]
                    + cam_model.principal_point[0])
    uv_rgb[:, 1] = cam_model.focal_length[1] * pc_rotated[:, 1] / pc_rotated[:, 2] + cam_model.principal_point[1]

    if reverse:
        flow = uv - uv_rgb
    else:
        flow = uv_rgb - uv
    flow[uv_rgb < 0] = 0.
    flow[uv_rgb[:, 0] >= image_shape[1]] = 0.
    flow[uv_rgb[:, 1] >= image_shape[0]] = 0.

    valid_index = torch.ones_like(flow[:, 0]).bool()
    valid_index = valid_index & (uv_rgb[:, 0] >= 0)
    valid_index = valid_index & (uv_rgb[:, 1] >= 0)
    valid_index = valid_index & (uv_rgb[:, 0] < image_shape[1])
    valid_index = valid_index & (uv_rgb[:, 1] < image_shape[0])

    # Remove tiny flow.
    flow[abs(flow) < 0.5] = 0

    # Apply PWC-Net's 1/20 flow scale.
    if scale_flow:
        flow /= 20.

    if get_valid_indexes:
        return flow.clone(), points_3D.clone(), valid_index
    else:
        return flow.clone(), points_3D.clone()


def project_points_pose(xyzw, hyp_pose, cam_mat, real_uv):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `project_points_pose` 操作。

    参数:
        xyzw: Z 轴向前的齐次三维点，形状为 4×N 或 N×4。
        hyp_pose: PnP 假设位姿，包含 Rodrigues 旋转向量和平移。
        cam_mat: 三乘三针孔相机内参矩阵。
        real_uv: 与三维点对应的真实像素坐标，形状为 2×N 或 N×2。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if xyzw.shape[1] == 4 and xyzw.shape[0] != 4:
        xyzw = xyzw.t()
    if real_uv.shape[1] == 2:
        real_uv = real_uv.t()
    rot_mat_zf, _ = cv2.Rodrigues(hyp_pose[1])
    RT_pred = torch.zeros((4, 4), dtype=xyzw.dtype)
    RT_pred[:3, :3] = torch.tensor(rot_mat_zf, dtype=xyzw.dtype)
    RT_pred[:3, 3] = torch.tensor(hyp_pose[2], dtype=xyzw.dtype).squeeze()
    RT_pred[3, 3] = 1.
    RT_pred = RT_pred.to(xyzw.device)
    xyzw = torch.mm(RT_pred, xyzw)
    uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
    uv[0, :] = cam_mat[0, 0] * xyzw[0, :] / xyzw[2, :] + cam_mat[0, 2]
    uv[1, :] = cam_mat[1, 1] * xyzw[1, :] / xyzw[2, :] + cam_mat[1, 2]

    diffs = (uv - real_uv).norm(dim=0)

    return diffs


def downsample_flow_and_mask(flow, mask, downsample_ratio, scale_flow=False):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `downsample_flow_and_mask` 操作。

    参数:
        flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
        mask: 有效像素布尔或数值掩码；非零位置参与计算。
        downsample_ratio: 空间下采样倍率；大于等于 1。
        scale_flow: 是否按旧 PWC-Net 约定将光流数值除以 20。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    downsampled_flow = torch.zeros((flow.shape[0]//downsample_ratio, flow.shape[1]//downsample_ratio, 2),
                                   device=flow.device, dtype=torch.float)
    downsampled_flow = visibility.downsample_flow(flow, downsampled_flow, flow.shape[1]//downsample_ratio,
                                           flow.shape[0]//downsample_ratio, downsample_ratio)

    downsampled_mask = torch.zeros((mask.shape[0]//downsample_ratio, mask.shape[1]//downsample_ratio),
                            device=mask.device, dtype=torch.int)
    downsampled_mask = visibility.downsample_mask(mask, downsampled_mask, mask.shape[1]//downsample_ratio,
                                                  mask.shape[0]//downsample_ratio, downsample_ratio)

    if scale_flow:
        downsampled_flow /= downsample_ratio

    return downsampled_flow, downsampled_mask


def downsample_depth(depth, downsample_ratio):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `downsample_depth` 操作。

    参数:
        depth: 与投影像素对应的 Z 向深度，单位为米。
        downsample_ratio: 空间下采样倍率；大于等于 1。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    downsampled_depth = torch.zeros((depth.shape[0]//downsample_ratio, depth.shape[1]//downsample_ratio, 1),
                                   device=depth.device, dtype=torch.float)
    downsampled_depth = visibility.downsample_depth(depth, downsampled_depth, depth.shape[1]//downsample_ratio,
                                           depth.shape[0]//downsample_ratio, downsample_ratio)
    return downsampled_depth


# Adapted from HD3.
def disp2flow(disp):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `disp2flow` 操作。

    参数:
        disp: 水平视差图，单位为像素。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    assert disp.size(1) == 1
    padder = torch.zeros(disp.size(), device=disp.device)
    return torch.cat([disp, padder], dim=1)


# Adapted from HD3.
def resize_dense_vector(vec, des_height, des_width):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `resize_dense_vector` 操作。

    参数:
        vec: 待调整尺寸的稠密二维向量场，形状为 B×2×H×W。
        des_height: 目标向量场高度，单位为像素。
        des_width: 目标向量场宽度，单位为像素。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    ratio_height = float(des_height / vec.size(2))
    ratio_width = float(des_width / vec.size(3))
    vec = F.interpolate(
        vec, (des_height, des_width), mode='bilinear', align_corners=True)
    if vec.size(1) == 1:
        vec = vec * ratio_width
    else:
        vec = torch.stack(
            [vec[:, 0, :, :] * ratio_width, vec[:, 1, :, :] * ratio_height],
            dim=1)
    return vec


def EndPointError(output, gt):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `EndPointError` 操作。

    参数:
        output: 调用方预分配并由当前函数填充的输出张量或数组。
        gt: 真实光流与可选有效掩码组成的监督张量。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if output.size(1) == 1:  # A single channel represents disparity.
        output = disp2flow(output)
    output = resize_dense_vector(output, gt.size(2), gt.size(3))
    error = torch.norm(output - gt[:, :2, :, :], 2, 1, keepdim=False)
    if gt.size(1) == 3:
        mask = gt[:, 2, :, :]
    else:
        mask = torch.ones_like(error)
    epe = (error * mask).sum() / mask.sum()
    return epe.reshape(1)


def uncertainty_to_color(_tensor):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `uncertainty_to_color` 操作。

    参数:
        _tensor: 需要转换为伪彩色的不确定性张量。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    total_uncertainty = _tensor[0, :, :] + _tensor[1, :, :]
    total_uncertainty = torch.from_numpy(total_uncertainty)
    total_uncertainty -= torch.min(total_uncertainty)
    total_uncertainty /= torch.max(total_uncertainty)
    jet = cm.get_cmap('jet')
    color = jet(total_uncertainty)
    color = color[:, :, :3]
    color = F.max_pool2d(torch.tensor(color[:, :, :3]).permute(2, 0, 1).unsqueeze(0), 3, 1, 1)
    color = color[0].permute(1, 2, 0).numpy()
    return color

# Source: https://scikit-surgerycore.readthedocs.io
def average_quaternions(quaternions):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `average_quaternions` 操作。

    参数:
        quaternions: 待聚合的单位四元数集合，形状为 N×4。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """

    # Number of quaternions.
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)

    for i in range(0, samples):
        quat = quaternions[i, :]
        # Accumulate quaternion outer products.
        mat_a = np.outer(quat, quat) + mat_a

    # Normalize by sample count.
    mat_a = (1.0/ samples)*mat_a
    # Compute eigenpairs.
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)
    # Sort by descending eigenvalue.
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
    # The principal eigenvector is the mean quaternion.
    return np.real(np.ravel(eigen_vectors[:, 0]))


def voxelize_gpu(points, voxel_size):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `voxelize_gpu` 操作。

    参数:
        points: 输入点、点云或坐标集合；形状由当前函数约定。
        voxel_size: 体素边长，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    batch = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)
    voxel_cluster = voxel_grid(points, batch, size = voxel_size)
    voxel_cluster, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm]
    points_sampled = scatter_mean(points, voxel_cluster, dim = 0)
    
    return points_sampled


# Source: https://github.com/dilaragokay/central-tendency-rotations/
def hamilton_product(q1, q2):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `hamilton_product` 操作。

    参数:
        q1: Hamilton 乘积左侧四元数或批量四元数。
        q2: Hamilton 乘积右侧四元数或批量四元数。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    return torch.tensor(
        [
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ]
    )


# Source: https://github.com/dilaragokay/central-tendency-rotations/
def mean(Q, weights=None):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `mean` 操作。

    参数:
        Q: 待求均值、中位数或众数的单位四元数集合。
        weights: 按级联阶段顺序排列的 checkpoint 路径序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if weights is None:
        weights = torch.ones(len(Q), device=torch.device("cuda:0")) / len(Q)
    A = torch.zeros((4, 4), device=torch.device("cuda:0"))
    weight_sum = torch.sum(weights)

    oriented_Q = ((Q[:, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("bi,bk->bik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("bij,b->bij", (A, weights)), 0)
    A /= weight_sum

    q_avg = torch.linalg.eigh(A)[1][:, -1]
    if q_avg[0] < 0:
        return -q_avg
    return q_avg

# Source: https://github.com/dilaragokay/central-tendency-rotations/
def quaternion_median(Q, p=1, max_angular_update=0.0001, max_iterations=1000):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `quaternion_median` 操作。

    参数:
        Q: 待求均值、中位数或众数的单位四元数集合。
        p: 四元数中位数使用的距离指数。
        max_angular_update: 四元数中位数迭代允许的最大角更新阈值，单位为弧度。
        max_iterations: 四元数中位数求解的最大迭代次数。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    Q = Q.clone().detach()
    weights = torch.ones(len(Q)) / len(Q)
    q_median = mean(Q, weights)
    Q = Q.cpu()  # Sequential operations run on CPU.
    q_median = q_median.cpu()
    EPS_ANGLE = 0.0000001
    max_angular_update = max(max_angular_update, EPS_ANGLE)
    theta = 10 * max_angular_update
    i = 0

    while theta > max_angular_update and i <= max_iterations:
        delta = torch.zeros(3)
        weight_sum = 0
        for q in Q:
            qj = hamilton_product(
                q, torch.tensor([q_median[0], -q_median[1], -q_median[2], -q_median[3]])
            )
            theta = 2 * torch.acos(qj[0])
            if theta > EPS_ANGLE:
                axis_angle = qj[1:] / torch.sin(theta / 2)
                axis_angle *= theta
                weight = 1.0 / pow(theta, 2 - p)
                delta += weight * axis_angle
                weight_sum += weight
        if weight_sum > EPS_ANGLE:
            delta /= weight_sum
            theta = torch.linalg.norm(delta)
            if theta > EPS_ANGLE:
                stby2 = torch.sin(theta * 0.5)
                delta /= theta
                q = torch.tensor(
                    [
                        torch.cos(theta * 0.5),
                        stby2 * delta[0],
                        stby2 * delta[1],
                        stby2 * delta[2],
                    ]
                )
                q_median = hamilton_product(q, q_median)
                if q_median[0] < 0:
                    q_median *= -1
        else:
            theta = 0
        i += 1
    return q_median


# Source: https://github.com/dilaragokay/central-tendency-rotations/
def quaternion_mode(Q, precision=3):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `quaternion_mode` 操作。

    参数:
        Q: 待求均值、中位数或众数的单位四元数集合。
        precision: 四元数众数聚类使用的角度精度阈值。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if isinstance(Q, torch.Tensor):
        Q = Q.clone().detach()
    else:
        Q = torch.tensor(Q)
    rounded_quats = torch.tensor(np.around(Q.numpy(), precision))
    values, counts = torch.unique(rounded_quats, dim=0, return_counts=True)
    return values[torch.argmax(counts)]


def str2bool(v):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `str2bool` 操作。

    参数:
        v: 待解析为布尔值的字符串、整数或布尔对象。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_flow_rgb2lidar(*args):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `get_flow_rgb2lidar` 操作。

    参数:
        *args: 调用参数集合；字段含义由当前模块的网络或命令行约定定义。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    pass


def get_ECE(predicted_flow, predicted_uncertainty, target_flow, mask, loss_type="NLL"):
    """完成 点云变换、外参表示、标定光流和训练公共工具 中的 `get_ECE` 操作。

    参数:
        predicted_flow: 网络预测的二维光流或多次迭代预测列表。
        predicted_uncertainty: 网络对光流分量预测的不确定性参数张量或迭代列表。
        target_flow: 光流监督张量或按尺度组织的光流监督列表。
        mask: 有效像素布尔或数值掩码；非零位置参与计算。
        loss_type: 不确定性校准使用的分布类型，例如 NLL 或 DER。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    gamma1 = predicted_flow[:, 0, :, :]
    y1 = target_flow[:, 0, :, :]
    if loss_type == "NLL":
        sigma1 = predicted_uncertainty[:, 0, :, :]
    else:
        v1 = predicted_uncertainty[:, 0, :, :]
        alpha1 = predicted_uncertainty[:, 1, :, :]
        beta1 = predicted_uncertainty[:, 2, :, :]
        sigma1 = (beta1 / (v1 * (alpha1))).sqrt()
    expected_p = np.arange(41) / 40.
    observed_p1 = []
    for p in expected_p:
        ppf = scipy.stats.norm.ppf(p, loc=gamma1[mask[:, 0] != 0].detach().cpu().numpy(),
                                   scale=sigma1[mask[:, 0] != 0].detach().cpu().numpy())
        obs_p = y1[mask[:, 0] != 0].detach().cpu().numpy()
        obs_p = obs_p < ppf
        observed_p1.append(obs_p.mean())
    ece_u = np.abs(expected_p - observed_p1).mean()

    gamma2 = predicted_flow[:, 1, :, :]
    y2 = target_flow[:, 1, :, :]
    if loss_type == "NLL":
        sigma2 = predicted_uncertainty[:, 1, :, :]
    else:
        v2 = predicted_uncertainty[:, 3, :, :]
        alpha2 = predicted_uncertainty[:, 4, :, :]
        beta2 = predicted_uncertainty[:, 5, :, :]
        sigma2 = (beta2 / (v2 * (alpha2))).sqrt()
    observed_p2 = []
    for p in expected_p:
        ppf = scipy.stats.norm.ppf(p, loc=gamma2[mask[:, 1] != 0].detach().cpu().numpy(),
                                   scale=sigma2[mask[:, 1] != 0].detach().cpu().numpy())
        obs_p = y2[mask[:, 1] != 0].detach().cpu().numpy()
        obs_p = obs_p < ppf
        observed_p2.append(obs_p.mean())
    ece_v = np.abs(expected_p - observed_p2).mean()

    return torch.tensor(ece_u, device=predicted_flow.device), torch.tensor(ece_v, device=predicted_flow.device)
