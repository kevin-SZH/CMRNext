"""原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import json
import logging
import os
import sys
from math import radians

import cv2
import mathutils
import numpy as np
import pandas as pd
import pykitti
import pyntcloud
import torch
import yaml
from PIL import Image
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.json_utils import read_json_file
from pandaset.geometry import _heading_position_to_mat
from torch.utils.data import Dataset
from torchvision import transforms

from utils import (
    invert_pose,
    quat2mat,
    quaternion_from_matrix,
    rotate_forward,
    to_rotation_matrix,
    tvector2mat,
)

logging.getLogger('argoverse').setLevel(logging.ERROR)


def is_image(img):
    """判断文件扩展名是否属于当前加载器支持的图像格式。

    参数:
        img: 输入图像或特征张量；布局由当前工具函数说明。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    extensions = ['.jpg', '.png', '.tiff', '.jpeg', '.bmp']
    return os.path.splitext(img)[1] in extensions


# Generic point-cloud reader from https://github.com/PRBonn/kiss-icp
def _get_point_cloud_reader(file_extension, first_scan_file):
        """尝试可用点云库并返回能够读取当前扫描格式的调用对象。

        参数:
            file_extension: 待探测点云文件的扩展名，例如 bin、ply 或 pcd。
            first_scan_file: 当前操作读取或写入的文件路径。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        # Read the legacy KITTI binary format.
        if file_extension == "bin":
            print("[WARNING] Reading .bin files, the only format supported is the KITTI format")

            class ReadKITTI:
                """封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadKITTI` 状态和操作。

                主要职责:
                    封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadKITTI` 状态和操作，并集中维护该职责所需状态。

                返回值:
                    类实例由构造方法创建；各公共方法的返回值见对应方法说明。

                异常和副作用:
                    文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                坐标系和数据约定:
                    点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                """
                def __call__(self, file):
                    """使当前对象可调用，并对输入执行该类定义的数据读取、变换或相关查询。

                    参数:
                        self: 当前类实例。
                        file: 需要读取的点云、光流或图像文件路径。

                    返回值:
                        返回由 `实现中的返回表达式` 表示的计算结果。

                    异常和副作用:
                        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                    坐标系和数据约定:
                        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                    """
                    return np.fromfile(file, dtype=np.float32).reshape((-1, 4))

            return ReadKITTI()

        print('Trying to guess how to read your data')
        # Try Open3D first.
        try:
            import open3d as o3d

            try_pcd = o3d.io.read_point_cloud(first_scan_file)
            if try_pcd.is_empty():
                # Open3D returns an empty cloud for unreadable files.
                raise Exception("通用数据加载器：Open3D 点云文件为空")

            class ReadOpen3d:
                """封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadOpen3d` 状态和操作。

                主要职责:
                    封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadOpen3d` 状态和操作，并集中维护该职责所需状态。

                返回值:
                    类实例由构造方法创建；各公共方法的返回值见对应方法说明。

                异常和副作用:
                    文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                坐标系和数据约定:
                    点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                """
                def __call__(self, file):
                    """使当前对象可调用，并对输入执行该类定义的数据读取、变换或相关查询。

                    参数:
                        self: 当前类实例。
                        file: 需要读取的点云、光流或图像文件路径。

                    返回值:
                        返回由 `实现中的返回表达式` 表示的计算结果。

                    异常和副作用:
                        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                    坐标系和数据约定:
                        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                    """
                    pcd = o3d.io.read_point_cloud(file)
                    points = np.asarray(pcd.points)
                    return points

            return ReadOpen3d()
        except:
            pass

        try:
            import trimesh

            trimesh.load(first_scan_file)

            class ReadTriMesh:
                """封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadTriMesh` 状态和操作。

                主要职责:
                    封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadTriMesh` 状态和操作，并集中维护该职责所需状态。

                返回值:
                    类实例由构造方法创建；各公共方法的返回值见对应方法说明。

                异常和副作用:
                    文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                坐标系和数据约定:
                    点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                """
                def __call__(self, file):
                    """使当前对象可调用，并对输入执行该类定义的数据读取、变换或相关查询。

                    参数:
                        self: 当前类实例。
                        file: 需要读取的点云、光流或图像文件路径。

                    返回值:
                        返回由 `实现中的返回表达式` 表示的计算结果。

                    异常和副作用:
                        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                    坐标系和数据约定:
                        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                    """
                    return np.asarray(trimesh.load(file).vertices)

            return ReadTriMesh()
        except:
            pass

        try:
            from pyntcloud import PyntCloud

            PyntCloud.from_file(first_scan_file)

            class ReadPynt:
                """封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadPynt` 状态和操作。

                主要职责:
                    封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `ReadPynt` 状态和操作，并集中维护该职责所需状态。

                返回值:
                    类实例由构造方法创建；各公共方法的返回值见对应方法说明。

                异常和副作用:
                    文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                坐标系和数据约定:
                    点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                """
                def __call__(self, file):
                    """使当前对象可调用，并对输入执行该类定义的数据读取、变换或相关查询。

                    参数:
                        self: 当前类实例。
                        file: 需要读取的点云、光流或图像文件路径。

                    返回值:
                        返回由 `实现中的返回表达式` 表示的计算结果。

                    异常和副作用:
                        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

                    坐标系和数据约定:
                        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
                    """
                    return PyntCloud.from_file(file).points[["x", "y", "z"]].to_numpy()

            return ReadPynt()
        except:
            print("[ERROR], File format not supported")
            sys.exit(1)


def get_scan_kitti(path, cam='2', kitti=None):
    """读取 KITTI velodyne 点云，并返回目标相机外参和针孔内参。

    参数:
        path: 输入或输出文件路径。
        cam: 目标相机编号或名称；KITTI 常用校正相机 2 或 3。
        kitti: 可选已加载的 pykitti.odometry 对象；为空时由点云路径创建。

    返回值:
        读取或计算得到的数据、标定参数或点云。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    scan = np.fromfile(path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    split_path = path.split('/')
    base_folder = os.path.join('/', *split_path[:-4])
    if kitti is None:
        kitti = pykitti.odometry(base_folder, split_path[-3])
    if cam == '2' or cam == '02':
        cam_to_velo = torch.from_numpy(kitti.calib.T_cam2_velo).double()
        calib = kitti.calib.K_cam2
    elif cam == '3' or cam == '03':
        cam_to_velo = torch.from_numpy(kitti.calib.T_cam3_velo).double()
        calib = kitti.calib.K_cam3
    calib = torch.tensor([calib[0, 0], calib[1, 1], calib[0, 2], calib[1, 2]]).float()
    return scan, cam_to_velo.float(), calib


def _get_kitti_camera_calibration(kitti, cam):
    """按相机编号返回 KITTI 校正相机的真实外参和内参。

    参数:
        kitti: 可选已加载的 pykitti.odometry 对象；为空时由点云路径创建。
        cam: 目标相机编号或名称；KITTI 常用校正相机 2 或 3。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    cam = str(cam)
    if cam in ('2', '02'):
        extrinsic = kitti.calib.T_cam2_velo
        intrinsic = kitti.calib.K_cam2
    elif cam in ('3', '03'):
        extrinsic = kitti.calib.T_cam3_velo
        intrinsic = kitti.calib.K_cam3
    else:
        raise ValueError(f"不支持的 KITTI 相机：{cam}")
    camera_params = torch.tensor([
        intrinsic[0, 0], intrinsic[1, 1],
        intrinsic[0, 2], intrinsic[1, 2],
    ]).float()
    return torch.from_numpy(extrinsic).float(), camera_params


def compute_kitti_sampled_initial_state(
        true_extrinsic, correction_quaternion, correction_translation):
    """复现原版 CMRNext KITTI 逐轴采样并构造实际 T0 和左乘残差。

    参数:
        true_extrinsic: 逐帧真实 LiDAR 到相机外参 Ti，形状为 4×4。
        correction_quaternion: 原版数据加载器采样的输入旋转扰动四元数。
        correction_translation: 原版数据加载器采样的输入平移扰动，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    correction = torch.mm(
        tvector2mat(correction_translation),
        quat2mat(correction_quaternion),
    )
    initial_extrinsic = torch.mm(correction, true_extrinsic)
    residual = torch.linalg.inv(correction)
    return initial_extrinsic, residual


def get_scan_argo(path, camera):
    """读取 Argoverse 点云以及指定环视相机的外参和内参。

    参数:
        path: 输入或输出文件路径。
        camera: 数据集中目标相机的名称。

    返回值:
        读取或计算得到的数据、标定参数或点云。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    data = pyntcloud.PyntCloud.from_file(os.fspath(path))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    lidar_intensity = np.array(data.points.intensity, dtype=np.float32)[:, np.newaxis]
    lidar_pts = np.concatenate((x, y, z, lidar_intensity), axis=1)

    splitted_path = path.split('/')

    calib = read_json_file(path.replace(f'{splitted_path[-2]}/{splitted_path[-1]}', 'vehicle_calibration_info.json'))
    calib = get_calibration_config(calib, camera)
    cam2_to_velo = torch.from_numpy(calib.extrinsic)
    intrinsics = torch.tensor([calib.intrinsic[0, 0], calib.intrinsic[1, 1],
                               calib.intrinsic[0, 2], calib.intrinsic[1, 2]])

    return lidar_pts, cam2_to_velo.float(), intrinsics.float()


def get_scan_pandaset(path, sensor_id=0):
    """读取 PandaSet HDF5 地图点云及可选强度通道。

    参数:
        path: 输入或输出文件路径。
        sensor_id: PandaSet LiDAR 传感器编号。

    返回值:
        读取或计算得到的数据、标定参数或点云。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    scan = pd.read_pickle(path)
    scan = scan.loc[scan['d'] == sensor_id]

    return scan.values[:, :4]


def get_extrinsic_pandaset(camera):
    """返回 PandaSet 指定相机与 LiDAR 之间的固定标定外参。

    参数:
        camera: 数据集中目标相机的名称。

    返回值:
        读取或计算得到的数据、标定参数或点云。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    with open(os.path.join(os.path.dirname(__file__), 'pandaset_extrinsic.yaml')) as f:
        file_data = yaml.safe_load(f)
    camera_pose = file_data[camera]['extrinsic']['transform']
    camera_translation = torch.tensor([camera_pose['translation']['x'], camera_pose['translation']['y'],
                                       camera_pose['translation']['z']])
    camera_quaternion = torch.tensor([camera_pose['rotation']['w'], camera_pose['rotation']['x'],
                                      camera_pose['rotation']['y'], camera_pose['rotation']['z']])
    camera_pose = to_rotation_matrix(camera_quaternion, camera_translation)
    return camera_pose


class DatasetGeneralExtrinsicCalib(Dataset):

    """封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `DatasetGeneralExtrinsicCalib` 状态和操作。

    主要职责:
        封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `DatasetGeneralExtrinsicCalib` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=2., max_r=10.,
                 train=True, normalize_images=True, dataset='kitti', cam='2', change_frame=False,
                 camera_intrinsics=None, nominal_cam=None):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            dataset_dirs: 参与当前数据集的序列目录列表。
            transform: 四乘四齐次刚体变换矩阵，平移单位为米。
            augmentation: 是否启用训练期图像颜色和空间增强。
            use_reflectance: 是否把 LiDAR 反射强度作为额外输入通道。
            max_t: 逐轴平移扰动上限，单位为米。
            max_r: 逐轴欧拉角扰动上限，单位为度。
            train: 是否启用训练期随机增强或逐次重采样。
            normalize_images: 是否按 ImageNet 统计量归一化 RGB 图像。
            dataset: 数据集模式名称，用于选择尺寸、坐标和后处理约定。
            cam: 目标相机编号或名称；KITTI 常用校正相机 2 或 3。
            change_frame: 是否把输入点轴从 (x,y,z) 重排为历史数据集使用的 (y,z,x)。
            camera_intrinsics: 可选显式针孔相机内参 [fx,fy,cx,cy]，单位为像素。
            nominal_cam: 提供固定标称 T0 的另一相机编号。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        super(DatasetGeneralExtrinsicCalib, self).__init__()
        self.dataset = dataset
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dirs = dataset_dirs
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.maps_folder = None
        self.extension = None
        self.cam = str(cam)
        self.nominal_cam = None if nominal_cam is None else str(nominal_cam)
        self.kitti_fixed_calibrations = {}
        self.camera_folder = f'image_{cam}'
        self.change_frame = change_frame
        if dataset == 'kitti':
            self.maps_folder = 'velodyne'
            self.extension = '.bin'
            if self.nominal_cam is not None:
                if self.cam not in ('2', '3') or self.nominal_cam not in ('2', '3'):
                    raise ValueError("KITTI 固定标称模式仅支持相机 2 和 3")
                if self.cam == self.nominal_cam:
                    raise ValueError("KITTI 目标相机与标称相机不能相同")
        elif dataset == 'argoverse':
            self.maps_folder = 'lidar'
            self.extension = '.ply'
            self.sdbs = {}
        elif dataset == 'custom':
            self.maps_folder = 'lidar'
            self.camera_folder = 'camera'
        self.all_files = []

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for directory in dataset_dirs:

            if dataset == 'kitti' and self.nominal_cam is not None:
                sequence_directory = os.path.abspath(directory)
                sequence = os.path.basename(sequence_directory)
                base_folder = os.path.dirname(os.path.dirname(sequence_directory))
                kitti = pykitti.odometry(base_folder, sequence)
                true_extrinsic, calib = _get_kitti_camera_calibration(
                    kitti, self.cam)
                nominal_extrinsic, _ = _get_kitti_camera_calibration(
                    kitti, self.nominal_cam)
                residual = torch.mm(
                    true_extrinsic, torch.linalg.inv(nominal_extrinsic))
                self.kitti_fixed_calibrations[sequence_directory] = {
                    'true_extrinsic': true_extrinsic,
                    'nominal_extrinsic': nominal_extrinsic,
                    'residual': residual,
                    'calib': calib,
                }

            if dataset == 'argoverse':
                for log_id in sorted(os.listdir(directory)):
                    self.sdbs[log_id] = SynchronizationDB(directory, collect_single_log_id=log_id)

                    point_cloud_folder = os.path.join(directory, log_id, self.maps_folder)

                    sorted_filenames = sorted(os.listdir(point_cloud_folder))
                    for filename in sorted_filenames:
                        self.all_files.append(os.path.join(point_cloud_folder, filename))

            if dataset == 'custom':
                with open(os.path.join(directory, 'calibration.yaml')) as f:
                    file_data = yaml.safe_load(f)
                self.camera_intrinsics = torch.tensor(
                    [file_data['fx'], file_data['fy'], file_data['cx'], file_data['cy']])
                self.initial_extrinsic = torch.tensor(file_data['initial_extrinsic'], dtype=torch.float).reshape(4, 4)
                first_scan = os.listdir(os.path.join(directory, self.maps_folder))
                first_scan = sorted(first_scan)[0]
                self.extension = os.path.splitext(first_scan)[1]
                self.point_cloud_reader = _get_point_cloud_reader(self.extension[1:],
                                                                  os.path.join(directory, self.maps_folder, first_scan))

            if dataset == 'kitti' or dataset == 'custom':
                img_folder = os.path.join(directory, self.camera_folder)
                point_cloud_folder = os.path.join(directory, self.maps_folder)

                sorted_filenames = sorted(os.listdir(img_folder))
                for filename in sorted_filenames:
                    filename_no_extension = os.path.splitext(filename)[0]
                    point_cloud_path = os.path.join(point_cloud_folder, filename_no_extension + self.extension)
                    if not os.path.exists(point_cloud_path):
                        continue
                    self.all_files.append(os.path.join(img_folder, filename))

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        """对图像执行训练期颜色、翻转或旋转增强并同步更新相关几何量。

        参数:
            self: 当前类实例。
            rgb: RGB 图像张量或数组；通道顺序和数值范围遵循调用方约定。
            calib: 随图像旋转或翻转同步更新的相机内参数向量。
            img_rotation: 施加到图像的平面内旋转角度，单位为度。
            flip: 是否执行水平镜像并同步修正图像几何量。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2)
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))

        return torch.tensor(rgb).float()

    def __len__(self):
        """返回当前数据集或容器可访问的样本数量。

        参数:
            self: 当前类实例。

        返回值:
            可访问样本或元素数量，类型为整数。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        return len(self.all_files)

    def __getitem__(self, idx):
        """按全局样本索引读取并构造一个可供当前数据集调用方使用的样本。

        参数:
            self: 当前类实例。
            idx: 样本、批次或可视化项索引，从零开始。

        返回值:
            当前索引对应的数据样本、张量或容器元素。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        if self.dataset == 'kitti' or self.dataset == 'custom':
            img_path = self.all_files[idx]
            extension = os.path.basename(img_path)
            extension = os.path.splitext(extension)[1]
            pc_path = img_path.replace(f'/{self.camera_folder}/', f'/{self.maps_folder}/').replace(extension,
                                                                                                   self.extension)
        elif self.dataset == 'argoverse':
            pc_path = self.all_files[idx]

            splitted_path = pc_path.split('/')
            lidar_stamp = int(splitted_path[-1][3:-4])
            log_id = splitted_path[-3]
            sdb = self.sdbs[log_id]

        fixed_kitti = self.dataset == 'kitti' and self.nominal_cam is not None
        if fixed_kitti:
            pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
            sequence_directory = os.path.abspath(
                os.path.dirname(os.path.dirname(pc_path)))
            fixed_calibration = self.kitti_fixed_calibrations[sequence_directory]
            cam2vel = fixed_calibration['true_extrinsic'].clone()
            calib = fixed_calibration['calib'].clone()
        elif self.dataset == 'kitti':
            pc, cam2vel, calib = get_scan_kitti(pc_path, cam=self.cam)
        elif self.dataset == 'argoverse':
            cam_timestamp = sdb.get_closest_cam_channel_timestamp(lidar_stamp, self.cam, log_id)

            img_path = pc_path.replace('/' + self.maps_folder + '/', f'/{self.cam}/')
            img_path = img_path.replace(splitted_path[-1], f'{self.cam}_{cam_timestamp}.jpg')

            pc, cam2vel, calib = get_scan_argo(pc_path, self.cam)
        elif self.dataset == 'custom':
            pc = self.point_cloud_reader(pc_path)
            cam2vel = self.initial_extrinsic
            calib = self.camera_intrinsics
            if pc.shape[1] == 3:
                pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), 1)
            elif pc.shape[1] >= 4:
                pc = pc[:, :4]
            else:
                print("[ERROR], Point cloud has less than 3 channels")
                sys.exit(1)

        if self.use_reflectance:
            reflectance = torch.from_numpy(pc[:, -1]).float()
        pc[:, -1] = 1

        pc_in = torch.from_numpy(pc.astype(np.float32))
        pc_in = torch.mm(cam2vel, pc_in.t())
        if self.change_frame:
            pc_in = pc_in[[2, 0, 1, 3], :]

        img = Image.open(img_path)
        h_mirror = False
        if np.random.rand() > 0.5 and self.train:
            h_mirror = True
            if self.change_frame:
                pc_in[1, :] *= -1
            else:
                pc_in[0, :] *= -1
            calib[2] = img.size[0] - calib[2]

        img_rotation = 0.
        if self.train:
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate the point cloud with the image.
        if self.train:
            if self.change_frame:
                R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            else:
                R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if fixed_kitti:
            residual = fixed_calibration['residual'].clone()
            input_from_target = torch.linalg.inv(residual)
            R = quaternion_from_matrix(input_from_target).float()
            T = input_from_target[:3, 3].clone().float()
        else:
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)

            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))

            if self.change_frame:
                R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
                T = mathutils.Vector((transl_x, transl_y, transl_z))
            else:
                R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
                T = mathutils.Vector((transl_y, transl_z, transl_x))

            R, T = invert_pose(R, T)
            R, T = torch.tensor(R), torch.tensor(T)

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'tr_error': T,
                  'rot_error': R, 'rgb_name': img_path, 'idx': idx, 'cam2vel': cam2vel}
        if fixed_kitti:
            sample.update({
                'nominal_extrinsic': fixed_calibration['nominal_extrinsic'].clone(),
                'residual': residual,
                'dataset_mode': 'kitti_fixed_nominal',
                'target_cam': self.cam,
                'nominal_cam': self.nominal_cam,
            })
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample


class DatasetPandasetExtrinsicCalib(Dataset):

    """封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `DatasetPandasetExtrinsicCalib` 状态和操作。

    主要职责:
        封装 原版 KITTI、Argoverse 和 PandaSet 外参标定数据加载 中的 `DatasetPandasetExtrinsicCalib` 状态和操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=2., max_r=10.,
                 train=True, normalize_images=True, sensor_id=0, camera='front_camera', change_frame=False):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            dataset_dirs: 参与当前数据集的序列目录列表。
            transform: 四乘四齐次刚体变换矩阵，平移单位为米。
            augmentation: 是否启用训练期图像颜色和空间增强。
            use_reflectance: 是否把 LiDAR 反射强度作为额外输入通道。
            max_t: 逐轴平移扰动上限，单位为米。
            max_r: 逐轴欧拉角扰动上限，单位为度。
            train: 是否启用训练期随机增强或逐次重采样。
            normalize_images: 是否按 ImageNet 统计量归一化 RGB 图像。
            sensor_id: PandaSet LiDAR 传感器编号。
            camera: 数据集中目标相机的名称。
            change_frame: 是否把输入点轴从 (x,y,z) 重排为历史数据集使用的 (y,z,x)。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        super(DatasetPandasetExtrinsicCalib, self).__init__()
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dirs = dataset_dirs
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.sensor_id = sensor_id
        self.maps_folder = 'lidar'
        self.extension = 'pkl.gz'
        self.camera = camera

        self.all_files = []
        self.camera_poses = []
        self.camera_stamps = []

        self.change_frame = change_frame

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for directory in dataset_dirs:
            point_cloud_folder = os.path.join(directory, self.maps_folder)

            sorted_filenames = sorted(os.listdir(point_cloud_folder))
            for filename in sorted_filenames:
                if '.json' not in filename:
                    self.all_files.append(os.path.join(point_cloud_folder, filename))

            pose_file = os.path.join(directory, 'camera', camera, 'poses.json')
            timestamp_file = os.path.join(directory, 'camera', camera, 'timestamps.json')
            with open(pose_file, 'r') as f:
                file_data = json.load(f)
                for entry in file_data:
                    self.camera_poses.append(_heading_position_to_mat(entry['heading'], entry['position']))

            with open(timestamp_file, 'r') as f:
                file_data = json.load(f)
                for entry in file_data:
                    self.camera_stamps.append(entry)

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        """对图像执行训练期颜色、翻转或旋转增强并同步更新相关几何量。

        参数:
            self: 当前类实例。
            rgb: RGB 图像张量或数组；通道顺序和数值范围遵循调用方约定。
            calib: 随图像旋转或翻转同步更新的相机内参数向量。
            img_rotation: 施加到图像的平面内旋转角度，单位为度。
            flip: 是否执行水平镜像并同步修正图像几何量。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2)
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))

        return torch.tensor(rgb).float()

    def __len__(self):
        """返回当前数据集或容器可访问的样本数量。

        参数:
            self: 当前类实例。

        返回值:
            可访问样本或元素数量，类型为整数。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        return len(self.all_files)

    def __getitem__(self, idx):
        """按全局样本索引读取并构造一个可供当前数据集调用方使用的样本。

        参数:
            self: 当前类实例。
            idx: 样本、批次或可视化项索引，从零开始。

        返回值:
            当前索引对应的数据样本、张量或容器元素。

        异常和副作用:
            文件缺失、格式、张量形状或外参关系不合法时抛出异常；训练模式下可能推进随机状态或执行图像增强。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        pc_path = self.all_files[idx]
        img_path = pc_path.replace('/' + self.maps_folder + '/', f'/camera/{self.camera}/').replace(self.extension,
                                                                                                    'jpg')

        # Load camera intrinsics.
        calib_file = os.path.dirname(img_path)
        calib_file = os.path.join(calib_file, 'intrinsics.json')
        with open(calib_file, 'r') as f:
            calib = json.load(f)
        calib = torch.tensor([calib['fx'], calib['fy'], calib['cx'], calib['cy']]).float()

        pc = get_scan_pandaset(pc_path, self.sensor_id)
        if self.use_reflectance:
            reflectance = torch.from_numpy(pc[:, -1]).float()
        pc[:, -1] = 1
        cam_pose = torch.from_numpy(self.camera_poses[idx]).float().inverse()
        pc_in = torch.from_numpy(pc.astype(np.float32))
        pc_in = torch.mm(cam_pose, pc_in.t())
        if self.change_frame:
            pc_in = pc_in[[2, 0, 1, 3], :]

        cam2vel = get_extrinsic_pandaset(self.camera)

        img = Image.open(img_path)
        h_mirror = False
        if np.random.rand() > 0.5 and self.train:
            h_mirror = True
            if self.change_frame:
                pc_in[1, :] *= -1
            else:
                pc_in[0, :] *= -1
            calib[2] = img.size[0] - calib[2]

        img_rotation = 0.
        if self.train:
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate the point cloud with the image.
        if self.train:
            if self.change_frame:
                R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            else:
                R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        max_angle = self.max_r
        rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        transl_x = np.random.uniform(-self.max_t, self.max_t)
        transl_y = np.random.uniform(-self.max_t, self.max_t)
        transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))

        if self.change_frame:
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
        else:
            R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
            T = mathutils.Vector((transl_y, transl_z, transl_x))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'tr_error': T,
                  'rot_error': R, 'rgb_name': img_path, 'idx': idx, 'cam2vel': cam2vel}
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample
