"""针孔相机内参管理与三维点投影。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import numpy as np
import torch


class CameraModel:

    """保存针孔相机内参，并提供相机坐标系三维点到图像平面的投影操作。

    主要职责:
        保存针孔相机内参，并提供相机坐标系三维点到图像平面的投影操作，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
    """
    def __init__(self, focal_length=None, principal_point=None, skew=0.0):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            focal_length: 焦距 fx、fy，单位为像素。
            principal_point: 主点 cx、cy，单位为像素。
            skew: 相机内参的像素轴倾斜项，单位为像素。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.skew = skew

    def set_intrinsics(self, parameters):
        """从按 fx、fy、cx、cy、skew 排列的参数向量设置相机内参。

        参数:
            self: 当前类实例。
            parameters: 相机内参数向量，依次为 fx、fy、cx、cy 和可选 skew，单位为像素。

        返回值:
            当前 CameraModel 实例，便于链式设置后立即投影。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        self.focal_length = parameters[:2]
        self.principal_point = parameters[2:4]
        self.skew = parameters[4] if len(parameters) > 4 else 0.0
        return self

    def project_pytorch(self, xyz: torch.Tensor, image_size, reflectance=None, change_frame=False):
        """将相机坐标系齐次三维点投影到图像平面，并过滤相机后方或越界点。

        参数:
            self: 当前类实例。
            xyz: 相机坐标系三维点，形状为 3×N 或齐次 4×N，长度单位为米。
            image_size: 目标图像尺寸，按高度、宽度排列，单位为像素。
            reflectance: 可选 LiDAR 反射强度，顺序与点云一致。
            change_frame: 是否把输入点轴从 (x,y,z) 重排为历史数据集使用的 (y,z,x)。

        返回值:
            四元组 (uv, depth, xyz, reflectance)：图像内像素坐标、Z 深度、对应三维点及可选反射强度。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
        if change_frame:
            order = [1, 2, 0, 3]
            xyzw = xyz[order, :]
        else:
            order = [0, 1, 2, 3]
            xyzw = xyz[order, :]
        indexes = xyzw[2, :] >= 0
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]

        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = (self.focal_length[0] * xyzw[0, :] / xyzw[2, :]
                    + self.skew * xyzw[1, :] / xyzw[2, :]
                    + self.principal_point[0])
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        indexes = uv[0, :] >= 0.1
        indexes = indexes & (uv[1, :] >= 0.1)
        indexes = indexes & (uv[0,:] < image_size[1])
        indexes = indexes & (uv[1,:] < image_size[0])
        if reflectance is None:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], None
        else:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], reflectance[:, indexes]

        return uv

    def get_matrix(self):
        """把当前相机内参组装为三乘三针孔相机矩阵。

        参数:
            self: 当前类实例。

        返回值:
            NumPy 三乘三相机内参矩阵 K。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

        坐标系和数据约定:
            点云默认使用齐次列向量；相机坐标采用 Z 轴向前。外参为 4×4 刚体变换，平移单位为米，图像坐标顺序为 (u, v)。
        """
        matrix = np.zeros([3, 3])
        matrix[0, 0] = self.focal_length[0]
        matrix[0, 1] = self.skew
        matrix[1, 1] = self.focal_length[1]
        matrix[0, 2] = self.principal_point[0]
        matrix[1, 2] = self.principal_point[1]
        matrix[2, 2] = 1.0
        return matrix
