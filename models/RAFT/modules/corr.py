"""RAFT 全对相关体与局部相关查询。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import torch
import torch.nn.functional as F

from models.RAFT.utils.utils import bilinear_sampler


class CorrBlock:
    """构建多尺度全对相关金字塔，并在当前流坐标邻域查询相关特征。

    主要职责:
        构建多尺度全对相关金字塔，并在当前流坐标邻域查询相关特征，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            fmap1: 第一路八分之一分辨率特征图，形状为 B×C×H×W。
            fmap2: 第二路八分之一分辨率特征图，形状为 B×C×H×W。
            num_levels: 对应元素或操作的数量。
            radius: 相关查询、滤波或邻域半径，单位由当前算法约定。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Compute all-pairs correlation.
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """使当前对象可调用，并对输入执行该类定义的数据读取、变换或相关查询。

        参数:
            self: 当前类实例。
            coords: 当前匹配坐标网格，形状为 B×2×H×W，单位为低分辨率像素。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1).type(corr.dtype)
            dy = torch.linspace(-r, r, 2*r+1).type(corr.dtype)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """完成 RAFT 全对相关体与局部相关查询 中的 `corr` 操作。

        参数:
            fmap1: 第一路八分之一分辨率特征图，形状为 B×C×H×W。
            fmap2: 第二路八分之一分辨率特征图，形状为 B×C×H×W。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
