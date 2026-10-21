"""RAFT 坐标网格、采样和光流上采样工具。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """把像素坐标转换为归一化网格并执行可微双线性采样。

    参数:
        img: 输入图像或特征张量；布局由当前工具函数说明。
        coords: 当前匹配坐标网格，形状为 B×2×H×W，单位为低分辨率像素。
        mode: 插值或采样模式名称，例如 bilinear。
        mask: 有效像素布尔或数值掩码；非零位置参与计算。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def forward_interpolate(flow):
    """把前向光流散点投射到目标位置并插值得到稠密初始流。

    参数:
        flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='cubic', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='cubic', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def coords_grid(batch, ht, wd):
    """生成每个批次共享的二维像素坐标网格。

    参数:
        batch: 需要生成坐标网格的批次大小。
        ht: 坐标网格高度，单位为像素。
        wd: 坐标网格宽度，单位为像素。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    """将光流双线性放大八倍并同步缩放位移幅值。

    参数:
        flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
        mode: 插值或采样模式名称，例如 bilinear。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def upflow2(flow, mode='bilinear'):
    """将光流双线性放大两倍并同步缩放位移幅值。

    参数:
        flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
        mode: 插值或采样模式名称，例如 bilinear。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return  2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
