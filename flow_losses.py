"""RAFT 多阶段光流监督与不确定性损失。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def NegativeLogLikelihoodLoss(predicted_flow, predicted_uncertainty, target_flow, mask):
    """用预测均值和标准差计算有效光流像素的二维高斯负对数似然。

    参数:
        predicted_flow: 网络预测的二维光流或多次迭代预测列表。
        predicted_uncertainty: 网络对光流分量预测的不确定性参数张量或迭代列表。
        target_flow: 光流监督张量或按尺度组织的光流监督列表。
        mask: 有效像素布尔或数值掩码；非零位置参与计算。

    返回值:
        有效像素上 u、v 两个分量负对数似然均值之和。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    u_component_mean = predicted_flow[:, 0, :, :]
    u_component_sigma = predicted_uncertainty[:, 0, :, :]
    u_component_distribution = torch.distributions.normal.Normal(u_component_mean, u_component_sigma)
    u_component_loss = -u_component_distribution.log_prob(target_flow[:, 0, :, :])
    u_component_loss = u_component_loss * mask

    v_component_mean = predicted_flow[:, 1, :, :]
    v_component_sigma = predicted_uncertainty[:, 1, :, :]
    v_component_distribution = torch.distributions.normal.Normal(v_component_mean, v_component_sigma)
    v_component_loss = -v_component_distribution.log_prob(target_flow[:, 1, :, :])
    v_component_loss = v_component_loss * mask

    return u_component_loss.sum() / mask.sum() + v_component_loss.sum() / mask.sum()


def RAFT_loss2(flow_preds, uncertainty_preds, flow_gt, valid, upsample=True, gamma=0.8, weight_nll=-1.0,
               unc_type="DER"):
    """对 RAFT 各次迭代的光流预测施加指数衰减加权监督并计算 EPE 指标。

    参数:
        flow_preds: RAFT 各次循环输出的光流预测列表。
        uncertainty_preds: RAFT 各次循环输出的不确定性参数列表。
        flow_gt: 真实二维光流张量，形状为 B×2×H×W，单位为像素。
        valid: 光流监督有效区域掩码。
        upsample: 是否先将低分辨率预测放大并同步缩放流幅值后计算损失。
        gamma: RAFT 多次迭代监督的指数衰减系数，允许范围通常为零到一。
        weight_nll: 负对数似然损失权重；-1 表示仅使用似然项的兼容模式。
        unc_type: 不确定性表示类型，例如 NLL 或 DER。

    返回值:
        二元组 (总光流损失, 指标字典)，指标包含 EPE、像素阈值比例和 F1。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # Exclude invalid pixels and extreme displacements.
    valid = valid[:, 0]
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 1000)
    up_preds = []

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        if upsample:
            new_size = (4 * flow_preds[i].shape[2], 4 * flow_preds[i].shape[3])
            up_preds.append(4 * F.interpolate(flow_preds[i], size=new_size, mode='bilinear', align_corners=True))
            i_loss = (up_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        elif uncertainty_preds[i] is None:
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        else:
            if weight_nll == -1.0:
                if unc_type == "NLL":
                    i_loss = NegativeLogLikelihoodLoss(flow_preds[i], uncertainty_preds[i], flow_gt, valid)
            else:
                if unc_type == "NLL":
                    i_loss = NegativeLogLikelihoodLoss(flow_preds[i], uncertainty_preds[i], flow_gt, valid) * weight_nll
                i_loss += (valid[:, None] * (flow_preds[i] - flow_gt).abs()).mean()
            flow_loss += i_weight * i_loss

    if upsample:
        epe = torch.sum((up_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    else:
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'f1': ((epe > 3.0) & ((epe / mag.view(-1)[valid.view(-1)]) > 0.05)).float().mean() * 100.
    }

    return flow_loss, metrics
