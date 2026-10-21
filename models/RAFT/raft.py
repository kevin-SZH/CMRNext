"""CMRNext 双编码器 RAFT 主网络。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from models.RAFT.modules.corr import CorrBlock
from models.RAFT.modules.extractor import SmallEncoder, BasicEncoder
from models.RAFT.modules.update import SmallUpdateBlock, BasicUpdateBlock
from models.RAFT.utils.utils import coords_grid, upflow2, upflow8


class CMRNext(nn.Module):
    """使用独立 RGB/LiDAR 编码器和 RAFT 循环更新器预测跨模态二维标定光流。

    主要职责:
        使用独立 RGB/LiDAR 编码器和 RAFT 循环更新器预测跨模态二维标定光流，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, args, use_reflectance=False, with_uncertainty=False, fourier_levels=-1, unc_type="DER",
                 unc_freeze=False, context_encoder='rgb'):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            args: 调用参数集合；字段含义由当前模块的网络或命令行约定定义。
            use_reflectance: 是否把 LiDAR 反射强度作为额外输入通道。
            with_uncertainty: 控制对应功能是否启用的布尔标志。
            fourier_levels: 深度 Fourier 特征层数；负值禁用，非负值决定正余弦频率数量。
            unc_type: 不确定性表示类型，例如 NLL 或 DER。
            unc_freeze: 是否冻结主光流网络并仅训练独立不确定性更新分支。
            context_encoder: 上下文编码来源，取 rgb 或 lidar。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(CMRNext, self).__init__()
        self.args = args
        self.with_uncertainty = with_uncertainty
        self.use_reflectance = use_reflectance
        self.unc_type = unc_type
        self.unc_freeze = unc_freeze
        self.context_encoder = context_encoder

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in args._get_kwargs():
            args.dropout = 0

        self.elu = nn.ELU()

        lidar_feat = 1
        if fourier_levels >= 0:
            lidar_feat = 2 * fourier_levels
            lidar_feat = lidar_feat + 1
        if use_reflectance:
            lidar_feat = lidar_feat + 1
        # Initialize feature, context, and update networks.
        if args.small:
            self.fnet_img = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.fnet_lidar = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout, in_feat=lidar_feat)
            if self.context_encoder == 'rgb':
                self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            else:
                self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none',
                                         dropout=args.dropout, in_feat=lidar_feat)
            if not self.unc_freeze:
                self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim,
                                                     with_uncertainty=with_uncertainty, unc_type=unc_type)
            else:
                self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim, with_uncertainty=False)
                if unc_type == "NLL":
                    self.update_block_unc = SmallUpdateBlock(self.args, hidden_dim=hdim, force_out_dim=2)
                elif unc_type == "DER":
                    self.update_block_unc = SmallUpdateBlock(self.args, hidden_dim=hdim, force_out_dim=6)

        else:
            self.fnet_img = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.fnet_lidar = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, in_feat=lidar_feat)
            if self.context_encoder == 'rgb':
                self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            else:
                self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch',
                                         dropout=args.dropout, in_feat=lidar_feat)
            if not self.unc_freeze:
                self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim,
                                                     with_uncertainty=with_uncertainty, unc_type=unc_type)
            else:
                self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, with_uncertainty=False)
                if unc_type == "NLL":
                    self.update_block_unc = BasicUpdateBlock(self.args, hidden_dim=hdim, force_out_dim=2)
                elif unc_type == "DER":
                    self.update_block_unc = BasicUpdateBlock(self.args, hidden_dim=hdim, force_out_dim=6)


    def freeze_bn(self):
        """完成 CMRNext 双编码器 RAFT 主网络 中的 `freeze_bn` 操作。

        参数:
            self: 当前类实例。

        返回值:
            无显式返回值；函数完成所述状态更新、校验或输出操作。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, dtype=torch.float32):
        """完成 CMRNext 双编码器 RAFT 主网络 中的 `initialize_flow` 操作。

        参数:
            self: 当前类实例。
            img: 输入图像或特征张量；布局由当前工具函数说明。
            dtype: 坐标网格和初始光流使用的 PyTorch 数据类型。

        返回值:
            计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device).type(dtype)
        coords1 = coords_grid(N, H//8, W//8).to(img.device).type(dtype)

        # Flow is coords1 - coords0.
        return coords0, coords1

    def upsample_flow(self, flow, mask, uncertainty=None):
        """完成 CMRNext 双编码器 RAFT 主网络 中的 `upsample_flow` 操作。

        参数:
            self: 当前类实例。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
            mask: 有效像素布尔或数值掩码；非零位置参与计算。
            uncertainty: 可选低分辨率不确定性参数张量，与当前光流空间尺寸一致。

        返回值:
            计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        up_uncertainty = None
        if uncertainty is not None:
            if self.unc_type == "NLL":
                uncertainty = self.elu(uncertainty) + 2
                up_uncertainty = F.unfold(uncertainty, [3, 3], padding=1)
                up_uncertainty = up_uncertainty.view(N, 2, 9, 1, 1, H, W)
                up_uncertainty = torch.sum(mask * up_uncertainty, dim=2)
                up_uncertainty = up_uncertainty.permute(0, 1, 4, 2, 5, 3)
                up_uncertainty = up_uncertainty.reshape(N, 2, 8*H, 8*W)
            
            elif self.unc_type == "DER":
                up_uncertainty = F.unfold(uncertainty, [3, 3], padding=1)
                up_uncertainty = up_uncertainty.view(N, 6, 9, 1, 1, H, W)
                up_uncertainty = torch.sum(mask * up_uncertainty, dim=2)
                up_uncertainty = up_uncertainty.permute(0, 1, 4, 2, 5, 3)
                up_uncertainty = up_uncertainty.reshape(N, 6, 8*H, 8*W)
                
                up_uncertainty = F.softplus(up_uncertainty)

        return up_flow.reshape(N, 2, 8*H, 8*W), up_uncertainty


    def forward(self, image1, lidar_image, iters=12, flow_init=None, upsample=True, test_mode=False):
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            image1: 作为上下文和匹配目标的 RGB 图像张量，形状为 B×3×H×W。
            lidar_image: 由 LiDAR 投影生成的深度或深度特征图，形状为 B×C×H×W。
            iters: RAFT 循环更新次数。
            flow_init: 可选低分辨率初始光流，单位为像素。
            upsample: 是否先将低分辨率预测放大并同步缩放流幅值后计算损失。
            test_mode: 是否返回测试模式的紧凑输出而不是全部迭代预测。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """

        hdim = self.hidden_dim
        cdim = self.context_dim

        # Run feature encoders.
        fmap1 = self.fnet_lidar(lidar_image)
        fmap2 = self.fnet_img(image1)
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # Run the context encoder.
        if self.context_encoder == 'rgb':
            cnet = self.cnet(image1)
        else:
            cnet = self.cnet(lidar_image)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        if self.unc_freeze:
            net_unc = torch.tanh(net)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1, dtype=fmap1.dtype)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_uncertainties = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # Index the correlation volume.

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            if self.unc_freeze:
                net_unc, up_mask_unc, delta_flow_unc = self.update_block_unc(net_unc, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow[:, :2]
            uncertainty = None
            if self.with_uncertainty:
                if self.unc_freeze:
                    uncertainty = delta_flow_unc
                else:
                    uncertainty = delta_flow[:, 2:]

            uncertainty_up = None
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up, uncertainty_up = self.upsample_flow(coords1 - coords0, up_mask, uncertainty)

            flow_predictions.append(flow_up)
            flow_uncertainties.append(uncertainty_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions, flow_uncertainties
