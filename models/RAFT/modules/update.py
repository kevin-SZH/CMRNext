"""RAFT 运动编码、循环更新和光流预测头。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    """把循环隐藏状态映射为二维光流增量及可选不确定性参数。

    主要职责:
        把循环隐藏状态映射为二维光流增量及可选不确定性参数，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, input_dim=128, hidden_dim=256, with_uncertainty=False, unc_type="NLL", force_out_dim=None):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            input_dim: 当前神经网络层期望的输入通道数量。
            hidden_dim: 循环隐藏状态或输出特征的通道数量。
            with_uncertainty: 控制对应功能是否启用的布尔标志。
            unc_type: 不确定性表示类型，例如 NLL 或 DER。
            force_out_dim: 可选预测头强制输出通道数；用于独立不确定性分支。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(FlowHead, self).__init__()
        out_dim = 2
        if with_uncertainty:
            if unc_type == "NLL":
                out_dim = 4
            elif unc_type == "DER":
                out_dim = 8
        if force_out_dim is not None:
            out_dim = force_out_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            x: 当前层输入张量；通道和空间布局由所在网络模块定义。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    """使用二维卷积门控循环单元更新 RAFT 隐藏状态。

    主要职责:
        使用二维卷积门控循环单元更新 RAFT 隐藏状态，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, hidden_dim=128, input_dim=192+128):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            hidden_dim: 循环隐藏状态或输出特征的通道数量。
            input_dim: 当前神经网络层期望的输入通道数量。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            h: GRU 当前隐藏状态张量，形状为 B×C×H×W。
            x: 当前层输入张量；通道和空间布局由所在网络模块定义。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    """依次使用水平和垂直可分离卷积门控单元更新隐藏状态。

    主要职责:
        依次使用水平和垂直可分离卷积门控单元更新隐藏状态，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, hidden_dim=128, input_dim=192+128):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            hidden_dim: 循环隐藏状态或输出特征的通道数量。
            input_dim: 当前神经网络层期望的输入通道数量。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # Horizontal pass.
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            h: GRU 当前隐藏状态张量，形状为 B×C×H×W。
            x: 当前层输入张量；通道和空间布局由所在网络模块定义。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # Vertical pass.
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    """融合小模型相关特征和当前光流，生成循环更新所需运动特征。

    主要职责:
        融合小模型相关特征和当前光流，生成循环更新所需运动特征，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, args):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            args: 调用参数集合；字段含义由当前模块的网络或命令行约定定义。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
            corr: 在当前坐标邻域采样的多尺度相关特征张量。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    """融合标准模型相关特征和当前光流，生成循环更新所需运动特征。

    主要职责:
        融合标准模型相关特征和当前光流，生成循环更新所需运动特征，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, args):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            args: 调用参数集合；字段含义由当前模块的网络或命令行约定定义。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
            corr: 在当前坐标邻域采样的多尺度相关特征张量。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    """组合小型运动编码器、ConvGRU 和光流头生成一次 RAFT 更新。

    主要职责:
        组合小型运动编码器、ConvGRU 和光流头生成一次 RAFT 更新，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, args, hidden_dim=96, with_uncertainty=False, unc_type="NLL", force_out_dim=None):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            args: 调用参数集合；字段含义由当前模块的网络或命令行约定定义。
            hidden_dim: 循环隐藏状态或输出特征的通道数量。
            with_uncertainty: 控制对应功能是否启用的布尔标志。
            unc_type: 不确定性表示类型，例如 NLL 或 DER。
            force_out_dim: 可选预测头强制输出通道数；用于独立不确定性分支。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128, with_uncertainty=with_uncertainty, unc_type=unc_type, force_out_dim=force_out_dim)

    def forward(self, net, inp, corr, flow):
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            net: RAFT 循环更新器的隐藏状态张量。
            inp: 上下文编码器输出的静态输入特征张量。
            corr: 在当前坐标邻域采样的多尺度相关特征张量。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    """组合运动编码器、SepConvGRU、光流头和凸上采样 mask 生成一次 RAFT 更新。

    主要职责:
        组合运动编码器、SepConvGRU、光流头和凸上采样 mask 生成一次 RAFT 更新，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, args, hidden_dim=128, input_dim=128, with_uncertainty=False, unc_type="NLL", force_out_dim=None):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            args: 调用参数集合；字段含义由当前模块的网络或命令行约定定义。
            hidden_dim: 循环隐藏状态或输出特征的通道数量。
            input_dim: 当前神经网络层期望的输入通道数量。
            with_uncertainty: 控制对应功能是否启用的布尔标志。
            unc_type: 不确定性表示类型，例如 NLL 或 DER。
            force_out_dim: 可选预测头强制输出通道数；用于独立不确定性分支。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256, with_uncertainty=with_uncertainty,
                                  unc_type=unc_type, force_out_dim=force_out_dim)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        """执行当前神经网络模块的前向计算并返回下游模块需要的张量。

        参数:
            self: 当前类实例。
            net: RAFT 循环更新器的隐藏状态张量。
            inp: 上下文编码器输出的静态输入特征张量。
            corr: 在当前坐标邻域采样的多尺度相关特征张量。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
            upsample: 是否先将低分辨率预测放大并同步缩放流幅值后计算损失。

        返回值:
            当前网络层或模型计算得到的特征、状态、光流及可选不确定性张量。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # Scale the mask to balance gradients.
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
