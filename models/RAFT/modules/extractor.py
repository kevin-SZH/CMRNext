"""RGB 和 LiDAR 深度图特征编码器。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """实现带可选下采样投影的两层残差特征块。

    主要职责:
        实现带可选下采样投影的两层残差特征块，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            in_planes: 残差块输入特征通道数量。
            planes: 残差块中间特征通道数量。
            norm_fn: 归一化层类型名称，例如 group、batch、instance 或 none。
            stride: 卷积或残差块的空间步幅。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


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
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    """实现一乘一、三乘三、一乘一结构的瓶颈残差特征块。

    主要职责:
        实现一乘一、三乘三、一乘一结构的瓶颈残差特征块，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            in_planes: 残差块输入特征通道数量。
            planes: 残差块中间特征通道数量。
            norm_fn: 归一化层类型名称，例如 group、batch、instance 或 none。
            stride: 卷积或残差块的空间步幅。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


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
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    """使用残差块把 RGB 或 LiDAR 输入编码为八分之一分辨率特征。

    主要职责:
        使用残差块把 RGB 或 LiDAR 输入编码为八分之一分辨率特征，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, in_feat=3):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            output_dim: 编码器或预测头输出的通道数量。
            norm_fn: 归一化层类型名称，例如 group、batch、instance 或 none。
            dropout: 训练期随机失活概率，范围为 0 到 1。
            in_feat: 编码器输入通道数量。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(in_feat, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # Output convolution.
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """完成 RGB 和 LiDAR 深度图特征编码器 中的 `_make_layer` 操作。

        参数:
            self: 当前类实例。
            dim: 当前编码阶段输出的特征通道数量。
            stride: 卷积或残差块的空间步幅。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # Merge batch dimensions for list input.
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
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    """使用瓶颈块构建参数量更小的八分之一分辨率特征编码器。

    主要职责:
        使用瓶颈块构建参数量更小的八分之一分辨率特征编码器，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, in_feat=3):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            output_dim: 编码器或预测头输出的通道数量。
            norm_fn: 归一化层类型名称，例如 group、batch、instance 或 none。
            dropout: 训练期随机失活概率，范围为 0 到 1。
            in_feat: 编码器输入通道数量。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(in_feat, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """完成 RGB 和 LiDAR 深度图特征编码器 中的 `_make_layer` 操作。

        参数:
            self: 当前类实例。
            dim: 当前编码阶段输出的特征通道数量。
            stride: 卷积或残差块的空间步幅。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # Merge batch dimensions for list input.
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
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
