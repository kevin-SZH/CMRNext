"""稠密或稀疏光流训练数据增强。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import numpy as np
import random
import math
import cv2
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F

class FlowAugmentor:
    """对稠密光流图像对同步执行颜色、遮挡、缩放、裁剪和翻转增强。

    主要职责:
        对稠密光流图像对同步执行颜色、遮挡、缩放、裁剪和翻转增强，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            crop_size: 增强后裁剪尺寸，按高度、宽度排列，单位为像素。
            min_scale: 随机缩放指数的下界。
            max_scale: 随机缩放指数的上界。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        self.crop_size = crop_size
        self.augcolor = torchvision.transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.4, 
            hue=0.5/3.14)

        self.asymmetric_color_aug_prob = 0.2
        self.spatial_aug_prob = 0.8
        self.eraser_aug_prob = 0.5

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_stretch = 0.2
        self.stretch_prob = 0.8
        self.margin = 20

    def color_transform(self, img1, img2):

        """完成 稠密或稀疏光流训练数据增强 中的 `color_transform` 操作。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.augcolor(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.augcolor(Image.fromarray(img2)), dtype=np.uint8)

        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.augcolor(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """完成 稠密或稀疏光流训练数据增强 中的 `eraser_transform` 操作。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。
            bounds: 遮挡矩形边长的最小值与最大值，单位为像素。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # Sample scale.

        """完成 稠密或稀疏光流训练数据增强 中的 `spatial_transform` 操作。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        max_scale = self.max_scale
        min_scale = max(min_scale, self.min_scale)
        
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # Rescale images.
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if np.random.rand() < 0.5: # Horizontal flip.
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]

        if np.random.rand() < 0.1: # Vertical flip.
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(-self.margin, img1.shape[0] - self.crop_size[0] + self.margin)
        x0 = np.random.randint(-self.margin, img1.shape[1] - self.crop_size[1] + self.margin)
        
        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        """使当前对象可调用，并对输入执行该类定义的数据读取、变换或相关查询。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class FlowAugmentorKITTI:
    """对 KITTI 稀疏光流样本同步执行颜色、缩放、裁剪和翻转增强。

    主要职责:
        对 KITTI 稀疏光流样本同步执行颜色、缩放、裁剪和翻转增强，并集中维护该职责所需状态。

    返回值:
        类实例由构造方法创建；各公共方法的返回值见对应方法说明。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5):
        """初始化当前实例的配置、层结构、索引或运行状态。

        参数:
            self: 当前类实例。
            crop_size: 增强后裁剪尺寸，按高度、宽度排列，单位为像素。
            min_scale: 随机缩放指数的下界。
            max_scale: 随机缩放指数的上界。

        返回值:
            无显式返回值；初始化结果保存在当前实例中。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        self.crop_size = crop_size
        self.augcolor = torchvision.transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)

        self.max_scale = max_scale
        self.min_scale = min_scale

        self.spatial_aug_prob = 0.8
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        """完成 稠密或稀疏光流训练数据增强 中的 `color_transform` 操作。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.augcolor(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        """完成 稠密或稀疏光流训练数据增强 中的 `eraser_transform` 操作。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        """完成 稠密或稀疏光流训练数据增强 中的 `resize_sparse_flow_map` 操作。

        参数:
            self: 当前类实例。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
            valid: 光流监督有效区域掩码。
            fx: 水平方向缩放倍率。
            fy: 垂直方向缩放倍率。

        返回值:
            计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # Sample scale.

        """完成 稠密或稀疏光流训练数据增强 中的 `spatial_transform` 操作。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
            valid: 光流监督有效区域掩码。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # Rescale images.
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if np.random.rand() < 0.5: # Horizontal flip.
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]
            valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow, valid


    def __call__(self, img1, img2, flow, valid):
        """使当前对象可调用，并对输入执行该类定义的数据读取、变换或相关查询。

        参数:
            self: 当前类实例。
            img1: 需要与第二幅图同步增强的第一幅图像数组。
            img2: 需要与第一幅图同步增强的第二幅图像数组。
            flow: 二维光流张量或数组，最后一维或通道维为 u、v，单位为像素。
            valid: 光流监督有效区域掩码。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

        坐标系和数据约定:
            网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
        """
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
