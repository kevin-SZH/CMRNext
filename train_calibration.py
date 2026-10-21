"""CMRNext 训练、验证、输入构造和 checkpoint 管理。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import argparse
import json
from functools import partial
from itertools import chain

import glob
import os
import random
import re
import time
from pathlib import Path

import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import mathutils
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import visibility
import wandb
from skimage import io
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
from tqdm import tqdm

# Global TensorBoard writer, initialized only on rank 0.
TB_WRITER = None

from datasets.emd_extrinsic_calib import (
    DatasetEMDExtrinsicCalib,
    load_emd_training_config,
)
from datasets.emd_dynamic_nominal import (
    DatasetEMDDynamicNominalExtrinsicCalib,
    load_emd_dynamic_nominal_config,
)
from datasets.emd_fixed_t0_online import (
    DatasetEMDFixedT0OnlineExtrinsicCalib,
    DatasetEMDPerTiOnlineExtrinsicCalib,
    load_fixed_t0_online_perturbation_config,
    load_per_ti_online_perturbation_config,
)
from camera_model import CameraModel
from flow_losses import RAFT_loss2
from utils import resize_dense_vector
from models.get_model import get_model
from utils import merge_inputs, rotate_back, get_flow_zforward, \
    init_logger, get_ECE
from flow_vis import flow_to_color

torch.backends.cudnn.benchmark = True


EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    """根据基础 seed、工作进程编号和训练轮次初始化各随机数生成器。

    参数:
        worker_id: DataLoader 工作进程编号。
        epoch: 当前训练轮次，从零开始计数。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    seed = seed + worker_id + epoch * 100
    seed = seed % (2 ** 32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def uncertainty_to_color(_tensor, mask=None):
    """完成 CMRNext 训练、验证、输入构造和 checkpoint 管理 中的 `uncertainty_to_color` 操作。

    参数:
        _tensor: 需要转换为伪彩色的不确定性张量。
        mask: 有效像素布尔或数值掩码；非零位置参与计算。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    total_uncertainty = _tensor[0, :, :] + _tensor[1, :, :]
    total_uncertainty = torch.from_numpy(total_uncertainty)
    if mask is not None:
        total_uncertainty = total_uncertainty * mask[0]
    total_uncertainty -= torch.min(total_uncertainty)
    total_uncertainty /= torch.max(total_uncertainty)
    jet = cm.get_cmap('jet')
    color = jet(total_uncertainty)
    color = color[:, :, :3]
    return color


def _random_crop_offset(_config, key, maximum):
    """完成 CMRNext 训练、验证、输入构造和 checkpoint 管理 中的 `_random_crop_offset` 操作。

    参数:
        _config: 运行配置字典，包含数据、网络、几何和输出选项。
        key: 待读取的配置、指标或字典字段名。
        maximum: 随机裁剪或索引允许的最大整数值，包含端点。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    bounds = _config.get(key)
    if bounds is None:
        return random.randint(0, maximum)
    return random.randint(bounds[0], bounds[1])


def _mask_flow_endpoints_inside(flow_img, flow_mask):
    """排除目标像素落到当前裁剪视野外的稀疏光流监督。

    参数:
        flow_img: H×W×2 的 GT 光流，通道顺序为水平 u、垂直 v，单位为像素。
        flow_mask: H×W 有效源像素掩码；非零位置代表存在 LiDAR 投影点。

    返回值:
        二元组 (filtered_flow, filtered_mask)；形状与输入一致，目标终点
        越界的位置被清零。

    异常和副作用:
        输入形状不匹配时抛出 ValueError；函数克隆输入后修改，不改变调用
        方持有的原张量。

    坐标系和外参约定:
        源像素与目标像素都位于同一裁剪图像坐标系，目标位置等于
        source_uv + flow_gt；该筛选不改变 Ti @ inverse(T0) 左乘残差。
    """
    if flow_img.ndim != 3 or flow_img.shape[-1] != 2:
        raise ValueError("flow_img 形状必须为 H×W×2")
    if flow_mask.shape != flow_img.shape[:2]:
        raise ValueError("flow_mask 形状必须与 flow_img 的空间尺寸一致")
    filtered_flow = flow_img.clone()
    filtered_mask = flow_mask.clone()
    # Check only sparse source pixels.
    source_y, source_x = torch.where(filtered_mask != 0)
    if source_y.numel() == 0:
        return filtered_flow, filtered_mask
    target_x = source_x.to(filtered_flow.dtype) + filtered_flow[
        source_y, source_x, 0]
    target_y = source_y.to(filtered_flow.dtype) + filtered_flow[
        source_y, source_x, 1]
    height, width = filtered_mask.shape
    endpoint_valid = (
        (target_x >= 0) & (target_x < width)
        & (target_y >= 0) & (target_y < height))
    invalid_y = source_y[~endpoint_valid]
    invalid_x = source_x[~endpoint_valid]
    filtered_mask[invalid_y, invalid_x] = 0
    filtered_flow[invalid_y, invalid_x] = 0
    return filtered_flow, filtered_mask


def _configure_emd_spatial_geometry(_config, contract):
    """把 EMD 在线缩放、训练裁剪和固定评估裁剪写入运行配置。

    参数:
        _config: 可变训练配置字典；img_shape 为训练 crop 的 H×W。
        contract: 从 EMD YAML 解析出的传感器和空间预处理契约。

    返回值:
        无显式返回值；派生尺寸、裁剪范围和缩放倍率原地写入 _config。

    异常和副作用:
        训练或评估尺寸不是 64 的倍数、裁剪超出 resize 图像，或 YAML 随机
        范围非法时抛出 ValueError。不读取样本、不创建输出文件。

    坐标系和外参约定:
        resize 后的像素坐标使用同步缩放的内参；训练 crop 可移动坐标原点，
        固定评估 crop 的主点偏移由评估器处理，外参仍为 Ti @ inverse(T0)。
    """
    train_height, train_width = (int(value) for value in _config["img_shape"])
    if train_height % 64 != 0 or train_width % 64 != 0:
        raise ValueError("EMD --img_shape 的高度和宽度必须是 64 的倍数")
    resize_height = int(contract.resize_height)
    resize_width = int(contract.resize_width)
    crop_ranges = (
        ("emd_crop_top_range", contract.crop_top_range,
         resize_height - train_height),
        ("emd_crop_left_range", contract.crop_left_range,
         resize_width - train_width),
    )
    for key, bounds, maximum in crop_ranges:
        if maximum < 0:
            raise ValueError(
                f"EMD training crop {_config['img_shape']} exceeds "
                f"resize shape {[resize_height, resize_width]}")
        if bounds is not None and bounds[1] > maximum:
            raise ValueError(
                f"{key}={bounds} exceeds the valid [0, {maximum}] range")
        _config[key] = bounds

    crop_values = (
        contract.evaluation_crop_top,
        contract.evaluation_crop_left,
        contract.evaluation_crop_height,
        contract.evaluation_crop_width,
    )
    if all(value is None for value in crop_values):
        evaluation_crop = None
    elif any(value is None for value in crop_values):
        raise ValueError("必须完整指定 EMD 评估裁剪区域")
    else:
        crop_top, crop_left, crop_height, crop_width = (
            int(value) for value in crop_values)
        if crop_height % 64 != 0 or crop_width % 64 != 0:
            raise ValueError(
                "EMD 评估裁剪的高度和宽度必须是 64 的倍数")
        if (crop_top + crop_height > resize_height
                or crop_left + crop_width > resize_width):
            raise ValueError("EMD evaluation crop exceeds resize shape")
        evaluation_crop = {
            "top": crop_top,
            "left": crop_left,
            "height": crop_height,
            "width": crop_width,
        }

    _config["emd_source_image_shape"] = [
        int(contract.camera_height), int(contract.camera_width)]
    _config["emd_resize_shape"] = [resize_height, resize_width]
    _config["emd_image_scale_xy"] = [
        resize_width / contract.camera_width,
        resize_height / contract.camera_height,
    ]
    _config["emd_evaluation_crop"] = evaluation_crop
    _config["emd_filter_cropped_flow_endpoints"] = (
        evaluation_crop is not None)


def prepare_input(
        _config, device, idx, img_shape, mean, sample, std,
        spatial_mode="train"):
    """把一帧 RGB 和外参扰动点云构造成网络深度输入、稀疏 GT 光流及有效掩码。

    参数:
        _config: 运行配置字典，包含数据、网络、几何和输出选项。
        device: PyTorch 计算设备，例如 CUDA 设备或 CPU。
        idx: 样本、批次或可视化项索引，从零开始。
        img_shape: 当前模式的网络输入尺寸，按高度、宽度排列，单位为像素。
        mean: RGB 归一化均值张量。
        sample: DataLoader 返回的批次样本字典。
        std: RGB 归一化标准差张量。
        spatial_mode: train 使用随机 crop，eval 使用 YAML 声明的固定 crop。

    返回值:
        四元组 (lidar_input, flow_gt, flow_mask, rgb_input)，分别为深度输入、GT 光流、有效掩码和 RGB 输入。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]
    sample['point_cloud'][idx] = sample['point_cloud'][idx].to(device)
    pc_rotated = sample['point_cloud'][idx].clone()
    reflectance = None
    if _config['use_reflectance']:
        reflectance = sample['reflectance'][idx].to(device)
    R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(sample['tr_error'][idx])
    try:
        RT = T @ R
    except:
        RT = T * R
    pc_rotated = rotate_back(pc_rotated, RT)
    cam_params = sample['calib'][idx].to(device)
    cam_model = CameraModel().set_intrinsics(cam_params)
    uv_lidar, depth, _, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance)
    uv_lidar = uv_lidar.t().int().contiguous()
    depth_img = torch.zeros(real_shape[:2], device=device, dtype=torch.float)
    depth_img += 1000.
    depth_img = visibility.depth_image(uv_lidar, depth, depth_img, uv_lidar.shape[0], real_shape[1],
                                       real_shape[0])
    temp_index = (depth_img == 1000.)
    depth_img[temp_index] = 0.
    depth_img_no_occlusion = depth_img
    uv_lidar = uv_lidar.long()
    indexes = depth_img_no_occlusion[uv_lidar[:, 1], uv_lidar[:, 0]] == depth
    if _config['use_reflectance']:
        refl_img = torch.zeros(real_shape[:2], device=device, dtype=torch.float)
        refl_img[uv_lidar[indexes, 1], uv_lidar[indexes, 0]] = refl[0, indexes]
    depth_img_no_occlusion /= _config['max_depth']
    depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)
    if _config['use_reflectance']:
        depth_img_no_occlusion = torch.cat((depth_img_no_occlusion, refl_img.unsqueeze(0)))
    uv_lidar = uv_lidar[indexes]
    flow, _, new_indexes = get_flow_zforward(uv_lidar.float(), depth[indexes], RT, cam_model,
                                             [real_shape[0], real_shape[1], 3],
                                             scale_flow=False, reverse=False,
                                             get_valid_indexes=True)
    uv_flow = uv_lidar
    uv_flow = uv_flow[new_indexes].clone()
    flow = flow[new_indexes].clone()
    rgb = sample['rgb'][idx].to(device)
    if _config['normalize_images']:
        rgb = rgb / 255.
        rgb = (rgb - mean) / std
    rgb = rgb.permute(2, 0, 1)
    flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device=device, dtype=torch.float)
    flow_img[uv_flow[:, 1], uv_flow[:, 0]] = flow
    flow_mask = torch.zeros((real_shape[0], real_shape[1]), device=device, dtype=torch.int)
    flow_mask[uv_flow[:, 1], uv_flow[:, 0]] = 1
    target_height, target_width = (int(value) for value in img_shape)
    if target_height > real_shape[0] or target_width > real_shape[1]:
        raise ValueError(
            f"EMD input shape {img_shape} exceeds resized frame {real_shape[:2]}")
    evaluation_crop = _config.get("emd_evaluation_crop")
    if spatial_mode == "eval" and evaluation_crop is not None:
        crop_top = int(evaluation_crop["top"])
        crop_left = int(evaluation_crop["left"])
        target_height = int(evaluation_crop["height"])
        target_width = int(evaluation_crop["width"])
    else:
        crop_top = _random_crop_offset(
            _config, "emd_crop_top_range",
            real_shape[0] - target_height)
        crop_left = _random_crop_offset(
            _config, "emd_crop_left_range",
            real_shape[1] - target_width)
    crop_bottom = crop_top + target_height
    crop_right = crop_left + target_width
    rgb = rgb[:, crop_top:crop_bottom, crop_left:crop_right]
    depth_img_no_occlusion = depth_img_no_occlusion[
        :, crop_top:crop_bottom, crop_left:crop_right]
    flow_img = flow_img[
        crop_top:crop_bottom, crop_left:crop_right, :]
    flow_mask = flow_mask[
        crop_top:crop_bottom, crop_left:crop_right]
    if _config.get("emd_filter_cropped_flow_endpoints", False):
        flow_img, flow_mask = _mask_flow_endpoints_inside(
            flow_img, flow_mask)
    if _config['fourier_levels'] >= 0:
        depth_img_no_occlusion = depth_img_no_occlusion.squeeze()
        mask = (depth_img_no_occlusion > 0).clone()
        fourier_feats = []
        for L in range(_config['fourier_levels']):
            fourier_feat = depth_img_no_occlusion * np.pi * 2 ** L
            fourier_feats.append(fourier_feat.sin())
            fourier_feats.append(fourier_feat.cos())
        depth_img_no_occlusion = torch.stack(fourier_feats + [depth_img_no_occlusion])
        depth_img_no_occlusion = depth_img_no_occlusion * mask.unsqueeze(0)
    return depth_img_no_occlusion, flow_img, flow_mask, rgb


# Training
def train(model, optimizer, scaler, rgb_img, lidar_img, target_flow, target_mask, _config):
    """执行一次 CMRNext 前向传播、光流损失计算、反向传播和优化器更新。

    参数:
        model: 待训练、验证或评估的 PyTorch 模型。
        optimizer: 用于更新模型参数的 PyTorch 优化器。
        scaler: 自动混合精度梯度缩放器。
        rgb_img: 批量 RGB 输入张量，形状为 B×3×H×W。
        lidar_img: 批量 LiDAR 深度输入张量，形状为 B×C×H×W。
        target_flow: 光流监督张量或按尺度组织的光流监督列表。
        target_mask: 与光流监督对应的有效像素掩码。
        _config: 运行配置字典，包含数据、网络、几何和输出选项。

    返回值:
        分离计算图后的总损失与末次预测 EPE。

    异常和副作用:
        可能占用 GPU、更新模型或优化器状态并写入日志和 checkpoint；配置或运行依赖无效时抛出异常。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    model.train()

    optimizer.zero_grad()

    with amp.autocast(enabled=_config['amp']):
        # Run inference.
        predicted_flow = model(rgb_img, lidar_img)

        predicted_flow, predicted_uncertainty = predicted_flow

        # Compute the loss.
        flow_loss, metrics = RAFT_loss2(predicted_flow, predicted_uncertainty, target_flow[0], target_mask[0],
                                        upsample=False,
                                        weight_nll=_config['weight_nll'], unc_type=_config['der_type'])

    # Compute EPE.
    epe = metrics['epe']

    total_loss = flow_loss

    # Backpropagate and update.
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer)
    scaler.update()

    del predicted_flow
    return total_loss.detach(), epe.detach()


# Validation
def test(model, rgb_img, lidar_img, target_flow, target_mask, log_image, _config):
    """在不更新参数的条件下执行训练期验证并计算光流与不确定性指标。

    参数:
        model: 待训练、验证或评估的 PyTorch 模型。
        rgb_img: 批量 RGB 输入张量，形状为 B×3×H×W。
        lidar_img: 批量 LiDAR 深度输入张量，形状为 B×C×H×W。
        target_flow: 光流监督张量或按尺度组织的光流监督列表。
        target_mask: 与光流监督对应的有效像素掩码。
        log_image: 是否为当前验证批次生成 TensorBoard 或 W&B 可视化。
        _config: 运行配置字典，包含数据、网络、几何和输出选项。

    返回值:
        验证损失、EPE、不确定性校准指标及 F1 指标组成的元组。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    global EPOCH
    model.eval()

    # Run inference.
    with torch.no_grad():
        with amp.autocast(enabled=False):
            predicted_flow = model(rgb_img, lidar_img)

    predicted_flow, predicted_uncertainty = predicted_flow

    # Compute the loss.
    flow_loss, metrics = RAFT_loss2(predicted_flow, predicted_uncertainty, target_flow[0], target_mask[0],
                                    upsample=False,
                                    weight_nll=_config['weight_nll'], unc_type=_config['der_type'])

    total_loss = flow_loss

    # Compute EPE.
    gt = torch.cat((target_flow[0], target_mask[0][:, 0:1, :, :]), dim=1)
    epe = metrics['epe']
    f1 = metrics['f1']

    # Compute expected calibration error.
    ece_dict, ece_u, ece_v = None, None, None
    if _config['uncertainty']:
        ece_u, ece_v = get_ECE(predicted_flow[-1], predicted_uncertainty[-1], target_flow[0], target_mask[0],
                               loss_type=_config['der_type'])

    # Log visualizations.
    if log_image:
        rgb_images = []
        predicted_flow_images = []
        target_flow_images = []
        uncertainties = []
        uncertainties2 = []
        # Keep HWC float copies in [0, 1] for TensorBoard.
        tb_rgb_np = []
        tb_pred_flow_np = []
        tb_gt_flow_np = []
        tb_unc_np = []
        tb_unc_masked_np = []
        upsampled = resize_dense_vector(predicted_flow[-1], gt.size(2), gt.size(3))
        for i in range(min(rgb_img.shape[0], 16)):
            img_show = rgb_img[i].permute(1, 2, 0).cpu().numpy()
            std = [0.229, 0.224, 0.225]
            mean = [0.485, 0.456, 0.406]
            if _config['normalize_images']:
                img_show = img_show * std + mean
            else:
                img_show = img_show / 255.
            img_show = img_show.clip(0, 1)
            tb_rgb_np.append(img_show)
            if _config['wandb']:
                if predicted_uncertainty[-1] is not None:
                    rgb_images.append(wandb.Image(img_show, grouping=5))
                else:
                    rgb_images.append(wandb.Image(img_show, grouping=3))

            flowcolored_pred = flow_to_color(upsampled[i].permute(1, 2, 0).cpu().numpy())
            tb_pred_flow_np.append(flowcolored_pred)
            if _config['wandb']:
                predicted_flow_images.append(wandb.Image(flowcolored_pred))

            flowcolored_gt = flow_to_color((gt[i].permute(1, 2, 0)[:, :, :2]).cpu().numpy())
            tb_gt_flow_np.append(flowcolored_gt)
            if _config['wandb']:
                target_flow_images.append(wandb.Image(flowcolored_gt))
            if predicted_uncertainty[-1] is not None:
                unc_img = uncertainty_to_color((predicted_uncertainty[-1][i]).cpu().numpy())
                unc_masked = uncertainty_to_color((target_mask[0][i] * predicted_uncertainty[-1][i]).cpu().numpy())
                tb_unc_np.append(unc_img)
                tb_unc_masked_np.append(unc_masked)
                if _config['wandb']:
                    uncertainties.append(wandb.Image(unc_img))
                    uncertainties2.append(wandb.Image(unc_masked))
        if _config['wandb']:
            if predicted_uncertainty[-1] is not None:
                images_for_wandb = list(chain.from_iterable(zip(rgb_images, predicted_flow_images,
                                                                target_flow_images, uncertainties, uncertainties2)))
            else:
                images_for_wandb = list(chain.from_iterable(zip(rgb_images, predicted_flow_images,
                                                                target_flow_images)))
            wandb.log({'examples': images_for_wandb}, commit=False)

        # Log TensorBoard images once per epoch.
        if TB_WRITER is not None:
            def _stack_for_tb(np_list):
                # Convert HWC images to NCHW tensors in [0, 1].
                """完成 CMRNext 训练、验证、输入构造和 checkpoint 管理 中的 `_stack_for_tb` 操作。

                参数:
                    np_list: 待堆叠写入 TensorBoard 的 NumPy 图像数组列表。

                返回值:
                    返回由 `实现中的返回表达式` 表示的计算结果。

                异常和副作用:
                    输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

                坐标系和数据约定:
                    EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
                """
                arr = np.stack(np_list, axis=0).astype(np.float32)
                if arr.max() > 1.5:  # Normalize uint8-style flow colors.
                    arr = arr / 255.0
                arr = arr.clip(0, 1)
                return torch.from_numpy(arr).permute(0, 3, 1, 2)

            if len(tb_rgb_np) > 0:
                TB_WRITER.add_images('val/rgb', _stack_for_tb(tb_rgb_np), EPOCH)
                TB_WRITER.add_images('val/flow_pred', _stack_for_tb(tb_pred_flow_np), EPOCH)
                TB_WRITER.add_images('val/flow_gt', _stack_for_tb(tb_gt_flow_np), EPOCH)
                if len(tb_unc_np) > 0:
                    TB_WRITER.add_images('val/uncertainty', _stack_for_tb(tb_unc_np), EPOCH)
                    TB_WRITER.add_images('val/uncertainty_masked', _stack_for_tb(tb_unc_masked_np), EPOCH)

    del predicted_flow
    return total_loss.detach(), epe.detach(), ece_u, ece_v, ece_dict, f1


def main(gpu, _config, common_seed, world_size):
    """在一个 GPU 进程中执行 CMRNext 训练。

    参数:
        gpu: 当前分布式进程使用的本地 GPU 编号。
        _config: 完整训练配置，包含显式场景路径、输出目录和优化器参数。
        common_seed: 所有进程共享的基础随机种子。
        world_size: 参与训练的 GPU 进程总数。

    返回值:
        无显式返回值；训练结束后写出 checkpoint、训练日志和 TensorBoard 事件。

    异常和副作用:
        数据、CUDA、分布式初始化或 checkpoint 不合法时抛出异常。函数会占用
        GPU、创建输出文件并更新模型参数。

    坐标系和外参约定:
        EMD 目标由 Ti @ p_lidar 构造。存储模式以 T0 为输入并监督
        Ti @ inverse(T0)；逐帧 Ti 在线模式直接采样 delta，再以
        inverse(delta) @ Ti 作为输入外参，存储 T0 不参与训练样本。
    """
    global EPOCH
    emd_config = load_emd_training_config(_config['emd_config'])
    emd_dynamic_nominal_config = load_emd_dynamic_nominal_config(
        _config['emd_config'])
    emd_fixed_t0_online_config = load_fixed_t0_online_perturbation_config(
        _config['emd_config'])
    emd_per_ti_online_config = load_per_ti_online_perturbation_config(
        _config['emd_config'])
    enabled_online_modes = sum((
        emd_dynamic_nominal_config.enabled,
        emd_fixed_t0_online_config.enabled,
        emd_per_ti_online_config.enabled,
    ))
    if enabled_online_modes > 1:
        raise ValueError(
            "动态 T0、固定 T0 与逐帧 Ti 在线扰动模式互斥")
    if _config['emd_level'] not in set(emd_config.curriculum.values()):
        raise ValueError(
            f"EMD level {_config['emd_level']!r} is not declared by the "
            "configured curriculum")
    _configure_emd_spatial_geometry(
        _config, emd_config.contract)
    _config['emd_dynamic_nominal'] = emd_dynamic_nominal_config.enabled
    _config['emd_fixed_t0_online'] = emd_fixed_t0_online_config.enabled
    _config['emd_per_ti_online'] = emd_per_ti_online_config.enabled
    if emd_dynamic_nominal_config.enabled:
        dynamic_level = emd_dynamic_nominal_config.levels[
            _config['emd_level']]
        _config['emd_dynamic_nominal_translation_alpha_t'] = (
            dynamic_level.translation_alpha_t)
        _config['emd_dynamic_nominal_translation_min_floor_m'] = (
            dynamic_level.translation_min_floor_m)
        _config['emd_dynamic_nominal_max_rotation_deg'] = (
            dynamic_level.max_rotation_deg)
    if emd_fixed_t0_online_config.enabled:
        _config['emd_fixed_t0_online_max_translation_component_m'] = (
            emd_fixed_t0_online_config.max_translation_component_m)
        _config['emd_fixed_t0_online_max_rotation_euler_component_deg'] = (
            emd_fixed_t0_online_config.max_rotation_euler_component_deg)
        _config['emd_fixed_t0_online_seed'] = int(common_seed)
    if emd_per_ti_online_config.enabled:
        _config['emd_per_ti_online_max_translation_component_m'] = (
            emd_per_ti_online_config.max_translation_component_m)
        _config['emd_per_ti_online_max_rotation_euler_component_deg'] = (
            emd_per_ti_online_config.max_rotation_euler_component_deg)
        _config['emd_per_ti_online_seed'] = int(common_seed)
    rank = gpu

    # Initialize distributed training.
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Seed all random sources.
    local_seed = (common_seed + common_seed ** gpu) ** 2
    local_seed = local_seed % (2 ** 32 - 1)
    np.random.seed(common_seed)
    torch.random.manual_seed(common_seed)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    print(f"Process {rank}, seed {common_seed}")
    
    # Initialize Weights & Biases.
    wandb_run_id = None
    if _config['wandb'] and rank == 0:
        if _config['resume_id'] is None:
            wandb.init(config=_config)
        else:
            wandb.init(id=_config['resume_id'], resume="must")
        wandb_run_id = wandb.run.id
        print("RUN ID: ", wandb_run_id)

    if rank == 0:
        os.makedirs(_config['run_output_dir'], exist_ok=True)
        logger = init_logger(
            _config['log_file_resolved'], _config['resume'], True)

    # Training and validation canvases differ but remain multiples of 64.
    img_shape = _config['img_shape']
    evaluation_crop = _config.get("emd_evaluation_crop")
    eval_img_shape = (
        [evaluation_crop["height"], evaluation_crop["width"]]
        if evaluation_crop is not None else img_shape)

    _config["savemodel"] = _config['run_output_dir']

    # Initialize TensorBoard only on rank 0.
    global TB_WRITER
    if _config['tensorboard'] and rank == 0:
        # Store events with the run and expose them through a stable symlink.
        tb_root = _config['tb_logdir_resolved']
        tb_mount_path = _mount_tensorboard_logdir(
            tb_root, _config['tb_mount_root'], _config['run_name'])
        TB_WRITER = SummaryWriter(log_dir=tb_root)
        print(f"TensorBoard logging to: {tb_root}")
        print(f"TensorBoard mounted at: {tb_mount_path}")
    
    # Build training and validation sets.
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    starting_epoch = 0

    model = get_model(_config, img_shape)

    # Load model weights when resuming.
    if _config['weights'] is not None:
        if rank == 0:
            logger.info(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        clean_state_dict = saved_state_dict
        if not _config['unc_freeze'] or _config['finetune']:
            model.load_state_dict(clean_state_dict, strict=True)
        else:
            model.load_state_dict(clean_state_dict, strict=False)
            for name, param in model.named_parameters():
                if 'update_block_unc' not in name:
                    param.requires_grad = False
    
    model.train()
    model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank,
                                    find_unused_parameters=_config['find_unused_parameter'])


    # Initialize the optimizer and scheduler.
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
    scheduler = None

    scaler = amp.GradScaler(enabled=_config['amp'])
    if rank == 0 and _config['amp']:
        logger.info("Using Mixed Precision")

    # Restore optimizer and scheduler state.
    resumed_ckpt = None
    resumed_best_val_epe = 10000.0
    resumed_best_checkpoint = None
    if _config['weights'] is not None and _config['resume']:
        resumed_ckpt = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = resumed_ckpt['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        starting_epoch = resumed_ckpt['epoch'] + 1
        stored_best = resumed_ckpt.get('best_val_epe')
        if stored_best is None:
            stored_best = resumed_ckpt.get('val_epe')
        if stored_best is not None and np.isfinite(float(stored_best)):
            resumed_best_val_epe = float(stored_best)

        # A checkpoint from a skipped-validation epoch has no val_epe.
        # Recover the historical best from stable checkpoint names.
        best_pattern = re.compile(r'^checkpoint_\d+_([0-9]+(?:\.[0-9]+)?)\.tar$')
        best_checkpoint_metric = float('inf')
        for checkpoint_path in glob.glob(
                os.path.join(_config['savemodel'], 'checkpoint_*.tar')):
            match = best_pattern.match(os.path.basename(checkpoint_path))
            if match is None:
                continue
            candidate = float(match.group(1))
            if candidate < best_checkpoint_metric:
                best_checkpoint_metric = candidate
                resumed_best_checkpoint = checkpoint_path
            if candidate < resumed_best_val_epe:
                resumed_best_val_epe = candidate

    if rank == 0:
        logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if _config['wandb'] and rank == 0:
        wandb.watch(model)

    start_full_time = time.time()
    BEST_VAL_EPE = resumed_best_val_epe if (_config['weights'] is not None and _config['resume']) else 10000.
    old_save_filename = resumed_best_checkpoint

    def _atomic_save(obj, path):
        """完成 CMRNext 训练、验证、输入构造和 checkpoint 管理 中的 `_atomic_save` 操作。

        参数:
            obj: 需要原子写入 checkpoint 的可序列化对象。
            path: 输入或输出文件路径。

        返回值:
            无显式返回值；函数完成所述状态更新、校验或输出操作。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

        坐标系和数据约定:
            EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
        """
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, path)

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    total_iter = 0
    for epoch in range(starting_epoch, _config['epochs']):

        if emd_fixed_t0_online_config.enabled:
            dataset_train = DatasetEMDFixedT0OnlineExtrinsicCalib(
                _config['emd_train_scene_paths'],
                level=_config['emd_level'],
                contract=emd_config.contract,
                online_config=emd_fixed_t0_online_config,
                seed=common_seed,
                train=True,
                normalize_images=_config['normalize_images'],
                use_reflectance=_config['use_reflectance'],
            )
        elif emd_per_ti_online_config.enabled:
            dataset_train = DatasetEMDPerTiOnlineExtrinsicCalib(
                _config['emd_train_scene_paths'],
                level=_config['emd_level'],
                contract=emd_config.contract,
                online_config=emd_per_ti_online_config,
                seed=common_seed,
                train=True,
                normalize_images=_config['normalize_images'],
                use_reflectance=_config['use_reflectance'],
            )
        elif emd_dynamic_nominal_config.enabled:
            dataset_train = DatasetEMDDynamicNominalExtrinsicCalib(
                _config['emd_train_scene_paths'],
                level=_config['emd_level'],
                contract=emd_config.contract,
                dynamic_config=emd_dynamic_nominal_config,
                train=True,
                normalize_images=_config['normalize_images'],
                use_reflectance=_config['use_reflectance'],
            )
        else:
            dataset_train = DatasetEMDExtrinsicCalib(
                _config['emd_train_scene_paths'],
                level=_config['emd_level'],
                contract=emd_config.contract,
                train=True,
                normalize_images=_config['normalize_images'],
                use_reflectance=_config['use_reflectance'],
            )
        if rank == 0:
            logger.info(
                "[emd] train level=%s samples=%d scenes=%s "
                "dynamic_nominal=%s fixed_t0_online=%s per_ti_online=%s "
                "test_scenes=%s",
                _config['emd_level'], len(dataset_train),
                dataset_train.scene_paths,
                emd_dynamic_nominal_config.enabled,
                emd_fixed_t0_online_config.enabled,
                emd_per_ti_online_config.enabled,
                _config.get('emd_test_scene_paths'),
            )

        if epoch == starting_epoch:
            total_iter = starting_epoch * len(dataset_train)

        if _config['val_every'] <= 0:
            dataset_val = None
            if rank == 0 and epoch == starting_epoch:
                logger.info("Validation disabled; no validation dataset will be loaded")
        else:
            if emd_fixed_t0_online_config.enabled:
                dataset_val = DatasetEMDFixedT0OnlineExtrinsicCalib(
                    _config['emd_val_scene_paths'],
                    level=_config['emd_level'],
                    contract=emd_config.contract,
                    online_config=emd_fixed_t0_online_config,
                    seed=common_seed,
                    train=False,
                    normalize_images=_config['normalize_images'],
                    use_reflectance=_config['use_reflectance'],
                )
            else:
                dataset_val = DatasetEMDExtrinsicCalib(
                    _config['emd_val_scene_paths'],
                    level=_config['emd_level'],
                    contract=emd_config.contract,
                    train=False,
                    normalize_images=_config['normalize_images'],
                    use_reflectance=_config['use_reflectance'],
                )
            if rank == 0:
                logger.info(
                    "[emd] val level=%s samples=%d scenes=%s",
                    _config['emd_level'], len(dataset_val),
                    dataset_val.scene_paths,
                )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank,
            seed=common_seed
        )
        val_sampler = None
        if dataset_val is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val,
                num_replicas=world_size,
                rank=rank,
                seed=common_seed
            )
        train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        init_fn = partial(_init_fn, epoch=epoch, seed=local_seed)
        loader_common = {
            'num_workers': num_worker,
            'worker_init_fn': init_fn,
            'collate_fn': merge_inputs,
            'pin_memory': True,
        }
        if num_worker > 0:
            loader_common['prefetch_factor'] = _config['prefetch_factor']

        TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                     shuffle=False,
                                                     batch_size=batch_size,
                                                     drop_last=True,
                                                     sampler=train_sampler,
                                                     **loader_common)

        TestImgLoader = None
        if dataset_val is not None:
            TestImgLoader = torch.utils.data.DataLoader(
                dataset=dataset_val,
                shuffle=False,
                batch_size=_config['eval_batch_size'],
                drop_last=True,
                sampler=val_sampler,
                **loader_common)

        if rank == 0:
            logger.info(f'Len Train: {len(TrainImgLoader)}')
            logger.info(f'This is {epoch}-th epoch')

        train_steps_per_epoch = len(TrainImgLoader)
        if train_steps_per_epoch <= 0:
            raise ValueError("训练 DataLoader 为空，无法创建调度器或开始训练")

        if epoch == starting_epoch and _config['scheduler'] and _config['scheduler'].startswith('cycle'):
            scheduler_total_steps = _config['epochs'] * train_steps_per_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, _config['BASE_LEARNING_RATE'],
                                                            total_steps=scheduler_total_steps,
                                                            pct_start=0.4, div_factor=10,
                                                            final_div_factor=100000)
            ckpt_scheduler = resumed_ckpt.get('scheduler') if resumed_ckpt is not None else None
            ckpt_total_steps = ckpt_scheduler.get('total_steps') if ckpt_scheduler is not None else None
            if ckpt_scheduler is not None and not _config['rebuild_scheduler'] and ckpt_total_steps == scheduler_total_steps:
                scheduler.load_state_dict(ckpt_scheduler)
                if rank == 0:
                    logger.info("Restored scheduler state from checkpoint")
            elif rank == 0:
                if ckpt_scheduler is not None and not _config['rebuild_scheduler']:
                    logger.info("Checkpoint scheduler total_steps=%s does not match current total_steps=%s; "
                                "rebuilt scheduler for current DataLoader length.",
                                ckpt_total_steps, scheduler_total_steps)
                elif _config['rebuild_scheduler'] and resumed_ckpt is not None:
                    logger.info("Rebuilt OneCycleLR on current DataLoader length; scheduler state in ckpt ignored.")
                logger.info("OneCycleLR total_steps=%d, steps_per_epoch=%d",
                            scheduler_total_steps, train_steps_per_epoch)

        EPOCH = epoch
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        local_epe = 0.
        total_train_epe = 0.
        if epoch != starting_epoch:
            if scheduler is not None and _config['wandb'] and rank == 0:
                wandb.log({'LR': scheduler.get_last_lr()[0]}, commit=False)
            if scheduler is not None and TB_WRITER is not None:
                TB_WRITER.add_scalar('train/LR', scheduler.get_last_lr()[0], epoch)

        time_for_N_it = time.time()
        batch_idx = 0
        # Train.
        train_iter = enumerate(TrainImgLoader)
        if rank == 0:
            train_iter = tqdm(train_iter, total=len(TrainImgLoader),
                              desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True)
        for batch_idx, sample in train_iter:
            start_time = time.time()
            lidar_input = []
            rgb_input = []

            target_flow1 = []
            target_mask1 = []

            sample['tr_error'] = sample['tr_error'].to(device)
            sample['rot_error'] = sample['rot_error'].to(device)

            for idx in range(len(sample['rgb'])):
                # Project with the current pose.

                with torch.no_grad():
                    depth_img_no_occlusion, flow_img, flow_mask, rgb = prepare_input(_config, device,
                                                                                     idx, img_shape, mean, sample, std)

                flow_img = flow_img.contiguous()
                flow_mask = flow_mask.contiguous()
                rgb_input.append(rgb)
                lidar_input.append(depth_img_no_occlusion)
                target_flow1.append(flow_img.permute(2, 0, 1).clone())
                target_mask1.append(flow_mask.repeat(2, 1, 1).float().clone())

                if _config['debug']:
                    io.imshow(rgb.permute(1, 2, 0).cpu().numpy())
                    io.show()
                    io.imshow(flow_mask.cpu().numpy())
                    io.show()
                    io.imshow(flow_img[:, :, 1].cpu().numpy())
                    io.show()
                    io.imshow(flow_to_color(flow_img.cpu().numpy()))
                    io.show()
                    io.imshow(depth_img_no_occlusion[0].cpu().numpy())
                    io.show()

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            target_flow1 = torch.stack(target_flow1)
            target_mask1 = torch.stack(target_mask1)

            loss, epe = train(model, optimizer, scaler, rgb_input, lidar_input,
                              [target_flow1],
                              [target_mask1],
                              _config)

            dist.reduce(loss, 0)
            dist.reduce(epe, 0)
            if scheduler is not None and _config['scheduler'] and _config['scheduler'].startswith('cycle'):
                scheduler.step()
            if rank == 0:
                loss = loss / world_size
                epe = epe / world_size
                local_loss += loss.item()
                local_epe += epe.item()

                if batch_idx % _config['print_every'] == 0 and batch_idx != 0:
                    logger.info('Iter %d/%d training loss = %.3f , epe = %.3f, time = %.2f, time for %d it = %.2f' %
                                (batch_idx,
                                 len(TrainImgLoader),
                                 local_loss / _config['print_every'],
                                 local_epe / _config['print_every'],
                                 (time.time() - start_time) / lidar_input.shape[0],
                                 _config['print_every'],
                                 (time.time() - time_for_N_it)))
                    time_for_N_it = time.time()
                    if _config['wandb']:
                        wandb.log({'Loss': local_loss / _config['print_every']}, step=total_iter)
                        wandb.log({'EPE': local_epe / _config['print_every']}, step=total_iter)
                    # Log epoch-level training metrics.
                    local_loss = 0.
                    local_epe = 0.
                total_train_loss += loss.item()
                total_train_epe += epe.item()
                total_iter += len(sample['rgb']) * world_size
            del loss, epe
            del target_mask1
            del target_flow1
            del rgb_input, lidar_input

        train_num_batches = batch_idx
        if rank == 0:
            logger.info("------------------------------------")
            logger.info('epoch %d total training loss = %.3f, total training epe = %.3f' %
                        (epoch, total_train_loss / batch_idx, total_train_epe / batch_idx))
            logger.info('Total epoch time = %.2f' % (time.time() - epoch_start_time))
            logger.info("------------------------------------")
            if _config['wandb']:
                wandb.log({'Total training loss': total_train_loss / batch_idx, 'epoch': epoch}, commit=False)
                wandb.log({'Total training EPE': total_train_epe / batch_idx}, commit=False)
            if TB_WRITER is not None:
                TB_WRITER.add_scalar('train/Loss', total_train_loss / batch_idx, epoch)
                TB_WRITER.add_scalar('train/EPE', total_train_epe / batch_idx, epoch)

        # Validate.
        do_validate = (_config['val_every'] > 0 and
                       (epoch == starting_epoch or (epoch + 1) % _config['val_every'] == 0
                        or epoch == _config['epochs'] - 1))
        total_test_loss = None
        total_test_epe = None
        val_epe = None
        if do_validate:
            total_test_loss = 0.
            total_test_epe = 0.
            total_test_f1 = 0.
            total_test_ece_u = 0.
            total_test_ece_v = 0.

            local_loss = 0.0
            val_iter = enumerate(TestImgLoader)
            if rank == 0:
                val_iter = tqdm(val_iter, total=len(TestImgLoader),
                                desc=f"epoch {epoch} val", leave=False, dynamic_ncols=True)
            for batch_idx, sample in val_iter:
                start_time = time.time()
                lidar_input = []
                rgb_input = []

                target_flow1 = []
                target_mask1 = []

                sample['tr_error'] = sample['tr_error'].to(device)
                sample['rot_error'] = sample['rot_error'].to(device)

                for idx in range(len(sample['rgb'])):
                    # Project with the current pose.
                    with torch.no_grad():
                        depth_img_no_occlusion, flow_img, flow_mask, rgb = prepare_input(
                            _config, device, idx, eval_img_shape, mean, sample,
                            std, spatial_mode="eval")

                    flow_img = flow_img.contiguous()
                    flow_mask = flow_mask.contiguous()
                    rgb_input.append(rgb)
                    lidar_input.append(depth_img_no_occlusion)

                    target_flow1.append(flow_img.permute(2, 0, 1).clone())
                    target_mask1.append(flow_mask.repeat(2, 1, 1).float().clone())

                lidar_input = torch.stack(lidar_input)
                rgb_input = torch.stack(rgb_input)
                target_flow1 = torch.stack(target_flow1)
                target_mask1 = torch.stack(target_mask1)

                save_images = (batch_idx == 0) and rank == 0 and (_config['wandb'] or _config['tensorboard'])

                loss, epe, ece_u, ece_v, ece_dict, f1 = test(model, rgb_input, lidar_input,
                                                             [target_flow1],
                                                             [target_mask1],
                                                             save_images, _config)
                dist.reduce(loss, 0)
                dist.reduce(epe, 0)
                if f1 is not None:
                    dist.reduce(f1, 0)
                if _config['uncertainty']:
                    dist.reduce(ece_u, 0)
                    dist.reduce(ece_v, 0)
                if rank == 0:
                    loss = loss / world_size
                    epe = epe / world_size
                    if f1 is not None:
                        f1 = f1 / world_size
                    if _config['uncertainty']:
                        ece_u = ece_u / world_size
                        ece_v = ece_v / world_size
                    local_loss += loss.item()

                    if batch_idx % 50 == 0 and batch_idx != 0:
                        logger.info('Iter %d test loss = %.3f , time = %.2f' %
                                    (batch_idx, local_loss / 50,
                                     (time.time() - start_time) / lidar_input.shape[0]))
                        local_loss = 0.0
                    total_test_loss += loss.item()
                    total_test_epe += epe.item()
                    if f1 is not None:
                        total_test_f1 += f1.item()
                    if _config['uncertainty']:
                        total_test_ece_u += ece_u.item()
                        total_test_ece_v += ece_v.item()
                del loss, epe
                del target_mask1
                del target_flow1
                del rgb_input, flow_img, flow_mask, rgb, lidar_input
                del depth_img_no_occlusion

            if rank == 0:
                val_epe = total_test_epe / batch_idx
                logger.info("------------------------------------")
                logger.info('total test loss = %.3f' % (total_test_loss / batch_idx))
                logger.info('total test epe = %.3f' % val_epe)
                logger.info("------------------------------------")

                if _config['wandb']:
                    wandb.log({'Val Loss': total_test_loss / batch_idx,
                               'Val EPE': val_epe}, commit=False)
                    if f1 is not None:
                        wandb.log({'Val F1': total_test_f1 / batch_idx}, commit=False)
                    if _config['uncertainty']:
                        wandb.log({'ECE u': total_test_ece_u / batch_idx,
                                   'ECE v': total_test_ece_v / batch_idx}, commit=False)
                if TB_WRITER is not None:
                    TB_WRITER.add_scalar('val/Loss', total_test_loss / batch_idx, epoch)
                    TB_WRITER.add_scalar('val/EPE', val_epe, epoch)
                    if f1 is not None:
                        TB_WRITER.add_scalar('val/F1', total_test_f1 / batch_idx, epoch)
                    if _config['uncertainty']:
                        TB_WRITER.add_scalar('val/ECE_u', total_test_ece_u / batch_idx, epoch)
                        TB_WRITER.add_scalar('val/ECE_v', total_test_ece_v / batch_idx, epoch)
        elif rank == 0:
            logger.info(f"Skipping validation at epoch {epoch}; val_every={_config['val_every']}")

        # Save training state.
        if rank == 0:
            if _config['wandb']:
                wandb.save(_config['log_file_resolved'])

            best_val_epe_after_epoch = BEST_VAL_EPE
            if val_epe is not None:
                best_val_epe_after_epoch = min(
                    best_val_epe_after_epoch, float(val_epe))
            payload = {
                'config': _config,
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'train_loss': total_train_loss / train_num_batches,
                'test_loss': (total_test_loss / batch_idx) if total_test_loss is not None else None,
                'train_epe': total_train_epe / train_num_batches,
                'test_epe': (total_test_epe / batch_idx) if total_test_epe is not None else None,
                'val_epe': val_epe,
                'best_val_epe': best_val_epe_after_epoch,
            }

            _atomic_save(payload, f'{_config["savemodel"]}/last_iter_checkpoint.tar')

            rolling_path = f'{_config["savemodel"]}/epoch_{epoch:03d}.tar'
            _atomic_save(payload, rolling_path)
            _rolling_re = re.compile(r'^epoch_(\d+)\.tar$')
            existing = []
            for fn in glob.glob(f'{_config["savemodel"]}/epoch_*.tar'):
                m = _rolling_re.match(os.path.basename(fn))
                if m:
                    existing.append((int(m.group(1)), fn))
            existing.sort(key=lambda x: x[0], reverse=True)
            for _, old in existing[_config['ckpt_keep_last']:]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        if val_epe is not None and val_epe < BEST_VAL_EPE and rank == 0:
            BEST_VAL_EPE = val_epe
            savefilename = f'{_config["savemodel"]}/checkpoint_{epoch}_{val_epe:.3f}.tar'
            _atomic_save(payload, savefilename)
            logger.info(f'Model saved as {savefilename}')
            if _config['wandb']:
                wandb.run.summary["Best Val EPE"] = BEST_VAL_EPE
                _atomic_save(payload, './best_model_so_far.tar')
                wandb.save('./best_model_so_far.tar')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

        # Release EMD dataset and loader references.
        del sample, dataset_train, dataset_val, TrainImgLoader

    if rank == 0:
        logger.info('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
        if TB_WRITER is not None:
            TB_WRITER.close()


def str2bool(v):
    """完成 CMRNext 训练、验证、输入构造和 checkpoint 管理 中的 `str2bool` 操作。

    参数:
        v: 待解析为布尔值的字符串、整数或布尔对象。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _mount_tensorboard_logdir(tb_logdir, mount_root, run_name):
    """把一次训练的 TensorBoard events 目录持久链接到服务监听目录。

    参数:
        tb_logdir: 本次训练实际写入 events 的目录，路径无单位；目录中的 step
            和 scalar 均使用训练 epoch 或 batch 语义。
        mount_root: TensorBoard 服务通过 ``--logdir`` 监听的根目录，路径无
            单位；函数会在其中创建以 ``run_name`` 命名的目录软链接。
        run_name: 本次训练的唯一任务名，同时作为 TensorBoard 页面中的实验
            目录名，不得包含路径分隔符。

    返回值:
        返回创建或复用的软链接绝对路径，类型为 ``pathlib.Path``。

    异常和副作用:
        创建 events 目录和服务监听根目录；创建一个目录软链接。若挂载位置
        已被普通文件、普通目录或指向其他训练目录的软链接占用，则抛出
        ``FileExistsError``，且不会覆盖已有内容。

    坐标系和外参约定:
        本函数只管理训练日志路径，不读取点云、相机参数或外参，也不改变
        EMD 的 ``Ti @ inverse(T0)`` 左乘残差约定。
    """
    # Keep events with the run; expose only a stable service link.
    resolved_tb_logdir = Path(tb_logdir).expanduser().resolve()
    resolved_mount_root = Path(mount_root).expanduser().resolve()
    resolved_tb_logdir.mkdir(parents=True, exist_ok=True)
    resolved_mount_root.mkdir(parents=True, exist_ok=True)

    # Reuse the run link on resume without replacing another target.
    mount_path = resolved_mount_root / run_name
    if mount_path.is_symlink():
        mounted_target = Path(os.readlink(mount_path))
        if not mounted_target.is_absolute():
            mounted_target = mount_path.parent / mounted_target
        if mounted_target.resolve() == resolved_tb_logdir:
            return mount_path
        raise FileExistsError(
            f'TensorBoard 挂载名已指向其他目录: {mount_path} -> '
            f'{mounted_target.resolve()}')
    if mount_path.exists():
        raise FileExistsError(f'TensorBoard 挂载位置已被占用: {mount_path}')

    mount_path.symlink_to(resolved_tb_logdir, target_is_directory=True)
    return mount_path


def _write_training_dry_run(_config):
    """输出训练命令的解析结果，不创建 checkpoint 或启动 GPU。

    参数:
        _config: argparse 解析后的完整 EMD 配置；场景来自 YAML 或命令行整组
            覆盖，seed 为所有训练随机源的基础种子。

    返回值:
        无显式返回值；JSON 报告写到标准输出，并在指定时写入审计日志。

    异常和副作用:
        配置文件无法读取或审计日志已存在且未指定 overwrite 时抛出异常。
        除可选审计日志外不创建训练目录、不加载数据样本、不启动 CUDA。

    坐标系和外参约定:
        报告只描述输入。存储残差模式使用 Ti @ inverse(T0)；固定 T0 模式与
        逐帧 Ti 模式在训练读取时在线采样左乘残差。
    """
    emd_mode = None
    dynamic = load_emd_dynamic_nominal_config(_config['emd_config'])
    fixed = load_fixed_t0_online_perturbation_config(_config['emd_config'])
    per_ti = load_per_ti_online_perturbation_config(_config['emd_config'])
    if sum((dynamic.enabled, fixed.enabled, per_ti.enabled)) > 1:
        raise ValueError('动态 T0、固定 T0 与逐帧 Ti 在线扰动模式互斥')
    if dynamic.enabled:
        emd_mode = 'stored_t0_with_online_jitter'
    elif fixed.enabled:
        emd_mode = 'fixed_t0_online_perturbation'
    elif per_ti.enabled:
        emd_mode = 'per_ti_online_perturbation'
    else:
        emd_mode = 'stored_t0_residual'
    # Include per-Ti geometry and sampling limits in dry-run output.
    per_ti_online_audit = None
    if per_ti.enabled:
        per_ti_online_audit = {
            'reference_extrinsic': 'per_frame_ti',
            'stored_t0_used': False,
            'sampler': 'cmrnext_legacy_per_axis_box_v1',
            'max_translation_component_m': (
                per_ti.max_translation_component_m),
            'max_rotation_euler_component_deg': (
                per_ti.max_rotation_euler_component_deg),
        }
    report = {
        'dry_run': True,
        'seed': int(_config['seed']),
        'run_output_dir': _config['run_output_dir'],
        'emd_mode': emd_mode,
        'per_ti_online_perturbation': per_ti_online_audit,
        'emd_level': _config.get('emd_level'),
        'train_scene_paths': _config.get('emd_train_scene_paths'),
        'val_scene_paths': _config.get('emd_val_scene_paths'),
        'test_scene_paths': _config.get('emd_test_scene_paths'),
        'scene_path_sources': _config.get('emd_scene_path_sources'),
        'emd_config_audit': _config.get('emd_config_audit'),
        'epochs': int(_config['epochs']),
        'batch_size': int(_config['batch_size']),
        'training_img_shape': list(_config['img_shape']),
        'emd_source_image_shape': _config.get('emd_source_image_shape'),
        'emd_resize_shape': _config.get('emd_resize_shape'),
        'emd_image_scale_xy': _config.get('emd_image_scale_xy'),
        'emd_evaluation_crop': _config.get('emd_evaluation_crop'),
        'emd_crop_top_range': _config.get('emd_crop_top_range'),
        'emd_crop_left_range': _config.get('emd_crop_left_range'),
        'flow_epe_units': 'resized_pixels',
        'gpu': int(_config['gpu']),
        'gpu_count': int(_config['gpu_count']),
        'tensorboard': bool(_config['tensorboard']),
        'tensorboard_logdir': (
            _config['tb_logdir_resolved'] if _config['tensorboard'] else None),
        'tensorboard_mount_path': (
            _config['tb_mount_path'] if _config['tensorboard'] else None),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if _config.get('log_file'):
        log_path = Path(_config['log_file_resolved'])
        if log_path.exists() and not _config['overwrite']:
            raise FileExistsError(
                f'训练审计日志已存在，使用 --overwrite 才能替换: {log_path}')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + '\n', encoding='utf-8')


def _resolve_emd_training_scene_paths(_config, emd_config) -> None:
    """合并 YAML 默认场景和训练命令行的整组覆盖值。

    参数:
        _config: argparse 生成的可变训练配置字典。
        emd_config: 已解析的 EMD v2 配置，场景路径均为绝对路径。

    返回值:
        无显式返回值；最终 train、validation、test 场景列表及其来源会原地
        写回 _config，供 dry-run、日志、DataLoader 和 checkpoint 共用。

    异常和副作用:
        本函数不检查场景目录是否存在，也不读取 split、metadata 或 manifest；
        路径合法性由实际数据加载阶段检查。

    坐标系和外参约定:
        场景选择不改变 T0、Ti 或 delta_Ti = Ti @ inverse(T0)。
    """
    role_fields = (
        ("train", "emd_train_scene_paths"),
        ("validation", "emd_val_scene_paths"),
        ("test", "emd_test_scene_paths"),
    )
    path_sources = {}
    for role, field in role_fields:
        command_paths = _config.get(field)
        if command_paths is None:
            resolved_paths = [
                str(path) for path in emd_config.scene_paths(role)]
            path_sources[role] = "yaml"
        else:
            resolved_paths = [
                str(Path(path).expanduser().resolve())
                for path in command_paths]
            path_sources[role] = "command_line"
        _config[field] = resolved_paths
    _config["emd_scene_path_sources"] = path_sources
    _config["emd_config_audit"] = emd_config.audit_record()


def real_main():
    """解析唯一训练入口参数并启动 dry-run 或分布式训练。

    参数:
        无；参数来自命令行。平移单位为米，旋转单位为度，seed 为整数。

    返回值:
        无显式返回值；dry-run 输出 JSON，正式模式启动一个或多个 GPU 进程。

    异常和副作用:
        参数缺失时由 argparse 以退出码 2 结束；已有输出未显式允许覆盖时拒绝
        启动；正式模式会设置分布式环境变量并创建训练产物。

    坐标系和外参约定:
        EMD train/val/test 场景路径完全由用户指定，不读取 split 标记决定归属。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--savemodel', type=str, default='checkpoints')
    parser.add_argument('--run_name', type=str, default=None,
                        help='训练任务目录名；默认使用当前 UTC 时间生成唯一名称')
    parser.add_argument('--log_file', type=str, default=None,
                        help='审计日志路径；默认写到 <savemodel>/<run_name>/training.log')
    parser.add_argument('--dry-run', action='store_true',
                        help='只报告解析后的训练配置，不启动 GPU 或写 checkpoint')
    parser.add_argument('--overwrite', action='store_true',
                        help='显式允许复用已有训练输出目录或替换审计日志')
    parser.add_argument('--use_reflectance', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--BASE_LEARNING_RATE', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='num_worker 大于 0 时的 DataLoader 预取批次数；降低该值可减少内存和共享内存压力')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--max_depth', type=float, default=160.)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_id', type=str, default=None)
    parser.add_argument('--no_scheduler', default=False, action='store_true')
    parser.add_argument('--scheduler', type=str, default="cycle_one")
    parser.add_argument('--upsample_method', type=str, default="transposed")
    parser.add_argument('--img_shape', type=int, nargs=2, default=[320, 960])
    parser.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--tensorboard', type=str2bool, nargs='?', const=True, default=True,
                        help='启用 TensorBoard 日志(默认开)')
    parser.add_argument('--tb_logdir', type=str, default=None,
                        help='TensorBoard 日志根目录，默认写到 <run_output_dir>/tb')
    parser.add_argument(
        '--tb_mount_root', type=str,
        default=str(Path(__file__).resolve().parent / 'tf-logs'),
        help='TensorBoard 服务监听目录；默认使用项目根目录下的 ./tf-logs')
    parser.add_argument('--not_normalize_images', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--master_port', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--gpu_count', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--val_every', type=int, default=1,
                        help='每隔 N 个 epoch 验证一次；恢复后的首个 epoch 和最终 epoch 也会验证，设为 1 保持原行为')
    parser.add_argument('--uncertainty', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--config', dest='emd_config', metavar='PATH',
                        type=str, required=True,
                        help='场景、传感器、图像预处理和扰动方式 YAML。')
    parser.add_argument(
        '--train-scene-paths', dest='emd_train_scene_paths', metavar='SCENE', nargs='+',
        default=None, help='直接用于训练的场景目录列表。')
    parser.add_argument(
        '--val-scene-paths', dest='emd_val_scene_paths', metavar='SCENE', nargs='+',
        default=None, help='直接用于训练期验证的场景目录列表。')
    parser.add_argument(
        '--test-scene-paths', dest='emd_test_scene_paths', metavar='SCENE', nargs='+',
        default=None, help='记录到训练配置中的后续测试场景目录列表。')
    parser.add_argument('--level', dest='emd_level', metavar='NAME', required=True,
                        help='从 --config 选择当前训练阶段的 level。')
    parser.add_argument('--fourier_levels', type=int, default=12)
    parser.add_argument('--weight_nll', type=float, default=-1.0)
    parser.add_argument('--der_type', type=str, default="NLL",
                        choices=["NLL"])  # Uncertainty model.
    parser.add_argument('--unc_freeze', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--finetune', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--find_unused_parameter', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--context_encoder', type=str, default="lidar", choices=["lidar"])
    parser.add_argument('--ckpt_keep_last', type=int, default=5,
                        help='滚动保留最近 N 个 epoch_*.tar，自动删除更早的文件')
    parser.add_argument('--rebuild_scheduler', type=str2bool, nargs='?', const=True, default=False,
                        help='与 --resume 同用时不加载调度器状态，使 OneCycleLR 从当前 epoch 使用新的总轮数')

    args = parser.parse_args()
    _config = vars(args)
    if _config['use_reflectance']:
        parser.error('当前场景训练要求 use_reflectance=false，因为强度为占位值')
    emd_preflight_config = load_emd_training_config(_config['emd_config'])
    _resolve_emd_training_scene_paths(_config, emd_preflight_config)
    if not _config['emd_train_scene_paths']:
        parser.error('训练场景不能为空；请填写 YAML scenes.train 或使用命令行覆盖')
    if (_config['val_every'] > 0
            and not _config['emd_val_scene_paths']):
        parser.error(
            '验证场景为空时必须显式使用 --val_every 0，或提供验证场景')
    if _config['uncertainty']:
        raise NotImplementedError("不确定性训练尚未经过验证")
    if _config['no_scheduler']:
        _config['scheduler'] = False
    _config['normalize_images'] = not _config['not_normalize_images']
    # Dry-run and training share parsing and size validation.
    _configure_emd_spatial_geometry(_config, emd_preflight_config.contract)

    run_name = _config['run_name'] or time.strftime(
        'train_%Y%m%dT%H%M%SZ', time.gmtime())
    if Path(run_name).name != run_name or run_name in {'.', '..'}:
        parser.error('--run_name 必须是单个目录名，不能包含路径分隔符')
    run_output_dir = Path(_config['savemodel']).expanduser().resolve() / run_name
    _config['run_name'] = run_name
    _config['run_output_dir'] = str(run_output_dir)
    # Archive events with the run and publish the service path.
    tb_logdir = (
        Path(_config['tb_logdir']).expanduser().resolve()
        if _config['tb_logdir'] else run_output_dir / 'tb')
    tb_mount_root = Path(_config['tb_mount_root']).expanduser().resolve()
    _config['tb_logdir_resolved'] = str(tb_logdir)
    _config['tb_mount_path'] = str(tb_mount_root / run_name)
    _config['log_file_resolved'] = str(
        Path(_config['log_file']).expanduser().resolve()
        if _config['log_file'] else run_output_dir / 'training.log')
    if (not _config['dry_run'] and run_output_dir.exists()
            and not (_config['resume'] or _config['overwrite'])):
        parser.error(
            f'训练输出目录已存在: {run_output_dir}；请更换 --run_name，或显式使用 '
            '--resume/--overwrite')

    if _config['dry_run']:
        _write_training_dry_run(_config)
        return

    if _config['gpu_count'] == -1:
        _config['gpu_count'] = torch.cuda.device_count()
    world_size = _config['gpu_count']
    os.environ['MASTER_ADDR'] = 'localhost'
    _config['MASTER_PORT'] = f'{np.random.randint(8000, 9999)}'
    if args.master_port is not None:
        _config['MASTER_PORT'] = args.master_port
    os.environ['MASTER_PORT'] = _config['MASTER_PORT']
    if _config['gpu'] == -1:
        mp.spawn(main, nprocs=world_size, args=(_config, _config['seed'], world_size,))
    else:
        main(_config['gpu'], _config, _config['seed'], world_size)


if __name__ == '__main__':
    real_main()
