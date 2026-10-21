"""CMRNext 数据评估、光流到 PnP、级联外参细化和报告生成。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from matplotlib import cm, rcParams
from rich.console import Console
from rich.table import Table
from rich import box
from PIL import Image

import cv2
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data
import visibility
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from datasets.DatasetExtrinsicCalib import (
    DatasetGeneralExtrinsicCalib,
    DatasetPandasetExtrinsicCalib,
    compute_kitti_sampled_initial_state,
)
from datasets.emd_extrinsic_calib import (
    DatasetEMDExtrinsicCalib,
    load_emd_training_config,
)
from datasets.emd_fixed_t0_online import (
    DatasetEMDFixedT0OnlineExtrinsicCalib,
    load_fixed_t0_online_perturbation_config,
)
from camera_model import CameraModel

from models.get_model import get_model
from quaternion_distances import quaternion_loss

import matplotlib
matplotlib.use('Agg' if not os.environ.get('DISPLAY') else 'TkAgg')

import matplotlib.pyplot as plt
from utils import (downsample_depth, merge_inputs, get_flow_zforward, quat2mat, tvector2mat,
                   quaternion_from_matrix, EndPointError, rotate_forward, quaternion_median,
                   average_quaternions, rotate_back, quaternion_mode, str2bool, overlay_imgs,
                   mat2xyzrpy)


def _rot_angle_deg(T):
    """从四乘四刚体变换的旋转部分计算 SO(3) 测地角度。

    参数:
        T: 三维平移向量或齐次平移矩阵，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    if not torch.is_tensor(T):
        T = torch.as_tensor(T, dtype=torch.float)
    R = T[:3, :3]
    tr = (R[0, 0] + R[1, 1] + R[2, 2]).item()
    cos_t = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return math.degrees(math.acos(cos_t))


def _fmt_extrinsic(T):
    """把四乘四外参矩阵格式化为适合日志审阅的多行文本。

    参数:
        T: 三维平移向量或齐次平移矩阵，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    if not torch.is_tensor(T):
        T = torch.as_tensor(T, dtype=torch.float)
    T_cpu = T.detach().cpu().float()
    xyzrpy = mat2xyzrpy(T_cpu)
    x, y, z, r, p, yaw = xyzrpy.tolist()
    return (f"t=[{x:+.3f}, {y:+.3f}, {z:+.3f}]m  "
            f"rpy=[{math.degrees(r):+6.2f}, {math.degrees(p):+6.2f}, {math.degrees(yaw):+6.2f}]°")


def _nearest_percentile(values, percentile):
    """按最近秩方法计算数值序列的指定百分位，避免插值改变实际样本边界。

    参数:
        values: 待统计的一维有限数值序列。
        percentile: 目标百分位，范围为 0 到 100。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(round(percentile / 100.0 * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


def _distribution_summary(values):
    """汇总数值序列的样本数、最小值、最大值、均值和声明百分位。

    参数:
        values: 待统计的一维有限数值序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'p80': None,
            'p90': None,
            'p95': None,
            'p99': None,
            'max': None,
        }
    return {
        'count': len(finite),
        'mean': float(sum(finite) / len(finite)),
        'median': _nearest_percentile(finite, 50),
        'p80': _nearest_percentile(finite, 80),
        'p90': _nearest_percentile(finite, 90),
        'p95': _nearest_percentile(finite, 95),
        'p99': _nearest_percentile(finite, 99),
        'max': max(finite),
    }



def _build_flow_epe_report(epe_by_checkpoint, cascade_stages):
    """把各 checkpoint 的光流 EPE 按真实级联阶段编号整理为报告。

    参数:
        epe_by_checkpoint: 按 checkpoint 顺序保存的逐帧光流 EPE 序列。
        cascade_stages: 与 checkpoint 一一对应的真实级联阶段编号序列。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    if len(epe_by_checkpoint) < len(cascade_stages):
        raise ValueError("一个或多个阶段缺少光流 EPE")
    return {
        'units': 'pixels',
        'iterations': {
            f'iter{index + 1}': {
                'stage': int(stage),
                **_distribution_summary(epe_by_checkpoint[index]),
            }
            for index, stage in enumerate(cascade_stages)
        },
    }

def _validate_fixed_t0_checkpoint_limits(checkpoint_config, online_config):
    """核对固定 T0 评估扰动范围与 checkpoint 训练配置是否一致。

    参数:
        checkpoint_config: checkpoint 内保存的训练配置映射。
        online_config: 固定 T0 在线逐轴扰动配置对象。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    expected_limits = (
        online_config.max_translation_component_m,
        online_config.max_rotation_euler_component_deg,
    )
    actual_limits = (
        checkpoint_config.get(
            'emd_fixed_t0_online_max_translation_component_m'),
        checkpoint_config.get(
            'emd_fixed_t0_online_max_rotation_euler_component_deg'),
    )
    if (not all(value is not None for value in actual_limits)
            or not all(math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
                for actual, expected in zip(actual_limits, expected_limits))):
        raise ValueError(
            'fixed-T0 checkpoint/evaluation perturbation limits disagree: '
            f'{actual_limits} != {expected_limits}')


def _residual_components(transform):
    """把四乘四左乘残差分解为 xyz 平移和 xyz 欧拉角分量。

    参数:
        transform: 四乘四齐次刚体变换矩阵，平移单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    transform = torch.as_tensor(transform, dtype=torch.float).cpu()
    xyzrpy = mat2xyzrpy(transform)
    translation = [float(value) for value in xyzrpy[:3]]
    euler_deg = [math.degrees(float(value)) for value in xyzrpy[3:]]
    return {
        'translation_xyz_m': translation,
        'rotation_euler_xyz_deg': euler_deg,
        'translation_max_abs_component_m': max(abs(value) for value in translation),
        'rotation_max_abs_euler_component_deg': max(
            abs(value) for value in euler_deg),
        'convention': 'target_from_estimate_left_residual_xyz_euler',
    }


def _estimate_from_left_residual(true_extrinsic, left_residual):
    """根据 delta=Ti@inverse(T_est) 从真实外参和剩余残差恢复绝对外参估计。

    参数:
        true_extrinsic: 逐帧真实 LiDAR 到相机外参 Ti，形状为 4×4。
        left_residual: 目标相对当前估计的完整左乘残差，形状为 4×4。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    true_extrinsic = torch.as_tensor(
        true_extrinsic, dtype=torch.float).cpu()
    left_residual = torch.as_tensor(
        left_residual, dtype=torch.float).cpu()
    return torch.mm(torch.linalg.inv(left_residual), true_extrinsic)


def _normalized_joint_residual(record, iteration_key):
    """用当前样本有效平移和旋转尺度归一化残差并计算联合难度。

    参数:
        record: 一帧评估记录字典，包含场景、尺度和逐阶段误差。
        iteration_key: 待读取的阶段字段名，例如 iter1、iter2 或 iter3。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    max_t = float(record['effective_max_t_m'])
    max_r = float(record['effective_max_r_deg'])
    if max_t <= 0.0 or max_r <= 0.0:
        raise ValueError('EMD 归一化范围必须为正数')
    iteration = record[iteration_key]
    translation_normalized = float(iteration['t_cm']) / 100.0 / max_t
    rotation_normalized = float(iteration['r_deg']) / max_r
    return math.hypot(translation_normalized, rotation_normalized)


def _select_emd_worst_per_scene(frame_records, iteration_key, count):
    """在每个场景内部独立排序并选择联合归一化残差最大的样本。

    参数:
        frame_records: 完整评估期间累计的逐帧记录字典列表。
        iteration_key: 待读取的阶段字段名，例如 iter1、iter2 或 iter3。
        count: 需要选择或生成的元素数量。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    if count <= 0:
        raise ValueError('每场景最差样本数必须为正数')
    by_scene = {}
    for record in frame_records:
        if iteration_key not in record:
            continue
        scene = str(record.get('sequence', ''))
        by_scene.setdefault(scene, []).append(record)

    selected = []
    for scene in sorted(by_scene):
        ranked = sorted(
            by_scene[scene],
            key=lambda item: (
                -_normalized_joint_residual(item, iteration_key),
                int(item['batch_idx']),
            ),
        )
        for rank, record in enumerate(ranked[:count], start=1):
            selected.append({
                'scene': scene,
                'rank': rank,
                'score': _normalized_joint_residual(record, iteration_key),
                'record': record,
            })
    return selected


def _residual_annotation(iteration_record):
    """把单阶段残差记录格式化为重投影图使用的中文标注文本。

    参数:
        iteration_record: 单帧单阶段的残差与绝对外参记录字典。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    components = iteration_record['residual_components']
    tx, ty, tz = components['translation_xyz_m']
    rx, ry, rz = components['rotation_euler_xyz_deg']
    return (
        f"delta t xyz=[{tx:+.4f}, {ty:+.4f}, {tz:+.4f}] m | "
        f"delta r xyz=[{rx:+.3f}, {ry:+.3f}, {rz:+.3f}] deg\n"
        f"total: t={float(iteration_record['t_cm']) / 100.0:.4f} m | "
        f"r={float(iteration_record['r_deg']):.3f} deg"
    )


def _draw_reprojection_panels(plt_module, projections, rgb01, *, overlay_text):
    """按固定竖向布局绘制初始、估计和真实点云重投影面板。

    参数:
        plt_module: 已导入并配置后端的 matplotlib.pyplot 模块。
        projections: 按面板顺序排列的点云投影图像或像素集合。
        rgb01: 值域为 0 到 1 的 RGB 背景图像，形状为 H×W×3。
        overlay_text: 可选面板文字标注序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    panel_count = len(projections)
    fig, axes = plt_module.subplots(
        panel_count, 1, figsize=(12, 4.5 * panel_count), squeeze=False)
    axes = axes[:, 0]
    for axis, (label, depth_img) in zip(axes, projections):
        axis.imshow(per_frame_jet_overlay(rgb01, depth_img))
        if overlay_text:
            axis.text(
                0.01, 0.98, label,
                transform=axis.transAxes,
                ha='left', va='top', fontsize=9, color='black',
                bbox={
                    'facecolor': 'white', 'edgecolor': 'black',
                    'alpha': 0.82, 'pad': 4,
                },
            )
        else:
            axis.set_title(label, fontsize=9, loc='left')
        axis.axis('off')
    fig.subplots_adjust(hspace=0.08)
    return fig


def _residual_component_summary(records, iteration_key):
    """汇总指定阶段残差的逐轴平移、欧拉角和最大绝对分量分布。

    参数:
        records: 参与当前统计的逐帧评估记录序列。
        iteration_key: 待读取的阶段字段名，例如 iter1、iter2 或 iter3。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    components = [
        record[iteration_key]['residual_components']
        for record in records
        if record.get(iteration_key, {}).get('residual_components') is not None
    ]
    axes = ('x', 'y', 'z')
    translation_by_axis = {}
    rotation_by_axis = {}
    for index, axis in enumerate(axes):
        translation_by_axis[axis] = _distribution_summary([
            abs(item['translation_xyz_m'][index]) for item in components])
        rotation_by_axis[axis] = _distribution_summary([
            abs(item['rotation_euler_xyz_deg'][index]) for item in components])
    return {
        'count': len(components),
        'convention': 'absolute_components_of_target_from_estimate_left_residual',
        'translation_m': {
            'absolute_by_axis': translation_by_axis,
            'max_abs_component_per_frame': _distribution_summary([
                item['translation_max_abs_component_m'] for item in components]),
            'signed_mean_xyz': [
                (sum(item['translation_xyz_m'][index] for item in components)
                 / len(components)) if components else None
                for index in range(3)
            ],
        },
        'rotation_euler_deg': {
            'absolute_by_axis': rotation_by_axis,
            'max_abs_component_per_frame': _distribution_summary([
                item['rotation_max_abs_euler_component_deg']
                for item in components
            ]),
            'signed_mean_xyz': [
                (sum(item['rotation_euler_xyz_deg'][index] for item in components)
                 / len(components)) if components else None
                for index in range(3)
            ],
        },
    }


def _emd_iteration_summary(records, iteration_key, contract):
    """计算 EMD 单个级联阶段的完成数、成功率和外参误差统计。

    参数:
        records: 参与当前统计的逐帧评估记录序列。
        iteration_key: 待读取的阶段字段名，例如 iter1、iter2 或 iter3。
        contract: EMD 数据契约，包含传感器参数、裁剪范围和变换容差。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    translation_m = [record[iteration_key]['t_cm'] / 100.0 for record in records]
    rotation_deg = [record[iteration_key]['r_deg'] for record in records]
    translation_normalized = [
        record[iteration_key]['t_normalized'] for record in records]
    successes = [
        translation <= contract.success_translation_m
        and rotation <= contract.success_rotation_deg
        for translation, rotation in zip(translation_m, rotation_deg)
    ]
    return {
        'count': len(records),
        'translation_m': {
            'mean': _distribution_summary(translation_m)['mean'],
            'median': _nearest_percentile(translation_m, 50),
            'p80': _nearest_percentile(translation_m, 80),
            'p90': _nearest_percentile(translation_m, 90),
            'p95': _nearest_percentile(translation_m, 95),
            'p99': _nearest_percentile(translation_m, 99),
        },
        'translation_normalized': {
            'median': _nearest_percentile(translation_normalized, 50),
            'p90': _nearest_percentile(translation_normalized, 90),
            'p99': _nearest_percentile(translation_normalized, 99),
        },
        'rotation_deg': {
            'mean': _distribution_summary(rotation_deg)['mean'],
            'median': _nearest_percentile(rotation_deg, 50),
            'p80': _nearest_percentile(rotation_deg, 80),
            'p90': _nearest_percentile(rotation_deg, 90),
            'p95': _nearest_percentile(rotation_deg, 95),
            'p99': _nearest_percentile(rotation_deg, 99),
        },
        'success_rate': sum(successes) / len(successes) if successes else None,
    }


def _build_emd_report(
        frame_records, iteration_count, contract, cascade_stages):
    """组合 EMD 数据约定、阶段映射、逐阶段指标和残差分量报告。

    参数:
        frame_records: 完整评估期间累计的逐帧记录字典列表。
        iteration_count: 需要汇总的级联阶段数量。
        contract: EMD 数据契约，包含传感器参数、裁剪范围和变换容差。
        cascade_stages: 与 checkpoint 一一对应的真实级联阶段编号序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    complete = [
        record for record in frame_records
        if all(f'iter{index}' in record for index in range(iteration_count + 1))
    ]
    report = {
        'cascade_stages': cascade_stages,
        'thresholds': {
            'translation_m': contract.success_translation_m,
            'rotation_deg': contract.success_rotation_deg,
        },
        'iterations': {
            f'iter{index}': _emd_iteration_summary(
                [record for record in frame_records if f'iter{index}' in record],
                f'iter{index}', contract)
            for index in range(iteration_count + 1)
        },
        'residual_component_distributions': {
            f'iter{index}': _residual_component_summary(
                [record for record in frame_records if f'iter{index}' in record],
                f'iter{index}')
            for index in range(iteration_count + 1)
        },
        'final_by_scene': {},
        'final_by_initial_residual_magnitude': {},
    }
    final_key = f'iter{iteration_count}'
    for scene in sorted({record.get('sequence', '') for record in complete}):
        scene_records = [
            record for record in complete if record.get('sequence', '') == scene]
        report['final_by_scene'][scene] = _emd_iteration_summary(
            scene_records, final_key, contract)

    boundaries = contract.residual_magnitude_bins
    for lower, upper in zip(boundaries, boundaries[1:]):
        label = f'[{lower:.2f},{upper:.2f})'
        bin_records = [
            record for record in complete
            if lower <= record['initial_residual_magnitude'] < upper]
        if bin_records:
            report['final_by_initial_residual_magnitude'][label] = (
                _emd_iteration_summary(bin_records, final_key, contract))
    overflow_label = f'[{boundaries[-1]:.2f},inf)'
    overflow = [
        record for record in complete
        if record['initial_residual_magnitude'] >= boundaries[-1]]
    if overflow:
        report['final_by_initial_residual_magnitude'][overflow_label] = (
            _emd_iteration_summary(overflow, final_key, contract))
    return report


KITTI_SUCCESS_TRANSLATION_M = 0.10
KITTI_SUCCESS_ROTATION_DEG = 1.0


def _kitti_iteration_summary(records, iteration_key):
    """计算 KITTI 单个级联阶段的平移和旋转误差统计。

    参数:
        records: 参与当前统计的逐帧评估记录序列。
        iteration_key: 待读取的阶段字段名，例如 iter1、iter2 或 iter3。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    translation_m = [record[iteration_key]['t_cm'] / 100.0 for record in records]
    rotation_deg = [record[iteration_key]['r_deg'] for record in records]
    successes = [
        translation <= KITTI_SUCCESS_TRANSLATION_M
        and rotation <= KITTI_SUCCESS_ROTATION_DEG
        for translation, rotation in zip(translation_m, rotation_deg)
    ]
    return {
        'count': len(records),
        'translation_m': {
            'median': _nearest_percentile(translation_m, 50),
            'p90': _nearest_percentile(translation_m, 90),
            'p99': _nearest_percentile(translation_m, 99),
        },
        'rotation_deg': {
            'median': _nearest_percentile(rotation_deg, 50),
            'p90': _nearest_percentile(rotation_deg, 90),
            'p99': _nearest_percentile(rotation_deg, 99),
        },
        'success_count': sum(successes),
        'success_rate': sum(successes) / len(successes) if successes else None,
    }


def _build_kitti_stage_report(frame_records, iteration_count, cascade_stages):
    """组合 KITTI 级联阶段映射和逐阶段指标报告。

    参数:
        frame_records: 完整评估期间累计的逐帧记录字典列表。
        iteration_count: 需要汇总的级联阶段数量。
        cascade_stages: 与 checkpoint 一一对应的真实级联阶段编号序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    complete = [
        record for record in frame_records
        if all(f'iter{index}' in record for index in range(iteration_count + 1))
    ]
    iterations = {
        f'iter{index}': _kitti_iteration_summary(
            [record for record in frame_records if f'iter{index}' in record],
            f'iter{index}')
        for index in range(iteration_count + 1)
    }
    final = iterations[f'iter{iteration_count}']
    total = len(frame_records)
    initial_median = iterations['iter0']['translation_m']['median']
    final_median = final['translation_m']['median']
    return {
        'cascade_stages': cascade_stages,
        'thresholds': {
            'translation_m': KITTI_SUCCESS_TRANSLATION_M,
            'rotation_deg': KITTI_SUCCESS_ROTATION_DEG,
        },
        'n_frames_total': total,
        'n_frames_complete': len(complete),
        'n_frames_failed': total - len(complete),
        'completion_rate': len(complete) / total if total else None,
        'iterations': iterations,
        'final_success_rate_all_frames': (
            final['success_count'] / total if total else None),
        'median_translation_improvement_ratio': (
            final_median / initial_median
            if initial_median and final_median is not None else None),
    }


def per_frame_jet_overlay(rgb01, depth_img, lo_pct=2.0, hi_pct=98.0, alpha=0.5,
                          pooling=3):
    """按单帧深度百分位生成 jet 伪彩点云并叠加到 RGB 图像。

    参数:
        rgb01: 值域为 0 到 1 的 RGB 背景图像，形状为 H×W×3。
        depth_img: 用于颜色编码的单通道深度图，深度单位为米。
        lo_pct: 深度颜色归一化的低百分位，范围为 0 到 100。
        hi_pct: 深度颜色归一化的高百分位，范围为 0 到 100。
        alpha: 点云伪彩与 RGB 背景的混合权重，范围为 0 到 1。
        pooling: 可视化投影点向周围扩展的像素半径。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    from matplotlib import cm
    import torch.nn.functional as F

    rgb_np = rgb01.detach().cpu().permute(1, 2, 0).numpy()
    rgb_np = rgb_np.clip(0., 1.)

    # Use the final depth channel to match overlay_imgs.
    if depth_img.ndim == 3:
        d = depth_img[-1]
    else:
        d = depth_img
    d = d.detach().float().clone()

    # Expand points with the overlay_imgs near-point pooling rule.
    if pooling > 0:
        d4 = d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        nonzero_mask = (d4 > 0).float()
        # Exclude zero depth while taking the nearest positive depth.
        d_neg = -d4
        d_neg[d4 == 0] = -1e6  # Exclude zeros from max pooling.
        d_neg = F.max_pool2d(d_neg, pooling, 1, pooling // 2)
        d_pooled = -d_neg
        d_pooled[d_pooled == 1e6] = 0.
        d = d_pooled[0, 0]

    d_np = d.cpu().numpy()
    valid = d_np > 0
    if valid.sum() < 5:
        # Return RGB when too few points are valid.
        return rgb_np

    vals = d_np[valid]
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.zeros_like(d_np, dtype=np.float32)
    norm[valid] = ((d_np[valid] - lo) / (hi - lo)).clip(0., 1.)

    rgba = cm.jet(norm)  # (H,W,4)
    rgba[..., 3] = alpha
    rgba[~valid] = [0, 0, 0, 0]
    blended = rgba[..., :3] * rgba[..., 3:4] + rgb_np * (1. - rgba[..., 3:4])
    return blended.clip(0., 1.)

rcParams["figure.raise_window"] = False
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def _init_fn(worker_id, seed):
    """根据基础 seed、工作进程编号和训练轮次初始化各随机数生成器。

    参数:
        worker_id: DataLoader 工作进程编号。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def quaternion_distance(q, r):
    """计算两个四元数表示之间的最短旋转角距离。

    参数:
        q: 第一个四元数张量，分量顺序遵循当前模块实现。
        r: 第二个四元数张量，分量顺序遵循当前模块实现。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    dist = quaternion_loss(q.unsqueeze(0), r.unsqueeze(0), q.device)

    dist = 180. * dist.item() / math.pi
    return dist


def prepare_input(cam_params, pc_rotated, real_shape, reflectance, _config, change_frame=False):
    """把一帧 RGB 和外参扰动点云构造成网络深度输入、稀疏 GT 光流及有效掩码。

    参数:
        cam_params: 相机内参数向量 [fx, fy, cx, cy, skew]，单位为像素。
        pc_rotated: 当前剩余误差下的相机坐标齐次点云，形状为 4×N。
        real_shape: 裁剪或补齐前的实际图像尺寸，单位为像素。
        reflectance: 可选 LiDAR 反射强度，顺序与点云一致。
        _config: 运行配置字典，包含数据、网络、几何和输出选项。
        change_frame: 是否把输入点轴从 (x,y,z) 重排为历史数据集使用的 (y,z,x)。

    返回值:
        四元组 (lidar_input, flow_gt, flow_mask, rgb_input)，分别为深度输入、GT 光流、有效掩码和 RGB 输入。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    cam_model = CameraModel().set_intrinsics(cam_params)
    uv, depth, _, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance, change_frame)
    uv = uv.t().int().contiguous()
    depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
    depth_img += 1000.
    depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
    depth_img[depth_img == 1000.] = 0.

    depth_img_no_occlusion = torch.zeros_like(depth_img, device='cuda')
    depth_img_no_occlusion = visibility.visibility2(depth_img, cam_params[:4], depth_img_no_occlusion,
                                                    depth_img.shape[1], depth_img.shape[0],
                                                    _config['occlusion_threshold'], _config['occlusion_kernel'])

    uv = uv.long()

    # Keep only points that survive same-pixel projection conflicts.
    indexes = depth_img_no_occlusion[uv[:, 1], uv[:, 0]] == depth
    if _config['use_reflectance']:
        refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        refl_img[uv[indexes, 1], uv[indexes, 0]] = refl[0, indexes]

    depth_img_no_occlusion /= _config['max_depth']
    depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)

    if _config['use_reflectance']:
        depth_img_no_occlusion = torch.cat((depth_img_no_occlusion, refl_img.unsqueeze(0)))

    uv = uv[indexes]
    return depth_img_no_occlusion, uv, indexes, depth


def downsample_and_pad(_config, rgb, depth_img_no_occlusion, img_shape, real_shape, flow_img, flow_mask):
    """按数据集缩放约定调整 RGB、深度和光流，并补齐到网络输入尺寸。

    参数:
        _config: 运行配置字典，包含数据、网络、几何和输出选项。
        rgb: RGB 图像张量或数组；通道顺序和数值范围遵循调用方约定。
        depth_img_no_occlusion: 经过 z-buffer 去除遮挡的深度图，深度单位为米或按配置归一化。
        img_shape: 网络输入尺寸，按高度、宽度排列，单位为像素。
        real_shape: 裁剪或补齐前的实际图像尺寸，单位为像素。
        flow_img: 稀疏 GT 光流图，形状为 H×W×2，单位为像素。
        flow_mask: 标记存在有效点云光流的位置掩码，形状为 H×W。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    shape_pad = [0, 0, 0, 0]

    if _config['dataset'] in ['argoverse', 'pandaset'] or _config['downsample']:
        rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
        depth_img_no_occlusion = downsample_depth(depth_img_no_occlusion.permute(1, 2, 0).contiguous(), 2)
        depth_img_no_occlusion = depth_img_no_occlusion.permute(2, 0, 1)

        shape_pad[3] = (img_shape[0] - real_shape[0] // 2)  # // 2
        shape_pad[1] = (img_shape[1] - real_shape[1] // 2)  # // 2 + 1

        rgb = F.pad(rgb, shape_pad)
        depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)

        shape_pad[3] = (img_shape[0] * 2 - real_shape[0])  # // 2
        shape_pad[1] = (img_shape[1] * 2 - real_shape[1])  # // 2 + 1
        flow_img = F.pad(flow_img.permute(2, 0, 1), shape_pad).permute(1, 2, 0).contiguous()
        flow_mask = F.pad(flow_mask, shape_pad)

    else:
        shape_pad[3] = (img_shape[0] - real_shape[0])  # // 2
        shape_pad[1] = (img_shape[1] - real_shape[1])  # // 2 + 1

        rgb = F.pad(rgb, shape_pad)
        depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
        flow_img = F.pad(flow_img.permute(2, 0, 1), shape_pad).permute(1, 2, 0).contiguous()
        flow_mask = F.pad(flow_mask, shape_pad)

    # Encode depth with NeRF-style Fourier frequencies.
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

    return rgb, depth_img_no_occlusion, flow_img, flow_mask


def _resolve_evaluation_image_shape(checkpoint, dataset):
    """从 checkpoint 恢复固定评估尺寸，旧权重回退到训练尺寸。

    参数:
        checkpoint: 已加载的 checkpoint 字典，包含模型状态与训练配置。
        dataset: 数据集模式名称，用于选择尺寸、坐标和后处理约定。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    checkpoint_config = checkpoint.get('config', {})
    evaluation_crop = checkpoint_config.get('emd_evaluation_crop')
    if dataset == 'emd' and evaluation_crop is not None:
        if (not isinstance(evaluation_crop, dict)
                or set(evaluation_crop) != {"top", "left", "height", "width"}):
            raise ValueError(
                f'checkpoint 中的 EMD 评估裁剪无效：{evaluation_crop!r}')
        evaluation_shape = (
            int(evaluation_crop["height"]), int(evaluation_crop["width"]))
        if any(value <= 0 for value in evaluation_shape):
            raise ValueError(
                f'checkpoint 中的 EMD 评估尺寸无效：{evaluation_shape!r}')
        return evaluation_shape
    saved_shape = checkpoint_config.get('img_shape')
    if dataset == 'emd' and saved_shape is not None:
        if len(saved_shape) != 2 or any(int(value) <= 0 for value in saved_shape):
            raise ValueError(f'checkpoint 中的 EMD img_shape 无效：{saved_shape!r}')
        return (int(saved_shape[0]), int(saved_shape[1]))
    if dataset == 'emd':
        return (384, 1280)
    return None


def _checkpoint_training_domain_record(checkpoint, checkpoint_path, stage):
    """提取 checkpoint 中可用于交叉测试审计的训练域元数据。

    参数:
        checkpoint: 已加载的 checkpoint 字典；模型参数位于 state_dict，训练配置
            位于可选的 config 映射中。
        checkpoint_path: checkpoint 文件路径，仅用于写入评估报告，不修改文件。
        stage: 该 checkpoint 在当前级联中的真实阶段编号，取值为 1、2 或 3。

    返回值:
        训练域审计字典。旧版 checkpoint 缺少 EMD 字段时对应值保持为 None，
        不会被错误转换为 False。

    异常和副作用:
        checkpoint 的 config 不是字典时抛出 TypeError；本函数不加载模型、不写
        文件，也不改变 T0、Ti 或左乘残差。

    坐标系和外参约定:
        本函数只记录训练来源。评估几何仍由当前数据集决定，EMD 使用
        delta_Ti = Ti @ inverse(T0)，并满足 delta_Ti @ T0 = Ti。
    """
    checkpoint_config = checkpoint.get('config', {})
    if not isinstance(checkpoint_config, dict):
        raise TypeError('checkpoint config 必须是字典')
    return {
        'stage': int(stage),
        'checkpoint': str(checkpoint_path),
        'emd_only': checkpoint_config.get('emd_only'),
        'emd_level': checkpoint_config.get('emd_level'),
        'emd_dynamic_nominal': checkpoint_config.get('emd_dynamic_nominal'),
        'emd_fixed_t0_online': checkpoint_config.get('emd_fixed_t0_online'),
        'training_img_shape': checkpoint_config.get('img_shape'),
        'emd_source_image_shape': checkpoint_config.get(
            'emd_source_image_shape'),
        'emd_resize_shape': checkpoint_config.get('emd_resize_shape'),
        'emd_evaluation_crop': checkpoint_config.get(
            'emd_evaluation_crop'),
        'flow_epe_units': (
            'resized_pixels'
            if checkpoint_config.get('emd_resize_shape') is not None
            else 'pixels'),
        'fixed_t0_max_translation_component_m': checkpoint_config.get(
            'emd_fixed_t0_online_max_translation_component_m'),
        'fixed_t0_max_rotation_euler_component_deg': checkpoint_config.get(
            'emd_fixed_t0_online_max_rotation_euler_component_deg'),
    }


def _sample_emd_crop_offsets(config, source_shape, target_shape):
    """使用显式 seed 对应的随机状态采样与训练一致的 EMD 裁剪偏移。

    参数:
        config: 当前功能使用的配置对象或映射。
        source_shape: 裁剪前图像尺寸，按高度、宽度排列，单位为像素。
        target_shape: 裁剪后网络输入尺寸，按高度、宽度排列，单位为像素。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    source_height, source_width = int(source_shape[0]), int(source_shape[1])
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    if target_height > source_height or target_width > source_width:
        return 0, 0

    def sample(key, maximum):
        """完成 CMRNext 数据评估、光流到 PnP、级联外参细化和报告生成 中的 `sample` 操作。

        参数:
            key: 待读取的配置、指标或字典字段名。
            maximum: 随机裁剪或索引允许的最大整数值，包含端点。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

        坐标系和数据约定:
            EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
        """
        bounds = config.get(key)
        if bounds is None:
            return random.randint(0, maximum)
        lower, upper = (int(bounds[0]), int(bounds[1]))
        if lower < 0 or upper < lower or upper > maximum:
            raise ValueError(
                f'{key}={bounds!r} is outside the valid crop range [0, {maximum}]')
        return random.randint(lower, upper)

    return sample('emd_crop_top_range', source_height - target_height), sample(
        'emd_crop_left_range', source_width - target_width)


def _mask_evaluation_flow_endpoints(flow_img, flow_mask):
    """清除固定评估视野中目标终点越界的稀疏 GT 光流。

    参数:
        flow_img: 裁剪后的 H×W×2 GT 光流，单位为 resize 后像素。
        flow_mask: 裁剪后的 H×W LiDAR 有效源像素掩码。

    返回值:
        过滤后的光流和 mask；目标终点不在当前 H×W 内的位置被清零。

    异常和副作用:
        输入形状不匹配时抛出 ValueError；返回克隆张量，不修改输入。

    坐标系和外参约定:
        目标像素等于 source_uv + flow_gt；只改变评估对应集合，不改变
        delta_Ti = Ti @ inverse(T0) 或三维点坐标。
    """
    if flow_img.ndim != 3 or flow_img.shape[-1] != 2:
        raise ValueError("flow_img 形状必须为 H×W×2")
    if flow_mask.shape != flow_img.shape[:2]:
        raise ValueError("flow_mask 形状必须与 flow_img 一致")
    filtered_flow = flow_img.clone()
    filtered_mask = flow_mask.clone()
    source_y, source_x = torch.where(filtered_mask != 0)
    if source_y.numel() == 0:
        return filtered_flow, filtered_mask
    target_x = source_x.to(filtered_flow.dtype) + filtered_flow[
        source_y, source_x, 0]
    target_y = source_y.to(filtered_flow.dtype) + filtered_flow[
        source_y, source_x, 1]
    height, width = filtered_mask.shape
    valid = (
        (target_x >= 0) & (target_x < width)
        & (target_y >= 0) & (target_y < height))
    invalid_y = source_y[~valid]
    invalid_x = source_x[~valid]
    filtered_mask[invalid_y, invalid_x] = 0
    filtered_flow[invalid_y, invalid_x] = 0
    return filtered_flow, filtered_mask


def _crop_evaluation_frame(
        rgb, depth_img_no_occlusion, flow_img, flow_mask, uv, points_3D,
        target_shape, crop_top, crop_left, filter_flow_endpoints=False):
    """同步裁剪 RGB、深度、光流和 mask，并平移像素坐标而保持三维对应不变。

    参数:
        rgb: RGB 图像张量或数组；通道顺序和数值范围遵循调用方约定。
        depth_img_no_occlusion: 经过 z-buffer 去除遮挡的深度图，深度单位为米或按配置归一化。
        flow_img: 稀疏 GT 光流图，形状为 H×W×2，单位为像素。
        flow_mask: 标记存在有效点云光流的位置掩码，形状为 H×W。
        uv: 图像像素坐标，形状为 N×2，顺序为 u、v，单位为像素。
        points_3D: 与投影像素逐项对应的相机坐标三维齐次点，长度单位为米。
        target_shape: 裁剪后网络输入尺寸，按高度、宽度排列，单位为像素。
        crop_top: 从原图顶部移除的像素行数。
        crop_left: 从原图左侧移除的像素列数。
        filter_flow_endpoints: 是否清除 GT 光流终点落到裁剪视野外的点；
            新空间契约启用，旧 checkpoint 保持关闭。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    source_height, source_width = int(rgb.shape[-2]), int(rgb.shape[-1])
    if target_height > source_height or target_width > source_width:
        return rgb, depth_img_no_occlusion, flow_img, flow_mask, uv, points_3D
    crop_top, crop_left = int(crop_top), int(crop_left)
    if not (0 <= crop_top <= source_height - target_height):
        raise ValueError(f'crop_top={crop_top} is invalid for {source_height}x{target_height}')
    if not (0 <= crop_left <= source_width - target_width):
        raise ValueError(f'crop_left={crop_left} is invalid for {source_width}x{target_width}')

    crop_bottom = crop_top + target_height
    crop_right = crop_left + target_width
    rgb = rgb[..., crop_top:crop_bottom, crop_left:crop_right]
    depth_img_no_occlusion = depth_img_no_occlusion[..., crop_top:crop_bottom,
                                                     crop_left:crop_right]
    flow_img = flow_img[crop_top:crop_bottom, crop_left:crop_right]
    flow_mask = flow_mask[crop_top:crop_bottom, crop_left:crop_right]
    if filter_flow_endpoints:
        flow_img, flow_mask = _mask_evaluation_flow_endpoints(
            flow_img, flow_mask)

    in_crop = (
        (uv[:, 0] >= crop_left) & (uv[:, 0] < crop_right)
        & (uv[:, 1] >= crop_top) & (uv[:, 1] < crop_bottom))
    uv = uv[in_crop].clone()
    uv[:, 0] -= crop_left
    uv[:, 1] -= crop_top
    points_3D = points_3D[in_crop].clone()
    return rgb, depth_img_no_occlusion, flow_img, flow_mask, uv, points_3D


def _camera_matrix_for_crop(cam_model, crop_top, crop_left):
    """将相机主点减去裁剪偏移，生成裁剪图像对应的内参矩阵。

    参数:
        cam_model: 已经设置内参的 CameraModel 实例。
        crop_top: 从原图顶部移除的像素行数。
        crop_left: 从原图左侧移除的像素列数。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    camera_matrix = cam_model.get_matrix().copy()
    camera_matrix[0, 2] -= int(crop_left)
    camera_matrix[1, 2] -= int(crop_top)
    return camera_matrix

def _sha256_file(path):
    """以流式方式计算指定文件内容的 SHA-256 摘要，仅用于报告已有产物。

    参数:
        path: 输入或输出文件路径。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_pnp_pose(object_points, image_points, camera_matrix):
    """调用 CUDA RANSAC PnP，由三维点和二维像素估计相机位姿。

    参数:
        object_points: PnP 使用的三维目标点，形状为 N×3，长度单位为米。
        image_points: 与三维点对应的二维图像像素，形状为 N×2，单位为像素。
        camera_matrix: 三乘三针孔相机内参矩阵。

    返回值:
        四乘四齐次相机位姿矩阵；PnP 失败时按实现抛出异常。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    object_points = np.asarray(object_points, dtype=np.float32)
    image_points = np.asarray(image_points, dtype=np.float32)
    result = cv2.pythoncuda.cudaPnP(
        object_points.copy(),
        image_points.copy(),
        object_points.shape[0],
        200,
        2.0,
        np.asarray(camera_matrix, dtype=np.float32).copy(),
    )
    flat = np.asarray(result, dtype=np.float32).reshape(-1)
    if flat.size < 6 or not np.isfinite(flat[:6]).all():
        raise RuntimeError('CUDA PnP returned an invalid pose')
    rotation, _ = cv2.Rodrigues(flat[3:6].reshape(3, 1))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = flat[:3]
    return pose, max(0, flat.size - 6)


def _rotation_error_deg(target_pose, estimated_pose):
    """计算目标位姿与估计位姿之间的 SO(3) 测地旋转误差。

    参数:
        target_pose: 用于误差计算的目标四乘四相机位姿。
        estimated_pose: 待比较的估计四乘四相机位姿。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    delta_rotation = (
        np.asarray(target_pose, dtype=np.float64)[:3, :3]
        @ np.asarray(estimated_pose, dtype=np.float64)[:3, :3].T
    )
    cosine = np.clip((np.trace(delta_rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _reprojection_error_summary(
        object_points, image_points, pose, camera_matrix):
    """统计给定位姿下三维点重投影与目标像素之间的误差分布。

    参数:
        object_points: PnP 使用的三维目标点，形状为 N×3，长度单位为米。
        image_points: 与三维点对应的二维图像像素，形状为 N×2，单位为像素。
        pose: 三维点投影或 PnP 结果使用的四乘四相机位姿。
        camera_matrix: 三乘三针孔相机内参矩阵。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    camera_points = (
        pose[:3, :3] @ np.asarray(object_points, dtype=np.float64).T
        + pose[:3, 3:4]
    ).T
    valid = camera_points[:, 2] > 1e-6
    projected = np.empty((valid.sum(), 2), dtype=np.float64)
    projected[:, 0] = (
        camera_matrix[0, 0] * camera_points[valid, 0]
        / camera_points[valid, 2] + camera_matrix[0, 2]
    )
    projected[:, 1] = (
        camera_matrix[1, 1] * camera_points[valid, 1]
        / camera_points[valid, 2] + camera_matrix[1, 2]
    )
    errors = np.linalg.norm(
        projected - np.asarray(image_points, dtype=np.float64)[valid],
        axis=1,
    )
    summary = _distribution_summary(errors.tolist())
    summary['positive_depth_count'] = int(valid.sum())
    return summary
def _project_lidar_pixels(
        lidar_points, pose, camera_matrix, width, height, max_depth):
    """使用外参和内参投影 LiDAR 齐次点，并过滤深度和图像边界外点。

    参数:
        lidar_points: LiDAR 坐标点云数组，形状为 N×3 或 N×4，长度单位为米。
        pose: 三维点投影或 PnP 结果使用的四乘四相机位姿。
        camera_matrix: 三乘三针孔相机内参矩阵。
        width: 图像宽度，单位为像素。
        height: 图像高度，单位为像素。
        max_depth: 保留投影点的最大 Z 深度，单位为米。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    camera_points = (
        np.asarray(pose, dtype=np.float64)
        @ np.asarray(lidar_points, dtype=np.float64).T
    ).T
    depth = camera_points[:, 2]
    valid_depth = (depth > 0.1) & (depth <= max_depth)
    pixels = np.full((camera_points.shape[0], 2), np.nan, dtype=np.float64)
    pixels[valid_depth, 0] = (
        camera_matrix[0, 0] * camera_points[valid_depth, 0]
        / depth[valid_depth] + camera_matrix[0, 2]
    )
    pixels[valid_depth, 1] = (
        camera_matrix[1, 1] * camera_points[valid_depth, 1]
        / depth[valid_depth] + camera_matrix[1, 2]
    )
    visible = (
        valid_depth
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] < height)
    )
    return pixels, visible


def _metric_summary(trials, key):
    """从多次诊断试验中提取指定指标并计算数值分布摘要。

    参数:
        trials: 重复 PnP 或诊断运行产生的结果字典序列。
        key: 待读取的配置、指标或字典字段名。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    return _distribution_summary([trial[key] for trial in trials])


def _evaluate_gt_flow_pnp_frame(
        selected_frame,
        source_frame,
        camera_matrix,
        pnp_repetitions,
        projection_config):
    """用记录的 GT 光流构造精确 3D–2D 对应，重复 PnP 并比较目标位姿。

    参数:
        selected_frame: 选择清单中描述目标困难帧及其记录字段的字典。
        source_frame: errors.json 中与选中帧对应的完整逐阶段记录。
        camera_matrix: 三乘三针孔相机内参矩阵。
        pnp_repetitions: 为统计稳定性重复执行 PnP 的次数。
        projection_config: 图像尺寸、深度范围和相机投影相关配置。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    rgb_path = Path(selected_frame['rgb_path'])
    lidar_path = Path(selected_frame['lidar_path'])
    if not rgb_path.is_file() or not lidar_path.is_file():
        raise FileNotFoundError(
            f'缺少 KITTI 数据对：rgb={rgb_path}, lidar={lidar_path}')

    with Image.open(str(rgb_path)) as image:
        width, height = image.size
    real_shape = [height, width, 3]

    true_extrinsic = np.asarray(
        selected_frame['T_gt_cam2vel'], dtype=np.float64)
    initial_extrinsic = np.asarray(
        selected_frame['T0_initial_extrinsic'], dtype=np.float64)
    source_true = np.asarray(source_frame['T_gt_cam2vel'], dtype=np.float64)
    source_initial = np.asarray(
        source_frame['T0_initial_extrinsic'], dtype=np.float64)
    if not np.allclose(true_extrinsic, source_true, atol=1e-7, rtol=0.0):
        raise ValueError(
            f"Recorded T_gt mismatch for frame {selected_frame['frame_stem']}")
    if not np.allclose(initial_extrinsic, source_initial, atol=1e-7, rtol=0.0):
        raise ValueError(
            f"Recorded T0 mismatch for frame {selected_frame['frame_stem']}")

    residual = true_extrinsic @ np.linalg.inv(initial_extrinsic)
    if not np.allclose(
            residual @ initial_extrinsic,
            true_extrinsic,
            atol=2e-6,
            rtol=0.0):
        raise ValueError(
            f"帧 {selected_frame['frame_stem']} 不满足 delta_Ti @ T0 不变量")

    lidar = np.fromfile(str(lidar_path), dtype=np.float32).reshape((-1, 4))
    lidar[:, 3] = 1.0
    stage3_extrinsic = np.asarray(
        source_frame['iter3']['T_estimated_cam2vel'], dtype=np.float64)
    comparison_max_depth = float(
        projection_config['visualization_max_camera_depth_m'])
    gt_pixels, gt_visible = _project_lidar_pixels(
        lidar, true_extrinsic, camera_matrix, width, height,
        comparison_max_depth)
    initial_pixels, initial_visible = _project_lidar_pixels(
        lidar, initial_extrinsic, camera_matrix, width, height,
        comparison_max_depth)
    stage3_pixels, stage3_visible = _project_lidar_pixels(
        lidar, stage3_extrinsic, camera_matrix, width, height,
        comparison_max_depth)
    common_visible = gt_visible & initial_visible & stage3_visible
    if common_visible.sum() < 10:
        raise RuntimeError(
            f"三个投影中只有 {common_visible.sum()} 个共同可见点")
    input_projection_errors = np.linalg.norm(
        initial_pixels[common_visible] - gt_pixels[common_visible], axis=1)
    stage3_projection_errors = np.linalg.norm(
        stage3_pixels[common_visible] - gt_pixels[common_visible], axis=1)
    recorded_pose_reprojection = {
        'definition': (
            'same_lidar_point_pixel_distance_to_gt_on_common_visibility'),
        'max_camera_depth_m': comparison_max_depth,
        'common_visible_point_count': int(common_visible.sum()),
        'input_vs_gt_px': _distribution_summary(
            input_projection_errors.tolist()),
        'stage3_vs_gt_px': _distribution_summary(
            stage3_projection_errors.tolist()),
        'stage3_pointwise_improvement_rate': float(np.mean(
            stage3_projection_errors < input_projection_errors)),
    }
    lidar_gpu = torch.from_numpy(lidar.T.copy()).float().cuda()
    true_gpu = torch.from_numpy(true_extrinsic).float().cuda()
    residual_gpu = torch.from_numpy(residual).float().cuda()
    points_true = torch.mm(true_gpu, lidar_gpu)

    max_depth = float(projection_config['max_depth'])
    if max_depth < 100.0:
        points_true = points_true[:, points_true[0, :] < max_depth]

    points_initial = rotate_forward(points_true, residual_gpu)
    cam_params = torch.tensor([
        camera_matrix[0, 0],
        camera_matrix[1, 1],
        camera_matrix[0, 2],
        camera_matrix[1, 2],
        camera_matrix[0, 1],
    ], dtype=torch.float, device='cuda')
    _, uv, visible_indexes, depth = prepare_input(
        cam_params,
        points_initial,
        real_shape,
        None,
        projection_config,
    )
    cam_model = CameraModel().set_intrinsics(cam_params)

    gt_flow, object_points, target_valid = get_flow_zforward(
        uv.float(),
        depth[visible_indexes],
        residual_gpu.inverse(),
        cam_model,
        real_shape,
        scale_flow=False,
        reverse=False,
        get_valid_indexes=True,
    )
    uv = uv[target_valid].clone()
    gt_flow = gt_flow[target_valid].clone()
    object_points = object_points[target_valid].clone()
    target_pixels = uv.float() + gt_flow

    valid = (
        (target_pixels[:, 0] >= 0)
        & (target_pixels[:, 1] >= 0)
        & (target_pixels[:, 0] < width)
        & (target_pixels[:, 1] < height)
    )
    target_pixels = target_pixels[valid]
    object_points = object_points[valid]
    if object_points.shape[0] < 10:
        raise RuntimeError(
            f"仅剩 {object_points.shape[0]} 个真实光流对应点")

    relative_objects = object_points[:, :3].cpu().numpy()
    target_pixels_np = target_pixels.cpu().numpy()
    object_homogeneous = object_points.cpu().numpy()
    absolute_objects = (
        np.linalg.inv(true_extrinsic)
        @ residual
        @ object_homogeneous.T
    ).T[:, :3]

    trials = []
    for repetition in range(pnp_repetitions):
        relative_pose, relative_inliers = _cuda_pnp_pose(
            relative_objects, target_pixels_np, camera_matrix)
        absolute_pose, absolute_inliers = _cuda_pnp_pose(
            absolute_objects, target_pixels_np, camera_matrix)
        residual_after = residual @ np.linalg.inv(relative_pose)
        trials.append({
            'repetition': repetition,
            'translation_error_cm': float(
                np.linalg.norm(
                    absolute_pose[:3, 3] - true_extrinsic[:3, 3])
                * 100.0),
            'rotation_error_deg_evaluator_equivalent': (
                _rotation_error_deg(np.eye(4), residual_after)),
            'rotation_error_deg_absolute_pnp': _rotation_error_deg(
                true_extrinsic, absolute_pose),
            'relative_pose_translation_error_cm': float(
                np.linalg.norm(
                    relative_pose[:3, 3] - residual[:3, 3])
                * 100.0),
            'relative_pose_rotation_error_deg': _rotation_error_deg(
                residual, relative_pose),
            'residual_after_relative_pnp_translation_cm': float(
                np.linalg.norm(residual_after[:3, 3]) * 100.0),
            'relative_inlier_count': int(relative_inliers),
            'relative_inlier_rate': float(
                relative_inliers / relative_objects.shape[0]),
            'absolute_inlier_count': int(absolute_inliers),
            'absolute_inlier_rate': float(
                absolute_inliers / absolute_objects.shape[0]),
            'relative_reprojection_px': _reprojection_error_summary(
                relative_objects,
                target_pixels_np,
                relative_pose,
                camera_matrix,
            ),
            'absolute_reprojection_px': _reprojection_error_summary(
                absolute_objects,
                target_pixels_np,
                absolute_pose,
                camera_matrix,
            ),
            'T_relative_correction': relative_pose.tolist(),
            'T_estimated_cam2vel': absolute_pose.tolist(),
        })

    trial_summary = {
        key: _metric_summary(trials, key)
        for key in (
            'translation_error_cm',
            'rotation_error_deg_evaluator_equivalent',
            'rotation_error_deg_absolute_pnp',
            'relative_pose_translation_error_cm',
            'relative_pose_rotation_error_deg',
            'residual_after_relative_pnp_translation_cm',
            'relative_inlier_rate',
            'absolute_inlier_rate',
        )
    }
    trial_summary['relative_reprojection_median_px'] = (
        _distribution_summary([
            trial['relative_reprojection_px']['median']
            for trial in trials
        ]))
    trial_summary['absolute_reprojection_median_px'] = (
        _distribution_summary([
            trial['absolute_reprojection_px']['median']
            for trial in trials
        ]))

    return {
        'rank': int(selected_frame['rank']),
        'sequence': str(selected_frame['sequence']),
        'frame_stem': str(selected_frame['frame_stem']),
        'rgb_path': str(rgb_path),
        'lidar_path': str(lidar_path),
        'image_size': {'width': width, 'height': height},
        'correspondence_count': int(relative_objects.shape[0]),
        'T_gt_cam2vel': true_extrinsic.tolist(),
        'T0_initial_extrinsic': initial_extrinsic.tolist(),
        'T_gt_from_T0_left_residual': residual.tolist(),
        'recorded_predicted_flow_stage3': {
            'translation_error_cm': float(source_frame['iter3']['t_cm']),
            'rotation_error_deg': float(source_frame['iter3']['r_deg']),
            'T_estimated_cam2vel': source_frame[
                'iter3']['T_estimated_cam2vel'],
        },
        'recorded_pose_reprojection_vs_gt': (
            recorded_pose_reprojection),
        'gt_flow_pnp_across_repetitions': trial_summary,
        'gt_flow_pnp_trials': trials,
    }


def _representative_metric(frame_records, key):
    """从逐帧记录中提取指定指标的中位代表值。

    参数:
        frame_records: 完整评估期间累计的逐帧记录字典列表。
        key: 待读取的配置、指标或字典字段名。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    return [
        frame['gt_flow_pnp_across_repetitions'][key]['median']
        for frame in frame_records
    ]


def evaluate_gt_flow_pnp_diagnostic(
        source_errors_json,
        selection_manifest,
        output_dir,
        pnp_repetitions=5):
    """对选中失败帧执行 GT 光流 PnP 诊断，区分网络误差和 PnP 后端误差。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        selection_manifest: 明确记录诊断所选帧的清单文件路径。
        output_dir: 任务产物目录路径。
        pnp_repetitions: 为统计稳定性重复执行 PnP 的次数。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    source_path = Path(source_errors_json).expanduser().resolve()
    manifest_path = Path(selection_manifest).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if pnp_repetitions <= 0:
        raise ValueError('真实光流 PnP 重复次数必须为正数')
    if not torch.cuda.is_available():
        raise RuntimeError('真实光流 PnP 诊断需要 CUDA')
    if not hasattr(cv2, 'pythoncuda'):
        raise RuntimeError('cv2.pythoncuda is unavailable')

    with source_path.open('r', encoding='utf-8') as handle:
        source = json.load(handle)
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)

    source_sha256 = _sha256_file(source_path)
    expected_sha256 = manifest.get('source_errors_sha256')
    if source_sha256 != expected_sha256:
        raise ValueError(
            'Selection manifest does not match source errors.json: '
            f'{source_sha256} != {expected_sha256}')
    if source.get('dataset') != 'kitti':
        raise ValueError('真实光流 PnP 诊断目前仅支持 KITTI')
    if not source.get('weights'):
        raise ValueError('Source errors.json does not record checkpoints')

    first_checkpoint = Path(source['weights'][0]).expanduser().resolve()
    checkpoint = torch.load(str(first_checkpoint), map_location='cpu')
    max_depth = float(checkpoint['config']['max_depth'])
    projection_config = {
        'max_depth': max_depth,
        'use_reflectance': False,
        'occlusion_threshold': 3.9999,
        'occlusion_kernel': 9,
        'visualization_max_camera_depth_m': float(
            manifest['projection'].get(
                'max_camera_depth_m', 80.0)),
    }

    camera_matrix = np.asarray(
        manifest['projection']['intrinsic_matrix'], dtype=np.float64)
    source_frames = {
        (str(frame.get('sequence')), str(frame.get('frame_stem'))): frame
        for frame in source.get('frames', [])
    }

    frame_records = []
    failures = []
    for selected_frame in manifest.get('frames', []):
        key = (
            str(selected_frame.get('sequence')),
            str(selected_frame.get('frame_stem')),
        )
        source_frame = source_frames.get(key)
        if source_frame is None:
            raise ValueError(f'Selected frame is absent from errors.json: {key}')
        try:
            record = _evaluate_gt_flow_pnp_frame(
                selected_frame,
                source_frame,
                camera_matrix,
                pnp_repetitions,
                projection_config,
            )
            frame_records.append(record)
            gt_t = record['gt_flow_pnp_across_repetitions'][
                'translation_error_cm']['median']
            gt_r = record['gt_flow_pnp_across_repetitions'][
                'rotation_error_deg_evaluator_equivalent']['median']
            print(
                f"[gt-flow-pnp] rank={record['rank']:02d} "
                f"seq={record['sequence']} frame={record['frame_stem']} "
                f"corr={record['correspondence_count']} "
                f"t={gt_t:.4f}cm r={gt_r:.6f}deg")
        except Exception as error:
            failures.append({
                'rank': int(selected_frame.get('rank', -1)),
                'sequence': key[0],
                'frame_stem': key[1],
                'error': f'{type(error).__name__}: {error}',
            })
            print(
                f'[gt-flow-pnp] FAILED rank={selected_frame.get("rank")} '
                f'seq={key[0]} frame={key[1]}: {error}')

    if not frame_records:
        raise RuntimeError('所有选中帧的真实光流 PnP 均失败')

    gt_translation = _representative_metric(
        frame_records, 'translation_error_cm')
    gt_rotation = _representative_metric(
        frame_records, 'rotation_error_deg_evaluator_equivalent')
    gt_absolute_rotation = _representative_metric(
        frame_records, 'rotation_error_deg_absolute_pnp')
    predicted_translation = [
        frame['recorded_predicted_flow_stage3']['translation_error_cm']
        for frame in frame_records
    ]
    predicted_rotation = [
        frame['recorded_predicted_flow_stage3']['rotation_error_deg']
        for frame in frame_records
    ]
    gt_success = [
        translation <= 10.0 and rotation <= 1.0
        for translation, rotation in zip(gt_translation, gt_rotation)
    ]
    predicted_success = [
        translation <= 10.0 and rotation <= 1.0
        for translation, rotation
        in zip(predicted_translation, predicted_rotation)
    ]
    aggregate = {
        'frame_representative': (
            'median result across CUDA RANSAC repetitions per frame'),
        'gt_flow_pnp': {
            'translation_error_cm': _distribution_summary(gt_translation),
            'rotation_error_deg_evaluator_equivalent': (
                _distribution_summary(gt_rotation)),
            'rotation_error_deg_absolute_pnp': (
                _distribution_summary(gt_absolute_rotation)),
            'success_rate_10cm_1deg': float(
                sum(gt_success) / len(gt_success)),
        },
        'recorded_predicted_flow_stage3': {
            'translation_error_cm': _distribution_summary(
                predicted_translation),
            'rotation_error_deg': _distribution_summary(predicted_rotation),
            'success_rate_10cm_1deg': float(
                sum(predicted_success) / len(predicted_success)),
        },
        'gt_flow_pnp_relative_inlier_rate': _distribution_summary(
            _representative_metric(frame_records, 'relative_inlier_rate')),
        'gt_flow_pnp_absolute_inlier_rate': _distribution_summary(
            _representative_metric(frame_records, 'absolute_inlier_rate')),
        'gt_flow_pnp_relative_reprojection_median_px': _distribution_summary(
            _representative_metric(
                frame_records, 'relative_reprojection_median_px')),
        'gt_flow_pnp_absolute_reprojection_median_px': _distribution_summary(
            _representative_metric(
                frame_records, 'absolute_reprojection_median_px')),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / 'gt_flow_pnp_results.json'
    payload = {
        'diagnostic': 'recorded_bad_pairs_gt_flow_vs_predicted_flow_pnp',
        'source_errors_json': str(source_path),
        'source_errors_sha256': source_sha256,
        'selection_manifest': str(manifest_path),
        'selection_manifest_sha256': _sha256_file(manifest_path),
        'first_checkpoint': str(first_checkpoint),
        'first_checkpoint_sha256': _sha256_file(first_checkpoint),
        'projection': {
            'intrinsic_matrix': camera_matrix.tolist(),
            'max_depth_from_checkpoint_m': max_depth,
            'occlusion_threshold': 3.9999,
            'occlusion_kernel': 9,
            'source_pixel_quantization': 'truncate_to_integer',
            'gt_flow_component_suppression_px': 0.5,
        },
        'pnp': {
            'backend': 'cv2.pythoncuda.cudaPnP',
            'ransac_iterations': 200,
            'wrapper_reprojection_argument_px': 2.0,
            'internal_threshold': '2.0 / fx',
            'repetitions_per_frame': int(pnp_repetitions),
            'use_extrinsic_guess': False,
        },
        'n_selected': len(manifest.get('frames', [])),
        'n_complete': len(frame_records),
        'n_failed': len(failures),
        'aggregate': aggregate,
        'failures': failures,
        'frames': frame_records,
    }
    with results_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    csv_path = output_path / 'comparison.csv'
    csv_fields = [
        'rank', 'sequence', 'frame', 'correspondence_count',
        'gt_pnp_t_cm_median', 'gt_pnp_r_deg_median',
        'gt_pnp_absolute_r_deg_median',
        'gt_pnp_relative_inlier_rate_median',
        'gt_pnp_absolute_inlier_rate_median',
        'gt_pnp_absolute_reprojection_px_median',
        'predicted_stage3_t_cm', 'predicted_stage3_r_deg',
        'common_reprojection_point_count',
        'input_vs_gt_reprojection_mean_px',
        'input_vs_gt_reprojection_median_px',
        'input_vs_gt_reprojection_p80_px',
        'input_vs_gt_reprojection_p90_px',
        'stage3_vs_gt_reprojection_mean_px',
        'stage3_vs_gt_reprojection_median_px',
        'stage3_vs_gt_reprojection_p80_px',
        'stage3_vs_gt_reprojection_p90_px',
        'stage3_pointwise_improvement_rate',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for frame in frame_records:
            summary = frame['gt_flow_pnp_across_repetitions']
            predicted = frame['recorded_predicted_flow_stage3']
            reprojection = frame[
                'recorded_pose_reprojection_vs_gt']
            input_reprojection = reprojection['input_vs_gt_px']
            stage3_reprojection = reprojection['stage3_vs_gt_px']
            writer.writerow({
                'rank': frame['rank'],
                'sequence': frame['sequence'],
                'frame': frame['frame_stem'],
                'correspondence_count': frame['correspondence_count'],
                'gt_pnp_t_cm_median': summary[
                    'translation_error_cm']['median'],
                'gt_pnp_r_deg_median': summary[
                    'rotation_error_deg_evaluator_equivalent']['median'],
                'gt_pnp_absolute_r_deg_median': summary[
                    'rotation_error_deg_absolute_pnp']['median'],
                'gt_pnp_relative_inlier_rate_median': summary[
                    'relative_inlier_rate']['median'],
                'gt_pnp_absolute_inlier_rate_median': summary[
                    'absolute_inlier_rate']['median'],
                'gt_pnp_absolute_reprojection_px_median': summary[
                    'absolute_reprojection_median_px']['median'],
                'predicted_stage3_t_cm': predicted['translation_error_cm'],
                'predicted_stage3_r_deg': predicted['rotation_error_deg'],
                'common_reprojection_point_count': reprojection[
                    'common_visible_point_count'],
                'input_vs_gt_reprojection_mean_px': input_reprojection['mean'],
                'input_vs_gt_reprojection_median_px': input_reprojection[
                    'median'],
                'input_vs_gt_reprojection_p80_px': input_reprojection['p80'],
                'input_vs_gt_reprojection_p90_px': input_reprojection['p90'],
                'stage3_vs_gt_reprojection_mean_px': stage3_reprojection['mean'],
                'stage3_vs_gt_reprojection_median_px': stage3_reprojection[
                    'median'],
                'stage3_vs_gt_reprojection_p80_px': stage3_reprojection['p80'],
                'stage3_vs_gt_reprojection_p90_px': stage3_reprojection['p90'],
                'stage3_pointwise_improvement_rate': reprojection[
                    'stage3_pointwise_improvement_rate'],
            })

    gt_t_summary = aggregate['gt_flow_pnp']['translation_error_cm']
    gt_r_summary = aggregate['gt_flow_pnp'][
        'rotation_error_deg_evaluator_equivalent']
    pred_t_summary = aggregate[
        'recorded_predicted_flow_stage3']['translation_error_cm']
    pred_r_summary = aggregate[
        'recorded_predicted_flow_stage3']['rotation_error_deg']
    summary_lines = [
        'KITTI-04 Top-20 bad-pair GT-flow PnP diagnostic',
        '',
        f'Source errors SHA-256: {source_sha256}',
        f'Completed: {len(frame_records)}/{len(manifest.get("frames", []))}',
        f'CUDA RANSAC repetitions per frame: {pnp_repetitions}',
        '',
        'GT-flow PnP (per-frame median across repetitions):',
        (
            f'  translation cm: mean={gt_t_summary["mean"]:.6f}, '
            f'median={gt_t_summary["median"]:.6f}, '
            f'p90={gt_t_summary["p90"]:.6f}, '
            f'max={gt_t_summary["max"]:.6f}'
        ),
        (
            f'  rotation deg: mean={gt_r_summary["mean"]:.8f}, '
            f'median={gt_r_summary["median"]:.8f}, '
            f'p90={gt_r_summary["p90"]:.8f}, '
            f'max={gt_r_summary["max"]:.8f}'
        ),
        (
            '  success <=10cm/1deg: '
            f'{aggregate["gt_flow_pnp"]["success_rate_10cm_1deg"]:.2%}'
        ),
        '',
        'Recorded predicted-flow stage-3 PnP:',
        (
            f'  translation cm: mean={pred_t_summary["mean"]:.6f}, '
            f'median={pred_t_summary["median"]:.6f}, '
            f'p90={pred_t_summary["p90"]:.6f}, '
            f'max={pred_t_summary["max"]:.6f}'
        ),
        (
            f'  rotation deg: mean={pred_r_summary["mean"]:.8f}, '
            f'median={pred_r_summary["median"]:.8f}, '
            f'p90={pred_r_summary["p90"]:.8f}, '
            f'max={pred_r_summary["max"]:.8f}'
        ),
        (
            '  success <=10cm/1deg: '
            f'{aggregate["recorded_predicted_flow_stage3"]["success_rate_10cm_1deg"]:.2%}'
        ),
        '',
        'Interpretation: GT-flow and predicted-flow use the same CUDA PnP '
        'backend and recorded initial poses.',
    ]
    summary_path = output_path / 'summary.txt'
    summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

    fingerprints_path = output_path / 'fingerprints.sha256'
    fingerprints_path.write_text(
        f'{_sha256_file(results_path)}  {results_path.name}\n'
        f'{_sha256_file(csv_path)}  {csv_path.name}\n'
        f'{_sha256_file(summary_path)}  {summary_path.name}\n',
        encoding='utf-8',
    )
    print(f'[gt-flow-pnp] results: {results_path}')
    print('\n'.join(summary_lines))
    return payload

# noinspection PyUnreachableCode
def _recorded_percentage(value, reference):
    """计算当前数值相对参考值的百分比并处理零参考值。

    参数:
        value: 需要换算、比较或写入报告的数值。
        reference: 百分比或差异计算使用的参考数值。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    if abs(reference) < 1e-12:
        return 0.0 if abs(value) < 1e-12 else float("inf")
    return 100.0 * value / reference


def _load_recorded_projection_matrix(scene_directory):
    """从记录场景的标定文件加载三乘四投影矩阵。

    参数:
        scene_directory: 包含图像、点云、内参和外参文件的场景目录。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    calib_path = Path(scene_directory) / 'calib.txt'
    calib = {}
    with calib_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            key, separator, values = line.partition(':')
            if separator:
                calib[key.strip()] = np.fromstring(values, sep=' ')
    projection = calib.get('P2', calib.get('P0'))
    if projection is None or projection.size != 12:
        raise ValueError(f'{scene_directory}: calib.txt has no 3x4 P2/P0')
    return projection.reshape(3, 4)[:, :3].astype(np.float64)

def _recorded_depth_image(
        lidar_points, extrinsic, camera_matrix, width, height):
    """按记录外参投影 LiDAR 点云并生成带 z-buffer 的深度图。

    参数:
        lidar_points: LiDAR 坐标点云数组，形状为 N×3 或 N×4，长度单位为米。
        extrinsic: LiDAR 到相机的四乘四齐次外参矩阵。
        camera_matrix: 三乘三针孔相机内参矩阵。
        width: 图像宽度，单位为像素。
        height: 图像高度，单位为像素。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    camera_points = (
        np.asarray(extrinsic, dtype=np.float64)
        @ np.asarray(lidar_points, dtype=np.float64).T
    ).T
    depth = camera_points[:, 2]
    valid = depth > 0.1
    pixels_x = np.zeros(depth.shape, dtype=np.int64)
    pixels_y = np.zeros(depth.shape, dtype=np.int64)
    pixels_x[valid] = (
        camera_matrix[0, 0] * camera_points[valid, 0] / depth[valid]
        + camera_matrix[0, 2]
    ).astype(np.int64)
    pixels_y[valid] = (
        camera_matrix[1, 1] * camera_points[valid, 1] / depth[valid]
        + camera_matrix[1, 2]
    ).astype(np.int64)
    valid &= (
        (pixels_x >= 0) & (pixels_x < width)
        & (pixels_y >= 0) & (pixels_y < height)
    )
    flat_depth = np.full(height * width, np.inf, dtype=np.float32)
    flat_indexes = pixels_y[valid] * width + pixels_x[valid]
    np.minimum.at(flat_depth, flat_indexes, depth[valid].astype(np.float32))
    flat_depth[~np.isfinite(flat_depth)] = 0.0
    return torch.from_numpy(flat_depth.reshape(height, width))


def _recorded_lidar_path(rgb_path):
    """根据记录的 RGB 帧路径解析同帧 velodyne 点云路径。

    参数:
        rgb_path: 用于解析对应点云帧的 RGB 图像路径。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    rgb_path = Path(rgb_path).expanduser().resolve()
    lidar_path = (
        rgb_path.parent.parent / 'velodyne' / f'{rgb_path.stem}.bin')
    if not lidar_path.is_file():
        raise FileNotFoundError(f'缺少记录的点云：{lidar_path}')
    return lidar_path


def render_recorded_reprojections(
        source_errors_json, output_dir, per_scene_k=50):
    """从 errors.json 中的逐阶段绝对外参重新渲染点云投影对比图。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        output_dir: 任务产物目录路径。
        per_scene_k: 每个场景选择并渲染的样本数量。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    source_path = Path(source_errors_json).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if per_scene_k <= 0:
        raise ValueError('记录重投影的每场景样本数必须为正数')
    with source_path.open('r', encoding='utf-8') as handle:
        source = json.load(handle)
    if source.get('dataset') != 'emd':
        raise ValueError('Recorded reprojections currently require dataset=emd')

    frames = [
        frame for frame in source.get('frames', [])
        if all(f'iter{index}' in frame for index in range(4))
    ]
    by_scene = {}
    for frame in frames:
        by_scene.setdefault(str(frame.get('sequence', '')), []).append(frame)
    if not by_scene:
        raise ValueError('No complete three-stage frame records found')

    selections = []
    for scene in sorted(by_scene):
        ranked = sorted(
            by_scene[scene],
            key=lambda frame: (
                -float(frame['iter3']['t_cm']),
                int(frame['batch_idx']),
            ),
        )
        for rank, frame in enumerate(ranked[:per_scene_k], start=1):
            selections.append((scene, rank, frame))

    output_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 'cmrnext_recorded_reprojections_v1',
        'source_errors_json': str(source_path),
        'source_errors_sha256': _sha256_file(source_path),
        'source_frame_count': len(source.get('frames', [])),
        'complete_three_stage_frame_count': len(frames),
        'ranking_metric': 'iter3.t_cm descending within each scene',
        'per_scene_requested': int(per_scene_k),
        'scene_count': len(by_scene),
        'selected_count': len(selections),
        'layout': ['GT', 'Initial', 'iter1', 'iter2', 'iter3'],
        'network_or_pnp_rerun': False,
        'frames': [],
    }
    csv_rows = []

    for selection_index, (scene, rank, frame) in enumerate(selections, start=1):
        rgb_path = Path(frame['rgb_path']).expanduser().resolve()
        lidar_path = _recorded_lidar_path(rgb_path)
        if not rgb_path.is_file():
            raise FileNotFoundError(f'缺少记录的图像：{rgb_path}')
        with Image.open(str(rgb_path)) as image_handle:
            rgb_array = np.asarray(image_handle.convert('RGB'), dtype=np.float32)
        height, width = rgb_array.shape[:2]
        rgb01 = torch.from_numpy(rgb_array / 255.0).permute(2, 0, 1)
        lidar = np.fromfile(str(lidar_path), dtype=np.float32)
        if lidar.size == 0 or lidar.size % 4 != 0:
            raise ValueError(f'{lidar_path}：应为非空 Nx4 float32 点云')
        lidar = lidar.reshape(-1, 4).astype(np.float64)
        lidar[:, 3] = 1.0
        camera_matrix = _load_recorded_projection_matrix(
            rgb_path.parent.parent)

        true_extrinsic = np.asarray(
            frame['T_gt_cam2vel'], dtype=np.float64)
        t_gt_norm_m = float(np.linalg.norm(true_extrinsic[:3, 3]))
        r_gt_deg = _rot_angle_deg(true_extrinsic)
        projections = [(
            f'GT (true extrinsic) | cam2vel  {_fmt_extrinsic(true_extrinsic)}\n'
            f'|t_gt|={t_gt_norm_m * 100.0:.2f}cm, '
            f'theta_gt={r_gt_deg:.2f} deg  (baseline)',
            _recorded_depth_image(
                lidar, true_extrinsic, camera_matrix, width, height),
        )]
        for iteration in range(4):
            iteration_record = frame[f'iter{iteration}']
            estimated = np.asarray(
                iteration_record['T_estimated_cam2vel'], dtype=np.float64)
            label = (
                'Initial (perturbed)' if iteration == 0
                else f'After iter{iteration}')
            translation_cm = float(iteration_record['t_cm'])
            rotation_deg = float(iteration_record['r_deg'])
            title = (
                f'{label} | est cam2vel  {_fmt_extrinsic(estimated)}\n'
                f'abs err: |t_est-t_gt|={translation_cm:.2f}cm '
                f'({_recorded_percentage(translation_cm, t_gt_norm_m * 100.0):.1f}% of |t_gt|), '
                f'r={rotation_deg:.3f} deg '
                f'({_recorded_percentage(rotation_deg, r_gt_deg):.2f}% of theta_gt)'
            )
            projections.append((
                title,
                _recorded_depth_image(
                    lidar, estimated, camera_matrix, width, height),
            ))

        fig = _draw_reprojection_panels(
            plt, projections, rgb01, overlay_text=False)
        fig.suptitle(
            f'emd | seq={scene} frame={frame["frame_stem"]} | '
            f'tag=worst_rank{rank:03d}',
            fontsize=10,
        )
        scene_directory = output_path / scene
        scene_directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f'worst_rank{rank:03d}_b{int(frame["batch_idx"]):05d}_'
            f'{scene}_{frame["frame_stem"]}.png')
        image_path = scene_directory / filename
        fig.savefig(str(image_path), dpi=120, bbox_inches='tight')
        plt.close(fig)

        relative_image_path = str(image_path.relative_to(output_path))
        item = {
            'scene': scene,
            'rank': rank,
            'batch_idx': int(frame['batch_idx']),
            'frame_stem': str(frame['frame_stem']),
            'ranking_iter3_translation_error_cm': float(
                frame['iter3']['t_cm']),
            'iter3_rotation_error_deg': float(frame['iter3']['r_deg']),
            'rgb_path': str(rgb_path),
            'lidar_path': str(lidar_path),
            'image': relative_image_path,
        }
        for iteration in range(4):
            item[f'iter{iteration}'] = {
                'translation_error_cm': float(
                    frame[f'iter{iteration}']['t_cm']),
                'rotation_error_deg': float(
                    frame[f'iter{iteration}']['r_deg']),
                'T_estimated_cam2vel': frame[
                    f'iter{iteration}']['T_estimated_cam2vel'],
            }
        item['T_gt_cam2vel'] = frame['T_gt_cam2vel']
        manifest['frames'].append(item)
        csv_rows.append({
            'scene': scene,
            'rank': rank,
            'batch_idx': item['batch_idx'],
            'frame_stem': item['frame_stem'],
            **{
                f'iter{iteration}_translation_error_cm': item[
                    f'iter{iteration}']['translation_error_cm']
                for iteration in range(4)
            },
            **{
                f'iter{iteration}_rotation_error_deg': item[
                    f'iter{iteration}']['rotation_error_deg']
                for iteration in range(4)
            },
            'image': relative_image_path,
        })
        print(
            f'[recorded-reprojection] {selection_index}/{len(selections)} '
            f'scene={scene} rank={rank:03d} frame={frame["frame_stem"]} '
            f'iter3_t={frame["iter3"]["t_cm"]:.3f}cm')

    manifest_path = output_path / 'selection_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    csv_path = output_path / 'selection_manifest.csv'
    csv_fields = [
        'scene', 'rank', 'batch_idx', 'frame_stem',
        *[
            f'iter{iteration}_translation_error_cm'
            for iteration in range(4)
        ],
        *[
            f'iter{iteration}_rotation_error_deg'
            for iteration in range(4)
        ],
        'image',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    fingerprints_path = output_path / 'fingerprints.sha256'
    with fingerprints_path.open('w', encoding='utf-8') as handle:
        handle.write(
            f'{_sha256_file(manifest_path)}  {manifest_path.name}\n')
        handle.write(f'{_sha256_file(csv_path)}  {csv_path.name}\n')
    print(
        f'[recorded-reprojection] wrote {len(selections)} images '
        f'across {len(by_scene)} scenes to {output_path}')
    return manifest
def evaluate_calibration(_config, seed):
    """执行 CMRNext 标定评估并写出逐阶段审计结果。

    参数:
        _config: 命令行解析后的评估配置，包含显式场景路径、checkpoint 和
            可选诊断帧数上限。
        seed: NumPy、PyTorch、Python random 和 OpenCV 使用的显式随机种子。

    返回值:
        无显式返回值；指标打印到日志，并按 errors_json 写出结构化结果。

    异常和副作用:
        checkpoint、数据文件、阶段映射或几何约定不合法时抛出异常。函数会使用
        GPU 推理，并可能写入误差 JSON 和可视化文件。checkpoint 的训练数据标签
        不参与评估场景选择。

    坐标系和外参约定:
        点云采用列向量。EMD 初始输入由 T0 @ p_lidar 构造，目标由
        Ti @ p_lidar 构造，完整左乘残差为 delta_Ti = Ti @ inverse(T0)。
        级联后续阶段只细化上一阶段输出，不重复加入数据集残差。
    """
    global EPOCH

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    cv2.setRNGSeed(seed)
    if _config['deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    checkpoint = torch.load(_config['weights'][0], map_location='cpu')
    fixed_nominal_mode = (
        _config['dataset'] == 'kitti'
        and _config.get('kitti_nominal_cam') is not None)
    sampled_kitti_mode = (
        _config['dataset'] == 'kitti'
        and _config.get('kitti_sampled_perturbation', False))
    emd_checkpoint_mode = (
        _config['dataset'] == 'emd' or fixed_nominal_mode or sampled_kitti_mode)
    emd_config = None
    emd_contract = None
    fixed_t0_online_config = None
    fixed_t0_online_mode = False
    if emd_checkpoint_mode:
        emd_config = load_emd_training_config(_config['emd_config'])
        emd_contract = emd_config.contract
        if _config['dataset'] == 'emd':
            # New checkpoints define resize and crop; legacy weights keep source size.
            checkpoint_config = checkpoint.get("config", {})
            saved_resize_shape = checkpoint_config.get("emd_resize_shape")
            saved_evaluation_crop = checkpoint_config.get(
                "emd_evaluation_crop")
            if saved_resize_shape is None:
                saved_resize_shape = [
                    emd_contract.camera_height, emd_contract.camera_width]
                saved_evaluation_crop = None
            if (not isinstance(saved_resize_shape, (list, tuple))
                    or len(saved_resize_shape) != 2):
                raise ValueError(
                    f"checkpoint 中的 EMD 缩放尺寸无效：{saved_resize_shape!r}")
            resize_height, resize_width = (
                int(value) for value in saved_resize_shape)
            if saved_evaluation_crop is None:
                crop_fields = {
                    "evaluation_crop_top": None,
                    "evaluation_crop_left": None,
                    "evaluation_crop_height": None,
                    "evaluation_crop_width": None,
                }
            else:
                crop_fields = {
                    "evaluation_crop_top": int(
                        saved_evaluation_crop["top"]),
                    "evaluation_crop_left": int(
                        saved_evaluation_crop["left"]),
                    "evaluation_crop_height": int(
                        saved_evaluation_crop["height"]),
                    "evaluation_crop_width": int(
                        saved_evaluation_crop["width"]),
                }
            emd_contract = replace(
                emd_contract,
                resize_height=resize_height,
                resize_width=resize_width,
                **crop_fields)
            _config["emd_resize_shape"] = [
                resize_height, resize_width]
            _config["emd_evaluation_crop"] = saved_evaluation_crop
            fixed_t0_online_config = (
                load_fixed_t0_online_perturbation_config(
                    _config['emd_config']))
            fixed_t0_online_mode = fixed_t0_online_config.enabled
        start_stage = _config['emd_start_iteration']
        cascade_stages = list(range(
            start_stage, start_stage + len(_config['weights'])))
        unknown_stages = [stage for stage in cascade_stages
                          if stage not in (1, 2, 3)]
        if unknown_stages:
            raise ValueError(
                'Cascade stages are outside the supported range: '
                f'{unknown_stages}'
            )
    else:
        cascade_stages = list(range(1, len(_config['weights']) + 1))

    if _config['viz']:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as _plt_save  # Used only to save comparisons.
        viz_dir = _config.get('viz_dir') or os.path.join(
            os.path.dirname(os.path.abspath(_config['weights'][-1])), 'viz')
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[viz] 重投影图保存到: {viz_dir}")
    else:
        viz_dir = None

    # Prefer an explicit errors.json path, then the visualization directory.
    errors_json_path = _config.get('errors_json')
    if errors_json_path is None and viz_dir is not None:
        errors_json_path = os.path.join(viz_dir, 'errors.json')
    if errors_json_path is None and _config['dataset'] == 'emd':
        errors_json_path = os.path.join(
            os.path.dirname(os.path.abspath(_config['weights'][-1])),
            f"emd_{_config['emd_level']}_test_errors.json")
    if errors_json_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(errors_json_path)) or '.', exist_ok=True)
        print(f"[viz] 每帧误差 JSON 写到: {errors_json_path}")

    _config['network'] = 'RAFT'
    _config['use_reflectance'] = checkpoint['config']['use_reflectance']
    _config['initial_pool'] = True
    _config['upsample_method'] = checkpoint['config']['upsample_method']
    _config['occlusion_kernel'] = 9
    _config['occlusion_threshold'] = 3.9999
    _config['scaled_gt'] = False
    _config['uncertainty'] = False
    _config['der_type'] = "NLL"
    _config['unc_freeze'] = False
    _config['normalize_images'] = checkpoint['config']['normalize_images']
    _config['max_depth'] = checkpoint['config']['max_depth']
    if 'uncertainty' in checkpoint['config']:
        _config['uncertainty'] = checkpoint['config']['uncertainty']
    _config['fourier_levels'] = -1
    if 'fourier_levels' in checkpoint['config']:
        _config['fourier_levels'] = checkpoint['config']['fourier_levels']
    _config['num_scales'] = 1
    if 'num_scales' in checkpoint['config']:
        _config['num_scales'] = checkpoint['config']['num_scales']
    if 'der_type' in checkpoint['config']:
        _config['der_type'] = checkpoint['config']['der_type']
    if 'unc_freeze' in checkpoint['config']:
        _config['unc_freeze'] = checkpoint['config']['unc_freeze']
    _config['context_encoder'] = 'rgb'
    if 'context_encoder' in checkpoint['config']:
        _config['context_encoder'] = checkpoint['config']['context_encoder']

    mean_torch = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std_torch = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Configure the selected dataset.
    val_directories = []
    base_dir = _config['data_folder']
    if _config['dataset'] == 'argoverse':
        img_shape = (640, 1920 // 2)  # Multiple of 64 at half scale.
        if _config['cam'] is None:
            _config['cam'] = 'ring_front_center'
        else:
            assert _config['cam'] in ['ring_front_center'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif _config['dataset'] == 'kitti':
        img_shape = (384, 1280)  # Multiple of 64.
        for subdir in _config['kitti_test_sequences']:
            val_directories.append(os.path.join(base_dir, subdir))
        if _config['cam'] is None:
            _config['cam'] = '2'
        else:
            assert str(_config['cam']) in ['2', '3'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif _config['dataset'] == 'emd':
        img_shape = _resolve_evaluation_image_shape(checkpoint, 'emd')
        checkpoint_config = checkpoint.get("config", {})
        _config['emd_crop_top_range'] = checkpoint_config.get(
            "emd_crop_top_range", emd_contract.crop_top_range)
        _config['emd_crop_left_range'] = checkpoint_config.get(
            "emd_crop_left_range", emd_contract.crop_left_range)
        if _config['cam'] is None:
            _config['cam'] = '2'
    elif _config['dataset'] == 'pandaset':
        img_shape = (576, 1920 // 2)  # Multiple of 64 at half scale.
        if _config['cam'] is None:
            _config['cam'] = 'front_camera'
        else:
            assert _config['cam'] in ['front_camera'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif _config['dataset'] == 'custom':
        val_directories.append(base_dir)
        first_camera_path = os.listdir(os.path.join(_config['data_folder'], 'camera'))[0]
        first_camera_frame = np.asarray(Image.open(os.path.join(_config['data_folder'], 'camera', first_camera_path)))
        img_shape = [first_camera_frame.shape[0], first_camera_frame.shape[1]]
        if _config['downsample']:
            img_shape = [img_shape[0] // 2, img_shape[1] // 2]
        if img_shape[0] % 64 > 0:
            img_shape[0] = 64 * ((img_shape[0] // 64) + 1)
        if img_shape[1] % 64 > 0:
            img_shape[1] = 64 * ((img_shape[1] // 64) + 1)
    else:
        raise RuntimeError("Dataset unknown")

    if _config['dataset'] == 'kitti':
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'],
                                                   nominal_cam=_config.get('kitti_nominal_cam'))
        if fixed_nominal_mode:
            first_calibration = next(iter(
                dataset_val.kitti_fixed_calibrations.values()))
            residual = first_calibration['residual']
            print(
                '[kitti-fixed] target_cam=' + str(_config['cam'])
                + ' nominal_cam=' + str(_config['kitti_nominal_cam'])
                + f' samples={len(dataset_val)} '
                + f'initial_t={residual[:3, 3].norm().item():.9f} m '
                + f'initial_r={_rot_angle_deg(residual):.9f} deg'
            )
        elif sampled_kitti_mode:
            print(
                '[kitti-sampled] target_cam=' + str(_config['cam'])
                + f' samples={len(dataset_val)} '
                + f'max_t={_config["max_t"]:.3f} m '
                + f'max_r={_config["max_r"]:.3f} deg '
                + f'seed={seed} original_cmrnext_sampling=true'
            )
    elif _config['dataset'] == 'emd':
        scene_paths = _config.get('emd_scene_paths')
        if not scene_paths:
            raise ValueError('EMD 评估必须显式提供 --emd_scene_paths')
        if fixed_t0_online_mode:
            dataset_val = DatasetEMDFixedT0OnlineExtrinsicCalib(
                scene_paths,
                level=_config['emd_level'],
                contract=emd_contract,
                online_config=fixed_t0_online_config,
                seed=seed,
                train=False,
                normalize_images=_config['normalize_images'],
                use_reflectance=_config['use_reflectance'],
            )
        else:
            dataset_val = DatasetEMDExtrinsicCalib(
                scene_paths,
                level=_config['emd_level'],
                contract=emd_contract,
                train=False,
                normalize_images=_config['normalize_images'],
                use_reflectance=_config['use_reflectance'],
            )
        print(
            f"[emd] level={_config['emd_level']} "
            f"samples={len(dataset_val)} "
            f"scenes={dataset_val.scene_paths} "
            f"fixed_t0_online={fixed_t0_online_mode}"
        )
    elif _config['dataset'] == 'argoverse':
        for subdir in ['train4']:
            val_directories.append(os.path.join(_config['data_folder'], subdir))
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'])
    elif _config['dataset'] == 'pandaset':
        for subdir in ['011', '122', '124', '030', '109', '043', '084', '115', '090']:
            val_directories.append(os.path.join(_config['data_folder'], subdir))
        dataset_val = DatasetPandasetExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                    max_t=_config['max_t'],
                                                    use_reflectance=_config['use_reflectance'],
                                                    normalize_images=_config['normalize_images'],
                                                    sensor_id=0, camera=_config['cam'])
    elif _config['dataset'] == 'custom':
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'])

    def init_fn(x):
        """完成 CMRNext 数据评估、光流到 PnP、级联外参细化和报告生成 中的 `init_fn` 操作。

        参数:
            x: 当前层输入张量；通道和空间布局由所在网络模块定义。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

        坐标系和数据约定:
            EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
        """
        return _init_fn(x, seed)

    # Build the evaluation dataset.
    num_worker = _config['num_worker']
    batch_size = 1  # Evaluation requires single-sample batches.

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    print(len(TestImgLoader))
    # Frame limits truncate inference but preserve the full dataset size in JSON.
    max_frames = _config.get('max_frames')
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError('--max_frames 必须为正数')
    if max_frames is not None:
        max_frames = int(max_frames)
        print(
            f'[diagnostic] max_frames={max_frames} '
            f'dataset_frames={len(dataset_val)}')

    models = []
    # Preserve each checkpoint's training domain in errors.json.
    checkpoint_training_domains = []
    print(f"Loading weights from {_config['weights']}")
    for i in range(len(_config['weights'])):
        checkpoint = torch.load(_config['weights'][i], map_location='cpu')
        if emd_checkpoint_mode:
            stage = cascade_stages[i]
            training_domain = _checkpoint_training_domain_record(
                checkpoint, _config['weights'][i], stage)
            checkpoint_training_domains.append(training_domain)
            if training_domain['emd_only'] is not True:
                emd_label = training_domain['emd_only']
                if emd_label is None:
                    emd_label = '<missing>'
                print(
                    '[checkpoint-domain] '
                    f'stage={stage} emd_only={emd_label} '
                    '未声明为 EMD 专用训练或缺少训练域元数据，'
                    '按兼容的跨训练域权重继续评估: '
                    f'{_config["weights"][i]}'
                )
        model = get_model(_config, img_shape)
        saved_state_dict = checkpoint['state_dict']
        clean_state_dict = saved_state_dict
        model.load_state_dict(clean_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        models.append(model)

    errors_r = []
    errors_t = []
    list_quats = []
    list_transl = []
    epe = []
    ransac_time = []
    inference_time = []
    final_calib_RTs = []
    for i in range(len(_config['weights']) + 1):
        errors_r.append([])
        errors_t.append([])
        list_quats.append([])
        list_transl.append([])
        epe.append([])
        final_calib_RTs.append([])
    # Record per-frame paths and stage errors for JSON and ranking.
    frame_records = []
    tbar = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(tbar):
        if max_frames is not None and batch_idx >= max_frames:
            break
        lidar_input = []
        rgb_input = []

        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        for idx in range(len(sample['rgb'])):
            _T_gt_iter0 = sample['cam2vel'][idx].float()
            if _config['dataset'] == 'emd' or fixed_nominal_mode:
                _T_residual_iter0 = sample['residual'][idx].float().cpu()
                _T_est_iter0 = sample['nominal_extrinsic'][idx].float().cpu()
                _q_residual_iter0 = quaternion_from_matrix(
                    _T_residual_iter0.to(sample['rot_error'].device))
                errors_r[0].append(quaternion_distance(
                    _q_residual_iter0,
                    torch.tensor([1., 0., 0., 0.], device=_q_residual_iter0.device)))
            elif sampled_kitti_mode:
                _T_est_iter0, _T_residual_iter0 = (
                    compute_kitti_sampled_initial_state(
                        _T_gt_iter0.cpu(),
                        sample['rot_error'][idx].float().cpu(),
                        sample['tr_error'][idx].float().cpu(),
                    )
                )
                _q_residual_iter0 = quaternion_from_matrix(
                    _T_residual_iter0.to(sample['rot_error'].device))
                errors_r[0].append(quaternion_distance(
                    _q_residual_iter0,
                    torch.tensor([1., 0., 0., 0.],
                                 device=_q_residual_iter0.device)))
            else:
                errors_r[0].append(quaternion_distance(
                    sample['rot_error'][idx],
                    torch.tensor([1., 0., 0., 0.], device=sample['rot_error'].device)))
                # Legacy loaders define the initial estimate by right perturbation.
                _R_pert = quat2mat(sample['rot_error'][idx]).float().cpu()
                _T_pert = tvector2mat(sample['tr_error'][idx]).float().cpu()
                _RT_pert = torch.mm(_T_pert, _R_pert)
                _T_est_iter0 = torch.mm(_T_gt_iter0.cpu(), _RT_pert)
            errors_t[0].append((_T_est_iter0[:3, 3] - _T_gt_iter0.cpu()[:3, 3]).norm().item())
            list_quats[0].append(sample['rot_error'][idx].cpu().numpy())

            real_shape = [
                sample['rgb'][idx].shape[0],
                sample['rgb'][idx].shape[1],
                sample['rgb'][idx].shape[2],
            ]
            crop_top, crop_left = 0, 0
            if _config['dataset'] == 'emd':
                evaluation_crop = _config.get("emd_evaluation_crop")
                if evaluation_crop is None:
                    crop_top, crop_left = _sample_emd_crop_offsets(
                        _config, real_shape, img_shape)
                else:
                    crop_top = int(evaluation_crop["top"])
                    crop_left = int(evaluation_crop["left"])

            # Record the frame ID for JSON and visualization names.
            _rgb_name = sample.get('rgb_name')
            if isinstance(_rgb_name, (list, tuple)):
                _rgb_name = _rgb_name[idx]
            _frame_rec = {
                'batch_idx': int(batch_idx),
                'rgb_path': str(_rgb_name) if _rgb_name is not None else '',
                'iter0': {'t_cm': errors_t[0][-1] * 100.0,
                          'r_deg': float(errors_r[0][-1])},
            }
            # Parse sequence and frame index from the KITTI path.
            try:
                _parts = os.path.normpath(_frame_rec['rgb_path']).split(os.sep)
                _frame_rec['sequence'] = _parts[-3] if len(_parts) >= 3 else ''
                _frame_rec['frame_stem'] = os.path.splitext(_parts[-1])[0] if _parts else ''
            except Exception:
                _frame_rec['sequence'] = ''
                _frame_rec['frame_stem'] = ''
            # Store the per-frame 4x4 ground-truth cam2vel.
            try:
                _frame_rec['T_gt_cam2vel'] = sample['cam2vel'][idx].cpu().numpy().tolist()
            except Exception:
                _frame_rec['T_gt_cam2vel'] = None
            if _config['dataset'] == 'emd' or fixed_nominal_mode:
                _frame_rec['T0_nominal_extrinsic'] = (
                    sample['nominal_extrinsic'][idx].cpu().numpy().tolist())
            if _config['dataset'] == 'emd':
                _frame_rec['evaluation_img_shape'] = [
                    int(img_shape[0]), int(img_shape[1])]
                _frame_rec['crop_top'] = int(crop_top)
                _frame_rec['crop_left'] = int(crop_left)
            elif sampled_kitti_mode:
                _frame_rec['T0_initial_extrinsic'] = (
                    _T_est_iter0.numpy().tolist())
            if _config['dataset'] == 'emd':
                _frame_rec['effective_max_t_m'] = float(
                    sample['effective_max_t'][idx].item())
                _frame_rec['effective_max_r_deg'] = float(
                    sample['effective_max_r_deg'][idx].item())
                _frame_rec['iter0']['t_normalized'] = (
                    errors_t[0][-1] / _frame_rec['effective_max_t_m'])
                _frame_rec['iter0']['r_normalized'] = (
                    errors_r[0][-1] / _frame_rec['effective_max_r_deg'])
                _frame_rec['initial_residual_magnitude'] = math.sqrt(
                    _frame_rec['iter0']['t_normalized'] ** 2
                    + _frame_rec['iter0']['r_normalized'] ** 2)
            elif fixed_nominal_mode:
                _frame_rec['dataset_mode'] = 'kitti_fixed_nominal'
                _frame_rec['target_cam'] = str(_config['cam'])
                _frame_rec['nominal_cam'] = str(_config['kitti_nominal_cam'])
            elif sampled_kitti_mode:
                _frame_rec['dataset_mode'] = 'kitti_sampled_perturbation'
                _frame_rec['target_cam'] = str(_config['cam'])
                _frame_rec['perturbation_seed'] = int(seed)
                _frame_rec['max_t_m'] = float(_config['max_t'])
                _frame_rec['max_r_deg'] = float(_config['max_r'])
            frame_records.append(_frame_rec)

            list_transl[0].append(sample['tr_error'][idx].cpu().numpy())

            # Project with the current pose.

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            if _config['max_depth'] < 100.:
                sample['point_cloud'][idx] = sample['point_cloud'][idx][:,
                                             sample['point_cloud'][idx][0, :] < _config['max_depth']]

            pc_rotated = sample['point_cloud'][idx].clone()
            reflectance = None
            if _config['use_reflectance']:
                reflectance = sample['reflectance'][idx].cuda()

            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT1_inv = torch.mm(T, R)
            extrinsic_error = [RT1_inv.clone().inverse()]
            extrinsic_prediction = [torch.eye(4)]

            # Store the initial perturbation as iter0 and each later residual.
            if frame_records and frame_records[-1].get('batch_idx') == batch_idx:
                _T_gt = sample['cam2vel'][idx]
                if (
                        _config['dataset'] == 'emd'
                        or fixed_nominal_mode
                        or sampled_kitti_mode):
                    _resid_0 = _T_residual_iter0
                else:
                    _resid_0 = extrinsic_error[0].detach().cpu()
                frame_records[-1]['iter0']['T_residual'] = _resid_0.numpy().tolist()
                frame_records[-1]['iter0']['residual_components'] = (
                    _residual_components(_resid_0))
                # The cascade starts at cam2vel = GT @ residual^-1.
                try:
                    if _config['dataset'] == 'emd' or fixed_nominal_mode:
                        _T_est_0 = sample['nominal_extrinsic'][idx].float().cpu()
                    elif sampled_kitti_mode:
                        _T_est_0 = _T_est_iter0
                    else:
                        _T_est_0 = torch.mm(_T_gt.float(), _resid_0.inverse().float())
                    frame_records[-1]['iter0']['T_estimated_cam2vel'] = _T_est_0.numpy().tolist()
                except Exception:
                    frame_records[-1]['iter0']['T_estimated_cam2vel'] = None

            pc_rotated = rotate_forward(pc_rotated, extrinsic_error[0])

            # Project with the random initial extrinsic.
            cam_params = sample['calib'][idx].cuda()
            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, pc_rotated, real_shape,
                                                                       reflectance, _config)
            cam_model = CameraModel().set_intrinsics(cam_params)

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth[indexes], RT1_inv, cam_model,
                                                             [real_shape[0], real_shape[1], 3],
                                                             scale_flow=False, reverse=False,
                                                             get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()

            # Normalize the image.
            rgb = rgb / 255.
            if _config['normalize_images']:
                rgb = (rgb - mean_torch) / std_torch
            rgb = rgb.permute(2, 0, 1)
            sample['rgb'][idx] = rgb

            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            # Mark pixels with projected points in the flow mask.
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            if _config['viz']:
                viz_initial = overlay_imgs(rgb, depth_img_no_occlusion[-1].unsqueeze(0).unsqueeze(0), max_depth=0.5,
                                           close_thr=1000)

            points_3D = points_3D[new_indexes].clone()
            network_real_shape = real_shape
            if _config['dataset'] == 'emd':
                (rgb, depth_img_no_occlusion, flow_img, flow_mask, uv,
                 points_3D) = _crop_evaluation_frame(
                    rgb, depth_img_no_occlusion, flow_img, flow_mask, uv,
                    points_3D, img_shape, crop_top, crop_left,
                    filter_flow_endpoints=(
                        _config.get("emd_evaluation_crop") is not None))
                network_real_shape = [img_shape[0], img_shape[1], real_shape[2]]
            rgb, depth_img_no_occlusion, flow_img, flow_mask = downsample_and_pad(_config, rgb, depth_img_no_occlusion,
                                                                                  img_shape, network_real_shape, flow_img,
                                                                                  flow_mask)

            rgb_input.append(rgb)
            lidar_input.append(depth_img_no_occlusion)

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)

        for iteration in range(len(_config['weights'])):
            torch.cuda.synchronize()
            time1 = time.time()
            # Predict LiDAR-to-RGB pixel displacement.
            with torch.no_grad():
                predicted_flow = models[iteration](rgb_input, lidar_input)
                torch.cuda.synchronize()
                time2 = time.time()
                predicted_flow, predicted_uncertainty = predicted_flow
                inference_time.append(time2 - time1)
                # Upsample flow when needed.
                if _config['dataset'] in ['argoverse', 'pandaset'] or _config['downsample']:
                    predicted_flow = list(predicted_flow)
                    predicted_flow[-1] *= 2
                    predicted_flow[-1] = F.interpolate(predicted_flow[-1], scale_factor=2, mode='bilinear')
                    if _config['uncertainty']:
                        predicted_uncertainty[-1] = F.interpolate(predicted_uncertainty[-1], scale_factor=2,
                                                                  mode='bilinear')

            up_flow = predicted_flow[-1]

            # Compute EPE.
            gt = flow_img.clone().permute(2, 0, 1)
            gt = torch.cat((gt, flow_mask.unsqueeze(0).float()))
            gt = gt.unsqueeze(0)
            epe[iteration].append(EndPointError(up_flow, gt).item())

            up_flow = up_flow[0].permute(1, 2, 0)

            new_uv = uv.float() + up_flow[uv[:, 1], uv[:, 0]]

            valid_indexes = flow_mask[uv[:, 1], uv[:, 0]] == 1

            if flow_mask.sum() < 10:
                if frame_records and frame_records[-1].get('batch_idx') == batch_idx:
                    frame_records[-1]['failed_at_iter'] = iteration + 1
                    frame_records[-1]['failure_reason'] = (
                        'projected_point_mask_lt_10')
                    frame_records[-1]['projected_point_count'] = int(
                        flow_mask.sum().item())
                break

            if _config['uncertainty']:
                sum_uncertainty = predicted_uncertainty[-1][0, 0] + predicted_uncertainty[-1][0, 1]
                mean_uncertainty = sum_uncertainty * flow_mask
                try:
                    mean_uncertainty = np.quantile(mean_uncertainty[flow_mask != 0].detach().cpu().numpy(),
                                                   _config['quantile'])
                except:
                    pass
                valid_indexes = valid_indexes & (sum_uncertainty[uv[:, 1], uv[:, 0]] < mean_uncertainty)

            # Check only pixels inside the image.
            valid_indexes = valid_indexes & (new_uv[:, 0] < flow_mask.shape[1])
            valid_indexes = valid_indexes & (new_uv[:, 1] < flow_mask.shape[0])
            valid_indexes = valid_indexes & (new_uv[:, 0] >= 0)
            valid_indexes = valid_indexes & (new_uv[:, 1] >= 0)
            new_uv = new_uv[valid_indexes]

            valid_indexes2 = torch.ones(new_uv.shape[0], dtype=torch.bool).cuda()

            new_uv = new_uv[valid_indexes2]

            points_2d = new_uv.cpu().numpy()
            obj_coord = points_3D[valid_indexes][valid_indexes2][:, :3].cpu().numpy()
            obj_coord_zforward = np.zeros(obj_coord.shape)
            obj_coord_zforward[:, 0] = obj_coord[:, 0]
            obj_coord_zforward[:, 1] = obj_coord[:, 1]
            obj_coord_zforward[:, 2] = obj_coord[:, 2]
            cam_mat = _camera_matrix_for_crop(cam_model, crop_top, crop_left)

            torch.cuda.synchronize()

            time1 = time.time()
            if obj_coord_zforward.shape[0] < 10:
                for left_iter in range(iteration + 1):
                    errors_t[left_iter].pop(-1)
                    errors_r[left_iter].pop(-1)
                    list_transl[left_iter].pop(-1)
                    list_quats[left_iter].pop(-1)
                # Preserve the exact reason for an incomplete cascade.
                if frame_records and frame_records[-1].get('batch_idx') == batch_idx:
                    frame_records[-1]['failed_at_iter'] = iteration + 1
                    frame_records[-1]['failure_reason'] = (
                        'pnp_correspondences_lt_10')
                    frame_records[-1]['pnp_correspondence_count'] = int(
                        obj_coord_zforward.shape[0])
                break

            if _config['quantile'] < 1.0 and not _config['uncertainty']:
                num_corr = points_2d.shape[0]
                corr_to_keep = list(range(num_corr))
                random.shuffle(corr_to_keep)
                num_corr_to_keep = int(num_corr * _config['quantile'])
                if num_corr_to_keep > 10:
                    points_2d = points_2d[corr_to_keep[:num_corr_to_keep]]
                    obj_coord_zforward = obj_coord_zforward[corr_to_keep[:num_corr_to_keep]]

            # Refine the relative transform from CMRNext correspondences.
            cuda_pnp = cv2.pythoncuda.cudaPnP(obj_coord_zforward.astype(np.float32).copy(),
                                              points_2d.astype(np.float32).copy(), obj_coord_zforward.shape[0],
                                              200, 2., cam_mat.astype(np.float32)
                                              )

            transl = cuda_pnp[0, [0, 1, 2]]
            rot_mat = cuda_pnp[:, 3:6].T
            rot_mat, _ = cv2.Rodrigues(rot_mat)

            torch.cuda.synchronize()
            time2 = time.time()
            ransac_time.append(time2 - time1)

            transl = torch.tensor(transl).float().squeeze().cuda()

            rot_mat = torch.tensor(rot_mat)

            pred_quaternion = quaternion_from_matrix(rot_mat)

            R_predicted = quat2mat(pred_quaternion).cuda()
            T_predicted = tvector2mat(transl)
            RT_predicted = torch.mm(T_predicted, R_predicted)
            composed = torch.mm(extrinsic_error[iteration], RT_predicted.inverse())
            extrinsic_error.append(composed)
            extrinsic_prediction.append(torch.mm(extrinsic_prediction[iteration], RT_predicted.inverse().cpu()))

            T_composed = composed[:3, 3]
            R_composed = quaternion_from_matrix(composed)
            errors_r[iteration + 1].append(quaternion_distance(R_composed,
                                                               torch.tensor([1., 0., 0., 0.],
                                                                            device=R_composed.device)))

            list_transl[iteration + 1].append(T_composed.cpu().numpy())
            list_quats[iteration + 1].append(R_composed.cpu().numpy())

            # Compute the final camera-LiDAR matrix.
            points_3D_orig = torch.mm(extrinsic_prediction[-2].to(points_3D.device), points_3D[valid_indexes].T).T
            points_3D_orig = torch.mm(extrinsic_error[0], points_3D_orig.T).T
            points_3D_orig = rotate_forward(points_3D_orig, sample['cam2vel'][0].to(points_3D_orig.device))
            final_correspondences = new_uv, points_3D_orig
            cuda_pnp_final = cv2.pythoncuda.cudaPnP(
                final_correspondences[1][:, :3].cpu().numpy().astype(np.float32).copy(),
                final_correspondences[0].cpu().numpy().astype(np.float32).copy(),
                points_3D_orig.shape[0], 200, 2., cam_mat.astype(np.float32))
            transl_final = cuda_pnp_final[0, :3]
            rot_mat_final = cuda_pnp_final[:, 3:6].T
            rot_mat_final, _ = cv2.Rodrigues(rot_mat_final)
            rot_mat_final = torch.tensor(rot_mat_final)
            T_predicted_final = tvector2mat(torch.tensor(transl_final))
            R_predicted_final = torch.eye(4)
            R_predicted_final[:3, :3] = rot_mat_final.clone().detach()
            # Each stage predicts cam2vel.
            RT_predicted_final = torch.mm(T_predicted_final, R_predicted_final)
            _resid_n = extrinsic_error[-1].detach().cpu().float()
            if _config['dataset'] == 'emd':
                # Derive metrics, matrices, and reprojections from the same
                # complete left-residual cascade state.
                _T_est_full = _estimate_from_left_residual(
                    sample['cam2vel'][idx], _resid_n)
                _absolute_pose_source = (
                    'inverse(T_residual) @ T_gt_cam2vel')
            else:
                _T_est_full = RT_predicted_final.detach().cpu().float()
                _absolute_pose_source = 'legacy_final_absolute_pnp'
            final_calib_RTs[iteration + 1].append(_T_est_full)

            # Translation error is the absolute extrinsic translation difference.
            _t_gt_vec = sample['cam2vel'][idx][:3, 3].float().cpu()
            _t_est_vec = _T_est_full[:3, 3]
            errors_t[iteration + 1].append((_t_est_vec - _t_gt_vec).norm().item())

            # Append this stage's errors.
            if frame_records and frame_records[-1].get('batch_idx') == batch_idx:
                frame_records[-1][f'iter{iteration + 1}'] = {
                    't_cm': float(errors_t[iteration + 1][-1]) * 100.0,
                    'r_deg': float(errors_r[iteration + 1][-1]),
                    'T_residual': _resid_n.numpy().tolist(),
                    'residual_components': _residual_components(_resid_n),
                    'T_estimated_cam2vel': _T_est_full.numpy().tolist(),
                    'T_estimated_cam2vel_source': _absolute_pose_source,
                }
                if _config['dataset'] == 'emd':
                    frame_records[-1][f'iter{iteration + 1}']['t_normalized'] = (
                        float(errors_t[iteration + 1][-1])
                        / frame_records[-1]['effective_max_t_m'])

            if errors_t[iteration + 1][-1] > 4.:
                # This frame failed.
                if frame_records and frame_records[-1].get('batch_idx') == batch_idx:
                    frame_records[-1]['failed_after_iter'] = iteration + 1
                    frame_records[-1]['failure_reason'] = (
                        'translation_error_gt_4m')
                for left_iteration in range(iteration + 2, len(_config['weights']) + 1):
                    errors_t[left_iteration].append(errors_t[iteration + 1][-1])
                    errors_r[left_iteration].append(errors_r[iteration + 1][-1])
                break

            # Transform the cloud and build the next-stage input.
            rotated_point_cloud = rotate_forward(sample['point_cloud'][idx], extrinsic_error[-1])

            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, rotated_point_cloud, real_shape,
                                                                       reflectance, _config)

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth[indexes], extrinsic_error[-1].inverse(),
                                                             cam_model, [real_shape[0], real_shape[1], 3],
                                                             scale_flow=False, reverse=False,
                                                             get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()
            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            points_3D = points_3D[new_indexes].clone()
            network_real_shape = real_shape
            if _config['dataset'] == 'emd':
                (rgb, depth_img_no_occlusion, flow_img, flow_mask, uv,
                 points_3D) = _crop_evaluation_frame(
                    rgb, depth_img_no_occlusion, flow_img, flow_mask, uv,
                    points_3D, img_shape, crop_top, crop_left,
                    filter_flow_endpoints=(
                        _config.get("emd_evaluation_crop") is not None))
                network_real_shape = [img_shape[0], img_shape[1], real_shape[2]]
            rgb, depth_img_no_occlusion, flow_img, flow_mask = downsample_and_pad(_config, rgb, depth_img_no_occlusion,
                                                                                  img_shape, network_real_shape,
                                                                                  flow_img, flow_mask)

            rgb_input = rgb.unsqueeze(0)
            lidar_input = depth_img_no_occlusion.unsqueeze(0)

            # first_n renders by batch order; other strategies rank after inference.
            if (_config['viz'] and _config.get('viz_strategy', 'topk_iter3_t') == 'first_n'
                    and (_config.get('viz_max_frames', 0) == 0
                         or batch_idx * len(sample['rgb']) + idx < _config['viz_max_frames'])):
                gt_uv, gt_depth, _, _ = cam_model.project_pytorch(rotated_point_cloud, real_shape, reflectance)
                gt_uv = gt_uv.t().int().contiguous()

                new_depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                new_depth_img += 1000.
                new_depth_img = visibility.depth_image(gt_uv.int().contiguous(), gt_depth, new_depth_img,
                                                       gt_uv.shape[0],
                                                       real_shape[1], real_shape[0])
                new_depth_img[new_depth_img == 1000.] = 0.

                new_depth_img_no_occlusion = torch.zeros_like(new_depth_img, device='cuda')
                new_depth_img_no_occlusion = visibility.visibility2(new_depth_img, cam_params,
                                                                    new_depth_img_no_occlusion,
                                                                    new_depth_img.shape[1], new_depth_img.shape[0],
                                                                    _config['occlusion_threshold'],
                                                                    _config['occlusion_kernel'])

                lidar_flow = new_depth_img_no_occlusion.unsqueeze(0).unsqueeze(0)
                viz_final = overlay_imgs(sample['rgb'][idx].cuda(), lidar_flow, max_depth=_config['max_depth'] / 2,
                                         close_thr=1000)

                fig, axarr = _plt_save.subplots(2, 1, figsize=(14, 8))
                axarr[0].imshow(viz_initial); axarr[0].set_title('Initial (perturbed)')
                axarr[0].axis('off')
                axarr[1].imshow(viz_final);   axarr[1].set_title(f'After Iter {iteration + 1}')
                axarr[1].axis('off')
                frame_id = f"b{batch_idx:05d}_s{idx}_it{iteration + 1}"
                fig.savefig(os.path.join(viz_dir, f'{frame_id}.png'),
                            dpi=120, bbox_inches='tight')
                _plt_save.close(fig)

            try:
                if _config['dataset'] != 'custom':
                    tbar.set_postfix(t_mean=torch.tensor(errors_t[-1]).mean().item(),
                                     t_median=torch.tensor(errors_t[-1]).median().item(),
                                     r_mean=torch.tensor(errors_r[-1]).mean().item(),
                                     r_median=torch.tensor(errors_r[-1]).median().item(),
                                     epe_mean=torch.tensor(epe[0]).mean().item())
            except:
                pass

    print(f'Network Time: ', torch.tensor(inference_time).mean())
    print(f'PnP+RANSAC Time: ', torch.tensor(ransac_time).mean())
    flow_epe_report = _build_flow_epe_report(epe, cascade_stages)
    for iteration_key, metrics in flow_epe_report['iterations'].items():
        print(
            f"[flow-epe] {iteration_key} stage={metrics['stage']} "
            f"count={metrics['count']} mean={metrics['mean']:.6f} "
            f"median={metrics['median']:.6f}")
    for iteration in range(len(_config['weights']) + 1):
        errors_t[iteration] = torch.tensor(errors_t[iteration])
        errors_r[iteration] = torch.tensor(errors_r[iteration])
    emd_report = None
    kitti_fixed_nominal_report = None
    kitti_sampled_perturbation_report = None
    if _config['dataset'] == 'emd':
        emd_report = _build_emd_report(
            frame_records, len(_config['weights']), emd_config.contract,
            cascade_stages)
    elif fixed_nominal_mode:
        kitti_fixed_nominal_report = _build_kitti_stage_report(
            frame_records, len(_config['weights']), cascade_stages)
    elif sampled_kitti_mode:
        kitti_sampled_perturbation_report = _build_kitti_stage_report(
            frame_records, len(_config['weights']), cascade_stages)

    if _config['dataset'] == 'custom' and len(final_calib_RTs[len(_config['weights'])]) == 1:
        print("Predicted extrinsic calibration:")
        print(final_calib_RTs[len(_config['weights'])][0])
    elif _config['dataset'] == 'custom':
        iteration = len(_config['weights'])
        final_quats = np.stack([quaternion_from_matrix(t) for t in final_calib_RTs[iteration]])
        avg_quaternion = torch.from_numpy(average_quaternions(final_quats))
        median_quaternion = quaternion_median(torch.tensor(np.stack(final_quats)))
        mode_quaternion = quaternion_mode(final_quats, 4)

        avg_translation = torch.stack(final_calib_RTs[iteration])[:, :3, 3].mean(0)
        median_translation = torch.stack(final_calib_RTs[iteration])[:, :3, 3].median(0)[0]
        mode_translation = quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 2)

        R = quat2mat(avg_quaternion)
        T = tvector2mat(avg_translation)
        avg_extrinsic_calib = torch.mm(T, R)
        torch.set_printoptions(5, sci_mode=False)
        print("Predicted extrinsic calibration using average aggregation:")
        print(avg_extrinsic_calib)

        R = quat2mat(median_quaternion)
        T = tvector2mat(median_translation)
        median_extrinsic_calib = torch.mm(T, R)
        torch.set_printoptions(5, sci_mode=False)
        print("Predicted extrinsic calibration using median aggregation:")
        print(median_extrinsic_calib)

        R = quat2mat(mode_quaternion)
        T = tvector2mat(mode_translation)
        mode_extrinsic_calib = torch.mm(T, R)
        print("Predicted extrinsic calibration using mode aggregation:")
        print(mode_extrinsic_calib)

    else:
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
        table.title = f"CMRNext Results on {_config['dataset']}, camera {_config['cam']}"
        table.add_column("Iteration")
        table.add_column("Median Translation error (cm)", justify="center", max_width=20)
        table.add_column("Median Rotation error (˚)", justify="center", max_width=20)
        table.add_row(
            f"Initial Pose",
            f"{errors_t[0].median().item() * 100:.2f}",
            f"{errors_r[0].median().item():.2f}"
        )
        for iteration in range(1, len(_config['weights']) + 1):
            table.add_row(
                (f"Stage {cascade_stages[iteration - 1]}"
                 if emd_checkpoint_mode else f"Iteration {iteration}"),
                f"{errors_t[iteration].median().item() * 100:.2f}",
                f"{errors_r[iteration].median().item():.2f}"
            )
        print("")
        print("")
        console.print(table)

        if _config['dataset'] == 'emd':
            table = Table(
                show_header=True, header_style="bold magenta",
                box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
            thresholds = emd_report['thresholds']
            table.title = (
                "EMD 场景指标 "
                f"(success <= {thresholds['translation_m']:.2f} m, "
                f"{thresholds['rotation_deg']:.2f} deg)")
            table.add_column("Iteration")
            table.add_column("Median t (m)", justify="center")
            table.add_column("Median t / effective_max_t", justify="center")
            table.add_column("Median r (deg)", justify="center")
            table.add_column("Success", justify="center")
            for iteration in range(len(_config['weights']) + 1):
                metrics = emd_report['iterations'][f'iter{iteration}']
                median_t = metrics['translation_m']['median']
                median_t_normalized = metrics['translation_normalized']['median']
                median_r = metrics['rotation_deg']['median']
                success_rate = metrics['success_rate']
                table.add_row(
                    ("Initial Pose" if iteration == 0
                     else f"Stage {cascade_stages[iteration - 1]}"),
                    f"{median_t:.4f}" if median_t is not None else "n/a",
                    (f"{median_t_normalized:.4f}"
                     if median_t_normalized is not None else "n/a"),
                    f"{median_r:.3f}" if median_r is not None else "n/a",
                    f"{100.0 * success_rate:.2f}%" if success_rate is not None else "n/a",
                )
            print("")
            console.print(table)
            print("[emd] Temporal aggregation skipped: Ti varies per sample.")
        elif fixed_nominal_mode or sampled_kitti_mode:
            if fixed_nominal_mode:
                kitti_report = kitti_fixed_nominal_report
                mode_title = 'KITTI fixed nominal metrics'
                mode_prefix = '[kitti-fixed]'
            else:
                kitti_report = kitti_sampled_perturbation_report
                mode_title = 'KITTI sampled perturbation metrics'
                mode_prefix = '[kitti-sampled]'
            table = Table(
                show_header=True, header_style='bold magenta',
                box=box.MINIMAL_HEAVY_HEAD, title_style='bold red')
            thresholds = kitti_report['thresholds']
            table.title = (
                mode_title + ' '
                f"(success <= {thresholds['translation_m']:.2f} m, "
                f"{thresholds['rotation_deg']:.2f} deg)")
            table.add_column('Stage')
            table.add_column('Count', justify='center')
            table.add_column('Median t (m)', justify='center')
            table.add_column('Median r (deg)', justify='center')
            table.add_column('Success', justify='center')
            for iteration in range(len(_config['weights']) + 1):
                metrics = kitti_report['iterations'][f'iter{iteration}']
                median_t = metrics['translation_m']['median']
                median_r = metrics['rotation_deg']['median']
                success_rate = metrics['success_rate']
                table.add_row(
                    ('Initial Pose' if iteration == 0
                     else f'Stage {cascade_stages[iteration - 1]}'),
                    str(metrics['count']),
                    f'{median_t:.4f}' if median_t is not None else 'n/a',
                    f'{median_r:.3f}' if median_r is not None else 'n/a',
                    f'{100.0 * success_rate:.2f}%' if success_rate is not None else 'n/a',
                )
            print('')
            console.print(table)
            print(
                mode_prefix + ' completion='
                f"{100.0 * kitti_report['completion_rate']:.2f}% "
                'all-frame final success='
                f"{100.0 * kitti_report['final_success_rate_all_frames']:.2f}%")
        else:
            table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
            table.title = f"Temporal Aggregation Results on {_config['dataset']}"
            table.add_column("Aggregation Measure", max_width=13)
            table.add_column("Translation Error (cm)", justify="center")
            table.add_column("Rotation Error (˚)", justify="center")

            iteration = len(_config['weights'])
            final_quats = np.stack([quaternion_from_matrix(t) for t in final_calib_RTs[iteration]])
            r_error_avg = quaternion_distance(
                torch.from_numpy(average_quaternions(final_quats)),
                quaternion_from_matrix(sample['cam2vel'][0])
            )
            r_error_mode = quaternion_distance(
                quaternion_mode(final_quats, 4),
                quaternion_from_matrix(sample['cam2vel'][0])
            )
            if r_error_mode > quaternion_distance(
                    quaternion_mode(final_quats, 3),
                    quaternion_from_matrix(sample['cam2vel'][0])
            ):
                r_error_mode = quaternion_distance(
                    quaternion_mode(final_quats, 3),
                    quaternion_from_matrix(sample['cam2vel'][0])
                )

            t_error_avg = (torch.stack(final_calib_RTs[iteration])[:, :3, 3].mean(0)
                           - sample['cam2vel'][0][:3, 3]).norm() * 100.
            t_error_median = (torch.stack(final_calib_RTs[iteration])[:, :3, 3].median(0)[0]
                              - sample['cam2vel'][0][:3, 3]).norm() * 100.
            t_error_mode = (quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 2)
                            - sample['cam2vel'][0][:3, 3]).norm() * 100.
            if t_error_mode > (quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 1)
                               - sample['cam2vel'][0][:3, 3]).norm() * 100.:
                t_error_mode = (quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 1)
                                - sample['cam2vel'][0][:3, 3]).norm() * 100.
            table.add_row(
                "Mean",
                f"[bold green]{t_error_avg.item():.2f}[/bold green]" if t_error_avg.item() <= t_error_median.item() and
                                                                        t_error_avg.item() <= t_error_mode.item() else
                f"{t_error_avg.item():.2f}",
                f"[bold green]{r_error_avg:.2f}[/bold green]" if r_error_avg <= r_error_mode else
                f"{r_error_avg:.2f}",
            )
            table.add_row(
                "Median",
                f"[bold green]{t_error_median.item():.2f}[/bold green]" if t_error_median.item() <= t_error_avg.item() and
                                                                           t_error_median.item() <= t_error_mode.item() else
                f"{t_error_median.item():.2f}",
                f"---"
            )
            table.add_row(
                "Mode",
                f"[bold green]{t_error_mode.item():.2f}[/bold green]" if t_error_mode.item() <= t_error_avg.item() and
                                                                         t_error_mode.item() <= t_error_median.item() else
                f"{t_error_mode.item():.2f}",
                f"[bold green]{r_error_mode:.2f}[/bold green]" if r_error_mode <= r_error_avg else
                f"{r_error_mode:.2f}"
            )
            print("")
            print("")
            console.print(table)

    if _config['save_file'] is not None:
        torch.save(errors_t, f'./{_config["save_file"]}_errors_t.torch')
        torch.save(errors_r, f'./{_config["save_file"]}_errors_r.torch')

    # Write per-frame errors to the resolved JSON path.
    if errors_json_path is not None:
        import json
        complete = [r for r in frame_records if all(f'iter{i}' in r for i in range(len(_config['weights']) + 1))]
        # Aggregate centimeter and degree metrics.
        n_iter = len(_config['weights'])
        aggregate = {}
        for it in range(n_iter + 1):
            ts = [r[f'iter{it}']['t_cm'] for r in complete]
            rs = [r[f'iter{it}']['r_deg'] for r in complete]
            aggregate[f'iter{it}'] = {
                't_cm': _distribution_summary(ts),
                'r_deg': _distribution_summary(rs),
            }
        if fixed_t0_online_mode:
            init_perturbation = {
                'mode': 'fixed_t0_deterministic_per_sample_box',
                'source_mode': emd_contract.source_mode,
                'physical_extrinsic_mode': (
                    'recorded_pose_Ti_equals_product_T0'
                    if emd_contract.source_mode == 'recorded_pose_v1'
                    else 'stored_fixed_Ti_equals_T0'),
                'training_semantics': 'nominal_to_true_residual_v1',
                'residual_convention': 'delta_Ti = Ti @ inverse(T_input)',
                'stochastic_during_evaluation': False,
                'deterministic_realization': True,
                'seed': int(seed),
                'seed_derivation': 'SeedSequence([seed, sample_index])',
                'max_translation_component_m': (
                    fixed_t0_online_config.max_translation_component_m),
                'max_rotation_euler_component_deg': (
                    fixed_t0_online_config.max_rotation_euler_component_deg),
                'curriculum_level': _config['emd_level'],
            }
        elif (
                _config['dataset'] == 'emd'
                and emd_contract.source_mode == 'recorded_pose_v1'):
            init_perturbation = {
                'mode': 'recorded_pose_fixed_product_T0',
                'source_mode': emd_contract.source_mode,
                'training_semantics': 'nominal_to_true_residual_v1',
                'pose_format': emd_contract.recorded_pose_format,
                'true_extrinsic_source': (
                    f'each_scene/{emd_contract.true_extrinsics_file}'),
                'nominal_extrinsic_source': (
                    'dataset.recorded_pose.nominal_extrinsic'),
                'T0_nominal_extrinsic': np.asarray(
                    emd_contract.recorded_nominal_extrinsic,
                    dtype=np.float64,
                ).reshape(4, 4).tolist(),
                'residual_convention': 'delta_Ti = Ti @ inverse(T0)',
                'random_perturbation_applied': False,
                'curriculum_level': _config['emd_level'],
            }
        elif _config['dataset'] == 'emd':
            init_perturbation = {
                'mode': 'stored_emd_residuals',
                'training_semantics': 'nominal_to_true_residual_v1',
                'curriculum_level': _config['emd_level'],
                'random_perturbation_applied': False,
            }
        elif fixed_nominal_mode:
            first_record = frame_records[0]
            first_calibration = next(iter(
                dataset_val.kitti_fixed_calibrations.values()))
            init_perturbation = {
                'mode': 'kitti_fixed_camera_extrinsic',
                'training_semantics': 'nominal_to_true_residual_v1',
                'random_perturbation_applied': False,
                'target_cam': str(_config['cam']),
                'nominal_cam': str(_config['kitti_nominal_cam']),
                'T0_nominal_extrinsic': (
                    first_calibration['nominal_extrinsic'].cpu().numpy().tolist()),
                'Ti_true_extrinsic': (
                    first_calibration['true_extrinsic'].cpu().numpy().tolist()),
                'delta_Ti': first_calibration[
                    'residual'].cpu().numpy().tolist(),
                'translation_m': first_record['iter0']['t_cm'] / 100.0,
                'rotation_deg': first_record['iter0']['r_deg'],
            }
        elif sampled_kitti_mode:
            init_perturbation = {
                'mode': 'cmrnext_original_online_uniform',
                'training_semantics': 'cmrnext_original_online_perturbation_v1',
                'residual_convention': 'delta_Ti = Ti @ inverse(T0)',
                'random_perturbation_applied': True,
                'target_cam': str(_config['cam']),
                'seed': int(seed),
                'worker_seed_formula': 'seed + worker_id + 100',
                'max_r_deg': float(_config['max_r']),
                'max_t_m': float(_config['max_t']),
                'rotation_internal_component_ranges_deg': [
                    [-float(_config['max_r']), float(_config['max_r'])],
                    [-float(_config['max_r']), float(_config['max_r'])],
                    [-float(_config['max_r']), float(_config['max_r'])],
                ],
                'translation_internal_component_ranges_m': [
                    [-float(_config['max_t']), float(_config['max_t'])],
                    [-float(_config['max_t']), float(_config['max_t'])],
                    [-float(_config['max_t']), min(float(_config['max_t']), 1.0)],
                ],
                'kitti_axis_remapping': {
                    'euler_xyz': ['sampled_roty', 'sampled_rotz', 'sampled_rotx'],
                    'translation_xyz': [
                        'sampled_transl_y', 'sampled_transl_z',
                        'sampled_transl_x',
                    ],
                },
            }
        else:
            init_perturbation = {
                'mode': 'absolute',
                'max_r_deg': _config['max_r'],
                'max_t_m': _config['max_t'],
            }
        complete_batch_indices = {record['batch_idx'] for record in complete}
        failed_frames = [
            record for record in frame_records
            if record['batch_idx'] not in complete_batch_indices]
        with open(errors_json_path, 'w') as f:
            json.dump({
                'dataset': _config['dataset'],
                'emd_level': (_config['emd_level']
                              if _config['dataset'] == 'emd' else None),
                'evaluation_img_shape': (
                    [int(img_shape[0]), int(img_shape[1])]
                    if _config['dataset'] == 'emd' else None),
                'emd_resize_shape': (
                    list(_config['emd_resize_shape'])
                    if _config['dataset'] == 'emd' else None),
                'emd_evaluation_crop': (
                    _config.get('emd_evaluation_crop')
                    if _config['dataset'] == 'emd' else None),
                'flow_epe_units': (
                    'resized_pixels'
                    if _config['dataset'] == 'emd'
                    and checkpoint.get('config', {}).get(
                        'emd_resize_shape') is not None
                    else 'pixels'),
                'emd_crop_top_range': (
                    list(_config['emd_crop_top_range'])
                    if _config['dataset'] == 'emd'
                    and _config.get('emd_crop_top_range') is not None else None),
                'emd_crop_left_range': (
                    list(_config['emd_crop_left_range'])
                    if _config['dataset'] == 'emd'
                    and _config.get('emd_crop_left_range') is not None else None),
                'crop_sampling': (
                    ('fixed_from_checkpoint'
                     if _config.get('emd_evaluation_crop') is not None
                     else 'seeded_random_per_frame_reused_across_cascade')
                    if _config['dataset'] == 'emd' else None),
                'seed': int(seed),
                'diagnostic_max_frames': max_frames,
                'n_frames_available': (
                    len(dataset_val) if _config['dataset'] == 'emd' else None),
                'n_frames_total': len(frame_records),
                'n_frames_complete': len(complete),
                'n_frames_failed': len(frame_records) - len(complete),
                'init_perturbation': init_perturbation,
                'weights': _config['weights'],
                'cascade_stages': cascade_stages,
                'checkpoint_training_domains': checkpoint_training_domains,
                'evaluation_data_domain': ({
                    'level': _config['emd_level'],
                    'fixed_t0_online': fixed_t0_online_mode,
                    'source_mode': emd_contract.source_mode,
                    'scene_paths': list(dataset_val.scene_paths),
                } if _config['dataset'] == 'emd' else None),
                'checkpoint_stage_mapping': [
                    {'stage': int(stage), 'checkpoint': checkpoint_path}
                    for stage, checkpoint_path in zip(
                        cascade_stages, _config['weights'])
                ],
                'aggregate': aggregate,
                'flow_epe': flow_epe_report,
                'emd_evaluation': emd_report,
                'kitti_fixed_nominal_evaluation': kitti_fixed_nominal_report,
                'kitti_sampled_perturbation_evaluation': (
                    kitti_sampled_perturbation_report),
                'failed_frames': failed_frames,
                'frames': complete,
            }, f, indent=2)
        print(f"[viz] errors.json 已写 ({len(complete)} 完整帧 / {len(frame_records)} 总帧)")

    # Reuse stored residuals for reprojection without rerunning the model or PnP.
    if (_config['viz'] and _config.get('viz_strategy', 'topk_iter3_t')
            in ('topk_iter3_t', 'all_split', 'per_scene_worst_joint')):
        strategy = _config['viz_strategy']
        n_iter = len(_config['weights'])
        complete = [r for r in frame_records if all(f'iter{i}' in r for i in range(n_iter + 1))]
        if not complete:
            print(f"[viz] 无完整帧, 跳过 {strategy} 可视化")
        else:
            last_iter_key = f'iter{n_iter}'

            # Each selection contains a subdirectory, tag, and frame record.
            picks = []
            selection_manifest = None
            if strategy == 'per_scene_worst_joint':
                selections = _select_emd_worst_per_scene(
                    complete, last_iter_key,
                    _config['viz_per_scene_worst_k'])
                for item in selections:
                    record = item['record']
                    scene = item['scene'] or 'unknown_scene'
                    picks.append((
                        scene,
                        f"rank_{item['rank']:03d}",
                        record,
                    ))
                selection_manifest = {
                    'schema_version': 'cmrnext_emd_per_scene_worst_v1',
                    'dataset': _config['dataset'],
                    'emd_level': _config['emd_level'],
                    'scene_paths': list(dataset_val.scene_paths),
                    'training_semantics': 'nominal_to_true_residual_v1',
                    'initial_extrinsic': 'stored T0_nominal_extrinsic',
                    'seed': int(seed),
                    'checkpoint_stage_mapping': [
                        {'stage': int(stage), 'checkpoint': checkpoint_path}
                        for stage, checkpoint_path in zip(
                            cascade_stages, _config['weights'])
                    ],
                    'final_iteration': last_iter_key,
                    'ranking_metric': (
                        'sqrt((translation_m/effective_max_t_m)^2 + '
                        '(rotation_deg/effective_max_r_deg)^2)'),
                    'per_scene_requested': _config['viz_per_scene_worst_k'],
                    'scene_count': len({item['scene'] for item in selections}),
                    'selected_count': len(selections),
                    'frames': [],
                }
                for item in selections:
                    record = item['record']
                    initial = record['iter0']
                    final = record[last_iter_key]
                    selection_manifest['frames'].append({
                        'scene': item['scene'],
                        'rank': item['rank'],
                        'normalized_joint_residual': item['score'],
                        'batch_idx': record['batch_idx'],
                        'frame_stem': record['frame_stem'],
                        'rgb_path': record['rgb_path'],
                        'initial': {
                            'translation_m': initial['t_cm'] / 100.0,
                            'rotation_deg': initial['r_deg'],
                            'residual_components': initial['residual_components'],
                            'T_residual': initial['T_residual'],
                            'T_estimated_cam2vel': initial['T_estimated_cam2vel'],
                        },
                        'final': {
                            'translation_m': final['t_cm'] / 100.0,
                            'rotation_deg': final['r_deg'],
                            'residual_components': final['residual_components'],
                            'T_residual': final['T_residual'],
                            'T_estimated_cam2vel': final['T_estimated_cam2vel'],
                        },
                        'T_gt_cam2vel': record['T_gt_cam2vel'],
                        'T0_nominal_extrinsic': record['T0_nominal_extrinsic'],
                        'effective_max_t_m': record['effective_max_t_m'],
                        'effective_max_r_deg': record['effective_max_r_deg'],
                    })
                print(
                    f"[viz] per-scene worst: {len(picks)} frames from "
                    f"{selection_manifest['scene_count']} scenes")
            elif strategy == 'topk_iter3_t':
                ranked = sorted(complete, key=lambda r: r[last_iter_key]['t_cm'], reverse=True)
                n_worst = min(_config.get('viz_top_worst', 20), len(ranked))
                n_best = min(_config.get('viz_top_best', 5), len(ranked))
                for i, r in enumerate(ranked[:n_worst]):
                    picks.append(('', f'worst_rank{i + 1:03d}', r))
                for i, r in enumerate(ranked[-n_best:][::-1]):  # Smallest error is best_rank001.
                    picks.append(('', f'best_rank{i + 1:03d}', r))
                print(f"[viz] topk: {len(picks)} 帧 ({n_worst} worst + {n_best} best, 按 {last_iter_key} t_cm 排序)")
            else:  # all_split uses a median split.
                t_cms = sorted(r[last_iter_key]['t_cm'] for r in complete)
                median_t = t_cms[len(t_cms) // 2]
                os.makedirs(os.path.join(viz_dir, 'good'), exist_ok=True)
                os.makedirs(os.path.join(viz_dir, 'bad'), exist_ok=True)
                for r in complete:
                    sub = 'good' if r[last_iter_key]['t_cm'] <= median_t else 'bad'
                    picks.append((sub, f"b{r['batch_idx']:05d}", r))
                n_good = sum(1 for s, _, _ in picks if s == 'good')
                print(f"[viz] all_split: {len(picks)} 帧 (good {n_good} / bad {len(picks) - n_good}, "
                      f"以 {last_iter_key} t_cm 中位数 {median_t:.2f}cm 二分)")

            # Group selections by batch index; best and worst sets may overlap.
            meta_for_batch = {}
            for sd, tg, rc in picks:
                meta_for_batch.setdefault(rc['batch_idx'], []).append((sd, tg, rc))
            target_batch_indices = sorted(meta_for_batch.keys())

            # The second pass loads only selected frames.
            inner_ds = dataset_val.dataset if isinstance(dataset_val, torch.utils.data.Subset) else dataset_val
            inner_indices = dataset_val.indices if isinstance(dataset_val, torch.utils.data.Subset) else list(range(len(dataset_val)))
            sub_indices = [inner_indices[b] for b in target_batch_indices]
            picked_subset = torch.utils.data.Subset(inner_ds, sub_indices)
            VizLoader = torch.utils.data.DataLoader(dataset=picked_subset, shuffle=False, batch_size=1,
                                                    num_workers=0, worker_init_fn=init_fn,
                                                    collate_fn=merge_inputs, drop_last=False, pin_memory=False)

            def _pct(num, den, eps=1e-9):
                """完成 CMRNext 数据评估、光流到 PnP、级联外参细化和报告生成 中的 `_pct` 操作。

                参数:
                    num: 百分比计算的分子。
                    den: 百分比计算的分母。
                    eps: 防止除零使用的最小正数。

                返回值:
                    返回由 `实现中的返回表达式` 表示的计算结果。

                异常和副作用:
                    输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

                坐标系和数据约定:
                    EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
                """
                return 100.0 * num / max(den, eps)

            rendered_files = {}
            for new_bidx, sample in enumerate(tqdm(VizLoader, desc=f'[viz] {strategy}')):
                orig_batch_idx = target_batch_indices[new_bidx]

                # Prepare shared frame inputs once.
                idx = 0
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]
                sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
                if _config['max_depth'] < 100.:
                    sample['point_cloud'][idx] = sample['point_cloud'][idx][:,
                                                 sample['point_cloud'][idx][0, :] < _config['max_depth']]
                pc_orig_full = sample['point_cloud'][idx].clone()
                reflectance = sample['reflectance'][idx].cuda() if _config['use_reflectance'] else None

                cam_params = sample['calib'][idx].cuda()
                # per_frame_jet_overlay handles colors without ImageNet normalization.
                rgb01 = (sample['rgb'][idx].cuda() / 255.).permute(2, 0, 1)

                # Ground-truth subtitle denominator.
                T_gt = sample['cam2vel'][idx].float()
                t_gt_norm_m = float(torch.linalg.norm(T_gt[:3, 3]).item())
                r_gt_deg = _rot_angle_deg(T_gt)
                gt_extr_str = _fmt_extrinsic(T_gt)

                # Project the original cloud with the true extrinsic.
                depth_img_gt, _, _, _ = prepare_input(
                    cam_params, pc_orig_full.clone(), real_shape, reflectance, _config)

                # Project each stage from its stored residual.
                iter_panels = []  # Stage title and depth image only.
                panel_iterations = (
                    [0, n_iter]
                    if strategy == 'per_scene_worst_joint'
                    else list(range(n_iter + 1)))
                for it_id in panel_iterations:
                    # Records in one batch differ only by tag and subdirectory.
                    rec = meta_for_batch[orig_batch_idx][0][2]
                    iter_rec = rec.get(f'iter{it_id}')
                    if iter_rec is None or iter_rec.get('T_residual') is None:
                        continue
                    T_resid = torch.as_tensor(iter_rec['T_residual'], dtype=torch.float).to(pc_orig_full.device)
                    T_est_raw = iter_rec.get('T_estimated_cam2vel')
                    T_est = (torch.as_tensor(T_est_raw, dtype=torch.float)
                             if T_est_raw is not None else torch.eye(4))

                    pc_rot = rotate_forward(pc_orig_full.clone(), T_resid)
                    depth_n, _, _, _ = prepare_input(cam_params, pc_rot, real_shape, reflectance, _config)

                    t_cm = float(iter_rec['t_cm'])
                    r_deg = float(iter_rec['r_deg'])
                    t_pct = _pct(t_cm, t_gt_norm_m * 100.0)
                    r_pct = _pct(r_deg, r_gt_deg)
                    est_str = _fmt_extrinsic(T_est)
                    if strategy == 'per_scene_worst_joint':
                        label = ('T0 initial projection' if it_id == 0
                                 else 'Stage-1 inferred projection')
                        title = f"{label} | {_residual_annotation(iter_rec)}"
                    else:
                        label = 'Initial (perturbed)' if it_id == 0 else f'After iter{it_id}'
                        title = (f"{label} | est cam2vel  {est_str}\n"
                                 f"abs err: |t_est-t_gt|={t_cm:.2f}cm ({t_pct:.1f}% of |t_gt|), "
                                 f"r={r_deg:.3f}° ({r_pct:.2f}% of θ_gt)")
                    iter_panels.append((title, depth_n))

                gt_title = (
                    f"scene={rec.get('sequence', '?')} | "
                    f"frame={rec.get('frame_stem', '?')} | "
                    "Ti true-aligned projection\nresidual to Ti: "
                    "delta t xyz=[+0.0000, +0.0000, +0.0000] m | "
                    "delta r xyz=[+0.000, +0.000, +0.000] deg"
                    if strategy == 'per_scene_worst_joint'
                    else (f"GT (true extrinsic) | cam2vel  {gt_extr_str}\n"
                          f"|t_gt|={t_gt_norm_m * 100:.2f}cm, θ_gt={r_gt_deg:.2f}°  (分母 baseline)"))
                projections = [(gt_title, depth_img_gt)] + iter_panels

                fig = _draw_reprojection_panels(
                    _plt_save, projections, rgb01,
                    overlay_text=(strategy == 'per_scene_worst_joint'))

                # Save one result per selection when sets overlap.
                for subdir, tag, rec in meta_for_batch[orig_batch_idx]:
                    fname = f"{tag}_b{orig_batch_idx:05d}_{rec.get('sequence', '')}_{rec.get('frame_stem', '')}.png"
                    fname = fname.replace('/', '_')
                    if strategy != 'per_scene_worst_joint':
                        fig.suptitle(
                            f"{_config['dataset']} | seq={rec.get('sequence', '?')} "
                            f"frame={rec.get('frame_stem', '?')} | tag={tag}",
                            fontsize=10)
                    save_dir = os.path.join(viz_dir, subdir) if subdir else viz_dir
                    os.makedirs(save_dir, exist_ok=True)
                    fig.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches='tight')
                    rendered_files[(str(rec.get('sequence', '')), int(rec['batch_idx']))] = (
                        os.path.relpath(os.path.join(save_dir, fname), viz_dir))
                _plt_save.close(fig)
            if selection_manifest is not None:
                for frame in selection_manifest['frames']:
                    frame['triptych_png'] = rendered_files[
                        (frame['scene'], frame['batch_idx'])]
                manifest_path = os.path.join(viz_dir, 'selection_manifest.json')
                with open(manifest_path, 'w', encoding='utf-8') as handle:
                    json.dump(selection_manifest, handle, indent=2)
                csv_path = os.path.join(viz_dir, 'selection_manifest.csv')
                axes = ('x', 'y', 'z')
                fields = [
                    'scene', 'rank', 'batch_idx', 'frame_stem',
                    'normalized_joint_residual', 'initial_translation_m',
                    'initial_rotation_deg', 'final_translation_m',
                    'final_rotation_deg',
                    *[f'initial_t_{axis}_m' for axis in axes],
                    *[f'initial_r_{axis}_deg' for axis in axes],
                    *[f'final_t_{axis}_m' for axis in axes],
                    *[f'final_r_{axis}_deg' for axis in axes],
                    'triptych_png',
                ]
                with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for frame in selection_manifest['frames']:
                        row = {
                            'scene': frame['scene'], 'rank': frame['rank'],
                            'batch_idx': frame['batch_idx'],
                            'frame_stem': frame['frame_stem'],
                            'normalized_joint_residual': frame['normalized_joint_residual'],
                            'initial_translation_m': frame['initial']['translation_m'],
                            'initial_rotation_deg': frame['initial']['rotation_deg'],
                            'final_translation_m': frame['final']['translation_m'],
                            'final_rotation_deg': frame['final']['rotation_deg'],
                            'triptych_png': frame['triptych_png'],
                        }
                        for prefix in ('initial', 'final'):
                            components = frame[prefix]['residual_components']
                            for index, axis in enumerate(axes):
                                row[f'{prefix}_t_{axis}_m'] = components['translation_xyz_m'][index]
                                row[f'{prefix}_r_{axis}_deg'] = components['rotation_euler_xyz_deg'][index]
                        writer.writerow(row)
                print(f"[viz] selection manifests: {manifest_path}, {csv_path}")
            print(f"[viz] {strategy}: 共写出 {len(picks)} 张到 {viz_dir}")


def _geometry_numeric_summary(values):
    """计算几何指标序列的计数、均值、标准差和分位数。

    参数:
        values: 待统计的一维有限数值序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    array = np.asarray([
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ], dtype=np.float64)
    if not array.size:
        return {
            key: None for key in (
                'count', 'mean', 'std', 'min', 'p05', 'p10', 'p20',
                'median', 'p80', 'p90', 'p95', 'p99', 'max')
        }
    return {
        'count': int(array.size),
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'min': float(np.min(array)),
        'p05': float(np.quantile(array, 0.05)),
        'p10': float(np.quantile(array, 0.10)),
        'p20': float(np.quantile(array, 0.20)),
        'median': float(np.median(array)),
        'p80': float(np.quantile(array, 0.80)),
        'p90': float(np.quantile(array, 0.90)),
        'p95': float(np.quantile(array, 0.95)),
        'p99': float(np.quantile(array, 0.99)),
        'max': float(np.max(array)),
    }


def _geometry_values(rows, key):
    """从几何分析记录中提取指定字段的有限浮点数值。

    参数:
        rows: 待统计或写出的记录字典序列。
        key: 待读取的配置、指标或字典字段名。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    return np.asarray([
        float(row[key]) for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ], dtype=np.float64)


def _geometry_compare(selected_rows, reference_rows, key):
    """比较选中帧与参考帧在指定几何指标上的分布和标准化差异。

    参数:
        selected_rows: 困难帧或目标组的几何指标记录。
        reference_rows: 用于比较的参考帧几何指标记录。
        key: 待读取的配置、指标或字典字段名。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    selected = _geometry_values(selected_rows, key)
    reference = _geometry_values(reference_rows, key)
    result = {
        'selected': _geometry_numeric_summary(selected),
        'reference': _geometry_numeric_summary(reference),
    }
    if not selected.size or not reference.size:
        return result
    reference_sorted = np.sort(reference)
    selected_sorted = np.sort(selected)
    combined = np.sort(np.concatenate((reference_sorted, selected_sorted)))
    reference_cdf = np.searchsorted(
        reference_sorted, combined, side='right') / reference_sorted.size
    selected_cdf = np.searchsorted(
        selected_sorted, combined, side='right') / selected_sorted.size
    reference_p05, reference_p95 = np.quantile(reference, [0.05, 0.95])
    reference_std = float(np.std(reference))
    result.update({
        'selected_median_reference_percentile': float(
            np.mean(reference <= np.median(selected))),
        'selected_fraction_outside_reference_p05_p95': float(np.mean(
            (selected < reference_p05) | (selected > reference_p95))),
        'ks_distance': float(np.max(np.abs(reference_cdf - selected_cdf))),
        'standardized_mean_difference': (
            float((np.mean(selected) - np.mean(reference)) / reference_std)
            if reference_std > 1e-12 else None
        ),
        'median_ratio': (
            float(np.median(selected) / np.median(reference))
            if abs(np.median(reference)) > 1e-12 else None
        ),
    })
    return result


def _geometry_average_ranks(values):
    """对并列数值分配平均秩，供无 SciPy 的秩相关计算使用。

    参数:
        values: 待统计的一维有限数值序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _geometry_spearman(rows, left_key, right_key):
    """用平均秩计算两项几何指标的 Spearman 相关系数。

    参数:
        rows: 待统计或写出的记录字典序列。
        left_key: 秩相关分析左侧指标字段名。
        right_key: 秩相关分析右侧指标字段名。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    pairs = [
        (float(row[left_key]), float(row[right_key]))
        for row in rows
        if row.get(left_key) is not None
        and row.get(right_key) is not None
        and np.isfinite(float(row[left_key]))
        and np.isfinite(float(row[right_key]))
    ]
    if len(pairs) < 3:
        return {'count': len(pairs), 'rho': None}
    values = np.asarray(pairs, dtype=np.float64)
    left = _geometry_average_ranks(values[:, 0])
    right = _geometry_average_ranks(values[:, 1])
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return {'count': len(pairs), 'rho': None}
    return {
        'count': len(pairs),
        'rho': float(np.corrcoef(left, right)[0, 1]),
    }


def _geometry_metadata_row(sample, split, sequence):
    """从数据样本整理场景、帧号、划分和外参尺度等元数据。

    参数:
        sample: DataLoader 返回的批次样本字典。
        split: 仅用于记录来源的数据角色文本，不参与显式场景选择。
        sequence: 当前数据序列编号或名称。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    conditioning = sample['view_rotation_conditioning']
    gravity = np.asarray(
        conditioning['gravity_in_camera'], dtype=np.float64)
    gravity /= np.linalg.norm(gravity)
    level_tilt = float(conditioning['camera_level_tilt_deg'])
    horizon_roll = float(conditioning['camera_horizon_roll_deg'])
    quality = sample['generated_quality']
    camera_depth = sample['generated_sensor']['camera_depth']
    covariance = np.asarray(
        quality.get('depth_covariance_eigenvalues', []), dtype=np.float64)
    return {
        'split': split,
        'scene': sequence,
        'frame_stem': str(sample['frame_id']),
        'source_frame': str(sample['source_frame']),
        'target_frame': str(sample['target_frame']),
        'frame_gap': abs(
            int(sample['target_frame']) - int(sample['source_frame'])),
        'depth_scale': float(sample['depth_scale']),
        'camera_level_tilt_deg': level_tilt,
        'camera_unsigned_level_deviation_deg': min(
            level_tilt, 180.0 - level_tilt),
        'camera_inverted': float(level_tilt > 90.0),
        'camera_horizon_roll_deg': horizon_roll,
        'camera_abs_horizon_roll_deg': abs(horizon_roll),
        'gravity_camera_x': float(gravity[0]),
        'gravity_camera_y': float(gravity[1]),
        'gravity_camera_z': float(gravity[2]),
        'gravity_abs_optical_axis_pitch_deg': abs(float(np.rad2deg(
            np.arctan2(gravity[2], np.hypot(gravity[0], gravity[1]))
        ))),
        'dense_camera_min_depth_m': float(
            camera_depth['minimum_depth_m']),
        'dense_camera_max_depth_m': float(
            camera_depth['maximum_depth_m']),
        'raw_point_count': int(quality['point_count']),
        'target_visible_ratio': float(quality['target_visible_ratio']),
        'gt_projected_point_count_metadata': int(
            quality['true_projected_points']),
        'gt_zbuffer_point_count_metadata': int(
            quality['true_zbuffer_points']),
        'gt_zbuffer_point_ratio_metadata': float(
            quality['true_zbuffer_point_ratio']),
        'gt_zbuffer_retention_ratio_metadata': float(
            quality['true_zbuffer_retention_ratio']),
        'gt_dense_tile_ratio_metadata': float(
            quality['true_dense_tile_ratio']),
        'gt_tile_ratio_metadata': float(quality['true_tile_ratio']),
        'depth_edge_pixel_ratio': float(
            quality['depth_edge_pixel_ratio']),
        'depth_edge_tile_ratio': float(quality['depth_edge_tile_ratio']),
        'long_depth_edge_pixel_ratio': float(
            quality['long_depth_edge_pixel_ratio']),
        'long_depth_edge_tile_ratio': float(
            quality['long_depth_edge_tile_ratio']),
        'depth_nonplanarity': float(quality['depth_nonplanarity']),
        'depth_covariance_min_m2': (
            float(np.min(covariance)) if covariance.size else None),
        'depth_covariance_max_m2': (
            float(np.max(covariance)) if covariance.size else None),
    }


def _geometry_euler_xyz_matrix(euler_xyz_deg):
    """把 xyz 欧拉角度数组转换为四乘四齐次旋转矩阵。

    参数:
        euler_xyz_deg: 角度量或角度上限，单位为度。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    x, y, z = np.deg2rad(np.asarray(euler_xyz_deg, dtype=np.float64))
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    return np.asarray([
        [cz * cy, cz * sy * sx - sz * cx,
         cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx,
         sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ], dtype=np.float64)


def _geometry_projection(points_h, transform, camera_matrix, width, height):
    """用给定变换投影齐次点云并返回有效像素、深度和点索引。

    参数:
        points_h: 齐次点云，形状为 N×4，长度单位为米。
        transform: 四乘四齐次刚体变换矩阵，平移单位为米。
        camera_matrix: 三乘三针孔相机内参矩阵。
        width: 图像宽度，单位为像素。
        height: 图像高度，单位为像素。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    camera_points = (
        np.asarray(transform, dtype=np.float64)
        @ np.asarray(points_h, dtype=np.float64).T
    ).T
    depth = camera_points[:, 2]
    positive = depth >= 0.0
    uv = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    usable = positive & (depth > 1e-12)
    uv[usable, 0] = (
        camera_matrix[0, 0] * camera_points[usable, 0] / depth[usable]
        + camera_matrix[0, 2]
    )
    uv[usable, 1] = (
        camera_matrix[1, 1] * camera_points[usable, 1] / depth[usable]
        + camera_matrix[1, 2]
    )
    in_image = (
        usable
        & (uv[:, 0] >= 0.1) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.1) & (uv[:, 1] < height)
    )
    owner_mask = np.zeros(len(camera_points), dtype=bool)
    candidates = np.flatnonzero(in_image)
    if candidates.size:
        pixel_x = np.trunc(uv[candidates, 0]).astype(np.int64)
        pixel_y = np.trunc(uv[candidates, 1]).astype(np.int64)
        flat = pixel_y * width + pixel_x
        depth_order = candidates[np.argsort(
            depth[candidates], kind='mergesort')]
        ordered_flat = (
            np.trunc(uv[depth_order, 1]).astype(np.int64) * width
            + np.trunc(uv[depth_order, 0]).astype(np.int64)
        )
        _, first = np.unique(ordered_flat, return_index=True)
        owner_mask[depth_order[first]] = True
    return {
        'uv': uv,
        'depth': depth,
        'positive': positive,
        'in_image': in_image,
        'owner_mask': owner_mask,
    }


def _geometry_tile_coverage(projection, width, height):
    """统计投影像素覆盖固定图像网格单元的比例。

    参数:
        projection: 有效点云在图像中的投影结果，至少包含 u、v 像素坐标。
        width: 图像宽度，单位为像素。
        height: 图像高度，单位为像素。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    owners = np.flatnonzero(projection['owner_mask'])
    if not owners.size:
        return 0.0
    uv = projection['uv'][owners]
    tile_x = np.minimum(
        7, np.floor(uv[:, 0] * 8.0 / width).astype(np.int64))
    tile_y = np.minimum(
        3, np.floor(uv[:, 1] * 4.0 / height).astype(np.int64))
    return float(np.unique(tile_y * 8 + tile_x).size / 32.0)


def _geometry_add_quantiles(row, prefix, values):
    """把数值序列的声明分位数以指定前缀写入报告行。

    参数:
        row: 正在补充字段的单条报告记录字典。
        prefix: 写入报告字段名时使用的稳定前缀。
        values: 待统计的一维有限数值序列。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        for suffix in (
                'min', 'p02', 'p10', 'median', 'p80', 'p90',
                'p95', 'p98', 'max', 'mean'):
            row[f'{prefix}_{suffix}'] = None
        return
    quantiles = np.quantile(
        values, [0.02, 0.10, 0.50, 0.80, 0.90, 0.95, 0.98])
    row.update({
        f'{prefix}_min': float(np.min(values)),
        f'{prefix}_p02': float(quantiles[0]),
        f'{prefix}_p10': float(quantiles[1]),
        f'{prefix}_median': float(quantiles[2]),
        f'{prefix}_p80': float(quantiles[3]),
        f'{prefix}_p90': float(quantiles[4]),
        f'{prefix}_p95': float(quantiles[5]),
        f'{prefix}_p98': float(quantiles[6]),
        f'{prefix}_max': float(np.max(values)),
        f'{prefix}_mean': float(np.mean(values)),
    })


def _geometry_frame_metrics(
        point_path, true_extrinsic, transforms, camera_matrix,
        width, height, metadata_row):
    """计算一帧在多个候选外参下的可见点、深度、覆盖率和位移指标。

    参数:
        point_path: 当前帧 Nx4 float32 LiDAR 点云文件路径。
        true_extrinsic: 逐帧真实 LiDAR 到相机外参 Ti，形状为 4×4。
        transforms: 名称到候选四乘四外参矩阵的映射。
        camera_matrix: 三乘三针孔相机内参矩阵。
        width: 图像宽度，单位为像素。
        height: 图像高度，单位为像素。
        metadata_row: 当前帧场景、序列和尺度元数据字典。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    raw = np.fromfile(str(point_path), dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f'{point_path}：应为非空 Nx4 float32 点云')
    raw = raw.reshape(-1, 4).astype(np.float64)
    raw[:, 3] = 1.0
    row = dict(metadata_row)
    row['point_path'] = str(point_path)
    row['point_count_file'] = int(len(raw))
    radial = np.linalg.norm(raw[:, :3], axis=1)
    _geometry_add_quantiles(row, 'lidar_range_m', radial)

    gt_projection = _geometry_projection(
        raw, true_extrinsic, camera_matrix, width, height)
    gt_owners = gt_projection['owner_mask']
    row.update({
        'gt_positive_depth_point_count': int(np.count_nonzero(
            gt_projection['positive'])),
        'gt_in_image_point_count': int(np.count_nonzero(
            gt_projection['in_image'])),
        'gt_zbuffer_point_count': int(np.count_nonzero(gt_owners)),
        'gt_tile_coverage_8x4': _geometry_tile_coverage(
            gt_projection, width, height),
    })
    _geometry_add_quantiles(
        row, 'gt_positive_depth_m',
        gt_projection['depth'][gt_projection['positive']])
    _geometry_add_quantiles(
        row, 'gt_in_image_depth_m',
        gt_projection['depth'][gt_projection['in_image']])
    _geometry_add_quantiles(
        row, 'gt_zbuffer_depth_m',
        gt_projection['depth'][gt_owners])

    for label, transform in transforms.items():
        projection = _geometry_projection(
            raw, transform, camera_matrix, width, height)
        owners = projection['owner_mask']
        row.update({
            f'{label}_in_image_point_count': int(np.count_nonzero(
                projection['in_image'])),
            f'{label}_zbuffer_point_count': int(np.count_nonzero(owners)),
            f'{label}_tile_coverage_8x4': _geometry_tile_coverage(
                projection, width, height),
        })
        _geometry_add_quantiles(
            row, f'{label}_zbuffer_depth_m',
            projection['depth'][owners])

        flow_mask = owners & gt_projection['in_image']
        raw_flow = (
            gt_projection['uv'][flow_mask] - projection['uv'][flow_mask])
        suppressed_flow = raw_flow.copy()
        suppressed_flow[np.abs(suppressed_flow) < 0.5] = 0.0
        raw_disparity = np.linalg.norm(raw_flow, axis=1)
        disparity = np.linalg.norm(suppressed_flow, axis=1)
        row[f'{label}_gt_flow_correspondence_count'] = int(
            np.count_nonzero(flow_mask))
        _geometry_add_quantiles(
            row, f'{label}_gt_flow_disparity_px', disparity)
        _geometry_add_quantiles(
            row, f'{label}_gt_flow_raw_disparity_px', raw_disparity)

        common_zbuffer = owners & gt_owners
        common_disparity = np.linalg.norm(
            gt_projection['uv'][common_zbuffer]
            - projection['uv'][common_zbuffer],
            axis=1,
        )
        row[f'{label}_gt_common_zbuffer_count'] = int(
            np.count_nonzero(common_zbuffer))
        _geometry_add_quantiles(
            row, f'{label}_gt_common_zbuffer_disparity_px',
            common_disparity)
    return row


def _geometry_write_csv(path, rows):
    """按所有记录字段的稳定并集顺序写出几何分析 CSV。

    参数:
        path: 输入或输出文件路径。
        rows: 待统计或写出的记录字典序列。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    fields = sorted({key for row in rows for key in row})
    preferred = [
        'group', 'split', 'scene', 'frame_stem', 'rank',
        'batch_idx', 'stage3_translation_error_cm',
        'stage3_rotation_error_deg', 'point_path',
    ]
    fields = [field for field in preferred if field in fields] + [
        field for field in fields if field not in preferred]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _geometry_group_summaries(groups, keys):
    """按数据分组汇总指定几何指标的数值分布。

    参数:
        groups: 组名到几何指标记录序列的映射。
        keys: 需要汇总或绘图的指标字段名序列。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    return {
        group: {
            key: _geometry_numeric_summary(
                row.get(key) for row in rows)
            for key in keys
        }
        for group, rows in groups.items()
    }


def _geometry_plot_distributions(output_path, groups, metrics):
    """绘制不同数据组的几何指标分布对比图。

    参数:
        output_path: 目标报告、CSV 或图表文件路径。
        groups: 组名到几何指标记录序列的映射。
        metrics: 需要绘制或归档的几何指标字段序列。

    返回值:
        无显式返回值；函数完成所述状态更新、校验或输出操作。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    colors = {
        'train_reference': '#3973ac',
        'test_reference': '#3a944f',
        'bad_selected': '#c44335',
    }
    for axis, (key, title) in zip(axes.flat, metrics):
        for group in ('train_reference', 'test_reference', 'bad_selected'):
            values = _geometry_values(groups[group], key)
            if not values.size:
                continue
            axis.hist(
                values, bins=35, density=True, histtype='step',
                linewidth=1.8, color=colors[group], label=group)
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=140)
    plt.close(fig)


def evaluate_geometry_domain_gap(
        source_errors_json, selection_manifest, dataset_root, output_dir,
        train_reference_per_scene=100, test_reference_per_scene=200,
        seed=42):
    """比较训练、测试和困难帧的几何输入分布并输出 JSON、CSV 和图表。

    参数:
        source_errors_json: 已有逐帧评估结果 JSON 的路径。
        selection_manifest: 明确记录诊断所选帧的清单文件路径。
        dataset_root: 待分析数据集的根目录路径。
        output_dir: 任务产物目录路径。
        train_reference_per_scene: 每个训练场景抽取的几何参考帧数量。
        test_reference_per_scene: 每个测试场景抽取的几何参考帧数量。
        seed: 显式随机种子，用于复现采样、裁剪或数据顺序。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    source_path = Path(source_errors_json).expanduser().resolve()
    selection_path = Path(selection_manifest).expanduser().resolve()
    dataset_path = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if train_reference_per_scene <= 0 or test_reference_per_scene <= 0:
        raise ValueError('几何参考样本数必须为正数')
    with source_path.open('r', encoding='utf-8') as handle:
        source = json.load(handle)
    with selection_path.open('r', encoding='utf-8') as handle:
        selection = json.load(handle)
    if source.get('dataset') != 'emd':
        raise ValueError('几何域差异分析要求 dataset=emd')
    selected_frames = selection.get('frames', [])
    if not selected_frames:
        raise ValueError('Selection manifest contains no frames')

    source_by_key = {
        (str(frame['sequence']), str(frame['frame_stem'])): frame
        for frame in source.get('frames', [])
        if all(f'iter{index}' in frame for index in range(4))
    }
    selected_by_key = {
        (str(frame['scene']), str(frame['frame_stem'])): frame
        for frame in selected_frames
    }
    if len(selected_by_key) != len(selected_frames):
        raise ValueError('Selection manifest contains duplicate frames')

    level_root = dataset_path / 'fixed_baseline'
    metadata_rows = {'train_all': [], 'test_all': [], 'bad_selected': []}
    test_metadata = {}
    train_reference_specs = []
    train_dirs = sorted(
        path for path in (level_root / 'train').iterdir()
        if path.is_dir())
    test_dirs = sorted(
        path for path in (level_root / 'test').iterdir()
        if path.is_dir())

    for split, directories in (('train', train_dirs), ('test', test_dirs)):
        for scene_index, directory in enumerate(directories):
            with (directory / 'metadata.json').open(
                    'r', encoding='utf-8') as handle:
                metadata = json.load(handle)
            samples = metadata.get('samples', [])
            rows = [
                _geometry_metadata_row(sample, split, directory.name)
                for sample in samples
            ]
            metadata_rows[
                'train_all' if split == 'train' else 'test_all'
            ].extend(rows)
            if split == 'test':
                for sample, row in zip(samples, rows):
                    key = (directory.name, str(sample['frame_id']))
                    test_metadata[key] = (sample, row, directory)
                    if key in selected_by_key:
                        metadata_rows['bad_selected'].append(row)
            else:
                count = min(train_reference_per_scene, len(samples))
                rng = np.random.default_rng(np.random.SeedSequence([
                    int(seed), 0, int(scene_index),
                ]))
                indexes = np.sort(rng.choice(
                    len(samples), size=count, replace=False))
                perturbation_rng = np.random.default_rng(
                    np.random.SeedSequence([
                        int(seed), 0, int(scene_index), 1,
                    ]))
                for sample_index in indexes:
                    sample = samples[int(sample_index)]
                    row = rows[int(sample_index)]
                    translation = perturbation_rng.uniform(
                        -0.6, 0.6, size=3)
                    euler_deg = perturbation_rng.uniform(
                        -10.0, 10.0, size=3)
                    residual = np.eye(4, dtype=np.float64)
                    residual[:3, :3] = _geometry_euler_xyz_matrix(euler_deg)
                    residual[:3, 3] = translation
                    true_extrinsic = np.asarray(
                        sample['true_extrinsic'], dtype=np.float64)
                    input_extrinsic = (
                        np.linalg.inv(residual) @ true_extrinsic)
                    train_reference_specs.append({
                        'metadata': row,
                        'directory': directory,
                        'frame_stem': str(sample['frame_id']),
                        'true_extrinsic': true_extrinsic,
                        'input_extrinsic': input_extrinsic,
                        'sampled_translation_xyz_m': translation.tolist(),
                        'sampled_euler_xyz_deg': euler_deg.tolist(),
                    })

    if len(metadata_rows['bad_selected']) != len(selected_frames):
        raise ValueError(
            'Not every selected frame was found in test metadata')

    test_reference_specs = []
    complete_by_scene = {}
    for key, frame in source_by_key.items():
        if key not in selected_by_key:
            complete_by_scene.setdefault(key[0], []).append(frame)
    for scene_index, scene in enumerate(sorted(complete_by_scene)):
        candidates = sorted(
            complete_by_scene[scene],
            key=lambda frame: int(frame['batch_idx']))
        count = min(test_reference_per_scene, len(candidates))
        rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), 2, int(scene_index),
        ]))
        indexes = np.sort(rng.choice(
            len(candidates), size=count, replace=False))
        test_reference_specs.extend(candidates[int(index)] for index in indexes)

    first_directory = test_dirs[0]
    camera_matrix = _load_recorded_projection_matrix(first_directory)
    sensor = json.load(
        (first_directory / 'metadata.json').open('r', encoding='utf-8')
    )['sensor_setup']['camera']
    width, height = int(sensor['width']), int(sensor['height'])

    bad_rows = []
    for index, selected in enumerate(selected_frames, start=1):
        key = (str(selected['scene']), str(selected['frame_stem']))
        record = source_by_key[key]
        sample, metadata_row, directory = test_metadata[key]
        row = _geometry_frame_metrics(
            directory / 'velodyne' / f'{key[1]}.bin',
            np.asarray(record['T_gt_cam2vel'], dtype=np.float64),
            {
                'initial': np.asarray(
                    record['iter0']['T_estimated_cam2vel'],
                    dtype=np.float64),
                'iter1': np.asarray(
                    record['iter1']['T_estimated_cam2vel'],
                    dtype=np.float64),
                'iter2': np.asarray(
                    record['iter2']['T_estimated_cam2vel'],
                    dtype=np.float64),
                'iter3': np.asarray(
                    record['iter3']['T_estimated_cam2vel'],
                    dtype=np.float64),
            },
            camera_matrix, width, height, metadata_row)
        row.update({
            'group': 'bad_selected',
            'rank': int(selected['rank']),
            'batch_idx': int(record['batch_idx']),
            'stage3_translation_error_cm': float(record['iter3']['t_cm']),
            'stage3_rotation_error_deg': float(record['iter3']['r_deg']),
        })
        bad_rows.append(row)
        print(
            f'[geometry-domain-gap] bad {index}/{len(selected_frames)} '
            f'{key[0]}/{key[1]}')

    test_reference_rows = []
    for index, record in enumerate(test_reference_specs, start=1):
        key = (str(record['sequence']), str(record['frame_stem']))
        _, metadata_row, directory = test_metadata[key]
        row = _geometry_frame_metrics(
            directory / 'velodyne' / f'{key[1]}.bin',
            np.asarray(record['T_gt_cam2vel'], dtype=np.float64),
            {'initial': np.asarray(
                record['iter0']['T_estimated_cam2vel'],
                dtype=np.float64)},
            camera_matrix, width, height, metadata_row)
        row.update({
            'group': 'test_reference',
            'batch_idx': int(record['batch_idx']),
            'stage3_translation_error_cm': float(record['iter3']['t_cm']),
            'stage3_rotation_error_deg': float(record['iter3']['r_deg']),
        })
        test_reference_rows.append(row)
        if index % 100 == 0:
            print(
                f'[geometry-domain-gap] test reference '
                f'{index}/{len(test_reference_specs)}')

    train_reference_rows = []
    for index, spec in enumerate(train_reference_specs, start=1):
        row = _geometry_frame_metrics(
            spec['directory'] / 'velodyne'
            / f"{spec['frame_stem']}.bin",
            spec['true_extrinsic'],
            {'initial': spec['input_extrinsic']},
            camera_matrix, width, height, spec['metadata'])
        row.update({
            'group': 'train_reference',
            'sampled_translation_xyz_m': json.dumps(
                spec['sampled_translation_xyz_m']),
            'sampled_euler_xyz_deg': json.dumps(
                spec['sampled_euler_xyz_deg']),
        })
        train_reference_rows.append(row)
        if index % 100 == 0:
            print(
                f'[geometry-domain-gap] train reference '
                f'{index}/{len(train_reference_specs)}')

    error_metadata_rows = []
    for key, record in source_by_key.items():
        metadata = test_metadata.get(key)
        if metadata is None:
            continue
        row = dict(metadata[1])
        row.update({
            'stage3_translation_error_cm': float(record['iter3']['t_cm']),
            'stage3_rotation_error_deg': float(record['iter3']['r_deg']),
        })
        error_metadata_rows.append(row)

    metadata_keys = [
        'camera_level_tilt_deg',
        'camera_unsigned_level_deviation_deg',
        'camera_inverted',
        'camera_abs_horizon_roll_deg',
        'gravity_abs_optical_axis_pitch_deg',
        'depth_scale',
        'frame_gap',
        'dense_camera_min_depth_m',
        'dense_camera_max_depth_m',
        'gt_zbuffer_point_count_metadata',
        'gt_zbuffer_point_ratio_metadata',
        'gt_tile_ratio_metadata',
        'depth_edge_pixel_ratio',
        'depth_edge_tile_ratio',
        'long_depth_edge_pixel_ratio',
        'depth_nonplanarity',
    ]
    point_keys = [
        'lidar_range_m_min',
        'lidar_range_m_median',
        'lidar_range_m_p90',
        'lidar_range_m_p98',
        'lidar_range_m_max',
        'gt_zbuffer_depth_m_min',
        'gt_zbuffer_depth_m_p02',
        'gt_zbuffer_depth_m_median',
        'gt_zbuffer_depth_m_p90',
        'gt_zbuffer_depth_m_p98',
        'gt_zbuffer_depth_m_max',
        'gt_zbuffer_point_count',
        'gt_tile_coverage_8x4',
        'initial_zbuffer_point_count',
        'initial_tile_coverage_8x4',
        'initial_gt_flow_correspondence_count',
        'initial_gt_flow_disparity_px_mean',
        'initial_gt_flow_disparity_px_median',
        'initial_gt_flow_disparity_px_p80',
        'initial_gt_flow_disparity_px_p90',
        'initial_gt_flow_disparity_px_p95',
        'initial_gt_flow_disparity_px_max',
        'initial_gt_common_zbuffer_count',
        'initial_gt_common_zbuffer_disparity_px_median',
        'initial_gt_common_zbuffer_disparity_px_p90',
    ]
    groups = {
        'train_reference': train_reference_rows,
        'test_reference': test_reference_rows,
        'bad_selected': bad_rows,
    }
    comparisons = {
        'bad_vs_train_reference': {
            key: _geometry_compare(
                bad_rows, train_reference_rows, key)
            for key in metadata_keys + point_keys
        },
        'bad_vs_test_reference': {
            key: _geometry_compare(
                bad_rows, test_reference_rows, key)
            for key in metadata_keys + point_keys
        },
        'bad_vs_train_metadata_all': {
            key: _geometry_compare(
                bad_rows, metadata_rows['train_all'], key)
            for key in metadata_keys
        },
        'bad_vs_test_metadata_all': {
            key: _geometry_compare(
                bad_rows, metadata_rows['test_all'], key)
            for key in metadata_keys
        },
    }
    correlations = {
        metric: {
            'stage3_translation_error_cm': _geometry_spearman(
                error_metadata_rows, metric,
                'stage3_translation_error_cm'),
            'stage3_rotation_error_deg': _geometry_spearman(
                error_metadata_rows, metric,
                'stage3_rotation_error_deg'),
        }
        for metric in metadata_keys
    }
    per_scene_bad = {}
    for scene in sorted({row['scene'] for row in bad_rows}):
        scene_rows = [row for row in bad_rows if row['scene'] == scene]
        per_scene_bad[scene] = {
            key: _geometry_numeric_summary(
                row.get(key) for row in scene_rows)
            for key in metadata_keys + point_keys + [
                'stage3_translation_error_cm',
                'stage3_rotation_error_deg',
            ]
        }

    output_path.mkdir(parents=True, exist_ok=True)
    bad_csv = output_path / 'bad_frames_geometry.csv'
    train_csv = output_path / 'train_reference_geometry.csv'
    test_csv = output_path / 'test_reference_geometry.csv'
    _geometry_write_csv(bad_csv, bad_rows)
    _geometry_write_csv(train_csv, train_reference_rows)
    _geometry_write_csv(test_csv, test_reference_rows)

    summary = {
        'schema_version': 'cmrnext_geometry_domain_gap_v1',
        'source_errors_json': str(source_path),
        'source_errors_sha256': _sha256_file(source_path),
        'selection_manifest': str(selection_path),
        'selection_manifest_sha256': _sha256_file(selection_path),
        'dataset_root': str(dataset_path),
        'task_manifest_sha256': _sha256_file(
            dataset_path / 'task_manifest.json'),
        'seed': int(seed),
        'sample_counts': {
            'train_metadata_all': len(metadata_rows['train_all']),
            'test_metadata_all': len(metadata_rows['test_all']),
            'bad_selected': len(bad_rows),
            'train_point_reference': len(train_reference_rows),
            'test_point_reference': len(test_reference_rows),
        },
        'definitions': {
            'gravity': (
                'camera_level_tilt_deg is angle between camera up (-Y) '
                'and world gravity; 0=upright, 90=sideways, 180=upside-down'),
            'unsigned_level_deviation': (
                'min(level_tilt, 180-level_tilt); inversion is reported '
                'separately as level_tilt>90'),
            'depth': (
                'camera forward Z in meters; per-image median uses GT '
                'z-buffer owner points at original 1226x370 resolution'),
            'lidar_range': 'Euclidean norm in raw LiDAR coordinates',
            'disparity': (
                'pixel norm from initial source z-buffer projection to GT '
                'projection for the same point, with CMRNext component-wise '
                '<0.5px suppression; raw unsuppressed values are also in CSV'),
            'reference_sampling': (
                'fixed per-scene sampling with SeedSequence; train initial '
                'poses use deterministic draws from the exact stage-1 '
                'per-axis uniform box, not recorded training accesses'),
            'network_max_depth_m': 160.0,
            'network_depth_crop_behavior': (
                'evaluation code only pre-crops when max_depth<100, so no '
                'point pre-crop is applied at max_depth=160'),
        },
        'gravity_tilt_fractions': {
            group: {
                'level_le_15_deg': float(np.mean(
                    _geometry_values(rows, 'camera_level_tilt_deg') <= 15.0)),
                'level_gt_60_deg': float(np.mean(
                    _geometry_values(rows, 'camera_level_tilt_deg') > 60.0)),
                'inverted_gt_90_deg': float(np.mean(
                    _geometry_values(rows, 'camera_level_tilt_deg') > 90.0)),
                'level_gt_150_deg': float(np.mean(
                    _geometry_values(rows, 'camera_level_tilt_deg') > 150.0)),
            }
            for group, rows in metadata_rows.items()
        },
        'metadata_full_summaries': _geometry_group_summaries(
            metadata_rows, metadata_keys),
        'point_reference_summaries': _geometry_group_summaries(
            groups, metadata_keys + point_keys),
        'comparisons': comparisons,
        'full_test_spearman_correlations': correlations,
        'per_scene_bad_summaries': per_scene_bad,
    }
    summary_path = output_path / 'geometry_domain_gap_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    plot_path = output_path / 'geometry_domain_gap_distributions.png'
    _geometry_plot_distributions(
        plot_path, groups, [
            ('camera_level_tilt_deg', 'Camera level tilt (deg)'),
            ('camera_abs_horizon_roll_deg', 'Absolute horizon roll (deg)'),
            ('depth_scale', 'Physical depth scale'),
            ('lidar_range_m_median', 'LiDAR range median (m)'),
            ('gt_zbuffer_depth_m_median', 'GT z-buffer depth median (m)'),
            ('gt_zbuffer_depth_m_p98', 'GT z-buffer depth p98 (m)'),
            ('initial_gt_flow_disparity_px_median',
             'Initial-to-GT disparity median (px)'),
            ('initial_gt_flow_disparity_px_p90',
             'Initial-to-GT disparity p90 (px)'),
            ('gt_zbuffer_point_count', 'GT z-buffer point count'),
        ])

    def _median(group, key):
        """完成 CMRNext 数据评估、光流到 PnP、级联外参细化和报告生成 中的 `_median` 操作。

        参数:
            group: 当前参与统计的一组记录。
            key: 待读取的配置、指标或字典字段名。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

        坐标系和数据约定:
            EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
        """
        return summary['point_reference_summaries'][group][key]['median']

    report_lines = [
        '# Geometry domain-gap report',
        '',
        f'- Bad selected frames: {len(bad_rows)}',
        f'- Full train metadata frames: {len(metadata_rows["train_all"])}',
        f'- Full test metadata frames: {len(metadata_rows["test_all"])}',
        (
            f'- Point references: train={len(train_reference_rows)}, '
            f'test={len(test_reference_rows)}'
        ),
        '',
        '## Core medians',
        '',
        '| Metric | Train reference | Test reference | Bad selected |',
        '|---|---:|---:|---:|',
    ]
    core_metrics = [
        ('camera_level_tilt_deg', 'level tilt (deg)'),
        ('camera_abs_horizon_roll_deg', 'abs horizon roll (deg)'),
        ('depth_scale', 'physical depth scale'),
        ('lidar_range_m_median', 'per-frame LiDAR range median (m)'),
        ('gt_zbuffer_depth_m_median',
         'per-frame GT z-buffer depth median (m)'),
        ('gt_zbuffer_depth_m_p98',
         'per-frame GT z-buffer depth p98 (m)'),
        ('initial_gt_flow_disparity_px_median',
         'initial-to-GT disparity median (px)'),
        ('initial_gt_flow_disparity_px_p90',
         'initial-to-GT disparity p90 (px)'),
        ('gt_zbuffer_point_count', 'GT z-buffer point count'),
    ]
    for key, label in core_metrics:
        report_lines.append(
            f'| {label} | {_median("train_reference", key):.4f} | '
            f'{_median("test_reference", key):.4f} | '
            f'{_median("bad_selected", key):.4f} |')
    report_lines.extend([
        '', '## Full metadata medians', '',
        '| Metric | Full train | Full test | Bad selected |',
        '|---|---:|---:|---:|',
    ])
    for key, label in core_metrics[:3]:
        metadata = summary['metadata_full_summaries']
        report_lines.append(
            f'| {label} | {metadata["train_all"][key]["median"]:.4f} '
            f'| {metadata["test_all"][key]["median"]:.4f} '
            f'| {metadata["bad_selected"][key]["median"]:.4f} |')
    report_lines.extend([
        '', '## Gravity tilt fractions', '',
        '| Group | <=15 deg | >60 deg | >90 deg inverted | >150 deg |',
        '|---|---:|---:|---:|---:|',
    ])
    for group in ('train_all', 'test_all', 'bad_selected'):
        fractions = summary['gravity_tilt_fractions'][group]
        report_lines.append(
            f'| {group} | {fractions["level_le_15_deg"]:.2%} '
            f'| {fractions["level_gt_60_deg"]:.2%} '
            f'| {fractions["inverted_gt_90_deg"]:.2%} '
            f'| {fractions["level_gt_150_deg"]:.2%} |')
    report_lines.extend([
        '',
        '## Notes',
        '',
        (
            '- This dataset used target_translation_conditioned_uniform_roll '
            'with no gravity-level constraint.'
        ),
        (
            '- Disparity is the calibration-flow displacement used by '
            'CMRNext, not stereo left-right disparity.'
        ),
        (
            '- See bad_frames_geometry.csv for every selected frame and '
            'geometry_domain_gap_summary.json for full distributions, '
            'effect sizes, per-scene summaries, and correlations.'
        ),
    ])
    report_path = output_path / 'REPORT.md'
    report_path.write_text(
        '\n'.join(report_lines) + '\n', encoding='utf-8')

    fingerprint_path = output_path / 'fingerprints.sha256'
    artifacts = [
        bad_csv, train_csv, test_csv, summary_path, plot_path, report_path,
    ]
    fingerprint_path.write_text(
        ''.join(
            f'{_sha256_file(path)}  {path.name}\n'
            for path in artifacts
        ),
        encoding='utf-8',
    )
    print(f'[geometry-domain-gap] report: {report_path}')
    return summary

def main():
    """执行当前模块的主流程；在入口模块中负责解析或运行完整训练、评估任务。

    参数:
        无。

    返回值:
        无显式返回值；成功或失败通过进程退出码、日志和输出文件体现。

    异常和副作用:
        输入、checkpoint 或几何条件无效时抛出异常；执行模式可能占用 GPU 并写入日志、JSON、图像或报告。

    坐标系和数据约定:
        EMD 使用左乘残差 delta=Ti@inverse(T0)，使 delta@T0=Ti；图像张量通常为 B×C×H×W，光流通道为 (u, v)，单位为像素。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--data_folder', type=str, default='/data/KITTI/sequences/')
    parser.add_argument('--cam', type=str, nargs='?', default=None)
    parser.add_argument('--max_t', type=float, default=1.5)
    parser.add_argument('--max_r', type=float, default=20.)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--weights', type=str, nargs='+', default=None)
    parser.add_argument('--img_shape', type=int, nargs=1, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_file', type=str, nargs='?', default=None)
    parser.add_argument('--quantile', type=float, default=1.0)
    parser.add_argument('--downsample', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--viz', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--viz_dir', type=str, default=None,
                        help='--viz 时保存 PNG 的目录, 默认 <最后一个 ckpt 同目录>/viz')
    parser.add_argument('--viz_max_frames', type=int, default=20,
                        help='[已弃用] 仅在 --viz_strategy=first_n 时生效')
    parser.add_argument('--viz_strategy', type=str, default='topk_iter3_t',
                        choices=['first_n', 'topk_iter3_t', 'all_split',
                                 'per_scene_worst_joint'],
                        help='topk_iter3_t (默认): 按 iter3 残余 t_err 选 top-N 大 + top-M 小; '
                             'first_n: 旧逻辑, 按 batch_idx 取前 viz_max_frames 个 (与 errors 排序无关); '
                             'all_split: 全部测试帧都画图, 按 iter3 t_err 中位数二分到 <viz_dir>/good/ 和 <viz_dir>/bad/')
    parser.add_argument('--viz_top_worst', type=int, default=20,
                        help='--viz_strategy=topk_iter3_t 时, 保存 iter3 t_err 最大的 N 个样本')
    parser.add_argument('--viz_top_best', type=int, default=5,
                        help='--viz_strategy=topk_iter3_t 时, 同时保存 iter3 t_err 最小的 N 个作为对比')
    parser.add_argument('--viz_per_scene_worst_k', type=int, default=50,
                        help='按场景选择归一化联合残差最大的样本数。')
    parser.add_argument('--errors_json', type=str, default=None,
                        help='把每帧 iter0/1/2/3 的 (t_err_cm, r_err_deg) 写到此 JSON; '
                             '默认写到 <viz_dir>/errors.json (启用 --viz 时), 或不写 (未启用 --viz 时, 除非显式指定)')
    parser.add_argument('--emd_config', type=str, default='config/train_mid360_multi_extrinsic.yaml',
                        help='包含 EMD 传感器、几何语义和课程阶段的 YAML。')
    parser.add_argument('--emd_level',
                        choices=['large', 'medium', 'small', 'fixed_baseline', 'perturbed_t0'], default='large',
                        help='级联评估使用的权威 EMD level。')
    parser.add_argument('--emd_start_iteration', type=int, choices=[1, 2, 3], default=1,
                        help='第一个 EMD checkpoint 对应的阶段编号，默认为 1。')
    parser.add_argument(
        '--emd_scene_paths', nargs='+', default=None,
        help='直接读取的 EMD 场景目录，不使用 metadata.split 进行筛选。')
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help='仅诊断时限制推理帧数；正式评估保持为空。')
    parser.add_argument('--kitti_test_sequences', nargs='+', default=['00'],
                        help='dataset=kitti 时评估的 sequence 列表 (默认: 00).')
    parser.add_argument(
        '--kitti_nominal_cam', choices=['2', '3'], default=None,
        help=('将此 KITTI 相机外参作为固定 T0，目标相机仍由 --cam 指定；'
              '不采样 max_t/max_r 随机扰动。'))
    parser.add_argument(
        '--kitti_sampled_perturbation', action='store_true',
        help=('对 KITTI 使用原始 CMRNext 在线均匀 max_t/max_r 扰动，'
              '而不是固定标称相机外参。'))
    parser.add_argument(
        '--gt_flow_pnp_source_errors_json', type=str, default=None)
    parser.add_argument(
        '--gt_flow_pnp_selection_manifest', type=str, default=None)
    parser.add_argument(
        '--gt_flow_pnp_output_dir', type=str, default=None)
    parser.add_argument(
        '--gt_flow_pnp_repetitions', type=int, default=5)
    parser.add_argument(
        '--recorded_reprojection_errors_json', type=str, default=None)
    parser.add_argument(
        '--recorded_reprojection_output_dir', type=str, default=None)
    parser.add_argument(
        '--recorded_reprojection_per_scene_k', type=int, default=50)
    parser.add_argument(
        '--geometry_source_errors_json', type=str, default=None)
    parser.add_argument(
        '--geometry_selection_manifest', type=str, default=None)
    parser.add_argument(
        '--geometry_dataset_root', type=str, default=None)
    parser.add_argument(
        '--geometry_output_dir', type=str, default=None)
    parser.add_argument(
        '--geometry_train_reference_per_scene', type=int, default=100)
    parser.add_argument(
        '--geometry_test_reference_per_scene', type=int, default=200)
    parser.add_argument('--geometry_seed', type=int, default=42)

    args = parser.parse_args()
    _config = vars(args)
    diagnostic_values = (
        _config['gt_flow_pnp_source_errors_json'],
        _config['gt_flow_pnp_selection_manifest'],
        _config['gt_flow_pnp_output_dir'],
    )
    if any(value is not None for value in diagnostic_values):
        if not all(value is not None for value in diagnostic_values):
            parser.error(
                '真实光流 PnP 需要源误差 JSON、选择清单和输出目录')
        if _config['gt_flow_pnp_repetitions'] <= 0:
            parser.error('--gt_flow_pnp_repetitions 必须为正数')
        evaluate_gt_flow_pnp_diagnostic(
            _config['gt_flow_pnp_source_errors_json'],
            _config['gt_flow_pnp_selection_manifest'],
            _config['gt_flow_pnp_output_dir'],
            _config['gt_flow_pnp_repetitions'],
        )
        return
    recorded_values = (
        _config['recorded_reprojection_errors_json'],
        _config['recorded_reprojection_output_dir'],
    )
    if any(value is not None for value in recorded_values):
        if not all(value is not None for value in recorded_values):
            parser.error(
                '记录重投影需要源误差 JSON 和输出目录')
        if _config['recorded_reprojection_per_scene_k'] <= 0:
            parser.error(
                '--recorded_reprojection_per_scene_k 必须为正数')
        render_recorded_reprojections(
            _config['recorded_reprojection_errors_json'],
            _config['recorded_reprojection_output_dir'],
            _config['recorded_reprojection_per_scene_k'],
        )
        return
    geometry_values = (
        _config['geometry_source_errors_json'],
        _config['geometry_selection_manifest'],
        _config['geometry_dataset_root'],
        _config['geometry_output_dir'],
    )
    if any(value is not None for value in geometry_values):
        if not all(value is not None for value in geometry_values):
            parser.error(
                '几何域差异分析需要源误差、选择清单、数据根目录和输出目录')
        if (
            _config['geometry_train_reference_per_scene'] <= 0
            or _config['geometry_test_reference_per_scene'] <= 0
        ):
            parser.error('几何参考样本数必须为正数')
        evaluate_geometry_domain_gap(
            _config['geometry_source_errors_json'],
            _config['geometry_selection_manifest'],
            _config['geometry_dataset_root'],
            _config['geometry_output_dir'],
            _config['geometry_train_reference_per_scene'],
            _config['geometry_test_reference_per_scene'],
            _config['geometry_seed'],
        )
        return
    if _config['weights'] is None:
        parser.error('评估至少需要一个 --weights')
    if (_config['viz_strategy'] == 'per_scene_worst_joint'
            and _config['viz_per_scene_worst_k'] <= 0):
        parser.error('--viz_per_scene_worst_k 必须为正数')
    if (_config['viz_strategy'] == 'per_scene_worst_joint'
            and _config['dataset'] != 'emd'):
        parser.error('per_scene_worst_joint 要求 --dataset emd')
    if _config['kitti_nominal_cam'] is not None:
        if _config['dataset'] != 'kitti':
            parser.error('--kitti_nominal_cam 要求 --dataset kitti')
        if _config['cam'] is None:
            parser.error('--kitti_nominal_cam 要求显式指定目标 --cam')
        if str(_config['cam']) == _config['kitti_nominal_cam']:
            parser.error('KITTI 目标 --cam 与 --kitti_nominal_cam 不能相同')
    if _config['kitti_sampled_perturbation']:
        if _config['dataset'] != 'kitti':
            parser.error('--kitti_sampled_perturbation 要求 --dataset kitti')
        if _config['cam'] is None:
            parser.error('--kitti_sampled_perturbation 要求显式指定 --cam')
        if _config['kitti_nominal_cam'] is not None:
            parser.error(
                '--kitti_sampled_perturbation 不能与 --kitti_nominal_cam 同时使用')
        if _config['max_t'] <= 0 or _config['max_r'] <= 0:
            parser.error(
                '--kitti_sampled_perturbation 要求 max_t/max_r 为正数')
    evaluate_calibration(_config, _config['seed'])


if __name__ == '__main__':
    main()
