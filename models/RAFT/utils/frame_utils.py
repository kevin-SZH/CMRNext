"""光流、视差和图像文件读写。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import numpy as np
from PIL import Image
from os.path import *
import re
import cv2

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """读取并解析当前函数约定的文件格式。

    参数:
        fn: Middlebury .flo 光流文件路径。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    # Adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # Supports little-endian architectures only.
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape to (h, w, 2) for visualization.
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    """读取并解析当前函数约定的文件格式。

    参数:
        file: 需要读取的点云、光流或图像文件路径。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # Little-endian.
        endian = '<'
        scale = -scale
    else:
        endian = '>' # Big-endian.

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writeFlow(filename,uv,v=None):
    """把当前函数接收的数据编码并写入目标文件。

    参数:
        filename: 需要读取或写入的光流文件路径。
        uv: 图像像素坐标，形状为 N×2，顺序为 u、v，单位为像素。
        v: 光流垂直分量数组，正方向向下，单位为像素。

    返回值:
        无显式返回值；结果写入调用方指定的文件或输出流。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # Write the header.
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # Arrange as a matrix.
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    """读取并解析当前函数约定的文件格式。

    参数:
        filename: 需要读取或写入的光流文件路径。

    返回值:
        计算得到的光流、有效索引、指标或可视化结果；具体组合遵循调用点解包方式。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def writeFlowKITTI(filename, uv):
    """把当前函数接收的数据编码并写入目标文件。

    参数:
        filename: 需要读取或写入的光流文件路径。
        uv: 图像像素坐标，形状为 N×2，顺序为 u、v，单位为像素。

    返回值:
        无显式返回值；结果写入调用方指定的文件或输出流。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def read_gen(file_name, pil=False):
    """读取并解析当前函数约定的文件格式。

    参数:
        file_name: 需要通用读取的图像、光流、视差或数组文件路径。
        pil: 是否优先保留 PIL 图像对象而非转换为 NumPy 数组。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果；其他分支按函数中的条件返回相同语义结果。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。

    坐标系和数据约定:
        网络张量采用 B×C×H×W；二维光流通道依次为水平 u 和垂直 v，数值单位为像素。
    """
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        return flow[:, :, :-1]
    return []
