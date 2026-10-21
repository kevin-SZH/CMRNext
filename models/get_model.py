"""CMRNext 网络实例构造。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

from argparse import Namespace

from models.RAFT.raft import CMRNext


def get_model(_config, img_shape):
    """根据训练或评估配置构造 CMRNext RAFT 网络。

    参数:
        _config: 运行配置字典，包含数据、网络、几何和输出选项。
        img_shape: 网络输入尺寸，按高度、宽度排列，单位为像素。

    返回值:
        根据配置初始化但尚未加载 checkpoint 的 CMRNext 模型。

    异常和副作用:
        输入通道、空间尺寸或设备不一致时由 PyTorch 抛出异常；训练模式可能更新批归一化统计。
    """
    if 'fourier_levels' not in _config:
        _config['fourier_levels'] = -1
    raft_args = Namespace()
    raft_args.small = False
    raft_args.iters = 12
    model = CMRNext(raft_args, use_reflectance=_config['use_reflectance'],
                            with_uncertainty=_config['uncertainty'],
                            fourier_levels=_config['fourier_levels'], unc_type=_config['der_type'],
                            unc_freeze=_config["unc_freeze"], context_encoder=_config["context_encoder"])
    return model
