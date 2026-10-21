"""项目源码模块 tests/test_training_crop.py。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import subprocess
import sys
from pathlib import Path

from train_calibration import (
    _mask_flow_endpoints_inside,
    _random_crop_offset,
)


def test_random_crop_offset_uses_configured_bounds(monkeypatch):
    """验证回归场景 `test_random_crop_offset_uses_configured_bounds`，确保相关数据、几何或入口行为保持预期。

    参数:
        monkeypatch: pytest 提供的属性、环境变量和函数替换工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    calls = []

    def fake_randint(minimum, maximum):
        """完成 tests/test_training_crop.py 中的 `fake_randint` 操作。

        参数:
            minimum: 随机整数或裁剪偏移允许的最小值，包含端点。
            maximum: 随机裁剪或索引允许的最大整数值，包含端点。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。
        """
        calls.append((minimum, maximum))
        return maximum

    monkeypatch.setattr("train_calibration.random.randint", fake_randint)

    config = {
        "emd_crop_top_range": (921, 976),
        "emd_crop_left_range": (358, 640),
    }
    assert _random_crop_offset(config, "emd_crop_top_range", 976) == 976
    assert _random_crop_offset(config, "emd_crop_left_range", 640) == 640
    assert calls == [(921, 976), (358, 640)]


def test_random_crop_offset_preserves_default_full_range(monkeypatch):
    """验证回归场景 `test_random_crop_offset_preserves_default_full_range`，确保相关数据、几何或入口行为保持预期。

    参数:
        monkeypatch: pytest 提供的属性、环境变量和函数替换工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    calls = []

    def fake_randint(minimum, maximum):
        """完成 tests/test_training_crop.py 中的 `fake_randint` 操作。

        参数:
            minimum: 随机整数或裁剪偏移允许的最小值，包含端点。
            maximum: 随机裁剪或索引允许的最大整数值，包含端点。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。
        """
        calls.append((minimum, maximum))
        return minimum

    monkeypatch.setattr("train_calibration.random.randint", fake_randint)

    assert _random_crop_offset({}, "emd_crop_top_range", 50) == 0
    assert calls == [(0, 50)]


def test_training_flow_mask_excludes_cropped_endpoints():
    """验证训练 crop 后目标终点越界的稀疏光流不参与监督。

    参数:
        无。

    返回值:
        无显式返回值；越界点被清零且内部点保持有效即通过。

    异常和副作用:
        断言失败时使测试失败；只创建内存张量，不写文件。

    坐标系和外参约定:
        源和目标均使用裁剪后的像素坐标，目标位置为 source_uv + flow_gt。
    """
    import torch

    flow = torch.zeros((4, 5, 2), dtype=torch.float32)
    mask = torch.zeros((4, 5), dtype=torch.int32)
    mask[1, 1] = 1
    mask[2, 2] = 1
    flow[1, 1, 1] = -2.0
    flow[2, 2] = torch.tensor([1.0, 0.0])

    filtered_flow, filtered_mask = _mask_flow_endpoints_inside(
        flow, mask)

    assert filtered_mask[1, 1].item() == 0
    assert filtered_flow[1, 1].tolist() == [0.0, 0.0]
    assert filtered_mask[2, 2].item() == 1
    assert filtered_flow[2, 2].tolist() == [1.0, 0.0]


def test_resolve_emd_training_scene_paths_uses_yaml_defaults(tmp_path):
    """验证训练入口使用 YAML 场景默认值并记录来源。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；三个场景组和来源写入训练配置即通过。

    异常和副作用:
        仅构造内存配置，不读取实际数据文件。

    坐标系和外参约定:
        场景解析不改变 T0、Ti 或 delta_Ti。
    """
    from train_calibration import _resolve_emd_training_scene_paths

    class Config:
        """提供训练场景解析测试所需的最小 EMD 配置接口。

        参数:
            无；测试临时目录通过外层函数闭包提供。

        返回值:
            实例提供场景路径与审计记录两个只读接口。

        异常和副作用:
            不访问文件系统中的数据场景，无外部副作用。

        坐标系和外参约定:
            仅返回路径和 schema，不读取或修改 T0、Ti 与左乘残差。
        """

        def scene_paths(self, role):
            """返回指定角色的单个测试路径。

            参数:
                role: train、validation 或 test 数据角色。

            返回值:
                对应角色的绝对路径元组。

            异常和副作用:
                无异常与外部副作用。

            坐标系和外参约定:
                本方法只返回路径，不读取外参。
            """
            return ((tmp_path / role).resolve(),)

        def audit_record(self):
            """返回最小审计记录。

            参数:
                无。

            返回值:
                包含 v2 schema 名称的字典。

            异常和副作用:
                无异常与外部副作用。

            坐标系和外参约定:
                审计记录不包含几何矩阵。
            """
            return {"schema_version": "cmrnext_emd_training_v2"}

    runtime = {
        "emd_train_scene_paths": None,
        "emd_val_scene_paths": None,
        "emd_test_scene_paths": None,
    }
    _resolve_emd_training_scene_paths(runtime, Config())

    assert runtime["emd_train_scene_paths"] == [
        str((tmp_path / "train").resolve())]
    assert runtime["emd_val_scene_paths"] == [
        str((tmp_path / "validation").resolve())]
    assert runtime["emd_test_scene_paths"] == [
        str((tmp_path / "test").resolve())]
    assert runtime["emd_scene_path_sources"] == {
        "train": "yaml", "validation": "yaml", "test": "yaml"}


def test_resolve_emd_training_scene_paths_command_group_overrides(tmp_path):
    """验证训练命令行只整组覆盖其显式提供的场景角色。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；train 来自命令行而其他角色来自 YAML 即通过。

    异常和副作用:
        仅构造内存配置，不读取实际数据文件。

    坐标系和外参约定:
        路径覆盖不改变任何外参矩阵。
    """
    from train_calibration import _resolve_emd_training_scene_paths

    class Config:
        """提供命令行覆盖测试所需的最小 EMD 配置接口。

        参数:
            无；测试临时目录通过外层函数闭包提供。

        返回值:
            实例提供 YAML 默认路径与最小审计记录接口。

        异常和副作用:
            不访问文件系统中的数据场景，无外部副作用。

        坐标系和外参约定:
            仅返回路径和 schema，不读取或修改 T0、Ti 与左乘残差。
        """

        def scene_paths(self, role):
            """返回 YAML 中指定角色的默认路径。

            参数:
                role: train、validation 或 test 数据角色。

            返回值:
                对应角色的路径元组。

            异常和副作用:
                无异常与外部副作用。

            坐标系和外参约定:
                本方法只返回路径，不读取外参。
            """
            return ((tmp_path / f"yaml_{role}").resolve(),)

        def audit_record(self):
            """返回 v2 最小审计记录。

            参数:
                无。

            返回值:
                schema 字典。

            异常和副作用:
                无异常与外部副作用。

            坐标系和外参约定:
                审计记录不包含几何矩阵。
            """
            return {"schema_version": "cmrnext_emd_training_v2"}

    command_train = tmp_path / "command_train"
    runtime = {
        "emd_train_scene_paths": [str(command_train)],
        "emd_val_scene_paths": None,
        "emd_test_scene_paths": None,
    }
    _resolve_emd_training_scene_paths(runtime, Config())

    assert runtime["emd_train_scene_paths"] == [str(command_train.resolve())]
    assert runtime["emd_scene_path_sources"]["train"] == "command_line"
    assert runtime["emd_scene_path_sources"]["validation"] == "yaml"


def test_training_help_excludes_raw_dataset_options():
    """验证训练入口不再公开原始数据集训练参数。

    参数:
        无。

    返回值:
        无显式返回值；帮助文本不含 KITTI、Argoverse、PandaSet 路径及旧扰动
        参数即通过。

    异常和副作用:
        启动一个只打印帮助的 Python 子进程；不加载数据、不占用 GPU、不写产物。

    坐标系和外参约定:
        本测试只检查 CLI，EMD 的 T0、Ti 与左乘残差约定不变。
    """
    # Resolve the canonical training entry point from the repository root.
    entrypoint = Path(__file__).resolve().parents[1] / "train_calibration.py"
    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        cwd=str(entrypoint.parent),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    removed_options = (
        "--data_folder_kitti",
        "--data_folder_argo",
        "--data_folder_panda",
        "--kitti_only",
        "--max_t",
        "--max_r",
        "--emd_only",
        "--emd_config",
        "--emd_level",
    )
    assert all(option not in completed.stdout for option in removed_options)
    assert "--config" in completed.stdout
    assert "--level" in completed.stdout
    assert "--train-scene-paths" in completed.stdout


def test_training_entrypoint_rejects_legacy_and_raw_dataset_options():
    """验证训练入口拒绝旧 EMD 开关或继续传入旧原始数据集参数。

    参数:
        无。

    返回值:
        无显式返回值；两类旧调用均以 argparse 退出码 2 结束即通过。

    异常和副作用:
        启动两个参数解析子进程；错误发生在配置、数据、CUDA 和输出初始化之前。

    坐标系和外参约定:
        拒绝旧入口不会读取或修改任何 T0、Ti 与 residual。
    """
    # Cover legacy EMD flags and rejected KITTI training paths.
    entrypoint = Path(__file__).resolve().parents[1] / "train_calibration.py"
    config_path = entrypoint.parent / "config/train_mid360_multi_extrinsic.yaml"
    required_options = [
        "--config", str(config_path), "--level", "perturbed_t0"]
    legacy_emd = subprocess.run(
        [sys.executable, str(entrypoint), *required_options,
         "--emd_only", "false", "--dry-run"],
        cwd=str(entrypoint.parent),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    raw_kitti = subprocess.run(
        [sys.executable, str(entrypoint), *required_options,
         "--data_folder_kitti", "/datasets/kitti", "--dry-run"],
        cwd=str(entrypoint.parent),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert legacy_emd.returncode == 2
    assert "unrecognized arguments: --emd_only false" in legacy_emd.stderr
    assert raw_kitti.returncode == 2
    assert "unrecognized arguments: --data_folder_kitti" in raw_kitti.stderr
