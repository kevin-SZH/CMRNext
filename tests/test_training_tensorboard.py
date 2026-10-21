"""训练 TensorBoard 日志持久挂载的聚焦回归测试。"""

from pathlib import Path

import pytest

from train_calibration import _mount_tensorboard_logdir


def test_mount_tensorboard_logdir_creates_and_reuses_symlink(tmp_path):
    """验证首次训练创建挂载链接且恢复同一任务时安全复用。

    参数:
        tmp_path: pytest 提供的临时目录，路径无单位，用于隔离 events 和服务
            监听目录。

    返回值:
        无显式返回值；软链接存在、目标正确且重复调用返回同一路径即通过。

    异常和副作用:
        在临时目录创建真实日志目录、监听目录和目录软链接；断言失败时由
        pytest 报告，不涉及点云或相机外参。
    """
    # Model run output and the TensorBoard service root separately.
    tb_logdir = tmp_path / 'checkpoints' / 'run-a' / 'tb'
    mount_root = tmp_path / 'tf-logs'

    mount_path = _mount_tensorboard_logdir(tb_logdir, mount_root, 'run-a')
    reused_mount_path = _mount_tensorboard_logdir(
        tb_logdir, mount_root, 'run-a')

    assert mount_path == mount_root / 'run-a'
    assert reused_mount_path == mount_path
    assert mount_path.is_symlink()
    assert mount_path.resolve() == tb_logdir.resolve()


def test_mount_tensorboard_logdir_rejects_different_existing_target(tmp_path):
    """验证同名挂载已属于其他任务目录时不会被静默覆盖。

    参数:
        tmp_path: pytest 提供的临时目录，路径无单位，用于构造挂载名冲突。

    返回值:
        无显式返回值；函数抛出 ``FileExistsError`` 且旧链接保持不变即通过。

    异常和副作用:
        在临时目录创建两个 events 目录及一个软链接；预期捕获
        ``FileExistsError``，不涉及点云或相机外参。
    """
    # Model an existing same-name run in the service directory.
    old_tb_logdir = tmp_path / 'old-checkpoints' / 'run-a' / 'tb'
    new_tb_logdir = tmp_path / 'new-checkpoints' / 'run-a' / 'tb'
    mount_root = tmp_path / 'tf-logs'
    old_mount_path = _mount_tensorboard_logdir(
        old_tb_logdir, mount_root, 'run-a')

    with pytest.raises(FileExistsError, match='已指向其他目录'):
        _mount_tensorboard_logdir(new_tb_logdir, mount_root, 'run-a')

    assert old_mount_path.resolve() == old_tb_logdir.resolve()
    assert Path(old_mount_path).is_symlink()
