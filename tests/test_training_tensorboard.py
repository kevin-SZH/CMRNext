"""Regression tests for training tensorboard.
"""

from pathlib import Path

import pytest

from train_calibration import _mount_tensorboard_logdir


def test_mount_tensorboard_logdir_creates_and_reuses_symlink(tmp_path):
    """Verify that mount tensorboard logdir creates and reuses symlink.

    Args:
        tmp_path: Tmp path.
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
    """Verify that mount tensorboard logdir rejects different existing target.

    Args:
        tmp_path: Tmp path.
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
