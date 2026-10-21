"""Regression tests for training crop.
"""

import subprocess
import sys
from pathlib import Path

from train_calibration import (
    _mask_flow_endpoints_inside,
    _random_crop_offset,
)


def test_random_crop_offset_uses_configured_bounds(monkeypatch):
    """Verify that random crop offset uses configured bounds.

    Args:
        monkeypatch: Monkeypatch.
    """
    calls = []

    def fake_randint(minimum, maximum):
        """Fake randint.

        Args:
            minimum: Minimum.
            maximum: Largest allowed integer offset, inclusive.
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
    """Verify that random crop offset preserves default full range.

    Args:
        monkeypatch: Monkeypatch.
    """
    calls = []

    def fake_randint(minimum, maximum):
        """Fake randint.

        Args:
            minimum: Minimum.
            maximum: Largest allowed integer offset, inclusive.
        """
        calls.append((minimum, maximum))
        return minimum

    monkeypatch.setattr("train_calibration.random.randint", fake_randint)

    assert _random_crop_offset({}, "emd_crop_top_range", 50) == 0
    assert calls == [(0, 50)]


def test_training_flow_mask_excludes_cropped_endpoints():
    """Verify that training flow mask excludes cropped endpoints.
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
    """Verify that resolve emd training scene paths uses yaml defaults.

    Args:
        tmp_path: Tmp path.
    """
    from train_calibration import _resolve_emd_training_scene_paths

    class Config:
        """Config.
        """

        def scene_paths(self, role):
            """Return the configured scene paths for train, validation, or test without filtering by metadata.

            Args:
                role: Scene role: train, validation, or test.
            """
            return ((tmp_path / role).resolve(),)

        def audit_record(self):
            """Return a serializable record of explicit scenes, sensor geometry, and training semantics.
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
    """Verify that resolve emd training scene paths command group overrides.

    Args:
        tmp_path: Tmp path.
    """
    from train_calibration import _resolve_emd_training_scene_paths

    class Config:
        """Config.
        """

        def scene_paths(self, role):
            """Return the configured scene paths for train, validation, or test without filtering by metadata.

            Args:
                role: Scene role: train, validation, or test.
            """
            return ((tmp_path / f"yaml_{role}").resolve(),)

        def audit_record(self):
            """Return a serializable record of explicit scenes, sensor geometry, and training semantics.
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
    """Verify that training help excludes raw dataset options.
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
    """Verify that training entrypoint rejects legacy and raw dataset options.
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
