"""Regression tests for test entrypoint.
"""

import pytest

import test as test_entrypoint


def _checkpoints(tmp_path, count=3):
    """Checkpoints.

    Args:
        tmp_path: Tmp path.
        count: Count.
    """
    checkpoints = []
    for index in range(count):
        checkpoint = tmp_path / f"stage_{index + 1}.tar"
        checkpoint.touch()
        checkpoints.append(str(checkpoint))
    return checkpoints


def test_build_emd_args_uses_explicit_scene_paths(tmp_path):
    """Verify that build emd args uses explicit scene paths.

    Args:
        tmp_path: Tmp path.
    """
    arguments = test_entrypoint.build_emd_test_args(
        _checkpoints(tmp_path),
        level="large",
        start_stage=1,
        scene_paths=("/datasets/scene_a", "/datasets/scene_b"),
        errors_json="/results/large.json",
    )

    assert arguments[:2] == ["--dataset", "emd"]
    assert arguments[arguments.index("--emd_level") + 1] == "large"
    assert arguments[arguments.index("--emd_start_iteration") + 1] == "1"
    scene_index = arguments.index("--emd_scene_paths")
    assert arguments[scene_index + 1:scene_index + 3] == [
        "/datasets/scene_a", "/datasets/scene_b"]
    assert "--emd_split" not in arguments
    assert "--emd_data_root" not in arguments
    assert "--kitti_test_sequences" not in arguments
    assert "--max_t" not in arguments
    assert "--max_r" not in arguments


def test_build_emd_train_args_selects_per_scene_worst_triptychs(tmp_path):
    """Verify that build emd train args selects per scene worst triptychs.

    Args:
        tmp_path: Tmp path.
    """
    arguments = test_entrypoint.build_emd_test_args(
        _checkpoints(tmp_path, count=1),
        level="large",
        scene_paths=("/datasets/scene_a",),
        per_scene_worst_k=50,
        triptych_dir="/results/triptychs",
        errors_json="/results/errors.json",
    )

    assert arguments[arguments.index("--viz_strategy") + 1] == (
        "per_scene_worst_joint")
    assert arguments[arguments.index("--viz_per_scene_worst_k") + 1] == "50"
    assert arguments[arguments.index("--viz_dir") + 1] == "/results/triptychs"


def test_build_emd_args_rejects_nonpositive_worst_count(tmp_path):
    """Verify that build emd args rejects nonpositive worst count.

    Args:
        tmp_path: Tmp path.
    """
    with pytest.raises(ValueError, match="必须为正数"):
        test_entrypoint.build_emd_test_args(
            _checkpoints(tmp_path, count=1),
            level="large",
            scene_paths=("/datasets/scene_a",),
            per_scene_worst_k=0,
            triptych_dir="/results/triptychs",
            errors_json="/results/errors.json",
        )


def test_emd_parser_accepts_scene_triptychs(tmp_path):
    """Verify that emd parser accepts scene triptychs.

    Args:
        tmp_path: Tmp path.
    """
    args = test_entrypoint.build_parser().parse_args([
        "scene", "--weights", *_checkpoints(tmp_path, count=1),
        "--config", test_entrypoint.DEFAULT_SCENE_CONFIG,
        "--level", "large", "--scene-paths", "/datasets/scene_a",
        "--per-scene-worst-k", "50",
        "--triptych-dir", "/results/triptychs",
        "--errors-json", "/results/errors.json",
        "--log-file", "/results/evaluation.log",
    ])

    assert args.scene_paths == ["/datasets/scene_a"]
    assert args.per_scene_worst_k == 50


def test_build_fixed_t0_emd_args_uses_explicit_scene(tmp_path):
    """Verify that build fixed t0 emd args uses explicit scene.

    Args:
        tmp_path: Tmp path.
    """
    arguments = test_entrypoint.build_emd_test_args(
        _checkpoints(tmp_path, count=1),
        level="fixed_baseline",
        scene_paths=("/datasets/fixed_scene",),
        start_stage=1,
        emd_config="config/train_emd_fixed_t0_iter1.yaml",
        errors_json="/results/fixed_t0_iter1.json",
        seed=42,
    )

    assert arguments[arguments.index("--emd_level") + 1] == "fixed_baseline"
    assert arguments[arguments.index("--emd_start_iteration") + 1] == "1"
    assert arguments[arguments.index("--seed") + 1] == "42"
    assert "--max_t" not in arguments
    assert "--max_r" not in arguments


def test_build_kitti_args_uses_fixed_camera_extrinsics(tmp_path):
    """Verify that build kitti args uses fixed camera extrinsics.

    Args:
        tmp_path: Tmp path.
    """
    arguments = test_entrypoint.build_kitti_fixed_nominal_test_args(
        _checkpoints(tmp_path),
        sequences=("00", "05"),
        target_cam="3",
        nominal_cam="2",
        errors_json="/results/kitti.json",
    )

    assert arguments[:2] == ["--dataset", "kitti"]
    sequence_index = arguments.index("--kitti_test_sequences")
    assert arguments[sequence_index + 1:sequence_index + 3] == ["00", "05"]
    assert arguments[arguments.index("--cam") + 1] == "3"
    assert arguments[arguments.index("--kitti_nominal_cam") + 1] == "2"
    assert "--max_t" not in arguments
    assert "--max_r" not in arguments


def test_build_kitti_sampled_args_uses_original_cmrnext_ranges(tmp_path):
    """Verify that build kitti sampled args uses original cmrnext ranges.

    Args:
        tmp_path: Tmp path.
    """
    arguments = test_entrypoint.build_kitti_sampled_perturbation_test_args(
        _checkpoints(tmp_path),
        sequences=("00",),
        target_cam="2",
        errors_json="/results/kitti_left.json",
    )

    assert arguments[:2] == ["--dataset", "kitti"]
    assert arguments[arguments.index("--cam") + 1] == "2"
    assert "--kitti_sampled_perturbation" in arguments
    assert arguments[arguments.index("--max_t") + 1] == "1.5"
    assert arguments[arguments.index("--max_r") + 1] == "20.0"
    assert "--kitti_nominal_cam" not in arguments


def test_kitti_sampled_args_reject_nonpositive_range(tmp_path):
    """Verify that kitti sampled args reject nonpositive range.

    Args:
        tmp_path: Tmp path.
    """
    with pytest.raises(ValueError, match="必须为正数"):
        test_entrypoint.build_kitti_sampled_perturbation_test_args(
            _checkpoints(tmp_path),
            sequences=("00",),
            target_cam="2",
            max_t=0.0,
            errors_json="/results/kitti_left.json",
        )


def test_kitti_fixed_nominal_rejects_identical_cameras(tmp_path):
    """Verify that kitti fixed nominal rejects identical cameras.

    Args:
        tmp_path: Tmp path.
    """
    with pytest.raises(ValueError, match="不能相同"):
        test_entrypoint.build_kitti_fixed_nominal_test_args(
            _checkpoints(tmp_path),
            sequences=("00",),
            target_cam="2",
            nominal_cam="2",
            errors_json="/results/kitti.json",
        )


def test_post_training_suite_runs_generated_test_before_kitti(
        tmp_path, monkeypatch):
    """Verify that post training suite runs generated test before kitti.

    Args:
        tmp_path: Tmp path.
        monkeypatch: Monkeypatch.
    """
    calls = []
    monkeypatch.setattr(
        test_entrypoint,
        "run_emd_test",
        lambda *args, **kwargs: calls.append(("emd", kwargs["level"])),
    )
    monkeypatch.setattr(
        test_entrypoint,
        "run_kitti_fixed_nominal_test",
        lambda *args, **kwargs: calls.append(
            ("kitti", tuple(kwargs["sequences"]))),
    )
    monkeypatch.setattr(
        test_entrypoint,
        "run_kitti_sampled_perturbation_test",
        lambda *args, **kwargs: calls.append(
            ("kitti-perturbed", tuple(kwargs["sequences"]))),
    )
    case = test_entrypoint.KITTIFixedNominalCase(
        sequences=("00",),
        target_cam="3",
        nominal_cam="2",
        errors_json=str(tmp_path / "kitti.json"),
        log_file=str(tmp_path / "kitti.log"),
    )
    sampled_case = test_entrypoint.KITTISampledPerturbationCase(
        sequences=("00",),
        target_cam="2",
        errors_json=str(tmp_path / "kitti_left.json"),
        log_file=str(tmp_path / "kitti_left.log"),
    )

    test_entrypoint.run_post_training_suite(
        _checkpoints(tmp_path),
        emd_level="large",
        emd_scene_paths=("/datasets/scene_a",),
        emd_errors_json=str(tmp_path / "emd.json"),
        emd_log_file=str(tmp_path / "emd.log"),
        kitti_cases=(case,),
        kitti_sampled_cases=(sampled_case,),
        dry_run=True,
    )

    assert calls == [
        ("emd", "large"),
        ("kitti", ("00",)),
        ("kitti-perturbed", ("00",)),
    ]


def test_evaluator_runs_from_project_root(monkeypatch):
    """Verify that evaluator runs from project root.

    Args:
        monkeypatch: Monkeypatch.
    """
    captured = {}

    def fake_run(command, **kwargs):
        """Fake run.

        Args:
            command: Command.
            kwargs: Keyword arguments forwarded to the operation.
        """
        captured["command"] = command
        captured.update(kwargs)
        return "completed"

    monkeypatch.setattr(test_entrypoint.subprocess, "run", fake_run)
    result = test_entrypoint._run_evaluator(
        (), log_file=None, gpu=None, dry_run=False)

    assert result == "completed"
    assert captured["cwd"] == str(test_entrypoint.PROJECT_ROOT)

def test_build_gt_flow_pnp_diagnostic_args_uses_recorded_inputs(tmp_path):
    """Verify that build gt flow pnp diagnostic args uses recorded inputs.

    Args:
        tmp_path: Tmp path.
    """
    source = tmp_path / "errors.json"
    manifest = tmp_path / "manifest.json"
    source.touch()
    manifest.touch()

    arguments = test_entrypoint.build_gt_flow_pnp_diagnostic_args(
        source_errors_json=str(source),
        selection_manifest=str(manifest),
        output_dir=str(tmp_path / "gt_flow"),
        repetitions=7,
    )

    assert arguments[arguments.index(
        "--gt_flow_pnp_source_errors_json") + 1] == str(source.resolve())
    assert arguments[arguments.index(
        "--gt_flow_pnp_selection_manifest") + 1] == str(manifest.resolve())
    assert arguments[arguments.index(
        "--gt_flow_pnp_repetitions") + 1] == "7"
    assert "--weights" not in arguments


def test_gt_flow_pnp_parser_does_not_require_checkpoints(tmp_path):
    """Verify that gt flow pnp parser does not require checkpoints.

    Args:
        tmp_path: Tmp path.
    """
    source = tmp_path / "errors.json"
    manifest = tmp_path / "manifest.json"
    source.touch()
    manifest.touch()

    args = test_entrypoint.build_parser().parse_args([
        "gt-flow-pnp",
        "--source-errors-json", str(source),
        "--selection-manifest", str(manifest),
        "--output-dir", str(tmp_path / "output"),
        "--log-file", str(tmp_path / "evaluate.log"),
    ])

    assert args.test_mode == "gt-flow-pnp"
    assert args.repetitions == 5


def test_build_recorded_reprojection_args_uses_recorded_poses(tmp_path):
    """Verify that build recorded reprojection args uses recorded poses.

    Args:
        tmp_path: Tmp path.
    """
    source = tmp_path / "errors.json"
    source.touch()
    arguments = test_entrypoint.build_recorded_reprojection_args(
        source_errors_json=str(source),
        output_dir=str(tmp_path / "reprojections"),
        per_scene_k=17,
    )
    assert arguments[arguments.index(
        "--recorded_reprojection_errors_json") + 1] == str(source.resolve())
    assert arguments[arguments.index(
        "--recorded_reprojection_output_dir") + 1] == str(
            tmp_path / "reprojections")
    assert arguments[arguments.index(
        "--recorded_reprojection_per_scene_k") + 1] == "17"
    assert "--weights" not in arguments


def test_build_recorded_reprojection_args_rejects_nonpositive_count(tmp_path):
    """Verify that build recorded reprojection args rejects nonpositive count.

    Args:
        tmp_path: Tmp path.
    """
    source = tmp_path / "errors.json"
    source.touch()
    with pytest.raises(ValueError, match="必须为正数"):
        test_entrypoint.build_recorded_reprojection_args(
            source_errors_json=str(source),
            output_dir=str(tmp_path / "reprojections"),
            per_scene_k=0,
        )


def test_recorded_reprojection_parser_does_not_require_checkpoints(tmp_path):
    """Verify that recorded reprojection parser does not require checkpoints.

    Args:
        tmp_path: Tmp path.
    """
    source = tmp_path / "errors.json"
    source.touch()
    args = test_entrypoint.build_parser().parse_args([
        "recorded-reprojections",
        "--source-errors-json", str(source),
        "--output-dir", str(tmp_path / "reprojections"),
        "--log-file", str(tmp_path / "render.log"),
    ])
    assert args.test_mode == "recorded-reprojections"
    assert args.per_scene_k == 50


def test_emd_parser_and_builder_accept_perturbed_t0(tmp_path):
    """Verify that emd parser and builder accept perturbed t0.

    Args:
        tmp_path: Tmp path.
    """
    checkpoints = _checkpoints(tmp_path, count=1)
    args = test_entrypoint.build_parser().parse_args([
        "scene", "--weights", *checkpoints,
        "--config", test_entrypoint.DEFAULT_SCENE_CONFIG,
        "--level", "perturbed_t0", "--scene-paths", "/datasets/scene_a",
        "--errors-json", "/results/errors.json",
        "--log-file", "/results/evaluation.log",
    ])
    arguments = test_entrypoint.build_emd_test_args(
        checkpoints, level=args.level, scene_paths=args.scene_paths,
        errors_json=args.errors_json)

    assert args.level == "perturbed_t0"
    assert arguments[arguments.index("--emd_level") + 1] == "perturbed_t0"
    assert "--emd_split" not in arguments


def test_emd_cross_training_is_normal_evaluation(tmp_path):
    """Verify that emd cross training is normal evaluation.

    Args:
        tmp_path: Tmp path.
    """
    checkpoints = _checkpoints(tmp_path, count=3)
    arguments = test_entrypoint.build_emd_test_args(
        checkpoints,
        level="perturbed_t0",
        scene_paths=("/datasets/scene_a",),
        errors_json="/results/cross_domain.json",
        max_frames=2,
    )

    assert "--cross_training_domain" not in arguments
    assert arguments[arguments.index("--max_frames") + 1] == "2"


def test_emd_rejects_nonpositive_frame_limit(tmp_path):
    """Verify that emd rejects nonpositive frame limit.

    Args:
        tmp_path: Tmp path.
    """
    with pytest.raises(ValueError, match="必须为正数"):
        test_entrypoint.build_emd_test_args(
            _checkpoints(tmp_path, count=1),
            level="perturbed_t0",
            scene_paths=("/datasets/scene_a",),
            errors_json="/results/cross_domain.json",
            max_frames=0,
        )


def test_common_parser_accepts_external_reprojection_switch(tmp_path) -> None:
    """Verify that common parser accepts external reprojection switch.

    Args:
        tmp_path: Tmp path.

    Returns:
        None.
    """
    checkpoint = _checkpoints(tmp_path, count=1)[0]
    tool = tmp_path / "reproject.py"
    tool.touch()

    args = test_entrypoint.build_parser().parse_args([
        "scene",
        "--weights", checkpoint,
        "--config", test_entrypoint.DEFAULT_SCENE_CONFIG,
        "--level", "perturbed_t0",
        "--scene-paths", str(tmp_path / "scene"),
        "--errors-json", str(tmp_path / "errors.json"),
        "--log-file", str(tmp_path / "evaluation.log"),
        "--reprojection",
        "--reprojection-output", str(tmp_path / "reprojections"),
        "--reprojection-tool", str(tool),
        "--dry-run",
    ])

    assert args.reprojection is True
    assert args.reprojection_output == str(tmp_path / "reprojections")
    assert args.reprojection_tool == str(tool)


def test_emd_dry_run_prints_evaluator_then_external_reprojection(
        tmp_path, capsys) -> None:
    """Verify that emd dry run prints evaluator then external reprojection.

    Args:
        tmp_path: Tmp path.
        capsys: Capsys.

    Returns:
        None.
    """
    checkpoint = _checkpoints(tmp_path, count=1)[0]
    tool = tmp_path / "reproject.py"
    tool.touch()
    errors_json = tmp_path / "errors.json"
    output_dir = tmp_path / "reprojections"

    test_entrypoint.main([
        "scene",
        "--weights", checkpoint,
        "--config", test_entrypoint.DEFAULT_SCENE_CONFIG,
        "--level", "perturbed_t0",
        "--scene-paths", str(tmp_path / "scene"),
        "--errors-json", str(errors_json),
        "--log-file", str(tmp_path / "evaluation.log"),
        "--reprojection",
        "--reprojection-output", str(output_dir),
        "--reprojection-tool", str(tool),
        "--dry-run",
    ])

    output_lines = capsys.readouterr().out.strip().splitlines()
    assert len(output_lines) == 2
    assert "evaluate_flow_calibration.py" in output_lines[0]
    assert output_lines[1].endswith(
        "--input-type errors --input {} --output {}".format(
            errors_json.resolve(), output_dir.resolve()))


def test_emd_scene_paths_are_forwarded_without_split_filtering(tmp_path):
    """Verify that emd scene paths are forwarded without split filtering.

    Args:
        tmp_path: Tmp path.
    """
    checkpoint = _checkpoints(tmp_path, count=1)
    arguments = test_entrypoint.build_emd_test_args(
        checkpoint,
        level="perturbed_t0",
        errors_json="/results/direct_scenes.json",
        scene_paths=("/data/scene_a", "/data/scene_b"),
        max_frames=3,
    )

    index = arguments.index("--emd_scene_paths")
    assert arguments[index + 1:index + 3] == [
        "/data/scene_a", "/data/scene_b"]
    assert arguments[arguments.index("--max_frames") + 1] == "3"


def test_emd_parser_allows_yaml_test_scenes(tmp_path):
    """Verify that emd parser allows yaml test scenes.

    Args:
        tmp_path: Tmp path.
    """
    args = test_entrypoint.build_parser().parse_args([
        "scene", "--weights", *_checkpoints(tmp_path, count=1),
        "--config", test_entrypoint.DEFAULT_SCENE_CONFIG,
        "--level", "perturbed_t0",
        "--errors-json", "/results/errors.json",
        "--log-file", "/results/evaluation.log",
    ])

    assert args.scene_paths is None


def test_emd_yaml_test_scenes_are_resolved(monkeypatch, tmp_path):
    """Verify that emd yaml test scenes are resolved.

    Args:
        monkeypatch: Monkeypatch.
        tmp_path: Tmp path.
    """
    configured_scene = tmp_path / "dataset/test/scene_a"
    fake_config = type("FakeEMDConfig", (), {
        "test_scene_paths": (configured_scene,),
    })()
    monkeypatch.setattr(
        test_entrypoint, "load_emd_training_config",
        lambda _: fake_config)

    resolved = test_entrypoint._resolve_emd_test_scene_paths(
        "/configs/mid360.yaml", None)

    assert resolved == [str(configured_scene)]


def test_emd_command_scenes_override_yaml(monkeypatch, tmp_path):
    """Verify that emd command scenes override yaml.

    Args:
        monkeypatch: Monkeypatch.
        tmp_path: Tmp path.
    """
    monkeypatch.setattr(
        test_entrypoint, "load_emd_training_config",
        lambda _: pytest.fail("命令行覆盖时不应读取 YAML 场景"))
    scene = tmp_path / "command_scene"

    resolved = test_entrypoint._resolve_emd_test_scene_paths(
        "/configs/mid360.yaml", [str(scene)])

    assert resolved == [str(scene.resolve())]


def test_emd_empty_yaml_test_scenes_are_rejected(monkeypatch):
    """Verify that emd empty yaml test scenes are rejected.

    Args:
        monkeypatch: Monkeypatch.
    """
    fake_config = type("FakeEMDConfig", (), {
        "test_scene_paths": (),
    })()
    monkeypatch.setattr(
        test_entrypoint, "load_emd_training_config",
        lambda _: fake_config)

    with pytest.raises(ValueError, match="测试场景为空"):
        test_entrypoint._resolve_emd_test_scene_paths(
            "/configs/mid360.yaml", None)
