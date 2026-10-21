"""项目源码模块 tests/test_test_entrypoint.py。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

import pytest

import test as test_entrypoint


def _checkpoints(tmp_path, count=3):
    """完成 tests/test_test_entrypoint.py 中的 `_checkpoints` 操作。

    参数:
        tmp_path: pytest 提供的独立临时目录。
        count: 需要选择或生成的元素数量。

    返回值:
        返回由 `实现中的返回表达式` 表示的计算结果。

    异常和副作用:
        输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。
    """
    checkpoints = []
    for index in range(count):
        checkpoint = tmp_path / f"stage_{index + 1}.tar"
        checkpoint.touch()
        checkpoints.append(str(checkpoint))
    return checkpoints


def test_build_emd_args_uses_explicit_scene_paths(tmp_path):
    """验证回归场景 `test_build_emd_args_uses_explicit_scene_paths`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_build_emd_train_args_selects_per_scene_worst_triptychs`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_build_emd_args_rejects_nonpositive_worst_count`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_emd_parser_accepts_scene_triptychs`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_build_fixed_t0_emd_args_uses_explicit_scene`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_build_kitti_args_uses_fixed_camera_extrinsics`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_build_kitti_sampled_args_uses_original_cmrnext_ranges`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_kitti_sampled_args_reject_nonpositive_range`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_kitti_fixed_nominal_rejects_identical_cameras`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_post_training_suite_runs_generated_test_before_kitti`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。
        monkeypatch: pytest 提供的属性、环境变量和函数替换工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_evaluator_runs_from_project_root`，确保相关数据、几何或入口行为保持预期。

    参数:
        monkeypatch: pytest 提供的属性、环境变量和函数替换工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
    """
    captured = {}

    def fake_run(command, **kwargs):
        """完成 tests/test_test_entrypoint.py 中的 `fake_run` 操作。

        参数:
            command: 测试替身捕获的子进程命令参数列表。
            **kwargs: 测试替身捕获的其余 subprocess.run 关键字参数。

        返回值:
            返回由 `实现中的返回表达式` 表示的计算结果。

        异常和副作用:
            输入类型、形状或数值不满足实现要求时抛出相应异常；是否原地修改输入以函数体实现为准。
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
    """验证回归场景 `test_build_gt_flow_pnp_diagnostic_args_uses_recorded_inputs`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_gt_flow_pnp_parser_does_not_require_checkpoints`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_build_recorded_reprojection_args_uses_recorded_poses`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_build_recorded_reprojection_args_rejects_nonpositive_count`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_recorded_reprojection_parser_does_not_require_checkpoints`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_emd_parser_and_builder_accept_perturbed_t0`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_emd_cross_training_is_normal_evaluation`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_emd_rejects_nonpositive_frame_limit`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_common_parser_accepts_external_reprojection_switch`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_emd_dry_run_prints_evaluator_then_external_reprojection`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。
        capsys: pytest 提供的标准输出与标准错误捕获工具。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证回归场景 `test_emd_scene_paths_are_forwarded_without_split_filtering`，确保相关数据、几何或入口行为保持预期。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；测试通过断言表达成功，失败时由 pytest 报告。

    异常和副作用:
        断言不满足时使测试失败；临时文件和替身对象由 pytest fixture 管理。
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
    """验证 EMD 测试入口允许省略命令行场景并回退到 YAML。

    参数:
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；解析结果的 scene_paths 为 None 即通过。

    异常和副作用:
        创建临时 checkpoint；断言失败时使测试失败。
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
    """验证 YAML test 场景会解析为共享评估器使用的绝对路径。

    参数:
        monkeypatch: pytest 属性替换工具。
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；返回 YAML 场景绝对路径即通过。

    异常和副作用:
        替换当前测试进程中的配置加载函数；不访问真实数据目录。

    坐标系和外参约定:
        路径解析不改变 T0、Ti 或左乘 residual。
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
    """验证命令行场景整组覆盖 YAML test 场景且不触发 YAML 加载。

    参数:
        monkeypatch: pytest 属性替换工具。
        tmp_path: pytest 提供的独立临时目录。

    返回值:
        无显式返回值；仅返回命令行绝对路径即通过。

    异常和副作用:
        替换配置加载函数为失败哨兵；不读取数据文件。
    """
    monkeypatch.setattr(
        test_entrypoint, "load_emd_training_config",
        lambda _: pytest.fail("命令行覆盖时不应读取 YAML 场景"))
    scene = tmp_path / "command_scene"

    resolved = test_entrypoint._resolve_emd_test_scene_paths(
        "/configs/mid360.yaml", [str(scene)])

    assert resolved == [str(scene.resolve())]


def test_emd_empty_yaml_test_scenes_are_rejected(monkeypatch):
    """验证 YAML 与命令行均无测试场景时给出明确错误。

    参数:
        monkeypatch: pytest 属性替换工具。

    返回值:
        无显式返回值；预期 ValueError 被捕获即通过。

    异常和副作用:
        替换当前测试进程中的配置加载函数；不访问数据文件。
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
