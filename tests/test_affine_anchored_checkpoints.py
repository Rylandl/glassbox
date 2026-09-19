"""Toy-only parity and externally anchored initializer witness integrity."""

import json
import sys

import jax
import numpy as np
import pytest
from test_fit_checkpoints import (
    PROVENANCE,
    SETTINGS,
    TOY_RECIPE,
    assert_model_equal,
    toy_scale,
    windows,
)
from test_fit_checkpoints import (
    source_map as inherited_source_map,
)

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.experimental import affine_anchored_checkpoints as checkpoints


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def source_map():
    paths = (
        "src/glassbox/experimental/affine_anchored_checkpoints.py",
        "tests/test_affine_anchored_checkpoints.py",
        "src/glassbox/experimental/affine_anchored_model.py",
        "tests/test_affine_anchored_model.py",
    )
    return {
        **inherited_source_map(),
        **{path: checkpoints._base._sha(checkpoints.ROOT / path) for path in paths},
    }


@pytest.fixture(scope="module")
def fitted():
    result = {}
    train, dev = windows(1), windows(2)
    original_solve = np.linalg.solve
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original_solve(*args, **kwargs)

    with jax.enable_x64(True), pytest.MonkeyPatch.context() as patch:
        patch.setattr(np.linalg, "solve", counted)
        for arm in ("baseline", "quadratic", "candidate"):
            function = checkpoints._arm(arm)[0]
            calls.clear()
            plain, plain_report = function(
                train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
            )
            plain_solves = len(calls)
            calls.clear()
            previous = sys.gettrace()
            with checkpoints.capture_fit(
                arm, PROVENANCE, source_sha256=source_map(), expected_steps=(0, 1, 2)
            ) as captured:
                model, report = function(
                    train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
                )
            assert sys.gettrace() is previous
            assert len(calls) == plain_solves == (1 if arm == "baseline" else 2)
            result[arm] = dict(
                train=train,
                dev=dev,
                plain=plain,
                plain_report=plain_report,
                model=model,
                report=report,
                captured=captured,
            )
    return result


def save_fitted(path, arm, fitted):
    values = fitted[arm]
    recipe = {**TOY_RECIPE, "id": arm}
    evidence = checkpoints.save_checkpoints(
        path,
        values["captured"],
        train=values["train"],
        development=values["dev"],
        recipe=recipe,
        selected_step=values["report"]["selected_step"],
        selected_model=values["model"],
    )
    arguments = dict(
        train=values["train"],
        development=values["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        selected_model=values["model"],
        expectation=dict(
            arm=arm,
            recipe=recipe,
            expected_steps=[0, 1, 2],
            completed=True,
            failure=None,
            selected_step=values["report"]["selected_step"],
            trace=values["report"]["trace"],
            status="complete",
        ),
    )
    return evidence, arguments


@pytest.mark.parametrize("arm", ("baseline", "quadratic", "candidate"))
def test_exact_plain_instrumented_parity_and_actual_current_snapshots(fitted, arm):
    values = fitted[arm]
    assert_model_equal(values["plain"], values["model"])
    assert values["plain_report"] == values["report"]
    captured = values["captured"]
    assert captured.completed and captured.calls == 1
    assert [s["step"] for s in captured.snapshots] == [0, 1, 2]
    for snap, trace in zip(captured.snapshots, values["report"]["trace"], strict=True):
        assert snap["selection_loss"] == trace["validation_rollout_mse"]
    assert any(
        not np.array_equal(value, captured.snapshots[-1]["params"][key])
        for key, value in captured.snapshots[0]["params"].items()
    )
    if arm != "baseline":
        witness = captured.initializer_observation["witness"]
        assert captured.initializer_observation["calls"] == 1
        assert witness["stage"] == "joint_coefficients_installed"
        initial = np.vstack(
            [captured.snapshots[0]["params"][key] for key in checkpoints.BLOCKS]
        )
        np.testing.assert_array_equal(witness["coefficients"], initial)
        assert not witness["precursor"]["linear"].flags.writeable
    assert checkpoints.historical._arm is not checkpoints._arm


@pytest.mark.parametrize("arm", ("baseline", "quadratic", "candidate"))
def test_nested_replay_no_solve_and_coefficient_diagnostics(
    tmp_path, fitted, arm, monkeypatch
):
    path = tmp_path / arm
    evidence, arguments = save_fitted(path, arm, fitted)
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: pytest.fail("extra solve"))
    replay = checkpoints.replay_checkpoints(path, **arguments)
    assert replay["exact"] and replay["coefficient_checkpoints"] == 3
    assert not replay["normal_equations_resolved"]
    monkeypatch.setattr(
        checkpoints._base, "_diagnostics", lambda *a: pytest.fail("link check rollout")
    )
    assert checkpoints.check_checkpoint_links(path, **arguments)["exact"]
    assert evidence["manifest_sha256"] == checkpoints._base._sha(path / "manifest.json")
    meta, arrays = load_arrays(path / "initializer-witness.npz")
    witness = meta["witness"]
    assert witness["applied_prior"] == ("affine" if arm == "candidate" else "zero")
    if arm != "baseline":
        assert witness["backward_residual"]["relative_backward_residual"] < 1e-14
        reference = arrays["common_affine_reference"]
        np.testing.assert_array_equal(
            reference[: len(arrays["affine_linear"])], arrays["affine_linear"]
        )
        expected = reference if arm == "candidate" else np.zeros_like(reference)
        np.testing.assert_array_equal(arrays["applied_prior"], expected)
        if arm == "candidate":
            assert set(witness["captured_fingerprints"]) == {
                "D",
                "Y",
                "penalty",
                "A",
                "B0",
                "W_affine",
                "centered_rhs",
            }
        else:
            assert set(witness["captured_fingerprints"]) == {"D", "Y", "penalty"}
            assert "not captured" in witness["matrix_capture_scope"]
    rows = json.loads((path / "coefficient-diagnostics.json").read_text())[
        "checkpoints"
    ]
    if arm == "baseline":
        assert rows[0]["blocks"]["linear"]["displacement_frobenius"] == 0
        assert not rows[0]["blocks"]["autonomous"]["present"]
    else:
        assert rows[0]["blocks"]["linear"]["displacement_frobenius"] > 0
        assert rows[0]["blocks"]["autonomous"]["relative_displacement"] is None


def reseal(path):
    file = path / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["files"] = {
        str(p.relative_to(path)): checkpoints._base._sha(p)
        for p in path.rglob("*")
        if p.is_file() and p != file
    }
    file.write_text(json.dumps(manifest))
    return checkpoints._base._sha(file)


@pytest.mark.parametrize(
    "change",
    (
        "missing_rhs",
        "changed_rhs",
        "changed_prior",
        "changed_lambda",
        "changed_precursor",
        "changed_stage",
    ),
)
def test_coherent_witness_tamper_rejected(tmp_path, fitted, change):
    path = tmp_path / "candidate"
    _, arguments = save_fitted(path, "candidate", fitted)
    file = path / "initializer-witness.npz"
    meta, arrays = load_arrays(file)
    witness = meta["witness"]
    if change == "missing_rhs":
        witness["captured_fingerprints"].pop("centered_rhs")
        witness["captured_local_names"].pop("centered_rhs")
    elif change == "changed_rhs":
        witness["captured_fingerprints"]["centered_rhs"] = witness[
            "captured_fingerprints"
        ]["B0"]
    elif change == "changed_prior":
        arrays["applied_prior"][:] = 0
    elif change == "changed_lambda":
        witness["settings"]["ridge"] *= 2
    elif change == "changed_precursor":
        arrays["affine_linear"][0, 0] += 1
    else:
        witness["stage"] = "before_existing_solve"
    save_arrays(file, meta, arrays)
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.check_checkpoint_links(
            path, **{**arguments, "expected_sha256": reseal(path)}
        )


def test_cross_arm_exact_initialization_and_missing_failed_witness(tmp_path, fitted):
    paths = {arm: tmp_path / arm for arm in fitted}
    for arm, path in paths.items():
        save_fitted(path, arm, fitted)
    report = checkpoints.check_shared_initialization(paths)
    assert report["all_available_precursors_exact"] and len(report["comparisons"]) == 3
    assert report["additional_fits_or_solves"] == 0
    assert not report["unavailable"]
    file = paths["candidate"] / "initializer-witness.npz"
    meta, payload = load_arrays(file)
    payload["affine_linear"][0, 0] += 1
    save_arrays(file, meta, payload)
    reseal(paths["candidate"])
    with pytest.raises(
        checkpoints.CheckpointError, match="actual affine precursor differs"
    ):
        checkpoints.check_shared_initialization(paths)


def failed_bundle(path, arm="candidate"):
    previous = sys.gettrace()

    def tracer(frame, event, argument):
        return tracer

    try:
        sys.settrace(tracer)
        with pytest.raises(RuntimeError, match="preparation unavailable"):
            with checkpoints.capture_fit(
                arm, PROVENANCE, source_sha256=source_map(), expected_steps=(0, 1, 2)
            ) as captured:
                raise RuntimeError("preparation unavailable")
        assert sys.gettrace() is tracer
    finally:
        sys.settrace(previous)
    failure = {"preparation_unavailable": True}
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=windows(1),
        development=windows(2),
        recipe=TOY_RECIPE,
        failure=failure,
    )
    arguments = dict(
        train=windows(1),
        development=windows(2),
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        expectation=dict(
            arm=arm,
            recipe=TOY_RECIPE,
            expected_steps=[0, 1, 2],
            completed=False,
            failure=failure,
            selected_step=None,
            trace=None,
            status="preparation_unavailable",
        ),
    )
    return captured, arguments


def test_restore_failure_no_call_and_available_pairs_still_checked(tmp_path, fitted):
    paths = {arm: tmp_path / arm for arm in fitted}
    captured, arguments = failed_bundle(paths["candidate"])
    assert captured.calls == 0 and captured.initializer_observation["calls"] == 0
    evidence = checkpoints.replay_checkpoints(paths["candidate"], **arguments)
    assert not evidence["initializer_available"] and evidence["checkpoints"] == 0
    for arm in ("baseline", "quadratic"):
        save_fitted(paths[arm], arm, fitted)
    report = checkpoints.check_shared_initialization(paths)
    assert report["available_arms"] == ["baseline", "quadratic"]
    assert list(report["unavailable"]) == ["candidate"]
    assert len(report["comparisons"]) == 1


def test_external_anchor_cache_provenance_and_source_tamper(tmp_path, fitted):
    path = tmp_path / "candidate"
    _, arguments = save_fitted(path, "candidate", fitted)
    for change in (
        {"train": windows(8)},
        {"provenance": {"changed": True}},
        {"source_sha256": {}},
        {"expected_sha256": "0" * 64},
    ):
        with pytest.raises(checkpoints.CheckpointError):
            checkpoints.check_checkpoint_links(path, **{**arguments, **change})
    (path / "initializer-witness.npz").unlink()
    with pytest.raises(checkpoints.CheckpointError, match="inventory"):
        checkpoints.check_checkpoint_links(path, **arguments)


def test_candidate_numeric_failure_preserves_actual_initializer_and_prefix(tmp_path):
    train, dev = windows(1), windows(2)
    previous = sys.gettrace()
    with pytest.raises(ValueError, match="nonfinite"):
        with checkpoints.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as captured:
            checkpoints._arm("candidate")[0](
                train.batch,
                dev.batch,
                **{**SETTINGS, "learning_rate": 1e200},
                error_scale=toy_scale(train),
            )
    assert sys.gettrace() is previous
    assert [snap["step"] for snap in captured.snapshots] == [0]
    failure = captured.failure["message"]
    recipe = {**TOY_RECIPE, "learning_rate": 1e200}
    path = tmp_path / "partial"
    evidence = checkpoints.save_checkpoints(
        path, captured, train=train, development=dev, recipe=recipe, failure=failure
    )
    replay = checkpoints.replay_checkpoints(
        path,
        train=train,
        development=dev,
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        expectation=dict(
            arm="candidate",
            recipe=recipe,
            expected_steps=[0, 1, 2],
            completed=False,
            failure=failure,
            selected_step=None,
            trace=None,
            status="fit_failure",
        ),
    )
    assert replay["initializer_available"] and replay["checkpoints"] == 1


def test_failed_joint_solve_retains_actual_input_witness_without_coefficients(
    tmp_path, monkeypatch
):
    train, dev = windows(1), windows(2)
    solve = np.linalg.solve
    calls = []

    def fail_joint(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise np.linalg.LinAlgError("toy joint solve failure")
        return solve(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", fail_joint)
    with pytest.raises(np.linalg.LinAlgError, match="toy joint solve failure"):
        with checkpoints.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as captured:
            checkpoints._arm("candidate")[0](
                train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
            )
    assert not captured.snapshots and len(calls) == 2
    path = tmp_path / "failed-solve"
    failure = captured.failure["message"]
    evidence = checkpoints.save_checkpoints(
        path, captured, train=train, development=dev, recipe=TOY_RECIPE, failure=failure
    )
    replay = checkpoints.replay_checkpoints(
        path,
        train=train,
        development=dev,
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        expectation=dict(
            arm="candidate",
            recipe=TOY_RECIPE,
            expected_steps=[0, 1, 2],
            completed=False,
            failure=failure,
            selected_step=None,
            trace=None,
            status="fit_failure",
        ),
    )
    assert (
        replay["initializer_available"]
        and replay["checkpoints"] == 0
        and len(calls) == 2
    )
    metadata, arrays = load_arrays(path / "initializer-witness.npz")
    assert metadata["witness"]["stage"] == "before_existing_solve"
    assert not metadata["witness"]["backward_residual"]["available"]
    assert "initial_joint_coefficients" not in arrays


def test_empty_capture_and_changed_initializer_code_rejected(monkeypatch):
    previous = sys.gettrace()
    with pytest.raises(checkpoints.CheckpointError, match="missed initializer"):
        with checkpoints.capture_fit(
            "candidate", PROVENANCE, source_sha256=source_map()
        ):
            pass
    assert sys.gettrace() is previous
    module = checkpoints._candidate()
    original = module.initialize_candidate
    changed = original.__code__.replace(
        co_consts=(*original.__code__.co_consts, "changed")
    )
    monkeypatch.setattr(original, "__code__", changed)
    with pytest.raises(checkpoints.CheckpointError, match="differs from pinned source"):
        checkpoints._initializer_binding("candidate")
