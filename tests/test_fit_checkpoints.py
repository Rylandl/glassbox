"""Bounded toy optimizer parity and external checkpoint integrity witnesses."""

import hashlib
import json
import sys
from dataclasses import replace

import jax
import numpy as np
import pytest

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox._sequence_model import SequenceBatch
from glassbox.experimental import fit_checkpoints as checkpoints
from glassbox.recordings import SequenceWindows, WindowKey


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def source_map():
    paths = [
        "src/glassbox/experimental/fit_checkpoints.py",
        "tests/test_fit_checkpoints.py",
        "src/glassbox/_sequence_model.py",
        "src/glassbox/experimental/state_input_model.py",
        "src/glassbox/experimental/state_quadratic_model.py",
    ]
    return {path: checkpoints._sha(checkpoints.ROOT / path) for path in paths}


def windows(seed):
    rng = np.random.default_rng(seed)
    command = rng.normal(size=(12, 9, 2))
    state = np.empty((12, 10, 15))
    state[:, 0] = rng.normal(size=(12, 15))
    force = rng.normal(size=(2, 15)) * 0.03
    for i in range(9):
        state[:, i + 1] = 0.96 * state[:, i] + command[:, i] @ force
    batch = SequenceBatch(
        state[:, :5], command[:, :4], command[:, 4:], state[:, 5:], 0.05
    )
    return SequenceWindows(
        batch,
        tuple(WindowKey(f"parent-{i % 3}", "seg", i + 4) for i in range(12)),
        tuple(range(4, 16)),
    )


PROVENANCE = {"protocol_sha256": "a" * 64, "runtime": {"toy": True}, "stage": "fit"}
SETTINGS = dict(
    steps=2, check_every=1, width=3, memory=2, delay_steps=1, batch_size=6, ridge=1.0
)
TOY_RECIPE = dict(
    id="toy",
    kind="filter_mlp",
    seed=0,
    steps=2,
    check_every=1,
    width=3,
    memory=2,
    batch_size=6,
    learning_rate=0.002,
    ridge_fraction=1 / 60,
    delay_s=0.05,
    context_s=0.2,
    horizon_s=0.25,
    training_windows=12,
    development_windows=12,
    hold_scale_floor=0.01,
)


def toy_scale(train):
    batch = train.batch
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    scale = physical[:, 4:-1].std((0, 1))
    scale = np.where(scale > 1e-8, scale, 1)
    hold = np.repeat(batch.past_states[:, -1:], 5, axis=1)
    return np.maximum(
        np.sqrt(np.mean((hold - batch.future_states) ** 2, axis=0)), 0.01 * scale
    )


@pytest.fixture(scope="module", params=("baseline", "candidate", "quadratic"))
def fitted(request):
    arm = request.param
    train, development = windows(1), windows(2)
    function = checkpoints._arm(arm)[0]
    with jax.enable_x64(True):
        plain, plain_report = function(
            train.batch, development.batch, **SETTINGS, error_scale=toy_scale(train)
        )
        with checkpoints.capture_fit(
            arm, PROVENANCE, source_sha256=source_map(), expected_steps=(0, 1, 2)
        ) as captured:
            observed, observed_report = function(
                train.batch, development.batch, **SETTINGS, error_scale=toy_scale(train)
            )
    return (
        arm,
        train,
        development,
        plain,
        plain_report,
        observed,
        observed_report,
        captured,
    )


def assert_model_equal(left, right):
    assert (left.kind, left.dt_s, left.history_steps, left.delay_steps) == (
        right.kind,
        right.dt_s,
        right.history_steps,
        right.delay_steps,
    )
    for group in ("params", "norms"):
        assert getattr(left, group).keys() == getattr(right, group).keys()
        for key, value in getattr(left, group).items():
            np.testing.assert_array_equal(value, getattr(right, group)[key])


def test_actual_optimizer_parity_all_three_arms(fitted):
    arm, train, dev, plain, report, observed, observed_report, captured = fitted
    assert report == observed_report
    assert_model_equal(plain, observed)
    assert captured.calls == 1 and captured.completed and captured.failure is None
    assert [s["step"] for s in captured.snapshots] == [0, 1, 2]
    assert [s["selection_loss"] for s in captured.snapshots] == [
        row["validation_rollout_mse"] for row in report["trace"]
    ]
    for snapshot in captured.snapshots:
        assert snapshot["train_fingerprint"] == checkpoints._batch_fingerprint(train)
        assert snapshot["development_fingerprint"] == checkpoints._batch_fingerprint(
            dev
        )
        assert all(not value.flags.writeable for value in snapshot["params"].values())
    chosen = captured.snapshots[report["selected_step"]]
    checkpoints._selected(chosen, observed)
    if arm == "quadratic":
        assert "_state_quadratic_shared" in captured.binding["module"]


def save_fitted(path, fitted):
    arm, train, dev, _, _, model, report, captured = fitted
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=train,
        development=dev,
        recipe={**TOY_RECIPE, "id": arm},
        selected_step=report["selected_step"],
        selected_model=model,
    )
    arguments = dict(
        train=train,
        development=dev,
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        selected_model=model,
        expectation=dict(
            arm=arm,
            recipe={**TOY_RECIPE, "id": arm},
            expected_steps=[0, 1, 2],
            completed=True,
            failure=None,
            selected_step=report["selected_step"],
            trace=report["trace"],
            status="complete",
        ),
    )
    return evidence, arguments


def test_save_replay_physical_decomposition_and_cheap_links(
    tmp_path, fitted, monkeypatch
):
    path = tmp_path / "checkpoints"
    evidence, arguments = save_fitted(path, fitted)
    assert evidence["checkpoints"] == 3
    replayed = checkpoints.replay_checkpoints(path, **arguments)
    assert replayed["exact"] and replayed["checkpoints"] == 3
    metadata, arrays = load_arrays(path / "step-0000-development.npz")
    prediction = arrays["prediction"]
    target = fitted[2].batch.future_states
    squared = (prediction - target) ** 2
    np.testing.assert_array_equal(arrays["channel_mse"], squared.mean(0))
    np.testing.assert_allclose(
        arrays["group_endpoint_rmse"][:, 0],
        np.sqrt(squared[:, :, :3].mean((0, 2))),
        rtol=1e-14,
    )
    assert metadata["diagnostics"]["horizon_steps"] == [1, 3, 5]
    assert arrays["parent_endpoint_rmse"].shape == (3, 3, 3)
    assert np.isclose(
        sum(metadata["diagnostics"]["group_loss_contributions"]),
        metadata["diagnostics"]["selection_loss"],
        rtol=1e-10,
        atol=1e-12,
    )
    monkeypatch.setattr(
        checkpoints,
        "_diagnostics",
        lambda *args: pytest.fail("cheap check ran rollout"),
    )
    assert checkpoints.check_checkpoint_links(path, **arguments)["exact"]


def test_snapshot_and_metric_tampering_are_rejected(tmp_path, fitted):
    path = tmp_path / "checkpoints"
    _, arguments = save_fitted(path, fitted)
    snapshot = path / "step-0000.npz"
    metadata, arrays = load_arrays(snapshot)
    arrays["param_bias"][0] += 0.5
    save_arrays(snapshot, metadata, arrays)
    with pytest.raises(checkpoints.CheckpointError, match="payload seal"):
        checkpoints.replay_checkpoints(path, **arguments)
    # Coherently rehashing the local manifest still fails the external anchor.
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][snapshot.name] = checkpoints._sha(snapshot)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(checkpoints.CheckpointError, match="external anchor"):
        checkpoints.replay_checkpoints(path, **arguments)


def test_changed_cache_provenance_source_and_selected_model_rejected(tmp_path, fitted):
    path = tmp_path / "checkpoints"
    _, arguments = save_fitted(path, fitted)
    for changes in (
        {"development": windows(11)},
        {"provenance": {"changed": True}},
        {"source_sha256": {**source_map(), "tests/test_fit_checkpoints.py": "0" * 64}},
    ):
        with pytest.raises(checkpoints.CheckpointError):
            checkpoints.check_checkpoint_links(path, **{**arguments, **changes})
    model = arguments["selected_model"]
    params = {k: np.array(v, copy=True) for k, v in model.params.items()}
    params["bias"][0] += 1
    changed = replace(model, params=params)
    with pytest.raises(checkpoints.CheckpointError, match="selected model differs"):
        checkpoints.check_checkpoint_links(
            path, **{**arguments, "selected_model": changed}
        )
    (path / "extra.txt").write_text("extra")
    with pytest.raises(checkpoints.CheckpointError, match="inventory"):
        checkpoints.check_checkpoint_links(path, **arguments)


def test_restores_previous_tracer_and_records_zero_call_failure(tmp_path):
    previous = sys.gettrace()

    def tracer(frame, event, arg):
        return tracer

    try:
        sys.settrace(tracer)
        with pytest.raises(RuntimeError, match="preparation unavailable"):
            with checkpoints.capture_fit(
                "candidate",
                PROVENANCE,
                source_sha256=source_map(),
                expected_steps=(0, 1, 2),
            ) as captured:
                raise RuntimeError("preparation unavailable")
        assert sys.gettrace() is tracer
    finally:
        sys.settrace(previous)
    assert captured.calls == 0 and not captured.snapshots
    path = tmp_path / "failed"
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=windows(1),
        development=windows(2),
        recipe=TOY_RECIPE,
        failure={"preparation_unavailable": True},
    )
    result = checkpoints.replay_checkpoints(
        path,
        train=windows(1),
        development=windows(2),
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        expectation=dict(
            arm="candidate",
            recipe=TOY_RECIPE,
            expected_steps=[0, 1, 2],
            completed=False,
            failure={"preparation_unavailable": True},
            selected_step=None,
            trace=None,
            status="preparation_unavailable",
        ),
    )
    assert result["checkpoints"] == 0 and result["unavailable_checkpoints"] == 3
    assert not result["optimizer_attempted"]


def test_missing_fit_or_checkpoint_fails_and_restores_tracer():
    previous = sys.gettrace()
    with pytest.raises(checkpoints.CheckpointError, match="missed required"):
        with checkpoints.capture_fit(
            "baseline", PROVENANCE, source_sha256=source_map()
        ):
            pass
    assert sys.gettrace() is previous
    with pytest.raises(checkpoints.CheckpointError, match="missed required"):
        with checkpoints.capture_fit(
            "baseline", PROVENANCE, source_sha256=source_map(), expected_steps=(0, 1)
        ):
            checkpoints.public.fit_sequence_model(
                windows(1).batch, windows(2).batch, **{**SETTINGS, "steps": 0}
            )
    assert sys.gettrace() is previous


def test_wrong_source_code_and_missing_map_rejected(monkeypatch):
    with pytest.raises(checkpoints.CheckpointError, match="required modules"):
        with checkpoints.capture_fit("baseline", PROVENANCE, source_sha256={}):
            pass
    function = checkpoints.public.fit_sequence_model
    monkeypatch.setattr(checkpoints.public, "fit_sequence_model", lambda *a, **k: None)
    with pytest.raises(checkpoints.CheckpointError, match="AST identity"):
        checkpoints._binding("baseline")
    monkeypatch.setattr(checkpoints.public, "fit_sequence_model", function)


def test_capture_is_current_tree_not_best_tree(fitted):
    *_, captured = fitted
    # Existing finite checkpoints observe current params even before best selection.
    # Independently reconstruct the final chronological snapshot's loss and compare
    # it to that trace entry, without relying on its selection status.
    _, cls, _ = checkpoints._arm(fitted[0])
    snapshot = captured.snapshots[-1]
    model = cls(
        snapshot["kind"],
        snapshot["dt_s"],
        snapshot["history_steps"],
        snapshot["params"],
        snapshot["norms"],
        snapshot["delay_steps"],
    )
    b = fitted[2].batch
    predicted = np.asarray(model.rollout(b.past_states, b.past_inputs, b.future_inputs))
    loss = float(
        np.mean(((predicted - b.future_states) / snapshot["error_scale"]) ** 2)
    )
    assert np.isclose(
        loss, fitted[6]["trace"][-1]["validation_rollout_mse"], rtol=1e-10, atol=1e-12
    )
    assert any(
        not np.array_equal(
            snapshot["params"][key], captured.snapshots[0]["params"][key]
        )
        for key in snapshot["params"]
    )


def test_portable_code_identity_omits_absolute_worktree():
    identity = checkpoints._binding("baseline")[3]
    assert identity["path"] == "src/glassbox/_sequence_model.py"
    assert len(identity["code_sha256"]) == len(hashlib.sha256().hexdigest())


def test_actual_current_checkpoint_can_be_worse_than_saved_best():
    train = windows(1)
    with checkpoints.capture_fit(
        "baseline", PROVENANCE, source_sha256=source_map(), expected_steps=(0, 1)
    ) as captured:
        model, report = checkpoints.public.fit_sequence_model(
            train.batch, train.batch, **{**SETTINGS, "steps": 1, "learning_rate": 0.5}
        )
    assert report["selected_step"] == 0
    assert (
        captured.snapshots[1]["selection_loss"]
        > captured.snapshots[0]["selection_loss"]
    )
    checkpoints._selected(captured.snapshots[0], model)
    with pytest.raises(checkpoints.CheckpointError, match="selected model differs"):
        checkpoints._selected(captured.snapshots[1], model)


def test_numeric_abort_preserves_partial_prefix_and_replays_without_fit(
    tmp_path, monkeypatch
):
    train, dev = windows(1), windows(2)
    prior = sys.gettrace()
    with pytest.raises(ValueError, match="nonfinite sequence training"):
        with checkpoints.capture_fit(
            "baseline", PROVENANCE, source_sha256=source_map(), expected_steps=(0, 1, 2)
        ) as captured:
            checkpoints.public.fit_sequence_model(
                train.batch,
                dev.batch,
                **{**SETTINGS, "learning_rate": 1e200},
                error_scale=toy_scale(train),
            )
    assert sys.gettrace() is prior
    assert [s["step"] for s in captured.snapshots] == [0]
    assert captured.calls == 1 and not captured.completed
    reason = captured.failure["message"]
    path = tmp_path / "partial"
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=train,
        development=dev,
        recipe={**TOY_RECIPE, "learning_rate": 1e200},
        failure=reason,
    )
    # Reproduction must not initialize or fit; the canonical function identity is
    # still checked, so disable initialization instead of replacing that function.
    monkeypatch.setattr(
        checkpoints.public,
        "initialize_sequence_model",
        lambda *a, **k: pytest.fail("replay attempted initialization"),
    )
    replay = checkpoints.replay_checkpoints(
        path,
        train=train,
        development=dev,
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        expectation=dict(
            arm="baseline",
            recipe={**TOY_RECIPE, "learning_rate": 1e200},
            expected_steps=[0, 1, 2],
            completed=False,
            failure=reason,
            selected_step=None,
            trace=None,
            status="fit_failure",
        ),
    )
    assert replay["checkpoints"] == 1 and replay["unavailable_checkpoints"] == 2


def reseal(path, change):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    change(manifest)
    manifest_path.write_text(json.dumps(manifest))
    return checkpoints._sha(manifest_path)


@pytest.mark.parametrize(
    "field,changed",
    [
        ("expected_steps", [0]),
        ("recipe", {"renamed": True}),
        ("completed", False),
        ("selected_step", 999),
    ],
)
def test_external_expectations_reject_coherently_resealed_roster_and_outcome(
    tmp_path, fitted, field, changed
):
    path = tmp_path / "checkpoints"
    _, arguments = save_fitted(path, fitted)
    new_anchor = reseal(path, lambda manifest: manifest.update({field: changed}))
    with pytest.raises(checkpoints.CheckpointError, match="external"):
        checkpoints.check_checkpoint_links(
            path, **{**arguments, "expected_sha256": new_anchor}
        )


def test_original_report_trace_and_recomputed_diagnostic_are_independent_checks(
    tmp_path, fitted
):
    path = tmp_path / "checkpoints"
    _, arguments = save_fitted(path, fitted)
    changed = json.loads(json.dumps(arguments["expectation"]))
    changed["trace"][0]["validation_rollout_mse"] += 1
    with pytest.raises(checkpoints.CheckpointError, match="trace differs"):
        checkpoints.check_checkpoint_links(
            path, **{**arguments, "expectation": changed}
        )
    file = path / "step-0000-development.npz"
    metadata, arrays = load_arrays(file)
    arrays["channel_mse"][0, 0] += 1
    save_arrays(file, metadata, arrays)
    anchor = reseal(
        path,
        lambda manifest: manifest["files"].update({file.name: checkpoints._sha(file)}),
    )
    with pytest.raises(checkpoints.CheckpointError, match="diagnostic replay differs"):
        checkpoints.replay_checkpoints(path, **{**arguments, "expected_sha256": anchor})


@pytest.mark.parametrize("changed", ["settings", "normalization", "loss_scale"])
def test_resealed_snapshot_settings_and_normalization_are_bound_to_recipe_and_cache(
    tmp_path, fitted, changed
):
    path = tmp_path / "checkpoints"
    _, arguments = save_fitted(path, fitted)
    # Avoid the selected snapshot: constant normalization must be checked too.
    step = next(s for s in (0, 1, 2) if s != fitted[6]["selected_step"])
    file = path / f"step-{step:04d}.npz"
    metadata, arrays = load_arrays(file)
    if changed == "settings":
        metadata["snapshot"]["settings"]["learning_rate"] *= 2
    elif changed == "normalization":
        arrays["norm_input_mean"][0] += 0.1
    else:
        arrays["error_scale"] *= 2
    save_arrays(file, metadata, arrays)
    anchor = reseal(
        path,
        lambda manifest: manifest["files"].update({file.name: checkpoints._sha(file)}),
    )
    with pytest.raises(
        checkpoints.CheckpointError, match=r"settings|normalization|loss scale"
    ):
        checkpoints.check_checkpoint_links(
            path, **{**arguments, "expected_sha256": anchor}
        )
