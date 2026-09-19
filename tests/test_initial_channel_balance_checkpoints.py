"""Toy-only exact observer parity, weighted evidence replay and failure integrity."""

import json
import sys
from dataclasses import replace

import jax
import numpy as np
import pytest
from test_affine_anchored_checkpoints import source_map as anchored_sources
from test_fit_checkpoints import (
    PROVENANCE,
    SETTINGS,
    TOY_RECIPE,
    assert_model_equal,
    toy_scale,
    windows,
)

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.experimental import initial_channel_balance_checkpoints as checkpoints
from glassbox.experimental import initial_channel_balance_model as model


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def source_map():
    paths = (
        "src/glassbox/experimental/initial_channel_balance_checkpoints.py",
        "src/glassbox/experimental/initial_channel_balance_model.py",
        "tests/test_initial_channel_balance_checkpoints.py",
        "tests/test_initial_channel_balance_model.py",
    )
    return {
        **anchored_sources(),
        **{path: checkpoints._sha(checkpoints.ROOT / path) for path in paths},
    }


def reference_for(path, snapshot):
    save_arrays(
        path,
        {"snapshot": checkpoints._base._snapshot_metadata(snapshot)},
        checkpoints._base._snapshot_arrays(snapshot),
    )
    return dict(
        path=path,
        sha256=checkpoints._sha(path),
        source_bundle_sha256="a" * 64,
        source_checkpoint_manifest_sha256="b" * 64,
        source_relative_path="toy/candidate/checkpoints/trajectory/step-0000.npz",
    )


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    train, dev = windows(1), windows(2)

    def with_residuals(data, seed):
        rng = np.random.default_rng(seed)
        target = data.batch.future_states
        target = target + rng.normal(size=target.shape) * (0.003 * np.arange(1, 16))
        return replace(data, batch=replace(data.batch, future_states=target))

    train, dev = with_residuals(train, 3), with_residuals(dev, 4)
    with jax.enable_x64(True):
        old = checkpoints.inherited
        with old.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as anchor_capture:
            old._arm("candidate")[0](
                train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
            )
        reference = reference_for(
            tmp_path_factory.mktemp("balanced-anchor") / "step0.npz",
            anchor_capture.snapshots[0],
        )
        solve = np.linalg.solve
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return solve(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(np.linalg, "solve", counted)
            plain, plain_report = model.fit_candidate_sequence(
                train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
            )
            plain_calls = len(calls)
            calls.clear()
            previous = sys.gettrace()
            with checkpoints.capture_fit(
                "candidate",
                PROVENANCE,
                source_sha256=source_map(),
                expected_steps=(0, 1, 2),
            ) as captured:
                observed, report = model.fit_candidate_sequence(
                    train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
                )
            assert sys.gettrace() is previous
            assert plain_calls == len(calls) == 2
    return dict(
        train=train,
        dev=dev,
        reference=reference,
        plain=plain,
        plain_report=plain_report,
        observed=observed,
        report=report,
        captured=captured,
    )


def save_fitted(path, fitted):
    evidence = checkpoints.save_checkpoints(
        path,
        fitted["captured"],
        train=fitted["train"],
        development=fitted["dev"],
        recipe=TOY_RECIPE,
        anchor_reference=fitted["reference"],
        selected_step=fitted["report"]["selected_step"],
        selected_model=fitted["observed"],
    )
    kwargs = dict(
        train=fitted["train"],
        development=fitted["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        anchor_reference=fitted["reference"],
        selected_model=fitted["observed"],
        expectation=dict(
            arm="candidate",
            recipe=TOY_RECIPE,
            expected_steps=[0, 1, 2],
            completed=True,
            failure=None,
            selected_step=fitted["report"]["selected_step"],
            trace=fitted["report"]["trace"],
            unweighted_trace=fitted["report"]["unweighted_development_trace"],
            objective=fitted["report"]["objective"],
            status="complete",
        ),
    )
    return evidence, kwargs


def test_exact_fitter_parity_actual_weights_and_original_scalars(fitted):
    assert_model_equal(fitted["plain"], fitted["observed"])
    assert fitted["plain_report"] == fitted["report"]
    captured = fitted["captured"]
    assert captured.completed and captured.calls == 1
    state = captured.weight_observation
    assert state["forecast_calls"] == state["boundary_calls"] == 1
    assert state["available"] and all(
        not value.flags.writeable for value in state["arrays"].values()
    )
    for snapshot, weighted, original in zip(
        captured.snapshots,
        fitted["report"]["trace"],
        fitted["report"]["unweighted_development_trace"],
        strict=True,
    ):
        assert snapshot["selection_loss"] == weighted["validation_rollout_mse"]
        assert (
            snapshot["observed_unweighted_loss"] == original["validation_rollout_mse"]
        )
    assert any(
        not np.array_equal(value, captured.snapshots[-1]["params"][key])
        for key, value in captured.snapshots[0]["params"].items()
    )
    assert not np.array_equal(state["arrays"]["channel_weights"], np.ones(15))
    np.testing.assert_array_equal(
        state["arrays"]["channel_weights"], state["arrays"]["fixed_weights"]
    )


def test_save_full_replay_without_solve_and_cheap_links_without_forecast(
    tmp_path, fitted, monkeypatch
):
    path = tmp_path / "checkpoints"
    evidence, kwargs = save_fitted(path, fitted)
    monkeypatch.setattr(
        np.linalg, "solve", lambda *args: pytest.fail("replay solved ridge")
    )
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    assert (
        replay["exact"] and replay["checkpoints"] == replay["training_checkpoints"] == 3
    )
    assert replay["initial_identity"]["actual_initial_forecast_replayed"]
    assert not replay["initializers_or_optimizers_reexecuted"]
    assert evidence["initial_identity"]["exact"]
    meta, arrays = load_arrays(path / "step-0000-training.npz")
    assert len(arrays["original_channel_loss_contributions"]) == 15
    assert np.isclose(
        arrays["channel_loss_contributions"].sum(),
        meta["diagnostics"]["reconstructed_selection_loss"],
        atol=1e-12,
    )
    # Cheap checks reduce saved training predictions, but never execute a forecast.
    monkeypatch.setattr(
        jax,
        "jit",
        lambda *args, **kwargs: pytest.fail("cheap check compiled a rollout"),
    )
    assert checkpoints.check_checkpoint_links(path, **kwargs)["exact"]
    meta, _ = load_arrays(path / "objective-witness.npz")
    assert "path" not in meta["witness"]["anchor_reference"]
    selection = json.loads((path / "selection-evidence.json").read_text())
    assert selection["actual_selected_step"] == fitted["report"]["selected_step"]


def reseal(path):
    file = path / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["files"] = {
        str(item.relative_to(path)): checkpoints._sha(item)
        for item in path.rglob("*")
        if item.is_file() and item != file
    }
    file.write_text(json.dumps(manifest))
    return checkpoints._sha(file)


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "weights",
        "fixed_weights",
        "mse",
        "floor",
        "prediction",
        "normalizer",
        "initial",
        "reference",
    ),
)
def test_objective_witness_coherent_tamper_rejected(tmp_path, fitted, change):
    path = tmp_path / "checkpoints"
    _, kwargs = save_fitted(path, fitted)
    file = path / "objective-witness.npz"
    metadata, arrays = load_arrays(file)
    if change == "missing":
        arrays.pop("raw_channel_weights")
    elif change == "reference":
        metadata["witness"]["anchor_reference"]["source_bundle_sha256"] = "0" * 64
    else:
        key = dict(
            weights="channel_weights",
            fixed_weights="fixed_weights",
            mse="initial_channel_mse",
            floor="weight_floor",
            prediction="initial_training_prediction",
            normalizer="weight_normalizer",
            initial="param_bias",
        )[change]
        arrays[key].flat[0] += 0.1
    save_arrays(file, metadata, arrays)
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.check_checkpoint_links(
            path, **{**kwargs, "expected_sha256": reseal(path)}
        )


def test_external_report_objective_trace_and_anchor_bindings(tmp_path, fitted):
    path = tmp_path / "checkpoints"
    _, kwargs = save_fitted(path, fitted)
    for name in ("objective", "unweighted_trace"):
        expected = json.loads(json.dumps(kwargs["expectation"]))
        if name == "objective":
            expected[name]["channel_weights"][0] *= 2
        else:
            expected[name][0]["validation_rollout_mse"] += 1
        with pytest.raises(checkpoints.CheckpointError):
            checkpoints.check_checkpoint_links(
                path, **{**kwargs, "expectation": expected}
            )
    for changes in (
        {"development": windows(8)},
        {"provenance": {"different": True}},
        {"source_sha256": {}},
        {"anchor_reference": {**fitted["reference"], "sha256": "0" * 64}},
    ):
        with pytest.raises(checkpoints.CheckpointError):
            checkpoints.check_checkpoint_links(path, **{**kwargs, **changes})


def failed_kwargs(path, fitted, captured, recipe, status):
    failure = captured.failure["message"]
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=fitted["train"],
        development=fitted["dev"],
        recipe=recipe,
        anchor_reference=fitted["reference"],
        failure=failure,
    )
    return dict(
        train=fitted["train"],
        development=fitted["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        anchor_reference=fitted["reference"],
        expectation=dict(
            arm="candidate",
            recipe=recipe,
            expected_steps=[0, 1, 2],
            completed=False,
            failure=failure,
            selected_step=None,
            trace=None,
            unweighted_trace=None,
            objective=None,
            status=status,
        ),
    )


def test_preparation_failure_and_tracer_restoration(tmp_path, fitted):
    previous = sys.gettrace()

    def tracer(frame, event, argument):
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
    path = tmp_path / "failed"
    kwargs = failed_kwargs(
        path, fitted, captured, TOY_RECIPE, "preparation_unavailable"
    )
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    assert not replay["weighting_available"] and replay["checkpoints"] == 0
    assert captured.weight_observation["forecast_calls"] == 0


@pytest.mark.parametrize("stage", ("weighting", "initial_development", "training"))
def test_numerical_failure_preserves_reached_weight_witness(
    tmp_path, fitted, monkeypatch, stage
):
    settings = dict(SETTINGS)
    scale = toy_scale(fitted["train"])
    if stage == "weighting":
        scale *= 1e-200
    elif stage == "training":
        settings["learning_rate"] = 1e200
    else:
        original_jit = jax.jit

        def replace_evaluate(function, *args, **kwargs):
            if function.__name__ == "evaluate":
                return lambda params: (np.inf, np.inf)
            return original_jit(function, *args, **kwargs)

        monkeypatch.setattr(jax, "jit", replace_evaluate)
    previous = sys.gettrace()
    with pytest.raises(ValueError, match="nonfinite"):
        with checkpoints.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as captured:
            model.fit_candidate_sequence(
                fitted["train"].batch,
                fitted["dev"].batch,
                **settings,
                error_scale=scale,
            )
    assert sys.gettrace() is previous
    monkeypatch.undo()
    recipe = {**TOY_RECIPE, **({"learning_rate": 1e200} if stage == "training" else {})}
    path = tmp_path / "failed"
    kwargs = failed_kwargs(path, fitted, captured, recipe, "fit_failure")
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    assert replay["weighting_available"] == (stage != "weighting")
    assert replay["checkpoints"] == (1 if stage == "training" else 0)
