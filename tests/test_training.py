"""Small analytic fits exercise the fixed recipe without simulator calls."""

from dataclasses import replace
from itertools import pairwise

import jax
import numpy as np
import pytest

from glassbox import _dynamics as core
from glassbox._training import SequenceBatch, SequenceFitError


def analytic_batch(commands=1):
    n, context, horizon, dt = 4, 3, 3, 0.05
    rng = np.random.default_rng(42)
    command = rng.normal(size=(n, context + horizon, commands)) * 0.3
    states = np.zeros((n, context + horizon + 1, 15))
    states[..., 6:] = np.eye(3).reshape(9)
    for t in range(context + horizon):
        acceleration = np.zeros((n, 3))
        acceleration[:, 0] = command[:, t].sum(-1)
        acceleration[:, 2] = core.GRAVITY[2]
        states[:, t + 1, :3] = states[:, t, :3] + dt * acceleration
    return SequenceBatch(
        states[:, : context + 1],
        command[:, :context],
        command[:, context:],
        states[:, context + 1 :],
        dt,
    )


def test_batch_contract_and_copies():
    batch = analytic_batch()
    assert not batch.past_states.flags.writeable
    for invalid in [np.nan, -0.1]:
        with pytest.raises(ValueError, match="inconsistent"):
            replace(batch, dt_s=invalid)
    with pytest.raises(ValueError, match="inconsistent"):
        replace(batch, past_inputs=batch.past_inputs[:, :-1])
    broken = batch.future_states.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(batch, future_states=broken)


def test_initializer_fixed_recipe_training_support_and_two_solves(monkeypatch):
    batch = analytic_batch(3)
    solve, calls = np.linalg.solve, []
    monkeypatch.setattr(
        np.linalg, "solve", lambda a, b: (calls.append(a.shape), solve(a, b))[1]
    )
    model = core.initialize(batch)
    assert len(calls) == 2
    assert model.fingerprint == core.initialize(batch).fingerprint
    assert model.params["w1"].shape[-1] == 32
    assert model.params["memory"].shape[-1] == 8
    np.testing.assert_array_equal(model.params["w2"], 0)
    assert all(value.dtype == np.float64 for value in model.arrays().values())
    normalized = np.concatenate(
        [
            (
                (core._body_features(state, np) - model.norms["body_mean"])
                / model.norms["body_scale"]
            )[..., :6]
            for state in [batch.past_states, batch.future_states]
        ],
        axis=1,
    )
    np.testing.assert_array_equal(
        model.norms["motion_bound_scale"],
        4 * np.maximum(1, np.max(np.abs(normalized), axis=(0, 1))),
    )
    with jax.enable_x64(True):
        np.testing.assert_array_equal(
            core.supported_motion(normalized, model.norms["motion_bound_scale"]),
            normalized,
        )
    broken = batch.future_states.copy()
    broken[..., 6] = 2
    with pytest.raises(ValueError, match="proper"):
        core.initialize(replace(batch, future_states=broken))


def test_fixed_fit_reduces_training_loss_and_selects_development_checkpoint():
    batch = analytic_batch()
    before = bool(jax.config.x64_enabled)
    model, report = core.fit_sequence(batch, batch, _steps=2, _check_every=1)
    assert bool(jax.config.x64_enabled) is before
    assert report["gradient"]["known_gradient_window_visits"] == 8
    assert report["initial_weight_forecasts"] == 1
    assert report["actual_initializers"] == 1
    assert report["actual_ridge_solves"] == 2
    losses = report["safeguard"]["checkpoint_full_training_losses"]
    assert all(
        b["full_training_loss"] <= a["full_training_loss"] for a, b in pairwise(losses)
    )
    selected = min(report["trace"], key=lambda row: row["validation_rollout_mse"])
    assert report["selected_step"] == selected["step"]
    assert report["selected_fingerprint"] == model.fingerprint
    assert report["recipe"] == core.RECIPE_ID
    assert np.isfinite(
        model.rollout(batch.past_states, batch.past_inputs, batch.future_inputs)
    ).all()


def test_training_only_weights_support_and_precision_scope():
    batch = analytic_batch()
    different = batch.future_states.copy()
    different[..., :6] *= 2
    with jax.enable_x64(False):
        model, report = core.fit_sequence(
            batch, replace(batch, future_states=different), _steps=0
        )
        assert not jax.config.x64_enabled
    with jax.enable_x64(True):
        prediction = np.asarray(
            model.rollout(batch.past_states, batch.past_inputs, batch.future_inputs)
        )
    scale = np.asarray(report["error_scale"])
    mse = np.mean(((prediction - batch.future_states) / scale) ** 2, (0, 1))
    raw = 1 / np.maximum(mse, 0.0001)
    np.testing.assert_array_equal(report["channel_weights"], raw / raw.mean())
    assert report["selected_step"] == 0
    assert report["initial_fingerprint"] == core.initialize(batch).fingerprint


def test_equal_loss_rejects_all_scales_and_preserves_initial(monkeypatch):
    monkeypatch.setattr(
        core, "trial_parameters", lambda current, proposal, alpha: current
    )
    batch = analytic_batch()
    model, report = core.fit_sequence(batch, batch, _steps=2, _check_every=1)
    assert report["selected_step"] == 0
    assert report["safeguard"]["rejected_attempts"] == 2
    assert report["full_training_objective_calls"] == 17
    assert model.fingerprint == report["initial_fingerprint"]


def test_failed_initialization_and_timeout_restore_precision_scope(monkeypatch):
    batch = analytic_batch()

    def fail(*args):
        raise np.linalg.LinAlgError("test singular")

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "solve", fail)
        with jax.enable_x64(False), pytest.raises(np.linalg.LinAlgError):
            core.fit_sequence(batch, batch, _steps=0)
        assert not jax.config.x64_enabled
    with monkeypatch.context() as patch:
        patch.setattr(core, "FIT_WALL_TIME_LIMIT_S", -1)
        with (
            jax.enable_x64(False),
            pytest.raises(SequenceFitError, match="fit_time_limit"),
        ):
            core.fit_sequence(batch, batch, _steps=0)
        assert not jax.config.x64_enabled


def test_no_configurable_recipe_and_contract_mismatch():
    batch = analytic_batch()
    with pytest.raises(TypeError):
        core.initialize(batch, width=4)
    with pytest.raises(TypeError):
        core.fit_sequence(batch, batch, channel_weights=np.ones(15))
    with pytest.raises(ValueError, match="contracts differ"):
        core.fit_sequence(batch, replace(batch, dt_s=0.04), _steps=0)


def test_saved_window_consistency_checks_both_sides_of_history_boundary():
    from types import SimpleNamespace

    from glassbox._training import validate_window_consistency

    # Three actual, overlapping windows cut from a single recording.
    states = np.arange(10 * 15, dtype=float).reshape(10, 15)
    commands = np.arange(9 * 2, dtype=float).reshape(9, 2)
    origins = (3, 4, 5)
    batch = SequenceBatch(
        np.stack([states[i - 3 : i + 1] for i in origins]),
        np.stack([commands[i - 3 : i] for i in origins]),
        np.stack([commands[i : i + 2] for i in origins]),
        np.stack([states[i + 1 : i + 3] for i in origins]),
        0.05,
    )
    keys = tuple(SimpleNamespace(recording_id="a", segment_id="whole") for _ in origins)
    validate_window_consistency(batch, keys, origins)
    for field in ("future_states", "future_inputs", "past_states", "past_inputs"):
        values = getattr(batch, field).copy()
        values[1, 0, 0] += 0.5
        with pytest.raises(ValueError, match="disagree"):
            validate_window_consistency(
                replace(batch, **{field: values}), keys, origins
            )
    other_segments = tuple(
        SimpleNamespace(recording_id="a", segment_id=str(i)) for i in origins
    )
    with pytest.raises(ValueError, match="segments overlap"):
        validate_window_consistency(batch, other_segments, origins)
    independent = tuple(
        SimpleNamespace(recording_id=str(i), segment_id="whole") for i in origins
    )
    # Source row numbers are local to a recording; distinct recordings may reuse them.
    validate_window_consistency(batch, independent, origins)
    # Same recording, disjoint retained segments are legal.
    validate_window_consistency(batch, other_segments, (3, 104, 205))
