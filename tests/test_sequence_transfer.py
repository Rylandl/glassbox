"""Causal telemetry, horizon balancing, and rejection of hidden tradeoffs."""

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.causal_sampling import causal_hold
from glassbox.experimental.sequence_model import fit_sequence_model, sequence_windows
from glassbox.experimental.sequence_objective import SequenceGuard, relative_error_scale


def test_causal_hold_uses_publication_time_and_last_duplicate():
    held = causal_hold(
        [0.02, 0.10, 0.10, 0.3],
        [[2], [10], [11], [30]],
        [0, 0.02, 0.09, 0.10, 0.2, 0.31],
        maximum_age_s=0.08,
    )
    np.testing.assert_array_equal(held.source_indices, [-1, 0, 0, 2, -1, 3])
    np.testing.assert_allclose(held.values[[1, 2, 3, 5], 0], [2, 2, 11, 30])
    assert np.isnan(held.values[[0, 4]]).all()
    np.testing.assert_allclose(held.age_s[1:], [0, 0.07, 0, 0.1, 0.01])


def test_future_publications_cannot_change_earlier_sampled_features():
    time = np.array([0, 0.03, 0.07, 0.2])
    a = causal_hold(time, np.arange(4.0)[:, None], [0.05, 0.1], maximum_age_s=0.1)
    b = causal_hold(
        np.r_[time, 0.25],
        np.array([0, 1, 2, -900, 1000])[:, None],
        [0.05, 0.1],
        maximum_age_s=0.1,
    )
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.source_indices, b.source_indices)


@pytest.mark.parametrize(
    "time,values,age",
    [([1, 0], [[1], [2]], 1), ([0, 1], [[1], [np.nan]], 1), ([0, 1], [[1], [2]], -1)],
)
def test_invalid_causal_streams_are_rejected(time, values, age):
    with pytest.raises(ValueError):
        causal_hold(time, values, [0.5], maximum_age_s=age)


def test_relative_scale_equalizes_horizons_and_uses_explicit_floor():
    target = np.zeros((4, 2, 3))
    residual = np.broadcast_to([[1, 2, 0], [10, 20, 0]], target.shape)
    actual = relative_error_scale(residual, target, [1, 1, 2], minimum_fraction=0.01)
    np.testing.assert_allclose(actual, [[1, 2, 0.02], [10, 20, 0.02]])
    np.testing.assert_allclose((residual / actual)[:, 0], (residual / actual)[:, 1])


def test_guard_checks_each_horizon_and_group_not_just_average():
    guard = SequenceGuard((1, 3), ((0, 1), (2,)), 1.05)
    target = jnp.zeros((5, 3, 3))
    prediction = jnp.ones_like(target)
    errors = guard.errors(prediction, target)
    np.testing.assert_allclose(errors, [[np.sqrt(2), 1], [np.sqrt(2), 1]])
    changed = prediction.at[:, 0, 2].set(1.2).at[:, 2, :].set(0.1)
    assert np.mean(np.asarray(changed) ** 2) < np.mean(np.asarray(prediction) ** 2)
    assert not guard.accepts(guard.errors(changed, target), errors)
    assert guard.accepts(errors, errors)


def test_guard_allows_float32_roundoff_but_rejects_real_regression():
    guard = SequenceGuard((1,), ((0,),), 1.0)
    baseline = np.array([[0.001]], dtype=np.float32)
    # A serialized reference loses its original dtype; candidate precision
    # must still determine the comparison tolerance.
    assert guard.accepts(
        baseline * (1 + 32 * np.finfo(np.float32).eps), baseline.tolist()
    )
    assert not guard.accepts(baseline * 1.001, baseline.tolist())


@pytest.mark.parametrize(
    "errors", [np.array([[np.inf]]), np.array([[-1.0]]), np.ones((1, 2))]
)
def test_guard_rejects_invalid_error_evidence(errors):
    assert not SequenceGuard((1,), ((0,),)).accepts(errors, [[1.0]])


@pytest.mark.parametrize(
    "h,g,r",
    [
        ((0,), ((0,),), 1.1),
        ((1,), ((),), 1.1),
        ((1,), ((-1,),), 1.1),
        ((1,), ((0,),), 0.9),
    ],
)
def test_invalid_guard_is_rejected(h, g, r):
    with pytest.raises(ValueError):
        SequenceGuard(h, g, r)


def test_guarded_fit_preserves_a_valid_initialization_and_reports_scale():
    rng = np.random.default_rng(3)
    u = rng.normal(size=(99, 1))
    x = np.zeros((100, 1))
    for t in range(99):
        x[t + 1] = 0.9 * x[t] + 0.1 * u[t]
    train = sequence_windows(
        x, u, np.arange(10, 60, 2), history_steps=3, horizon_steps=5, dt_s=0.1
    )
    development = sequence_windows(
        x, u, np.arange(65, 90, 2), history_steps=3, horizon_steps=5, dt_s=0.1
    )
    guard = SequenceGuard((1, 5), ((0,),), 1.0)
    model, report = fit_sequence_model(
        train,
        development,
        kind="delay_mlp",
        steps=10,
        check_every=5,
        error_scale=np.ones((5, 1)) * 0.1,
        selection_guard=guard,
    )
    assert report["loss_scale_mode"] == "explicit_horizon_channel"
    accepted = [r for r in report["trace"] if r["guard_accepted"]]
    assert (
        report["selected_step"]
        == min(accepted, key=lambda r: r["validation_rollout_mse"])["step"]
    )
    actual = model.rollout(
        development.past_states, development.past_inputs, development.future_inputs
    )
    assert guard.accepts(
        guard.errors(actual, development.future_states),
        report["trace"][0]["guard_errors"],
    )
    with pytest.raises(ValueError, match="error_scale"):
        fit_sequence_model(train, development, steps=0, error_scale=np.zeros((5, 1)))
