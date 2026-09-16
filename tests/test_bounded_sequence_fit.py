"""Budget, failure, selection, precision and analytic fit contracts."""

import copy

import jax
import numpy as np
import pytest

from glassbox.experimental import _sequence_fit as bounded
from glassbox.experimental.sequence_model import (
    initialize_sequence_model,
    sequence_windows,
)


def scores(vector):
    value = float(np.sum(vector**2))
    return dict(mean=value, recordings={"a": value, "b": value})


def test_exact_line_search_budget_retains_last_accepted_point(monkeypatch):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return float(x @ x), 2 * x

    def solver(fun, x0, callback, **kwargs):
        fun(np.array([1.0]))
        callback(np.array([1.0]))
        for value in range(2, 20):
            fun(np.array([float(value)]))
        raise AssertionError("budget was not enforced")

    monkeypatch.setattr(bounded, "fmin_l_bfgs_b", solver)
    settings = dict(bounded._BOUNDED_FIT, max_function_evaluations=3)
    model, report = bounded._run_lbfgs(np.array([4.0]), objective, scores, settings)
    np.testing.assert_array_equal(model, [1.0])
    assert len(calls) == report["resource_usage"]["training_value_gradient_calls"] == 3
    assert report["termination"]["reason"] == "evaluation_budget"
    assert report["termination"]["accepted_iterations"] == 1


def test_optimizer_convergence_does_not_claim_selected_checkpoint_convergence():
    def objective(x):
        return float(np.sum((x - 2) ** 2)), 2 * (x - 2)

    def development(x):
        a = float(x[0] ** 2)
        b = float((x[0] - 2) ** 2)
        return dict(mean=0.9 * a + 0.1 * b, recordings={"a": a, "b": b})

    chosen, report = bounded._run_lbfgs(
        np.array([0.0]), objective, development, bounded._BOUNDED_FIT
    )
    np.testing.assert_array_equal(chosen, [0.0])
    assert report["termination"]["gradient_converged"]
    assert report["selected_training_gradient_inf_norm"] == 4.0
    assert report["selection"]["recording_winner_disagreement"]


def test_nonfinite_line_trial_is_not_selected(monkeypatch):
    def objective(x):
        if x[0] > 1:
            return np.inf, np.full_like(x, np.nan)
        return float(x @ x), 2 * x

    def solver(fun, x0, callback, **kwargs):
        bad, _ = fun(np.array([2.0]))
        assert np.isinf(bad)
        fun(np.array([0.5]))
        callback(np.array([0.5]))
        return (
            np.array([0.5]),
            0.25,
            dict(task="test failure after finite progress", warnflag=2),
        )

    monkeypatch.setattr(bounded, "fmin_l_bfgs_b", solver)
    chosen, report = bounded._run_lbfgs(
        np.array([1.0]), objective, scores, bounded._BOUNDED_FIT
    )
    np.testing.assert_array_equal(chosen, [0.5])
    assert report["resource_usage"]["nonfinite_trials"] == 1
    assert report["termination"]["reason"] == "line_search_failed"


def test_nonfinite_development_is_counted_and_keeps_valid_checkpoint():
    def objective(x):
        return float(np.sum((x - 2) ** 2)), 2 * (x - 2)

    def development(x):
        value = 1.0 if x[0] == 0 else np.inf
        return dict(mean=value, recordings={"a": value})

    chosen, report = bounded._run_lbfgs(
        np.array([0.0]), objective, development, bounded._BOUNDED_FIT
    )
    np.testing.assert_array_equal(chosen, [0.0])
    assert report["resource_usage"]["development_passes"] == 2
    assert report["resource_usage"]["nonfinite_development_checkpoints"] == 1
    assert len(report["trace"]) == 1


def test_invalid_initial_objective_fails_before_selection():
    with pytest.raises(ValueError, match="initial training"):
        bounded._run_lbfgs(
            np.zeros(1), lambda x: (np.nan, np.zeros(1)), scores, bounded._BOUNDED_FIT
        )


def fixture(seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, (100, 1))
    x = np.zeros((101, 1))
    for t in range(100):
        x[t + 1] = 0.83 * x[t] + 0.17 * u[t]
    return sequence_windows(
        x, u, np.arange(2, 95, 3), history_steps=2, horizon_steps=5, dt_s=0.05
    )


def test_affine_response_fits_and_precision_scope_is_preserved():
    train, dev = fixture(81), fixture(82)
    initial = initialize_sequence_model(
        train, kind="delay_mlp", width=4, ridge=0.01 * len(train.past_states) * 5
    )
    before = copy.deepcopy(initial.params)
    enabled = jax.config.x64_enabled
    scales = np.maximum(
        np.sqrt(
            np.mean((train.future_states - train.past_states[:, -1:]) ** 2, axis=0)
        ),
        0.01 * initial.norms["state_scale"],
    )
    model, report = bounded._fit_bounded_sequence_model(
        train,
        dev,
        initial,
        scales,
        np.array(["a"] * 16 + ["b"] * (len(dev.past_states) - 16)),
    )
    assert jax.config.x64_enabled == enabled
    for k, v in before.items():
        np.testing.assert_array_equal(initial.params[k], v)
    with jax.enable_x64(True):
        predicted = np.asarray(
            model.rollout(dev.past_states, dev.past_inputs, dev.future_inputs)
        )
        assert np.sqrt(np.mean((predicted - dev.future_states) ** 2)) < 1e-5
        jac = jax.jacfwd(
            lambda commands: model.rollout(
                dev.past_states[0], dev.past_inputs[0], commands
            )
        )(dev.future_inputs[0])
        assert np.isfinite(jac).all()
        for h in range(5):
            np.testing.assert_array_equal(jac[h, :, h + 1 :], 0)
    usage = report["resource_usage"]
    assert (
        usage["training_value_gradient_calls"] <= 160
        and usage["development_passes"] <= 12
    )
    assert report["precision"] == "float64"
