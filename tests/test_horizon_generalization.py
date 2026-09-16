"""Independent synthetic dynamics identities and evaluation contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    sys.modules.pop("experiment_horizon_generalization", None)
    import experiment_horizon_generalization

    return experiment_horizon_generalization


@pytest.fixture
def plan():
    familiar = dict(
        input_persistence=0.65, input_innovation_weight=0.35, input_amplitude=1.0
    )
    return dict(
        intervals=160,
        dt_s=0.05,
        calibration_recordings=8,
        evaluation_recordings_per_regime=4,
        rng_salt=1207,
        regimes=dict(
            calibration=familiar,
            matched=familiar,
            shifted=dict(
                input_persistence=0.25,
                input_innovation_weight=0.75,
                input_amplitude=1.25,
            ),
        ),
    )


def test_affine_superposition_and_deadzone_contract(experiment):
    zero = np.zeros(2)

    def f(x, u):
        return experiment.step("stable_affine", np.array(x), np.array(u), zero, zero)

    x, y, u, v = [0.2, -0.4, 0.1], [-0.1, 0.3, 0.5], [0.1, 0.2], [-0.2, 0.1]
    np.testing.assert_allclose(
        f(np.array(x) + y, np.array(u) + v), f(x, u) + f(y, v), atol=1e-15
    )
    for command, effective in [(0.1, 0), (-0.2, 0), (0.4, 0.2), (1, 0.45), (-1, -0.45)]:
        output = experiment.step(
            "deadzone_saturation",
            np.array([0.3]),
            np.array([command]),
            np.zeros(1),
            np.zeros(1),
        )
        np.testing.assert_allclose(output, 0.88 * 0.3 + 0.22 * effective)


def test_delay_is_four_steps_and_hysteresis_retains_memory(experiment):
    immediate = experiment.step(
        "delayed_nonlinear", np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1)
    )
    delayed = experiment.step(
        "delayed_nonlinear", np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1)
    )
    np.testing.assert_array_equal(immediate, [0])
    np.testing.assert_allclose(delayed, [0.22 * np.tanh(2)])
    negative = experiment.step(
        "hidden_hysteresis",
        np.array([0.0, -0.3]),
        np.zeros(1),
        np.zeros(1),
        np.zeros(2),
    )
    positive = experiment.step(
        "hidden_hysteresis", np.array([0.0, 0.3]), np.zeros(1), np.zeros(1), np.zeros(2)
    )
    assert positive[0] > 0 > negative[0]


@pytest.mark.parametrize("family", ["near_periodic", "off_periodic"])
def test_unforced_oscillator_norm_decays_without_exact_return(experiment, family):
    x = np.array([0.3, -0.2])
    original = x.copy()
    for _ in range(5):
        next_x = experiment.step(family, x, np.zeros(2), np.zeros(2), np.zeros(2))
        np.testing.assert_allclose(
            np.linalg.norm(next_x), 0.98 * np.linalg.norm(x), rtol=1e-14
        )
        x = next_x
    assert not np.allclose(x, original)


@pytest.mark.parametrize(
    "family",
    [
        "stable_affine",
        "coupled_nonlinear",
        "deadzone_saturation",
        "hidden_hysteresis",
        "delayed_nonlinear",
        "near_periodic",
        "off_periodic",
        "noisy_observation",
    ],
)
def test_generation_and_evaluation_keep_targets_out_of_inputs(experiment, plan, family):
    calibration, _ = experiment.generate(plan, family, 4101, "calibration")
    matched, truth = experiment.generate(plan, family, 4101, "matched")
    shifted, _ = experiment.generate(plan, family, 4101, "shifted")
    ids = [
        {s.recording_id for s in c.segments} for c in (calibration, matched, shifted)
    ]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    arrays = experiment.evaluate_arrays(matched, 5)
    assert len(arrays["past_states"]) == 124
    s = matched.segments[0]
    np.testing.assert_array_equal(arrays["past_states"][0], s.states[:3])
    np.testing.assert_array_equal(arrays["past_inputs"][0], s.inputs[:2])
    np.testing.assert_array_equal(arrays["future_inputs"][0], s.inputs[2:7])
    np.testing.assert_array_equal(arrays["targets"][0], s.states[3:8])
    again, _ = experiment.generate(plan, family, 4101, "matched")
    np.testing.assert_array_equal(again.segments[0].states, s.states)
    if family == "hidden_hysteresis":
        assert s.states.shape[1] == 1 and truth[s.recording_id].shape[1] == 2
    if family == "noisy_observation":
        assert not np.array_equal(s.states, truth[s.recording_id])
    if family == "delayed_nonlinear":
        np.testing.assert_allclose(
            s.states[:5], 0.72 ** np.arange(5)[:, None] * s.states[0]
        )
        np.testing.assert_allclose(
            s.states[5], 0.72 * s.states[4] + 0.22 * np.tanh(2 * s.inputs[0])
        )


def test_error_metric_preserves_units_and_screen_needs_both_changes(experiment):
    rng = np.random.default_rng(71)
    y = rng.normal(size=(7, 5, 2))
    p = y + 0.1 * rng.normal(size=y.shape)
    scale = np.array([2.0, 0.5])
    first = experiment.measure(p, y, scale)
    factors = np.array([1000.0, 0.01])
    offset = np.array([100.0, -0.2])
    encoded = experiment.measure(
        p * factors + offset, y * factors + offset, scale * factors
    )
    np.testing.assert_allclose(
        first["horizon_scaled_rmse"], encoded["horizon_scaled_rmse"], rtol=1e-12
    )
    protocol = dict(relative_rmse_increase=0.1, absolute_scaled_rmse_increase=0.01)
    assert not experiment.screen(0.001, 0.002, protocol)
    assert not experiment.screen(1, 1.05, protocol)
    assert experiment.screen(0.1, 0.13, protocol)


def test_first_step_floor_preserves_unaffected_objectives_and_units(experiment):
    from experiment_first_step_floor import first_step_floor

    scales = np.array([[0.2, 0.3], [0.4, 0.5], [0.1, 0.8], [0.01, 0.7]])
    copied = scales.copy()
    revised = first_step_floor(scales)
    np.testing.assert_array_equal(scales, copied)
    np.testing.assert_array_equal(revised[:, 1], scales[:, 1])
    np.testing.assert_array_equal(revised[:, 0], [0.2, 0.4, 0.2, 0.2])
    assert np.all(1 / revised**2 <= 1 / revised[0] ** 2)
    factors = np.array([0.01, 1000.0])
    np.testing.assert_allclose(first_step_floor(scales * factors), revised * factors)
    monotone = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.5]])
    np.testing.assert_array_equal(first_step_floor(monotone), monotone)
