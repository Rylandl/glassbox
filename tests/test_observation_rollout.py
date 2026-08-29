from types import SimpleNamespace

import numpy as np

from glassbox.observation_compatibility import FirstOrderObservationFilter
from glassbox.observation_rollout import (
    body_rate_observation_metrics_from_predictions,
    evaluate_body_rate_observation_rollouts,
    first_order_body_rate_observations,
)


def _filter(time_constant_s: float) -> FirstOrderObservationFilter:
    return FirstOrderObservationFilter(
        velocity_time_constant_s=np.zeros(3),
        velocity_scale=np.ones(3),
        velocity_bias_m_s=np.zeros(3),
        angular_rate_time_constant_s=np.full(3, time_constant_s),
        angular_rate_scale=np.ones(3),
        angular_rate_bias_rad_s=np.zeros(3),
    )


def test_body_rate_observation_filter_is_causal_and_batched() -> None:
    dt_s = 0.02
    time_s = np.arange(101) * dt_s
    physical = np.stack(
        (
            np.sin(time_s),
            np.cos(0.7 * time_s),
            np.sin(1.3 * time_s + 0.2),
        ),
        axis=1,
    )
    batched = np.stack((physical, 2.0 * physical))

    observed = first_order_body_rate_observations(
        batched,
        initial_reported_rate_rad_s=batched[:, 0],
        model=_filter(0.08),
        dt_s=dt_s,
    )

    assert observed.shape == batched.shape
    np.testing.assert_allclose(observed[:, 0], batched[:, 0])
    assert np.max(np.abs(observed[:, 1:] - batched[:, 1:])) > 0.01


def test_observation_aware_metrics_change_only_reported_body_rate() -> None:
    dt_s = 0.02
    time_s = np.arange(101) * dt_s
    physical_rate = np.stack(
        (
            np.sin(2.0 * time_s),
            np.cos(1.4 * time_s),
            np.sin(1.7 * time_s + 0.3),
        ),
        axis=1,
    )
    candidate = _filter(0.08)
    reference = _filter(0.0)
    reported_rate = first_order_body_rate_observations(
        physical_rate,
        initial_reported_rate_rad_s=physical_rate[0],
        model=candidate,
        dt_s=dt_s,
    )
    predicted = np.zeros((len(time_s), 13), dtype=np.float64)
    target = np.zeros_like(predicted)
    predicted[:, 6] = 1.0
    target[:, 6] = 1.0
    predicted[:, 0] = 0.1 * time_s
    target[:, 0] = 0.08 * time_s
    predicted[:, 10:13] = physical_rate
    target[:, 10:13] = reported_rate

    candidate_metrics = body_rate_observation_metrics_from_predictions(
        predicted,
        target,
        model=candidate,
        dt_s=dt_s,
    )
    reference_metrics = body_rate_observation_metrics_from_predictions(
        predicted,
        target,
        model=reference,
        dt_s=dt_s,
    )

    assert candidate_metrics["angular_velocity_rmse_rad_s"] < 1e-12
    assert reference_metrics["angular_velocity_rmse_rad_s"] > 0.01
    for name in (
        "position_rmse_m",
        "velocity_rmse_m_s",
        "attitude_rmse_deg",
    ):
        assert candidate_metrics[name] == reference_metrics[name]


def test_rollout_gate_aggregates_all_horizons_without_changing_dynamics(
    monkeypatch,
) -> None:
    dt_s = 0.02
    candidate = _filter(0.08)
    reference = _filter(0.0)

    trajectory = SimpleNamespace(
        nominal_dt_s=dt_s,
        labels={
            "source_group": "synthetic_validation",
            "profile": "temporal_rate",
            "replicate": 1,
        },
    )

    def fake_predictions(params, trajectory, *, horizon_steps, stride_steps):
        del params, trajectory, stride_steps
        time_s = np.arange(horizon_steps + 1) * dt_s
        physical_rate = np.stack(
            (
                np.sin(5.0 * time_s),
                np.cos(4.0 * time_s),
                np.sin(3.0 * time_s + 0.2),
            ),
            axis=1,
        )
        reported_rate = first_order_body_rate_observations(
            physical_rate,
            initial_reported_rate_rad_s=physical_rate[0],
            model=candidate,
            dt_s=dt_s,
        )
        predicted = np.zeros((1, horizon_steps + 1, 13))
        target = np.zeros_like(predicted)
        predicted[..., 6] = 1.0
        target[..., 6] = 1.0
        predicted[0, :, 10:13] = physical_rate
        target[0, :, 10:13] = reported_rate
        return predicted, target, dt_s

    monkeypatch.setattr(
        "glassbox.observation_rollout.windowed_rollout_predictions",
        fake_predictions,
    )
    report = evaluate_body_rate_observation_rollouts(
        None,
        candidate,
        reference,
        [trajectory],
    )

    assert report["gate"]["research_rollout_passes"] is True
    assert report["invariants"] == {
        "dynamics_parameters_changed": False,
        "physical_rollout_changed": False,
        "position_velocity_attitude_metrics_changed": False,
        "only_reported_body_rate_changed": True,
    }
