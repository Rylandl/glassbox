from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.core.streaming_evaluation import StreamingOneStepEvaluator


class ConstantAccelerationModel:
    command_size = 1
    runtime_spec = SimpleNamespace(sample_period_s=0.1)

    def initial_latent_state(self, command_history: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(command_history)[-1:]

    def transition(
        self,
        state: jnp.ndarray,
        latent_state: jnp.ndarray,
        command: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.transition_at_interval(
            state,
            latent_state,
            command,
            self.runtime_spec.sample_period_s,
        )

    def transition_at_interval(
        self,
        state: jnp.ndarray,
        latent_state: jnp.ndarray,
        command: jnp.ndarray,
        interval_s: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        acceleration = command[0]
        next_state = state.at[0].add(0.5 * acceleration * interval_s**2)
        next_state = next_state.at[3].add(acceleration * interval_s)
        return next_state, command


def state(*, position_m: float = 0.0, velocity_m_s: float = 0.0) -> np.ndarray:
    value = np.zeros(13)
    value[0] = position_m
    value[3] = velocity_m_s
    value[6] = 1.0
    return value


def test_streaming_evaluator_scores_nominal_intervals_against_persistence() -> None:
    evaluator = StreamingOneStepEvaluator(ConstantAccelerationModel())  # type: ignore[arg-type]

    assert evaluator.observe(state(), np.asarray([2.0]), elapsed_s=0.0) is None
    result = evaluator.observe(
        state(position_m=0.01, velocity_m_s=0.2),
        np.asarray([2.0]),
        elapsed_s=0.1,
    )

    assert result is not None
    assert result["status"] == "evaluated"
    assert result["interval_s"] == pytest.approx(0.1)
    assert result["model_error"]["position_error_m"] < 1e-8
    assert result["model_error"]["velocity_error_m_s"] < 1e-8
    assert result["kinematic_error"]["position_error_m"] == pytest.approx(0.01)
    assert result["kinematic_error"]["velocity_error_m_s"] == pytest.approx(0.2)

    summary = evaluator.summary()
    assert summary["evaluated_transition_count"] == 1
    assert summary["timing_ineligible_transition_count"] == 0
    assert summary["evaluated_transition_fraction"] == 1.0
    assert summary["model"]["position_rmse_m"] < 1e-8
    assert summary["model_to_kinematic_ratio"]["position_rmse_m"] < 1e-6


def test_streaming_evaluator_skips_gaps_and_reanchors_latent_state() -> None:
    evaluator = StreamingOneStepEvaluator(ConstantAccelerationModel())  # type: ignore[arg-type]
    evaluator.observe(state(), np.asarray([1.0]), elapsed_s=0.0)

    skipped = evaluator.observe(state(), np.asarray([2.0]), elapsed_s=0.3)
    resumed = evaluator.observe(
        state(position_m=0.01, velocity_m_s=0.2),
        np.asarray([2.0]),
        elapsed_s=0.4,
    )

    assert skipped == {"status": "timing_ineligible", "interval_s": 0.3}
    assert resumed is not None
    assert resumed["status"] == "evaluated"
    summary = evaluator.summary()
    assert summary["transition_count"] == 2
    assert summary["evaluated_transition_count"] == 1
    assert summary["timing_ineligible_transition_count"] == 1
    assert summary["evaluated_transition_fraction"] == 0.5


def test_streaming_evaluator_integrates_short_irregular_intervals() -> None:
    evaluator = StreamingOneStepEvaluator(ConstantAccelerationModel())  # type: ignore[arg-type]
    evaluator.observe(state(), np.asarray([2.0]), elapsed_s=0.0)

    result = evaluator.observe(
        state(position_m=0.0256, velocity_m_s=0.32),
        np.asarray([2.0]),
        elapsed_s=0.16,
    )

    assert result is not None
    assert result["status"] == "evaluated"
    assert result["interval_s"] == pytest.approx(0.16)
    assert result["model_error"]["position_error_m"] < 1e-8
    assert result["model_error"]["velocity_error_m_s"] < 1e-8
    summary = evaluator.summary()
    assert summary["evaluated_transition_fraction"] == 1.0
    assert summary["observed_interval_median_s"] == pytest.approx(0.16)


def test_streaming_evaluator_validates_canonical_observations() -> None:
    evaluator = StreamingOneStepEvaluator(ConstantAccelerationModel())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unit quaternion"):
        evaluator.observe(np.zeros(13), np.asarray([1.0]), elapsed_s=0.0)
    with pytest.raises(ValueError, match="1 finite"):
        evaluator.observe(state(), np.asarray([1.0, 2.0]), elapsed_s=0.0)
