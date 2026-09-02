"""Online logged-input checks for executable Glassbox dynamics models."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.evaluation import METRIC_FLOORS, ROLLOUT_METRICS
from glassbox.core.runtime import RuntimeDynamicsModel

_MINIMUM_INTERVAL_MULTIPLE = 0.5
_MAXIMUM_INTERVAL_MULTIPLE = 2.5
_INTERVAL_QUANTIZATION_US = 1_000


def _validated_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (13,) or not np.all(np.isfinite(state)):
        raise ValueError("streaming evaluation state must contain 13 finite values")
    quaternion_norm = np.linalg.norm(state[6:10])
    if not np.isclose(quaternion_norm, 1.0, atol=1e-6):
        raise ValueError("streaming evaluation attitude must be a unit quaternion")
    return state


def _validated_command(command: np.ndarray, *, expected_size: int) -> np.ndarray:
    command = np.asarray(command, dtype=np.float64)
    if command.shape != (expected_size,) or not np.all(np.isfinite(command)):
        raise ValueError(
            f"streaming command must contain {expected_size} finite values"
        )
    return command


def _kinematic_prediction(state: np.ndarray, dt_s: float) -> np.ndarray:
    predicted = state.copy()
    quaternion = state[6:10]
    angular_velocity = state[10:13]
    w, x, y, z = quaternion
    wx, wy, wz = angular_velocity
    quaternion_rate = 0.5 * np.asarray(
        [
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy - x * wz + z * wx,
            w * wz + x * wy - y * wx,
        ]
    )
    predicted[0:3] += dt_s * state[3:6]
    predicted[6:10] += dt_s * quaternion_rate
    predicted[6:10] /= np.linalg.norm(predicted[6:10])
    return predicted


def _state_error(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    quaternion_dot = abs(float(np.dot(predicted[6:10], actual[6:10])))
    attitude_error_deg = np.rad2deg(2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0)))
    return {
        "position_error_m": float(np.linalg.norm(predicted[0:3] - actual[0:3])),
        "velocity_error_m_s": float(np.linalg.norm(predicted[3:6] - actual[3:6])),
        "attitude_error_deg": float(attitude_error_deg),
        "angular_velocity_error_rad_s": float(
            np.linalg.norm(predicted[10:13] - actual[10:13])
        ),
    }


def _aggregate_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    position_error = predicted[:, 0:3] - actual[:, 0:3]
    velocity_error = predicted[:, 3:6] - actual[:, 3:6]
    angular_velocity_error = predicted[:, 10:13] - actual[:, 10:13]
    quaternion_dot = np.abs(np.sum(predicted[:, 6:10] * actual[:, 6:10], axis=1))
    attitude_error_deg = np.rad2deg(2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0)))
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "attitude_rmse_deg": float(np.sqrt(np.mean(np.square(attitude_error_deg)))),
        "angular_velocity_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(angular_velocity_error)))
        ),
    }


class StreamingOneStepEvaluator:
    """Audit short observed transitions against live logged inputs.

    Each prediction is reset to the latest measured rigid-body state while the
    model's latent actuator state is carried forward. The continuous dynamics
    are integrated across the observation's actual source-time interval with a
    zero-order hold on its starting command. Large telemetry gaps are counted
    but not scored because their intermediate commands were not observed.
    """

    def __init__(self, model: RuntimeDynamicsModel) -> None:
        self.model = model
        self._initial_latent_compiled = jax.jit(model.initial_latent_state)
        self._compiled_transitions: dict[int, Any] = {}
        self._previous_state: np.ndarray | None = None
        self._previous_command: np.ndarray | None = None
        self._previous_elapsed_s: float | None = None
        self._latent_state: np.ndarray | None = None
        self._intervals_s: list[float] = []
        self._model_predictions: list[np.ndarray] = []
        self._kinematic_predictions: list[np.ndarray] = []
        self._actual_states: list[np.ndarray] = []
        self._observation_count = 0
        self._timing_ineligible_count = 0

    def _initial_latent(self, command: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._initial_latent_compiled(jnp.asarray(command)),
            dtype=np.float64,
        )

    def _set_anchor(
        self,
        state: np.ndarray,
        command: np.ndarray,
        elapsed_s: float,
    ) -> None:
        self._previous_state = state
        self._previous_command = command
        self._previous_elapsed_s = elapsed_s
        self._latent_state = self._initial_latent(command)

    def _transition(
        self,
        state: np.ndarray,
        latent_state: np.ndarray,
        command: np.ndarray,
        interval_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        interval_us = max(
            _INTERVAL_QUANTIZATION_US,
            round(interval_s * 1e6 / _INTERVAL_QUANTIZATION_US)
            * _INTERVAL_QUANTIZATION_US,
        )
        transition = self._compiled_transitions.get(interval_us)
        if transition is None:
            quantized_interval_s = interval_us * 1e-6

            def transition_at_interval(
                current_state: jnp.ndarray,
                current_latent: jnp.ndarray,
                current_command: jnp.ndarray,
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                return self.model.transition_at_interval(
                    current_state,
                    current_latent,
                    current_command,
                    quantized_interval_s,
                )

            transition = jax.jit(transition_at_interval)
            self._compiled_transitions[interval_us] = transition
        predicted, next_latent = transition(
            jnp.asarray(state),
            jnp.asarray(latent_state),
            jnp.asarray(command),
        )
        return (
            np.asarray(predicted, dtype=np.float64),
            np.asarray(next_latent, dtype=np.float64),
        )

    def observe(
        self,
        state: np.ndarray,
        command: np.ndarray,
        *,
        elapsed_s: float,
    ) -> dict[str, Any] | None:
        """Consume one observation and return its preceding transition audit."""

        state = _validated_state(state)
        command = _validated_command(command, expected_size=self.model.command_size)
        elapsed_s = float(elapsed_s)
        if not np.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("streaming observation elapsed_s must be nonnegative")
        self._observation_count += 1
        if self._previous_state is None:
            self._set_anchor(state, command, elapsed_s)
            return None

        previous_elapsed_s = self._previous_elapsed_s
        previous_state = self._previous_state
        previous_command = self._previous_command
        latent_state = self._latent_state
        if (
            previous_elapsed_s is None
            or previous_command is None
            or latent_state is None
        ):
            raise RuntimeError("streaming evaluator anchor is incomplete")
        interval_s = elapsed_s - previous_elapsed_s
        self._intervals_s.append(interval_s)
        nominal_period_s = self.model.runtime_spec.sample_period_s
        if (
            interval_s < _MINIMUM_INTERVAL_MULTIPLE * nominal_period_s
            or interval_s > _MAXIMUM_INTERVAL_MULTIPLE * nominal_period_s
        ):
            self._timing_ineligible_count += 1
            self._set_anchor(state, command, elapsed_s)
            return {
                "status": "timing_ineligible",
                "interval_s": interval_s,
            }

        predicted, next_latent = self._transition(
            previous_state,
            latent_state,
            previous_command,
            interval_s,
        )
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(next_latent)):
            raise RuntimeError("streaming one-step model prediction is non-finite")
        kinematic = _kinematic_prediction(previous_state, interval_s)

        self._model_predictions.append(predicted)
        self._kinematic_predictions.append(kinematic)
        self._actual_states.append(state)
        self._previous_state = state
        self._previous_command = command
        self._previous_elapsed_s = elapsed_s
        self._latent_state = next_latent
        return {
            "status": "evaluated",
            "interval_s": interval_s,
            "model_predicted_state": predicted.tolist(),
            "kinematic_predicted_state": kinematic.tolist(),
            "model_error": _state_error(predicted, state),
            "kinematic_error": _state_error(kinematic, state),
        }

    def summary(self) -> dict[str, Any]:
        """Return aggregate model and persistence errors for scored transitions."""

        transition_count = max(0, self._observation_count - 1)
        evaluated_count = len(self._model_predictions)
        intervals = np.asarray(self._intervals_s, dtype=np.float64)
        result: dict[str, Any] = {
            "scope": "logged_input_one_step_prediction_not_flight_safety",
            "nominal_sample_period_s": self.model.runtime_spec.sample_period_s,
            "minimum_eligible_interval_s": (
                _MINIMUM_INTERVAL_MULTIPLE * self.model.runtime_spec.sample_period_s
            ),
            "maximum_eligible_interval_s": (
                _MAXIMUM_INTERVAL_MULTIPLE * self.model.runtime_spec.sample_period_s
            ),
            "observation_count": self._observation_count,
            "transition_count": transition_count,
            "evaluated_transition_count": evaluated_count,
            "timing_ineligible_transition_count": self._timing_ineligible_count,
            "evaluated_transition_fraction": (
                evaluated_count / transition_count if transition_count else 0.0
            ),
            "observed_interval_median_s": (
                float(np.median(intervals)) if len(intervals) else None
            ),
            "observed_interval_minimum_s": (
                float(np.min(intervals)) if len(intervals) else None
            ),
            "observed_interval_maximum_s": (
                float(np.max(intervals)) if len(intervals) else None
            ),
            "model": None,
            "kinematic_persistence": None,
            "model_to_kinematic_ratio": None,
        }
        if not evaluated_count:
            return result

        actual = np.asarray(self._actual_states)
        model = _aggregate_metrics(np.asarray(self._model_predictions), actual)
        kinematic = _aggregate_metrics(np.asarray(self._kinematic_predictions), actual)
        result["model"] = model
        result["kinematic_persistence"] = kinematic
        result["model_to_kinematic_ratio"] = {
            name: model[name] / max(kinematic[name], METRIC_FLOORS[name])
            for name in ROLLOUT_METRICS
        }
        return result
