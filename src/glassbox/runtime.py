"""Typed execution contract for fitted Glassbox dynamics models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.data import ControlChannel, Trajectory, TrajectorySpec
from glassbox.dynamics import (
    ModelParams,
    control_state_after_history,
    model_family,
    quaternion_to_rotation,
    step_with_latent,
)

ACTIONABLE_CONTROL_SEMANTICS = frozenset(
    {"normalized_command", "normalized_generalized_command"}
)


class NonActionableModelError(ValueError):
    """Raised when model inputs cannot safely be interpreted as commands."""


def _finite_triplet(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class ModelValidityEnvelope:
    """Training-supported body-velocity and body-rate operating envelope."""

    body_velocity_center_m_s: tuple[float, float, float]
    body_velocity_half_width_m_s: tuple[float, float, float]
    angular_velocity_center_rad_s: tuple[float, float, float]
    angular_velocity_half_width_rad_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in (
            "body_velocity_center_m_s",
            "body_velocity_half_width_m_s",
            "angular_velocity_center_rad_s",
            "angular_velocity_half_width_rad_s",
        ):
            object.__setattr__(self, name, _finite_triplet(name, getattr(self, name)))
        if np.any(np.asarray(self.body_velocity_half_width_m_s) <= 0.0):
            raise ValueError("body velocity envelope half-widths must be positive")
        if np.any(np.asarray(self.angular_velocity_half_width_rad_s) <= 0.0):
            raise ValueError("angular velocity envelope half-widths must be positive")

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "body_velocity_center_m_s": list(self.body_velocity_center_m_s),
            "body_velocity_half_width_m_s": list(self.body_velocity_half_width_m_s),
            "angular_velocity_center_rad_s": list(self.angular_velocity_center_rad_s),
            "angular_velocity_half_width_rad_s": list(
                self.angular_velocity_half_width_rad_s
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelValidityEnvelope:
        return cls(
            body_velocity_center_m_s=tuple(payload["body_velocity_center_m_s"]),
            body_velocity_half_width_m_s=tuple(payload["body_velocity_half_width_m_s"]),
            angular_velocity_center_rad_s=tuple(
                payload["angular_velocity_center_rad_s"]
            ),
            angular_velocity_half_width_rad_s=tuple(
                payload["angular_velocity_half_width_rad_s"]
            ),
        )


@dataclass(frozen=True)
class RuntimeModelSpec:
    """Numerical and evidence contract required to execute a fitted model."""

    sample_period_s: float
    validity_envelope: ModelValidityEnvelope
    certified_prediction_horizon_s: float | None = None
    certification_source: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        if self.certified_prediction_horizon_s is None:
            if self.certification_source is not None:
                raise ValueError(
                    "certification_source requires a certified prediction horizon"
                )
        else:
            if (
                not np.isfinite(self.certified_prediction_horizon_s)
                or self.certified_prediction_horizon_s <= 0.0
            ):
                raise ValueError(
                    "certified_prediction_horizon_s must be finite and positive"
                )
            if not self.certification_source or not self.certification_source.strip():
                raise ValueError(
                    "a certified prediction horizon requires certification_source"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_period_s": self.sample_period_s,
            "validity_envelope": self.validity_envelope.to_dict(),
            "certified_prediction_horizon_s": self.certified_prediction_horizon_s,
            "certification_source": self.certification_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeModelSpec:
        return cls(
            sample_period_s=float(payload["sample_period_s"]),
            validity_envelope=ModelValidityEnvelope.from_dict(
                payload["validity_envelope"]
            ),
            certified_prediction_horizon_s=(
                None
                if payload.get("certified_prediction_horizon_s") is None
                else float(payload["certified_prediction_horizon_s"])
            ),
            certification_source=(
                None
                if payload.get("certification_source") is None
                else str(payload["certification_source"])
            ),
        )


def runtime_spec_from_fit_report(
    report: Mapping[str, Any],
    *,
    model_name: str = "learned_lag",
    certified_prediction_horizon_s: float | None = None,
    certification_source: str | None = None,
) -> RuntimeModelSpec:
    """Extract the runtime period and training envelope from a fit report."""

    try:
        if "dataset" in report:
            sample_rate_hz = float(report["dataset"]["sample_rate_hz"])
            envelope = report["models"][model_name]["fit"]["rollout_loss"][
                "dynamic_envelope"
            ]
        else:
            horizon_steps = int(report["configuration"]["horizon_steps"])
            horizon_duration_s = float(report["configuration"]["horizon_duration_s"])
            sample_rate_hz = horizon_steps / horizon_duration_s
            envelope = report["fit"]["rollout_loss"]["dynamic_envelope"]
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError(
            f"fit report does not contain runtime data for model {model_name!r}"
        ) from error
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("fit report sample rate must be finite and positive")
    return RuntimeModelSpec(
        sample_period_s=1.0 / sample_rate_hz,
        validity_envelope=ModelValidityEnvelope.from_dict(envelope),
        certified_prediction_horizon_s=certified_prediction_horizon_s,
        certification_source=certification_source,
    )


def _wind_world(trajectory: Trajectory) -> np.ndarray:
    wind = np.zeros((len(trajectory.states), 3), dtype=np.float64)
    roles = trajectory.spec.exogenous_roles
    for axis, role in enumerate(("wind_north", "wind_west", "wind_up")):
        if role in roles:
            wind[:, axis] = trajectory.exogenous[:, roles.index(role)]
    return wind


def _robust_half_width(values: np.ndarray, *, floor: float) -> np.ndarray:
    median = np.median(values, axis=0)
    deviation = values - median
    robust_scale = np.maximum(
        np.median(np.abs(deviation), axis=0) / 0.6744897501960817,
        floor,
    )
    return np.maximum(
        np.quantile(np.abs(deviation), 0.995, axis=0),
        4.0 * robust_scale,
    )


def runtime_spec_from_trajectory(
    trajectory: Trajectory,
    *,
    certified_prediction_horizon_s: float | None = None,
    certification_source: str | None = None,
) -> RuntimeModelSpec:
    """Build a runtime contract for synthetic or externally supplied models."""

    quaternions = jnp.asarray(trajectory.states[:, 6:10])
    rotations = np.asarray(jax.vmap(quaternion_to_rotation)(quaternions))
    relative_velocity = trajectory.states[:, 3:6] - _wind_world(trajectory)
    body_velocity = np.einsum("nji,nj->ni", rotations, relative_velocity)
    angular_velocity = trajectory.states[:, 10:13]
    body_center = np.median(body_velocity, axis=0)
    angular_center = np.median(angular_velocity, axis=0)
    return RuntimeModelSpec(
        sample_period_s=trajectory.nominal_dt_s,
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=tuple(body_center),
            body_velocity_half_width_m_s=tuple(
                _robust_half_width(body_velocity, floor=0.1)
            ),
            angular_velocity_center_rad_s=tuple(angular_center),
            angular_velocity_half_width_rad_s=tuple(
                _robust_half_width(angular_velocity, floor=0.1)
            ),
        ),
        certified_prediction_horizon_s=certified_prediction_horizon_s,
        certification_source=certification_source,
    )


@runtime_checkable
class ActuationMap(Protocol):
    """JAX-compatible mapping from bounded commands to model input channels."""

    command_channels: tuple[ControlChannel, ...]
    model_control_size: int

    def model_control(self, command: Array) -> Array:
        """Map one command vector into the fitted model's control coordinates."""


@dataclass(frozen=True)
class DirectActuationMap:
    """Identity command mapping for explicitly actionable model inputs."""

    command_channels: tuple[ControlChannel, ...]

    def __post_init__(self) -> None:
        channels = tuple(self.command_channels)
        unsupported = [
            channel.semantic
            for channel in channels
            if channel.semantic not in ACTIONABLE_CONTROL_SEMANTICS
        ]
        if unsupported:
            raise NonActionableModelError(
                "direct NMPC actuation requires command semantics; got "
                + ", ".join(unsupported)
            )
        missing_bounds = [
            channel.name
            for channel in channels
            if channel.minimum is None or channel.maximum is None
        ]
        if missing_bounds:
            raise NonActionableModelError(
                "direct NMPC actuation requires finite bounds for "
                + ", ".join(missing_bounds)
            )
        object.__setattr__(self, "command_channels", channels)

    @property
    def model_control_size(self) -> int:
        return len(self.command_channels)

    def model_control(self, command: Array) -> Array:
        return jnp.asarray(command)


@dataclass(frozen=True)
class RuntimeDynamicsModel:
    """A fitted model bound to its executable timing and actuation contract."""

    params: ModelParams
    input_spec: TrajectorySpec
    runtime_spec: RuntimeModelSpec
    actuation: ActuationMap

    def __post_init__(self) -> None:
        family = model_family(self.params)
        if self.input_spec.vehicle.family != family.platform:
            raise ValueError("runtime input spec does not match model family")
        family.validate_control_schema(
            self.input_spec.control_names, self.input_spec.control_roles
        )
        if self.actuation.model_control_size != len(self.input_spec.controls):
            raise ValueError(
                "actuation map output size does not match model control size"
            )
        if not self.actuation.command_channels:
            raise ValueError("actuation map needs at least one command channel")
        for channel in self.actuation.command_channels:
            if channel.minimum is None or channel.maximum is None:
                raise ValueError("NMPC command channels require finite bounds")
            if (
                not np.isfinite(channel.minimum)
                or not np.isfinite(channel.maximum)
                or channel.minimum >= channel.maximum
            ):
                raise ValueError(f"invalid command bounds for channel {channel.name!r}")
        minimum = np.asarray(
            [channel.minimum for channel in self.actuation.command_channels]
        )
        maximum = np.asarray(
            [channel.maximum for channel in self.actuation.command_channels]
        )
        expected_shape = (len(self.input_spec.controls),)
        midpoint = 0.5 * (minimum + maximum)
        for sample in (minimum, midpoint, maximum):
            try:
                mapped = np.asarray(
                    jax.jit(self.actuation.model_control)(jnp.asarray(sample))
                )
            except Exception as error:
                raise ValueError("actuation map must be JAX-traceable") from error
            if mapped.shape != expected_shape:
                raise ValueError(
                    "actuation map produced shape "
                    f"{mapped.shape}, expected {expected_shape}"
                )
            if not np.all(np.isfinite(mapped)):
                raise ValueError("actuation map produced non-finite model controls")
        try:
            jacobian = np.asarray(
                jax.jacrev(self.actuation.model_control)(jnp.asarray(midpoint))
            )
        except Exception as error:
            raise ValueError("actuation map must be JAX-differentiable") from error
        expected_jacobian_shape = expected_shape + (len(minimum),)
        if jacobian.shape != expected_jacobian_shape or not np.all(
            np.isfinite(jacobian)
        ):
            raise ValueError("actuation map produced an invalid command Jacobian")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        actuation: ActuationMap | None = None,
    ) -> RuntimeDynamicsModel:
        from glassbox.model_io import load_dynamics_model

        params, payload = load_dynamics_model(path)
        input_spec = TrajectorySpec.from_dict(payload["input_spec"])
        runtime_spec = RuntimeModelSpec.from_dict(payload["runtime_spec"])
        if actuation is None:
            actuation = DirectActuationMap(input_spec.controls)
        return cls(params, input_spec, runtime_spec, actuation)

    @property
    def command_size(self) -> int:
        return len(self.actuation.command_channels)

    @property
    def command_minimum(self) -> Array:
        return jnp.asarray(
            [channel.minimum for channel in self.actuation.command_channels]
        )

    @property
    def command_maximum(self) -> Array:
        return jnp.asarray(
            [channel.maximum for channel in self.actuation.command_channels]
        )

    @property
    def exogenous_size(self) -> int:
        return len(self.input_spec.exogenous)

    @property
    def latent_size(self) -> int:
        return len(self.input_spec.controls) + (
            3 if model_family(self.params).platform == "multirotor" else 0
        )

    def initial_latent_state(self, command_history: Array) -> Array:
        history = jnp.asarray(command_history)
        if history.ndim == 1:
            history = history[jnp.newaxis, :]
        if history.ndim != 2 or history.shape[1] != self.command_size:
            raise ValueError("command history must have shape (time, command_size)")
        model_history = jax.vmap(self.actuation.model_control)(history)
        return control_state_after_history(
            self.params,
            model_history,
            self.runtime_spec.sample_period_s,
            self.input_spec.control_roles,
        )

    def transition(
        self,
        state: Array,
        latent_state: Array,
        command: Array,
        exogenous: Array | None = None,
    ) -> tuple[Array, Array]:
        return self.transition_at_interval(
            state,
            latent_state,
            command,
            self.runtime_spec.sample_period_s,
            exogenous,
        )

    def transition_at_interval(
        self,
        state: Array,
        latent_state: Array,
        command: Array,
        interval_s: float,
        exogenous: Array | None = None,
    ) -> tuple[Array, Array]:
        """Advance the continuous dynamics across one explicit interval."""

        if not np.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("runtime transition interval must be finite and positive")
        if command.shape[-1] != self.command_size:
            raise ValueError("command does not match runtime command size")
        if exogenous is None:
            exogenous = jnp.zeros(self.exogenous_size)
        if exogenous.shape[-1] != self.exogenous_size:
            raise ValueError("exogenous input does not match runtime spec")
        return step_with_latent(
            self.params,
            state,
            latent_state,
            self.actuation.model_control(command),
            interval_s,
            self.input_spec.control_roles,
            exogenous,
            self.input_spec.exogenous_roles,
        )

    def validity_utilization(
        self,
        state: Array,
        exogenous: Array | None = None,
    ) -> Array:
        """Return per-axis utilization of the fitted dynamic envelope."""

        if exogenous is None:
            exogenous = jnp.zeros(self.exogenous_size)
        if exogenous.shape[-1] != self.exogenous_size:
            raise ValueError("exogenous input does not match runtime spec")
        roles = self.input_spec.exogenous_roles
        wind = jnp.stack(
            tuple(
                exogenous[roles.index(role)] if role in roles else jnp.asarray(0.0)
                for role in ("wind_north", "wind_west", "wind_up")
            )
        )
        rotation = quaternion_to_rotation(state[6:10])
        body_velocity = rotation.T @ (state[3:6] - wind)
        envelope = self.runtime_spec.validity_envelope
        return jnp.concatenate(
            (
                jnp.abs(body_velocity - jnp.asarray(envelope.body_velocity_center_m_s))
                / jnp.asarray(envelope.body_velocity_half_width_m_s),
                jnp.abs(
                    state[10:13] - jnp.asarray(envelope.angular_velocity_center_rad_s)
                )
                / jnp.asarray(envelope.angular_velocity_half_width_rad_s),
            )
        )
