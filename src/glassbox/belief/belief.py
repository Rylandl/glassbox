"""Executable dynamics with local parameter information and held-out forecast error."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.belief.forecast_error import ForecastErrorEnvelope
from glassbox.belief.information import ParameterInformation
from glassbox.core.data import TrajectorySpec
from glassbox.core.dynamics import (
    ModelParams,
    control_state_after_history,
    rollout_with_latent,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import TANGENT_STATE_SIZE, rigid_body_local_error
from glassbox.core.model import (
    ActuationMap,
    ExecutableModel,
    ModelValidityEnvelope,
    RuntimeModelSpec,
    commands_within_declared_bounds,
)

if TYPE_CHECKING:
    from glassbox.belief.update import UpdateResult
    from glassbox.core.data import Trajectory


@dataclass(frozen=True)
class PredictiveTrajectory:
    """One rollout with empirical error and propagated parameter information."""

    states: Array
    latent_states: Array
    commands: Array
    forecast_error_covariance: Array
    parameter_covariance: Array
    validity_utilization: Array
    forecast_error_available: bool
    forecast_error_horizon_supported: bool
    parameter_information_rank: int
    parameter_information_complete: bool
    unresolved_parameter_basis: Array

    @property
    def uncertainty_available(self) -> bool:
        return self.forecast_error_available or self.parameter_information_rank > 0


@dataclass(frozen=True)
class DynamicsBelief:
    """One executable model, what the evidence resolved, and its error envelope.

    ``information`` may be omitted at construction, in which case the belief
    starts at rank zero: a point estimate that claims nothing. It is never
    ``None`` afterwards. ``forecast_error`` is ``None`` when no held-out
    evidence has been measured, which is different from a zero envelope and is
    reported as such rather than as certainty.

    Declared command bounds are enforced on every concrete rollout: a command
    outside them raises and names the channel. The validity envelope stays
    advisory. :attr:`PredictiveTrajectory.validity_utilization` reports how far
    a forecast leaves the training support and the caller decides, because an
    unsupported forecast is still a forecast while an unexecutable command is
    not a command.
    """

    model: ExecutableModel
    information: ParameterInformation | None = None
    forecast_error: ForecastErrorEnvelope | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = structured_parameter_names(self.params)
        information = self.information
        if information is None:
            information = ParameterInformation.unknown(self.params)
        if information.names != names:
            raise ValueError(
                "parameter information does not match the structured parameters"
            )
        object.__setattr__(self, "information", information)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def params(self) -> ModelParams:
        return self.model.params

    @property
    def input_spec(self) -> TrajectorySpec:
        return self.model.input_spec

    @property
    def runtime_spec(self) -> RuntimeModelSpec:
        return self.model.runtime_spec

    @property
    def support(self) -> ModelValidityEnvelope:
        """Return the operating region the model's evidence supports."""

        return self.model.runtime_spec.validity_envelope

    @property
    def sample_period_s(self) -> float:
        return self.model.runtime_spec.sample_period_s

    @property
    def forecast_error_available(self) -> bool:
        return self.forecast_error is not None

    @property
    def uncertainty_available(self) -> bool:
        assert self.information is not None
        return self.forecast_error is not None or self.information.resolved_rank() > 0

    @property
    def parameter_information_complete(self) -> bool:
        assert self.information is not None
        return self.information.complete

    @property
    def maximum_error_horizon_s(self) -> float | None:
        return (
            None
            if self.forecast_error is None
            else self.forecast_error.maximum_horizon_s
        )

    @property
    def update_count(self) -> int:
        """Return how many telemetry blocks this belief has absorbed."""

        return int(self.provenance.get("update_count", 0))

    @property
    def parameter_distance_since_measurement(self) -> float:
        """Return how far the parameters moved since the envelope was measured.

        The distance is the accumulated normalized structured-parameter step
        length, so it is comparable across coordinates of different physical
        units. Nothing gates on it; it is the number a caller reads to decide
        whether an envelope measured around older parameters still describes
        the model in hand.
        """

        return float(self.provenance.get("parameter_distance_since_measurement", 0.0))

    def error_covariance(self, horizon_s: Array | float) -> Array:
        """Return the held-out forecast-error covariance at one horizon."""

        if self.forecast_error is None:
            return jnp.zeros((TANGENT_STATE_SIZE, TANGENT_STATE_SIZE))
        return self.forecast_error.covariance_at(horizon_s)

    def _rollout_with_params(
        self,
        params: ModelParams,
        initial_state: Array,
        commands: Array,
        command_history: Array,
        initial_latent_state: Array | None,
        exogenous: Array,
    ) -> tuple[Array, Array]:
        model_controls = jax.vmap(self.model.actuation_map.model_control)(commands)
        if initial_latent_state is None:
            history_controls = jax.vmap(self.model.actuation_map.model_control)(
                command_history
            )
            initial_latent = control_state_after_history(
                params,
                history_controls,
                self.model.runtime_spec.sample_period_s,
                self.model.input_spec.control_roles,
            )
        else:
            initial_latent = initial_latent_state

        return rollout_with_latent(
            params,
            initial_state,
            model_controls,
            self.model.runtime_spec.sample_period_s,
            initial_motor_state=initial_latent,
            control_roles=self.model.input_spec.control_roles,
            exogenous=exogenous,
            exogenous_roles=self.model.input_spec.exogenous_roles,
        )

    def _parameter_covariance(
        self,
        params: ModelParams,
        initial_state: Array,
        commands: Array,
        command_history: Array,
        initial_latent_state: Array | None,
        exogenous: Array,
        future_states: Array,
    ) -> Array:
        """Propagate the resolved parameter covariance along one rollout.

        Written through a factor of the covariance, the contribution is a sum
        of one directional derivative per resolved direction, so a belief that
        resolves two directions costs two extra rollouts rather than a full
        parameter Jacobian.
        """

        assert self.information is not None
        covariance = self.information.covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        retained = eigenvalues > 0.0
        if not np.any(retained):
            return jnp.zeros(
                (len(future_states), TANGENT_STATE_SIZE, TANGENT_STATE_SIZE)
            )
        factor = eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])
        center = structured_parameter_vector(params)

        def varied_error(vector: Array) -> Array:
            varied_states, _ = self._rollout_with_params(
                with_structured_parameter_vector(params, vector),
                initial_state,
                commands,
                command_history,
                initial_latent_state,
                exogenous,
            )
            return jax.vmap(rigid_body_local_error)(future_states, varied_states[1:])

        directions = jax.vmap(
            lambda column: jax.jvp(varied_error, (center,), (column,))[1]
        )(jnp.asarray(factor.T))
        return jnp.einsum("kti,ktj->tij", directions, directions)

    def rollout(
        self,
        initial_state: Array,
        commands: Array,
        *,
        model_parameters: ModelParams | None = None,
        command_history: Array | None = None,
        initial_latent_state: Array | None = None,
        exogenous: Array | None = None,
    ) -> PredictiveTrajectory:
        """Roll out the model and attach the covariances the belief supports.

        Concrete commands and command history must lie within the declared
        channel bounds; see ``commands_within_declared_bounds``.
        """

        commands = jnp.asarray(commands)
        if commands.ndim != 2 or commands.shape[1] != self.model.command_size:
            raise ValueError("commands must have shape (time, command_size)")
        if len(commands) < 1:
            raise ValueError("belief rollout requires at least one command")
        commands = commands_within_declared_bounds(
            commands,
            self.model.actuation_map.command_channels,
        )
        if exogenous is None:
            exogenous = jnp.zeros((len(commands), self.model.exogenous_size))
        else:
            exogenous = jnp.asarray(exogenous)
        if exogenous.shape != (len(commands), self.model.exogenous_size):
            raise ValueError("exogenous forecast does not match command timeline")
        initial_state = jnp.asarray(initial_state)
        history = (
            commands[0:1] if command_history is None else jnp.asarray(command_history)
        )
        if history.ndim == 1:
            history = history[None, :]
        if history.ndim != 2 or history.shape[1] != self.model.command_size:
            raise ValueError("command history must have shape (time, command_size)")
        history = commands_within_declared_bounds(
            history,
            self.model.actuation_map.command_channels,
            label="command history",
        )
        provided_latent = (
            None if initial_latent_state is None else jnp.asarray(initial_latent_state)
        )
        selected_parameters = (
            self.model.params if model_parameters is None else model_parameters
        )
        states, latent_states = self._rollout_with_params(
            selected_parameters,
            initial_state,
            commands,
            history,
            provided_latent,
            exogenous,
        )
        future_states = states[1:]
        horizons = self.model.runtime_spec.sample_period_s * jnp.arange(
            1, len(commands) + 1
        )
        forecast_covariance = jax.vmap(self.error_covariance)(horizons)
        parameter_covariance = self._parameter_covariance(
            selected_parameters,
            initial_state,
            commands,
            history,
            provided_latent,
            exogenous,
            future_states,
        )
        zero = jnp.zeros((1, TANGENT_STATE_SIZE, TANGENT_STATE_SIZE))
        initial_context = exogenous[0]
        validity = jnp.concatenate(
            (
                self.model.validity_utilization(initial_state, initial_context)[
                    None, :
                ],
                jax.vmap(self.model.validity_utilization)(future_states, exogenous),
            )
        )
        maximum_horizon = self.maximum_error_horizon_s
        assert self.information is not None
        return PredictiveTrajectory(
            states=states,
            latent_states=latent_states,
            commands=commands,
            forecast_error_covariance=jnp.concatenate((zero, forecast_covariance)),
            parameter_covariance=jnp.concatenate((zero, parameter_covariance)),
            validity_utilization=validity,
            forecast_error_available=self.forecast_error is not None,
            forecast_error_horizon_supported=(
                maximum_horizon is not None
                and len(commands) * self.model.runtime_spec.sample_period_s
                <= maximum_horizon + 1e-12
            ),
            parameter_information_rank=self.information.resolved_rank(),
            parameter_information_complete=self.information.complete,
            unresolved_parameter_basis=jnp.asarray(
                self.information.unresolved_subspace()
            ),
        )

    def absorb(self, telemetry: Trajectory) -> tuple[DynamicsBelief, UpdateResult]:
        """Update with fresh, nonoverlapping telemetry."""

        from glassbox.belief.update import absorb

        return absorb(self, telemetry)

    def save(self, path: str | Path) -> None:
        from glassbox.belief.belief_io import save_dynamics_belief

        save_dynamics_belief(self, path)

    @classmethod
    def load(
        cls, path: str | Path, *, actuation: ActuationMap | None = None
    ) -> DynamicsBelief:
        from glassbox.belief.belief_io import load_dynamics_belief

        return load_dynamics_belief(path, actuation=actuation)
