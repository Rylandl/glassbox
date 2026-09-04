"""The plan model a dynamics belief presents to a bounded solver.

This module is the control boundary. Everything that knows what a
:class:`~glassbox.belief.belief.DynamicsBelief` is lives here; nothing on the
other side of :class:`~glassbox.control.plan.PlanModel` does. One adapter
serves both families: a belief fitted from a corpus and a belief the in-flight
identifier built from nothing state their mean, their resolved parameter
covariance and their forecast-error evidence in the same three objects, so the
solver never learns which it is planning over.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.plan import (
    PlanMeasurements,
    PlanValues,
    Prediction,
    SafetyEnvelope,
    SolverPolicy,
    TrackingTolerances,
    maintained_block_count,
)
from glassbox.control.solver import BoundedShootingSolver
from glassbox.core.data import duration_to_steps
from glassbox.core.dynamics import (
    ModelParams,
    quaternion_to_rotation,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.model import ExecutableModel, NonActionableModelError

_MINIMUM_COVARIANCE_EIGENVALUE_FRACTION = 1e-10


def _marginal_standard_deviation(variance: Array) -> Array:
    """Square root of a variance whose gradient is finite where it vanishes.

    The square root has an infinite slope at zero, and a point model predicts
    exactly zero spread on every axis at every stage. Differentiating the naive
    expression therefore hands the line search a non-finite gradient on the one
    case that must reduce to the point objective. The masked form returns zero
    with a zero derivative there, which is the derivative of the constant the
    function actually is when the belief carries no covariance.
    """

    positive = variance > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, variance, 1.0)), 0.0)


def parameter_covariance_factor(belief: DynamicsBelief) -> np.ndarray | None:
    """Return a factor ``L`` with ``L @ L.T`` equal to the parameter covariance.

    The objective charges predicted spread, and the parameter contribution to
    that spread is ``J C J.T`` for the plan's parameter Jacobian ``J``. Written
    through a factor it becomes a sum of ``len(L.T)`` directional derivatives,
    each one forward-mode rollout, so a covariance the evidence resolved along
    one direction costs one extra rollout instead of a full Jacobian.

    ``None`` means the belief resolves no parameter direction at all, which is
    a point model: it contributes exactly zero spread and is priced by the
    point objective. A direction the evidence did not resolve is absent from
    the factor rather than present with a large variance, because the
    information state says nothing about it and inventing a spread there would
    be an assumption the belief does not hold.
    """

    assert belief.information is not None
    covariance = belief.information.covariance()
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = _MINIMUM_COVARIANCE_EIGENVALUE_FRACTION * max(float(eigenvalues[-1]), 0.0)
    retained = eigenvalues > max(floor, 0.0)
    if not np.any(retained):
        return None
    return eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])


def default_solver_policy(model: ExecutableModel) -> SolverPolicy:
    """The maintained horizon and command-block layout for one model.

    The horizon is a property of the vehicle rather than of how its model was
    obtained, so both multirotor families share one.
    """

    dt_s = model.runtime_spec.sample_period_s
    fixed_wing = model.input_spec.vehicle.family == "fixedwing"
    target_horizon_s = 1.0 if fixed_wing else 0.6
    maximum_steps = 50 if fixed_wing else 40
    steps = min(maximum_steps, max(2, duration_to_steps(target_horizon_s, dt_s)))
    return SolverPolicy(
        horizon_steps=steps,
        block_count=maintained_block_count(steps),
    )


def _plan_values(belief: DynamicsBelief, policy: SolverPolicy) -> PlanValues:
    """Read one belief's numbers out into the arguments the kernels take.

    All three are read once, here, rather than inside a traced kernel. The
    forecast-error covariance is evaluated at the horizon's own stage times,
    which the policy fixes, so the envelope's interpolation table never enters
    a compiled kernel as a constant and a re-measured envelope of the same
    length costs no recompile.
    """

    horizons = belief.model.runtime_spec.sample_period_s * jnp.arange(
        1, policy.horizon_steps + 1
    )
    factor = parameter_covariance_factor(belief)
    return PlanValues(
        parameters=belief.model.params,
        covariance_factor=None if factor is None else jnp.asarray(factor),
        forecast_error_covariance=jax.vmap(belief.error_covariance)(horizons),
    )


def _compile_signature(
    belief: DynamicsBelief,
    tolerances: TrackingTolerances,
    safety_envelope: SafetyEnvelope,
    policy: SolverPolicy,
    values: PlanValues,
) -> str:
    """Digest the static structure a compiled kernel for this plan model traces.

    No belief value reaches this digest. The parameters, the parameter
    covariance factor and the stage forecast-error covariance all travel
    through every kernel as :class:`PlanValues`, so a belief that absorbs
    telemetry every control interval keeps the code compiled for the belief it
    came from. What is here is what the trace itself depends on: the model's
    declared specs, the tolerances, the envelope and the policy the objective
    is written from, the parameter tree's structure and leaf shapes, whether
    the belief resolves any parameter direction at all and how wide the factor
    of that resolution is, and the rank the information state says it
    resolved. A kernel compiled for one of those cannot answer for another.
    """

    digest = hashlib.sha256()

    def add(value: object) -> None:
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\x00")

    model = belief.model
    add(json.dumps(model.input_spec.to_dict(), sort_keys=True))
    add(json.dumps(model.runtime_spec.to_dict(), sort_keys=True))
    add(type(model.actuation).__name__)
    add(tolerances)
    add(safety_envelope)
    add(policy)
    leaves, structure = jax.tree_util.tree_flatten(model.params)
    add(structure)
    for leaf in leaves:
        array = np.asarray(leaf)
        add((array.shape, str(array.dtype)))
    add(values.forecast_error_covariance.shape)
    if values.covariance_factor is None:
        add("no_parameter_covariance")
    else:
        add(values.covariance_factor.shape)
    assert belief.information is not None
    add(belief.information.resolved_rank())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class BeliefPlanModel:
    """One belief seen as something a bounded solver can plan over.

    The belief supplies the mean the plan is rolled out through and the spread
    the plan is charged for. Both robustness terms in :meth:`stage_cost` are
    exactly zero when the belief carries no covariance, so a point model, and
    equally a bootstrap belief that has resolved nothing yet, is priced by the
    point objective.
    """

    belief: DynamicsBelief
    tolerances: TrackingTolerances
    safety_envelope: SafetyEnvelope
    policy: SolverPolicy
    values: PlanValues
    compile_signature: str

    @property
    def model(self) -> ExecutableModel:
        return self.belief.model

    @property
    def horizon_steps(self) -> int:
        return self.policy.horizon_steps

    @property
    def block_count(self) -> int:
        return self.policy.block_count

    @property
    def sample_period_s(self) -> float:
        return self.model.runtime_spec.sample_period_s

    @property
    def uncertainty_available(self) -> bool:
        return self.belief.uncertainty_available

    @property
    def command_size(self) -> int:
        return self.model.command_size

    @property
    def exogenous_size(self) -> int:
        return self.model.exogenous_size

    @property
    def latent_size(self) -> int:
        return self.model.latent_size

    @property
    def command_minimum(self) -> Array:
        return self.model.command_minimum

    @property
    def command_maximum(self) -> Array:
        return self.model.command_maximum

    def initial_latent(self, command_history: Array, values: PlanValues) -> Array:
        return self.model.initial_latent_state_with_parameters(
            values.parameters, command_history
        )

    def _expand_normalized_blocks(self, blocks: Array) -> Array:
        """Hold each block over its model steps, covering the whole horizon.

        The maintained layout divides the horizon exactly, so the trailing
        slice is a no-op. It only ever shortens the final block of a horizon
        whose length admits no usable divisor, and every block still drives at
        least one prediction step.
        """

        expanded = jnp.repeat(blocks, self.policy.block_steps, axis=0)
        return expanded[: self.horizon_steps]

    def _commands_from_normalized(self, normalized: Array) -> Array:
        minimum = self.command_minimum
        command_range = self.command_maximum - minimum
        return minimum + 0.5 * (jnp.clip(normalized, -1.0, 1.0) + 1.0) * command_range

    def _mean_rollout(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        parameters: ModelParams,
    ) -> tuple[Array, Array, Array]:
        """Predict the horizon one plan drives through one parameter vector.

        This is the mean only. Both parts of the spread are added by
        :meth:`rollout`: the belief's stage forecast-error covariance travels
        with the plan values, and the parameter contribution depends on the
        plan and is computed by :meth:`_tangent_covariance`.
        """

        model = self.model
        normalized_commands = self._expand_normalized_blocks(blocks)
        commands = self._commands_from_normalized(normalized_commands)

        def transition(
            carry: tuple[Array, Array], inputs: tuple[Array, Array]
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            state, latent = carry
            command, context = inputs
            next_state, next_latent = model.transition_at_interval_with_parameters(
                parameters,
                state,
                latent,
                command,
                model.runtime_spec.sample_period_s,
                context,
            )
            return (next_state, next_latent), (next_state, next_latent)

        _, (future_states, future_latent) = jax.lax.scan(
            transition,
            (initial_state, initial_latent),
            (commands, exogenous),
        )
        states = jnp.concatenate((initial_state[None, :], future_states), axis=0)
        latent = jnp.concatenate((initial_latent[None, :], future_latent), axis=0)
        return states, latent, commands

    def _tangent_covariance(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        values: PlanValues,
        states: Array,
    ) -> Array:
        """Add the plan's parameter spread to the forecast-error covariance.

        The parameter contribution is ``J C J.T`` for the Jacobian of the
        predicted tangent error with respect to the fitted parameters. Written
        through the covariance factor it is the sum of the outer products of
        one directional derivative per retained direction, and each of those is
        a single forward-mode rollout rather than a column of a full Jacobian.
        """

        error_covariance = values.forecast_error_covariance
        if values.covariance_factor is None:
            return error_covariance
        parameters = values.parameters
        center = structured_parameter_vector(parameters)

        def varied_error(vector: Array) -> Array:
            varied_parameters = with_structured_parameter_vector(parameters, vector)
            varied_states, _, _ = self._mean_rollout(
                blocks,
                initial_state,
                initial_latent,
                exogenous,
                varied_parameters,
            )
            return jax.vmap(rigid_body_local_error)(states[1:], varied_states[1:])

        def direction(column: Array) -> Array:
            return jax.jvp(varied_error, (center,), (column,))[1]

        directions = jax.vmap(direction)(values.covariance_factor.T)
        return error_covariance + jnp.einsum("kti,ktj->tij", directions, directions)

    def rollout(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        values: PlanValues,
    ) -> Prediction:
        """Predict the horizon this plan drives, with its tangent covariance."""

        states, latent, commands = self._mean_rollout(
            blocks,
            initial_state,
            initial_latent,
            exogenous,
            values.parameters,
        )
        covariance = self._tangent_covariance(
            blocks,
            initial_state,
            initial_latent,
            exogenous,
            values,
            states,
        )
        return Prediction(
            mean_states=states,
            tangent_covariance=covariance,
            commands=commands,
            latent_states=latent,
            exogenous=exogenous,
        )

    def _validity_utilization(self, state: Array, exogenous: Array) -> Array:
        return self.model.validity_utilization(state, exogenous)

    def _robust_validity_utilization(
        self,
        mean_state: Array,
        covariance: Array,
        exogenous: Array,
    ) -> Array:
        """Return per-axis validity utilization widened by its own spread.

        The envelope is stated on body velocity and body rates, so the tangent
        covariance is mapped onto those six features and the marginal standard
        deviation of each is added to that feature's mean utilization. A belief
        with no covariance adds exactly zero and the metric is the mean
        utilization it has always been.
        """

        envelope = self.model.runtime_spec.validity_envelope
        mean_utilization = self.model.validity_utilization(mean_state, exogenous)

        roles = self.model.input_spec.exogenous_roles
        wind = jnp.stack(
            tuple(
                exogenous[roles.index(role)] if role in roles else jnp.asarray(0.0)
                for role in ("wind_north", "wind_west", "wind_up")
            )
        )
        rotation = quaternion_to_rotation(mean_state[6:10])
        body_velocity = rotation.T @ (mean_state[3:6] - wind)
        body_velocity_cross = jnp.asarray(
            (
                (0.0, -body_velocity[2], body_velocity[1]),
                (body_velocity[2], 0.0, -body_velocity[0]),
                (-body_velocity[1], body_velocity[0], 0.0),
            )
        )
        feature_jacobian = jnp.zeros((6, 12))
        feature_jacobian = feature_jacobian.at[0:3, 3:6].set(rotation.T)
        feature_jacobian = feature_jacobian.at[0:3, 6:9].set(body_velocity_cross)
        feature_jacobian = feature_jacobian.at[3:6, 9:12].set(jnp.eye(3))
        feature_covariance = feature_jacobian @ covariance @ feature_jacobian.T
        feature_half_width = jnp.concatenate(
            (
                jnp.asarray(envelope.body_velocity_half_width_m_s),
                jnp.asarray(envelope.angular_velocity_half_width_rad_s),
            )
        )
        marginal_radius = (
            _marginal_standard_deviation(jnp.diag(feature_covariance))
            / feature_half_width
        )
        return mean_utilization + marginal_radius

    def _safety_violation(self, state: Array) -> Array:
        envelope = self.safety_envelope
        violations: list[Array] = []
        position_scale = self.tolerances.local_state_scale[0:3]
        if envelope.minimum_position_m is not None:
            violations.append(
                jax.nn.relu(jnp.asarray(envelope.minimum_position_m) - state[0:3])
                / position_scale
            )
        if envelope.maximum_position_m is not None:
            violations.append(
                jax.nn.relu(state[0:3] - jnp.asarray(envelope.maximum_position_m))
                / position_scale
            )
        if envelope.maximum_speed_m_s is not None:
            violations.append(
                jnp.atleast_1d(
                    jax.nn.relu(
                        jnp.sqrt(jnp.sum(jnp.square(state[3:6])) + 1e-12)
                        - envelope.maximum_speed_m_s
                    )
                    / envelope.maximum_speed_m_s
                )
            )
        if envelope.maximum_angular_velocity_rad_s is not None:
            violations.append(
                jnp.atleast_1d(
                    jax.nn.relu(
                        jnp.sqrt(jnp.sum(jnp.square(state[10:13])) + 1e-12)
                        - envelope.maximum_angular_velocity_rad_s
                    )
                    / envelope.maximum_angular_velocity_rad_s
                )
            )
        return jnp.concatenate(violations) if violations else jnp.zeros(1)

    def stage_cost(
        self,
        prediction: Prediction,
        reference_states: Array,
        previous_command: Array,
        policy: SolverPolicy,
    ) -> Array:
        """Score one command plan as the expected cost of its own forecast.

        Tracking charges ``E[l] = l(mean) + trace(W Sigma)`` at every predicted
        stage, where ``W`` is the diagonal tracking weight built from the
        declared tolerances and ``Sigma`` is the belief's predicted tangent
        covariance. Validity charges the robust utilization, the mean
        utilization plus the belief's own marginal radius on the same envelope
        features. A belief that carries no covariance contributes exactly zero
        to both, so a point model is scored by the point objective.
        """

        states = prediction.mean_states
        commands = prediction.commands
        covariance = prediction.tangent_covariance
        exogenous = prediction.exogenous
        local_error = jax.vmap(rigid_body_local_error)(reference_states[1:], states[1:])
        normalized_error = local_error / self.tolerances.local_state_scale
        normalized_spread = jnp.sum(
            jnp.diagonal(covariance, axis1=-2, axis2=-1)
            / jnp.square(self.tolerances.local_state_scale),
            axis=1,
        )
        tracking_cost = jnp.mean(
            jnp.sum(jnp.square(normalized_error), axis=1) + normalized_spread
        )
        terminal_cost = policy.terminal_weight * (
            jnp.sum(jnp.square(normalized_error[-1])) + normalized_spread[-1]
        )

        command_range = self.command_maximum - self.command_minimum
        command_delta = jnp.diff(
            jnp.concatenate((previous_command[None, :], commands), axis=0), axis=0
        )
        normalized_delta = command_delta / (
            policy.command_change_fraction * command_range
        )
        smoothness_cost = policy.command_change_weight * jnp.mean(
            jnp.square(normalized_delta)
        )

        utilization = jax.vmap(self._robust_validity_utilization)(
            states[1:], covariance, exogenous
        )
        validity_cost = policy.validity_weight * jnp.mean(
            jnp.square(jax.nn.relu(utilization - 1.0))
        )
        safety_violation = jax.vmap(self._safety_violation)(states[1:])
        safety_cost = policy.safety_weight * jnp.mean(jnp.square(safety_violation))
        return (
            tracking_cost
            + terminal_cost
            + smoothness_cost
            + validity_cost
            + safety_cost
        )

    def measure(self, prediction: Prediction) -> PlanMeasurements:
        """Measure the margins the result reports for one finished plan."""

        states = prediction.mean_states
        exogenous = prediction.exogenous
        standard_deviation = _marginal_standard_deviation(
            jnp.diagonal(prediction.tangent_covariance, axis1=-2, axis2=-1)
        )
        return PlanMeasurements(
            maximum_validity_utilization=jnp.max(
                jax.vmap(self._validity_utilization)(states[1:], exogenous)
            ),
            maximum_normalized_safety_violation=jnp.max(
                jax.vmap(self._safety_violation)(states[1:])
            ),
            maximum_normalized_uncertainty=jnp.max(
                standard_deviation / self.tolerances.local_state_scale[None, :]
            ),
        )


def plan_model(
    belief: DynamicsBelief,
    tolerances: TrackingTolerances,
    safety_envelope: SafetyEnvelope,
    *,
    policy: SolverPolicy | None = None,
) -> BeliefPlanModel:
    """Present one belief to a bounded solver, or refuse to.

    The belief may come from a fit or from the in-flight identifier, and
    nothing here distinguishes them. The horizon contract is settled here
    rather than in the solver, because it is a statement about evidence rather
    than about optimization: the maintained default horizon is shortened to
    the forecast-error evidence that supports it, and that envelope is the
    only thing that caps it.
    """

    model = belief.model
    if model.actuation is None:
        raise NonActionableModelError(
            "planning needs a command space; this model has no actuation map, "
            "because its inputs are observations of actuation rather than "
            "commands"
        )
    resolved = default_solver_policy(model) if policy is None else policy
    if policy is None and belief.maximum_error_horizon_s is not None:
        supported_steps = math.floor(
            belief.maximum_error_horizon_s / model.runtime_spec.sample_period_s + 1e-9
        )
        if supported_steps < 1:
            raise ValueError("predictive-error evidence is shorter than one model step")
        if supported_steps < resolved.horizon_steps:
            resolved = replace(
                resolved,
                horizon_steps=supported_steps,
                block_count=maintained_block_count(supported_steps),
            )
    values = _plan_values(belief, resolved)
    return BeliefPlanModel(
        belief=belief,
        tolerances=tolerances,
        safety_envelope=safety_envelope,
        policy=resolved,
        values=values,
        compile_signature=_compile_signature(
            belief,
            tolerances,
            safety_envelope,
            resolved,
            values,
        ),
    )


class NMPCController:
    """Opinionated NMPC over one belief or one executable model.

    A thin factory: it resolves the maintained tolerances and envelope, builds
    the plan model that is the control boundary, and hands the result to a
    :class:`~glassbox.control.solver.BoundedShootingSolver`, which is where
    every solve actually happens.
    """

    def __init__(
        self,
        model: ExecutableModel | DynamicsBelief,
        tolerances: TrackingTolerances | None = None,
        safety_envelope: SafetyEnvelope | None = None,
        *,
        policy: SolverPolicy | None = None,
    ) -> None:
        belief = model if isinstance(model, DynamicsBelief) else DynamicsBelief(model)
        self.belief = belief
        self.model = belief.model
        self.tolerances = (
            TrackingTolerances.for_platform(self.model.input_spec.vehicle.family)
            if tolerances is None
            else tolerances
        )
        self.safety_envelope = (
            SafetyEnvelope() if safety_envelope is None else safety_envelope
        )
        self.plan = plan_model(
            belief,
            self.tolerances,
            self.safety_envelope,
            policy=policy,
        )
        self.solver = BoundedShootingSolver(self.plan, self.plan.policy)

    @property
    def prediction_steps(self) -> int:
        return self.solver.prediction_steps

    @property
    def sample_period_s(self) -> float:
        return self.solver.sample_period_s

    @property
    def prediction_horizon_s(self) -> float:
        return self.solver.prediction_horizon_s

    def hold_reference(self, state, *, exogenous=None):
        """Build the common regulation reference for this controller."""

        return self.solver.hold_reference(state, exogenous=exogenous)

    def solve(self, *arguments, **keywords):
        """Optimize one bounded command and return an auditable horizon."""

        return self.solver.solve(*arguments, **keywords)
