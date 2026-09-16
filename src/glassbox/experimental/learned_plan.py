"""The generic learner seen as something a bounded NMPC solver can plan over.

This module is the control boundary for the generic learner, exactly as
:mod:`glassbox.control.fitted` is the control boundary for a dynamics belief.
Nothing on the solver's side of :class:`~glassbox.control.plan.PlanModel`
learns which one it is planning over, and nothing here changes how a belief is
presented.

Two things have to be bridged, and neither is negotiable.

*Coordinates.* The learner predicts fifteen Euclidean observation channels:
world-frame velocity, body rates, and the nine body-to-world rotation entries,
in the order the platform-tier adapter builds them. The controller plans over a
thirteen-row canonical rigid-body state. Going in, the state's quaternion
becomes rotation entries. Coming out, position is integrated trapezoidally from
the predicted world velocity, and the predicted rotation entries are projected
onto the nearest rotation and read back as a quaternion. Both directions use
the library's own geometry helpers, so nothing about rotations is reinvented
here.

*History.* The learner is recursive: a forecast means nothing without the
observed transitions before its origin. A control loop has those, interval by
interval, from the states it observes and the commands it applies, so
:class:`LearnedPlanController` carries them and hands them to the plan model
through :attr:`~glassbox.control.plan.PlanValues.observed_history`. A trial
start is a recording boundary: the history is reset and the memory starts again
at rest there, and nothing before the first observation is implied. Until the
loop has observed enough transitions to fill the explicit-difference window the
model is not usable at all, and this module says so rather than padding.

No uncertainty is claimed. The learner carries no error envelope and resolves
no parameter direction, so ``uncertainty_available`` and
``uncertainty_complete`` are both false, both robustness terms are exactly
zero, and a solve runs only under the seam's explicit no-evidence override
(``SolverPolicy.allow_unresolved_parameters``). The learner also establishes no
support envelope, so validity utilization is reported as zero, meaning *no
envelope was declared*, not *the envelope was checked and is clear*.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.control.plan import (
    PlanMeasurements,
    PlanValues,
    Prediction,
    ReferenceTrajectory,
    SafetyEnvelope,
    SolverPolicy,
    TrackingTolerances,
    maintained_block_count,
)
from glassbox.control.solver import BoundedShootingSolver
from glassbox.core.geometry import (
    nearest_rotation,
    quaternion_to_rotation_batch,
    rigid_body_local_error,
    rotation_to_quaternion,
)

from .default_model import LearnedDynamics, steps_for
from .sequence_model import KIND, SequenceModel

VELOCITY_ROWS = slice(3, 6)
QUATERNION_ROWS = slice(6, 10)
BODY_RATE_ROWS = slice(10, 13)
"""Where ``rigid_body_13_nwu_flu_wxyz_v1`` keeps each observed state group."""

OBSERVED_CHANNELS = (
    "velocity_north [m/s,world_nwu]",
    "velocity_west [m/s,world_nwu]",
    "velocity_up [m/s,world_nwu]",
    "body_rate_x [rad/s,body_flu]",
    "body_rate_y [rad/s,body_flu]",
    "body_rate_z [rad/s,body_flu]",
) + tuple(
    f"rotation_{row}{column} [unitless,body_flu_to_world_nwu]"
    for row in range(3)
    for column in range(3)
)
"""The fifteen observed channels this adapter requires a learner to declare.

The same contract the platform tier's adapter builds, in the same order. A
learner fitted on any other channel set describes something else and is
refused rather than reinterpreted.
"""

OBSERVED_SIZE = len(OBSERVED_CHANNELS)
VELOCITY_CHANNELS = slice(0, 3)
BODY_RATE_CHANNELS = slice(3, 6)
ROTATION_CHANNELS = slice(6, 15)


class ObservedHistory(NamedTuple):
    """The observed transitions a horizon starting now continues from.

    A named tuple, so it is an ordinary JAX pytree and travels through every
    compiled kernel as an argument rather than as part of its signature.
    ``states`` holds the ``delay_steps`` observations *before* the current one
    and ``commands`` the ``delay_steps`` commands applied before it, both in
    the learner's own channels. ``memory`` is the recurrent state after
    consuming every retained transition, which is what makes this short
    explicit window equivalent to consuming the whole context at once.
    """

    states: Array
    commands: Array
    memory: Array


def observed_from_state(state: Array) -> Array:
    """Return one canonical rigid-body state in the learner's channels.

    World-frame velocity, body rates, then the nine body-to-world rotation
    entries in row-major order: the same fifteen channels, built the same way,
    that the platform tier's adapter cuts every recording to.
    """

    state = jnp.asarray(state)
    rotation = quaternion_to_rotation_batch(state[QUATERNION_ROWS][None, :])[0]
    return jnp.concatenate(
        (state[VELOCITY_ROWS], state[BODY_RATE_ROWS], rotation.reshape(9))
    )


def states_from_observed(initial_state: Array, observed: Array, dt_s: float) -> Array:
    """Return the canonical states a run of predicted observations describes.

    Velocity and body rates are predicted channels. Position is not modeled, so
    it is integrated from the predicted world velocity with the trapezoidal
    rule, starting at the supplied state's own position: the same rule on the
    same grid the observations were sampled on. Attitude is not modeled as a
    rotation either, so the predicted rotation entries are projected onto the
    nearest rotation and read back as a quaternion. The supplied initial state
    is returned unchanged as the first row, because it is an observation rather
    than a prediction.
    """

    initial_state = jnp.asarray(initial_state)
    observed = jnp.asarray(observed)
    velocity = observed[:, VELOCITY_CHANNELS]
    body_rate = observed[:, BODY_RATE_CHANNELS]
    entries = observed[:, ROTATION_CHANNELS].reshape(-1, 3, 3)
    quaternion = jax.vmap(
        lambda matrix: rotation_to_quaternion(nearest_rotation(matrix))
    )(entries)
    speeds = jnp.concatenate((initial_state[VELOCITY_ROWS][None, :], velocity), axis=0)
    position = initial_state[0:3] + dt_s * jnp.cumsum(
        0.5 * (speeds[:-1] + speeds[1:]), axis=0
    )
    future = jnp.concatenate((position, velocity, quaternion, body_rate), axis=1)
    return jnp.concatenate((initial_state[None, :], future), axis=0)


def _command_bounds(name: str, values) -> np.ndarray:
    bounds = np.asarray(values, dtype=float)
    if bounds.ndim != 1 or not bounds.size or not np.all(np.isfinite(bounds)):
        raise ValueError(f"{name} must be a finite command vector")
    return bounds


@dataclass(frozen=True, eq=False)
class LearnedPlanModel:
    """One fitted generic learner seen as a bounded solver's dynamics.

    Both robustness terms are exactly zero: the learner carries no covariance,
    so the plan is priced by the point objective and the caller must accept
    that explicitly through the solver policy.
    """

    learned: LearnedDynamics
    tolerances: TrackingTolerances
    safety_envelope: SafetyEnvelope
    policy: SolverPolicy
    values: PlanValues
    compile_signature: str
    command_minimum_array: np.ndarray
    command_maximum_array: np.ndarray

    uncertainty_available: bool = False
    uncertainty_complete: bool = False
    exogenous_size: int = 0
    latent_size: int = 0

    @property
    def horizon_steps(self) -> int:
        return self.policy.horizon_steps

    @property
    def block_count(self) -> int:
        return self.policy.block_count

    @property
    def sample_period_s(self) -> float:
        return float(self.learned.contract["dt_s"])

    @property
    def context_steps(self) -> int:
        """Observed transitions the fitted recipe consumes before an origin."""

        return int(self.learned.history_steps)

    @property
    def delay_steps(self) -> int:
        """The explicit-difference window inside the consumed context."""

        return int(self.learned._model.delay_steps)

    @property
    def memory_size(self) -> int:
        return int(self.learned._model.params["memory"].shape[1])

    @property
    def command_size(self) -> int:
        return len(self.command_minimum_array)

    @property
    def command_minimum(self) -> Array:
        return jnp.asarray(self.command_minimum_array)

    @property
    def command_maximum(self) -> Array:
        return jnp.asarray(self.command_maximum_array)

    def with_history(self, history: ObservedHistory) -> LearnedPlanModel:
        """Return the same plan model reading one more observed interval.

        The history is a value, not part of the traced structure, so the
        compile signature is unchanged and the returned model reuses every
        kernel compiled for this one.
        """

        return replace(
            self, values=PlanValues(*self.values[:3], observed_history=history)
        )

    def _sequence_model(self, values: PlanValues) -> SequenceModel:
        parameters = values.parameters
        return SequenceModel(
            kind=KIND,
            dt_s=self.sample_period_s,
            history_steps=self.context_steps,
            params=parameters["params"],
            norms=parameters["norms"],
            delay_steps=self.delay_steps,
        )

    def initial_latent(self, command_history: Array, values: PlanValues) -> Array:
        """Return the empty actuator state this model declares.

        The learner has no actuator state to infer: whatever lag the vehicle
        has is already inside its recursive memory, which travels with the
        observed history rather than being reconstructed from commands.
        """

        del command_history, values
        return jnp.zeros(self.latent_size)

    def _expand_normalized_blocks(self, blocks: Array) -> Array:
        expanded = jnp.repeat(blocks, self.policy.block_steps, axis=0)
        return expanded[: self.horizon_steps]

    def _commands_from_normalized(self, normalized: Array) -> Array:
        """Map feasible solver variables affinely onto the declared bounds.

        The solver owns projection to ``[-1, 1]``; clipping again here would
        attenuate the derivative at an active bound. The bounds themselves are
        the caller's declared telemetry contract, never anything the learner
        inferred.
        """

        minimum = self.command_minimum
        maximum = self.command_maximum
        return 0.5 * (1.0 - normalized) * minimum + 0.5 * (1.0 + normalized) * maximum

    def rollout(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        values: PlanValues,
    ) -> Prediction:
        """Predict the horizon this plan drives, with a zero tangent covariance."""

        commands = self._commands_from_normalized(
            self._expand_normalized_blocks(blocks)
        )
        return self.rollout_commands(
            commands, initial_state, initial_latent, exogenous, values
        )

    def rollout_commands(
        self,
        commands: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        values: PlanValues,
    ) -> Prediction:
        """Predict an exact physical command sequence over the fitted horizon."""

        del initial_latent
        if commands.shape != (self.horizon_steps, self.command_size):
            raise ValueError("commands must have one row per prediction interval")
        history = values.observed_history
        if history is None:
            raise ValueError(
                "a recursive learner cannot start a horizon without observed "
                "history; supply PlanValues.observed_history"
            )
        current = observed_from_state(initial_state)
        past_states = jnp.concatenate(
            (jnp.asarray(history.states), current[None, :]), axis=0
        )
        observed = self._sequence_model(values).rollout(
            past_states,
            jnp.asarray(history.commands),
            commands,
            memory=jnp.asarray(history.memory),
        )
        states = states_from_observed(initial_state, observed, self.sample_period_s)
        return Prediction(
            mean_states=states,
            tangent_covariance=jnp.zeros((self.horizon_steps, 12, 12)),
            commands=commands,
            latent_states=jnp.zeros((self.horizon_steps + 1, self.latent_size)),
            exogenous=exogenous,
        )

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
        """Price one plan by the same objective a belief is priced by, minus spread.

        Tracking, terminal, smoothness and safety terms are written exactly as
        :class:`~glassbox.control.fitted.BeliefPlanModel` writes them. The two
        robustness terms are absent because the evidence they charge for is
        absent: there is no predicted spread to charge and no declared support
        envelope to be near the edge of.
        """

        states = prediction.mean_states
        local_error = jax.vmap(rigid_body_local_error)(reference_states[1:], states[1:])
        error = local_error / self.tolerances.local_state_scale
        delta = jnp.diff(
            jnp.concatenate((previous_command[None, :], prediction.commands), axis=0),
            axis=0,
        ) / (
            policy.command_change_fraction
            * (self.command_maximum - self.command_minimum)
        )
        safety = jax.vmap(self._safety_violation)(states[1:])
        tracking_cost = jnp.mean(jnp.sum(jnp.square(error), axis=1))
        terminal_cost = policy.terminal_weight * jnp.sum(jnp.square(error[-1]))
        smoothness_cost = policy.command_change_weight * jnp.mean(jnp.square(delta))
        safety_cost = policy.safety_weight * jnp.mean(jnp.square(safety))
        return tracking_cost + terminal_cost + smoothness_cost + safety_cost

    def measure(self, prediction: Prediction) -> PlanMeasurements:
        """Measure the margins the result reports for one finished plan.

        Validity utilization is zero because this model declares no support
        envelope, not because a support check passed. Normalized uncertainty is
        zero because no spread is claimed.
        """

        return PlanMeasurements(
            maximum_validity_utilization=jnp.zeros(()),
            maximum_normalized_safety_violation=jnp.max(
                jax.vmap(self._safety_violation)(prediction.mean_states[1:])
            ),
            maximum_normalized_uncertainty=jnp.zeros(()),
        )


def _compile_signature(
    learned: LearnedDynamics,
    tolerances: TrackingTolerances,
    safety_envelope: SafetyEnvelope,
    policy: SolverPolicy,
    command_minimum: np.ndarray,
    command_maximum: np.ndarray,
) -> str:
    """Digest the static structure a kernel compiled for this plan model traces.

    No fitted number reaches this digest. Parameters, normalizations and the
    observed history all travel through every kernel as
    :class:`~glassbox.control.plan.PlanValues`, so a loop that hands the model
    one more observed interval every control interval keeps the code compiled
    for the interval before it.
    """

    digest = hashlib.sha256()

    def add(value: object) -> None:
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\x00")

    model = learned._model
    add(json.dumps(learned.contract, sort_keys=True))
    add(json.dumps(learned.recipe, sort_keys=True))
    add(model.metadata())
    for name, array in sorted(model.arrays().items()):
        add((name, np.asarray(array).shape, str(np.asarray(array).dtype)))
    add(tolerances)
    add(safety_envelope)
    add(policy)
    add((tuple(command_minimum.tolist()), tuple(command_maximum.tolist())))
    add("no_parameter_covariance")
    return digest.hexdigest()


def fitted_solver_policy(learned: LearnedDynamics) -> SolverPolicy:
    """The longest horizon this learner's own evidence supports, and its blocks.

    The recipe forecasts ``horizon_s`` and rejects anything longer rather than
    extrapolating, so that is the horizon, full stop. It is a statement about
    evidence, exactly as a belief's forecast-error envelope is, and it is the
    only thing that sets the horizon here.
    """

    steps = steps_for(learned.contract["dt_s"])["horizon"]
    if steps != learned.horizon_steps:
        raise ValueError("the fitted horizon differs from the recipe's own plan")
    return SolverPolicy(horizon_steps=steps, block_count=maintained_block_count(steps))


def learned_plan_model(
    learned: LearnedDynamics,
    tolerances: TrackingTolerances,
    safety_envelope: SafetyEnvelope,
    *,
    command_minimum,
    command_maximum,
    policy: SolverPolicy | None = None,
) -> LearnedPlanModel:
    """Present one fitted learner to a bounded solver, or refuse to.

    ``command_minimum`` and ``command_maximum`` are the caller's declared
    telemetry contract. They are facts about the vehicle's command space and
    are required here rather than read off the learner, which observed commands
    but was never told what the actuators accept.

    A horizon longer than the fitted one is refused rather than rolled past:
    the recipe's own ``predict`` rejects it, and a plan model that quietly
    extrapolated where ``predict`` refuses would be claiming evidence the fit
    never produced.
    """

    contract = learned.contract
    if tuple(contract["state_channels"]) != OBSERVED_CHANNELS:
        raise ValueError(
            "this plan model requires the fifteen rigid-body observation "
            "channels the platform-tier adapter builds, in that order"
        )
    minimum = _command_bounds("command_minimum", command_minimum)
    maximum = _command_bounds("command_maximum", command_maximum)
    if minimum.shape != maximum.shape or np.any(minimum >= maximum):
        raise ValueError("command bounds must be paired and strictly increasing")
    if len(minimum) != len(contract["input_channels"]):
        raise ValueError(
            "the declared command bounds and the learner's input channels differ"
        )
    resolved = fitted_solver_policy(learned) if policy is None else policy
    if resolved.horizon_steps > learned.horizon_steps:
        raise ValueError(
            f"the requested {resolved.horizon_steps}-step horizon is longer than "
            f"the learner's fitted {learned.horizon_steps} steps; the recipe "
            "rejects horizons beyond its fitted range rather than extrapolating"
        )
    model = learned._model
    values = PlanValues(
        parameters=dict(params=model.params, norms=model.norms),
        covariance_factor=None,
        forecast_error_covariance=jnp.zeros((resolved.horizon_steps, 12, 12)),
        observed_history=None,
    )
    return LearnedPlanModel(
        learned=learned,
        tolerances=tolerances,
        safety_envelope=safety_envelope,
        policy=resolved,
        values=values,
        compile_signature=_compile_signature(
            learned, tolerances, safety_envelope, resolved, minimum, maximum
        ),
        command_minimum_array=minimum,
        command_maximum_array=maximum,
    )


class LearnedPlanController:
    """A bounded NMPC loop over the generic learner, carrying its own history.

    The controller owns the one thing the seam cannot supply: the observed
    transitions before the forecast origin. :meth:`reset` declares a recording
    boundary, :meth:`observe` records one observed state, and
    :meth:`command_applied` records the command the loop actually applied over
    the interval that follows it. Nothing is padded, imputed or assumed: until
    the loop has observed the explicit-difference window the learner reads,
    :attr:`ready` is false and there is no forecast to plan with.
    """

    def __init__(
        self,
        learned: LearnedDynamics,
        tolerances: TrackingTolerances | None = None,
        safety_envelope: SafetyEnvelope | None = None,
        *,
        command_minimum,
        command_maximum,
        policy: SolverPolicy | None = None,
        platform: str = "fixedwing",
    ) -> None:
        self.learned = learned
        self.plan = learned_plan_model(
            learned,
            TrackingTolerances.for_platform(platform)
            if tolerances is None
            else tolerances,
            SafetyEnvelope() if safety_envelope is None else safety_envelope,
            command_minimum=command_minimum,
            command_maximum=command_maximum,
            policy=policy,
        )
        self.policy = self.plan.policy
        # Both are ordinary host-side kernels over one frozen model: the
        # channel map every observation goes through and the memory recurrence
        # over the retained context. Compiled, they cost microseconds an
        # interval; interpreted, their dozens of small array operations cost
        # more than the solve they prepare. JAX caches one per input shape, and
        # the context grows by one observation an interval until it is full, so
        # a timed loop compiles the shapes it will use before it starts.
        self._observe_kernel = jax.jit(observed_from_state)
        self._memory_kernel = jax.jit(learned._model.memory_state)
        self.reset()

    @property
    def prediction_steps(self) -> int:
        return self.plan.horizon_steps

    @property
    def sample_period_s(self) -> float:
        return self.plan.sample_period_s

    @property
    def prediction_horizon_s(self) -> float:
        return self.prediction_steps * self.sample_period_s

    @property
    def context_steps(self) -> int:
        return self.plan.context_steps

    @property
    def required_observations(self) -> int:
        """Observed states the loop needs before any forecast exists at all."""

        return self.plan.delay_steps + 1

    def reset(self) -> None:
        """Declare a recording boundary: forget every observed transition.

        The memory starts again at rest at the next observation, which is what
        the learner's memory contract says a recording boundary means.
        """

        self._states: deque = deque(maxlen=self.context_steps + 1)
        self._commands: deque = deque(maxlen=self.context_steps)
        self._observed = 0
        self._applied = 0

    def observe(self, state) -> None:
        """Record one observed canonical rigid-body state."""

        if self._observed != self._applied:
            raise ValueError("observe and command_applied must alternate")
        observed = np.asarray(self._observe_kernel(np.asarray(state, dtype=float)))
        if not np.isfinite(observed).all():
            raise ValueError("an observed state must be finite")
        self._states.append(observed)
        self._observed += 1

    def command_applied(self, command) -> None:
        """Record the command the loop applied over the interval just observed."""

        if self._applied != self._observed - 1:
            raise ValueError("a command follows the state it was computed from")
        applied = np.asarray(command, dtype=float)
        if applied.shape != (self.plan.command_size,) or not np.isfinite(applied).all():
            raise ValueError("an applied command must be a finite command vector")
        self._commands.append(applied)
        self._applied += 1

    @property
    def observed_states(self) -> int:
        """Observed states recorded since the last recording boundary."""

        return self._observed

    @property
    def ready(self) -> bool:
        """Whether enough observed transitions exist for a forecast to mean anything."""

        return self._observed >= self.required_observations

    def history(self) -> ObservedHistory:
        """The observed history a horizon starting now continues from.

        The origin is the newest observed state, so this is only meaningful
        between observing it and applying the command computed from it. Asked
        for after that command has been recorded, it would shift the commands
        one interval past the observations they belong to, so it refuses.
        """

        if not self.ready:
            raise ValueError(
                f"the learner needs {self.required_observations} observed states "
                f"and {self.plan.delay_steps} applied commands; nothing is padded"
            )
        if self._applied != self._observed - 1:
            raise ValueError(
                "a horizon starts at the newest observed state; ask for the "
                "history between observing it and applying its command"
            )
        delay = self.plan.delay_steps
        states = np.asarray(self._states)
        commands = np.asarray(self._commands)
        memory = np.asarray(self._memory_kernel(states, commands[: len(states) - 1]))
        return ObservedHistory(
            states=states[-delay - 1 : -1],
            commands=commands[-delay:],
            memory=memory,
        )

    def solve(self, state, reference, previous_command, **keywords):
        """Optimize one bounded command over the carried history.

        A fresh solver is built for each interval because the history is a new
        value. Kernels are cached on the model's compile signature and the
        policy, which neither the history nor any fitted number enters, so this
        is a lookup rather than a rebuild.
        """

        plan = self.plan.with_history(self.history())
        solver = BoundedShootingSolver(plan, self.policy)
        return solver.solve(state, reference, previous_command, **keywords)

    def hold_reference(self, state, *, exogenous=None):
        """Build the common regulation reference for this controller."""

        return ReferenceTrajectory.hold(
            state, self.prediction_steps, exogenous=exogenous
        )
