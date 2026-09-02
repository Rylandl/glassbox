"""Dual-control NMPC: one bounded optimization that learns while it recovers.

This is the controller described in ``docs/concepts/dual-control-nmpc.md``.  It
replaces a hand-gained cascade and a scan-based excitation with a single
objective over bounded command blocks.

The first pass paid for information with a separate additive log-determinant
term evaluated on raw commands, and it did not fly: at zero information a
motor-uniform seed is a fixed point of the whole objective, and raw commands
report information a uniform plan cannot actually buy.  Two changes answer
that, both selectable per variant:

``residualize_information``
    The planned information features are residualized exactly the way the
    recursive identifier residualizes its own command Gram, against the
    intercept and the angular nuisance regressors.  A uniform plan then earns
    exactly zero information, and the information gradient points along
    differential inputs rather than along the all-ones direction.

``objective``
    ``"expected_cost"`` deletes the additive information term and charges the
    tracking cost for predicted spread instead, ``E[l(x_k)] = l(x_hat_k) +
    trace(W Sigma_x,k)``, with the parameter posterior the planned inputs would
    produce.  Excitation then pays for itself in tracking units and there is no
    information weight left to choose.

A small multi-start over bounded orthogonal designs breaks the motor symmetry
before the gradient refinement runs.  The amplitudes and sign patterns of those
designs are declared design constants: they are the one action-side prior in
the controller, alongside the command box and the regularizing ``epsilon``.

Nothing in this module knows anything about a particular vehicle.  The
prediction model is rigid-body kinematics, gravity, and the posterior mean of
the recursive bootstrap belief; every other number lives in
:class:`DualControlConfig` and describes either the command box, the recovery
goal, the declared excitation family, or the optimizer.  There is no cascade,
no gain, and no motor geometry.

Research tier: names, signatures, and semantics may change without notice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.control.nmpc.types import SolveStatus
from glassbox.control.online_bootstrap import RecursiveBootstrapBelief
from glassbox.core.dynamics import (
    GRAVITY_M_S2,
    quaternion_multiply,
    quaternion_to_rotation,
)

_STATE_SIZE = 13
_COMMAND_SIZE = 4
#: Nuisance regressors the identifier's angular regression residualizes its
#: command Gram against, minus the intercept, which is handled exactly by
#: centering: three body rates and three rate products.
_ANGULAR_NUISANCE_SIZE = 6
#: Squared quantities below this are held constant before a square root is
#: taken, so a spread of exactly zero, which is what a posterior carrying no
#: information reports, leaves a finite gradient instead of an infinite one.
_SQUARE_FLOOR = 1e-18
#: Magnitude at which the predicted accelerations and states are saturated.
#:
#: A posterior fitted from a handful of rank-deficient samples routinely carries
#: a rate-product coefficient of tens, and ``omega' = a_p omega^2`` blows up in
#: finite time, so the mean model's own thirty-step prediction reaches infinity
#: within the horizon and every plan scores the same infinite cost.  Saturating
#: the prediction keeps the objective finite and, more usefully, makes the
#: tracking term go flat exactly where the model has stopped saying anything, so
#: the spread and information terms decide instead.  The bound is six orders of
#: magnitude above any state a multirotor can be in, so it is a numerical guard
#: in the same sense as the square-root floor above, not a vehicle number: no
#: reachable flight touches it and nothing about its value is tuned.
_PREDICTION_GUARD = 1.0e6
#: Sign patterns the bounded orthogonal designs are built from.  A Hadamard
#: matrix of order four is the smallest bounded design whose rows are mutually
#: orthogonal and span every command direction, and its first row is the
#: collective direction, so a horizon that cycles the rows in both polarities
#: excites the collective and all three differentials at equal amplitude.
_HADAMARD_ROWS = np.asarray(
    (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0, 1.0),
    )
)

#: Objective form.  ``"information_gain"`` is the first pass's additive
#: log-determinant term; ``"expected_cost"`` charges predicted spread inside the
#: tracking cost and carries no information weight.
ObjectiveForm = Literal["information_gain", "expected_cost"]

#: The three studied configurations.  Pass one is kept selectable so its
#: failure stays reproducible; the default is the current design.
DUAL_CONTROL_VARIANTS: dict[str, dict[str, Any]] = {
    "pass1": {
        "multi_start": False,
        "residualize_information": False,
        "objective": "information_gain",
    },
    "pass2a": {
        "multi_start": True,
        "residualize_information": True,
        "objective": "information_gain",
    },
    "pass2b": {
        "multi_start": True,
        "residualize_information": True,
        "objective": "expected_cost",
    },
}


def _safe_sqrt(value: Array) -> Array:
    """Square root with a bounded gradient at zero."""

    return jnp.sqrt(jnp.maximum(value, _SQUARE_FLOOR))


def _guard(value: Array) -> Array:
    """Saturate a predicted quantity at :data:`_PREDICTION_GUARD`."""

    return jnp.clip(value, -_PREDICTION_GUARD, _PREDICTION_GUARD)


def design_sign_pattern(block_count: int) -> np.ndarray:
    """Signs of the declared orthogonal design, one row per command block.

    Blocks cycle the four Hadamard rows and flip polarity every fourth block.
    Over eight or more blocks every row appears in both polarities, so the
    pattern is exactly zero-mean along every command direction and its
    intercept-residualized Gram has full rank four within a single horizon.
    Cycling the rows before flipping the polarity, rather than alternating the
    polarity every block, is the ordering that moves the fewest commands per
    block boundary and so costs the least command rate for the same design.
    """

    if block_count < 1:
        raise ValueError("block_count must be positive")
    index = np.arange(block_count)
    polarity = np.where((index // len(_HADAMARD_ROWS)) % 2 == 0, 1.0, -1.0)
    return polarity[:, None] * _HADAMARD_ROWS[index % len(_HADAMARD_ROWS)]


@dataclass(frozen=True)
class DualControlConfig:
    """Horizon, objective form, task tolerances, and the command box.

    Every field is either a property of the control loop (sample period,
    horizon, block length, optimizer budget), a statement of the recovery goal
    (the four tolerances, the altitude floor, the tilt maximum), a declared
    action-side choice the posterior cannot supply (the command bounds, the
    multi-start amplitudes, and the regularizing ``epsilon``), or an objective
    weight.  None of them is a vehicle number.
    """

    #: Control interval the horizon is expressed in.
    sample_period_s: float = 0.01
    horizon_steps: int = 30
    block_steps: int = 3
    #: Objective form; see :data:`ObjectiveForm`.
    objective: ObjectiveForm = "expected_cost"
    #: Residualize the planned information features the way the identifier
    #: residualizes its own command Gram.
    residualize_information: bool = True
    #: Evaluate the declared orthogonal designs before the gradient refinement.
    multi_start: bool = True
    #: Amplitudes, as fractions of the command range, at which the declared
    #: orthogonal designs are laid on top of the current command.  Both sign
    #: polarities of each design are evaluated.  This tuple and
    #: :func:`design_sign_pattern` are the controller's one action-side prior
    #: beyond the command box.
    #:
    #: A geometric ladder spanning a factor of four.  The top rung is set so
    #: that a single horizon of it dominates the ``epsilon`` prior by three
    #: orders of magnitude, which is what lets one horizon settle the command
    #: map instead of the optimizer having to grow an amplitude over many
    #: intervals; the bottom rung is small enough to leave a settled hover
    #: alone.  Nothing about the ladder refers to a vehicle.
    multi_start_amplitudes: tuple[float, ...] = (0.06, 0.12, 0.25)
    #: Weight on the squared command move between consecutive horizon steps.
    #: A simultaneous ``0.1`` move on all four commands then costs exactly one
    #: task-tolerance unit of tracking error.
    w_rate: float = 25.0
    #: Weight on the expected log-determinant information gain.  Used only by
    #: the ``"information_gain"`` objective; ``"expected_cost"`` has no
    #: information weight at all.
    w_info: float = 1.0
    #: Standard deviations of predicted spread the chance penalties reserve.
    beta: float = 2.0
    #: Regularizing information that makes the log-determinant gain finite
    #: before the first observation, and the parameter posterior proper.  It is
    #: the only prior on the belief side of the controller.
    epsilon: float = 1e-3
    #: Ridge, relative to the mean nuisance energy, used when residualizing the
    #: planned commands against the planned nuisance regressors.  A purely
    #: numerical regularizer: an unexcited nuisance direction explains nothing
    #: rather than everything.
    nuisance_ridge: float = 1e-6
    #: Multiples of the matching task tolerance beyond which a predicted spread
    #: is treated as saturated, and its chance penalty is dropped entirely.
    spread_cap: float = 3.0
    velocity_tolerance_m_s: float = 0.10
    body_rate_tolerance_rad_s: float = 0.10
    tilt_tolerance_rad: float = 0.05
    altitude_tolerance_m: float = 0.10
    altitude_floor_m: float = 1.0
    maximum_tilt_rad: float = 0.50
    #: Outer projected-gradient iterations per solve.
    iteration_count: int = 10
    #: Halvings the backtracking search may take.  The objective's curvature
    #: spans many orders of magnitude between a tumbling release and a settled
    #: hover, so a search that can only reach a step of ``1e-3`` gives up at the
    #: hover end of that range and reports a line-search failure at a point that
    #: is merely stiff.
    line_search_steps: int = 24
    initial_step_size: float = 0.5
    armijo_fraction: float = 1e-4
    #: Absolute infinity norm of the bound-projected gradient below which the
    #: solve is called converged.
    gradient_tolerance: float = 1e-3
    relative_improvement_tolerance: float = 1e-5
    command_minimum: float | tuple[float, float, float, float] = 0.0
    command_maximum: float | tuple[float, float, float, float] = 1.0

    @property
    def block_count(self) -> int:
        """Command blocks the horizon is parameterized by."""

        return math.ceil(self.horizon_steps / self.block_steps)

    @property
    def horizon_s(self) -> float:
        return self.horizon_steps * self.sample_period_s

    @property
    def candidate_count(self) -> int:
        """Plans the multi-start scores before the gradient refinement.

        The shifted warm start, the previous command held, and both polarities
        of the declared design at every declared amplitude.
        """

        if not self.multi_start:
            return 2
        return 2 + 2 * len(self.multi_start_amplitudes)

    @property
    def variant(self) -> str:
        """Name of the studied configuration this config matches, if any."""

        for name, switches in DUAL_CONTROL_VARIANTS.items():
            if all(getattr(self, key) == value for key, value in switches.items()):
                return name
        return "custom"

    def __post_init__(self) -> None:
        minimum = _finite_command("command_minimum", self.command_minimum)
        maximum = _finite_command("command_maximum", self.command_maximum)
        if np.any(minimum >= maximum):
            raise ValueError("command_minimum must be below command_maximum")
        if self.horizon_steps < 1 or self.block_steps < 1:
            raise ValueError("horizon_steps and block_steps must be positive")
        if self.block_steps > self.horizon_steps:
            raise ValueError("block_steps cannot exceed the prediction horizon")
        if self.iteration_count < 1 or self.line_search_steps < 1:
            raise ValueError("solver iteration counts must be positive")
        if self.objective not in ("information_gain", "expected_cost"):
            raise ValueError("objective must be 'information_gain' or 'expected_cost'")
        amplitudes = np.asarray(self.multi_start_amplitudes, dtype=np.float64)
        if amplitudes.ndim != 1 or amplitudes.size == 0:
            raise ValueError("multi_start_amplitudes must be a non-empty sequence")
        if not np.all(np.isfinite(amplitudes)) or np.any(
            (amplitudes <= 0.0) | (amplitudes > 1.0)
        ):
            raise ValueError(
                "multi_start_amplitudes must be fractions of the command range"
            )
        positive = (
            self.sample_period_s,
            self.w_rate,
            self.w_info,
            self.beta,
            self.epsilon,
            self.nuisance_ridge,
            self.spread_cap,
            self.velocity_tolerance_m_s,
            self.body_rate_tolerance_rad_s,
            self.tilt_tolerance_rad,
            self.altitude_tolerance_m,
            self.altitude_floor_m,
            self.maximum_tilt_rad,
            self.initial_step_size,
            self.armijo_fraction,
            self.gradient_tolerance,
            self.relative_improvement_tolerance,
        )
        if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0.0):
            raise ValueError("dual-control weights and tolerances must be positive")
        object.__setattr__(self, "command_minimum", tuple(minimum))
        object.__setattr__(self, "command_maximum", tuple(maximum))
        object.__setattr__(self, "multi_start_amplitudes", tuple(amplitudes.tolist()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "sample_period_s": self.sample_period_s,
            "horizon_steps": self.horizon_steps,
            "block_steps": self.block_steps,
            "block_count": self.block_count,
            "objective": self.objective,
            "residualize_information": self.residualize_information,
            "multi_start": self.multi_start,
            "multi_start_amplitudes": list(self.multi_start_amplitudes),
            "candidate_count": self.candidate_count,
            "w_rate": self.w_rate,
            "w_info": self.w_info,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "nuisance_ridge": self.nuisance_ridge,
            "spread_cap": self.spread_cap,
            "velocity_tolerance_m_s": self.velocity_tolerance_m_s,
            "body_rate_tolerance_rad_s": self.body_rate_tolerance_rad_s,
            "tilt_tolerance_rad": self.tilt_tolerance_rad,
            "altitude_tolerance_m": self.altitude_tolerance_m,
            "altitude_floor_m": self.altitude_floor_m,
            "maximum_tilt_rad": self.maximum_tilt_rad,
            "iteration_count": self.iteration_count,
            "line_search_steps": self.line_search_steps,
            "command_minimum": list(self.command_minimum),  # type: ignore[arg-type]
            "command_maximum": list(self.command_maximum),  # type: ignore[arg-type]
        }


def dual_control_config(variant: str = "pass2b", **overrides: Any) -> DualControlConfig:
    """Config for one of the studied variants, with optional overrides."""

    if variant not in DUAL_CONTROL_VARIANTS:
        known = ", ".join(sorted(DUAL_CONTROL_VARIANTS))
        raise ValueError(f"unknown dual-control variant {variant!r}; known: {known}")
    return DualControlConfig(**DUAL_CONTROL_VARIANTS[variant], **overrides)


def _finite_command(name: str, value: Any) -> np.ndarray:
    array = np.broadcast_to(
        np.asarray(value, dtype=np.float64),
        (_COMMAND_SIZE,),
    ).astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array)


def command_information_log_determinant(
    belief: RecursiveBootstrapBelief,
    config: DualControlConfig | None = None,
) -> float:
    """Log-determinant of the command information the objective differences.

    This is exactly the ``log det(I_u)`` that the information gain subtracts, so
    a study can record its trajectory and read the gain against it without
    restating the definition.
    """

    settings = DualControlConfig() if config is None else config
    variance = max(float(np.mean(np.square(belief.angular_residual_std_rad_s2))), 1e-12)
    information = np.asarray(
        belief.normalized_command_information, dtype=np.float64
    ) / variance + settings.epsilon * np.eye(_COMMAND_SIZE)
    return float(np.linalg.slogdet(information)[1])


class _Posterior(NamedTuple):
    """Every posterior quantity the plan consumes, as one dynamic pytree.

    These are traced inputs to the single compiled program, so a fresh belief
    every interval never triggers a recompilation.
    """

    collective_per_command: Array
    collective_velocity_coefficient: Array
    collective_intercept: Array
    angular_per_command: Array
    angular_rate_coefficient: Array
    angular_rate_product_coefficient: Array
    angular_intercept: Array
    command_information: Array
    collective_covariance: Array
    angular_covariance: Array
    residual_variance: Array
    collective_residual_variance: Array


class _Terms(NamedTuple):
    """The objective decomposition and the horizon diagnostics it implies."""

    tracking: Array
    spread_charge: Array
    command_rate: Array
    information_gain: Array
    altitude_penalty: Array
    tilt_penalty: Array
    maximum_rate_spread: Array
    maximum_tilt_spread: Array
    maximum_altitude_spread: Array
    altitude_active_steps: Array
    tilt_active_steps: Array
    altitude_saturated_steps: Array
    tilt_saturated_steps: Array


@dataclass(frozen=True)
class DualControlResult:
    """One bounded command, why it was chosen, and what it is predicted to risk."""

    command: np.ndarray
    command_usable: bool
    status: SolveStatus
    iterations: int
    objective_value: float
    seed_objective_value: float
    tracking_cost: float
    spread_charge: float
    command_rate_cost: float
    information_gain: float
    altitude_penalty: float
    tilt_penalty: float
    maximum_rate_spread_rad_s: float
    maximum_tilt_spread_rad: float
    maximum_altitude_spread_m: float
    altitude_constraint_active_steps: int
    tilt_constraint_active_steps: int
    altitude_constraint_saturated_steps: int
    tilt_constraint_saturated_steps: int
    used_warm_start: bool
    #: Which multi-start candidate the solve refined.
    selected_candidate: str
    selected_candidate_index: int
    #: Declared amplitude of the winning candidate, zero for the warm start and
    #: the held command.
    selected_amplitude: float
    #: Amplitude the refined plan actually carries: the largest deviation of any
    #: block from the plan's own mean, as a fraction of the command range.
    plan_amplitude: float
    #: Numerical rank of the planned command information the refined plan buys.
    planned_information_rank: int
    plan: np.ndarray
    reason: str = "dual_control_nmpc"

    def __post_init__(self) -> None:
        command = np.asarray(self.command, dtype=np.float64)
        if command.shape != (_COMMAND_SIZE,):
            raise ValueError("dual-control command must have four entries")
        command.setflags(write=False)
        object.__setattr__(self, "command", command)
        plan = np.asarray(self.plan, dtype=np.float64)
        if plan.ndim != 2 or plan.shape[1] != _COMMAND_SIZE:
            raise ValueError("dual-control plan must be blocks of four commands")
        plan.setflags(write=False)
        object.__setattr__(self, "plan", plan)
        scalars = (
            self.objective_value,
            self.seed_objective_value,
            self.tracking_cost,
            self.spread_charge,
            self.command_rate_cost,
            self.information_gain,
            self.altitude_penalty,
            self.tilt_penalty,
            self.maximum_rate_spread_rad_s,
            self.maximum_tilt_spread_rad,
            self.maximum_altitude_spread_m,
            self.selected_amplitude,
            self.plan_amplitude,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("dual-control diagnostics must be finite")
        if not np.all(np.isfinite(command)):
            raise ValueError("dual-control command must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.tolist(),
            "command_usable": self.command_usable,
            "status": str(self.status),
            "reason": self.reason,
            "iterations": self.iterations,
            "objective_value": self.objective_value,
            "seed_objective_value": self.seed_objective_value,
            "tracking_cost": self.tracking_cost,
            "spread_charge": self.spread_charge,
            "command_rate_cost": self.command_rate_cost,
            "information_gain": self.information_gain,
            "altitude_penalty": self.altitude_penalty,
            "tilt_penalty": self.tilt_penalty,
            "maximum_rate_spread_rad_s": self.maximum_rate_spread_rad_s,
            "maximum_tilt_spread_rad": self.maximum_tilt_spread_rad,
            "maximum_altitude_spread_m": self.maximum_altitude_spread_m,
            "altitude_constraint_active_steps": (self.altitude_constraint_active_steps),
            "tilt_constraint_active_steps": self.tilt_constraint_active_steps,
            "altitude_constraint_saturated_steps": (
                self.altitude_constraint_saturated_steps
            ),
            "tilt_constraint_saturated_steps": (self.tilt_constraint_saturated_steps),
            "used_warm_start": self.used_warm_start,
            "selected_candidate": self.selected_candidate,
            "selected_candidate_index": self.selected_candidate_index,
            "selected_amplitude": self.selected_amplitude,
            "plan_amplitude": self.plan_amplitude,
            "planned_information_rank": self.planned_information_rank,
        }


# iteration, blocks, value, gradient, step size, converged, stalled,
# line-search failure
_OuterCarry = tuple[Array, Array, Array, Array, Array, Array, Array, Array]
_LineCarry = tuple[Array, Array, Array, Array, Array, Array]


def _projected_gradient_norm(blocks: Array, gradient: Array) -> Array:
    """Infinity norm of the bound-projected gradient on the normalized box.

    A raw gradient component pointing outward at an active bound never shrinks,
    so the projected step is the honest first-order residual for this bounded
    problem: it vanishes exactly when no feasible descent direction remains.
    """

    return jnp.max(jnp.abs(blocks - jnp.clip(blocks - gradient, -1.0, 1.0)))


class DualControlNMPC:
    """Plan bounded commands that recover the vehicle and identify it at once.

    The whole solve, multi-start included, is one jitted program.  The posterior
    enters as traced arrays, so a belief that changes every interval never
    recompiles anything.
    """

    def __init__(self, config: DualControlConfig | None = None) -> None:
        self.config = DualControlConfig() if config is None else config
        self._minimum = np.asarray(self.config.command_minimum, dtype=np.float64)
        self._maximum = np.asarray(self.config.command_maximum, dtype=np.float64)
        self._span = self._maximum - self._minimum
        self._midpoint = 0.5 * (self._minimum + self._maximum)
        self._jit_minimum = jnp.asarray(self._minimum)
        self._jit_span = jnp.asarray(self._span)
        self._jit_midpoint = jnp.asarray(self._midpoint)
        self._jit_maximum = jnp.asarray(self._maximum)
        self._signs = jnp.asarray(design_sign_pattern(self.config.block_count))
        # Amplitude labels line up with the candidate order the program builds:
        # warm start, held command, then both polarities of every amplitude.
        polarities = (1.0, -1.0)
        self._candidate_names: tuple[str, ...] = (
            "warm_start",
            "hold",
            *(
                f"design_{amplitude:g}_{'plus' if polarity > 0 else 'minus'}"
                for amplitude in self.config.multi_start_amplitudes
                for polarity in polarities
            ),
        )
        self._candidate_amplitudes = np.asarray(
            (
                0.0,
                0.0,
                *(
                    amplitude
                    for amplitude in self.config.multi_start_amplitudes
                    for _ in polarities
                ),
            )
        )
        if not self.config.multi_start:
            self._candidate_names = self._candidate_names[:2]
            self._candidate_amplitudes = self._candidate_amplitudes[:2]
        self._value_and_gradient = jax.value_and_grad(self._objective)
        self._program = jax.jit(self._solve_program)

    # ------------------------------------------------------------------
    # command-block parameterization
    # ------------------------------------------------------------------

    @property
    def block_count(self) -> int:
        return self.config.block_count

    @property
    def candidate_names(self) -> tuple[str, ...]:
        """Multi-start candidates in the order the program scores them."""

        return self._candidate_names

    @property
    def jit_cache_size(self) -> int:
        """Compiled variants of the one solve program, for the no-recompile test."""

        return int(self._program._cache_size())

    def _expand(self, blocks: Array) -> Array:
        """Hold each block over its steps and truncate to the horizon."""

        expanded = jnp.repeat(blocks, self.config.block_steps, axis=0)
        return expanded[: self.config.horizon_steps]

    def _commands_from_normalized(self, normalized: Array) -> Array:
        return (
            self._jit_minimum
            + 0.5 * (jnp.clip(normalized, -1.0, 1.0) + 1.0) * self._jit_span
        )

    def _normalized_from_commands(self, commands: Array) -> Array:
        return 2.0 * (commands - self._jit_minimum) / self._jit_span - 1.0

    def _cold_blocks(self, previous_command: Array) -> Array:
        normalized = jnp.clip(
            self._normalized_from_commands(previous_command), -1.0, 1.0
        )
        return jnp.repeat(normalized[None, :], self.block_count, axis=0)

    def _design_blocks(self, previous_command: Array, offset: Array) -> Array:
        """One declared orthogonal design laid on top of the current command.

        The offset is applied in raw command units and clipped to the box.  At a
        bound the clip turns the two-sided design into a one-sided one, which
        halves its amplitude but leaves its sign structure, and therefore its
        rank, untouched.
        """

        raw = previous_command[None, :] + offset * self._signs * self._jit_span
        clipped = jnp.clip(raw, self._jit_minimum, self._jit_maximum)
        return jnp.clip(self._normalized_from_commands(clipped), -1.0, 1.0)

    def _candidate_blocks(
        self,
        warm_blocks: Array,
        previous_command: Array,
    ) -> Array:
        """Every multi-start candidate, stacked for one vmapped evaluation."""

        cold = self._cold_blocks(previous_command)
        candidates = [warm_blocks, cold]
        if self.config.multi_start:
            for amplitude in self.config.multi_start_amplitudes:
                for polarity in (1.0, -1.0):
                    candidates.append(
                        self._design_blocks(
                            previous_command,
                            jnp.asarray(polarity * amplitude),
                        )
                    )
        return jnp.stack(candidates)

    # ------------------------------------------------------------------
    # prediction, information features, spread, and objective
    # ------------------------------------------------------------------

    def _rollout(
        self,
        blocks: Array,
        state: Array,
        posterior: _Posterior,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Roll the known kinematics forward on the posterior-mean command maps.

        Returns the commands, the per-step tracking cost, the chord tilt, the
        altitude, and the two angular nuisance regressor blocks the identifier
        residualizes its command Gram against.
        """

        commands = self._commands_from_normalized(self._expand(blocks))
        period = self.config.sample_period_s
        gravity = jnp.asarray((0.0, 0.0, -GRAVITY_M_S2))

        def step(
            carry: tuple[Array, Array, Array, Array],
            command: Array,
        ) -> tuple[
            tuple[Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array],
        ]:
            altitude, velocity, quaternion, angular_velocity = carry
            rotation = quaternion_to_rotation(quaternion)
            body_velocity = rotation.T @ velocity
            specific_force = (
                posterior.collective_per_command @ command
                + posterior.collective_velocity_coefficient @ body_velocity
                + posterior.collective_intercept
            )
            specific_force = _guard(specific_force)
            world_acceleration = gravity + rotation[:, 2] * specific_force
            rate_products = jnp.stack(
                (
                    angular_velocity[0] * angular_velocity[1],
                    angular_velocity[0] * angular_velocity[2],
                    angular_velocity[1] * angular_velocity[2],
                )
            )
            angular_acceleration = (
                posterior.angular_per_command @ command
                + posterior.angular_rate_coefficient @ angular_velocity
                + posterior.angular_rate_product_coefficient @ rate_products
                + posterior.angular_intercept
            )

            angular_acceleration = _guard(angular_acceleration)

            next_altitude = _guard(altitude + period * velocity[2])
            next_velocity = _guard(velocity + period * world_acceleration)
            quaternion_rate = 0.5 * quaternion_multiply(
                quaternion,
                jnp.concatenate((jnp.zeros(1), angular_velocity)),
            )
            unnormalized = quaternion + period * quaternion_rate
            next_quaternion = unnormalized / jnp.maximum(
                jnp.linalg.norm(unnormalized), 1e-9
            )
            next_angular_velocity = _guard(
                angular_velocity + period * angular_acceleration
            )

            # Chord-squared tilt: ``2 (1 - cos tilt)`` equals ``tilt**2`` to
            # second order and, unlike ``arccos``, has a bounded gradient at
            # exactly level flight.
            _, x, y, _ = next_quaternion
            chord_squared = 4.0 * (x * x + y * y)
            chord_tilt = _safe_sqrt(chord_squared)
            floor_error = jnp.maximum(self.config.altitude_floor_m - next_altitude, 0.0)
            tracking = (
                jnp.sum(jnp.square(next_velocity))
                / self.config.velocity_tolerance_m_s**2
                + jnp.sum(jnp.square(next_angular_velocity))
                / self.config.body_rate_tolerance_rad_s**2
                + chord_squared / self.config.tilt_tolerance_rad**2
                + jnp.square(floor_error) / self.config.altitude_tolerance_m**2
            )
            return (
                next_altitude,
                next_velocity,
                next_quaternion,
                next_angular_velocity,
            ), (
                tracking,
                chord_tilt,
                next_altitude,
                # The identifier regresses the interval's angular acceleration
                # on the rates measured at the *start* of the interval, so the
                # nuisance regressors the plan implies are the pre-step ones.
                angular_velocity,
                rate_products,
            )

        quaternion = state[6:10] / jnp.maximum(jnp.linalg.norm(state[6:10]), 1e-9)
        initial = (state[2], state[3:6], quaternion, state[10:13])
        _, outputs = jax.lax.scan(step, initial, commands)
        tracking, tilt, altitude, angular_velocity, rate_products = outputs
        return commands, tracking, tilt, altitude, angular_velocity, rate_products

    def _information_features(
        self,
        commands: Array,
        angular_velocity: Array,
        rate_products: Array,
    ) -> Array:
        """Planned command features on the identifier's own residual basis.

        The identifier accumulates ``outer(f, f)`` on ``f = [(u - mid) / span,
        omega, omega products, 1]`` and reports as command information the Schur
        complement of the nuisance block, that is the command features
        residualized against the intercept and the nuisance regressors.  A
        planned Gram is only comparable with that incumbent if it is built the
        same way, so the planned commands are centered over the horizon (which
        is exact residualization against the intercept) and then regressed out
        of the planned nuisance regressors.

        Centering is what makes a uniform plan earn exactly zero: its centered
        features are identically zero, so its Gram is zero and its
        log-determinant gain is zero rather than the rank-one credit raw
        commands report for doing nothing differential.
        """

        features = (commands - self._jit_midpoint) / self._jit_span
        if not self.config.residualize_information:
            return features
        centered = features - jnp.mean(features, axis=0)
        nuisance = jnp.concatenate((angular_velocity, rate_products), axis=1)
        nuisance = nuisance - jnp.mean(nuisance, axis=0)
        gram = nuisance.T @ nuisance
        ridge = jnp.maximum(
            self.config.nuisance_ridge * jnp.trace(gram) / _ANGULAR_NUISANCE_SIZE,
            1e-12,
        )
        coefficient = jnp.linalg.solve(
            gram + ridge * jnp.eye(_ANGULAR_NUISANCE_SIZE),
            nuisance.T @ centered,
        )
        return centered - nuisance @ coefficient

    def _current_information(self, posterior: _Posterior) -> Array:
        """``I_u``: the incumbent command information as a precision."""

        variance = jnp.maximum(posterior.residual_variance, 1e-12)
        return posterior.command_information / variance + self.config.epsilon * jnp.eye(
            _COMMAND_SIZE
        )

    def _information_gain(self, features: Array, posterior: _Posterior) -> Array:
        """Expected log-determinant gain about the command maps."""

        variance = jnp.maximum(posterior.residual_variance, 1e-12)
        current = self._current_information(posterior)
        planned = features.T @ features / variance
        return jnp.linalg.slogdet(current + planned)[1] - jnp.linalg.slogdet(current)[1]

    def _planned_parameter_covariance(
        self,
        features: Array,
        posterior: _Posterior,
    ) -> Array:
        """``Sigma_theta' = (I_u + Phi^T Phi / s^2)^-1`` on residualized features.

        This is the posterior the planned inputs would leave behind, used from
        the start of the horizon.  It is the standard value-of-information
        approximation of the dual effect: the plan is charged for the spread it
        would still be carrying after its own excitation had been absorbed.
        """

        variance = jnp.maximum(posterior.residual_variance, 1e-12)
        planned = self._current_information(posterior) + features.T @ features / (
            variance
        )
        return jnp.linalg.inv(planned)

    def expected_information_gain(
        self,
        commands: Any,
        belief: RecursiveBootstrapBelief,
        angular_velocity: Any = None,
        rate_products: Any = None,
    ) -> float:
        """Information the given command sequence is expected to buy.

        This is the objective's own information term evaluated on an arbitrary
        command sequence rather than on a plan, which is what lets a caller ask
        what a candidate excitation would be worth without solving anything.
        The nuisance regressors default to zero, which reduces the
        residualization to the exact intercept term.
        """

        sequence = np.asarray(commands, dtype=np.float64)
        if sequence.ndim != 2 or sequence.shape[1] != _COMMAND_SIZE:
            raise ValueError("commands must be a sequence of four-entry commands")
        rates = (
            np.zeros((len(sequence), 3))
            if angular_velocity is None
            else np.asarray(angular_velocity, dtype=np.float64)
        )
        products = (
            np.zeros((len(sequence), 3))
            if rate_products is None
            else np.asarray(rate_products, dtype=np.float64)
        )
        if rates.shape != (len(sequence), 3) or products.shape != (len(sequence), 3):
            raise ValueError("nuisance regressors must be three per command")
        posterior = self._posterior(belief)
        features = self._information_features(
            jnp.asarray(sequence),
            jnp.asarray(rates),
            jnp.asarray(products),
        )
        return float(self._information_gain(features, posterior))

    def _spreads(
        self,
        commands: Array,
        features: Array,
        posterior: _Posterior,
    ) -> tuple[Array, Array, Array, Array]:
        """Per-step predicted spread in rate, tilt, vertical speed, and altitude.

        The command-map covariance is the covariance of a fixed unknown
        coefficient, so its effect is perfectly correlated across the horizon
        and the standard deviations, not the variances, are what integrate.

        Under ``"information_gain"`` the per-step acceleration spread is the
        incumbent posterior evaluated at the planned command, which is what the
        first pass propagated.  Under ``"expected_cost"`` it is the
        command-averaged spread of the *planned* posterior: the variance a
        command drawn uniformly from the box would still have after the plan's
        own excitation.  Averaging over the box rather than evaluating at the
        planned command is what stops the charge from being zeroed by parking on
        whatever command the quadratic form happens to vanish at; it asks how
        well the vehicle's response is known over the whole box it will have to
        act in, which is the quantity that decides whether the goal is
        reachable.
        """

        period = self.config.sample_period_s
        if self.config.objective == "expected_cost":
            covariance = self._planned_parameter_covariance(features, posterior)
            # Uniform commands on the box give the normalized features
            # covariance ``I / 12`` and zero mean, so the box-averaged
            # predictive variance is ``trace(Sigma) / 12`` per output axis.
            box_variance = jnp.trace(covariance) / 12.0
            angular_variance = 3.0 * box_variance
            collective_variance = (
                box_variance
                * posterior.collective_residual_variance
                / jnp.maximum(posterior.residual_variance, 1e-12)
            )
            steps = commands.shape[0]
            angular_spread = jnp.full((steps,), _safe_sqrt(angular_variance))
            collective_spread = jnp.full((steps,), _safe_sqrt(collective_variance))
        else:
            collective_spread = _safe_sqrt(
                jnp.einsum(
                    "ti,ij,tj->t",
                    commands,
                    posterior.collective_covariance,
                    commands,
                )
            )
            angular_spread = _safe_sqrt(
                jnp.einsum(
                    "ti,aij,tj->t",
                    commands,
                    posterior.angular_covariance,
                    commands,
                )
            )
        rate_spread = jnp.cumsum(period * angular_spread)
        tilt_spread = jnp.cumsum(period * rate_spread)
        vertical_speed_spread = jnp.cumsum(period * collective_spread)
        altitude_spread = jnp.cumsum(period * vertical_speed_spread)
        return rate_spread, tilt_spread, vertical_speed_spread, altitude_spread

    def _terms(
        self,
        blocks: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
    ) -> _Terms:
        """Every objective term, plus the chance-constraint activity it implies."""

        (
            commands,
            tracking,
            tilt,
            altitude,
            angular_velocity,
            rate_products,
        ) = self._rollout(blocks, state, posterior)
        features = self._information_features(
            commands,
            angular_velocity,
            rate_products,
        )
        (
            rate_spread,
            tilt_spread,
            vertical_speed_spread,
            altitude_spread,
        ) = self._spreads(commands, features, posterior)
        moves = jnp.diff(
            jnp.concatenate((previous_command[None, :], commands), axis=0),
            axis=0,
        )
        command_rate = self.config.w_rate * jnp.sum(jnp.square(moves))
        gain = self._information_gain(features, posterior)
        if self.config.objective == "expected_cost":
            # ``trace(W Sigma_x,k)`` on the same tolerances the tracking cost
            # normalizes by, so a metre of predicted spread and a metre of
            # predicted error cost the same.
            spread_charge = jnp.sum(
                jnp.square(rate_spread) / self.config.body_rate_tolerance_rad_s**2
                + jnp.square(tilt_spread) / self.config.tilt_tolerance_rad**2
                + jnp.square(vertical_speed_spread)
                / self.config.velocity_tolerance_m_s**2
                + jnp.square(altitude_spread) / self.config.altitude_tolerance_m**2
            )
        else:
            spread_charge = jnp.asarray(0.0)

        altitude_cap = self.config.spread_cap * self.config.altitude_tolerance_m
        tilt_cap = self.config.spread_cap * self.config.tilt_tolerance_rad
        altitude_supported = altitude_spread <= altitude_cap
        tilt_supported = tilt_spread <= tilt_cap
        altitude_breach = jnp.maximum(
            self.config.altitude_floor_m
            + self.config.beta * altitude_spread
            - altitude,
            0.0,
        )
        tilt_breach = jnp.maximum(
            tilt + self.config.beta * tilt_spread - self.config.maximum_tilt_rad,
            0.0,
        )
        # A spread wider than the cap makes the chance constraint say nothing,
        # so the penalty is dropped from the objective and therefore from the
        # gradient.  At zero information this leaves the spread charge, the
        # information term, and the command bounds in charge, which is the
        # intent.
        altitude_penalty = jnp.sum(
            jnp.where(altitude_supported, jnp.square(altitude_breach), 0.0)
        )
        tilt_penalty = jnp.sum(jnp.where(tilt_supported, jnp.square(tilt_breach), 0.0))
        return _Terms(
            tracking=jnp.sum(tracking),
            spread_charge=spread_charge,
            command_rate=command_rate,
            information_gain=gain,
            altitude_penalty=altitude_penalty,
            tilt_penalty=tilt_penalty,
            maximum_rate_spread=jnp.max(rate_spread),
            maximum_tilt_spread=jnp.max(tilt_spread),
            maximum_altitude_spread=jnp.max(altitude_spread),
            altitude_active_steps=jnp.sum((altitude_breach > 0.0) & altitude_supported),
            tilt_active_steps=jnp.sum((tilt_breach > 0.0) & tilt_supported),
            altitude_saturated_steps=jnp.sum(~altitude_supported),
            tilt_saturated_steps=jnp.sum(~tilt_supported),
        )

    def _objective(
        self,
        blocks: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
    ) -> Array:
        terms = self._terms(blocks, state, posterior, previous_command)
        information = (
            jnp.asarray(0.0)
            if self.config.objective == "expected_cost"
            else self.config.w_info * terms.information_gain
        )
        return (
            terms.tracking
            + terms.spread_charge
            + terms.command_rate
            - information
            + terms.altitude_penalty
            + terms.tilt_penalty
        )

    def _plan_diagnostics(
        self,
        blocks: Array,
        state: Array,
        posterior: _Posterior,
    ) -> tuple[Array, Array]:
        """Realized amplitude and planned information rank of a finished plan.

        Kept out of the objective because the eigenvalue decomposition the rank
        needs has no usable gradient at the repeated eigenvalues an unexcited
        design produces.
        """

        (
            commands,
            _tracking,
            _tilt,
            _altitude,
            angular_velocity,
            rate_products,
        ) = self._rollout(blocks, state, posterior)
        features = self._information_features(
            commands,
            angular_velocity,
            rate_products,
        )
        amplitude = jnp.max(jnp.abs(features - jnp.mean(features, axis=0)))
        planned = features.T @ features
        eigenvalues = jnp.linalg.eigvalsh(planned)
        threshold = jnp.maximum(jnp.max(eigenvalues), 0.0) * 1e-6
        rank = jnp.sum(eigenvalues > jnp.maximum(threshold, 1e-12))
        return amplitude, rank

    # ------------------------------------------------------------------
    # bounded projected-gradient solve
    # ------------------------------------------------------------------

    def _optimize(
        self,
        blocks: Array,
        value: Array,
        gradient: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Projected gradient with a bounded Armijo backtracking line search.

        The search direction is the gradient scaled to unit infinity norm.  The
        objective spans several orders of magnitude between a tumbling release
        and a settled hover, so a raw-gradient step of a fixed size would be
        either a bang-bang jump or a no-op depending only on where in the
        flight the solve happens; a unit direction makes the step sizes mean the
        same thing throughout, and any positive multiple of the gradient is
        still a descent direction, so the Armijo test is unchanged.

        Every accepted step strictly decreases the objective and a rejected one
        leaves the iterate alone, so the refinement can never return a plan
        worse than the multi-start candidate it was handed.
        """

        def finite(candidate_value: Array, candidate_gradient: Array) -> Array:
            return jnp.isfinite(candidate_value) & jnp.all(
                jnp.isfinite(candidate_gradient)
            )

        def continue_outer(carry: _OuterCarry) -> Array:
            iteration, _, current, current_gradient, _, converged, stalled, failed = (
                carry
            )
            return (
                (iteration < self.config.iteration_count)
                & ~converged
                & ~stalled
                & ~failed
                & finite(current, current_gradient)
            )

        def outer_step(carry: _OuterCarry) -> _OuterCarry:
            iteration, current_blocks, current, current_gradient, step, _, _, _ = carry
            scale = jnp.maximum(jnp.max(jnp.abs(current_gradient)), 1e-12)
            direction = current_gradient / scale
            # Absolute, not relative to the objective value: at zero
            # information the tracking term is a large constant with no
            # gradient at all, so a value-scaled tolerance would call the very
            # first solve converged before it had moved a single command.
            converged = (
                _projected_gradient_norm(current_blocks, current_gradient)
                <= self.config.gradient_tolerance
            )

            def continue_line_search(line_carry: _LineCarry) -> Array:
                line_iteration, accepted, _, _, _, _ = line_carry
                return (line_iteration < self.config.line_search_steps) & ~accepted

            def line_search_step(line_carry: _LineCarry) -> _LineCarry:
                (
                    line_iteration,
                    accepted,
                    best_blocks,
                    best_value,
                    best_gradient,
                    accepted_step,
                ) = line_carry
                candidate_step = step * jnp.power(0.5, line_iteration)
                candidate = jnp.clip(
                    current_blocks - candidate_step * direction, -1.0, 1.0
                )
                candidate_value, candidate_gradient = self._value_and_gradient(
                    candidate,
                    state,
                    posterior,
                    previous_command,
                )
                projected_decrease = jnp.sum(
                    current_gradient * (current_blocks - candidate)
                )
                candidate_accepted = finite(candidate_value, candidate_gradient) & (
                    candidate_value
                    <= current
                    - self.config.armijo_fraction * jnp.maximum(projected_decrease, 0.0)
                )
                return (
                    line_iteration + 1,
                    accepted | candidate_accepted,
                    jnp.where(candidate_accepted, candidate, best_blocks),
                    jnp.where(candidate_accepted, candidate_value, best_value),
                    jnp.where(candidate_accepted, candidate_gradient, best_gradient),
                    jnp.where(candidate_accepted, candidate_step, accepted_step),
                )

            (
                _,
                accepted,
                next_blocks,
                next_value,
                next_gradient,
                accepted_step,
            ) = jax.lax.while_loop(
                continue_line_search,
                line_search_step,
                (
                    jnp.asarray(0),
                    converged,
                    current_blocks,
                    current,
                    current_gradient,
                    step,
                ),
            )
            relative_improvement = (current - next_value) / jnp.maximum(
                jnp.abs(current), 1.0
            )
            stalled = (
                accepted
                & ~converged
                & (relative_improvement <= self.config.relative_improvement_tolerance)
            )
            return (
                iteration + 1,
                next_blocks,
                next_value,
                next_gradient,
                jnp.minimum(self.config.initial_step_size, 2.0 * accepted_step),
                converged,
                stalled,
                ~accepted,
            )

        (
            iteration,
            final_blocks,
            final_value,
            _final_gradient,
            _step,
            converged,
            stalled,
            failed,
        ) = jax.lax.while_loop(
            continue_outer,
            outer_step,
            (
                jnp.asarray(0),
                blocks,
                value,
                gradient,
                jnp.asarray(self.config.initial_step_size),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
            ),
        )
        return (
            final_blocks,
            final_value,
            iteration,
            converged,
            stalled,
            failed,
        )

    def _solve_program(
        self,
        warm_blocks: Array,
        warm_valid: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
    ) -> tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        _Terms,
    ]:
        """The whole solve as one compiled operation on traced posteriors.

        The multi-start scores every declared candidate in one vmap and refines
        the best.  Because the warm start and the held command are both in that
        set, the seed is structurally no worse than either, and because the
        refinement is monotone the returned plan is no worse than the seed.
        """

        candidates = self._candidate_blocks(warm_blocks, previous_command)
        values = jax.vmap(
            lambda candidate: self._objective(
                candidate,
                state,
                posterior,
                previous_command,
            )
        )(candidates)
        # A warm start that could not be shifted is not a candidate at all.
        usable = jnp.isfinite(values).at[0].set(jnp.isfinite(values[0]) & warm_valid)
        values = jnp.where(usable, values, jnp.inf)
        selected = jnp.argmin(values)
        blocks = candidates[selected]
        seed_value = values[selected]
        value, gradient = self._value_and_gradient(
            blocks,
            state,
            posterior,
            previous_command,
        )
        final_blocks, final_value, iteration, converged, stalled, failed = (
            self._optimize(
                blocks,
                value,
                gradient,
                state,
                posterior,
                previous_command,
            )
        )
        terms = self._terms(final_blocks, state, posterior, previous_command)
        amplitude, rank = self._plan_diagnostics(final_blocks, state, posterior)
        commands = self._commands_from_normalized(self._expand(final_blocks))
        plan = self._commands_from_normalized(final_blocks)
        return (
            commands[0],
            plan,
            final_value,
            seed_value,
            iteration,
            converged,
            stalled,
            failed,
            selected,
            amplitude,
            rank,
            terms,
        )

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def _held_command(self, previous_command: Any) -> np.ndarray:
        try:
            previous = np.asarray(previous_command, dtype=np.float64)
        except (TypeError, ValueError):
            return self._midpoint.copy()
        if previous.shape != (_COMMAND_SIZE,) or not np.all(np.isfinite(previous)):
            return self._midpoint.copy()
        return np.clip(previous, self._minimum, self._maximum)

    def _unusable(
        self,
        command: np.ndarray,
        status: SolveStatus,
        reason: str,
    ) -> DualControlResult:
        return DualControlResult(
            command=command,
            command_usable=False,
            status=status,
            iterations=0,
            objective_value=0.0,
            seed_objective_value=0.0,
            tracking_cost=0.0,
            spread_charge=0.0,
            command_rate_cost=0.0,
            information_gain=0.0,
            altitude_penalty=0.0,
            tilt_penalty=0.0,
            maximum_rate_spread_rad_s=0.0,
            maximum_tilt_spread_rad=0.0,
            maximum_altitude_spread_m=0.0,
            altitude_constraint_active_steps=0,
            tilt_constraint_active_steps=0,
            altitude_constraint_saturated_steps=0,
            tilt_constraint_saturated_steps=0,
            used_warm_start=False,
            selected_candidate="none",
            selected_candidate_index=-1,
            selected_amplitude=0.0,
            plan_amplitude=0.0,
            planned_information_rank=0,
            plan=np.repeat(command[None, :], self.block_count, axis=0),
            reason=reason,
        )

    def _posterior(self, belief: RecursiveBootstrapBelief) -> _Posterior:
        variance = float(np.mean(np.square(belief.angular_residual_std_rad_s2)))
        return _Posterior(
            collective_per_command=jnp.asarray(
                belief.collective_acceleration_per_command
            ),
            collective_velocity_coefficient=jnp.asarray(
                belief.collective_velocity_coefficient
            ),
            collective_intercept=jnp.asarray(belief.collective_intercept_m_s2),
            angular_per_command=jnp.asarray(belief.angular_acceleration_per_command),
            angular_rate_coefficient=jnp.asarray(belief.angular_rate_coefficient),
            angular_rate_product_coefficient=jnp.asarray(
                belief.angular_rate_product_coefficient
            ),
            angular_intercept=jnp.asarray(belief.angular_intercept_rad_s2),
            command_information=jnp.asarray(belief.normalized_command_information),
            collective_covariance=jnp.asarray(
                belief.supported_collective_effect_covariance
            ),
            angular_covariance=jnp.asarray(belief.supported_angular_effect_covariance),
            residual_variance=jnp.asarray(variance),
            collective_residual_variance=jnp.asarray(
                float(belief.collective_residual_std_m_s2) ** 2
            ),
        )

    @staticmethod
    def _belief_is_finite(belief: RecursiveBootstrapBelief) -> bool:
        return bool(
            np.all(np.isfinite(belief.collective_acceleration_per_command))
            and np.all(np.isfinite(belief.collective_velocity_coefficient))
            and math.isfinite(belief.collective_intercept_m_s2)
            and np.all(np.isfinite(belief.angular_acceleration_per_command))
            and np.all(np.isfinite(belief.angular_rate_coefficient))
            and np.all(np.isfinite(belief.angular_rate_product_coefficient))
            and np.all(np.isfinite(belief.angular_intercept_rad_s2))
            and np.all(np.isfinite(belief.normalized_command_information))
            and np.all(np.isfinite(belief.supported_collective_effect_covariance))
            and np.all(np.isfinite(belief.supported_angular_effect_covariance))
            and math.isfinite(belief.collective_residual_std_m_s2)
        )

    def _warm_blocks(
        self,
        warm_start: Any,
        held: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Shift the previous plan by one block, or say it cannot be used.

        The plan is parameterized at block granularity, so the seed is shifted
        at that granularity too: its first block is the previous plan's second,
        and its final block repeats the previous plan's last.
        """

        cold = np.repeat(
            (2.0 * (held - self._minimum) / self._span - 1.0)[None, :],
            self.block_count,
            axis=0,
        )
        if warm_start is None:
            return cold, False
        plan = getattr(warm_start, "plan", warm_start)
        try:
            commands = np.asarray(plan, dtype=np.float64)
        except (TypeError, ValueError):
            return cold, False
        if commands.shape != (self.block_count, _COMMAND_SIZE) or not np.all(
            np.isfinite(commands)
        ):
            return cold, False
        bounded = np.clip(commands, self._minimum, self._maximum)
        shifted = np.concatenate((bounded[1:], bounded[-1:]), axis=0)
        normalized = 2.0 * (shifted - self._minimum) / self._span - 1.0
        return np.clip(normalized, -1.0, 1.0), True

    def solve(
        self,
        state: Sequence[float],
        belief: RecursiveBootstrapBelief,
        previous_command: Sequence[float],
        warm_start: Any = None,
    ) -> DualControlResult:
        """Return one bounded command, or the previous one when it cannot.

        A state or posterior this controller cannot act on never raises: the
        result comes back with ``command_usable`` false, a status, and the
        previous command clipped into bounds.  There is no fallback controller.
        """

        held = self._held_command(previous_command)
        state_array = np.asarray(state, dtype=np.float64)
        if (
            state_array.shape != (_STATE_SIZE,)
            or not np.all(np.isfinite(state_array))
            or float(np.linalg.norm(state_array[6:10])) < 1e-9
            or not self._belief_is_finite(belief)
        ):
            return self._unusable(held, SolveStatus.INVALID_INPUT, "unusable_input")

        warm, warm_valid = self._warm_blocks(warm_start, held)
        (
            command,
            plan,
            value,
            seed_value,
            iteration,
            converged,
            stalled,
            failed,
            selected,
            plan_amplitude,
            planned_rank,
            terms,
        ) = self._program(
            jnp.asarray(warm),
            jnp.asarray(warm_valid),
            jnp.asarray(state_array),
            self._posterior(belief),
            jnp.asarray(held),
        )
        command_array = np.asarray(command, dtype=np.float64)
        value_float = float(value)
        finite = bool(np.all(np.isfinite(command_array))) and math.isfinite(value_float)
        if not finite:
            return self._unusable(
                held,
                SolveStatus.NONFINITE_OBJECTIVE,
                "nonfinite_solve",
            )
        bounded = np.clip(command_array, self._minimum, self._maximum)
        if bool(failed):
            status = SolveStatus.LINE_SEARCH_FAILED
        elif bool(converged):
            status = SolveStatus.CONVERGED
        elif bool(stalled):
            status = SolveStatus.STALLED
        else:
            status = SolveStatus.ITERATION_LIMIT
        index = int(selected)
        return DualControlResult(
            command=bounded,
            command_usable=True,
            status=status,
            iterations=int(iteration),
            objective_value=value_float,
            seed_objective_value=float(seed_value),
            tracking_cost=float(terms.tracking),
            spread_charge=float(terms.spread_charge),
            command_rate_cost=float(terms.command_rate),
            information_gain=float(terms.information_gain),
            altitude_penalty=float(terms.altitude_penalty),
            tilt_penalty=float(terms.tilt_penalty),
            maximum_rate_spread_rad_s=float(terms.maximum_rate_spread),
            maximum_tilt_spread_rad=float(terms.maximum_tilt_spread),
            maximum_altitude_spread_m=float(terms.maximum_altitude_spread),
            altitude_constraint_active_steps=int(terms.altitude_active_steps),
            tilt_constraint_active_steps=int(terms.tilt_active_steps),
            altitude_constraint_saturated_steps=int(terms.altitude_saturated_steps),
            tilt_constraint_saturated_steps=int(terms.tilt_saturated_steps),
            used_warm_start=index == 0,
            selected_candidate=self._candidate_names[index],
            selected_candidate_index=index,
            selected_amplitude=float(self._candidate_amplitudes[index]),
            plan_amplitude=float(plan_amplitude),
            planned_information_rank=int(planned_rank),
            plan=np.clip(
                np.asarray(plan, dtype=np.float64),
                self._minimum,
                self._maximum,
            ),
        )
