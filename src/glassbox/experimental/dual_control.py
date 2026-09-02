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

The fourth pass changes where those designs sit and what the rate cost charges
for, and nothing else:

``charge_unowned_transition``
    False makes the command-rate cost a slew cost on the controller's *own*
    consecutive actions rather than a prior about the state the vehicle was
    handed over in.  The move out of a command this controller never issued —
    the first plan after enable, or any interval whose predecessor was somebody
    else's command — is not charged; every later move is charged exactly as
    before.

``center_designs_on_base_action``
    True centers the declared designs on the base action rather than on the
    previous command.  See :meth:`DualControlNMPC.base_action`: it is the box
    midpoint while the posterior has no supported hover estimate, and the
    posterior's own hover estimate once it has one.

The fifth pass removes the midpoint, the amplitude ladder, and the declared
designs entirely.  There is one goal — stabilize — and learning is valued only
for what it buys towards that goal.  The only declared quantities left on the
action side are the command box, a per-interval slew limit, and the outcome
limits; everything else is derived from the posterior and the state.  Three
switches carry it, and every earlier variant leaves them where they were:

``plan_parameterization``
    ``"slew_moves"`` makes each block a bounded *move* from the previous one.
    The declared slew becomes a box on the decision variables rather than a
    cost, so the executed command can never jump more than the declared
    fraction of the range, and the projection the bounded solver already
    performs is exactly the slew projection.

``spread_model``
    ``"planned_trajectory"`` inverts the *full* regressor information — the
    accumulated Gram plus what the plan itself would add — along the planned
    mean trajectory, and couples the attitude spread into the thrust: the
    velocity spread carries ``|f| sigma_tilt`` alongside the collective's own
    spread, so thrusting while the attitude is uncertain is charged for the
    velocity it might produce in the wrong direction.
    ``"command_marginal_coupled"`` keeps that coupling and that propagation
    but charges the box-averaged per-step spread the second pass charged, so
    the charge cannot be lowered by a plan that visits fewer points.

``seed_family``
    ``"posterior_moves"`` replaces the declared designs with seeds the
    posterior and the state supply: an excitation cycle along the eigenvectors
    of the current command covariance, weakest-known direction first; a
    righting move allocated through the pseudo-inverse of the posterior's own
    angular map; and a collective move along the posterior's own collective
    map.  At zero information the excitation cycle is the only seed that moves,
    which is the symmetry break the ladder used to provide.

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
#: Columns of the identifier's collective regression: the normalized command,
#: the body velocity, and an intercept.
_COLLECTIVE_FEATURE_SIZE = 8
#: Columns of the identifier's angular regression: the normalized command, the
#: body rates, the three rate products, and an intercept.
_ANGULAR_FEATURE_SIZE = 11
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

#: How the horizon's decision variables become commands.  ``"command_blocks"``
#: makes every block a bounded command in its own right.  ``"slew_moves"``
#: makes every block a bounded *move* from the one before it, so the declared
#: per-interval slew is a box on the decision variables and the plan is the
#: cumulative sum from the command the vehicle is already holding.
PlanParameterization = Literal["command_blocks", "slew_moves"]

#: Where the predicted spread comes from.  ``"command_marginal"`` reads it off
#: the command-map covariance alone — evaluated at the planned command under
#: ``"information_gain"``, averaged over the command box under
#: ``"expected_cost"`` — and treats the four channels as independent chains of
#: integrators.  ``"planned_trajectory"`` inverts the full regressor
#: information along the planned mean trajectory and couples the attitude
#: spread into the thrust.
SpreadModel = Literal["command_marginal", "planned_trajectory"]

#: What the multi-start scores.  ``"declared_designs"`` is the Hadamard
#: amplitude ladder laid on a command center; ``"posterior_moves"`` derives
#: every seed from the current posterior and the current state.
SeedFamily = Literal["declared_designs", "posterior_moves"]
GoalHorizon = Literal["declared", "posterior"]
InformationNeighbourhood = Literal["box", "slew", "visited"]
ExcitationBasis = Literal["motor", "hadamard"]

#: The posterior-derived seeds, in the order the program stacks them after the
#: warm start and the held command.  Four excitation cycles along the command
#: covariance's own eigenvectors — both polarities of the plain cycle and both
#: of the cycle whose polarity flips every four blocks, which is zero-mean over
#: two cycles — then the righting move and the collective move.
_POSTERIOR_SEED_NAMES: tuple[str, ...] = (
    "excite+",
    "excite-",
    "excite_alt+",
    "excite_alt-",
    "righting",
    "collective",
)
#: The sixth pass's additional seeds: one descent plan per goal term.
_GOAL_SEED_NAMES: tuple[str, ...] = (
    "goal_velocity",
    "goal_rate",
    "goal_tilt",
    "goal_floor",
)

#: The studied configurations.  Pass one is kept selectable so its failure stays
#: reproducible; the default is the current design.  Every variant states every
#: switch, so a config matches at most one of them.
DUAL_CONTROL_VARIANTS: dict[str, dict[str, Any]] = {
    "pass1": {
        "multi_start": False,
        "residualize_information": False,
        "objective": "information_gain",
        "charge_unowned_transition": True,
        "center_designs_on_base_action": False,
        "plan_parameterization": "command_blocks",
        "spread_model": "command_marginal",
        "seed_family": "declared_designs",
        "charge_body_rate_limit": False,
    },
    "pass2a": {
        "multi_start": True,
        "residualize_information": True,
        "objective": "information_gain",
        "charge_unowned_transition": True,
        "center_designs_on_base_action": False,
        "plan_parameterization": "command_blocks",
        "spread_model": "command_marginal",
        "seed_family": "declared_designs",
        "charge_body_rate_limit": False,
    },
    "pass2b": {
        "multi_start": True,
        "residualize_information": True,
        "objective": "expected_cost",
        "charge_unowned_transition": True,
        "center_designs_on_base_action": False,
        "plan_parameterization": "command_blocks",
        "spread_model": "command_marginal",
        "seed_family": "declared_designs",
        "charge_body_rate_limit": False,
    },
    "pass4": {
        "multi_start": True,
        "residualize_information": True,
        "objective": "expected_cost",
        "charge_unowned_transition": False,
        "center_designs_on_base_action": True,
        "plan_parameterization": "command_blocks",
        "spread_model": "command_marginal",
        "seed_family": "declared_designs",
        "charge_body_rate_limit": False,
    },
    # One goal, a one-second horizon, and no declared action prior beyond the
    # command box, the slew limit, and the two outcome limits.
    "pass5": {
        "multi_start": True,
        "residualize_information": True,
        "objective": "expected_cost",
        "charge_unowned_transition": False,
        "center_designs_on_base_action": False,
        "plan_parameterization": "slew_moves",
        "spread_model": "planned_trajectory",
        "seed_family": "posterior_moves",
        "charge_body_rate_limit": True,
        "horizon_steps": 100,
        "block_steps": 5,
    },
    # Pass five with the spread charged on the box average again: the
    # one-second horizon, the slew-bounded moves, the posterior seeds, and the
    # tilt-to-thrust coupling are kept, and the charge no longer depends on
    # which commands the plan happens to visit.
    "pass6": {
        "multi_start": True,
        "residualize_information": True,
        "objective": "expected_cost",
        "charge_unowned_transition": False,
        "center_designs_on_base_action": False,
        "plan_parameterization": "slew_moves",
        "spread_model": "command_marginal_full",
        "seed_family": "posterior_moves",
        "charge_body_rate_limit": True,
        "goal_horizon": "posterior",
        "clip_action_spread": True,
        "information_neighbourhood": "box",
        "empirical_prior_scale": False,
        "regressor_trust": False,
        "excitation_basis": "hadamard",
        "goal_seeds": True,
        "horizon_steps": 100,
        "block_steps": 5,
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
    #: Charge the command-rate cost for the move out of a command this
    #: controller did not issue.
    #:
    #: The rate cost exists to keep consecutive *planned* actions close to each
    #: other, which is a statement about how fast this controller is willing to
    #: move.  Applied to the handover it becomes something else: it anchors the
    #: whole first plan to whatever command the vehicle happened to be carrying,
    #: and the throw diagnostic releases with the motors off, so that anchor is
    #: a prior toward zero thrust that nothing in the objective ever declared.
    #: With this false the rate cost is a slew cost on the controller's own
    #: actions and nothing else: the first transition after enable, and any
    #: transition out of an interval the controller did not command, is free,
    #: while every transition between two of its own consecutive commands is
    #: charged exactly as before.
    charge_unowned_transition: bool = True
    #: Center the declared orthogonal designs on
    #: :meth:`DualControlNMPC.base_action` rather than on the previous command.
    center_designs_on_base_action: bool = False
    #: How the decision variables become commands; see
    #: :data:`PlanParameterization`.
    plan_parameterization: PlanParameterization = "command_blocks"
    #: Where the predicted spread comes from; see :data:`SpreadModel`.
    spread_model: SpreadModel = "command_marginal"
    #: What the multi-start scores; see :data:`SeedFamily`.
    seed_family: SeedFamily = "declared_designs"
    #: Charge a chance penalty for exceeding :attr:`maximum_body_rate_rad_s`.
    #: The rate limit is only an outcome limit where it is charged; the earlier
    #: passes declare no such limit and leave this off.
    charge_body_rate_limit: bool = False
    #: How far along the horizon the goal is charged.  ``"declared"`` charges
    #: the mean prediction on every step.  ``"posterior"`` charges each goal
    #: term, and its chance penalty, only on the steps where that channel's
    #: predicted spread is still under its cap: past that point the posterior
    #: says the mean prediction is unknown, so it carries no cost, and the
    #: spread charge alone speaks for those steps.  The horizon the goal sees
    #: is then a consequence of what has been learned rather than a number.
    goal_horizon: GoalHorizon = "declared"
    #: Under the ``"sequential"`` spread model, whether the plan's own outcome
    #: spread is clipped at the cap before it is charged.  Unclipped, acting
    #: along a direction the posterior has not seen is charged at its full
    #: predicted spread, which is the honest expected cost of doing so.
    clip_action_spread: bool = True
    #: Which commands the knowledge term averages over: the whole command box,
    #: or the commands within one slew of the held command.
    information_neighbourhood: InformationNeighbourhood = "box"
    #: Take the prior spread of an unlearned command direction from the
    #: effects already learned: once any direction of a map is supported, an
    #: unlearned direction is presumed as strong as the learned ones, which is
    #: a statement that the motors are alike and not a number for any vehicle.
    #: Before anything is learned the regularizing ``epsilon`` stands.
    empirical_prior_scale: bool = False
    #: Trust each fitted nuisance regressor only within the range of the data:
    #: in the mean rollout the body velocity, the body rates, and their
    #: products entering the fitted nuisance terms are clipped to the half-width
    #: of a uniform distribution with the second moment the identifier has
    #: accumulated for that regressor.  A local linear fit says nothing about
    #: rates it has never seen, and extrapolating its damping and coupling
    #: coefficients over a one-second horizon is what blows the mean prediction
    #: up.  The command maps and the known kinematics are never clipped.
    regressor_trust: bool = False
    #: Basis the excitation seeds decompose the command precision in.  It only
    #: decides the degenerate case: at zero information ``"motor"`` probes the
    #: four motors one at a time and ``"hadamard"`` probes the collective first
    #: and then the three zero-sum patterns, which is the order that treats the
    #: motors as exchangeable and nothing more.  Once the posterior has learned
    #: anything the eigenvectors are its own and the basis is immaterial.
    excitation_basis: ExcitationBasis = "motor"
    #: Add one descent seed per goal term to the posterior multi-start: the
    #: steepest-descent plan of the velocity, rate, tilt, and floor costs with
    #: respect to the moves, at the held command, scaled to the slew.
    goal_seeds: bool = False
    #: The declared per-interval slew limit, as a fraction of the command
    #: range.  Under ``"slew_moves"`` this is a hard box on every block's move,
    #: so the executed command never leaves the previous one by more than this
    #: fraction of the range, whatever the objective would prefer.  It is a
    #: statement about how fast this controller is willing to move an actuator
    #: it has not identified, in the same class as the command bounds.
    slew_per_interval: float = 0.10
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
    #: The declared maximum body rate, charged only when
    #: :attr:`charge_body_rate_limit` is set.  It is an outcome limit of the
    #: same kind as :attr:`maximum_tilt_rad`: a statement about the states this
    #: controller is willing to plan through, not a vehicle capability.
    maximum_body_rate_rad_s: float = 5.0
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

        The shifted warm start, the previous command held, and either both
        polarities of the declared design at every declared amplitude or, under
        ``"posterior_moves"``, the six seeds the posterior and the state
        supply.
        """

        if not self.multi_start:
            return 2
        if self.seed_family == "posterior_moves":
            return 2 + len(self.posterior_seed_names)
        return 2 + 2 * len(self.multi_start_amplitudes)

    @property
    def posterior_seed_names(self) -> tuple[str, ...]:
        """Names of the posterior-derived seeds this config runs."""

        if self.goal_seeds:
            return (*_POSTERIOR_SEED_NAMES, *_GOAL_SEED_NAMES)
        return _POSTERIOR_SEED_NAMES

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
        if self.plan_parameterization not in ("command_blocks", "slew_moves"):
            raise ValueError(
                "plan_parameterization must be 'command_blocks' or 'slew_moves'"
            )
        if self.spread_model not in (
            "command_marginal",
            "planned_trajectory",
            "command_marginal_coupled",
            "command_marginal_full",
            "sequential",
            "act_know",
        ):
            raise ValueError(
                "spread_model must be 'command_marginal', 'planned_trajectory', "
                "'command_marginal_coupled', 'command_marginal_full', "
                "'sequential', or 'act_know'"
            )
        if self.excitation_basis not in ("motor", "hadamard"):
            raise ValueError("excitation_basis must be 'motor' or 'hadamard'")
        if self.information_neighbourhood not in ("box", "slew", "visited"):
            raise ValueError(
                "information_neighbourhood must be 'box', 'slew', or 'visited'"
            )
        if self.goal_horizon not in ("declared", "posterior"):
            raise ValueError("goal_horizon must be 'declared' or 'posterior'")
        if self.seed_family not in ("declared_designs", "posterior_moves"):
            raise ValueError(
                "seed_family must be 'declared_designs' or 'posterior_moves'"
            )
        if (
            not math.isfinite(self.slew_per_interval)
            or not 0.0 < self.slew_per_interval <= 1.0
        ):
            raise ValueError(
                "slew_per_interval must be a fraction of the command range"
            )
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
            self.maximum_body_rate_rad_s,
            self.slew_per_interval,
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
            "charge_unowned_transition": self.charge_unowned_transition,
            "center_designs_on_base_action": self.center_designs_on_base_action,
            "plan_parameterization": self.plan_parameterization,
            "spread_model": self.spread_model,
            "seed_family": self.seed_family,
            "charge_body_rate_limit": self.charge_body_rate_limit,
            "goal_horizon": self.goal_horizon,
            "clip_action_spread": self.clip_action_spread,
            "information_neighbourhood": self.information_neighbourhood,
            "empirical_prior_scale": self.empirical_prior_scale,
            "regressor_trust": self.regressor_trust,
            "excitation_basis": self.excitation_basis,
            "goal_seeds": self.goal_seeds,
            "slew_per_interval": self.slew_per_interval,
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
            "maximum_body_rate_rad_s": self.maximum_body_rate_rad_s,
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
    #: The identifier's own accumulated Grams, over its own feature order.
    #: Only ``"planned_trajectory"`` reads them.
    collective_information: Array
    angular_information: Array
    #: Per-axis angular residual variance.  ``residual_variance`` above is the
    #: mean of these three, which is what the command-marginal spread and the
    #: information term have always used.
    angular_residual_variance: Array
    #: Prior precision of an unlearned coefficient in each regression.
    collective_prior_precision: Array = jnp.asarray(1e-3)
    angular_prior_precision: Array = jnp.asarray(1e-3)


class _Rollout(NamedTuple):
    """Everything one mean rollout leaves behind, over the horizon's steps.

    The nuisance regressors are the ones measured at the *start* of each step,
    because that is what the identifier regresses the interval's acceleration
    on; the tracking cost, the tilt, the altitude, and the rate norm are the
    predicted post-step quantities.
    """

    commands: Array
    tracking: Array
    tilt: Array
    altitude: Array
    angular_velocity: Array
    rate_products: Array
    body_velocity: Array
    specific_force: Array
    rate_norm: Array
    #: Per-step goal terms in the order velocity, body rate, tilt, floor, each
    #: already normalized by its tolerance; ``tracking`` is their row sum.
    tracking_terms: Array


class _Terms(NamedTuple):
    """The objective decomposition and the horizon diagnostics it implies."""

    tracking: Array
    spread_charge: Array
    command_rate: Array
    information_gain: Array
    altitude_penalty: Array
    tilt_penalty: Array
    body_rate_penalty: Array
    maximum_rate_spread: Array
    maximum_tilt_spread: Array
    maximum_velocity_spread: Array
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
    #: Chance penalty on the declared maximum body rate, zero for every
    #: configuration that does not declare one.
    body_rate_penalty: float
    maximum_rate_spread_rad_s: float
    maximum_tilt_spread_rad: float
    maximum_velocity_spread_m_s: float
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
    #: The command the declared designs were centered on this interval, and
    #: which of the two rules supplied it: ``"previous_command"`` when the
    #: designs sit on the incumbent command, ``"box_midpoint"`` when the base
    #: action is the declared midpoint because no supported hover estimate
    #: exists yet, and ``"hover_estimate"`` once the posterior supplies one.
    design_center: np.ndarray
    design_center_source: str
    #: Whether the rate cost charged the move out of the previous command.
    charged_initial_transition: bool
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
        center = np.asarray(self.design_center, dtype=np.float64)
        if center.shape != (_COMMAND_SIZE,) or not np.all(np.isfinite(center)):
            raise ValueError("dual-control design center must have four finite entries")
        center.setflags(write=False)
        object.__setattr__(self, "design_center", center)
        scalars = (
            self.objective_value,
            self.seed_objective_value,
            self.tracking_cost,
            self.spread_charge,
            self.command_rate_cost,
            self.information_gain,
            self.altitude_penalty,
            self.tilt_penalty,
            self.body_rate_penalty,
            self.maximum_rate_spread_rad_s,
            self.maximum_tilt_spread_rad,
            self.maximum_velocity_spread_m_s,
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
            "body_rate_penalty": self.body_rate_penalty,
            "maximum_rate_spread_rad_s": self.maximum_rate_spread_rad_s,
            "maximum_tilt_spread_rad": self.maximum_tilt_spread_rad,
            "maximum_velocity_spread_m_s": self.maximum_velocity_spread_m_s,
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
            "design_center": self.design_center.tolist(),
            "design_center_source": self.design_center_source,
            "charged_initial_transition": self.charged_initial_transition,
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
        # One decision-variable unit is a full declared slew on that command,
        # so the projected gradient's clip to ``[-1, 1]`` is exactly the slew
        # projection and no other bound has to be threaded through the solver.
        self._slew_step = self.config.slew_per_interval * self._span
        self._jit_slew_step = jnp.asarray(self._slew_step)
        self._jit_hadamard_basis = jnp.asarray(_HADAMARD_ROWS / 2.0)
        # Amplitude labels line up with the candidate order the program builds:
        # warm start, held command, then both polarities of every amplitude.
        polarities = (1.0, -1.0)
        if self.config.seed_family == "posterior_moves":
            self._candidate_names: tuple[str, ...] = (
                "warm",
                "hold",
                *self.config.posterior_seed_names,
            )
            # Every posterior seed moves by exactly the declared slew on at
            # least one command, so the declared amplitude of all six is the
            # slew itself; the warm start and the held command declare none.
            self._candidate_amplitudes = np.asarray(
                (
                    0.0,
                    0.0,
                    *(
                        self.config.slew_per_interval
                        for _ in self.config.posterior_seed_names
                    ),
                )
            )
        else:
            self._candidate_names = (
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

    def _plan_command_blocks(self, blocks: Array, previous_command: Array) -> Array:
        """Raw command per block under this config's plan parameterization.

        Under ``"slew_moves"`` each decision variable is a move of at most one
        declared slew, the plan is their cumulative sum from the command the
        vehicle is holding, and the box is applied to the running total.
        Clipping the total rather than the moves keeps the slew bound exact: a
        clip can only pull a command back towards the one before it.
        """

        if self.config.plan_parameterization == "slew_moves":
            moves = jnp.clip(blocks, -1.0, 1.0) * self._jit_slew_step
            return jnp.clip(
                previous_command[None, :] + jnp.cumsum(moves, axis=0),
                self._jit_minimum,
                self._jit_maximum,
            )
        return self._commands_from_normalized(blocks)

    def _cold_blocks(self, previous_command: Array) -> Array:
        if self.config.plan_parameterization == "slew_moves":
            # Holding the current command is the zero move, whatever it is.
            return jnp.zeros((self.block_count, _COMMAND_SIZE))
        normalized = jnp.clip(
            self._normalized_from_commands(previous_command), -1.0, 1.0
        )
        return jnp.repeat(normalized[None, :], self.block_count, axis=0)

    def _design_blocks(self, center: Array, offset: Array) -> Array:
        """One declared orthogonal design laid on top of a command center.

        The offset is applied in raw command units and clipped to the box.  At a
        bound the clip turns the two-sided design into a one-sided one, which
        halves its amplitude but leaves its sign structure, and therefore its
        rank, untouched.
        """

        raw = center[None, :] + offset * self._signs * self._jit_span
        clipped = jnp.clip(raw, self._jit_minimum, self._jit_maximum)
        return jnp.clip(self._normalized_from_commands(clipped), -1.0, 1.0)

    def _excitation_directions(self, posterior: _Posterior) -> Array:
        """Unit-slew moves along the command covariance's own eigenvectors.

        The command block of the accumulated Gram, scaled by the residual
        variance and regularized by ``epsilon``, is the incumbent command
        precision.  Its eigenvectors are the covariance's eigenvectors, and
        ``eigh`` returns them in ascending eigenvalue order, which is exactly
        descending posterior variance: row zero is the direction the posterior
        knows least about.

        The decomposition is taken in the configured basis of the command
        block.  Eigenvectors do not depend on the basis, so once the posterior
        has learned anything the rows are its own weakest directions either
        way; what the basis decides is the degenerate case.  At zero
        information the precision is a multiple of the identity and every
        direction is equally unknown, and the rows are then the basis vectors
        in index order.  In the motor basis (the fifth pass) they are the four
        motors one at a time, which is the probe order that buys the least
        thrust per unit of torque on any multirotor.  In the orthonormal
        Hadamard basis (the sixth pass) they are the collective first and then
        the three zero-sum patterns: the order that treats the motors as
        exchangeable and nothing more, naming no vehicle and no direction a
        vehicle prefers, only the symmetry the command contract already
        states.

        Each row is scaled so its largest entry is one, so every excitation
        block moves at least one command by the full declared slew.
        """

        variance = jnp.maximum(posterior.collective_residual_variance, 1e-12)
        precision = posterior.collective_information[
            :_COMMAND_SIZE, :_COMMAND_SIZE
        ] / variance + self.config.epsilon * jnp.eye(_COMMAND_SIZE)
        if self.config.excitation_basis == "hadamard":
            basis = self._jit_hadamard_basis
        else:
            basis = jnp.eye(_COMMAND_SIZE)
        _eigenvalues, eigenvectors = jnp.linalg.eigh(basis @ precision @ basis.T)
        rows = eigenvectors.T @ basis
        scale = jnp.maximum(jnp.max(jnp.abs(rows), axis=1, keepdims=True), 1e-12)
        return rows / scale

    def _righting_move(self, state: Array, posterior: _Posterior) -> Array:
        """One slew-sized move towards levelling the vehicle and stopping it.

        The desired body-frame angular acceleration is the axis-angle that
        takes the body ``z`` axis onto world up, less the current body rate, so
        the seed both rights the vehicle and damps what it is already doing.
        It is allocated through the pseudo-inverse of the posterior's own
        angular map, with a relative cutoff that zeroes the directions the
        posterior cannot yet actuate; at zero information the map is zero, the
        pseudo-inverse is zero, and this seed is the held command.
        """

        quaternion = state[6:10] / jnp.maximum(jnp.linalg.norm(state[6:10]), 1e-9)
        rotation = quaternion_to_rotation(quaternion)
        # World up expressed in the body frame, which is the third row of the
        # body-to-world rotation.
        up_body = rotation[2, :]
        levelling = jnp.stack((-up_body[1], up_body[0], jnp.zeros(())))
        desired = levelling - state[10:13]
        norm = jnp.linalg.norm(desired)
        direction = jnp.where(
            norm > 1e-9,
            desired / jnp.maximum(norm, 1e-12),
            jnp.zeros(3),
        )
        # Angular acceleration per unit decision variable, before the slew
        # scale: the posterior's map composed with the command range.
        effect = posterior.angular_per_command * self._jit_span[None, :]
        move = jnp.linalg.pinv(effect, rtol=1e-3) @ direction
        scale = jnp.max(jnp.abs(move))
        return jnp.where(
            scale > 1e-12,
            move / jnp.maximum(scale, 1e-12),
            jnp.zeros(_COMMAND_SIZE),
        )

    def _collective_move(self, posterior: _Posterior) -> Array:
        """One slew-sized move along the posterior's own collective map.

        The move is the map itself, rescaled so its largest entry is one, which
        raises the predicted specific force by construction and does so through
        the motors the posterior believes actually produce it.  At zero
        information the map is zero and this seed is the held command.
        """

        effect = posterior.collective_per_command * self._jit_span
        scale = jnp.max(jnp.abs(effect))
        return jnp.where(
            scale > 1e-12,
            effect / jnp.maximum(scale, 1e-12),
            jnp.zeros(_COMMAND_SIZE),
        )

    def _goal_moves(
        self,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
    ) -> Array:
        """One slew-scaled descent plan per goal term, from the hold plan.

        Each seed is the steepest-descent direction of one goal term — the
        velocity, body-rate, tilt, or floor cost summed over the steps the
        current posterior can still see — with respect to the whole plan of
        moves, taken at the held command and rescaled so its largest move is
        exactly the slew.  Nothing is allocated by hand: the direction comes
        from differentiating the objective's own mean rollout, so it already
        composes the posterior's maps with the kinematics, and at zero
        information every map is zero and every seed is the held command.
        """

        hold = jnp.zeros((self.block_count, _COMMAND_SIZE))
        current = self._spreads_marginal_full(
            self._rollout(hold, state, posterior, previous_command),
            posterior,
            state,
            False,
            previous_command,
        )
        known = jnp.stack(
            (
                current[2]
                <= self.config.spread_cap * self.config.velocity_tolerance_m_s,
                current[0]
                <= self.config.spread_cap * self.config.body_rate_tolerance_rad_s,
                current[1] <= self.config.spread_cap * self.config.tilt_tolerance_rad,
                current[3] <= self.config.spread_cap * self.config.altitude_tolerance_m,
            ),
            axis=1,
        )

        def term(blocks: Array, index: int) -> Array:
            rollout = self._rollout(blocks, state, posterior, previous_command)
            return jnp.sum(
                jnp.where(known[:, index], rollout.tracking_terms[:, index], 0.0)
            )

        def seed(index: int) -> Array:
            gradient = jax.grad(term)(hold, index)
            scale = jnp.max(jnp.abs(gradient))
            finite = jnp.all(jnp.isfinite(gradient)) & (scale > 1e-12)
            return jnp.where(finite, -gradient / jnp.maximum(scale, 1e-12), hold)

        return jnp.stack([seed(index) for index in range(4)])

    def _posterior_seed_blocks(
        self,
        warm_blocks: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
    ) -> Array:
        """The twelve seeds of the posterior-derived multi-start."""

        hold = jnp.zeros((self.block_count, _COMMAND_SIZE))
        directions = self._excitation_directions(posterior)
        index = jnp.arange(self.block_count)
        cycle = directions[index % _COMMAND_SIZE]
        # Flip the polarity every completed cycle, so the alternating pattern
        # is exactly zero-mean over any two cycles.
        polarity = jnp.where((index // _COMMAND_SIZE) % 2 == 0, 1.0, -1.0)
        alternating = polarity[:, None] * cycle
        righting = hold.at[0].set(self._righting_move(state, posterior))
        collective = hold.at[0].set(self._collective_move(posterior))
        seeds = jnp.stack(
            (
                warm_blocks,
                hold,
                cycle,
                -cycle,
                alternating,
                -alternating,
                righting,
                collective,
            )
        )
        if not self.config.goal_seeds:
            return seeds
        goals = self._goal_moves(state, posterior, previous_command)
        return jnp.concatenate((seeds, goals))

    def _candidate_blocks(
        self,
        warm_blocks: Array,
        previous_command: Array,
        design_center: Array,
        state: Array,
        posterior: _Posterior,
    ) -> Array:
        """Every multi-start candidate, stacked for one vmapped evaluation."""

        if self.config.multi_start and self.config.seed_family == "posterior_moves":
            return self._posterior_seed_blocks(
                warm_blocks, state, posterior, previous_command
            )
        cold = self._cold_blocks(previous_command)
        candidates = [warm_blocks, cold]
        if self.config.multi_start:
            for amplitude in self.config.multi_start_amplitudes:
                for polarity in (1.0, -1.0):
                    candidates.append(
                        self._design_blocks(
                            design_center,
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
        previous_command: Array,
    ) -> _Rollout:
        """Roll the known kinematics forward on the posterior-mean command maps.

        Returns the commands, the per-step tracking cost, the chord tilt, the
        altitude, the regressors the identifier would see at each step, and the
        predicted specific force and rate magnitude.
        """

        if self.config.plan_parameterization == "slew_moves":
            commands = self._expand(self._plan_command_blocks(blocks, previous_command))
        else:
            commands = self._commands_from_normalized(self._expand(blocks))
        period = self.config.sample_period_s
        gravity = jnp.asarray((0.0, 0.0, -GRAVITY_M_S2))
        if self.config.regressor_trust:
            # Half-width of a uniform distribution with the accumulated second
            # moment of each nuisance regressor: ``sqrt(3) * rms``.  With no
            # samples the half-width is zero and the nuisance terms are off.
            collective_count = jnp.maximum(posterior.collective_information[7, 7], 1.0)
            angular_count = jnp.maximum(posterior.angular_information[10, 10], 1.0)
            velocity_trust = jnp.sqrt(
                3.0 * jnp.diag(posterior.collective_information)[4:7] / collective_count
            )
            rate_trust = jnp.sqrt(
                3.0 * jnp.diag(posterior.angular_information)[4:7] / angular_count
            )
            product_trust = jnp.sqrt(
                3.0 * jnp.diag(posterior.angular_information)[7:10] / angular_count
            )
        else:
            velocity_trust = jnp.full((3,), jnp.inf)
            rate_trust = jnp.full((3,), jnp.inf)
            product_trust = jnp.full((3,), jnp.inf)

        def step(
            carry: tuple[Array, Array, Array, Array],
            command: Array,
        ) -> tuple[
            tuple[Array, Array, Array, Array],
            tuple[Array, ...],
        ]:
            altitude, velocity, quaternion, angular_velocity = carry
            rotation = quaternion_to_rotation(quaternion)
            body_velocity = rotation.T @ velocity
            trusted_velocity = jnp.clip(body_velocity, -velocity_trust, velocity_trust)
            specific_force = (
                posterior.collective_per_command @ command
                + posterior.collective_velocity_coefficient @ trusted_velocity
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
            trusted_rates = jnp.clip(angular_velocity, -rate_trust, rate_trust)
            trusted_products = jnp.clip(rate_products, -product_trust, product_trust)
            angular_acceleration = (
                posterior.angular_per_command @ command
                + posterior.angular_rate_coefficient @ trusted_rates
                + posterior.angular_rate_product_coefficient @ trusted_products
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
            tracking_terms = jnp.stack(
                (
                    jnp.sum(jnp.square(next_velocity))
                    / self.config.velocity_tolerance_m_s**2,
                    jnp.sum(jnp.square(next_angular_velocity))
                    / self.config.body_rate_tolerance_rad_s**2,
                    chord_squared / self.config.tilt_tolerance_rad**2,
                    jnp.square(floor_error) / self.config.altitude_tolerance_m**2,
                )
            )
            tracking = jnp.sum(tracking_terms)
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
                body_velocity,
                specific_force,
                jnp.linalg.norm(next_angular_velocity),
                tracking_terms,
            )

        quaternion = state[6:10] / jnp.maximum(jnp.linalg.norm(state[6:10]), 1e-9)
        initial = (state[2], state[3:6], quaternion, state[10:13])
        _, outputs = jax.lax.scan(step, initial, commands)
        return _Rollout(commands, *outputs)

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

    def _spreads_marginal_coupled(
        self,
        rollout: _Rollout,
        features: Array,
        posterior: _Posterior,
    ) -> tuple[Array, Array, Array, Array]:
        """Box-averaged per-step spread, propagated with the attitude coupling.

        The per-step acceleration spread is the command-marginal one: the
        variance a command drawn uniformly from the box would still have after
        the plan's own excitation, which is plan-independent except through
        the information the plan buys, so a plan cannot lower its charge by
        promising to visit fewer points.  The propagation is the planned-
        trajectory one: the collective spread and ``|f| sigma_tilt`` together
        make up the acceleration spread, so thrust spent while the attitude is
        uncertain is charged for the velocity it might produce sideways.
        """

        period = self.config.sample_period_s
        covariance = self._planned_parameter_covariance(features, posterior)
        box_variance = jnp.trace(covariance) / 12.0
        angular_variance = 3.0 * box_variance
        collective_variance = (
            box_variance
            * posterior.collective_residual_variance
            / jnp.maximum(posterior.residual_variance, 1e-12)
        )
        steps = rollout.commands.shape[0]
        angular_spread = jnp.full((steps,), _safe_sqrt(angular_variance))
        collective_spread = jnp.full((steps,), _safe_sqrt(collective_variance))
        rate_spread = jnp.cumsum(period * angular_spread)
        tilt_spread = jnp.cumsum(period * rate_spread)
        acceleration_spread = (
            collective_spread + jnp.abs(rollout.specific_force) * tilt_spread
        )
        velocity_spread = jnp.cumsum(period * acceleration_spread)
        altitude_spread = jnp.cumsum(period * velocity_spread)
        return rate_spread, tilt_spread, velocity_spread, altitude_spread

    def _spreads_marginal_full(
        self,
        rollout: _Rollout,
        posterior: _Posterior,
        state: Array,
        planned: bool,
        held: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Full-regressor spread at the measured state, over one slew of command.

        The per-step acceleration spread is evaluated on the identifier's whole
        regressor set — command block, nuisance block, intercept — with the
        nuisance block fixed at the *measured* state (the body velocity and
        rates the vehicle has now, not the ones the plan predicts) and the
        command block averaged over the commands within one declared slew of
        the command the vehicle is holding: a uniform box of half-width one
        slew around it, which has that command as its mean and ``slew^2 / 3``
        per axis as its variance.  That is the neighbourhood the next re-plan
        can actually reach, so the spread is a statement about how well the
        response is known where the vehicle is and for the commands it can
        issue next, and a plan can change it only by buying information.
        Averaging over the whole box instead would keep valuing information
        about commands a settled hover will never issue, and the charge would
        never fall below what the goal costs.

        ``planned`` selects the posterior the plan would leave behind (its own
        features absorbed), which is what the charge is levied on; the current
        posterior is what decides how far along the horizon the goal is worth
        charging at all, and that must not depend on the plan.

        Propagation carries the attitude coupling: the acceleration spread is
        the collective's own plus ``|f| sigma_tilt`` on the plan's mean thrust.
        """

        period = self.config.sample_period_s
        steps = rollout.commands.shape[0]
        quaternion = state[6:10] / jnp.maximum(jnp.linalg.norm(state[6:10]), 1e-9)
        rotation = quaternion_to_rotation(quaternion)
        body_velocity = rotation.T @ state[3:6]
        rates = state[10:13]
        rate_products = jnp.stack(
            (rates[0] * rates[1], rates[0] * rates[2], rates[1] * rates[2])
        )
        if planned:
            normalized = (rollout.commands - self._jit_midpoint) / self._jit_span
            ones = jnp.ones((steps, 1))
            collective_planned = jnp.concatenate(
                (normalized, rollout.body_velocity, ones), axis=1
            )
            angular_planned = jnp.concatenate(
                (normalized, rollout.angular_velocity, rollout.rate_products, ones),
                axis=1,
            )
            collective_gram = (
                posterior.collective_information
                + collective_planned.T @ collective_planned
            )
            angular_gram = (
                posterior.angular_information + angular_planned.T @ angular_planned
            )
        else:
            collective_gram = posterior.collective_information
            angular_gram = posterior.angular_information

        held_feature = (held - self._jit_midpoint) / self._jit_span
        local_variance = self.config.slew_per_interval**2 / 3.0

        if self.config.information_neighbourhood == "box":
            # Uniform on the box: zero-mean command features with variance
            # ``1 / 12`` per axis.
            held_feature = jnp.zeros(_COMMAND_SIZE)
            local_variance = 1.0 / 12.0

        visited = self.config.information_neighbourhood == "visited"

        def box_variance(
            gram: Array, variance: Array, rest: Array, prior: Array
        ) -> Array:
            precision = gram / jnp.maximum(variance, 1e-12) + prior * jnp.eye(
                gram.shape[0]
            )
            covariance = jnp.linalg.inv(precision)
            if visited:
                # Average the predictive variance over the regressors the
                # identifier has actually seen, with the box as a single
                # pseudo-sample: the intercept column of the accumulated Gram
                # counts the samples, so this is the empirical second moment of
                # the visited regressors blended with the box at zero
                # information and dominated by the data thereafter.
                size = gram.shape[0]
                count = gram[size - 1, size - 1]
                pseudo = jnp.zeros((size, size)).at[:4, :4].set(jnp.eye(4) / 12.0)
                pseudo = pseudo.at[size - 1, size - 1].set(1.0)
                # The accumulated Gram is the one before this plan's samples;
                # ``gram`` may include them, so read the count from the
                # posterior's own accumulator.
                base = (
                    posterior.collective_information
                    if size == _COLLECTIVE_FEATURE_SIZE
                    else posterior.angular_information
                )
                count = base[size - 1, size - 1]
                moment = (base + pseudo) / (count + 1.0)
                return jnp.trace(covariance @ moment)
            feature = jnp.concatenate((held_feature, rest))
            return feature @ covariance @ feature + local_variance * jnp.trace(
                covariance[:4, :4]
            )

        collective_rest = jnp.concatenate((body_velocity, jnp.ones(1)))
        collective_variance = box_variance(
            collective_gram,
            posterior.collective_residual_variance,
            collective_rest,
            posterior.collective_prior_precision,
        )
        angular_rest = jnp.concatenate((rates, rate_products, jnp.ones(1)))
        angular_variance = jnp.sum(
            jax.vmap(
                lambda variance: box_variance(
                    angular_gram,
                    variance,
                    angular_rest,
                    posterior.angular_prior_precision,
                )
            )(posterior.angular_residual_variance)
        )
        angular_spread = jnp.full((steps,), _safe_sqrt(angular_variance))
        collective_spread = jnp.full((steps,), _safe_sqrt(collective_variance))
        rate_spread = jnp.cumsum(period * angular_spread)
        tilt_spread = jnp.cumsum(period * rate_spread)
        acceleration_spread = (
            collective_spread + jnp.abs(rollout.specific_force) * tilt_spread
        )
        velocity_spread = jnp.cumsum(period * acceleration_spread)
        altitude_spread = jnp.cumsum(period * velocity_spread)
        return rate_spread, tilt_spread, velocity_spread, altitude_spread

    def _spreads_sequential(
        self,
        rollout: _Rollout,
        posterior: _Posterior,
    ) -> tuple[Array, Array, Array, Array]:
        """The plan's own outcome spread, with its samples absorbed as taken.

        At every step the acceleration spread is the predictive spread at that
        step's own regressors under the posterior that has absorbed the plan's
        *earlier* steps and nothing later.  A move into a direction the
        posterior has not seen is therefore charged at its full spread until
        the plan's own samples of it arrive, and a command the plan has been
        holding is cheap because the plan has already learned it.  Neither a
        plan that promises to visit fewer points nor one that spends its
        credit before earning it can lower this charge.

        The posterior is carried as a coefficient covariance and updated by
        the rank-one step of recursive least squares, so the cost is one
        matrix-vector product per regression per step.
        """

        period = self.config.sample_period_s
        steps = rollout.commands.shape[0]
        normalized = (rollout.commands - self._jit_midpoint) / self._jit_span
        ones = jnp.ones((steps, 1))
        collective_features = jnp.concatenate(
            (normalized, rollout.body_velocity, ones), axis=1
        )
        angular_features = jnp.concatenate(
            (normalized, rollout.angular_velocity, rollout.rate_products, ones),
            axis=1,
        )
        collective_variance = jnp.maximum(posterior.collective_residual_variance, 1e-12)
        angular_variances = jnp.maximum(posterior.angular_residual_variance, 1e-12)
        collective_covariance = jnp.linalg.inv(
            posterior.collective_information / collective_variance
            + self.config.epsilon * jnp.eye(_COLLECTIVE_FEATURE_SIZE)
        )
        angular_covariances = jax.vmap(
            lambda variance: jnp.linalg.inv(
                posterior.angular_information / variance
                + self.config.epsilon * jnp.eye(_ANGULAR_FEATURE_SIZE)
            )
        )(angular_variances)

        def absorb(
            covariance: Array, feature: Array, variance: Array
        ) -> tuple[Array, Array]:
            gain = covariance @ feature
            predictive = feature @ gain
            updated = covariance - jnp.outer(gain, gain) / (variance + predictive)
            return updated, predictive

        def step(
            carry: tuple[Array, Array],
            features: tuple[Array, Array],
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            collective_cov, angular_cov = carry
            collective_feature, angular_feature = features
            collective_cov, force_variance = absorb(
                collective_cov, collective_feature, collective_variance
            )
            angular_cov, axis_variances = jax.vmap(absorb, in_axes=(0, None, 0))(
                angular_cov, angular_feature, angular_variances
            )
            return (collective_cov, angular_cov), (
                force_variance,
                jnp.sum(axis_variances),
            )

        _, (force_variance, angular_variance) = jax.lax.scan(
            step,
            (collective_covariance, angular_covariances),
            (collective_features, angular_features),
        )
        collective_spread = _safe_sqrt(force_variance)
        angular_spread = _safe_sqrt(angular_variance)
        rate_spread = jnp.cumsum(period * angular_spread)
        tilt_spread = jnp.cumsum(period * rate_spread)
        acceleration_spread = (
            collective_spread + jnp.abs(rollout.specific_force) * tilt_spread
        )
        velocity_spread = jnp.cumsum(period * acceleration_spread)
        altitude_spread = jnp.cumsum(period * velocity_spread)
        return rate_spread, tilt_spread, velocity_spread, altitude_spread

    def _spreads_trajectory(
        self,
        rollout: _Rollout,
        posterior: _Posterior,
        planned: bool = True,
    ) -> tuple[Array, Array, Array, Array]:
        """Per-step spread from the full-regressor posterior the plan implies.

        The command-marginal model above asks what a command drawn from the box
        would be worth on average.  This one asks the narrower question the goal
        actually needs: given the states this plan says the vehicle will be in,
        and given that the plan's own regressors will have been absorbed by the
        time the horizon ends, how well is the response known *at the points
        the plan visits*?  The regressors are the identifier's own — the
        normalized command, the nuisance block, and the intercept — so the
        planned information is the accumulated Gram plus the plan's own outer
        products, in the same basis, and no residualization is needed or
        wanted: a plan that moves a nuisance regressor is buying real
        information about the coefficient it multiplies.

        The one coupling the command-marginal model has no way to express is
        the decisive one here.  Specific force acts along the body ``z`` axis,
        so an uncertain attitude turns a *known* thrust into an unknown
        acceleration direction: the acceleration spread carries ``|f|
        sigma_tilt`` alongside the collective's own spread.  Thrusting while
        the attitude is uncertain is therefore charged for the velocity it
        might produce sideways, and levelling first is worth something in the
        same units as the goal.
        """

        period = self.config.sample_period_s
        steps = rollout.commands.shape[0]
        normalized = (rollout.commands - self._jit_midpoint) / self._jit_span
        ones = jnp.ones((steps, 1))
        collective_features = jnp.concatenate(
            (normalized, rollout.body_velocity, ones),
            axis=1,
        )
        angular_features = jnp.concatenate(
            (normalized, rollout.angular_velocity, rollout.rate_products, ones),
            axis=1,
        )
        collective_variance = jnp.maximum(posterior.collective_residual_variance, 1e-12)
        # ``planned`` credits the plan's own samples from the start of the
        # horizon (the fifth pass's form); without it the spread is the plan's
        # outcome under the posterior as it stands, which is what an action is
        # actually charged for.
        if planned:
            collective_gram = (
                posterior.collective_information
                + collective_features.T @ collective_features
            )
            angular_gram = (
                posterior.angular_information + angular_features.T @ angular_features
            )
        else:
            collective_gram = posterior.collective_information
            angular_gram = posterior.angular_information
        collective_precision = (
            collective_gram / collective_variance
            + posterior.collective_prior_precision * jnp.eye(_COLLECTIVE_FEATURE_SIZE)
        )
        force_spread = _safe_sqrt(
            jnp.einsum(
                "ti,ij,tj->t",
                collective_features,
                jnp.linalg.inv(collective_precision),
                collective_features,
            )
        )
        # The three angular axes share one Gram and one planned design and
        # differ only in their residual variance, so one inverse per axis is
        # the whole cost.

        def axis_variance(variance: Array) -> Array:
            precision = angular_gram / jnp.maximum(
                variance, 1e-12
            ) + posterior.angular_prior_precision * jnp.eye(_ANGULAR_FEATURE_SIZE)
            return jnp.einsum(
                "ti,ij,tj->t",
                angular_features,
                jnp.linalg.inv(precision),
                angular_features,
            )

        angular_spread = _safe_sqrt(
            jnp.sum(
                jax.vmap(axis_variance)(posterior.angular_residual_variance),
                axis=0,
            )
        )
        rate_spread = jnp.cumsum(period * angular_spread)
        tilt_spread = jnp.cumsum(period * rate_spread)
        acceleration_spread = (
            force_spread + jnp.abs(rollout.specific_force) * tilt_spread
        )
        velocity_spread = jnp.cumsum(period * acceleration_spread)
        altitude_spread = jnp.cumsum(period * velocity_spread)
        return rate_spread, tilt_spread, velocity_spread, altitude_spread

    def _terms(
        self,
        blocks: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
        initial_rate_scale: Array,
    ) -> _Terms:
        """Every objective term, plus the chance-constraint activity it implies."""

        rollout = self._rollout(blocks, state, posterior, previous_command)
        commands = rollout.commands
        features = self._information_features(
            commands,
            rollout.angular_velocity,
            rollout.rate_products,
        )
        if self.config.spread_model == "planned_trajectory":
            spreads = self._spreads_trajectory(rollout, posterior)
        elif self.config.spread_model == "command_marginal_coupled":
            spreads = self._spreads_marginal_coupled(rollout, features, posterior)
        elif self.config.spread_model == "command_marginal_full":
            spreads = self._spreads_marginal_full(
                rollout, posterior, state, True, previous_command
            )
        elif self.config.spread_model == "sequential":
            spreads = self._spreads_sequential(rollout, posterior)
        elif self.config.spread_model == "act_know":
            # The action charge: this plan's own outcome under the posterior as
            # it stands.  No credit for samples not yet taken, so a plan cannot
            # lower it by promising to learn, and a move into a direction the
            # posterior has not seen is charged at its full predicted spread.
            spreads = self._spreads_trajectory(rollout, posterior, planned=False)
        else:
            spreads = self._spreads(commands, features, posterior)
        (
            rate_spread,
            tilt_spread,
            vertical_speed_spread,
            altitude_spread,
        ) = spreads
        moves = jnp.diff(
            jnp.concatenate((previous_command[None, :], commands), axis=0),
            axis=0,
        )
        squared_moves = jnp.square(moves)
        if self.config.charge_unowned_transition:
            command_rate = self.config.w_rate * jnp.sum(squared_moves)
        else:
            # ``initial_rate_scale`` is one when the previous command was this
            # controller's own and zero when it was not, so the sum below is
            # exactly the slew of consecutive planned actions.
            command_rate = self.config.w_rate * (
                initial_rate_scale * jnp.sum(squared_moves[0])
                + jnp.sum(squared_moves[1:])
            )
        gain = self._information_gain(features, posterior)
        rate_cap = self.config.spread_cap * self.config.body_rate_tolerance_rad_s
        tilt_cap = self.config.spread_cap * self.config.tilt_tolerance_rad
        velocity_cap = self.config.spread_cap * self.config.velocity_tolerance_m_s
        altitude_cap = self.config.spread_cap * self.config.altitude_tolerance_m
        # Which steps each channel's prediction is still worth charging on.
        # Decided by the *current* posterior, so every candidate in a solve is
        # charged over the same steps and no plan can profit by learning less.
        if self.config.spread_model in (
            "command_marginal_full",
            "sequential",
            "act_know",
        ):
            current = self._spreads_marginal_full(
                rollout, posterior, state, False, previous_command
            )
        else:
            current = (rate_spread, tilt_spread, vertical_speed_spread, altitude_spread)
        # The executed block is always charged: the goal must never vanish
        # from the objective, however poor the posterior.
        executed = jnp.arange(rate_spread.shape[0]) < self.config.block_steps
        rate_known = (current[0] <= rate_cap) | executed
        tilt_known = (current[1] <= tilt_cap) | executed
        velocity_known = (current[2] <= velocity_cap) | executed
        altitude_known = (current[3] <= altitude_cap) | executed
        if self.config.spread_model in (
            "planned_trajectory",
            "command_marginal_coupled",
            "command_marginal_full",
            "sequential",
            "act_know",
        ):
            # Clipping, rather than dropping, is what keeps every chance
            # penalty live.  A spread past the cap says "unknown", and what a
            # bounded charge for "unknown" buys is that no plan can profit by
            # making the prediction worse, while a plan whose own excitation
            # brings a channel back under the cap is charged strictly less.
            # The gradient goes flat inside a saturated channel; the seeds, not
            # the gradient, are what move the plan there.
            altitude_saturated = altitude_spread > altitude_cap
            tilt_saturated = tilt_spread > tilt_cap
            if (
                self.config.spread_model != "sequential"
                or self.config.clip_action_spread
            ):
                rate_spread = jnp.minimum(rate_spread, rate_cap)
                tilt_spread = jnp.minimum(tilt_spread, tilt_cap)
                vertical_speed_spread = jnp.minimum(vertical_speed_spread, velocity_cap)
                altitude_spread = jnp.minimum(altitude_spread, altitude_cap)
            altitude_supported = jnp.ones_like(altitude_spread, dtype=bool)
            tilt_supported = jnp.ones_like(tilt_spread, dtype=bool)
        else:
            altitude_supported = altitude_spread <= altitude_cap
            tilt_supported = tilt_spread <= tilt_cap
            altitude_saturated = ~altitude_supported
            tilt_saturated = ~tilt_supported
        if (
            self.config.spread_model == "act_know"
            and not self.config.clip_action_spread
        ):
            # An unclipped action spread is charged only where the posterior
            # can see; beyond that the knowledge term is the whole statement.
            rate_spread = jnp.where(rate_known, rate_spread, 0.0)
            tilt_spread = jnp.where(tilt_known, tilt_spread, 0.0)
            vertical_speed_spread = jnp.where(
                velocity_known, vertical_speed_spread, 0.0
            )
            altitude_spread = jnp.where(altitude_known, altitude_spread, 0.0)
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
            if self.config.spread_model in ("sequential", "act_know"):
                # What later plans will need: the spread within one slew of the
                # held command under the posterior this plan leaves behind,
                # clipped at the cap.
                # This is the term that values learning beyond the plan's own
                # commands, and it is the only term that can prefer a probe
                # to a hold when the mean model cannot tell them apart.
                know = self._spreads_marginal_full(
                    rollout, posterior, state, True, previous_command
                )
                spread_charge = spread_charge + jnp.sum(
                    jnp.square(jnp.minimum(know[0], rate_cap))
                    / self.config.body_rate_tolerance_rad_s**2
                    + jnp.square(jnp.minimum(know[1], tilt_cap))
                    / self.config.tilt_tolerance_rad**2
                    + jnp.square(jnp.minimum(know[2], velocity_cap))
                    / self.config.velocity_tolerance_m_s**2
                    + jnp.square(jnp.minimum(know[3], altitude_cap))
                    / self.config.altitude_tolerance_m**2
                )
        else:
            spread_charge = jnp.asarray(0.0)

        altitude_breach = jnp.maximum(
            self.config.altitude_floor_m
            + self.config.beta * altitude_spread
            - rollout.altitude,
            0.0,
        )
        tilt_breach = jnp.maximum(
            rollout.tilt
            + self.config.beta * tilt_spread
            - self.config.maximum_tilt_rad,
            0.0,
        )
        if self.config.goal_horizon == "posterior":
            # The goal and its chance penalties are charged only where the
            # posterior can still see: each term on the steps its own channel's
            # spread is under the cap.  A mean rollout that has run past what
            # the data supports carries no cost there, and the clipped spread
            # charge is the whole statement about those steps.
            known = jnp.stack(
                (velocity_known, rate_known, tilt_known, altitude_known), axis=1
            )
            tracking = jnp.sum(jnp.where(known, rollout.tracking_terms, 0.0))
            altitude_supported = altitude_known
            tilt_supported = tilt_known
            rate_supported = rate_known
        else:
            tracking = jnp.sum(rollout.tracking)
            rate_supported = jnp.ones_like(rate_spread, dtype=bool)
        if self.config.charge_body_rate_limit:
            # The declared rate limit, charged in exactly the form the tilt
            # limit is: predicted magnitude plus a reserved spread, past the
            # declared maximum, squared.
            body_rate_penalty = jnp.sum(
                jnp.where(
                    rate_supported,
                    jnp.square(
                        jnp.maximum(
                            rollout.rate_norm
                            + self.config.beta * rate_spread
                            - self.config.maximum_body_rate_rad_s,
                            0.0,
                        )
                    ),
                    0.0,
                )
            )
        else:
            body_rate_penalty = jnp.asarray(0.0)
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
            tracking=tracking,
            spread_charge=spread_charge,
            command_rate=command_rate,
            information_gain=gain,
            altitude_penalty=altitude_penalty,
            tilt_penalty=tilt_penalty,
            body_rate_penalty=body_rate_penalty,
            maximum_rate_spread=jnp.max(rate_spread),
            maximum_tilt_spread=jnp.max(tilt_spread),
            maximum_velocity_spread=jnp.max(vertical_speed_spread),
            maximum_altitude_spread=jnp.max(altitude_spread),
            altitude_active_steps=jnp.sum((altitude_breach > 0.0) & altitude_supported),
            tilt_active_steps=jnp.sum((tilt_breach > 0.0) & tilt_supported),
            altitude_saturated_steps=jnp.sum(altitude_saturated),
            tilt_saturated_steps=jnp.sum(tilt_saturated),
        )

    def _objective(
        self,
        blocks: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
        initial_rate_scale: Array,
    ) -> Array:
        terms = self._terms(
            blocks,
            state,
            posterior,
            previous_command,
            initial_rate_scale,
        )
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
            + terms.body_rate_penalty
        )

    def _plan_diagnostics(
        self,
        blocks: Array,
        state: Array,
        posterior: _Posterior,
        previous_command: Array,
    ) -> tuple[Array, Array]:
        """Realized amplitude and planned information rank of a finished plan.

        Kept out of the objective because the eigenvalue decomposition the rank
        needs has no usable gradient at the repeated eigenvalues an unexcited
        design produces.
        """

        rollout = self._rollout(blocks, state, posterior, previous_command)
        features = self._information_features(
            rollout.commands,
            rollout.angular_velocity,
            rollout.rate_products,
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
        initial_rate_scale: Array,
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
                    initial_rate_scale,
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
        design_center: Array,
        initial_rate_scale: Array,
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

        candidates = self._candidate_blocks(
            warm_blocks,
            previous_command,
            design_center,
            state,
            posterior,
        )
        values = jax.vmap(
            lambda candidate: self._objective(
                candidate,
                state,
                posterior,
                previous_command,
                initial_rate_scale,
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
            initial_rate_scale,
        )
        final_blocks, final_value, iteration, converged, stalled, failed = (
            self._optimize(
                blocks,
                value,
                gradient,
                state,
                posterior,
                previous_command,
                initial_rate_scale,
            )
        )
        terms = self._terms(
            final_blocks,
            state,
            posterior,
            previous_command,
            initial_rate_scale,
        )
        amplitude, rank = self._plan_diagnostics(
            final_blocks,
            state,
            posterior,
            previous_command,
        )
        # The executed command is the plan's first block, whatever the
        # parameterization: the horizon holds each block over its own steps.
        plan = self._plan_command_blocks(final_blocks, previous_command)
        return (
            plan[0],
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

    def base_action(
        self,
        belief: RecursiveBootstrapBelief,
    ) -> tuple[np.ndarray, str]:
        """The command the designs are centered on, and why that one.

        A multi-start design is a spread of commands around a center, and at
        zero information the center is the only thing in the controller that
        says where in the command box the vehicle is likely to want to be.
        Centering it on the previous command answers that question with
        whatever the vehicle happened to be carrying at handover, which in a
        motors-off release is the lower bound, and no posterior ever says so.

        This answers it from the command contract instead.  The contract is
        that each command is a normalized thrust fraction on ``[0, 1]`` and
        that hover is somewhere in that box; with no information about where,
        the box midpoint is the maximum-entropy choice, and it is the unique
        point that minimizes the worst-case distance to hover over the box.  It
        is a statement about the interface, in the same class as the bounds
        themselves, and it stays the same number for every vehicle: nothing
        here is fitted, measured, or tuned to an airframe.

        The moment the posterior can say where hover is, the declaration stops
        being the best available answer and is replaced by the estimate.  That
        handover is this method's return value, and the caller records which of
        the two was in use, so the annealing is visible rather than implicit.

        The condition is the identifier's own support rule and nothing new: the
        command evidence spans all four motors, the fitted angular effect spans
        all three body axes, and the collective effect implies a hover command
        inside the command box.  That is the same rule the certification
        transaction requires of a candidate and the same one working mode hands
        control over on, so this controller anneals off its declaration at
        exactly the moment the rest of the stack agrees the posterior can
        answer the question.  A weaker rule is not merely less careful, it is
        wrong here: a hover command fitted from a handful of rank-deficient
        samples exists long before it means anything, and centering on it would
        replace a declaration that is true by construction with an estimate
        that is not yet true at all.
        """

        hover = belief.hover_command
        if (
            hover is not None
            and int(belief.command_evidence_rank) == _COMMAND_SIZE
            and int(belief.angular_effect_rank) == 3
        ):
            candidate = np.asarray(hover, dtype=np.float64)
            if candidate.shape == (_COMMAND_SIZE,) and np.all(np.isfinite(candidate)):
                clipped = np.clip(candidate, self._minimum, self._maximum)
                return clipped, "hover_estimate"
        return self._midpoint.copy(), "box_midpoint"

    def _design_center(
        self,
        belief: RecursiveBootstrapBelief,
        held: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        if self.config.center_designs_on_base_action:
            return self.base_action(belief)
        return held, "previous_command"

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
        design_center: np.ndarray | None = None,
        design_center_source: str = "previous_command",
        charged_initial_transition: bool = True,
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
            body_rate_penalty=0.0,
            maximum_rate_spread_rad_s=0.0,
            maximum_tilt_spread_rad=0.0,
            maximum_velocity_spread_m_s=0.0,
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
            design_center=(command if design_center is None else design_center),
            design_center_source=design_center_source,
            charged_initial_transition=charged_initial_transition,
            reason=reason,
        )

    def _prior_precisions(
        self, belief: RecursiveBootstrapBelief
    ) -> tuple[float, float]:
        """Prior precision of an unlearned direction, from what is learned.

        With ``empirical_prior_scale`` the mean squared effect per supported
        direction of each map is the presumed squared scale of an unlearned
        direction, and the prior precision is its reciprocal, never narrower
        than ``epsilon``: the data can only widen the prior.  Without it, or
        before anything is supported, ``epsilon`` stands.
        """

        epsilon = float(self.config.epsilon)
        if not self.config.empirical_prior_scale:
            return epsilon, epsilon
        angular = np.asarray(belief.angular_acceleration_per_command, dtype=np.float64)
        angular_rank = int(belief.angular_effect_rank)
        angular_precision = epsilon
        if angular_rank >= 1:
            scale = (
                float(np.sum(np.square(angular * self._span[None, :]))) / angular_rank
            )
            if scale > 0.0:
                angular_precision = min(epsilon, 1.0 / scale)
        collective = np.asarray(
            belief.collective_acceleration_per_command, dtype=np.float64
        )
        collective_rank = int(belief.command_evidence_rank)
        collective_precision = epsilon
        if collective_rank >= 1:
            scale = float(np.sum(np.square(collective * self._span))) / collective_rank
            if scale > 0.0:
                collective_precision = min(epsilon, 1.0 / scale)
        return collective_precision, angular_precision

    def _posterior(self, belief: RecursiveBootstrapBelief) -> _Posterior:
        variance = float(np.mean(np.square(belief.angular_residual_std_rad_s2)))
        collective_prior, angular_prior = self._prior_precisions(belief)
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
            collective_prior_precision=jnp.asarray(collective_prior),
            angular_prior_precision=jnp.asarray(angular_prior),
            collective_information=jnp.asarray(belief.collective_information),
            angular_information=jnp.asarray(belief.angular_information),
            angular_residual_variance=jnp.asarray(
                np.square(belief.angular_residual_std_rad_s2)
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
            and np.all(np.isfinite(belief.collective_information))
            and np.all(np.isfinite(belief.angular_information))
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

        Under ``"slew_moves"`` the shift happens on the previous plan's
        absolute commands and the result is converted back into moves from the
        command the vehicle is now holding.  The previous plan's first block is
        the one that was executed, so the held command is where the shifted
        plan already starts, and every move in the shifted plan is a move the
        previous solve had already chosen; the clip to the slew box is
        therefore a formality except when the box clipped the plan.
        """

        moves = self.config.plan_parameterization == "slew_moves"
        cold = (
            np.zeros((self.block_count, _COMMAND_SIZE))
            if moves
            else np.repeat(
                (2.0 * (held - self._minimum) / self._span - 1.0)[None, :],
                self.block_count,
                axis=0,
            )
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
        if moves:
            steps = np.diff(
                np.concatenate((held[None, :], shifted), axis=0),
                axis=0,
            )
            return np.clip(steps / self._slew_step, -1.0, 1.0), True
        normalized = 2.0 * (shifted - self._minimum) / self._span - 1.0
        return np.clip(normalized, -1.0, 1.0), True

    def solve(
        self,
        state: Sequence[float],
        belief: RecursiveBootstrapBelief,
        previous_command: Sequence[float],
        warm_start: Any = None,
        previous_command_owned: bool = True,
    ) -> DualControlResult:
        """Return one bounded command, or the previous one when it cannot.

        ``previous_command_owned`` says whether the command the vehicle is
        carrying was issued by this controller.  It is false exactly once per
        flight in the throw diagnostic — the first interval after enable, whose
        predecessor is the motors-off release — and the rate cost then does not
        charge for leaving it, provided the configuration says the rate cost is
        a slew cost on the controller's own actions.  It defaults to true, so a
        caller that does not distinguish gets the behaviour it always had.

        A state or posterior this controller cannot act on never raises: the
        result comes back with ``command_usable`` false, a status, and the
        previous command clipped into bounds.  There is no fallback controller.
        """

        held = self._held_command(previous_command)
        center, center_source = self._design_center(belief, held)
        charged = bool(previous_command_owned) or self.config.charge_unowned_transition
        state_array = np.asarray(state, dtype=np.float64)
        if (
            state_array.shape != (_STATE_SIZE,)
            or not np.all(np.isfinite(state_array))
            or float(np.linalg.norm(state_array[6:10])) < 1e-9
            or not self._belief_is_finite(belief)
        ):
            return self._unusable(
                held,
                SolveStatus.INVALID_INPUT,
                "unusable_input",
                design_center=center,
                design_center_source=center_source,
                charged_initial_transition=charged,
            )

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
            jnp.asarray(center),
            jnp.asarray(1.0 if charged else 0.0),
        )
        command_array = np.asarray(command, dtype=np.float64)
        value_float = float(value)
        finite = bool(np.all(np.isfinite(command_array))) and math.isfinite(value_float)
        if not finite:
            return self._unusable(
                held,
                SolveStatus.NONFINITE_OBJECTIVE,
                "nonfinite_solve",
                design_center=center,
                design_center_source=center_source,
                charged_initial_transition=charged,
            )
        bounded = np.clip(command_array, self._minimum, self._maximum)
        if self.config.plan_parameterization == "slew_moves":
            # The declared limits are enforced here, in double precision, for
            # the same reason the command box is: the solve runs in the JAX
            # default precision and a rounded decision variable can land a
            # fraction of an ulp outside a bound the caller was promised.  The
            # slew window always contains the held command, so this clip can
            # never push the command back out of the box.
            bounded = np.clip(
                bounded,
                held - self._slew_step,
                held + self._slew_step,
            )
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
            body_rate_penalty=float(terms.body_rate_penalty),
            maximum_rate_spread_rad_s=float(terms.maximum_rate_spread),
            maximum_tilt_spread_rad=float(terms.maximum_tilt_spread),
            maximum_velocity_spread_m_s=float(terms.maximum_velocity_spread),
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
            design_center=center,
            design_center_source=center_source,
            charged_initial_transition=charged,
        )
