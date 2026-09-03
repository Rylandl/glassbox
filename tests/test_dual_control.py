"""Contracts for the experimental dual-control NMPC.

The beliefs these tests feed the controller are produced by driving the real
recursive identifier over transitions from a small synthetic multirotor, rather
than by fabricating posterior arrays.  That keeps every belief internally
consistent with the identifier's own support rules, which is what the
controller consumes, and it means "zero information" here is the same object
the flight loop starts from.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control._common import quaternion_to_rotation
from glassbox.control.online_bootstrap import (
    RecursiveBootstrapBelief,
    RecursiveBootstrapIdentifier,
)
from glassbox.core.dynamics import GRAVITY_M_S2
from glassbox.experimental.dual_control import (
    DUAL_CONTROL_VARIANTS,
    DualControlConfig,
    DualControlNMPC,
    _Rollout,
    command_information_log_determinant,
    design_sign_pattern,
    dual_control_config,
)

#: A synthetic four-motor vehicle used only to manufacture posteriors.  It is a
#: test fixture, not a prior: the controller never sees these numbers.
_COLLECTIVE_PER_COMMAND = np.asarray((4.6, 4.7, 4.5, 4.6))
_ANGULAR_PER_COMMAND = np.asarray(
    (
        (-90.0, -88.0, 92.0, 90.0),
        (86.0, -90.0, -88.0, 92.0),
        (-24.0, 26.0, -25.0, 23.0),
    )
)
_SAMPLE_PERIOD_S = 0.01


def _synthetic_transition(
    state: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    """Advance the synthetic vehicle one interval under one command."""

    quaternion = state[6:10] / np.linalg.norm(state[6:10])
    w, x, y, z = quaternion
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    specific_force = float(_COLLECTIVE_PER_COMMAND @ command)
    world_acceleration = rotation[:, 2] * specific_force - np.asarray(
        (0.0, 0.0, GRAVITY_M_S2)
    )
    angular_velocity = state[10:13]
    angular_acceleration = _ANGULAR_PER_COMMAND @ command - 0.4 * angular_velocity
    nxt = state.copy()
    nxt[0:3] = state[0:3] + _SAMPLE_PERIOD_S * state[3:6]
    nxt[3:6] = state[3:6] + _SAMPLE_PERIOD_S * world_acceleration
    half = 0.5 * _SAMPLE_PERIOD_S
    rate = np.asarray(
        (
            -half * (x * angular_velocity[0] + y * angular_velocity[1])
            - half * z * angular_velocity[2],
            half * (w * angular_velocity[0] + y * angular_velocity[2])
            - half * z * angular_velocity[1],
            half * (w * angular_velocity[1] + z * angular_velocity[0])
            - half * x * angular_velocity[2],
            half * (w * angular_velocity[2] + x * angular_velocity[1])
            - half * y * angular_velocity[0],
        )
    )
    updated = quaternion + rate
    nxt[6:10] = updated / np.linalg.norm(updated)
    nxt[10:13] = angular_velocity + _SAMPLE_PERIOD_S * angular_acceleration
    return nxt


def _belief_after(interval_count: int, seed: int = 7) -> RecursiveBootstrapBelief:
    """Return the working belief after this many excited synthetic intervals."""

    identifier = RecursiveBootstrapIdentifier()
    if interval_count == 0:
        return identifier.belief
    generator = np.random.default_rng(seed)
    state = np.zeros(13)
    state[2] = 3.0
    state[6] = 1.0
    hover = GRAVITY_M_S2 / float(np.sum(_COLLECTIVE_PER_COMMAND))
    belief = identifier.belief
    for _ in range(interval_count):
        command = np.clip(hover + 0.15 * generator.standard_normal(4), 0.0, 1.0)
        nxt = _synthetic_transition(state, command)
        belief = identifier.update(state, nxt, command, _SAMPLE_PERIOD_S)
        state = nxt
        state[10:13] = np.clip(state[10:13], -3.0, 3.0)
    return belief


def _random_states(count: int, seed: int = 3) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    states = []
    for _ in range(count):
        state = np.zeros(13)
        state[0:3] = generator.uniform(-2.0, 2.0, 3)
        state[2] = generator.uniform(0.2, 6.0)
        state[3:6] = generator.uniform(-8.0, 8.0, 3)
        quaternion = generator.standard_normal(4)
        state[6:10] = quaternion / np.linalg.norm(quaternion)
        state[10:13] = generator.uniform(-4.0, 4.0, 3)
        states.append(state)
    return states


def _controller() -> DualControlNMPC:
    return DualControlNMPC(DualControlConfig(sample_period_s=_SAMPLE_PERIOD_S))


def test_commands_stay_finite_and_bounded_on_random_states_and_posteriors() -> None:
    controller = _controller()
    beliefs = [_belief_after(count) for count in (0, 1, 6, 40, 200)]
    minimum = np.asarray(controller.config.command_minimum)
    maximum = np.asarray(controller.config.command_maximum)
    for belief in beliefs:
        for state in _random_states(6):
            previous = np.clip(np.full(4, 0.4), minimum, maximum)
            result = controller.solve(state, belief, previous)
            assert np.all(np.isfinite(result.command))
            assert np.all(result.command >= minimum - 1e-9)
            assert np.all(result.command <= maximum + 1e-9)
            assert np.isfinite(result.objective_value)
            assert np.isfinite(result.information_gain)
            json.dumps(result.to_dict(), allow_nan=False)


def test_unusable_state_returns_the_previous_bounded_command() -> None:
    controller = _controller()
    belief = _belief_after(0)
    previous = np.full(4, 0.3)
    for bad in (np.full(13, np.nan), np.zeros(13)):
        result = controller.solve(bad, belief, previous)
        assert not result.command_usable
        assert np.allclose(result.command, previous)
    out_of_bounds = controller.solve(_random_states(1)[0], belief, np.full(4, 5.0))
    assert np.all(out_of_bounds.command <= 1.0 + 1e-9)


def test_information_gain_is_nonnegative_and_grows_with_excitation() -> None:
    controller = _controller()
    belief = _belief_after(0)
    midpoint = 0.5 * (
        np.asarray(controller.config.command_minimum)
        + np.asarray(controller.config.command_maximum)
    )
    pattern = np.asarray(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        )
    )
    gains = []
    for amplitude in (0.0, 0.05, 0.15, 0.35, 0.5):
        commands = np.clip(midpoint + amplitude * pattern, 0.0, 1.0)
        gain = controller.expected_information_gain(commands, belief)
        assert gain >= -1e-6
        gains.append(gain)
    assert gains[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(gains) > 0.0)


def test_tracking_dominates_a_well_informed_posterior_at_hover() -> None:
    controller = _controller()
    belief = _belief_after(200)
    assert belief.command_evidence_rank == 4
    assert belief.hover_command is not None
    hover = np.asarray(belief.hover_command)
    state = np.zeros(13)
    state[2] = 3.0
    state[6] = 1.0

    result = controller.solve(state, belief, hover)
    for _ in range(20):
        result = controller.solve(state, belief, result.command, result)
    # The plan holds a hover: the collective matches the identified hover
    # command closely, and the small per-motor spread is the differential the
    # belief's own angular intercept asks for, not excitation.
    assert float(np.mean(result.command)) == pytest.approx(
        float(np.mean(hover)), abs=2e-3
    )
    assert np.max(np.abs(result.command - hover)) < 2e-2
    # The information term has annealed to a fraction of a nat, so the plan is
    # a tracking plan even though nothing froze or switched the objective off.
    assert controller.config.w_info * result.information_gain < 1.0
    assert result.tracking_cost < 1.0


def test_warm_start_is_never_worse_than_cold_on_the_objective() -> None:
    """The seed a warm solve starts from is never worse than the cold one.

    The solve evaluates both the shifted plan and the held command and starts
    from whichever scores lower, so this is a structural property rather than an
    empirical one.  It is stated on the seed and not on the final iterate
    because ten projected-gradient steps on a nonconvex objective can descend
    further from a worse starting point; the aggregate below is the honest
    version of that weaker claim.
    """

    controller = _controller()
    belief = _belief_after(200)
    previous = np.asarray(belief.hover_command)
    cold_objectives = []
    warm_objectives = []
    for state in _random_states(5, seed=11):
        seed = controller.solve(state, belief, previous)
        cold = controller.solve(state, belief, seed.command)
        warm = controller.solve(state, belief, seed.command, warm_start=seed)
        assert warm.seed_objective_value <= cold.seed_objective_value + 1e-4 * max(
            abs(cold.seed_objective_value), 1.0
        )
        assert warm.used_warm_start or warm.seed_objective_value == pytest.approx(
            cold.seed_objective_value
        )
        cold_objectives.append(cold.objective_value)
        warm_objectives.append(warm.objective_value)
    assert float(np.mean(warm_objectives)) <= float(np.mean(cold_objectives)) * 1.02


def test_the_solve_program_is_compiled_once_across_posteriors() -> None:
    controller = _controller()
    assert controller.jit_cache_size == 0
    states = _random_states(4, seed=5)
    for count in (0, 1, 6, 40, 200):
        belief = _belief_after(count)
        result = None
        for state in states:
            result = controller.solve(state, belief, np.full(4, 0.45), result)
    assert controller.jit_cache_size == 1


def test_log_determinant_helper_matches_the_objectives_information_floor() -> None:
    config = DualControlConfig(sample_period_s=_SAMPLE_PERIOD_S)
    empty = command_information_log_determinant(_belief_after(0), config)
    informed = command_information_log_determinant(_belief_after(200), config)
    assert empty == pytest.approx(4.0 * np.log(config.epsilon))
    assert informed > empty


@pytest.mark.crazyflow
def test_the_superseded_first_pass_mode_still_runs_the_canonical_case() -> None:
    """The first pass stays runnable so its recorded failure is reproducible.

    It is not the default arm any more, and it is not expected to fly: the
    study report is where its behaviour is read.  What is asserted here is only
    that the switch still selects it and that it still produces a bounded,
    serializable run.
    """

    pytest.importorskip("crazyflow")

    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        DUAL_CONTROL_MODEL,
        run_crazyflow_throw_study,
    )

    canonical = next(
        case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == "canonical"
    )
    report = run_crazyflow_throw_study((canonical,), (DUAL_CONTROL_MODEL,))

    assert report["control_models"] == [DUAL_CONTROL_MODEL]
    metrics = report["cases"][0]["modes"][DUAL_CONTROL_MODEL]
    assert metrics["dual_control"]["config"]["variant"] == "pass1"
    assert metrics["flight"]["non_finite_value_count"] == 0
    assert metrics["flight"]["command_bound_violation_count"] == 0
    dual = metrics["dual_control"]
    # The first pass falls to the floor, and a trial now stops at its first
    # floor contact: the per-interval series run to the contact and no further.
    assert metrics["flight"]["floor_contact_time_s"] is not None
    interval_count = metrics["identification"]["working_interval_count"]
    assert 0 < interval_count < 900
    assert len(dual["information_gain_per_step"]) == interval_count
    assert len(dual["command_information_log_determinant"]) == interval_count
    assert len(dual["early_commands"]) == min(30, interval_count)
    assert dual["solve_iterations"]["total"] > 0
    assert set(dual["status_counts"]) <= {
        "converged",
        "iteration_limit",
        "stalled",
        "line_search_failed",
        "nonfinite_objective",
        "invalid_input",
    }
    assert "altitude_active_step_total" in dual["chance_constraints"]
    json.dumps(report, allow_nan=False)


def _uniform_plan(level: float, steps: int = 30) -> np.ndarray:
    return np.repeat(np.full(4, level)[None, :], steps, axis=0)


def _declared_plan(
    controller: DualControlNMPC,
    amplitude: float,
    base: float = 0.5,
) -> np.ndarray:
    """The controller's own declared design, expanded to horizon steps."""

    blocks = np.clip(
        base + amplitude * design_sign_pattern(controller.block_count),
        0.0,
        1.0,
    )
    return np.repeat(blocks, controller.config.block_steps, axis=0)[
        : controller.config.horizon_steps
    ]


def test_the_declared_design_is_full_rank_after_intercept_residualization() -> None:
    """One horizon of the design spans all four command directions.

    Centering is exactly what the identifier's intercept does to the command
    Gram, so a design whose centered Gram is rank deficient could never buy
    rank four inside a single horizon however large its amplitude.
    """

    signs = design_sign_pattern(DualControlConfig().block_count)
    centered = signs - signs.mean(axis=0)
    assert np.linalg.matrix_rank(centered.T @ centered) == 4


def test_a_uniform_plan_earns_exactly_zero_planned_information() -> None:
    """The identifier's intercept absorbs a uniform command, so it is free.

    Not approximately zero: the planned features are centered over the horizon
    before the Gram is formed, so a uniform plan contributes an exactly zero
    matrix and the log-determinant difference is the same float twice.  The
    first pass, which regressed on raw commands, credited a uniform plan with
    several nats it could never collect, and that credit is what let the
    optimizer sit still.
    """

    controller = _controller()
    superseded = DualControlNMPC(
        dual_control_config("pass1", sample_period_s=_SAMPLE_PERIOD_S)
    )
    for count in (0, 6, 200):
        belief = _belief_after(count)
        for level in (0.0, 0.317, 0.5, 1.0):
            plan = _uniform_plan(level)
            assert controller.expected_information_gain(plan, belief) == 0.0
    assert (
        superseded.expected_information_gain(_uniform_plan(0.317), _belief_after(0))
        > 1.0
    )


def test_the_declared_design_outearns_every_uniform_plan() -> None:
    controller = _controller()
    for count in (0, 6, 200):
        belief = _belief_after(count)
        design = controller.expected_information_gain(
            _declared_plan(controller, 0.12), belief
        )
        uniform = [
            controller.expected_information_gain(_uniform_plan(level), belief)
            for level in np.linspace(0.0, 1.0, 21)
        ]
        assert design > max(uniform)
        assert max(uniform) == 0.0


def test_the_multi_start_never_seeds_worse_than_the_warm_start() -> None:
    """Adding candidates can only lower the objective the refinement starts at.

    The warm start and the held command are both in the candidate set and the
    seed is their argmin, so this is structural.  The refinement is monotone on
    top of that: every accepted line-search step strictly decreases the
    objective and a rejected one leaves the iterate alone.
    """

    controller = _controller()
    single = DualControlNMPC(
        DualControlConfig(sample_period_s=_SAMPLE_PERIOD_S, multi_start=False)
    )
    for count in (0, 6, 200):
        belief = _belief_after(count)
        for state in _random_states(4, seed=17):
            previous = np.full(4, 0.4)
            warm = controller.solve(state, belief, previous)
            multi = controller.solve(state, belief, previous, warm_start=warm)
            plain = single.solve(state, belief, previous, warm_start=warm)
            tolerance = 1e-4 * max(abs(plain.seed_objective_value), 1.0)
            assert multi.seed_objective_value <= plain.seed_objective_value + tolerance
            assert multi.objective_value <= multi.seed_objective_value + tolerance


def test_expected_cost_probes_a_wide_posterior_and_tracks_a_tight_one() -> None:
    """The spread charge, not a weight, decides between probing and tracking.

    With no information the spread charge is the only term with a usable
    gradient and an informative plan collapses it, so the optimizer takes one of
    the declared designs and buys rank four.  With a tight posterior the same
    charge is already small, excitation cannot repay its command-rate cost, and
    the optimizer holds the tracking plan.  Nothing switched: the same objective
    reports both.
    """

    controller = _controller()
    state = np.zeros(13)
    state[2] = 3.0
    state[6] = 1.0
    state[3:6] = (0.0, 0.0, -2.0)

    wide = controller.solve(state, _belief_after(0), np.full(4, 0.3))
    assert wide.selected_amplitude > 0.0
    assert wide.selected_candidate.startswith("design_")
    assert wide.planned_information_rank == 4
    assert wide.spread_charge > 0.0

    belief = _belief_after(200)
    hover = np.asarray(belief.hover_command)
    tight = controller.solve(state, belief, hover)
    for _ in range(20):
        tight = controller.solve(state, belief, tight.command, tight)
    assert tight.selected_amplitude == 0.0
    assert tight.plan_amplitude < wide.plan_amplitude
    assert tight.spread_charge < wide.spread_charge


def test_every_variant_compiles_one_program_and_stays_bounded() -> None:
    beliefs = [_belief_after(count) for count in (0, 1, 6, 40, 200)]
    for variant in ("pass1", "pass2a", "pass2b"):
        controller = DualControlNMPC(
            dual_control_config(variant, sample_period_s=_SAMPLE_PERIOD_S)
        )
        assert controller.config.variant == variant
        assert controller.jit_cache_size == 0
        for belief in beliefs:
            result = None
            for state in _random_states(4, seed=5):
                result = controller.solve(state, belief, np.full(4, 0.45), result)
                assert np.all(np.isfinite(result.command))
                assert np.all(result.command >= -1e-9)
                assert np.all(result.command <= 1.0 + 1e-9)
        assert controller.jit_cache_size == 1


@pytest.mark.crazyflow
def test_the_canonical_dual_control_smoke_stays_under_a_minute() -> None:
    pytest.importorskip("crazyflow")

    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        DUAL_CONTROL_PASS2B_MODEL,
        run_crazyflow_throw_study,
    )

    canonical = next(
        case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == "canonical"
    )
    started = time.perf_counter()
    report = run_crazyflow_throw_study((canonical,), (DUAL_CONTROL_PASS2B_MODEL,))
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0

    metrics = report["cases"][0]["modes"][DUAL_CONTROL_PASS2B_MODEL]
    dual = metrics["dual_control"]
    assert dual["config"]["variant"] == "pass2b"
    assert metrics["flight"]["non_finite_value_count"] == 0
    assert metrics["flight"]["command_bound_violation_count"] == 0
    assert dual["unusable_command_count"] == 0
    assert len(dual["multi_start"]["selected_candidate"]) == 900
    assert len(dual["information_rank"]) == 900
    assert set(dual["multi_start"]["selection_counts"]) <= set(
        (
            "warm_start",
            "hold",
            "none",
            *(
                f"design_{amplitude:g}_{polarity}"
                for amplitude in dual["config"]["multi_start_amplitudes"]
                for polarity in ("plus", "minus")
            ),
        )
    )
    assert dual["charge_series"][0]["spread_charge"] >= 0.0
    json.dumps(report, allow_nan=False)


@pytest.mark.crazyflow
def test_the_pass_three_arm_stages_regressors_and_records_the_sign_projection() -> None:
    """Pass three is pass 2b's objective on a staged, sign-constrained belief.

    Both changes live in the identifier, so what this asserts is that the arm
    selects them, that the staging transitions and the projection are recorded
    where the study says they are, and that the run stays bounded.  Whether the
    changes help is a study result, not a contract.
    """

    pytest.importorskip("crazyflow")

    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        DUAL_CONTROL_PASS3_MODEL,
        run_crazyflow_throw_study,
    )

    canonical = next(
        case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == "canonical"
    )
    started = time.perf_counter()
    report = run_crazyflow_throw_study((canonical,), (DUAL_CONTROL_PASS3_MODEL,))
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0

    metrics = report["cases"][0]["modes"][DUAL_CONTROL_PASS3_MODEL]
    dual = metrics["dual_control"]
    assert dual["config"]["variant"] == "pass2b"
    assert dual["identifier"] == {
        "staged_regressors": True,
        "staging_sample_multiple": 4.0,
        "enforce_collective_sign": True,
        "transition_aggregation_steps": 1,
    }
    assert metrics["flight"]["non_finite_value_count"] == 0
    assert metrics["flight"]["command_bound_violation_count"] == 0
    assert dual["unusable_command_count"] == 0
    staging = dual["staging"]
    assert staging["staged_regressors"]
    # Four samples per column on eight and eleven columns, at a hundred hertz.
    assert staging["collective_transition"]["step"] == 31
    assert staging["angular_transition"]["step"] == 43
    assert staging["angular_transition"]["time_from_enable_s"] == pytest.approx(0.44)
    assert staging["collective_sign_projection"]["enforced"]
    for moment in ("first_supported_model", "command_rank_four"):
        assert set(dual[moment]) == {
            "step",
            "time_s",
            "time_from_enable_s",
            "altitude_m",
            "descent_rate_m_s",
        }
    json.dumps(report, allow_nan=False)


# ----------------------------------------------------------------------
# fourth pass: the base action at zero information
# ----------------------------------------------------------------------


def _pass_four() -> DualControlNMPC:
    return DualControlNMPC(
        dual_control_config("pass4", sample_period_s=_SAMPLE_PERIOD_S)
    )


def _released_state() -> np.ndarray:
    """A tumbling, descending release of the kind the throw diagnostic hands over."""

    state = np.zeros(13)
    state[2] = 4.16
    state[3:6] = (1.0, -0.6, -3.25)
    state[6:10] = (0.9899, 0.0993, -0.0747, -0.0074)
    state[6:10] /= np.linalg.norm(state[6:10])
    state[10:13] = (0.8, -0.6, 0.4)
    return state


def _rate_cost(
    config: DualControlConfig,
    previous_command: np.ndarray,
    plan: np.ndarray,
    charge_first: bool,
) -> float:
    """The command-rate term recomputed from a finished plan, in numpy."""

    commands = np.repeat(plan, config.block_steps, axis=0)[: config.horizon_steps]
    moves = np.diff(np.concatenate((previous_command[None, :], commands)), axis=0)
    squared = np.square(moves)
    total = np.sum(squared) if charge_first else np.sum(squared[1:])
    return float(config.w_rate * total)


def test_the_first_transition_after_enable_is_free_and_a_later_one_is_not() -> None:
    """The rate cost is a slew cost on this controller's own consecutive actions.

    Both solves below run the same objective on the same state and posterior and
    differ only in whether the command the vehicle is carrying was issued by
    this controller.  The recomputed cost is asserted term for term rather than
    compared between the two runs, because the two runs return different plans:
    what is being tested is which moves the charge is summed over, not that one
    number is smaller than another.
    """

    controller = _pass_four()
    config = controller.config
    belief = _belief_after(0)
    state = _released_state()
    previous = np.zeros(4)

    unowned = controller.solve(state, belief, previous, previous_command_owned=False)
    assert not unowned.charged_initial_transition
    assert unowned.command_rate_cost == pytest.approx(
        _rate_cost(config, previous, unowned.plan, charge_first=False)
    )
    # The transition that was not charged is a real one, not a zero move.
    assert unowned.command_rate_cost != pytest.approx(
        _rate_cost(config, previous, unowned.plan, charge_first=True)
    )

    owned = controller.solve(state, belief, previous, previous_command_owned=True)
    assert owned.charged_initial_transition
    assert owned.command_rate_cost == pytest.approx(
        _rate_cost(config, previous, owned.plan, charge_first=True)
    )

    # Pass 2b charges the handover whatever the caller says, which is the
    # behaviour every earlier arm was measured under.
    legacy = _controller()
    for owned_flag in (False, True):
        result = legacy.solve(
            state, belief, previous, previous_command_owned=owned_flag
        )
        assert result.charged_initial_transition
        assert result.command_rate_cost == pytest.approx(
            _rate_cost(legacy.config, previous, result.plan, charge_first=True)
        )


def test_designs_are_midpoint_centered_until_the_posterior_supports_a_hover() -> None:
    """The declared center is the box midpoint, then the posterior's own hover.

    The midpoint is a statement about the command box: commands are normalized
    thrust fractions on ``[0, 1]`` and hover is somewhere inside, so with
    nothing known the midpoint minimizes the worst-case distance to it.  The
    handover condition is the identifier's own support rule, which is why a
    posterior with a hover command but without rank four is still centered on
    the midpoint.
    """

    controller = _pass_four()
    empty = _belief_after(0)
    center, source = controller.base_action(empty)
    assert source == "box_midpoint"
    assert center == pytest.approx(np.full(4, 0.5))

    informed = _belief_after(200)
    assert informed.command_evidence_rank == 4
    assert informed.angular_effect_rank == 3
    assert informed.hover_command is not None
    center, source = controller.base_action(informed)
    assert source == "hover_estimate"
    assert center == pytest.approx(np.asarray(informed.hover_command))

    # A posterior that has an unsupported hover estimate keeps the declaration.
    for count in (1, 2, 3):
        partial = _belief_after(count)
        if partial.hover_command is None:
            continue
        if partial.command_evidence_rank == 4 and partial.angular_effect_rank == 3:
            continue
        assert controller.base_action(partial)[1] == "box_midpoint"

    # And the center a solve actually used is recorded on its result.
    state = _released_state()
    zero_information = controller.solve(state, empty, np.zeros(4))
    assert zero_information.design_center_source == "box_midpoint"
    assert zero_information.design_center == pytest.approx(np.full(4, 0.5))
    supported = controller.solve(state, informed, np.zeros(4))
    assert supported.design_center_source == "hover_estimate"

    # Pass 2b centers on the previous command whatever the posterior says,
    # which at a motors-off release is the lower bound.
    legacy = _controller()
    for belief in (empty, informed):
        result = legacy.solve(state, belief, np.full(4, 0.3))
        assert result.design_center_source == "previous_command"
        assert result.design_center == pytest.approx(np.full(4, 0.3))


def test_the_base_action_leaves_the_released_zero_and_pass_two_b_does_not() -> None:
    """The two changes together move the first command off the released zero.

    This is the whole point of the pass, stated as the smallest behavioural
    claim that carries it: from the same zero-information posterior and the same
    motors-off release, pass 2b's first command is exactly the command it was
    released with and pass 4's is a substantial fraction of the command range.
    """

    state = _released_state()
    belief = _belief_after(0)
    legacy = _controller().solve(
        state, belief, np.zeros(4), previous_command_owned=False
    )
    fourth = _pass_four().solve(
        state, belief, np.zeros(4), previous_command_owned=False
    )
    assert float(np.mean(legacy.command)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.mean(fourth.command)) > 0.25


def test_every_pass_four_solve_stays_bounded_and_compiles_once() -> None:
    controller = _pass_four()
    assert controller.jit_cache_size == 0
    for count in (0, 1, 6, 40, 200):
        belief = _belief_after(count)
        result = None
        for index, state in enumerate(_random_states(4, seed=5)):
            result = controller.solve(
                state,
                belief,
                np.full(4, 0.45),
                result,
                previous_command_owned=index > 0,
            )
            assert np.all(np.isfinite(result.command))
            assert np.all(result.command >= -1e-9)
            assert np.all(result.command <= 1.0 + 1e-9)
            assert result.design_center_source in ("box_midpoint", "hover_estimate")
    assert controller.jit_cache_size == 1


def test_wilson_intervals_bracket_the_rate_and_stay_inside_the_unit_interval() -> None:
    from glassbox.integrations.crazyflow_throw_study import wilson_interval

    for successes, trials in ((0, 16), (1, 16), (8, 16), (15, 16), (16, 16)):
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= successes / trials <= high <= 1.0
        # A degenerate outcome still carries width, which is the reason this
        # interval is used instead of the normal approximation.
        assert high - low > 0.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    # Sixteen of sixteen is not certainty, and none of sixteen is not
    # impossibility.
    assert wilson_interval(16, 16)[0] < 1.0
    assert wilson_interval(0, 16)[1] > 0.0
    # The interval narrows with evidence.
    narrow = wilson_interval(8, 32)
    wide = wilson_interval(2, 8)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_the_ensemble_draws_come_from_the_declared_distribution() -> None:
    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        ThrowEnsembleConfig,
        build_ensemble_cases,
    )

    config = ThrowEnsembleConfig(replicate_count=8)
    base = CRAZYFLOW_THROW_STUDY_CASES[0]
    draws = build_ensemble_cases(base, 0, config)
    assert len(draws) == 8
    assert len({draw.name for draw in draws}) == 8
    for draw in draws:
        perturbation = draw.release_perturbation
        assert perturbation is not None
        assert np.all(np.asarray(perturbation.velocity_scale) >= 0.8)
        assert np.all(np.asarray(perturbation.velocity_scale) <= 1.2)
        assert np.all(np.asarray(perturbation.angular_velocity_scale) >= 0.8)
        assert np.all(np.asarray(perturbation.angular_velocity_scale) <= 1.2)
        rotation = np.asarray(perturbation.tilt_rotation_rad)
        assert float(np.linalg.norm(rotation)) <= 0.1 + 1e-12
        # The tilt perturbation is about a horizontal axis, so it never adds
        # yaw, and the release height is not perturbed at all.
        assert rotation[2] == 0.0
        assert draw.scenario.release_height_m == base.scenario.release_height_m
    # A draw is decided by the seed and the two indices, so it reproduces on its
    # own and does not move when other draws are added.
    assert config.draw(0, 3) == draws[3].release_perturbation
    assert config.draw(1, 3) != draws[3].release_perturbation
    assert (
        ThrowEnsembleConfig(replicate_count=32).draw(0, 3)
        == draws[3].release_perturbation
    )


@pytest.mark.crazyflow
def test_the_pass_four_canonical_smoke_stays_under_a_minute() -> None:
    pytest.importorskip("crazyflow")

    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        DUAL_CONTROL_PASS4_MODEL,
        run_crazyflow_throw_study,
    )

    canonical = next(
        case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == "canonical"
    )
    started = time.perf_counter()
    report = run_crazyflow_throw_study((canonical,), (DUAL_CONTROL_PASS4_MODEL,))
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0

    metrics = report["cases"][0]["modes"][DUAL_CONTROL_PASS4_MODEL]
    dual = metrics["dual_control"]
    assert dual["config"]["variant"] == "pass4"
    assert dual["config"]["charge_unowned_transition"] is False
    assert dual["config"]["center_designs_on_base_action"] is True
    assert metrics["flight"]["non_finite_value_count"] == 0
    assert metrics["flight"]["command_bound_violation_count"] == 0
    assert dual["unusable_command_count"] == 0
    # Exactly one interval per flight has a predecessor this controller did not
    # command: the first one after model enable.
    assert dual["base_action"]["uncharged_transition_count"] == 1
    assert dual["base_action"]["early_center_source"][0] == "box_midpoint"
    assert dual["base_action"]["first_center"] == [0.5, 0.5, 0.5, 0.5]
    assert len(dual["early_mean_collective_thirds"]) == 3
    assert 0.0 <= dual["early_mean_collective"] <= 1.0
    json.dumps(report, allow_nan=False)


@pytest.mark.crazyflow
def test_the_release_ensemble_is_deterministic_and_reports_wilson_intervals() -> None:
    pytest.importorskip("crazyflow")

    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        DUAL_CONTROL_PASS4_MODEL,
        ThrowEnsembleConfig,
        format_ensemble_table,
        run_crazyflow_throw_ensemble,
        wilson_interval,
    )

    canonical = next(
        case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == "canonical"
    )
    config = ThrowEnsembleConfig(replicate_count=2)
    arms = ("certified", DUAL_CONTROL_PASS4_MODEL)
    first = run_crazyflow_throw_ensemble((canonical,), arms, config)
    second = run_crazyflow_throw_ensemble((canonical,), arms, config)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    assert first["artifact_type"] == "glassbox_crazyflow_throw_release_ensemble"
    assert first["releases_per_arm"] == 2
    assert first["trial_count"] == 4
    assert first["ensemble"]["seed"] == config.seed
    entry = first["cases"][0]
    assert len(entry["releases"]) == 2
    for model in arms:
        summary = entry["arms"][model]
        assert summary["trial_count"] == 2
        assert summary["recovery_rate"] == summary["recovery_count"] / 2
        assert summary["recovery_rate_wilson_95"] == list(
            wilson_interval(summary["recovery_count"], 2)
        )
        low, high = summary["recovery_rate_wilson_95"]
        assert 0.0 <= low <= summary["recovery_rate"] <= high <= 1.0
        assert summary["all_values_finite_and_bounded"]
        assert 0.0 <= summary["early_mean_collective"]["mean"] <= 1.0
        pooled = first["pooled"][model]
        assert pooled["trial_count"] == 2
    # Every arm flies the same draws, so the comparison is paired.
    flown = {(trial["case"], trial["control_model"]) for trial in first["trials"]}
    assert flown == {
        (f"canonical#{replicate:02d}", model)
        for replicate in range(2)
        for model in arms
    }
    table = format_ensemble_table(first)
    assert table.splitlines()[0].startswith("| case")
    # One row per case and arm, plus the pooled rows, plus the two header rows.
    assert len(table.splitlines()) == 2 + len(arms) * 2
    json.dumps(first, allow_nan=False)


@pytest.mark.crazyflow
def test_the_cascade_arms_reproduce_the_archived_single_release_report() -> None:
    """Nothing in this pass touches the two cascade arms, asserted not claimed.

    The archive lives outside the repository, so this skips when it is absent
    rather than pinning a four-megabyte artifact into the test data.
    """

    pytest.importorskip("crazyflow")

    from pathlib import Path

    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        run_crazyflow_throw_study,
    )

    archive_path = Path("artifacts/crazyflow_throw_study/report-pass3.json")
    if not archive_path.is_file():
        pytest.skip("the archived third-pass report is not present")
    archive = json.loads(archive_path.read_text())
    archived = {entry["case"]["name"]: entry for entry in archive["cases"]}

    canonical = next(
        case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == "canonical"
    )
    report = run_crazyflow_throw_study((canonical,), ("certified", "working"))
    fresh = report["cases"][0]
    want = archived["canonical"]
    assert json.dumps(fresh["case"], sort_keys=True) == json.dumps(
        want["case"], sort_keys=True
    )

    def on_archived_keys(fresh_value: object, archived_value: object) -> object:
        """Restrict the fresh report to the keys the archive carries.

        The stop-at-floor-contact rule added ``floor_contact_time_s`` to every
        flight section after the archive was written; a canonical release that
        never touches the floor is otherwise unchanged, and that is what this
        asserts.
        """

        if isinstance(fresh_value, dict) and isinstance(archived_value, dict):
            return {
                key: on_archived_keys(fresh_value[key], archived_value[key])
                for key in archived_value
                if key in fresh_value
            }
        return fresh_value

    for model in ("certified", "working"):
        assert json.dumps(
            on_archived_keys(fresh["modes"][model], want["modes"][model]),
            sort_keys=True,
        ) == json.dumps(want["modes"][model], sort_keys=True), model
    assert json.dumps(
        on_archived_keys(
            fresh["difference_working_minus_certified"],
            want["difference_working_minus_certified"],
        ),
        sort_keys=True,
    ) == json.dumps(want["difference_working_minus_certified"], sort_keys=True)


# ----------------------------------------------------------------------
# pass five: one goal, slew-bounded moves, and posterior-derived seeds
# ----------------------------------------------------------------------


def _pass_five(**overrides: object) -> DualControlNMPC:
    return DualControlNMPC(
        dual_control_config("pass5", sample_period_s=_SAMPLE_PERIOD_S, **overrides)
    )


def _synthetic_belief(
    *,
    angular_per_command: np.ndarray,
    collective_per_command: np.ndarray,
) -> RecursiveBootstrapBelief:
    """A belief with the two command maps set and nothing else supported.

    The seeds under test read exactly these two arrays, so setting them
    directly is what makes the assertion about the seed rather than about
    whatever the identifier happened to fit.
    """

    empty = RecursiveBootstrapIdentifier().belief
    return replace(
        empty,
        angular_acceleration_per_command=angular_per_command,
        collective_acceleration_per_command=collective_per_command,
    )


def test_pass_five_declares_its_switches_and_refuses_a_malformed_slew() -> None:
    config = dual_control_config("pass5")

    assert config.variant == "pass5"
    assert config.horizon_steps == 100
    assert config.block_steps == 5
    assert config.block_count == 20
    assert config.plan_parameterization == "slew_moves"
    assert config.spread_model == "planned_trajectory"
    assert config.seed_family == "posterior_moves"
    assert config.charge_body_rate_limit is True
    assert config.charge_unowned_transition is False
    assert config.center_designs_on_base_action is False
    assert config.candidate_count == 8
    recorded = config.to_dict()
    for key in (
        "plan_parameterization",
        "spread_model",
        "seed_family",
        "charge_body_rate_limit",
        "slew_per_interval",
        "maximum_body_rate_rad_s",
    ):
        assert key in recorded, key

    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="slew_per_interval"):
            DualControlConfig(slew_per_interval=bad)
    with pytest.raises(ValueError, match="plan_parameterization"):
        DualControlConfig(plan_parameterization="absolute")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spread_model"):
        DualControlConfig(spread_model="box")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed_family"):
        DualControlConfig(seed_family="hadamard")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        DualControlConfig(maximum_body_rate_rad_s=0.0)


def test_every_pass_five_candidate_and_command_respects_the_declared_slew() -> None:
    """The slew limit is a box on the decision variables, not a preference.

    Every seed and every refined plan is checked block by block, and the
    executed command is checked against the command the vehicle was holding.
    The seeds are checked at the JAX default precision the solve runs in; the
    executed command is enforced in double precision by the solver itself, so
    it is checked exactly.
    """

    controller = _pass_five()
    step = controller.config.slew_per_interval
    beliefs = [_belief_after(count) for count in (0, 4, 60)]
    states = _random_states(4, seed=17)
    for belief in beliefs:
        posterior = controller._posterior(belief)
        for state in states:
            for held in (np.zeros(4), np.full(4, 0.35), np.full(4, 1.0)):
                warm, _ = controller._warm_blocks(None, held)
                candidates = controller._candidate_blocks(
                    jnp.asarray(warm),
                    jnp.asarray(held),
                    jnp.asarray(held),
                    jnp.asarray(state),
                    posterior,
                )
                for blocks in candidates:
                    plan = np.asarray(
                        controller._plan_command_blocks(blocks, jnp.asarray(held))
                    )
                    moves = np.diff(np.concatenate((held[None, :], plan)), axis=0)
                    assert np.max(np.abs(moves)) <= step + 1e-6
                result = controller.solve(state, belief, held)
                assert np.all(np.abs(result.command - held) <= step + 1e-9)
                assert np.all(result.command >= 0.0)
                assert np.all(result.command <= 1.0)


def _assert_directions_are_read_off_the_posterior(
    controller: DualControlNMPC,
) -> None:
    """The excitation basis follows the evidence, weakest-known direction first.

    At zero information any orthonormal basis is as good as any other, so the
    identity there proves nothing on its own.  This checks the informed case:
    the basis is no longer the identity, it is still orthogonal, and its rows
    are ordered by how little the accumulated command evidence says about them.
    """

    posterior = controller._posterior(_belief_after(60))
    directions = np.asarray(controller._excitation_directions(posterior))
    assert not np.allclose(np.abs(directions), np.eye(4), atol=1e-3)
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    assert np.allclose(unit @ unit.T, np.eye(4), atol=1e-4)
    gram = np.asarray(posterior.collective_information)[:4, :4]
    known = np.asarray([row @ gram @ row for row in unit])
    assert np.all(np.diff(known) >= -1e-6), known
    assert known[-1] > known[0]


def test_zero_information_seeds_span_the_box_but_the_objective_still_holds() -> None:
    """A measured negative result, pinned so it cannot be lost.

    At zero information the excitation directions are the four commands, one at
    a time — the posterior supplies the symmetry break the declared ladder used
    to declare — and the four excitation seeds between them move every command
    in both directions.  The objective nevertheless prefers ``hold``: the
    planned-trajectory spread is evaluated at the plan's *own* features, and a
    plan that visits one command direction knows that direction better than a
    plan that spreads the same horizon over four.  Standing still is therefore
    the cheapest plan the spread charge can see, and the fifth pass reproduces
    the first pass's fixed point by a different route.
    """

    controller = _pass_five()
    belief = _belief_after(0)
    posterior = controller._posterior(belief)
    state = jnp.asarray(_released_state())

    directions = np.asarray(controller._excitation_directions(posterior))
    assert np.allclose(np.abs(directions), np.eye(4), atol=1e-6)

    held = np.full(4, 0.4)
    _assert_directions_are_read_off_the_posterior(controller)
    warm, _ = controller._warm_blocks(None, held)
    candidates = controller._candidate_blocks(
        jnp.asarray(warm),
        jnp.asarray(held),
        jnp.asarray(held),
        state,
        posterior,
    )
    names = controller.candidate_names
    excitation = {
        name: np.asarray(controller._plan_command_blocks(blocks, jnp.asarray(held)))
        for name, blocks in zip(names, candidates)
        if name.startswith("excite")
    }
    assert len(excitation) == 4
    stacked = np.concatenate(list(excitation.values()))
    assert np.linalg.matrix_rank(stacked - held, tol=1e-6) == 4
    assert np.max(stacked - held) > 0.09
    assert np.min(stacked - held) < -0.09

    values = np.asarray(
        jax.vmap(
            lambda candidate: controller._objective(
                candidate,
                state,
                posterior,
                jnp.asarray(held),
                jnp.asarray(0.0),
            )
        )(candidates)
    )
    ranked = dict(zip(names, values))
    assert min(ranked[name] for name in excitation) > ranked["hold"]
    assert (
        controller.solve(np.asarray(state), belief, held).selected_candidate == "hold"
    )


def test_the_coupled_spread_grows_the_velocity_channel_with_thrust() -> None:
    """Thrusting under an uncertain attitude is charged for sideways velocity.

    The two rollouts below carry the same attitude spread by construction — the
    angular regressors and the angular posterior are identical — and differ only
    in the magnitude of the predicted specific force.  Only the coupled term
    ``|f| sigma_tilt`` can move the velocity spread between them.
    """

    controller = _pass_five()
    posterior = controller._posterior(_belief_after(30))
    steps = controller.config.horizon_steps
    rollout = _fixed_thrust_rollout(controller, steps, specific_force=0.0)
    quiet = controller._spreads_trajectory(rollout, posterior)
    loud = controller._spreads_trajectory(
        _fixed_thrust_rollout(controller, steps, specific_force=12.0),
        posterior,
    )
    assert np.allclose(np.asarray(quiet[0]), np.asarray(loud[0]))
    assert np.allclose(np.asarray(quiet[1]), np.asarray(loud[1]))
    assert float(loud[2][-1]) > float(quiet[2][-1])
    assert float(loud[3][-1]) > float(quiet[3][-1])
    tilt = np.asarray(quiet[1])
    expected = float(np.sum(_SAMPLE_PERIOD_S * 12.0 * tilt))
    assert float(loud[2][-1] - quiet[2][-1]) == pytest.approx(expected, rel=1e-3)


def _fixed_thrust_rollout(
    controller: DualControlNMPC,
    steps: int,
    *,
    specific_force: float,
) -> object:
    """One synthetic rollout with every regressor fixed but the thrust."""

    zeros = jnp.zeros((steps, 3))
    return _Rollout(
        commands=jnp.full((steps, 4), 0.4),
        tracking=jnp.zeros(steps),
        tilt=jnp.zeros(steps),
        altitude=jnp.full((steps,), 2.0),
        angular_velocity=zeros,
        rate_products=zeros,
        body_velocity=zeros,
        specific_force=jnp.full((steps,), specific_force),
        rate_norm=jnp.zeros(steps),
        tracking_terms=jnp.zeros((steps, 4)),
    )


def test_the_body_rate_penalty_fires_on_a_fast_spin_and_not_at_rest() -> None:
    """The declared rate limit is charged, and only pass five declares one."""

    controller = _pass_five()
    posterior = controller._posterior(_belief_after(60))
    blocks = jnp.zeros((controller.block_count, 4))
    previous = jnp.full(4, 0.3)

    calm = _released_state()
    calm[10:13] = (0.05, -0.05, 0.02)
    spinning = calm.copy()
    spinning[10:13] = (26.0, -18.0, 9.0)

    calm_terms = controller._terms(
        blocks, jnp.asarray(calm), posterior, previous, jnp.asarray(1.0)
    )
    spinning_terms = controller._terms(
        blocks, jnp.asarray(spinning), posterior, previous, jnp.asarray(1.0)
    )
    assert float(calm_terms.body_rate_penalty) == 0.0
    assert float(spinning_terms.body_rate_penalty) > 0.0

    quiet = DualControlNMPC(
        dual_control_config("pass4", sample_period_s=_SAMPLE_PERIOD_S)
    )
    quiet_terms = quiet._terms(
        jnp.zeros((quiet.block_count, 4)),
        jnp.asarray(spinning),
        quiet._posterior(_belief_after(60)),
        previous,
        jnp.asarray(1.0),
    )
    assert float(quiet_terms.body_rate_penalty) == 0.0


def test_the_righting_seed_accelerates_towards_level_under_a_known_map() -> None:
    """The righting seed is read off the posterior, not declared.

    With a known angular map the seed's first move must produce an angular
    acceleration that points the way the vehicle has to turn: positively along
    the levelling-plus-damping direction the seed was built from, and against
    the tilt the state carries.  With no map at all it is the held command.
    """

    controller = _pass_five()
    angular = np.asarray(
        (
            (-90.0, -88.0, 92.0, 90.0),
            (86.0, -90.0, -88.0, 92.0),
            (-24.0, 26.0, -25.0, 23.0),
        )
    )
    belief = _synthetic_belief(
        angular_per_command=angular,
        collective_per_command=np.asarray((4.6, 4.7, 4.5, 4.6)),
    )
    posterior = controller._posterior(belief)
    state = _released_state()
    state[10:13] = (0.0, 0.0, 0.0)
    move = np.asarray(controller._righting_move(jnp.asarray(state), posterior))
    assert np.max(np.abs(move)) == pytest.approx(1.0, abs=1e-5)

    rotation = quaternion_to_rotation(state[6:10])
    up_body = rotation[2, :]
    desired = np.asarray((-up_body[1], up_body[0], 0.0)) - state[10:13]
    desired /= np.linalg.norm(desired)
    acceleration = angular @ (move * controller.config.slew_per_interval)
    assert float(desired @ acceleration) > 0.0
    # The tilt error itself must shrink: the roll and pitch rates the move
    # produces oppose the tilt the state carries.
    assert float(np.asarray((up_body[0], up_body[1])) @ acceleration[:2]) < 0.0

    blind = _synthetic_belief(
        angular_per_command=np.zeros((3, 4)),
        collective_per_command=np.zeros(4),
    )
    blind_move = np.asarray(
        controller._righting_move(jnp.asarray(state), controller._posterior(blind))
    )
    assert np.allclose(blind_move, 0.0)


def test_the_collective_seed_raises_the_posterior_specific_force() -> None:
    controller = _pass_five()
    collective = np.asarray((4.6, 4.7, 4.5, 2.3))
    belief = _synthetic_belief(
        angular_per_command=np.zeros((3, 4)),
        collective_per_command=collective,
    )
    move = np.asarray(controller._collective_move(controller._posterior(belief)))
    assert np.max(np.abs(move)) == pytest.approx(1.0, abs=1e-5)
    assert float(collective @ move) > 0.0
    # The weakest motor is asked for the least, because the posterior says it
    # buys the least specific force.
    assert int(np.argmin(move)) == 3

    blind = _synthetic_belief(
        angular_per_command=np.zeros((3, 4)),
        collective_per_command=np.zeros(4),
    )
    assert np.allclose(
        np.asarray(controller._collective_move(controller._posterior(blind))),
        0.0,
    )


def test_the_warm_start_moves_round_trip_through_the_shift() -> None:
    """The shifted warm start replays the plan the previous solve committed to.

    The vehicle is holding the plan's first block, so the shifted plan's own
    commands must be the previous plan's second block onwards, with its last
    block repeated.  The seed is stored as moves; reconstructing the commands
    from them is what proves the conversion is lossless.
    """

    controller = _pass_five()
    slew = controller.config.slew_per_interval
    generator = np.random.default_rng(5)
    # A plan a previous solve could actually have produced: a cumulative sum of
    # moves inside the declared slew box, starting inside the command box.
    steps = generator.uniform(
        -slew,
        slew,
        (controller.block_count, 4),
    )
    plan = np.clip(np.cumsum(steps, axis=0) + 0.45, 0.0, 1.0)
    held = plan[0]
    moves, valid = controller._warm_blocks(plan, held)
    assert valid
    assert np.max(np.abs(moves)) <= 1.0
    rebuilt = np.asarray(
        controller._plan_command_blocks(jnp.asarray(moves), jnp.asarray(held))
    )
    expected = np.concatenate((plan[1:], plan[-1:]))
    assert np.allclose(rebuilt, expected, atol=1e-6)
    assert np.allclose(rebuilt[-1], rebuilt[-2])

    # A plan whose blocks are further apart than the declared slew comes back
    # clipped rather than refused: the seed is a starting point, not a promise.
    wide = np.tile(np.asarray(((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))), (10, 1))
    clipped, wide_valid = controller._warm_blocks(wide, wide[0])
    assert wide_valid
    assert np.max(np.abs(clipped)) == 1.0
    # Every move but the repeated final block saturates the slew box.
    assert np.allclose(np.abs(clipped[:-1]), 1.0)
    assert np.allclose(clipped[-1], 0.0)

    cold, cold_valid = controller._warm_blocks(None, held)
    assert not cold_valid
    assert np.allclose(cold, 0.0)


def test_the_earlier_variants_are_unchanged_by_the_fifth_pass() -> None:
    """The four earlier variants are pinned, not merely believed unchanged.

    These numbers were recorded from the pass-four tree before any of the fifth
    pass landed and reproduce bit for bit after it; the recorded study report
    reproduces alongside them.  Every switch the fifth pass adds is stated on
    each earlier variant at the value it already behaved as.
    """

    state = _released_state()
    belief = _belief_after(40)
    previous = np.full(4, 0.25)
    expected = {
        "pass2b": (
            (
                0.5503792762756348,
                0.4479737877845764,
                0.39375966787338257,
                0.43878284096717834,
            ),
            52114.84375,
        ),
        "pass4": (
            (
                0.8636729717254639,
                0.7502814531326294,
                0.6340652704238892,
                0.683287501335144,
            ),
            34851.47265625,
        ),
    }
    for variant, (command, objective) in expected.items():
        config = dual_control_config(variant, sample_period_s=_SAMPLE_PERIOD_S)
        assert config.variant == variant
        assert config.plan_parameterization == "command_blocks"
        assert config.spread_model == "command_marginal"
        assert config.seed_family == "declared_designs"
        assert config.charge_body_rate_limit is False
        result = DualControlNMPC(config).solve(
            state, belief, previous, previous_command_owned=True
        )
        assert result.command.tolist() == list(command), variant
        assert float(result.objective_value) == objective, variant
        assert float(result.body_rate_penalty) == 0.0, variant


def test_every_pass_five_solve_stays_bounded_and_compiles_once() -> None:
    controller = _pass_five()
    plan = None
    previous = np.full(4, 0.2)
    for index, state in enumerate(_random_states(5, seed=23)):
        for belief in (_belief_after(0), _belief_after(9), _belief_after(80)):
            result = controller.solve(
                state,
                belief,
                previous,
                warm_start=plan,
                previous_command_owned=index > 0,
            )
            assert result.command_usable
            assert np.all(np.isfinite(result.command))
            assert np.all(result.command >= 0.0)
            assert np.all(result.command <= 1.0)
            assert result.design_center_source == "previous_command"
            assert np.allclose(result.design_center, previous)
            assert result.selected_candidate in controller.candidate_names
            plan = result
            previous = np.asarray(result.command)
    assert controller.jit_cache_size == 1


def test_a_diverged_release_is_recorded_rather_than_ending_the_ensemble() -> None:
    """A simulator that cannot integrate one release must not lose the other 559.

    The plant refuses a non-finite state rather than returning one, which is
    right, and the ensemble turns that refusal into a trial: not recovered,
    every measured quantity absent rather than invented, and the arm's own
    finiteness criterion failed so the divergence cannot be read as a clean run.
    """

    from glassbox.integrations.crazyflow_throw_study import (
        CRAZYFLOW_THROW_STUDY_CASES,
        _diverged_ensemble_trial,
        _ensemble_summary,
    )

    case = CRAZYFLOW_THROW_STUDY_CASES[0]
    flown = {
        "case": case.name,
        "control_model": "arm",
        "simulator_diverged": False,
        "recovered": True,
        "reached_hover_envelope": True,
        "touched_floor": False,
        "terminal_speed_m_s": 0.10,
        "terminal_angular_rate_rad_s": 0.10,
        "terminal_tilt_rad": 0.01,
        "minimum_altitude_m": 1.0,
        "sustained_hover_duration_s": 3.0,
        "time_to_rank_four_s": 0.30,
        "early_mean_collective": 0.50,
        "settled_maximum_command_step": 0.01,
        "settled_maximum_relative_allocation_change": 0.01,
        "settled_maximum_allocation_change": 0.001,
        "non_finite_value_count": 0,
        "command_bound_violation_count": 0,
    }
    diverged = _diverged_ensemble_trial(case, "arm", "velocity must be finite")
    assert set(diverged) >= set(flown)
    assert diverged["simulator_diverged"] is True
    assert diverged["recovered"] is False
    assert diverged["early_mean_collective"] is None

    summary = _ensemble_summary([flown, diverged])
    assert summary["trial_count"] == 2
    assert summary["recovery_count"] == 1
    assert summary["simulator_diverged_count"] == 1
    assert summary["all_values_finite_and_bounded"] is False
    # The diverged release invents no number: every spread is built from the
    # one trial that actually flew.
    for key in (
        "terminal_speed_m_s",
        "time_to_rank_four_s",
        "settled_maximum_command_step",
    ):
        assert summary[key]["available_count"] == 1
    assert summary["early_mean_collective"]["available_count"] == 1
    assert summary["early_mean_collective"]["mean"] == 0.50
    json.dumps(summary, allow_nan=False)

    clean = _ensemble_summary([flown])
    assert clean["simulator_diverged_count"] == 0
    assert clean["all_values_finite_and_bounded"] is True


# ----------------------------------------------------------------------
# pass six
# ----------------------------------------------------------------------


def _pass_six(**overrides: object) -> DualControlNMPC:
    return DualControlNMPC(
        DualControlConfig(**(dict(DUAL_CONTROL_VARIANTS["pass6"]) | overrides))
    )


def test_pass_six_declares_its_switches_and_refuses_malformed_ones() -> None:
    config = dual_control_config("pass6")
    assert config.variant == "pass6"
    assert config.spread_model == "command_marginal_full"
    assert config.goal_horizon == "posterior"
    assert config.excitation_basis == "hadamard"
    assert config.information_neighbourhood == "hover"
    assert config.horizon_neighbourhood == "box_commands"
    assert config.authority_scaled_maps is True
    assert config.warm_start_shift == "step"
    # Measured and rejected; kept as switches, off.
    assert config.rate_limit_on_mean is False
    assert config.goal_seed_authority is False
    assert config.face_value_at_full_rank is False
    assert config.probe_until_supported is False
    assert config.horizon_steps == 100 and config.block_steps == 5
    assert dual_control_config("pass5").excitation_basis == "motor"
    for field, value in (
        ("goal_horizon", "always"),
        ("excitation_basis", "random"),
        ("information_neighbourhood", "everywhere"),
        ("spread_model", "box"),
    ):
        with pytest.raises(ValueError):
            DualControlConfig(**(dict(DUAL_CONTROL_VARIANTS["pass6"]) | {field: value}))


def test_pass_six_probes_the_collective_first_at_zero_information() -> None:
    """In the Hadamard basis the degenerate case is collective first.

    The eigenvectors of a multiple of the identity are whatever basis the
    decomposition is taken in, so at zero information the sixth pass's first
    excitation row moves all four motors together and the other three are
    zero-sum patterns; the fifth pass, decomposing in the motor basis, probes
    the four motors one at a time.  Once the posterior has learned anything
    the rows are its own weakest directions in either basis.
    """

    belief = _belief_after(0)
    six = _pass_six()
    rows = np.asarray(six._excitation_directions(six._posterior(belief)))
    assert np.allclose(np.abs(rows[0]), np.ones(4), atol=1e-6)
    assert np.allclose(np.sum(rows[1:], axis=1), 0.0, atol=1e-6)
    assert np.linalg.matrix_rank(rows, tol=1e-6) == 4
    five = _pass_five()
    motor_rows = np.asarray(five._excitation_directions(five._posterior(belief)))
    assert np.allclose(np.abs(motor_rows), np.eye(4), atol=1e-6)

    learned = _belief_after(60)
    six_learned = np.asarray(six._excitation_directions(six._posterior(learned)))
    five_learned = np.asarray(five._excitation_directions(five._posterior(learned)))
    # Same subspace ordering, up to sign: the basis is immaterial once learned.
    for a, b in zip(six_learned, five_learned):
        assert np.isclose(
            abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)), 1.0, atol=1e-5
        )


def test_pass_six_seeds_include_one_descent_plan_per_goal_term() -> None:
    controller = _pass_six()
    assert controller.config.candidate_count == 2 + 14
    names = controller.candidate_names
    assert names[-8:-4] == ("goal_velocity", "goal_rate", "goal_tilt", "goal_floor")
    assert all(name.endswith("+excite") for name in names[-4:])
    belief = _belief_after(60)
    posterior = controller._posterior(belief)
    held = jnp.full(4, 0.5)
    goals = np.asarray(
        controller._goal_moves(jnp.asarray(_released_state()), posterior, held)
    )
    assert goals.shape == (4, controller.block_count, 4)
    # Every goal seed is a slew-scaled descent plan: bounded by one in the
    # normalized move units, and non-trivial once the maps are learned.
    assert np.max(np.abs(goals)) <= 1.0 + 1e-9
    assert np.max(np.abs(goals)) > 0.5


def test_pass_six_solves_from_zero_information_within_the_slew() -> None:
    controller = _pass_six()
    belief = _belief_after(0)
    held = np.zeros(4)
    result = controller.solve(
        _released_state(), belief, held, previous_command_owned=False
    )
    assert result.command_usable
    assert np.all(
        np.abs(result.command - held) <= controller.config.slew_per_interval + 1e-9
    )
    assert np.all(result.command >= 0.0) and np.all(result.command <= 1.0)
    assert result.selected_candidate != "hold"
    assert np.max(result.command) > 0.05


def test_the_hover_neighbourhood_is_the_box_until_the_maps_pin_hover() -> None:
    """The knowledge term's neighbourhood is derived, never declared.

    At zero information the hover solution carries no information and the
    neighbourhood is the box prior: zero mean, ``1/12`` variance per axis.
    Once the maps are learned the mean is a command the belief's own collective
    map says produces hover thrust, and the spread around it is far tighter
    than the box.
    """

    controller = _pass_six()
    posterior = controller._posterior(_belief_after(0))
    rest_c = jnp.concatenate((jnp.zeros(3), jnp.ones(1)))
    rest_a = jnp.concatenate((jnp.zeros(6), jnp.ones(1)))
    controller._jit_held_feature = jnp.zeros(4)
    mean, moment = controller._hover_command_moment(posterior, rest_c, rest_a)
    assert np.allclose(np.asarray(mean), 0.0, atol=1e-9)
    assert np.allclose(np.asarray(moment), np.eye(4) / 12.0, atol=1e-9)

    belief = _belief_after(200)
    posterior = controller._posterior(belief)
    mean, moment = controller._hover_command_moment(posterior, rest_c, rest_a)
    command = np.asarray(controller._jit_midpoint + controller._jit_span * mean)
    predicted = float(
        belief.collective_acceleration_per_command @ command
        + belief.collective_intercept_m_s2
    )
    assert abs(predicted - GRAVITY_M_S2) < 0.1 * GRAVITY_M_S2
    covariance = np.asarray(moment) - np.outer(np.asarray(mean), np.asarray(mean))
    assert np.trace(covariance) < 0.1 * (4.0 / 12.0)


def test_the_step_warm_start_advances_the_plan_in_real_time() -> None:
    """A block plan is kept for ``block_steps`` intervals, then shifted.

    Under ``"block"`` every solve drops the first block, so a plan of
    five-step blocks is executed five times faster than it was planned.  Under
    ``"step"`` the phase rides on the result: the block is kept until it has
    been executed in full, and only then does the next block become first.
    """

    from types import SimpleNamespace

    controller = _pass_six()
    assert controller.config.warm_start_shift == "step"
    assert controller._warm_phase(None) == (True, 0)
    previous = SimpleNamespace(plan_phase=0)
    for phase in range(1, controller.config.block_steps):
        assert controller._warm_phase(previous) == (False, phase)
        previous = SimpleNamespace(plan_phase=phase)
    assert controller._warm_phase(previous) == (True, 0)

    blocked = _pass_six(warm_start_shift="block")
    assert blocked._warm_phase(SimpleNamespace(plan_phase=0)) == (True, 0)

    held = np.full(4, 0.4)
    plan = np.linspace(0.3, 0.6, controller.block_count)[:, None] * np.ones((1, 4))
    kept, valid = controller._warm_blocks(SimpleNamespace(plan=plan), held, shift=False)
    shifted, _ = controller._warm_blocks(SimpleNamespace(plan=plan), held, shift=True)
    assert valid
    kept_commands = np.asarray(
        controller._plan_command_blocks(jnp.asarray(kept), jnp.asarray(held))
    )
    shifted_commands = np.asarray(
        controller._plan_command_blocks(jnp.asarray(shifted), jnp.asarray(held))
    )
    assert np.allclose(kept_commands[0], plan[0], atol=1e-9)
    assert np.allclose(shifted_commands[0], plan[1], atol=1e-9)


def test_explicit_block_lengths_expand_bound_and_shift_exactly() -> None:
    """Short blocks first, long blocks last, and a real-time warm start.

    The plan expands to exactly the horizon with each block held for its own
    length, a block may move by one slew per step it lasts so the executed
    command still moves at most one slew per interval, and the step shift
    re-blocks the previous plan one step later: the first block becomes the
    second, exactly, because the leading blocks are one step long.
    """

    from types import SimpleNamespace

    controller = _pass_six(block_lengths=(1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 44))
    config = controller.config
    assert config.block_lengths is not None
    assert sum(config.block_lengths) == config.horizon_steps
    assert config.block_count == len(config.block_lengths)
    blocks = jnp.asarray(np.arange(config.block_count, dtype=np.float64))[
        :, None
    ] * jnp.ones((1, 4))
    expanded = np.asarray(controller._expand(blocks))
    assert expanded.shape == (config.horizon_steps, 4)
    starts = np.concatenate(([0], np.cumsum(config.block_lengths)[:-1]))
    assert np.allclose(expanded[starts, 0], np.arange(config.block_count))
    assert np.allclose(expanded[-1], config.block_count - 1)

    held = jnp.full(4, 0.5)
    full = jnp.ones((config.block_count, 4))
    commands = np.asarray(controller._plan_command_blocks(full, held))
    moves = np.diff(np.concatenate((np.full((1, 4), 0.5), commands)), axis=0)
    expected = np.minimum(
        config.slew_per_interval,
        np.maximum(1.0 - np.concatenate((np.full((1, 4), 0.5), commands))[:-1], 0.0),
    )
    assert np.allclose(moves, expected, atol=1e-9)

    plan = np.linspace(0.3, 0.6, config.block_count)[:, None] * np.ones((1, 4))
    shifted, valid = controller._warm_blocks(
        SimpleNamespace(plan=plan), np.full(4, 0.3)
    )
    assert valid
    shifted_commands = np.asarray(
        controller._plan_command_blocks(jnp.asarray(shifted), jnp.full(4, 0.3))
    )
    assert np.allclose(shifted_commands[:3], plan[1:4], atol=1e-9)
    with pytest.raises(ValueError):
        _pass_six(block_lengths=(1, 2, 3))


def test_authority_scaled_maps_use_the_posterior_at_the_identifier_trust() -> None:
    """A map the identifier has not earned is not acted on at face value.

    Each angular axis is scaled by its own authority and the collective map by
    the collective authority, so at zero authority the mean rollout sees no
    command effect at all, and at full authority it sees the fitted maps.
    """

    from dataclasses import replace

    belief = _belief_after(60)
    # The scaling itself, without the full-rank face-value rule on top.
    scaled = _pass_six(face_value_at_full_rank=False)
    plain = _pass_six(authority_scaled_maps=False, face_value_at_full_rank=False)
    full = replace(
        belief,
        collective_authority=1.0,
        angular_axis_authority=np.ones(3),
    )
    none = replace(
        belief,
        collective_authority=0.0,
        angular_axis_authority=np.zeros(3),
    )
    assert np.allclose(
        np.asarray(scaled._posterior(full).angular_per_command),
        np.asarray(plain._posterior(full).angular_per_command),
    )
    assert np.allclose(np.asarray(scaled._posterior(none).angular_per_command), 0.0)
    assert np.allclose(np.asarray(scaled._posterior(none).collective_per_command), 0.0)
    half = replace(
        belief,
        collective_authority=0.5,
        angular_axis_authority=np.asarray((1.0, 0.5, 0.0)),
    )
    rows = np.asarray(scaled._posterior(half).angular_per_command)
    reference = np.asarray(plain._posterior(half).angular_per_command)
    assert np.allclose(rows[0], reference[0])
    assert np.allclose(rows[1], 0.5 * reference[1])
    assert np.allclose(rows[2], 0.0)


def test_the_probe_overlay_rides_on_the_executed_command_until_supported() -> None:
    """A weakest-direction probe at one slew, held per block, until rank four.

    With the identifier's support incomplete the executed command carries the
    probe, the executed move still stays within one slew of the held command,
    and the probe's sign is held within a block and flipped at the next.  With
    the support complete there is no probe and the command is the plan's own.
    """

    from types import SimpleNamespace

    controller = _pass_six(probe_until_supported=True)
    plain = _pass_six()
    assert plain.config.probe_until_supported is False
    held = np.full(4, 0.4)
    state = _released_state()

    incomplete = _belief_after(0)
    with_probe = controller.solve(state, incomplete, held, previous_command_owned=False)
    without = plain.solve(state, incomplete, held, previous_command_owned=False)
    assert with_probe.command_usable and without.command_usable
    assert not np.allclose(with_probe.command, without.command)
    assert np.all(
        np.abs(with_probe.command - held) <= controller.config.slew_per_interval + 1e-9
    )
    assert with_probe.probe_sign == 1.0
    # Within the same block the sign holds; a new block flips it.
    same_block = controller.solve(
        state, incomplete, with_probe.command, warm_start=with_probe
    )
    assert same_block.probe_sign == 1.0
    last = SimpleNamespace(
        plan=with_probe.plan,
        plan_phase=controller.config.block_steps - 1,
        probe_sign=1.0,
    )
    next_block = controller.solve(
        state, incomplete, with_probe.command, warm_start=last
    )
    assert next_block.probe_sign == -1.0

    complete = _belief_after(200)
    assert complete.command_evidence_rank == 4 and complete.angular_effect_rank == 3
    probed = controller.solve(state, complete, held, previous_command_owned=False)
    bare = plain.solve(state, complete, held, previous_command_owned=False)
    assert np.allclose(probed.command, bare.command)


def test_the_command_only_horizon_ignores_nuisance_uncertainty_at_high_rates() -> None:
    """The goal horizon can be decided by the command maps alone.

    With ``"box"`` the spread that decides how far the goal is charged includes
    the fitted damping and coupling coefficients evaluated at the measured
    rates, so a fast tumble saturates the rate channel at once.  With
    ``"box_commands"`` only the command block's box-averaged uncertainty
    enters, and the spread at a fast tumble equals the spread at rest.
    """

    box = _pass_six(horizon_neighbourhood="box")
    commands = _pass_six(horizon_neighbourhood="box_commands")
    belief = _belief_after(60)
    posterior = box._posterior(belief)
    held = jnp.full(4, 0.5)
    calm = _released_state()
    calm[10:13] = (0.05, -0.05, 0.02)
    spinning = _released_state()
    spinning[10:13] = (6.0, -4.0, 2.0)
    hold = jnp.zeros((box.block_count, 4))

    def rate_spread(controller, state):
        rollout = controller._rollout(hold, jnp.asarray(state), posterior, held)
        return np.asarray(
            controller._spreads_marginal_full(
                rollout,
                posterior,
                jnp.asarray(state),
                False,
                held,
                controller.config.horizon_neighbourhood,
            )[0]
        )

    box_calm, box_spinning = rate_spread(box, calm), rate_spread(box, spinning)
    cmd_calm, cmd_spinning = (
        rate_spread(commands, calm),
        rate_spread(commands, spinning),
    )
    assert box_spinning[-1] > box_calm[-1]
    assert np.allclose(cmd_calm, cmd_spinning)
    assert cmd_spinning[-1] < box_spinning[-1]
