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

import numpy as np
import pytest

from glassbox.control.online_bootstrap import (
    RecursiveBootstrapBelief,
    RecursiveBootstrapIdentifier,
)
from glassbox.core.dynamics import GRAVITY_M_S2
from glassbox.experimental.dual_control import (
    DualControlConfig,
    DualControlNMPC,
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
    assert len(dual["information_gain_per_step"]) == 900
    assert len(dual["command_information_log_determinant"]) == 900
    assert len(dual["early_commands"]) == 30
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
    for model in ("certified", "working"):
        assert json.dumps(fresh["modes"][model], sort_keys=True) == json.dumps(
            want["modes"][model], sort_keys=True
        ), model
    assert json.dumps(
        fresh["difference_working_minus_certified"], sort_keys=True
    ) == json.dumps(want["difference_working_minus_certified"], sort_keys=True)
