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
def test_dual_control_study_mode_reports_the_canonical_case() -> None:
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
