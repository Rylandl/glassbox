"""Frozen tracking arms: causal isolation, work, precision and honest timing."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import test_precision_solver as precision_fixture
import test_task_qualification as fixture

from glassbox.control.plan import ReferenceTrajectory
from glassbox.control.solver import SolveStatus, _SolveAbort
from glassbox.experimental import quasi_newton
from glassbox.experimental import tracking_solver as q

learned = fixture.learned
PLAN = json.loads(
    (
        Path(__file__).resolve().parents[1] / "docs/harness/solver-tracking-v1.json"
    ).read_text()
)


def make_arm(learned, name, monkeypatch):
    monkeypatch.setattr(
        q, "PrecisionFactory", lambda model: precision_fixture.factory()
    )
    return q.TrackingOracleArm(
        PLAN, fixture.CONTROL, learned, None, name, _equations=fixture.Equations()
    )


def ready(arm):
    arm.reset(fixture.INITIAL, fixture.COMMAND)
    for _ in range(2):
        arm.observe(np.asarray(arm.equations.canonical(arm._state)))
        arm.command_applied(fixture.COMMAND)
    state = np.asarray(arm.equations.canonical(arm._state))
    arm.observe(state)
    return state, ReferenceTrajectory.hold(fixture.INITIAL, 5)


@pytest.mark.parametrize("name", q.ARMS)
def test_short_closed_loop_records_and_replay_are_exact(learned, monkeypatch, name):
    arm = make_arm(learned, name, monkeypatch)
    first = fixture.short_trial(arm)
    diagnostics = arm.diagnostic_arrays()
    records = copy.deepcopy(arm.records)
    assert len(records) == 2
    assert (
        set(diagnostics) == fixture.q.ORIGINAL_DIAGNOSTICS | fixture.q.EXTRA_DIAGNOSTICS
    )
    assert [record["origin"] for record in records] == [2, 3]
    for record in records:
        assert set(record) == q.RECORD_FIELDS
        assert record["arm"] == name
        assert "solve_time_s" not in json.dumps(record)
        assert set(record["audits"]) == {"lifted_seed64", "returned_plan64"}
        if name == q.PG4:
            assert record["work"] is None
        else:
            assert record["work"]["accepted_iterations"] <= 64
            assert record["work"]["new_objective_evaluations"] <= 1024
        if name == q.LBFGS64:
            assert record["seed_selection"]["successful"]
            assert record["work"]["seed_objective_evaluations"] == 1
            assert record["precision"]["floating_dtype"] == "float64"
            assert record["precision"]["jaxpr"]["float32_arithmetic"] == 0
            assert (
                record["audits"]["lifted_seed64"]["blocks"]
                == record["capture"]["blocks"]
            )
            assert (
                record["audits"]["returned_plan64"]["gradient"]
                == record["work"]["independent_gradient"]
            )
        else:
            assert (
                record["capture"]
                is record["precision"]
                is record["seed_selection"]
                is None
            )
    assert diagnostics["candidate_commands"].dtype == (
        np.float64 if name == q.LBFGS64 else np.float32
    )
    assert diagnostics["causal_states"].dtype == np.float32
    np.testing.assert_array_equal(
        diagnostics["variable_final_objectives"],
        diagnostics["final_objectives"] - diagnostics["constant_covariance_objectives"],
    )
    if name == q.LBFGS64:
        assert np.all(
            diagnostics["constant_covariance_objectives"] != arm.constant_objective
        )
    second = fixture.short_trial(arm)
    for field in first:
        np.testing.assert_array_equal(first[field], second[field])
    for field in diagnostics:
        np.testing.assert_array_equal(
            diagnostics[field], arm.diagnostic_arrays()[field]
        )
    assert arm.records == records
    assert not jax.config.x64_enabled


def test_pg4_trajectory_and_diagnostics_match_unchanged_task_arm(learned, monkeypatch):
    original = fixture.make_arm(learned, fixture.q.CANDIDATE)
    arm = make_arm(learned, q.PG4, monkeypatch)
    before = fixture.short_trial(original)
    after = fixture.short_trial(arm)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    for name, value in original.diagnostic_arrays().items():
        np.testing.assert_array_equal(value, arm.diagnostic_arrays()[name])
    assert arm.policy == original.policy
    assert not arm.summary()["objective_evaluation_counts_available"]


def test_seed_only_keeps_exact_original_seed_without_calling_minimize(
    learned, monkeypatch
):
    model, policy, original, previous = precision_fixture.baseline(learned)
    reference = ReferenceTrajectory.hold(fixture.INITIAL, 5)
    original.solve(
        fixture.INITIAL, reference, fixture.COMMAND, warm_start=previous.warm_start
    )

    def forbidden(*args, **kwargs):
        pytest.fail("seed selection invoked an optimizer")

    monkeypatch.setattr(quasi_newton, "minimize", forbidden)
    selector = q.SeedOnlySolver(model, policy)
    result = selector.solve(
        fixture.INITIAL, reference, fixture.COMMAND, warm_start=previous.warm_start
    )
    assert selector.last_capture.record() == original.last_capture.record()
    assert selector.last_work["seed_objective_evaluations"] == 2
    assert selector.last_work["new_objective_evaluations"] == 0
    assert selector.last_work["audit_objective_evaluations"] == 0
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.final_objective == selector.last_capture.native_seed_value


def test_float64_own_warm_plan_converts_through_original_float32_seed_path(
    learned, monkeypatch
):
    seen = []
    original = q.SeedOnlySolver._warm_blocks

    def observe(self, warm):
        assert not jax.config.x64_enabled
        blocks = original(self, warm)
        seen.append((warm, np.asarray(warm.commands).dtype, np.asarray(blocks).dtype))
        return blocks

    monkeypatch.setattr(q.SeedOnlySolver, "_warm_blocks", observe)
    arm = make_arm(learned, q.LBFGS64, monkeypatch)
    state, reference = ready(arm)
    first = arm.solve(state, reference, fixture.COMMAND)
    arm.command_applied(first.command)
    state = np.asarray(arm.equations.canonical(arm._state))
    arm.observe(state)
    arm.solve(state, reference, first.command, warm_start=first.warm_start)
    assert len(seen) == 1
    assert seen[0][0] is first.warm_start
    assert seen[0][1:] == (np.dtype("float64"), np.dtype("float32"))
    assert arm.records[0]["seed_selection"]["objective_evaluations"] == 1
    assert arm.records[1]["seed_selection"]["objective_evaluations"] == 2


def test_outer_duration_covers_seed_selection_precision_audits_and_synchronization(
    learned, monkeypatch
):
    arm = make_arm(learned, q.LBFGS64, monkeypatch)
    state, reference = ready(arm)
    clock = [100.0]
    monkeypatch.setattr(q, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    seed_solve = q.SeedOnlySolver.solve
    precise_solve = arm._precision_factory.solve
    block = jax.block_until_ready

    def seed(*args, **kwargs):
        result = seed_solve(*args, **kwargs)
        clock[0] += 3
        return result

    def precise(*args, **kwargs):
        result = precise_solve(*args, **kwargs)
        clock[0] += 11
        return result

    def synchronize(tree):
        result = block(tree)
        clock[0] += 5
        return result

    monkeypatch.setattr(q.SeedOnlySolver, "solve", seed)
    monkeypatch.setattr(arm._precision_factory, "solve", precise)
    monkeypatch.setattr(jax, "block_until_ready", synchronize)
    result = arm.solve(state, reference, fixture.COMMAND)
    assert result.diagnostics.solve_time_s == 19.0
    assert result.deadline_met is None
    assert "solve_time_s" not in json.dumps(arm.records)


@pytest.mark.parametrize("phase", ["selection", "selected_rollout"])
def test_failed_seed_selection_records_bounded_fallback_and_no_float64_work(
    learned, monkeypatch, phase
):
    arm = make_arm(learned, q.LBFGS64, monkeypatch)

    def fail_seed(self, *args, **kwargs):
        self.last_work["seed_objective_evaluations"] = 1
        raise _SolveAbort(SolveStatus.NONFINITE_OBJECTIVE, "injected nonfinite seed")

    def fail_selected_rollout(self, *args, **kwargs):
        assert self.last_capture is not None
        raise _SolveAbort(SolveStatus.NONFINITE_OBJECTIVE, "injected nonfinite rollout")

    def forbidden(*args, **kwargs):
        pytest.fail("failed seed invoked precision factory")

    if phase == "selection":
        monkeypatch.setattr(q.SeedOnlySolver, "_seed_plan", fail_seed)
    else:
        monkeypatch.setattr(q.SeedOnlySolver, "_evaluate_blocks", fail_selected_rollout)
    monkeypatch.setattr(arm._precision_factory, "solve", forbidden)
    tracking = fixture.short_trial(arm)
    assert len(tracking["commands"]) == 4
    for record in arm.records:
        assert record["seed_selection"]["objective_evaluations"] == 1
        assert not record["seed_selection"]["successful"]
        assert record["seed_selection"]["used_fallback"]
        assert record["capture"] is record["precision"] is None
        assert record["audits"] == dict(lifted_seed64=None, returned_plan64=None)
        assert record["work"]["seed_objective_evaluations"] == 0
        assert record["work"]["new_objective_evaluations"] == 0
        assert record["work"]["audit_objective_evaluations"] == 0
    diagnostics = arm.diagnostic_arrays()
    assert diagnostics["used_fallback"].all()
    assert np.isinf(diagnostics["final_objectives"]).all()
    assert not diagnostics["forecast_states"].any()


def test_finite_return_does_not_hide_backend_failure(learned, monkeypatch):
    arm = make_arm(learned, q.LBFGS64, monkeypatch)
    original = arm._precision_factory.solve

    def backend_failed(*args, **kwargs):
        result, work, audits, evidence = original(*args, **kwargs)
        work.update(backend_failure=True, backend_status=2, backend_message="ABNORMAL")
        return result, work, audits, evidence

    monkeypatch.setattr(arm._precision_factory, "solve", backend_failed)
    state, reference = ready(arm)
    result = arm.solve(state, reference, fixture.COMMAND)
    assert not result.used_fallback
    assert arm.records[0]["work"]["backend_failure"]
    assert arm.records[0]["audits"]["returned_plan64"] is not None


def test_scope_restores_after_backend_exception(learned, monkeypatch):
    arm = make_arm(learned, q.LBFGS64, monkeypatch)
    state, reference = ready(arm)

    def crash(*args, **kwargs):
        assert jax.config.x64_enabled
        raise RuntimeError("injected backend error")

    monkeypatch.setattr(quasi_newton, "minimize", crash)
    with pytest.raises(RuntimeError, match="injected backend error"):
        arm.solve(state, reference, fixture.COMMAND)
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("name", q.ARMS)
def test_causal_order_default_precision_and_no_deadline(learned, monkeypatch, name):
    arm = make_arm(learned, name, monkeypatch)
    arm.reset(fixture.INITIAL, fixture.COMMAND)
    with pytest.raises(ValueError, match="ready"):
        arm.solve(
            fixture.INITIAL,
            ReferenceTrajectory.hold(fixture.INITIAL, 5),
            fixture.COMMAND,
        )
    state, reference = ready(arm)
    with pytest.raises(ValueError, match="deadline"):
        arm.solve(state, reference, fixture.COMMAND, deadline_s=0.05)
    with jax.enable_x64(True):
        with pytest.raises(ValueError, match="default float32"):
            arm.solve(state, reference, fixture.COMMAND)
        with pytest.raises(ValueError, match="default float32"):
            arm.command_applied(fixture.COMMAND)
    assert not arm.records


@pytest.mark.cascade
@pytest.mark.parametrize("name", q.ARMS)
def test_public_cascade_short_closed_loop(learned, name):
    """Four equilibrium-origin transitions; none of the twelve declared trials."""
    cascade = pytest.importorskip("cascade")
    arm = q.TrackingOracleArm(
        PLAN, fixture.CONTROL, learned, cascade.skywalker_x8_spec().to_model(), name
    )
    arm.reset(fixture.INITIAL, fixture.COMMAND)
    previous, warm = fixture.COMMAND, None
    for _ in range(4):
        state = np.asarray(arm.equations.canonical(arm._state))
        arm.observe(state)
        command = previous
        if arm.ready:
            result = arm.solve(
                state,
                ReferenceTrajectory.hold(fixture.INITIAL, 5),
                previous,
                warm_start=warm,
            )
            assert not result.used_fallback
            assert result.diagnostics.solve_time_s > 0
            command, warm = result.command, result.warm_start
            assert np.isfinite(result.predicted_states).all()
        assert not jax.config.x64_enabled
        arm.command_applied(command)
        previous = command
    assert len(arm.records) == 2
    assert arm.diagnostic_arrays()["causal_states"].dtype == np.float32
    assert arm.diagnostic_arrays()["candidate_commands"].dtype == (
        np.float64 if name == q.LBFGS64 else np.float32
    )
