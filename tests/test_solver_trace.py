"""Tracing preserves work; fixed directional probes expose their resolution."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import test_task_qualification as task_fixture

from glassbox.control.plan import ReferenceTrajectory, SolverPolicy
from glassbox.experimental import quasi_newton as optimizer
from glassbox.experimental import solver_trace as q
from glassbox.experimental.first_order_qualification import FirstOrderSolver

learned = task_fixture.learned


def quadratic(point):
    return np.sum(point**2), 2 * point


def analytic_record(seed=102, origin=134):
    """Genuine small optimization and audit, independent of Cascade artifacts."""
    point = np.zeros((5, 3), np.float32)
    target = np.full((5, 3), 0.25, np.float32)
    target[-1, -1] = 2

    def objective(x):
        error = x - target
        return np.sum(error**2), 2 * error

    value, gradient = objective(point)
    policy = SolverPolicy(horizon_steps=5, block_count=5, maximum_iterations=64)
    with q.trace_minimize(point, value, gradient) as trace:
        outcome, work = optimizer.bounded_minimize(
            objective, point, value, gradient, policy, _ftol=0.0
        )
    trace["retained_work"] = copy.deepcopy(work)
    directions = q.direction_audit(
        objective, outcome.blocks, outcome.value, outcome.gradient, q.EXPONENTS
    )
    return dict(seed=seed, origin=origin, work=work, trace=trace, directions=directions)


def test_all_frozen_coordinate_directions_and_steps_are_explicit():
    point = np.full((5, 3), 0.25, np.float32)
    audit = q.direction_audit(quadratic, point, *quadratic(point), q.EXPONENTS)
    assert len(audit["directions"]) == 31
    assert len(audit["probes"]) == 31 * 6
    for coordinate in range(point.size):
        for offset, sign in enumerate((-1, 1)):
            vector = np.zeros(point.size)
            vector[coordinate] = sign
            np.testing.assert_array_equal(
                np.asarray(
                    audit["directions"][2 * coordinate + offset]["vector"]
                ).ravel(),
                vector,
            )
    for direction in range(31):
        rows = audit["probes"][direction * 6 : (direction + 1) * 6]
        assert [r["exponent"] for r in rows] == list(q.EXPONENTS)
        assert [r["requested_step"] for r in rows] == [2.0**e for e in q.EXPONENTS]
    assert audit["diagnostic_function_calls"] == 1 + len(audit["probes"])
    json.dumps(audit, allow_nan=False)


def test_active_bounds_skip_outward_proposals_without_clipping_or_calling():
    calls = []

    def objective(point):
        calls.append(point.copy())
        return quadratic(point)

    point = np.array([[1, -1]], np.float32)
    audit = q.direction_audit(objective, point, *quadratic(point), [-4])
    assert [r["status"] for r in audit["probes"][:4]] == [
        "evaluated",
        "infeasible",
        "infeasible",
        "evaluated",
    ]
    assert audit["probes"][1]["host_blocks"][0][0] > 1
    assert audit["probes"][1]["canonical_blocks"] is None
    assert len(calls) == audit["diagnostic_function_calls"] == 4
    assert all(np.abs(p).max() <= 1 for p in calls)


def test_actual_displacement_and_linearization_use_exact_float64_differences():
    point = np.array([[0.5, -0.25]], np.float32)
    audit = q.direction_audit(quadratic, point, *quadratic(point), [-4])
    row = audit["probes"][0]
    displacement = np.asarray(row["canonical_blocks"], dtype=np.float64) - point.astype(
        np.float64
    )
    np.testing.assert_array_equal(row["actual_displacement"], displacement)
    assert row["predicted_change"] == -0.0625
    assert row["observed_change"] == -0.05859375
    assert row["absolute_linearization_discrepancy"] == 0.0625**2
    assert row["observed_predicted_ratio"] == 0.9375
    assert row["predicted_change_in_spacings"] == 0.0625 / float(
        np.spacing(np.float32(0.3125))
    )


def test_collapsed_input_steps_are_evaluated_and_marked_without_fake_ratios():
    point = np.array([[0.75]], np.float32)
    audit = q.direction_audit(quadratic, point, *quadratic(point), [-26])
    assert all(row["status"] == "collapsed" for row in audit["probes"])
    assert audit["diagnostic_function_calls"] == 4
    for row in audit["probes"]:
        assert row["observed_change"] == row["predicted_change"] == 0
        assert row["observed_predicted_ratio"] is None


def test_objective_ties_do_not_hide_nonzero_predicted_changes():
    def offset(point):
        value, gradient = quadratic(point)
        return np.float32(1e8) + value, gradient

    point = np.array([[0.5]], np.float32)
    audit = q.direction_audit(offset, point, *offset(point), [-4])
    assert all(row["observed_change"] == 0 for row in audit["probes"])
    assert all(row["predicted_change"] != 0 for row in audit["probes"])
    assert all(row["status"] == "evaluated" for row in audit["probes"])


def test_stationary_projected_direction_is_explicitly_skipped():
    def objective(point):
        error = point - np.float32(2)
        return np.sum(error**2), 2 * error

    point = np.ones((1, 1), np.float32)
    audit = q.direction_audit(objective, point, *objective(point), [-4, -8])
    assert audit["directions"][-1]["zero_direction"]
    assert all(row["status"] == "zero_direction" for row in audit["probes"][-2:])
    assert audit["diagnostic_function_calls"] == 3


def test_base_audit_rejects_changed_value_or_gradient():
    point = np.ones((1, 1), np.float32) * 0.5
    value, gradient = quadratic(point)
    with pytest.raises(AssertionError):
        q.direction_audit(quadratic, point, value + 1, gradient, [-4])
    with pytest.raises(AssertionError):
        q.direction_audit(quadratic, point, value, gradient + 1, [-4])


def test_real_backend_trace_preserves_work_and_exact_output():
    original_backend = optimizer.minimize
    record = analytic_record()
    trace, work = record["trace"], record["work"]
    assert optimizer.minimize is original_backend
    assert len(trace["requests"]) == (
        work["new_objective_evaluations"] + work["cached_seed_requests"]
    )
    assert len(trace["callbacks"]) == work["accepted_iterations"]
    assert trace["raw_backend"]["nit"] == work["accepted_iterations"]
    assert trace["raw_backend"]["nfev"] == len(trace["requests"])
    assert trace["requests"][0]["cached_seed"]
    assert not trace["requests"][0]["best_updated"]
    assert all(row["matched_request"] is not None for row in trace["callbacks"])
    assert trace["retained_work"] == work
    json.dumps(record, allow_nan=False)


def test_trace_keeps_first_best_point_on_ties_and_records_raw_later_point(monkeypatch):
    def objective(point):
        return np.float32(100), np.full(point.shape, 0.01, np.float32)

    def backend(fun, initial, **options):
        fun(initial)
        later = np.full(initial.shape, 0.125)
        value, gradient = fun(later)
        options["callback"](later)
        return SimpleNamespace(
            x=later,
            fun=value,
            jac=gradient,
            status=0,
            success=True,
            message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
            nit=1,
            nfev=2,
            njev=2,
        )

    monkeypatch.setattr(optimizer, "minimize", backend)
    point = np.zeros((1, 1), np.float32)
    value, gradient = objective(point)
    policy = SolverPolicy(horizon_steps=1, block_count=1, maximum_iterations=64)
    with q.trace_minimize(point, value, gradient) as trace:
        outcome, work = optimizer.bounded_minimize(
            objective, point, value, gradient, policy, _ftol=0.0
        )
    assert not any(row["best_updated"] for row in trace["requests"])
    assert trace["raw_backend"]["x"] == [0.125]
    assert trace["callbacks"][0]["matched_request"] == 1
    assert work["returned_seed"] and not outcome.converged
    np.testing.assert_array_equal(outcome.blocks, point)
    assert optimizer.minimize is backend


def test_trace_restores_backend_on_programming_error(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("backend programming error")

    monkeypatch.setattr(optimizer, "minimize", broken)
    point = np.zeros((1, 1), np.float32)
    with pytest.raises(RuntimeError, match="programming error"):
        with q.trace_minimize(point, *quadratic(point)):
            optimizer.bounded_minimize(
                quadratic,
                point,
                *quadratic(point),
                SolverPolicy(horizon_steps=1, block_count=1, maximum_iterations=64),
                _ftol=0.0,
            )
    assert optimizer.minimize is broken


def solver_fixture(learned):
    arm = task_fixture.make_arm(learned, task_fixture.q.CANDIDATE)
    arm.reset(task_fixture.INITIAL, task_fixture.COMMAND)
    model = arm.plan.with_causal_state(arm._state)
    policy = replace(arm.policy, maximum_iterations=64)
    reference = ReferenceTrajectory.hold(task_fixture.INITIAL, 5)
    return model, policy, reference


def test_full_solver_trace_preserves_results_work_seeds_and_kernel_identity(learned):
    model, policy, reference = solver_fixture(learned)
    baseline, traced = (
        FirstOrderSolver(model, policy),
        q.TracedFirstOrderSolver(model, policy),
    )
    original_backend, kernels = optimizer.minimize, baseline._kernels
    warm = None
    for _ in range(2):
        a = baseline.solve(
            task_fixture.INITIAL, reference, task_fixture.COMMAND, warm_start=warm
        )
        b = traced.solve(
            task_fixture.INITIAL, reference, task_fixture.COMMAND, warm_start=warm
        )
        for key in ("command", "predicted_commands", "predicted_states"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        assert a.status == b.status
        assert (
            baseline.last_work == traced.last_work == traced.last_trace["retained_work"]
        )
        assert traced._kernels is baseline._kernels is kernels
        assert optimizer.minimize is original_backend
        assert traced.last_work["seed_objective_evaluations"] == (
            1 if warm is None else 2
        )
        assert traced.last_directions["diagnostic_function_calls"] > 1
        warm = a.warm_start
    invalid = traced.solve(np.zeros(2), reference, task_fixture.COMMAND)
    assert invalid.used_fallback
    assert traced.last_trace is None and traced.last_directions is None


def test_full_solver_restores_both_patched_surfaces_on_error(learned, monkeypatch):
    model, policy, reference = solver_fixture(learned)
    traced = q.TracedFirstOrderSolver(model, policy)
    kernels = traced._kernels

    def broken(fun, initial, **kwargs):
        fun(initial)
        raise RuntimeError("unexpected backend defect")

    monkeypatch.setattr(optimizer, "minimize", broken)
    with pytest.raises(RuntimeError, match="unexpected backend defect"):
        traced.solve(task_fixture.INITIAL, reference, task_fixture.COMMAND)
    assert optimizer.minimize is broken
    assert traced._kernels is kernels
    assert traced.last_trace["requests"][0]["cached_seed"]
    assert traced.last_directions is None
