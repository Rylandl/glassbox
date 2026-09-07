"""Check the arithmetic and feasibility contracts of the SQP experiment."""

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control.fitted import NMPCController
from glassbox.control.plan import SafetyEnvelope, SolverPolicy
from glassbox.control.solver import _SolveAbort
from glassbox.core.data import make_trajectory_spec
from glassbox.core.dynamics import QUADROTOR_CONTROL_NAMES
from glassbox.core.model import ExecutableModel, ModelValidityEnvelope, RuntimeModelSpec
from glassbox.core.synthetic import resting_state, true_parameters


@pytest.fixture
def investigation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_sqp_recovery")


@pytest.fixture
def small_controller():
    model = ExecutableModel(
        true_parameters(),
        make_trajectory_spec(
            QUADROTOR_CONTROL_NAMES,
            family="multirotor",
            observation_source="simulator_truth",
            configuration_id="sqp-test",
        ),
        RuntimeModelSpec(
            0.02, ModelValidityEnvelope((0.0,) * 3, (0.2,) * 3, (0.0,) * 3, (0.2,) * 3)
        ),
    )
    controller = NMPCController(
        model,
        policy=SolverPolicy(horizon_steps=2, block_count=2),
        safety_envelope=SafetyEnvelope(maximum_speed_m_s=0.2),
    )
    return controller


def test_residual_form_preserves_objective_and_gradient_with_uncertainty(
    investigation, small_controller
):
    controller = small_controller
    plan = controller.plan
    state = jnp.asarray(resting_state()).at[3].set(1.0)
    reference = controller.hold_reference(jnp.asarray(resting_state())).states
    previous = jnp.full(4, 0.5)

    def costs(blocks):
        prediction = plan.rollout(
            blocks, state, previous, jnp.zeros((2, 0)), plan.values
        )
        covariance = jnp.broadcast_to(
            0.01 * (1 + jnp.sum(blocks**2)) * jnp.eye(12), (2, 12, 12)
        )
        prediction = prediction._replace(tangent_covariance=covariance)
        residuals, _ = investigation.residuals_and_margins(
            plan, plan.policy, prediction, reference, previous
        )
        return jnp.asarray(
            (
                plan.stage_cost(prediction, reference, previous, plan.policy),
                residuals @ residuals,
            )
        )

    blocks = jnp.asarray([[0.1, -0.2, 0.2, -0.1]] * 2)
    values = costs(blocks)
    derivatives = jax.jacfwd(costs)(blocks)
    np.testing.assert_allclose(values[0], values[1], rtol=2e-6)
    np.testing.assert_allclose(derivatives[0], derivatives[1], rtol=2e-5, atol=2e-5)


def test_whitened_quadratic_step_handles_ill_conditioned_curvature(investigation):
    hessian = np.diag([1e-3, 1e3])
    gradient = -hessian @ np.asarray([0.8, 0.4])
    margins = np.asarray([0.5])
    jacobian = np.asarray([[-1.0, 0.0]])
    step, multipliers, usable, success = investigation.quadratic_step(
        hessian, gradient, margins, jacobian, np.zeros(2)
    )
    assert usable and success
    np.testing.assert_allclose(
        step, [0.5 - investigation.NUMERICAL_INTERIOR_MARGIN, 0.4], atol=1e-6
    )
    lagrangian = hessian @ step + gradient - jacobian.T @ multipliers[:1]
    np.testing.assert_allclose(lagrangian, 0.0, atol=1e-6)


def reference_problem(investigation, *, nonfinite=False):
    def packed(blocks, *_):
        multiplier = jnp.nan if nonfinite else 1.0
        return blocks.ravel() * multiplier, (blocks.ravel() - 0.5) * multiplier

    def with_aux(*args):
        values = packed(*args)
        return values, (values, None)

    solver = object.__new__(investigation.GaussNewtonReference)
    solver.model = SimpleNamespace(values=None)
    solver._kernels = SimpleNamespace(
        objective_and_gradient=jax.jit(
            jax.value_and_grad(lambda blocks, *_: jnp.sum(blocks**2))
        )
    )
    solver.work_estimates = None
    solver.fused_output = False
    solver.reports = []
    solver._seed_derivative = None
    solver._seed_prediction = None
    solver._seed_linearization_time_s = 0.0
    solver._iteration_budget = 8
    solver.legacy_seeding = False
    solver._warm_blocks = lambda warm: warm
    solver.warm_iterations = 2
    solver.linearize = jax.jit(jax.jacfwd(with_aux, has_aux=True))
    solver.evaluate = jax.jit(packed)
    return solver


def test_sqp_keeps_a_feasible_result_even_when_the_infeasible_seed_is_cheaper(
    investigation,
):
    solver = reference_problem(investigation)
    outcome = solver._optimize_plan(
        jnp.zeros((1, 1)),
        jnp.asarray(0.0),
        jnp.zeros((1, 1)),
        None,
        None,
        SimpleNamespace(states=None),
        None,
        None,
    )
    assert outcome.finite and not outcome.converged
    assert not outcome.progressed
    np.testing.assert_allclose(
        outcome.blocks, [[0.5 + investigation.NUMERICAL_INTERIOR_MARGIN]], atol=1e-6
    )
    assert solver.reports[-1]["feasible"]


def test_sqp_refuses_nonfinite_linearization(investigation):
    solver = reference_problem(investigation, nonfinite=True)
    with pytest.raises(_SolveAbort, match="no feasible command plan"):
        solver._optimize_plan(
            jnp.zeros((1, 1)),
            jnp.asarray(0.0),
            jnp.zeros((1, 1)),
            None,
            None,
            SimpleNamespace(states=None),
            None,
            None,
        )
    assert solver.reports[-1]["stop_reason"] == "nonfinite_linearization"
    assert not solver.reports[-1]["feasible"]


@pytest.mark.parametrize(
    "cold,warm,expected,warm_used",
    [
        (0.0, 0.7, 0.7, True),  # Feasible, despite greater objective.
        (0.6, 0.7, 0.6, False),  # Cheapest of two feasible seeds.
        (0.7, 0.7, 0.7, True),  # The existing exact tie selects the warm seed.
        (0.0, 0.4, 0.4, True),  # Less violation while still infeasible.
        (0.6, float("nan"), 0.6, False),
        (float("nan"), 0.6, 0.6, True),
    ],
)
@pytest.mark.parametrize("reuse_single_seed", [False, True])
def test_sqp_seed_selection_uses_feasibility_before_cost(
    investigation, cold, warm, expected, warm_used, reuse_single_seed
):
    solver = reference_problem(investigation)
    solver.reuse_single_seed = reuse_single_seed
    blocks, _, gradient, cost, used = solver._seed_plan(
        jnp.asarray([[cold]]),
        jnp.asarray([[warm]]),
        None,
        None,
        SimpleNamespace(states=None),
        None,
        None,
    )
    assert used is warm_used
    np.testing.assert_allclose(blocks, [[expected]])
    np.testing.assert_allclose(gradient, [[2 * expected]])
    assert cost == pytest.approx(expected**2)


@pytest.mark.parametrize("mode", ["cold", "warm", "incompatible_warm"])
def test_single_seed_reuses_values_gradient_and_constraints(investigation, mode):
    solver = reference_problem(investigation)
    cold = None if mode == "warm" else jnp.asarray([[0.7]])
    warm = None if mode == "cold" else jnp.asarray([[0.8]])
    if mode == "incompatible_warm":
        solver._warm_blocks = lambda _: None
    args = cold, warm, None, None, SimpleNamespace(states=None), None, None
    expected = solver._seed_plan(*args)
    derivative = solver._seed_derivative
    solver.reuse_single_seed = True
    solver.evaluate = lambda *_: pytest.fail("one seed needs no standalone evaluation")
    linearize, calls = solver.linearize, []

    def counted(*args):
        calls.append(True)
        return linearize(*args)

    solver.linearize = counted
    actual = solver._seed_plan(*args)
    assert len(calls) == 1
    assert solver._iteration_budget == (2 if mode == "warm" else 8)
    assert actual[-1] == (mode == "warm")
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(
        jax.tree.leaves(solver._seed_derivative), jax.tree.leaves(derivative)
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("remaining,accepted", [(0.009, True), (0.008, False)])
def test_single_seed_admission_charges_only_linearization_and_reserve(
    investigation, monkeypatch, remaining, accepted
):
    solver, reference, _, clock, budget = budgeted_problem(investigation, monkeypatch)
    clock.now = 0.020 - remaining
    solver.evaluate = lambda *_: pytest.fail("standalone evaluation is not needed")
    # This same budget cannot admit the old evaluation-plus-linearization path.
    with pytest.raises(_SolveAbort, match="insufficient SQP seed budget"):
        solver._seed_plan(
            jnp.ones((1, 1)), None, None, None, reference, None, None, budget=budget
        )
    solver.reuse_single_seed = True
    if accepted:
        solver._seed_plan(
            jnp.ones((1, 1)), None, None, None, reference, None, None, budget=budget
        )
        assert solver._seed_derivative is not None
    else:
        solver.linearize = lambda *_: pytest.fail("reserve must remain protected")
        with pytest.raises(_SolveAbort, match="insufficient SQP linearization budget"):
            solver._seed_plan(
                jnp.ones((1, 1)), None, None, None, reference, None, None, budget=budget
            )
        assert solver._seed_derivative is None
        assert solver._seed_prediction is None


@pytest.mark.parametrize(
    "bad_part",
    ["residual", "margin", "residual_jacobian", "margin_jacobian", "cost_overflow"],
)
def test_single_seed_rejects_nonfinite_values_before_caching(investigation, bad_part):
    solver = reference_problem(investigation)
    solver.reuse_single_seed = True
    solver.evaluate = lambda *_: pytest.fail("one seed must use linearization values")
    args = jnp.ones((1, 1)), None, None, None, SimpleNamespace(states=None), None, None
    original = solver.linearize
    solver._seed_plan(*args)

    def invalid(*_):
        parts = {
            name: np.ones((1,))
            for name in ("residual", "margin", "residual_jacobian", "margin_jacobian")
        }
        if bad_part == "cost_overflow":
            parts["residual"][:] = 1e200
        else:
            parts[bad_part][:] = np.nan
        return (
            (parts["residual_jacobian"], parts["margin_jacobian"]),
            ((parts["residual"], parts["margin"]), "invalid prepared prediction"),
        )

    solver.linearize = invalid
    with np.errstate(over="ignore"), pytest.raises(_SolveAbort, match="non-finite"):
        solver._seed_plan(*args)
    assert solver._seed_derivative is None
    assert solver._seed_prediction is None
    solver.linearize = original
    solver._seed_plan(*args)
    assert solver._seed_derivative is not None


def test_single_seed_does_not_change_legacy_seeding(investigation, monkeypatch):
    from glassbox.control.solver import BoundedShootingSolver

    solver = reference_problem(investigation)
    solver.reuse_single_seed = solver.legacy_seeding = True
    sentinel = object()
    monkeypatch.setattr(BoundedShootingSolver, "_seed_plan", lambda *_a, **_k: sentinel)
    solver.linearize = lambda *_: pytest.fail("legacy seeding owns its own evaluation")
    result = solver._seed_plan(
        jnp.ones((1, 1)), None, None, None, SimpleNamespace(states=None), None, None
    )
    assert result is sentinel


@pytest.mark.parametrize("use_observation", [False, True])
@pytest.mark.parametrize("precision_stopping", [False, True])
def test_single_seed_still_rejects_a_late_linearization_at_solve_boundary(
    investigation, small_controller, monkeypatch, use_observation, precision_stopping
):
    from glassbox.control import solver as solver_module

    plan = small_controller.plan
    solver = investigation.GaussNewtonReference(
        plan,
        replace(plan.policy, allow_unresolved_parameters=True),
        reuse_single_seed=True,
        use_observed_linearization_cost=use_observation,
        precision_stopping=precision_stopping,
    )
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        solver_module, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )
    linearize = solver.linearize

    def late(*args):
        result = linearize(*args)
        clock.now = 0.03
        return result

    solver.linearize = late
    solver._optimize_plan = lambda *_a, **_k: pytest.fail("late seeds cannot optimize")
    state = jnp.asarray(resting_state())
    previous = jnp.full(4, 0.5)
    result = solver.solve(
        state,
        solver.hold_reference(state),
        previous,
        latent_state=previous,
        deadline_s=0.02,
    )
    assert not result.command_usable and result.deadline_met is False
    assert result.nonlinear_feasibility.status == "not_assessed"
    assert result.message == "solver deadline expired before optimization"


def test_sqp_reuses_selected_linearization_and_resets_cold_budget(investigation):
    solver = reference_problem(investigation)
    linearize = solver.linearize
    calls = []

    def counted(*args):
        calls.append(np.asarray(args[0]).copy())
        return linearize(*args)

    solver.linearize = counted
    reference = SimpleNamespace(states=None)
    blocks, value, gradient, _, _ = solver._seed_plan(
        jnp.zeros((1, 1)),
        jnp.ones((1, 1)),
        None,
        None,
        reference,
        None,
        None,
    )
    assert solver._iteration_budget == 2
    solver._optimize_plan(blocks, value, gradient, None, None, reference, None, None)
    assert len(calls) == solver.reports[-1]["iterations"]
    assert solver._seed_derivative is None
    # A fresh request has no warm plan, regardless of reports or previous work.
    solver._seed_plan(jnp.zeros((1, 1)), None, None, None, reference, None, None)
    assert solver._iteration_budget == 8
    np.testing.assert_array_equal(solver._seed_derivative[1][0], [0.0])


@pytest.mark.parametrize("constant_margin,usable", [(0.0, True), (-0.1, False)])
def test_sqp_does_not_tighten_constant_initial_state_constraints(
    investigation, constant_margin, usable
):
    step, _, result_usable, _ = investigation.quadratic_step(
        np.eye(1),
        np.asarray([-0.2]),
        np.asarray([constant_margin]),
        np.zeros((1, 1)),
        np.zeros(1),
    )
    assert result_usable is usable
    if usable:
        np.testing.assert_allclose(step, [0.2], atol=1e-6)


def test_sqp_accepts_a_model_with_no_nonlinear_constraints(investigation):
    step, _, usable, _ = investigation.quadratic_step(
        np.eye(1),
        np.asarray([-0.2]),
        np.empty(0),
        np.empty((0, 1)),
        np.zeros(1),
    )
    assert usable
    np.testing.assert_allclose(step, [0.2], atol=1e-6)


def budgeted_problem(investigation, monkeypatch, *, seed=0.7):
    from glassbox.control import solver as solver_module

    solver = reference_problem(investigation)
    reference = SimpleNamespace(states=None)
    arguments = solver._seed_plan(
        jnp.asarray([[seed]]), None, None, None, reference, None, None
    )[:3]
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        solver_module, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )
    solver.work_estimates = investigation.SQPWorkEstimates()
    budget = solver_module._SolveBudget(0.020)
    return solver, reference, arguments, clock, budget


def observed_cost_problem(investigation, monkeypatch, *, scale=1.0, feasible=True):
    """All costs are synthetic relative units, independent of execution speed."""
    from glassbox.control.plan import PlanMeasurements, Prediction
    from glassbox.control.solver import _SolveBudget

    solver, reference, arguments, clock, _ = budgeted_problem(
        investigation, monkeypatch, seed=0.7 if feasible else 0.0
    )
    monkeypatch.setattr(
        investigation, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )
    solver.work_estimates = investigation.SQPWorkEstimates(
        linearization_s=scale,
        quadratic_step_s=0.1 * scale,
        evaluation_s=0.1 * scale,
        output_reserve_s=0.2 * scale,
    )
    solver.use_observed_linearization_cost = True
    solver._iteration_budget = 2
    solver._seed_linearization_time_s = 2 * scale
    solver._seed_prediction = (
        Prediction(
            mean_states=jnp.ones((2, 13)),
            tangent_covariance=jnp.zeros((1, 12, 12)),
            commands=arguments[0],
            latent_states=jnp.zeros((2, 1)),
            exogenous=jnp.empty((1, 0)),
        ),
        PlanMeasurements(0.5, 0.0, 0.0),
        arguments[1],
    )
    calls = {"linearize": 0, "quadratic": 0, "evaluate": 0}
    linearize, evaluate = solver.linearize, solver.evaluate

    def simulated_linearize(*args):
        result = linearize(*args)
        calls["linearize"] += 1
        clock.now += 2 * scale
        return result

    def simulated_evaluate(*args):
        result = evaluate(*args)
        calls["evaluate"] += 1
        clock.now += 0.1 * scale
        return result

    def simulated_quadratic(*args):
        calls["quadratic"] += 1
        clock.now += 0.1 * scale
        # A no-op preserves a feasible seed. The infeasible case improves its
        # merit but remains outside support, so no candidate can be retained.
        return (
            np.zeros_like(args[-1]) + (0.0 if feasible else 0.1),
            np.ones(1),
            True,
            True,
        )

    solver.linearize, solver.evaluate = simulated_linearize, simulated_evaluate
    monkeypatch.setattr(investigation, "quadratic_step", simulated_quadratic)
    clock.now = 2 * scale  # The measured seed work has already completed.
    return solver, reference, arguments, clock, _SolveBudget(3.5 * scale), calls


@pytest.mark.parametrize("scale", [0.001, 1.0, 17.0])
@pytest.mark.parametrize("feasible", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_observed_cost_controls_only_subsequent_linearization(
    investigation, monkeypatch, scale, feasible, enabled
):
    solver, reference, arguments, clock, budget, calls = observed_cost_problem(
        investigation, monkeypatch, scale=scale, feasible=feasible
    )
    solver.use_observed_linearization_cost = enabled
    configured = solver.work_estimates
    if feasible:
        outcome = solver._optimize_plan(
            *arguments, None, None, reference, None, None, budget=budget
        )
        assert outcome.evaluation.nonlinear_feasibility.status == "feasible"
        np.testing.assert_array_equal(outcome.blocks, arguments[0])
        assert solver.reports[-1]["output_source"] == "linearization_checkpoint"
    else:
        with pytest.raises(_SolveAbort, match="before finding a feasible plan"):
            solver._optimize_plan(
                *arguments, None, None, reference, None, None, budget=budget
            )
        assert not solver.reports[-1]["feasible"]
    assert calls == {"linearize": 0 if enabled else 1, "quadratic": 1, "evaluate": 1}
    assert (clock.now < budget.deadline_at) == enabled
    assert solver.work_estimates is configured
    assert configured.linearization_s == scale
    assert configured.output_reserve_s == 0.2 * scale
    report = solver.reports[-1]
    assert report["observed_linearization_floor_applied"] == enabled
    assert report["linearization_admission_estimate_s"] == (2 if enabled else 1) * scale
    assert report["stop_reason"] == "time_budget"
    assert solver._seed_derivative is None and solver._seed_prediction is None


@pytest.mark.parametrize(
    "mode", ["no_budget", "no_deadline", "no_estimates", "no_cache"]
)
def test_observed_cost_requires_current_seed_and_explicit_budget(
    investigation, monkeypatch, mode
):
    from glassbox.control.solver import _SolveBudget

    solver, reference, arguments, _, budget, calls = observed_cost_problem(
        investigation, monkeypatch
    )
    if mode == "no_budget":
        budget = None
    elif mode == "no_deadline":
        budget = _SolveBudget(None)
    elif mode == "no_estimates":
        solver.work_estimates = None
    else:
        solver._seed_derivative = solver._seed_prediction = None
    if mode == "no_cache":
        with pytest.raises(_SolveAbort, match="insufficient SQP output budget"):
            solver._optimize_plan(
                *arguments, None, None, reference, None, None, budget=budget
            )
    else:
        solver._optimize_plan(
            *arguments, None, None, reference, None, None, budget=budget
        )
    assert calls["linearize"] >= 1
    assert not solver.reports[-1]["observed_linearization_floor_applied"]
    assert solver.reports[-1]["linearization_admission_estimate_s"] == (
        1.0 if mode == "no_cache" else None
    )


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf"), 0.5])
def test_invalid_or_short_measurement_cannot_lower_configured_cost(
    investigation, monkeypatch, duration
):
    solver, reference, arguments, _, budget, calls = observed_cost_problem(
        investigation, monkeypatch
    )
    solver._seed_linearization_time_s = duration
    solver._optimize_plan(*arguments, None, None, reference, None, None, budget=budget)
    assert calls["linearize"] == 1
    assert solver.reports[-1]["linearization_admission_estimate_s"] == 1.0
    assert not solver.reports[-1]["observed_linearization_floor_applied"]


def test_next_request_uses_its_own_measurement(investigation, monkeypatch):
    from glassbox.control.solver import _SolveBudget

    solver, reference, arguments, clock, budget, _ = observed_cost_problem(
        investigation, monkeypatch
    )
    solver._optimize_plan(*arguments, None, None, reference, None, None, budget=budget)
    assert solver.reports[-1]["observed_linearization_floor_applied"]
    # A later request's faster linearization replaces the old observation.
    old = solver.linearize

    def faster(*args):
        before = clock.now
        result = old(*args)
        clock.now = before + 0.5
        return result

    solver.linearize = faster
    solver.reuse_single_seed = True
    arguments = solver._seed_plan(
        jnp.ones((1, 1)) * 0.7, None, None, None, reference, None, None
    )[:3]
    assert solver._seed_linearization_time_s == pytest.approx(0.5)
    solver._optimize_plan(
        *arguments,
        None,
        None,
        reference,
        None,
        None,
        budget=_SolveBudget(clock.now + 1.5),
    )
    assert not solver.reports[-1]["observed_linearization_floor_applied"]
    assert solver.reports[-1]["linearization_admission_estimate_s"] == 1.0


def test_observed_seed_duration_includes_materialization(investigation, monkeypatch):
    solver = reference_problem(investigation)
    solver.reuse_single_seed = True
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        investigation, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )

    class DeferredArray:
        def __init__(self, value):
            self.value = value

        def __array__(self, dtype=None, copy=None):
            clock.now += 0.5
            return np.asarray(self.value, dtype=dtype)

    def linearize(*_):
        clock.now += 0.25
        return (
            (DeferredArray(np.ones((1, 1, 1))), DeferredArray(np.zeros((1, 1, 1)))),
            ((DeferredArray(np.ones(1)), DeferredArray(np.ones(1))), None),
        )

    solver.linearize = linearize
    solver._seed_plan(
        jnp.ones((1, 1)), None, None, None, SimpleNamespace(states=None), None, None
    )
    assert (
        solver._seed_linearization_time_s == 2.25
    )  # Dispatch + four materializations.


def test_time_budget_returns_a_checked_seed_without_starting_another_step(
    investigation, monkeypatch
):
    solver, reference, arguments, clock, budget = budgeted_problem(
        investigation, monkeypatch
    )
    clock.now = 0.016
    monkeypatch.setattr(
        investigation, "quadratic_step", lambda *_: pytest.fail("must not start a QP")
    )
    outcome = solver._optimize_plan(
        *arguments, None, None, reference, None, None, budget=budget
    )
    assert outcome.finite and not outcome.converged
    np.testing.assert_allclose(outcome.blocks, [[0.7]])
    np.testing.assert_allclose(outcome.gradient, [[1.4]])
    assert solver.reports[-1]["iterations"] == 0
    assert solver.reports[-1]["stop_reason"] == "time_budget"
    assert "time budget" in outcome.stall_message


def test_time_budget_refuses_an_infeasible_seed(investigation, monkeypatch):
    solver, reference, arguments, clock, budget = budgeted_problem(
        investigation, monkeypatch, seed=0.0
    )
    clock.now = 0.016
    with pytest.raises(_SolveAbort, match="before finding a feasible plan") as caught:
        solver._optimize_plan(
            *arguments, None, None, reference, None, None, budget=budget
        )
    assert caught.value.status.value == "deadline_exceeded"
    assert not solver.reports[-1]["feasible"]


@pytest.mark.parametrize("finished_at,returns_plan", [(0.014, True), (0.017, False)])
def test_time_budget_retains_a_nonlinearly_checked_step_and_protects_output_reserve(
    investigation, monkeypatch, finished_at, returns_plan
):
    solver, reference, arguments, clock, budget = budgeted_problem(
        investigation, monkeypatch, seed=0.0
    )
    evaluate = solver.evaluate

    def slow_evaluation(*args):
        values = evaluate(*args)
        clock.now = finished_at
        return values

    solver.evaluate = slow_evaluation
    if returns_plan:
        outcome = solver._optimize_plan(
            *arguments, None, None, reference, None, None, budget=budget
        )
        np.testing.assert_allclose(
            outcome.blocks, [[0.5 + investigation.NUMERICAL_INTERIOR_MARGIN]], atol=1e-6
        )
        assert solver.reports[-1]["iterations"] == 1
        assert solver.reports[-1]["stop_reason"] == "time_budget"
    else:
        with pytest.raises(_SolveAbort, match="insufficient SQP output budget"):
            solver._optimize_plan(
                *arguments, None, None, reference, None, None, budget=budget
            )


def test_time_budget_does_not_trust_a_qp_step_without_nonlinear_evaluation(
    investigation, monkeypatch
):
    solver, reference, arguments, clock, budget = budgeted_problem(
        investigation, monkeypatch
    )
    quadratic_step = investigation.quadratic_step

    def slow_quadratic_step(*args):
        result = quadratic_step(*args)
        clock.now = 0.0165
        return result

    monkeypatch.setattr(investigation, "quadratic_step", slow_quadratic_step)
    solver.evaluate = lambda *_: pytest.fail("no time for a line-search evaluation")
    outcome = solver._optimize_plan(
        *arguments, None, None, reference, None, None, budget=budget
    )
    np.testing.assert_allclose(outcome.blocks, [[0.7]])
    assert solver.reports[-1]["stop_reason"] == "time_budget"


def test_time_budget_does_not_start_seed_work_that_cannot_finish(
    investigation, monkeypatch
):
    solver, reference, _, clock, budget = budgeted_problem(investigation, monkeypatch)
    clock.now = 0.019
    solver.evaluate = lambda *_: pytest.fail("no time for seed evaluation")
    with pytest.raises(_SolveAbort, match="insufficient SQP seed budget"):
        solver._seed_plan(
            jnp.zeros((1, 1)), None, None, None, reference, None, None, budget=budget
        )
    assert solver._seed_derivative is None


def test_fused_output_preserves_cost_gradient_prediction_and_constraints(
    investigation, small_controller
):
    plan = small_controller.plan
    solver = investigation.GaussNewtonReference(plan, plan.policy)
    state = jnp.asarray(resting_state()).at[3].set(0.1)
    previous = jnp.full(4, 0.5)
    blocks = jnp.full((2, 4), -0.2)
    reference = small_controller.hold_reference(jnp.asarray(resting_state())).states
    exogenous = jnp.zeros((2, 0))
    context = (state, previous, reference, previous, exogenous, plan.values)
    (value, (prediction, measurements, margins)), gradient = solver.finalize(
        blocks, *context
    )
    expected_value, expected_gradient = solver._kernels.objective_and_gradient(
        blocks, *context
    )
    expected_prediction = plan.rollout(blocks, state, previous, exogenous, plan.values)
    expected_measurements = plan.measure(expected_prediction)
    expected_margins = plan.optimization_terms(
        expected_prediction, reference, previous, plan.policy
    ).inequality_margins
    np.testing.assert_allclose(value, expected_value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-5)
    for actual, expected in zip(prediction, expected_prediction):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        measurements, expected_measurements, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(margins, expected_margins, rtol=2e-6, atol=2e-6)

    derivative, (terms, prepared) = solver.linearize(blocks, *context)
    prepared_gradient = (
        2 * np.asarray(derivative[0]).reshape(-1, blocks.size).T @ np.asarray(terms[0])
    )
    np.testing.assert_allclose(
        prepared_gradient.reshape(blocks.shape), expected_gradient, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(prepared[2], expected_value, rtol=2e-6, atol=2e-6)
    for actual, expected in zip(prepared[0], expected_prediction):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_fused_output_rechecks_the_returned_plans_constraints(investigation):
    solver = reference_problem(investigation)
    solver.fused_output = True
    solver.finalize = lambda blocks, *_: (
        (jnp.sum(blocks**2), (None, None, jnp.asarray([-0.01]))),
        2 * blocks,
    )
    with pytest.raises(_SolveAbort, match="final SQP prediction is not feasible"):
        solver._optimize_plan(
            jnp.ones((1, 1)),
            jnp.asarray(1.0),
            jnp.full((1, 1), 2.0),
            None,
            None,
            SimpleNamespace(states=None),
            None,
            None,
        )
    assert solver.reports[-1]["feasible"]
    assert not solver.reports[-1]["finalization_feasible"]


@pytest.mark.parametrize("finite_checkpoint", [True, False])
def test_prepared_checkpoint_needs_no_final_kernel_when_output_time_runs_short(
    investigation, monkeypatch, finite_checkpoint
):
    from glassbox.control.plan import PlanMeasurements, Prediction

    solver, reference, arguments, clock, budget = budgeted_problem(
        investigation, monkeypatch
    )
    prediction = Prediction(
        mean_states=jnp.ones((2, 13)) * (1.0 if finite_checkpoint else jnp.nan),
        tangent_covariance=jnp.zeros((1, 12, 12)),
        commands=arguments[0],
        latent_states=jnp.zeros((2, 1)),
        exogenous=jnp.empty((1, 0)),
    )
    solver._seed_prediction = (
        prediction,
        PlanMeasurements(0.5, 0.0, 0.0),
        arguments[1],
    )
    evaluate = solver.evaluate

    def slow_evaluation(*args):
        values = evaluate(*args)
        clock.now = 0.0175
        return values

    solver.evaluate = slow_evaluation
    solver._kernels = SimpleNamespace(
        objective_and_gradient=lambda *_: pytest.fail(
            "must return the ready checkpoint"
        )
    )
    if finite_checkpoint:
        outcome = solver._optimize_plan(
            *arguments, None, None, reference, None, None, budget=budget
        )
        assert outcome.evaluation is not None
        assert outcome.evaluation.nonlinear_feasibility.status == "feasible"
        assert outcome.evaluation.nonlinear_feasibility.constraint_count == 1
        np.testing.assert_allclose(outcome.blocks, [[0.7]])
        np.testing.assert_allclose(outcome.gradient, [[1.4]])
        assert solver.reports[-1]["output_source"] == "linearization_checkpoint"
        assert solver.reports[-1]["stop_reason"] == "time_budget"
        assert solver._seed_prediction is None
    else:
        with pytest.raises(_SolveAbort, match="insufficient SQP output budget"):
            solver._optimize_plan(
                *arguments, None, None, reference, None, None, budget=budget
            )


@pytest.mark.parametrize("prepared", [True, False])
def test_normal_sqp_result_exposes_only_returned_prediction_feasibility(
    investigation, small_controller, prepared
):
    from glassbox.control.plan import SolveStatus

    plan = small_controller.plan
    solver = investigation.GaussNewtonReference(
        plan,
        replace(plan.policy, allow_unresolved_parameters=True),
        prepared_checkpoints=prepared,
    )
    state = jnp.asarray(resting_state())
    previous = jnp.full(4, 0.5)
    reference = small_controller.hold_reference(state)
    result = solver.solve(state, reference, previous, applied_command=previous)
    assert result.command_usable
    assert result.status is SolveStatus.STALLED
    assert result.deadline_met is None
    assessment = result.nonlinear_feasibility
    assert assessment.status == "feasible"
    blocks = solver._normalized_from_commands(result.predicted_commands)
    prediction = plan.rollout(blocks, state, previous, jnp.empty((2, 0)), plan.values)
    margins = np.asarray(
        plan.optimization_terms(
            prediction, reference.states, previous, plan.policy
        ).inequality_margins
    )
    assert assessment.constraint_count == margins.size
    assert assessment.maximum_violation == pytest.approx(
        max(-np.min(margins), 0.0), abs=1e-7
    )
    np.testing.assert_allclose(prediction.mean_states, result.predicted_states)
    np.testing.assert_array_equal(result.warm_start.commands, result.predicted_commands)


def test_normal_sqp_infeasible_request_does_not_certify_fallback(
    investigation, small_controller
):
    plan = small_controller.plan
    solver = investigation.GaussNewtonReference(
        plan, replace(plan.policy, allow_unresolved_parameters=True)
    )
    state = jnp.asarray(resting_state()).at[3].set(1.0)
    previous = jnp.full(4, 0.5)
    result = solver.solve(state, small_controller.hold_reference(state), previous)
    assert not result.command_usable
    assert result.nonlinear_feasibility.status == "not_assessed"
    assert result.warm_start is None


def precision_problem(
    investigation,
    monkeypatch,
    *,
    objective_dtype=np.float32,
    scale=1.0,
    margin=0.1,
    checkpoint=True,
    step=-0.25,
    success=True,
):
    """A prepared synthetic point; expensive kernels are replaced by fixed values.

    QP steps are deliberately supplied so each stopping guard can be falsified
    independently. The unchanged line search still owns trial acceptance.
    """
    from glassbox.control.plan import PlanMeasurements, Prediction

    solver = reference_problem(investigation)
    solver.precision_stopping = True
    solver._iteration_budget = 1
    blocks = jnp.asarray([[0.5]])
    residuals = np.asarray([np.sqrt(scale)], dtype=float)
    derivative = (np.asarray([[[np.sqrt(scale) * 2**-26]]]), np.zeros((1, 1, 1)))
    values = (residuals, np.asarray([margin]))

    def prediction(current):
        return (
            Prediction(
                jnp.ones((2, 13)),
                jnp.zeros((1, 12, 12)),
                current,
                jnp.zeros((2, 1)),
                jnp.empty((1, 0)),
            ),
            PlanMeasurements(0.5, 0.0, 0.0),
            np.asarray(scale, dtype=objective_dtype),
        )

    solver._seed_derivative = derivative, values
    solver._seed_prediction = prediction(blocks) if checkpoint else None
    solver.linearize = lambda current, *_: (derivative, (values, prediction(current)))
    calls = []

    def evaluate(current, *_):
        calls.append(np.asarray(current).copy())
        return values

    solver.evaluate = evaluate
    monkeypatch.setattr(
        investigation,
        "quadratic_step",
        lambda *_: (np.asarray([step]), np.ones(1), True, success),
    )
    arguments = (
        blocks,
        jnp.asarray(scale),
        jnp.asarray([[scale * 2**-25]]),
        None,
        None,
        SimpleNamespace(states=None),
        None,
        None,
    )
    return solver, arguments, calls


@pytest.mark.parametrize("objective_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [2.0**-10, 1.0, 2.0**10])
def test_precision_resolution_uses_objective_dtype_and_scales_with_cost(
    investigation, monkeypatch, objective_dtype, scale
):
    solver, arguments, calls = precision_problem(
        investigation, monkeypatch, objective_dtype=objective_dtype, scale=scale
    )
    outcome = solver._optimize_plan(*arguments)
    stopped = objective_dtype == np.float32
    assert (solver.reports[-1]["stop_reason"] == "model_resolution") == stopped
    assert len(calls) == (1 if stopped else 12)
    if stopped:
        evidence = solver.reports[-1]["precision_stop"]
        assert evidence["rejected_trial"] == 0
        assert evidence["objective_dtype"] == "float32"
        assert evidence["objective_resolution"] == float(np.spacing(np.float32(scale)))
        assert 0 <= evidence["linear_decrease_bound"] < evidence["objective_resolution"]
    assert not outcome.converged and outcome.stalled
    np.testing.assert_array_equal(outcome.blocks, arguments[0])
    assert outcome.evaluation.nonlinear_feasibility.status == "feasible"


@pytest.mark.parametrize(
    "guard", ["repair", "missing_checkpoint", "failed_qp", "clipped_ray", "non_descent"]
)
def test_model_resolution_cannot_suppress_work_when_guard_is_missing(
    investigation, monkeypatch, guard
):
    solver, arguments, calls = precision_problem(
        investigation,
        monkeypatch,
        # Numerically feasible within tolerance, yet requires genuine repair.
        margin=-0.5 * investigation.FEASIBILITY_TOLERANCE if guard == "repair" else 0.1,
        checkpoint=guard != "missing_checkpoint",
        success=guard != "failed_qp",
        step=-2.0
        if guard == "clipped_ray"
        else (0.25 if guard == "non_descent" else -0.25),
    )
    outcome = solver._optimize_plan(*arguments)
    assert calls
    assert solver.reports[-1]["stop_reason"] != "model_resolution"
    assert not outcome.converged


def test_model_resolution_requires_checkpoint_at_current_iterate(
    investigation, monkeypatch
):
    solver, arguments, calls = precision_problem(investigation, monkeypatch)
    solver._iteration_budget = 2
    steps = iter((0.125, -0.125))
    monkeypatch.setattr(
        investigation,
        "quadratic_step",
        lambda *_: (np.asarray([next(steps)]), np.ones(1), True, True),
    )
    # The non-descent first step passes the original nonincrease Armijo test.
    # Its equal cost leaves the older checkpoint at .5, while current is .625.
    outcome = solver._optimize_plan(*arguments)
    assert len(calls) >= 2
    assert calls[0][0, 0] == 0.625
    assert solver.reports[-1]["stop_reason"] != "model_resolution"
    assert not outcome.converged


def test_precision_stop_cannot_return_infeasible_seed(investigation, monkeypatch):
    solver, arguments, calls = precision_problem(
        investigation, monkeypatch, margin=-0.1
    )
    with pytest.raises(_SolveAbort, match="no feasible command plan"):
        solver._optimize_plan(*arguments)
    assert calls
    assert not solver.reports[-1]["feasible"]
    assert solver.reports[-1]["stop_reason"] != "model_resolution"


def test_representable_stop_compares_actual_converted_input(investigation, monkeypatch):
    with jax.enable_x64(False):
        ulp = float(np.spacing(np.float32(0.5)))
        solver, arguments, calls = precision_problem(
            investigation, monkeypatch, step=0.25 * ulp
        )
        # Host arithmetic does move; the actual float32 evaluator input does not.
        assert np.float64(0.5) + 0.25 * ulp != np.float64(0.5)
        outcome = solver._optimize_plan(*arguments)
        assert not calls
        assert solver.reports[-1]["stop_reason"] == "representable_step"
        assert solver.reports[-1]["precision_stop"]["trial"] == 0
        assert not outcome.converged and outcome.stalled
        np.testing.assert_array_equal(outcome.blocks, arguments[0])
        assert outcome.evaluation.nonlinear_feasibility.status == "feasible"


def test_repeated_intermediate_trial_must_continue_backtracking(
    investigation, monkeypatch
):
    with jax.enable_x64(False):
        ulp = float(np.spacing(np.float32(0.5)))
        solver, arguments, calls = precision_problem(
            investigation, monkeypatch, step=1.1 * ulp
        )
        original = solver.evaluate

        def rejected(current, *context):
            original(current, *context)
            return np.asarray([2.0]), np.asarray([0.1])

        solver.evaluate = rejected
        outcome = solver._optimize_plan(*arguments)
        # alpha1 and alpha1/2 map to the same non-base float32 point. Only
        # alpha1/4 reaches the base point and permits representable stopping.
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0], calls[1])
        assert calls[0][0, 0] != 0.5
        assert solver.reports[-1]["precision_stop"]["trial"] == 2
        assert solver.reports[-1]["stop_reason"] == "representable_step"
        assert not outcome.converged


def test_representable_stop_with_no_feasible_candidate_rejects(
    investigation, monkeypatch
):
    with jax.enable_x64(False):
        step = 0.25 * float(np.spacing(np.float32(0.5)))
        solver, arguments, calls = precision_problem(
            investigation, monkeypatch, step=step, margin=-0.1
        )
        with pytest.raises(_SolveAbort, match="no feasible command plan"):
            solver._optimize_plan(*arguments)
        assert not calls
        assert solver.reports[-1]["stop_reason"] == "representable_step"
        assert not solver.reports[-1]["feasible"]


def test_precision_stopping_remains_opt_in(investigation, monkeypatch):
    solver, arguments, calls = precision_problem(investigation, monkeypatch)
    solver.precision_stopping = False
    outcome = solver._optimize_plan(*arguments)
    assert calls
    assert not solver.reports[-1]["precision_stopping_enabled"]
    assert solver.reports[-1]["precision_stop"] is None
    assert not outcome.converged


def test_unusable_qp_is_not_reclassified_as_precision_stop(investigation, monkeypatch):
    solver, arguments, calls = precision_problem(investigation, monkeypatch)
    monkeypatch.setattr(
        investigation,
        "quadratic_step",
        lambda *_: (np.zeros(1), np.ones(1), False, False),
    )
    outcome = solver._optimize_plan(*arguments)
    assert not calls
    assert solver.reports[-1]["stop_reason"] == "quadratic_step_failed"
    assert solver.reports[-1]["precision_stop"] is None
    assert not outcome.converged


def test_model_resolution_gives_successful_full_step_original_acceptance(
    investigation, monkeypatch
):
    solver, arguments, calls = precision_problem(investigation, monkeypatch)
    original = solver.evaluate

    def improved(current, *context):
        original(current, *context)
        return np.asarray([0.5]), np.asarray([0.1])

    solver.evaluate = improved
    outcome = solver._optimize_plan(*arguments)
    # Its GN decrease bound qualifies for the heuristic, but the checked full
    # step actually improves the nonlinear cost and must get Armijo priority.
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [[0.25]])
    np.testing.assert_array_equal(outcome.blocks, [[0.25]])
    assert solver.reports[-1]["stop_reason"] == "iteration_limit"
    assert solver.reports[-1]["precision_stop"] is None
    assert not outcome.converged


@pytest.mark.parametrize(
    "trial_kind",
    [
        "infeasible",
        "within_tolerance_violation",
        "nonfinite_residual",
        "nonfinite_margin",
    ],
)
def test_model_resolution_does_not_stop_after_unqualified_rejected_trial(
    investigation, monkeypatch, trial_kind
):
    solver, arguments, calls = precision_problem(investigation, monkeypatch)
    original = solver.evaluate

    def rejected(current, *context):
        original(current, *context)
        residual = np.nan if trial_kind == "nonfinite_residual" else 1.0
        margin = {
            "infeasible": -0.1,
            "within_tolerance_violation": -0.5 * investigation.FEASIBILITY_TOLERANCE,
            "nonfinite_residual": 0.1,
            "nonfinite_margin": np.nan,
        }[trial_kind]
        return np.asarray([residual]), np.asarray([margin])

    solver.evaluate = rejected
    outcome = solver._optimize_plan(*arguments)
    assert len(calls) == 12
    assert solver.reports[-1]["stop_reason"] == "line_search_stalled"
    assert solver.reports[-1]["precision_stop"] is None
    np.testing.assert_array_equal(outcome.blocks, arguments[0])
    assert outcome.evaluation.nonlinear_feasibility.status == "feasible"
    assert not outcome.converged
