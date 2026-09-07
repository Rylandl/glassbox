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
        (0.0, 0.4, 0.4, True),  # Less violation while still infeasible.
        (0.6, float("nan"), 0.6, False),
        (float("nan"), 0.6, 0.6, True),
    ],
)
def test_sqp_seed_selection_uses_feasibility_before_cost(
    investigation, cold, warm, expected, warm_used
):
    solver = reference_problem(investigation)
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
