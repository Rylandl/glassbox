"""Check the arithmetic and feasibility contracts of the SQP experiment."""

import importlib
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


def test_residual_form_preserves_objective_and_gradient_with_uncertainty(investigation):
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
        return jnp.concatenate((blocks.ravel(), blocks.ravel() - 0.5)) * (
            jnp.nan if nonfinite else 1
        )

    def with_aux(*args):
        values = packed(*args)
        return values, values

    solver = object.__new__(investigation.GaussNewtonReference)
    solver.model = SimpleNamespace(values=None)
    solver._kernels = SimpleNamespace(
        objective_and_gradient=jax.jit(
            jax.value_and_grad(lambda blocks, *_: jnp.sum(blocks**2))
        )
    )
    solver.reports = []
    solver.constraint_count = 1
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
