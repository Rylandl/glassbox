"""Precision isolation and exact seed reuse on analytic/fake equations only."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_quasi_newton as optimizer_tests
import test_task_qualification as task_fixture

from glassbox.control.plan import ReferenceTrajectory
from glassbox.experimental import precision_solver as q
from glassbox.experimental import quasi_newton as optimizer
from glassbox.experimental.first_order_qualification import FirstOrderSolver

learned = task_fixture.learned


def fixture(learned):
    arm = task_fixture.make_arm(learned, task_fixture.q.CANDIDATE)
    arm.reset(task_fixture.INITIAL, task_fixture.COMMAND)
    model = arm.plan.with_causal_state(arm._state)
    policy = replace(arm.policy, maximum_iterations=64)
    reference = ReferenceTrajectory.hold(task_fixture.INITIAL, 5)
    return model, policy, reference


def factory():
    return q.PrecisionFactory(
        {"coefficient": np.array([0.1], np.float32)},
        _environment={"gravity": np.array([0, 0, 9.80665], np.float32)},
        _equations_factory=lambda model, environment: task_fixture.Equations(),
    )


def baseline(learned):
    model, policy, reference = fixture(learned)
    solver = q.SeedCaptureSolver(model, policy)
    result = solver.solve(task_fixture.INITIAL, reference, task_fixture.COMMAND)
    assert not result.used_fallback
    return model, policy, solver, result


def test_capture_preserves_original_result_work_and_actual_jit_inputs(learned):
    model, policy, capture_solver, captured_result = baseline(learned)
    original = FirstOrderSolver(model, policy)
    original_result = original.solve(
        task_fixture.INITIAL,
        ReferenceTrajectory.hold(task_fixture.INITIAL, 5),
        task_fixture.COMMAND,
    )
    assert capture_solver.last_work == original.last_work
    for name in ("command", "predicted_commands", "predicted_states"):
        np.testing.assert_array_equal(
            getattr(captured_result, name), getattr(original_result, name)
        )
    capture = capture_solver.last_capture
    assert np.asarray(model.values.parameters["params"]["memory"]).dtype == np.float64
    for leaf in jax.tree.leaves(capture.runtime()):
        if np.asarray(leaf).dtype.kind == "f":
            assert np.asarray(leaf).dtype == np.float32
    np.testing.assert_array_equal(
        capture.state, task_fixture.INITIAL.astype(np.float32)
    )
    assert capture.record()["inputs"]["values"]["leaves"]
    assert not jax.config.x64_enabled


def test_float64_candidate_and_common_audits_use_exact_captured_seed(learned):
    model, policy, solver, result32 = baseline(learned)
    f = factory()
    result, work, audits, evidence = f.solve(
        model,
        policy,
        solver.last_capture,
        solver.last_work["returned_canonical_blocks"],
    )
    assert not result.used_fallback
    assert work["seed_objective_evaluations"] == 1
    assert work["audit_objective_evaluations"] == 1
    assert evidence["compile_signature"] != model.compile_signature
    assert evidence["floating_dtype"] == "float64"
    assert evidence["jaxpr"]["float32_arithmetic"] == 0
    assert evidence["jaxpr"]["jaxprs"] > 1
    assert np.asarray(result.predicted_states).dtype == np.float64
    assert np.asarray(result32.predicted_states).dtype == np.float32
    assert work["independent_objective"] == audits["candidate"]["value"]
    assert work["independent_gradient"] == audits["candidate"]["gradient"]
    assert (
        work["independent_projected_gradient_inf_norm"]
        == audits["candidate"]["residual"]
    )
    assert audits["baseline"]["blocks"] == solver.last_work["returned_canonical_blocks"]
    assert not jax.config.x64_enabled
    assert len(f._dtype_cache) == 1
    with jax.enable_x64(True):
        model64 = f._model(model, solver.last_capture)
        candidate = q.PrecisionSolver(model64, policy, solver.last_capture)
        assert candidate._kernels is not solver._kernels
        seed_value, _ = candidate._kernels.objective_and_gradient(
            q._lift(solver.last_capture.blocks),
            *q._lift(solver.last_capture.runtime()),
        )
        assert result.diagnostics.initial_objective == float(seed_value)


def test_warm_seed_is_not_reranked_or_renormalized(learned, monkeypatch):
    model, policy, _, previous = baseline(learned)
    solver = q.SeedCaptureSolver(model, policy)
    solver.solve(
        task_fixture.INITIAL,
        ReferenceTrajectory.hold(task_fixture.INITIAL, 5),
        task_fixture.COMMAND,
        warm_start=previous.warm_start,
    )
    capture = solver.last_capture
    assert solver.last_work["seed_objective_evaluations"] == 2
    assert capture.warm_start_used
    seen = []

    def backend(fun, x0, **kwargs):
        seen.append(x0.copy())
        assert kwargs["options"] == dict(
            maxiter=64, maxls=16, maxcor=10, ftol=0.0, gtol=0.002, maxfun=1024
        )
        fun(x0)
        return optimizer_tests.fake_result(x0, nit=0)

    monkeypatch.setattr(optimizer, "minimize", backend)
    result, work, _, _ = factory().solve(
        model, policy, capture, solver.last_work["returned_canonical_blocks"]
    )
    np.testing.assert_array_equal(
        seen[0], np.asarray(capture.blocks, np.float64).ravel()
    )
    assert result.diagnostics.warm_start_used
    assert work["seed_objective_evaluations"] == 1
    assert work["cached_seed_requests"] == 1
    assert work["new_objective_evaluations"] == 0
    assert not jax.config.x64_enabled


def test_restores_default_precision_when_backend_raises(learned, monkeypatch):
    model, policy, solver, _ = baseline(learned)

    def crash(*args, **kwargs):
        assert jax.config.x64_enabled
        raise RuntimeError("backend programming error")

    monkeypatch.setattr(optimizer, "minimize", crash)
    with pytest.raises(RuntimeError, match="programming error"):
        factory().solve(
            model,
            policy,
            solver.last_capture,
            solver.last_work["returned_canonical_blocks"],
        )
    assert not jax.config.x64_enabled


def test_snapshots_cached_model_and_lazy_scales_before_lifting(learned):
    model, _, solver, _ = baseline(learned)
    f = factory()
    capture = solver.last_capture
    with jax.enable_x64(True):
        model64 = f._model(model, capture)
        np.testing.assert_array_equal(
            model64.command_minimum, np.asarray(capture.command_minimum, np.float64)
        )
        np.testing.assert_array_equal(
            model64.tolerances.local_state_scale,
            np.asarray(capture.local_state_scale, np.float64),
        )
        np.testing.assert_array_equal(
            f.model64["coefficient"], np.array([0.1], np.float32).astype(np.float64)
        )
        assert model64.sample_period_s == model.sample_period_s == 0.05
        assert model64.policy == model.policy
    assert not jax.config.x64_enabled


def test_nested_jaxpr_rejects_float32_arithmetic_and_runtime_inputs():
    with jax.enable_x64(True):
        bad = jax.jit(
            lambda x: jax.lax.scan(
                lambda a, _: (a + jnp.float32(0.1), a),
                x.astype(jnp.float32),
                None,
                length=2,
            )[0].astype(jnp.float64)
        )
        graph = jax.make_jaxpr(bad)(jnp.ones(2, dtype=jnp.float64))
        with pytest.raises(ValueError, match="lower precision arithmetic"):
            q.audit_jaxpr_dtypes(graph)
        graph = jax.make_jaxpr(lambda x: x.astype(jnp.float64))(
            jnp.ones(2, dtype=jnp.float32)
        )
        with pytest.raises(ValueError, match="runtime input"):
            q.audit_jaxpr_dtypes(graph)


def test_nested_jaxpr_allows_only_exact_static_sign_conversion():
    sign = jnp.array([1.0, -1.0, -1.0], dtype=jnp.float32)
    with jax.enable_x64(True):
        graph = jax.make_jaxpr(jax.jit(lambda x: x * sign))(
            jnp.ones(3, dtype=jnp.float64)
        )
        evidence = q.audit_jaxpr_dtypes(graph)
        assert evidence["static_float32_to_float64_conversions"] == 1
        assert evidence["float32_arithmetic"] == 0
    inexact = jnp.array([0.1], dtype=jnp.float32)
    with jax.enable_x64(True):
        graph = jax.make_jaxpr(jax.jit(lambda x: x * inexact))(
            jnp.ones(1, dtype=jnp.float64)
        )
        with pytest.raises(ValueError, match="non-exact"):
            q.audit_jaxpr_dtypes(graph)


def test_offset_objective_preserves_subfloat32_changes_with_same_backend(monkeypatch):
    def backend(fun, x0, **kwargs):
        fun(np.array([0.1]))
        kwargs["callback"](np.array([0.1]))
        return optimizer_tests.fake_result([0.1])

    monkeypatch.setattr(optimizer, "minimize", backend)

    def run(dtype):
        def objective(blocks):
            assert np.asarray(blocks).dtype == dtype
            error = np.asarray(blocks) - dtype(0.1)
            return dtype(1e6) + np.sum(error**2), 2 * error

        seed = np.zeros((1, 1), dtype)
        return optimizer.bounded_minimize(
            objective, seed, *objective(seed), optimizer_tests.POLICY, _ftol=0.0
        )

    with jax.enable_x64(True):
        double, double_work = run(np.float64)
    single, single_work = run(np.float32)
    assert double_work["independent_projected_gradient_inf_norm"] <= 0.002
    assert (
        double_work["independent_projected_gradient_inf_norm"]
        < single_work["independent_projected_gradient_inf_norm"]
    )
    assert np.asarray(double.blocks).dtype == np.float64
    assert np.asarray(single.blocks).dtype == np.float32


def test_factory_and_capture_refuse_global_x64_context(learned):
    model, policy, reference = fixture(learned)
    with jax.enable_x64(True):
        with pytest.raises(ValueError, match="float32"):
            factory()
        with pytest.raises(ValueError, match="float32"):
            q.SeedCaptureSolver(model, policy).solve(
                task_fixture.INITIAL, reference, task_fixture.COMMAND
            )


@pytest.mark.parametrize("failure", ["budget", "nonfinite", "abnormal"])
def test_float64_preserves_hard_budget_and_explicit_failure(failure, monkeypatch):
    monkeypatch.setattr(optimizer, "MAXIMUM_EVALUATIONS", 2)

    def objective(x):
        x = np.asarray(x)
        assert x.dtype == np.float64
        if failure == "nonfinite" and np.any(x > 0.15):
            return np.float64(np.nan), np.full_like(x, np.nan)
        return np.sum((x - 0.7) ** 2), 2 * (x - 0.7)

    def backend(fun, x0, **kwargs):
        fun(np.array([0.1]))
        fun(np.array([0.2]))
        if failure == "budget":
            fun(np.array([0.3]))
        return optimizer_tests.fake_result(
            [0.2], status=2, success=False, message="ABNORMAL", nit=0
        )

    monkeypatch.setattr(optimizer, "minimize", backend)
    with jax.enable_x64(True):
        seed = np.zeros((1, 1), np.float64)
        outcome, work = optimizer.bounded_minimize(
            objective,
            seed,
            *objective(seed),
            optimizer_tests.POLICY,
            _ftol=0.0,
        )
    assert work["new_objective_evaluations"] == 2
    assert work["audit_objective_evaluations"] == 1
    np.testing.assert_array_equal(
        outcome.blocks, [[0.1 if failure == "nonfinite" else 0.2]]
    )
    assert np.asarray(outcome.blocks).dtype == np.float64
    assert not outcome.converged
    assert work["nonfinite_evaluation"] == (failure == "nonfinite")
    assert work["backend_failure"] == (failure != "budget")
    assert not jax.config.x64_enabled


def test_factory_restores_precision_when_equation_construction_raises():
    def fail(model, environment):
        assert jax.config.x64_enabled
        raise RuntimeError("constructor failure")

    with pytest.raises(RuntimeError, match="constructor failure"):
        q.PrecisionFactory({}, _environment={}, _equations_factory=fail)
    assert not jax.config.x64_enabled


@pytest.mark.cascade
def test_public_cascade_float64_planning_and_gradient_smoke(learned):
    """One synthetic reset/hold problem, independent of the128 saved origins."""
    cascade = pytest.importorskip("cascade")
    assert not jax.config.x64_enabled
    physical_model = cascade.skywalker_x8_spec().to_model()
    arm = task_fixture.q.TaskOracleArm(
        task_fixture.PLAN,
        task_fixture.CONTROL,
        learned,
        physical_model,
        task_fixture.q.CANDIDATE,
    )
    arm.reset(task_fixture.INITIAL, task_fixture.COMMAND)
    model = arm.plan.with_causal_state(arm._state)
    policy = replace(arm.policy, maximum_iterations=64)
    baseline_solver = q.SeedCaptureSolver(model, policy)
    original = baseline_solver.solve(
        task_fixture.INITIAL,
        ReferenceTrajectory.hold(task_fixture.INITIAL, 5),
        task_fixture.COMMAND,
    )
    assert not original.used_fallback
    f = q.PrecisionFactory(physical_model)
    result, work, audits, evidence = f.solve(
        model,
        policy,
        baseline_solver.last_capture,
        baseline_solver.last_work["returned_canonical_blocks"],
    )
    assert not result.used_fallback
    assert work["seed_objective_evaluations"] == 1
    assert work["independent_gradient"] == audits["candidate"]["gradient"]
    assert work["independent_objective"] == audits["candidate"]["value"]
    assert evidence["jaxpr"]["float32_arithmetic"] == 0
    assert evidence["jaxpr"]["static_float32_to_float64_conversions"] > 0
    assert evidence["model_float_leaf_count"] > 0
    assert evidence["environment_float_leaf_count"] > 0
    assert np.asarray(result.predicted_states).dtype == np.float64
    assert np.asarray(result.predicted_commands).dtype == np.float64
    assert np.asarray(f.model64.inertia_inverse).dtype == np.float64
    np.testing.assert_array_equal(
        f.model64.inertia_inverse,
        np.asarray(physical_model.inertia_inverse).astype(np.float64),
    )
    assert audits["baseline"]["bound_violation"] == 0
    assert audits["candidate"]["bound_violation"] == 0
    assert not jax.config.x64_enabled
