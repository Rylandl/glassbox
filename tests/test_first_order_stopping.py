"""One stopping-rule change, with historical replay and safeguards preserved."""

import copy
import json

import numpy as np
import pytest
import test_quasi_newton as optimizer_tests
import test_quasi_newton_qualification as previous_tests

from glassbox.experimental import first_order_qualification as q
from glassbox.experimental import quasi_newton as optimizer


def test_only_backend_cutoff_changes(monkeypatch):
    def backend(fun, x0, **kwargs):
        assert kwargs["method"] == "L-BFGS-B" and kwargs["jac"] is True
        assert kwargs["options"] == dict(
            maxiter=64, maxls=16, maxcor=10, ftol=0.0, gtol=0.002, maxfun=1024
        )
        return optimizer_tests.fake_result(x0, nit=0)

    monkeypatch.setattr(optimizer, "minimize", backend)
    objective, _ = optimizer_tests.quadratic()
    seed = np.zeros((1, 2), np.float32)
    value, gradient = objective(seed)
    optimizer.bounded_minimize(
        objective, seed, value, gradient, optimizer_tests.POLICY, _ftol=0.0
    )
    assert optimizer_tests.POLICY.relative_improvement_tolerance == 1e-5
    assert q.FirstOrderSolver._ftol == 0.0
    assert optimizer.QuasiNewtonSolver._ftol is None


@pytest.mark.parametrize("override", [False, 0, 1e-6, -1.0, np.nan, "0"])
def test_private_override_is_not_a_tolerance_menu(override):
    seed = np.zeros((1, 2), np.float32)
    objective, _ = optimizer_tests.quadratic()
    value, gradient = objective(seed)
    with pytest.raises(ValueError, match="frozen zero"):
        optimizer.bounded_minimize(
            objective, seed, value, gradient, optimizer_tests.POLICY, _ftol=override
        )


def test_removing_positive_cutoff_improves_an_offset_quadratic():
    def objective(x):
        error = np.asarray(x) - np.array([[0.7, -0.4]], np.float32)
        weights = np.array([[1.0, 0.01]], np.float32)
        return np.float32(100) + np.sum(weights * error**2), 2 * weights * error

    seed = np.zeros((1, 2), np.float32)
    value, gradient = objective(seed)
    old, old_work = optimizer.bounded_minimize(
        objective, seed, value, gradient, optimizer_tests.POLICY
    )
    new, new_work = optimizer.bounded_minimize(
        objective, seed, value, gradient, optimizer_tests.POLICY, _ftol=0.0
    )
    assert not old.converged and new.converged
    assert "RELATIVE REDUCTION" in old_work["backend_message"]
    assert float(new.value) <= float(old.value)
    assert new_work["independent_projected_gradient_inf_norm"] <= 0.002
    assert new_work["accepted_iterations"] > old_work["accepted_iterations"]


def test_zero_cutoff_does_not_turn_an_equal_objective_exit_into_convergence(
    monkeypatch,
):
    def backend(fun, x0, **kwargs):
        fun(x0)
        fun(np.array([0.1, 0.1]))
        kwargs["callback"](x0)
        return optimizer_tests.fake_result(
            x0, message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
        )

    monkeypatch.setattr(optimizer, "minimize", backend)
    seed = np.zeros((1, 2), np.float32)

    def objective(x):
        return np.float32(100), np.full((1, 2), 0.01, np.float32)

    outcome, work = optimizer.bounded_minimize(
        objective, seed, *objective(seed), optimizer_tests.POLICY, _ftol=0.0
    )
    assert not outcome.converged
    assert work["backend_success"] and work["returned_seed"]
    np.testing.assert_array_equal(outcome.blocks, seed)


@pytest.mark.parametrize(
    "case",
    [
        "test_hard_budget_retains_best_canonical_evaluated_point",
        "test_nonfinite_evaluation_remains_explicit_while_finite_seed_is_retained",
        "test_nonfinite_backend_proposal_is_an_explicit_failed_origin",
        "test_backend_abnormal_stop_is_not_silently_successful",
        "test_independent_audit_rejects_objective_drift",
    ],
)
def test_zero_cutoff_preserves_inherited_safeguards(monkeypatch, case):
    original = optimizer.bounded_minimize
    monkeypatch.setattr(
        optimizer,
        "bounded_minimize",
        lambda *args, **kwargs: original(*args, **kwargs, _ftol=0.0),
    )
    getattr(optimizer_tests, case)(monkeypatch)


def fixture():
    _, trials, inputs, work = previous_tests.fixture()
    plan, _ = q.frozen_plan()
    inputs = {"inputs/" + key: value for key, value in inputs.items()}
    return plan, trials, inputs, work


def test_frozen_roster_sources_and_single_option_change():
    plan, raw = q.frozen_plan()
    assert q.digest(raw) == q.PLAN_SHA256
    assert len(plan["input_sha256"]) == 67
    assert len(plan["source_sha256"]) == 83
    assert plan["comparison"]["candidate_options"] == plan["comparison"][
        "baseline_options"
    ] | {"ftol": 0.0}


def test_raw_backend_gradient_exit_is_separate_from_returned_point_convergence():
    plan, trials, inputs, work = fixture()
    trials[0]["projected_gradient_inf_norm"][0, 1] = 0.01
    work[0]["independent_projected_gradient_inf_norm"] = 0.01
    work[0]["independent_gradient"] = np.full((5, 3), 0.01).tolist()
    work[0]["backend_message"] = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
    result = q.report_from_inputs(plan, trials, inputs, work)
    assert result["qualification"]["converged_origins"] == 127
    assert (
        result["raw_backend_termination"]["pooled"][
            "raw_gradient_exit_with_nonstationary_return_count"
        ]
        == 1
    )
    assert result["plan_sha256"] == q.PLAN_SHA256


def test_relative_exit_report_does_not_claim_a_float32_plateau():
    plan, trials, inputs, work = fixture()
    work[0]["backend_message"] = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
    result = q.report_from_inputs(plan, trials, inputs, work)
    assert (
        result["raw_backend_termination"]["pooled"][
            "nonpositive_relative_decrease_count"
        ]
        == 1
    )


def test_historical_work_requires_exact_canonical_arrays():
    _, _, _, work = fixture()
    changed = copy.deepcopy(work[0])
    changed["returned_canonical_blocks"][0][0] += 1e-10
    with pytest.raises((ValueError, AssertionError)):
        q.historical_work_parity(changed, work[0])


@pytest.mark.parametrize("field", ["qualification", "commands", "work"])
def test_shared_verifier_rejects_forged_artifacts(tmp_path, monkeypatch, field):
    plan, trials, inputs, work = fixture()
    _, raw = q.frozen_plan()
    result = q.report_from_inputs(plan, trials, inputs, work)
    plan = plan | {
        "input_sha256": {name: q.digest(value) for name, value in inputs.items()}
    }
    monkeypatch.setattr(q, "frozen_plan", lambda: (plan, raw))
    monkeypatch.setattr(q, "check_environment", lambda _: {})
    monkeypatch.setattr(q, "probe", lambda *_: (trials, work, result))
    parent, output = tmp_path / "parent", tmp_path / "run"
    for name, value in inputs.items():
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    q.run(parent, output)
    assert q.verify(output)["verified_pairs"] == 128
    hashes = json.loads((output / "files.json").read_text())
    altered = copy.deepcopy(result)
    if field == "qualification":
        altered["qualification"]["eligible_for_future_tracking_experiment"] = False
    elif field == "commands":
        values = {k: v.copy() for k, v in trials[0].items()}
        values["commands"][0, 1, -1, 0] += 0.1
        np.savez_compressed(output / "trial-0/paired.npz", **values)
    else:
        values = copy.deepcopy(work)
        values[0]["new_objective_evaluations"] += 1
        q.parent.write_json(output / "work.json", values)
        altered = q.report_from_inputs(plan, trials, inputs, values)
    q.parent.write_json(output / "report.json", altered)
    for name in ("report.json", "work.json", "trial-0/paired.npz"):
        hashes[name] = q.digest((output / name).read_bytes())
    q.parent.write_json(output / "files.json", hashes)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
