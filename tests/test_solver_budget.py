"""Frozen budget-only diagnostic: causal inputs and honest optimizer outcomes."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control.plan import SolverPolicy
from glassbox.control.solver import _optimize_step, _projected_gradient_norm
from glassbox.experimental import solver_budget as q


def test_frozen_protocol_pins_complete_parent_and_fixed_origins():
    plan, raw = q.frozen_plan()
    assert q.digest(raw) == q.PLAN_SHA256
    assert plan["no_fit"] and not plan["new_policy_trials"]
    assert len(plan["input_sha256"]) == 48
    assert len(plan["source_sha256"]) == 82
    assert plan["selection"]["seeds"] == [101, 102, 104, 105]
    assert plan["selection"]["origins"] == np.linspace(2, 319, 32, dtype=int).tolist()
    assert len(plan["selection"]["origins"]) * 4 == 128


def test_policy_changes_only_iteration_cap():
    baseline = SolverPolicy(horizon_steps=5, block_count=5, maximum_iterations=4)
    candidate = q.candidate_policy(baseline)
    assert candidate.maximum_iterations == 64
    expected = asdict(baseline) | {"maximum_iterations": 64}
    assert asdict(candidate) == expected
    with pytest.raises(ValueError):
        q.candidate_policy(replace(baseline, maximum_iterations=8))


def optimize_quadratic(policy):
    # Distinct curvature makes four steps insufficient, while the same
    # projected-gradient algorithm can solve this bounded problem with 64.
    weights = jnp.array([[1.0, 0.2]])
    target = jnp.array([[0.8, -0.6]])

    def objective_gradient(blocks, *_):
        error = blocks - target
        return jnp.sum(weights * error**2), 2 * weights * error

    initial = jnp.zeros((1, 2))
    value, gradient = objective_gradient(initial)
    result = _optimize_step(
        policy,
        objective_gradient,
        initial,
        value,
        gradient,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    return value, result


def test_more_budget_improves_analytic_problem_from_identical_seed():
    policy = SolverPolicy(
        horizon_steps=1,
        block_count=1,
        maximum_iterations=4,
        relative_improvement_tolerance=1e-12,
    )
    start_a, a = optimize_quadratic(policy)
    start_b, b = optimize_quadratic(q.candidate_policy(policy))
    assert float(start_a) == float(start_b)
    assert float(b[1]) < float(a[1])
    assert int(a[3]) == 4 < int(b[3]) <= 64
    assert float(_projected_gradient_norm(a[0], a[2])) > policy.gradient_tolerance
    assert float(_projected_gradient_norm(b[0], b[2])) <= policy.gradient_tolerance
    assert np.max(np.abs(b[0])) <= 1


def test_more_budget_does_not_override_stopping_rule():
    policy = SolverPolicy(
        horizon_steps=1,
        block_count=1,
        maximum_iterations=4,
        relative_improvement_tolerance=1.0,
    )
    _, a = optimize_quadratic(policy)
    _, b = optimize_quadratic(q.candidate_policy(policy))
    assert bool(a[5]) and bool(b[5])  # Explicit stalls, not convergence.
    assert not bool(a[4]) and not bool(b[4])
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)


def warm_diagnostics():
    return {
        "solve_indices": np.arange(2, 7),
        "candidate_commands": np.arange(75, dtype=float).reshape(5, 5, 3),
        "used_fallback": np.zeros(5, dtype=bool),
    }


def test_original_warm_start_uses_immediately_preceding_full_original_plan():
    diagnostics = warm_diagnostics()
    assert q.original_warm_start(diagnostics, 2) is None
    for origin in (3, 5, 7):
        warm = q.original_warm_start(diagnostics, origin)
        np.testing.assert_array_equal(
            warm.commands, diagnostics["candidate_commands"][origin - 3]
        )


def test_missing_or_failed_predecessor_cannot_silently_use_another_seed():
    diagnostics = warm_diagnostics()
    with pytest.raises(ValueError):
        q.original_warm_start(diagnostics, 9)
    diagnostics["used_fallback"][1] = True
    with pytest.raises(ValueError):
        q.original_warm_start(diagnostics, 4)


def test_input_snapshot_rejects_wrong_parent_bytes(tmp_path):
    plan, _ = q.frozen_plan()
    small = copy.deepcopy(plan)
    small["input_sha256"] = {"manifest.json": q.digest(b"expected")}
    (tmp_path / "manifest.json").write_bytes(b"forged")
    with pytest.raises(ValueError):
        q.input_snapshot(small, tmp_path)


def test_inherited_source_tampering_is_rejected(tmp_path, monkeypatch):
    plan, _ = q.frozen_plan()
    for name in plan["source_sha256"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((q.ROOT / name).read_bytes())
    (tmp_path / next(iter(plan["source_sha256"]))).write_text("# changed\n")
    monkeypatch.setattr(q, "ROOT", Path(tmp_path))
    with pytest.raises(ValueError):
        q.frozen_plan()


def result(*, final=2.0, initial=5.0, warm=False, fallback=False):
    return SimpleNamespace(
        predicted_commands=np.zeros((5, 3)),
        predicted_states=np.zeros((6, 13)),
        command=np.zeros(3),
        status="iteration_limit" if not fallback else "nonfinite",
        used_fallback=fallback,
        diagnostics=SimpleNamespace(
            initial_objective=initial,
            final_objective=final,
            final_projected_gradient_inf_norm=0.1,
            iterations=4,
            warm_start_used=warm,
            maximum_command_bound_violation=0.0,
        ),
    )


def fake_solver(monkeypatch, outcomes, calls):
    class Solver:
        def __init__(self, model, policy):
            self.policy = policy

        def solve(self, *args, **kwargs):
            calls.append((self.policy, args, kwargs))
            return outcomes[len(calls) - 1]

    monkeypatch.setattr(q, "BoundedShootingSolver", Solver)


def test_pair_preserves_common_inputs_and_checks_baseline_before_probe(monkeypatch):
    calls, checked = [], []
    a, b = result(), result(final=1.0)
    fake_solver(monkeypatch, [a, b], calls)
    state, reference, previous, warm = object(), object(), object(), object()
    policy = SolverPolicy(horizon_steps=5, block_count=5, maximum_iterations=4)

    def baseline_check(value):
        assert len(calls) == 1
        checked.append(value)

    pair = q.paired_solve(
        object(),
        policy,
        state,
        reference,
        previous,
        warm_start=warm,
        baseline_check=baseline_check,
    )
    assert pair == dict(baseline=a, candidate=b)
    assert checked == [a]
    assert calls[0][1:] == calls[1][1:]


@pytest.mark.parametrize(
    "candidate",
    [
        result(initial=5.5),
        result(warm=True),
        result(final=2.5),
    ],
)
def test_pair_rejects_unequal_seed_or_regressed_objective(monkeypatch, candidate):
    fake_solver(monkeypatch, [result(), candidate], [])
    policy = SolverPolicy(horizon_steps=5, block_count=5, maximum_iterations=4)
    with pytest.raises((AssertionError, ValueError)):
        q.paired_solve(None, policy, None, None, None)


def saved_trials(plan):
    pair = q.serialize_pair(dict(baseline=result(), candidate=result(final=1.0)))
    n = len(plan["selection"]["origins"])
    return [
        dict(
            origins=np.asarray(plan["selection"]["origins"]),
            **{
                key: np.stack([value.copy() for _ in range(n)])
                for key, value in pair.items()
            },
        )
        for _ in plan["selection"]["seeds"]
    ]


def test_report_counts_failure_in_full_denominator_and_does_not_claim_task_success():
    plan, _ = q.frozen_plan()
    trials = saved_trials(plan)
    for trial in trials:
        trial["projected_gradient_inf_norm"][:, 1] = 0.001
    trial = trials[0]
    trial["used_fallback"][0, 1] = True
    trial["statuses"][0, 1] = "nonfinite"
    trial["final_objectives"][0, 1] = np.inf
    trial["projected_gradient_inf_norm"][0, 1] = np.inf
    report = q.report(plan, trials, np.ones(3), 0.002)
    assert report["pooled"]["origins"] == 128
    assert report["pooled"]["candidate"]["failure_count"] == 1
    assert report["pooled"]["candidate"]["below_gradient_threshold"] == 127
    assert not report["every_candidate_origin_below_threshold"]
    assert not report["task_success_assessed"]
    assert report["origins"][0]["objective_reduction"] is None
    assert report["pooled"]["objective_reduction"]["count"] == 127
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "defect", ["field", "shape", "origin", "iteration", "nan", "dtype", "trial"]
)
def test_schema_rejects_incomplete_or_invalid_output(defect):
    plan, _ = q.frozen_plan()
    trials = saved_trials(plan)
    if defect == "field":
        del trials[0]["statuses"]
    elif defect == "shape":
        trials[0]["commands"] = trials[0]["commands"][:-1]
    elif defect == "origin":
        trials[0]["origins"][1] += 1
    elif defect == "iteration":
        trials[0]["iterations"][0, 1] = 65
    elif defect == "nan":
        trials[0]["final_objectives"][0, 1] = np.nan
    elif defect == "dtype":
        trials[0]["used_fallback"] = trials[0]["used_fallback"].astype(float)
    else:
        trials.pop()
    with pytest.raises((ValueError, AssertionError)):
        q.validate_arrays(plan, trials)


@pytest.mark.parametrize(
    "field",
    ["commands", "states", "iterations", "projected_gradient_inf_norm", "report"],
)
def test_saved_artifact_replay_rejects_forged_outer_hashes(
    tmp_path, monkeypatch, field
):
    plan, raw = q.frozen_plan()
    trials = saved_trials(plan)
    report = q.report(plan, trials, np.ones(3), 0.002)
    control = json.dumps(
        {"telemetry": {"command_minimum": [0, 0, 0], "command_maximum": [1, 1, 1]}}
    ).encode()
    snapshot = {"manifest.json": b"parent", "inputs/control/manifest.json": control}
    plan = plan | {
        "input_sha256": {name: q.digest(data) for name, data in snapshot.items()}
    }
    monkeypatch.setattr(q, "frozen_plan", lambda: (plan, raw))
    monkeypatch.setattr(q, "check_environment", lambda _: {})
    monkeypatch.setattr(q, "probe", lambda *_: (trials, report))
    parent = tmp_path / "parent"
    parent.mkdir()
    for name, data in snapshot.items():
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    output = tmp_path / "output"
    q.run(parent, output)
    assert q.verify(output)["verified_pairs"] == 128
    if field == "report":
        name = "report.json"
        altered = copy.deepcopy(report)
        altered["every_candidate_origin_below_threshold"] = True
        q.write_json(output / name, altered)
    else:
        name = "trial-0/paired.npz"
        altered = {k: v.copy() for k, v in trials[0].items()}
        altered[field].flat[1] += 1 if field == "iterations" else 0.1
        np.savez_compressed(output / name, **altered)
    hashes = json.loads((output / "files.json").read_text())
    hashes[name] = q.digest((output / name).read_bytes())
    q.write_json(output / "files.json", hashes)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)


def test_probe_uses_only_original_commands_and_previous_original_plans(monkeypatch):
    plan, _ = q.frozen_plan()
    plan = plan | {"selection": plan["selection"] | {"seeds": [101], "origins": [2, 3]}}
    commands = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]])
    states = np.zeros((5, 13))
    states[:, 6] = 1
    states[1:, 0] = np.cumsum(commands[:, 0])
    tracking = dict(commands=commands, states=states, reference_anchor_state=states[0])
    diagnostics = warm_diagnostics()
    advances, solves = [], []

    class Arm:
        def __init__(self, *_, **__):
            self.policy = SolverPolicy(
                horizon_steps=5, block_count=5, maximum_iterations=4
            )
            self.plan = SimpleNamespace(
                with_causal_state=lambda state: state.copy(),
                command_minimum=np.zeros(3),
                command_maximum=np.ones(3),
            )

        def reset(self, state, _):
            self._state = state.copy()

        def observe(self, state):
            np.testing.assert_allclose(self._state, state)

        def command_applied(self, command):
            advances.append(command.copy())
            self._state[0] += command[0]

    def pair(model, policy, state, reference, previous, *, warm_start, baseline_check):
        solves.append((model.copy(), previous.copy(), warm_start))
        a, b = result(), result(final=1.0)
        b.predicted_commands[:] = 99.0  # Must never drive state or next warm start.
        return dict(baseline=a, candidate=b)

    context = SimpleNamespace(
        model=None,
        learned=None,
        initial_command=np.zeros(3),
        manifest={"trial": {"sample_interval_s": 0.05}, "tracking_reference": {}},
    )
    monkeypatch.setattr(q, "_context", lambda _: context)
    monkeypatch.setattr(
        q, "CascadeEquations", lambda _: SimpleNamespace(canonical=lambda s: s)
    )
    monkeypatch.setattr(q, "TaskOracleArm", Arm)
    monkeypatch.setattr(q, "_diagnostic_schema", lambda *_: None)
    monkeypatch.setattr(
        q, "arrays", lambda raw: tracking if raw == b"tracking" else diagnostics
    )
    monkeypatch.setattr(
        q,
        "control_reference",
        lambda anchor, times, _: np.repeat(anchor[None], len(times), axis=0),
    )
    monkeypatch.setattr(q, "paired_solve", pair)
    prefix = "trial-0/oracle_task_scales/"
    inputs = {
        "manifest.json": b"{}",
        "inputs/control/manifest.json": json.dumps(
            {"telemetry": {"command_minimum": [0, 0, 0], "command_maximum": [1, 1, 1]}}
        ).encode(),
        prefix + "trial.json": b'{"initial_state_seed": 101, "terminated": false}',
        prefix + "tracking.npz": b"tracking",
        prefix + "oracle.npz": b"diagnostics",
    }
    q.probe(plan, inputs)
    np.testing.assert_array_equal(advances, commands)
    assert solves[0][2] is None
    np.testing.assert_array_equal(
        solves[1][2].commands, diagnostics["candidate_commands"][0]
    )
    for i, (state, previous, _) in enumerate(solves):
        np.testing.assert_allclose(state, states[i + 2])
        np.testing.assert_array_equal(previous, commands[i + 1])
