"""Qualification requires measured progress and replayable, complete evidence."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import test_solver_budget as budget_fixture

from glassbox.experimental import quasi_newton_qualification as q


def fixture():
    plan, _ = q.frozen_plan()
    trials = budget_fixture.saved_trials(plan)
    work = []
    for seed, trial in zip(plan["selection"]["seeds"], trials, strict=True):
        trial["initial_objectives"][:] = 12.0
        trial["final_objectives"][:] = [10.0, 9.0]
        trial["projected_gradient_inf_norm"][:] = [0.1, 0.001]
        trial["statuses"][:, 1] = "converged"
        for origin in plan["selection"]["origins"]:
            work.append(
                dict(
                    seed=seed,
                    origin=origin,
                    backend_status=0,
                    backend_success=True,
                    backend_message="projected gradient",
                    stop_cause="projected_gradient",
                    accepted_iterations=4,
                    new_objective_evaluations=5,
                    seed_objective_evaluations=2,
                    cached_seed_requests=1,
                    audit_objective_evaluations=1,
                    nonfinite_evaluation=False,
                    backend_failure=False,
                    returned_seed=False,
                    independent_objective=9.0,
                    independent_projected_gradient_inf_norm=0.001,
                    returned_canonical_blocks=np.zeros((5, 3)).tolist(),
                    independent_gradient=np.full((5, 3), 0.001).tolist(),
                )
            )
    inputs = {
        "inputs/inputs/control/manifest.json": json.dumps(
            {
                "telemetry": {
                    "command_minimum": [0, 0, 0],
                    "command_maximum": [1, 1, 1],
                },
            }
        ).encode()
    }
    return plan, trials, inputs, work


def test_frozen_source_parent_and_protocol_pins():
    plan, raw = q.frozen_plan()
    assert q.digest(raw) == q.PLAN_SHA256
    assert len(plan["input_sha256"]) == 57
    assert len(plan["source_sha256"]) == 82
    assert plan["environment"]["scipy"] == "1.18.1"
    assert plan["comparison"]["baseline_maximum_iterations"] == 64
    assert plan["selection"]["origins"] == np.linspace(2, 319, 32, dtype=int).tolist()


def test_qualification_is_only_eligibility_for_future_tracking():
    plan, trials, inputs, work = fixture()
    report = q.report_from_inputs(plan, trials, inputs, work)
    assert report["qualification"]["eligible_for_future_tracking_experiment"]
    assert not report["qualification"]["maintained_solver_promoted"]
    assert not report["task_success_assessed"]
    assert report["plan_sha256"] == q.PLAN_SHA256
    assert report["qualification"]["converged_origins"] == 128


@pytest.mark.parametrize("converged,eligible", [(63, False), (64, True)])
def test_declared_breadth_threshold_does_not_require_every_origin_to_converge(
    converged, eligible
):
    plan, trials, inputs, work = fixture()
    for index in range(converged, 128):
        trial, row = divmod(index, 32)
        trials[trial]["projected_gradient_inf_norm"][row, 1] = 0.003
        trials[trial]["statuses"][row, 1] = "stalled"
        work[index]["independent_projected_gradient_inf_norm"] = 0.003
        work[index]["independent_gradient"] = np.full((5, 3), 0.003).tolist()
        work[index]["stop_cause"] = "relative_improvement"
    report = q.report_from_inputs(plan, trials, inputs, work)
    assert report["qualification"]["converged_origins"] == converged
    assert (
        report["qualification"]["eligible_for_future_tracking_experiment"] is eligible
    )


@pytest.mark.parametrize(
    "criterion",
    [
        "enough_converged_origins",
        "median_residual_ratio",
        "objective_regression",
        "candidate_failures",
        "command_bounds",
    ],
)
def test_every_frozen_qualification_limit_is_enforced(criterion):
    plan, trials, inputs, work = fixture()
    if criterion in {"enough_converged_origins", "median_residual_ratio"}:
        value = 0.003 if criterion == "enough_converged_origins" else 0.06
        for trial in trials:
            trial["projected_gradient_inf_norm"][:, 1] = value
        for row in work:
            row["independent_projected_gradient_inf_norm"] = value
            row["independent_gradient"] = np.full((5, 3), value).tolist()
    elif criterion == "objective_regression":
        trials[0]["final_objectives"][0, 1] = 10.2
        work[0]["independent_objective"] = 10.2
    elif criterion == "candidate_failures":
        work[0]["backend_failure"] = True
    else:
        trials[0]["command_bound_violation"][0, 1] = 0.001
    report = q.report_from_inputs(plan, trials, inputs, work)
    assert not report["qualification"]["criteria"][criterion]
    assert not report["qualification"]["eligible_for_future_tracking_experiment"]
    assert len(report["origins"]) == 128


def test_inherited_failure_is_retained_in_denominator():
    plan, trials, inputs, work = fixture()
    trials[0]["used_fallback"][0, 1] = True
    trials[0]["final_objectives"][0, 1] = np.inf
    trials[0]["projected_gradient_inf_norm"][0, 1] = np.inf
    trials[0]["iterations"][0, 1] = 0
    work[0].update(
        stop_cause="inherited_failure",
        backend_failure=True,
        independent_objective=None,
        independent_projected_gradient_inf_norm=None,
        audit_objective_evaluations=0,
        accepted_iterations=0,
        returned_canonical_blocks=None,
        independent_gradient=None,
    )
    report = q.report_from_inputs(plan, trials, inputs, work)
    assert len(report["origins"]) == 128
    assert report["qualification"]["candidate_failures"] == 1
    assert report["qualification"]["converged_origins"] == 127
    assert not report["qualification"]["eligible_for_future_tracking_experiment"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("new_objective_evaluations", 1025),
        ("accepted_iterations", 65),
        ("audit_objective_evaluations", 0),
        ("independent_objective", 8.0),
        ("independent_projected_gradient_inf_norm", 0.0),
        ("origin", 3),
        ("backend_failure", 0),
        ("stop_cause", "unknown"),
    ],
)
def test_work_schema_rejects_forged_accounting(field, value):
    plan, trials, _, work = fixture()
    work[0][field] = value
    with pytest.raises((ValueError, AssertionError)):
        q.validate_work(plan, trials, work)


@pytest.mark.parametrize("field", ["report", "commands", "work"])
def test_artifact_replay_rejects_forged_hashes_and_recomputed_summary(
    tmp_path, monkeypatch, field
):
    plan, trials, inputs, work = fixture()
    _, raw = q.frozen_plan()
    report = q.report_from_inputs(plan, trials, inputs, work)
    plan = plan | {
        "input_sha256": {name: q.digest(value) for name, value in inputs.items()}
    }
    monkeypatch.setattr(q, "frozen_plan", lambda: (plan, raw))
    monkeypatch.setattr(q, "check_environment", lambda _: {})
    monkeypatch.setattr(q, "probe", lambda *_: (trials, work, report))
    parent, output = tmp_path / "parent", tmp_path / "run"
    for name, value in inputs.items():
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    q.run(parent, output)
    assert q.verify(output)["verified_pairs"] == 128
    hashes = json.loads((output / "files.json").read_text())
    altered_report = copy.deepcopy(report)
    if field == "report":
        altered_report["qualification"]["eligible_for_future_tracking_experiment"] = (
            False
        )
    elif field == "commands":
        altered = {k: v.copy() for k, v in trials[0].items()}
        altered["commands"][0, 1, -1, 0] += 0.1
        name = "trial-0/paired.npz"
        np.savez_compressed(output / name, **altered)
        hashes[name] = q.digest((output / name).read_bytes())
    else:
        altered = copy.deepcopy(work)
        altered[0]["new_objective_evaluations"] += 1
        q.write_json(output / "work.json", altered)
        hashes["work.json"] = q.digest((output / "work.json").read_bytes())
        altered_report = q.report_from_inputs(plan, trials, inputs, altered)
    q.write_json(output / "report.json", altered_report)
    hashes["report.json"] = q.digest((output / "report.json").read_bytes())
    q.write_json(output / "files.json", hashes)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)


def test_replay_hook_does_not_allow_unrelated_harness_edits(tmp_path, monkeypatch):
    plan, _ = q.frozen_plan()
    for name in list(plan["source_sha256"]) + [plan["allowed_harness_edit"]["path"]]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((q.ROOT / name).read_bytes())
    path = tmp_path / plan["allowed_harness_edit"]["path"]
    path.write_text(path.read_text() + "\n# unrelated change\n")
    monkeypatch.setattr(q, "ROOT", Path(tmp_path))
    with pytest.raises(ValueError, match="beyond"):
        q.frozen_plan()
