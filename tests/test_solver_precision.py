"""Precision comparisons use common arithmetic and preserve saved evidence."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import test_quasi_newton_qualification as old_fixture

from glassbox.experimental import solver_precision as q


def test_frozen_population_and_backend_options_remain_fixed():
    plan, raw = q.frozen_plan()
    assert q.digest(raw) == q.PLAN_SHA256
    assert len(plan["input_sha256"]) == 77
    assert len(plan["source_sha256"]) == 88
    assert plan["selection"]["seeds"] == [101, 102, 104, 105]
    assert plan["selection"]["origins"] == np.linspace(2, 319, 32, dtype=int).tolist()
    assert len(plan["selection"]["seeds"]) * len(plan["selection"]["origins"]) == 128
    assert plan["no_fit"] and not plan["new_policy_trials"]
    assert plan["comparison"]["backend_options"] == dict(
        maxiter=64, maxls=16, maxcor=10, ftol=0.0, gtol=0.002, maxfun=1024
    )


def test_changed_manifest_cannot_change_the_scoring_precision(tmp_path, monkeypatch):
    _, raw = q.frozen_plan()
    path = tmp_path / "plan.json"
    path.write_bytes(raw + b" ")
    monkeypatch.setattr(q, "PLAN_PATH", path)
    with pytest.raises(ValueError):
        q.frozen_plan()


def test_every_inherited_source_remains_byte_pinned(tmp_path, monkeypatch):
    plan, _ = q.frozen_plan()
    for name in plan["source_sha256"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((q.ROOT / name).read_bytes())
    (tmp_path / next(iter(plan["source_sha256"]))).write_text("# altered\n")
    monkeypatch.setattr(q, "ROOT", Path(tmp_path))
    with pytest.raises(ValueError):
        q.frozen_plan()


def fixture():
    plan, _ = q.frozen_plan()
    _, trials, _, work_rows = old_fixture.fixture()
    records = []
    for index, previous in enumerate(work_rows):
        trial, row = divmod(index, 32)
        baseline = {
            k: copy.deepcopy(v)
            for k, v in previous.items()
            if k not in {"seed", "origin"}
        }
        baseline.update(
            independent_objective=10.0,
            independent_projected_gradient_inf_norm=0.1,
            independent_gradient=np.full((5, 3), 0.1).tolist(),
            seed_objective_evaluations=1 if row == 0 else 2,
        )
        candidate = {
            k: copy.deepcopy(v)
            for k, v in previous.items()
            if k not in {"seed", "origin"}
        }
        candidate["seed_objective_evaluations"] = 1

        def audit(value, residual):
            return dict(
                blocks=np.zeros((5, 3)).tolist(),
                value=value,
                gradient=np.full((5, 3), residual).tolist(),
                residual=residual,
                commands=np.zeros((5, 3)).tolist(),
                states=np.zeros((6, 13)).tolist(),
                bound_violation=0.0,
            )

        records.append(
            dict(
                seed=previous["seed"],
                origin=previous["origin"],
                capture=dict(
                    blocks=np.zeros((5, 3)).tolist(),
                    warm_start_used=False,
                    inputs={},
                    native_seed_value=12.0,
                    native_seed_gradient=np.zeros((5, 3)).tolist(),
                ),
                baseline_work=baseline,
                candidate_work=candidate,
                common=dict(baseline=audit(10.0, 0.1), candidate=audit(9.0, 0.001)),
                precision={},
            )
        )
        trials[trial]["initial_objectives"][row] = [12.0, 12.00001]
    return plan, trials, records


def candidate_residual(trials, records, index, value):
    trial, row = divmod(index, 32)
    trials[trial]["projected_gradient_inf_norm"][row, 1] = value
    work = records[index]["candidate_work"]
    work["independent_projected_gradient_inf_norm"] = value
    work["independent_gradient"] = np.full((5, 3), value).tolist()
    common = records[index]["common"]["candidate"]
    common["residual"] = value
    common["gradient"] = np.full((5, 3), value).tolist()


def test_primary_qualification_does_not_mix_native32_and_common64_metrics():
    plan, trials, records = fixture()
    for index, record in enumerate(records):
        trial, row = divmod(index, 32)
        trials[trial]["final_objectives"][row, 0] = 0.1
        trials[trial]["projected_gradient_inf_norm"][row, 0] = 0.0001
        record["baseline_work"].update(
            independent_objective=0.1,
            independent_projected_gradient_inf_norm=0.0001,
            independent_gradient=np.full((5, 3), 0.0001).tolist(),
        )
        candidate_residual(trials, records, index, 0.0015)
    report = q.report_from_records(plan, trials, records)
    gate = report["qualification"]
    assert gate["eligible_for_future_tracking_experiment"]
    assert gate["converged_origins"] == 128
    assert gate["median_residual_ratio"] == pytest.approx(0.015)
    assert not gate["maintained_solver_promoted"]
    assert not report["new_policy_trials"]


@pytest.mark.parametrize("converged,eligible", [(63, False), (64, True)])
def test_breadth_threshold_uses_all128_common_audits(converged, eligible):
    plan, trials, records = fixture()
    for index in range(converged, 128):
        candidate_residual(trials, records, index, 0.003)
    report = q.report_from_records(plan, trials, records)
    gate = report["qualification"]
    assert gate["converged_origins"] == converged
    assert gate["eligible_for_future_tracking_experiment"] is eligible


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
def test_every_frozen_common64_limit_is_enforced(criterion):
    plan, trials, records = fixture()
    if criterion == "enough_converged_origins":
        for index in range(128):
            candidate_residual(trials, records, index, 0.003)
    elif criterion == "median_residual_ratio":
        for index, record in enumerate(records):
            record["common"]["baseline"]["gradient"] = np.full((5, 3), 0.001).tolist()
            record["common"]["baseline"]["residual"] = 0.001
            candidate_residual(trials, records, index, 0.0012)
    elif criterion == "objective_regression":
        records[0]["common"]["candidate"]["value"] = 10.2
        records[0]["candidate_work"]["independent_objective"] = 10.2
        trials[0]["final_objectives"][0, 1] = 10.2
    elif criterion == "candidate_failures":
        records[0]["candidate_work"]["backend_failure"] = True
    else:
        records[0]["common"]["candidate"]["bound_violation"] = 1e-8
        trials[0]["command_bound_violation"][0, 1] = 1e-8
    report = q.report_from_records(plan, trials, records)
    assert not report["qualification"]["criteria"][criterion]
    assert not report["qualification"]["eligible_for_future_tracking_experiment"]


@pytest.mark.parametrize("candidate_value,passes", [(5e-15, True), (1.01e-14, False)])
def test_fractional_regression_uses_declared_common_objective_floor(
    candidate_value, passes
):
    plan, trials, records = fixture()
    records[0]["common"]["baseline"]["value"] = 0.0
    records[0]["common"]["candidate"]["value"] = candidate_value
    records[0]["candidate_work"]["independent_objective"] = candidate_value
    trials[0]["final_objectives"][0, 1] = candidate_value
    gate = q.report_from_records(plan, trials, records)["qualification"]
    assert gate["criteria"]["objective_regression"] is passes


def test_common_residual_uses_float64_subtraction_at_a_bound():
    plan, trials, records = fixture()
    common = records[0]["common"]["candidate"]
    blocks = np.full((5, 3), 0.99999998, dtype=np.float64)
    gradient = np.full((5, 3), 0.00199999, dtype=np.float64)
    residual = float(np.max(np.abs(blocks - np.clip(blocks - gradient, -1, 1))))
    rounded_blocks, rounded_gradient = (
        blocks.astype(np.float32),
        gradient.astype(np.float32),
    )
    rounded = float(
        np.max(
            np.abs(rounded_blocks - np.clip(rounded_blocks - rounded_gradient, -1, 1))
        )
    )
    assert abs(residual - rounded) > 1e-9 + 1e-7 * abs(residual)
    common.update(blocks=blocks.tolist(), gradient=gradient.tolist(), residual=residual)
    records[0]["candidate_work"].update(
        returned_canonical_blocks=blocks.tolist(),
        independent_gradient=gradient.tolist(),
        independent_projected_gradient_inf_norm=residual,
    )
    trials[0]["projected_gradient_inf_norm"][0, 1] = residual
    gate = q.report_from_records(plan, trials, records)["qualification"]
    assert gate["converged_origins"] == 128


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    plan, trials, records = fixture()
    _, raw = q.frozen_plan()
    inputs = {"parent.json": b"pinned synthetic parent"}
    plan = plan | {"input_sha256": {k: q.digest(v) for k, v in inputs.items()}}
    result = q.report_from_records(plan, trials, records)
    calls = []

    def replay(*_):
        calls.append(True)
        return copy.deepcopy((trials, records, result))

    monkeypatch.setattr(q, "frozen_plan", lambda: (plan, raw))
    monkeypatch.setattr(q, "check_environment", lambda _: {})
    monkeypatch.setattr(q, "probe", replay)
    parent, output = tmp_path / "parent", tmp_path / "run"
    parent.mkdir()
    for name, value in inputs.items():
        (parent / name).write_bytes(value)
    q.run(parent, output)
    assert q.verify(output)["verified"]
    return plan, trials, records, result, output, calls


@pytest.mark.parametrize(
    "field", ["command", "common_gradient", "work", "qualification"]
)
def test_forged_outputs_fail_after_hashes_and_summaries_are_recomputed(
    saved_run, field
):
    plan, trials, records, _, output, calls = saved_run
    changed, changed_trials = copy.deepcopy(records), copy.deepcopy(trials)
    if field == "command":
        changed[0]["common"]["candidate"]["commands"][0][0] += 0.0001
        changed_trials[0]["commands"][0, 1, 0, 0] += 0.0001
    elif field == "common_gradient":
        common = changed[0]["common"]["candidate"]
        common["gradient"][0][0] += 0.0001
        blocks, gradient = np.asarray(common["blocks"]), np.asarray(common["gradient"])
        common["residual"] = float(
            np.max(np.abs(blocks - np.clip(blocks - gradient, -1, 1)))
        )
        changed[0]["candidate_work"].update(
            independent_gradient=copy.deepcopy(common["gradient"]),
            independent_projected_gradient_inf_norm=common["residual"],
        )
        changed_trials[0]["projected_gradient_inf_norm"][0, 1] = common["residual"]
    elif field == "work":
        changed[0]["candidate_work"]["new_objective_evaluations"] += 1
    summary = q.report_from_records(plan, changed_trials, changed)
    if field == "qualification":
        summary["qualification"]["eligible_for_future_tracking_experiment"] = False
    q.write_json(output / "records.json", changed)
    q.write_json(output / "report.json", summary)
    np.savez_compressed(output / "trial-0/paired.npz", **changed_trials[0])
    hashes = json.loads((output / "files.json").read_text())
    for name in ("records.json", "report.json", "trial-0/paired.npz"):
        hashes[name] = q.digest((output / name).read_bytes())
    q.write_json(output / "files.json", hashes)
    before = len(calls)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
    if field != "qualification":
        assert len(calls) == before + 1


@pytest.mark.parametrize(
    "defect", ["input", "extra_file", "array_dtype", "array_schema"]
)
def test_verification_rejects_input_roster_and_array_schema_changes(saved_run, defect):
    _, trials, _, _, output, _ = saved_run
    if defect == "input":
        (output / "inputs/parent.json").write_bytes(b"changed parent")
    elif defect == "extra_file":
        (output / "extra.json").write_text("{}")
    else:
        changed = copy.deepcopy(trials[0])
        if defect == "array_dtype":
            changed["commands"] = changed["commands"].astype(np.float32)
        else:
            changed["extra"] = np.zeros(1)
        np.savez_compressed(output / "trial-0/paired.npz", **changed)
        hashes = json.loads((output / "files.json").read_text())
        hashes["trial-0/paired.npz"] = q.digest(
            (output / "trial-0/paired.npz").read_bytes()
        )
        q.write_json(output / "files.json", hashes)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
