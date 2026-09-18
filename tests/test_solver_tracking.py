"""Tracking diagnostics preserve application denominators and replay evidence."""

import ast
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import solver_tracking as q
from glassbox.experimental.harness import (
    SIMULATED_TIME_MEANING,
    control_pass_criterion,
    control_reference,
    simulated_time_wall,
)
from glassbox.experimental.task_qualification import ROW_FIELDS

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "docs/harness/solver-tracking-v1.json").read_text())
CONTROL = json.loads((ROOT / "docs/harness/control-v5.json").read_text())
ARMS = ["oracle_pg4", "oracle_lbfgs32", "oracle_lbfgs64"]
INITIAL = np.array([0, 0, 100, 18, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
COMMAND = np.array([0.5, 0.0, 0.0])


def npz_bytes(**values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def json_bytes(value):
    return json.dumps(value, allow_nan=False).encode()


def report_fixture():
    """Complete numeric evidence for twelve synthetic perfect tracking trials."""
    rows, saved, prewarm = [], {}, []
    target = control_reference(
        INITIAL, np.arange(321) * 0.05, CONTROL["tracking_reference"]
    )
    for repetition, order in enumerate(PLAN["control_qualification"]["arm_order"]):
        for arm in order:
            prefix = f"trial-{repetition}/{arm}"
            states = target.copy()
            error = {ARMS[0]: (0.4, 0.3), ARMS[1]: (0.2, 0.1), ARMS[2]: (0.1, 0.05)}[
                arm
            ]
            states[:, 1:3] += error
            commands = np.tile(COMMAND, (320, 1))
            tracking = dict(
                time_s=np.arange(321) * 0.05,
                states=states,
                reference_states=target,
                commands=commands,
                solver_used=np.arange(320) >= 2,
                used_fallback=np.zeros(320, dtype=bool),
                initial_state=states[0],
                reference_anchor_state=INITIAL,
            )
            duration = {ARMS[0]: 0.02, ARMS[1]: 0.08, ARMS[2]: 0.04}[arm]
            solve_times = np.r_[np.zeros(2), np.full(318, duration)]
            ticks = np.full(320, duration + 0.01)
            timing = dict(
                tick_times_s=ticks,
                solve_times_s=solve_times,
                deadline_assessed=np.zeros(320, dtype=bool),
            )
            diagnostics = dict(
                causal_states=states,
                observed_max_abs_error=np.zeros(320),
                solve_indices=np.arange(2, 320),
                candidate_commands=np.tile(COMMAND, (318, 5, 1)),
                forecast_states=np.tile(INITIAL, (318, 6, 1)),
                final_objectives=np.full(318, 9.0),
                initial_objectives=np.full(318, 12.0),
                projected_gradient_inf_norm=np.zeros(318),
                iterations=np.full(318, 4),
                statuses=np.full(318, "converged"),
                constant_covariance_objectives=np.full(318, 0.25),
                variable_final_objectives=np.full(318, 8.75),
                used_fallback=np.zeros(318, dtype=bool),
            )
            work = dict(
                backend_status=0,
                backend_success=True,
                backend_message="projected gradient",
                stop_cause="projected_gradient",
                accepted_iterations=4,
                new_objective_evaluations=5,
                seed_objective_evaluations=1 if arm == ARMS[2] else 2,
                cached_seed_requests=1,
                audit_objective_evaluations=1,
                nonfinite_evaluation=False,
                backend_failure=False,
                returned_seed=False,
                independent_objective=9.0,
                independent_projected_gradient_inf_norm=0.0,
                returned_canonical_blocks=np.zeros((5, 3)).tolist(),
                independent_gradient=np.zeros((5, 3)).tolist(),
            )
            audit = dict(
                blocks=np.zeros((5, 3)).tolist(),
                value=9.0,
                gradient=np.zeros((5, 3)).tolist(),
                residual=0.0,
                commands=np.tile(COMMAND, (5, 1)).tolist(),
                states=np.tile(INITIAL, (6, 1)).tolist(),
                bound_violation=0.0,
            )
            records = [
                dict(
                    arm=arm,
                    origin=origin,
                    work=None if arm == ARMS[0] else copy.deepcopy(work),
                    capture=(
                        dict(
                            blocks=np.zeros((5, 3)).tolist(),
                            warm_start_used=origin > 2,
                            inputs={},
                            native_seed_value=12.0,
                            native_seed_gradient=np.zeros((5, 3)).tolist(),
                        )
                        if arm == ARMS[2]
                        else None
                    ),
                    precision=(
                        dict(floating_dtype="float64", jaxpr=dict(float32_arithmetic=0))
                        if arm == ARMS[2]
                        else None
                    ),
                    audits=dict(
                        lifted_seed64=copy.deepcopy(audit) if arm == ARMS[2] else None,
                        returned_plan64=copy.deepcopy(audit)
                        if arm == ARMS[2]
                        else None,
                    ),
                    seed_selection=(
                        dict(
                            objective_evaluations=1 if origin == 2 else 2,
                            successful=True,
                            used_warm_start=origin > 2,
                            status="iteration_limit",
                            used_fallback=False,
                        )
                        if arm == ARMS[2]
                        else None
                    ),
                )
                for origin in range(2, 320)
            ]
            row = dict.fromkeys(ROW_FIELDS)
            row.update(
                arm=arm,
                completed_intervals=320,
                requested_intervals=320,
                terminated=False,
                failure=None,
                pass_criterion=control_pass_criterion(states, INITIAL, CONTROL),
                model_not_ready_intervals=2,
                fallback_count=0,
                solver_statuses=dict(model_not_ready=2, converged=318),
                maximum_command_bound_violation=0.0,
                wall=simulated_time_wall(
                    meaning=SIMULATED_TIME_MEANING,
                    dt_s=0.05,
                    deadline_s=CONTROL["trial"]["solve_deadline_s"],
                    tick_times=ticks.tolist(),
                    solve_times=solve_times.tolist(),
                    elapsed_s=float(sum(ticks)),
                    solve_deadline_applied=False,
                    deadline_assessed_intervals=0,
                ),
                controller={},
                repetition=repetition,
                initial_state_seed=PLAN["control_qualification"]["seeds"][repetition],
                directory=prefix,
            )
            saved[prefix + "/tracking.npz"] = npz_bytes(**tracking)
            saved[prefix + "/timing.npz"] = npz_bytes(**timing)
            row["files"] = {
                name: q.digest(saved[prefix + "/" + name])
                for name in ("tracking.npz", "timing.npz")
            }
            saved[prefix + "/oracle.npz"] = npz_bytes(**diagnostics)
            saved[prefix + "/records.json"] = json_bytes(records)
            saved[prefix + "/trial.json"] = json_bytes(row)
            rows.append(row)
            prewarm.append(dict(repetition=repetition, arm=arm, elapsed_seconds=0.5))
    saved["prewarm.json"] = json_bytes(prewarm)
    saved["inputs/control/manifest.json"] = json_bytes(CONTROL)
    return copy.deepcopy(PLAN), rows, saved


def failed_prefix(row, saved, completed):
    prefix = row["directory"] + "/"
    tracking = q.arrays(saved[prefix + "tracking.npz"])
    attempted = completed + 1
    for name in ("time_s", "states", "reference_states"):
        tracking[name] = tracking[name][: completed + 1]
    tracking["commands"] = tracking["commands"][:completed]
    for name in ("solver_used", "used_fallback"):
        tracking[name] = tracking[name][:attempted]
    diagnostics = q.arrays(saved[prefix + "oracle.npz"])
    for name in diagnostics:
        count = (
            completed + 1
            if name == "causal_states"
            else attempted
            if name == "observed_max_abs_error"
            else max(0, attempted - 2)
        )
        diagnostics[name] = diagnostics[name][:count]
    records = json.loads(saved[prefix + "records.json"])[: max(0, attempted - 2)]
    timing = q.arrays(saved[prefix + "timing.npz"])
    timing["tick_times_s"] = timing["tick_times_s"][:completed]
    for name in ("solve_times_s", "deadline_assessed"):
        timing[name] = timing[name][:attempted]
    row.update(
        completed_intervals=completed,
        terminated=True,
        failure="nonfinite plant state",
        pass_criterion=control_pass_criterion(tracking["states"], INITIAL, CONTROL),
        model_not_ready_intervals=min(2, attempted),
        solver_statuses=dict(model_not_ready=min(2, attempted), converged=len(records)),
    )
    saved[prefix + "tracking.npz"] = npz_bytes(**tracking)
    saved[prefix + "oracle.npz"] = npz_bytes(**diagnostics)
    saved[prefix + "timing.npz"] = npz_bytes(**timing)
    saved[prefix + "records.json"] = json_bytes(records)
    saved[prefix + "trial.json"] = json_bytes(row)


def test_frozen_population_sources_inputs_and_measurement_only_decision():
    plan, raw = q.frozen_plan()
    assert q.digest(raw) == q.PLAN_SHA256
    assert len(plan["source_sha256"]) == 90
    assert len(plan["input_sha256"]) == 5
    assert set(plan["input_paths"]) == set(plan["input_sha256"])
    assert plan["no_fit"]
    trial = plan["control_qualification"]
    assert trial["seeds"] == [106, 107, 108, 109]
    assert trial["arms"] == ARMS
    assert len(trial["arm_order"]) == 4
    assert all(sorted(order) == sorted(ARMS) for order in trial["arm_order"])
    assert trial["intervals"] == 320
    assert trial["sample_interval_s"] == 0.05
    assert trial["warmup_intervals"] == 2
    assert trial["horizon_steps"] == trial["block_count"] == 5
    assert trial["position_m"] == [5.0, 0.5, 0.5]
    assert trial["application"]["scored_samples_per_trial"] == 281
    assert not plan["policy_decision"]["historical_precision_qualification_passed"]
    names = q.artifact_names(plan)
    assert len([name for name in names if name.endswith("/tracking.npz")]) == 12
    assert len([name for name in names if name.endswith("/oracle.npz")]) == 12
    assert len([name for name in names if name.endswith("/records.json")]) == 12


def test_runner_does_not_fit_update_or_change_learner():
    tree = ast.parse(Path(q.__file__).read_text())
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        (isinstance(f, ast.Name) and f.id in {"fit", "update"})
        or (isinstance(f, ast.Attribute) and f.attr in {"fit", "update_recordings"})
        for f in calls
    )


def test_changed_manifest_is_rejected_before_any_measurement(tmp_path, monkeypatch):
    _, raw = q.frozen_plan()
    path = tmp_path / "plan.json"
    path.write_bytes(raw + b" ")
    monkeypatch.setattr(q, "PLAN_PATH", path)
    with pytest.raises(ValueError):
        q.frozen_plan()


def test_every_inherited_source_remains_byte_pinned(tmp_path, monkeypatch):
    plan, _ = q.frozen_plan()
    for name in plan["source_sha256"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((q.ROOT / name).read_bytes())
    (tmp_path / next(iter(plan["source_sha256"]))).write_text("# altered\n")
    monkeypatch.setattr(q, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        q.frozen_plan()


def test_input_snapshot_keeps_preflighted_bytes_and_rejects_changed_input(tmp_path):
    (tmp_path / "model.npz").write_bytes(b"original")
    plan = dict(
        input_paths={"control/model.npz": "model.npz"},
        input_sha256={"control/model.npz": q.digest(b"original")},
    )
    inputs = q.input_snapshot(plan, tmp_path)
    (tmp_path / "model.npz").write_bytes(b"changed")
    assert inputs["control/model.npz"] == b"original"
    with pytest.raises(ValueError):
        q.input_snapshot(plan, tmp_path)


def test_environment_check_rejects_unpinned_runtime():
    plan = copy.deepcopy(PLAN)
    plan["environment"]["python"] = "0.0.0"
    with pytest.raises((ValueError, AssertionError)):
        q.check_environment(plan)


def test_environment_versions_accept_separate_cascade_provenance_note():
    import platform

    import jax
    import scipy

    plan = copy.deepcopy(PLAN)
    actual = dict(
        python=platform.python_version(),
        jax=jax.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
    )
    plan["environment"].update(actual)
    assert "cascade" in plan["environment"]
    assert q.check_environment(plan) == actual


@pytest.mark.parametrize("within,passed", [(266, False), (267, True)])
def test_application_threshold_is_95_percent_of_fixed281_samples(within, passed):
    control = copy.deepcopy(CONTROL)
    control["tracking_reference"]["lateral_amplitude_m"] = 0.0
    control["tracking_reference"]["altitude_amplitude_m"] = 0.0
    target = control_reference(
        INITIAL, np.arange(321) * 0.05, control["tracking_reference"]
    )
    states = target.copy()
    states[:, 1:3] += 0.5
    states[40 + within :, 1] += 1e-6
    actual = control_pass_criterion(states, INITIAL, control)
    assert actual["scored_samples"] == 281
    assert actual["within_samples"] == within
    assert actual["within_tolerance_fraction"] == within / 281
    assert actual["met"] is passed


@pytest.mark.parametrize("completed", [0, 39, 40, 100, 319])
def test_perfect_failed_prefix_keeps_missing_suffix_outside_tolerance(completed):
    target = control_reference(
        INITIAL, np.arange(321) * 0.05, CONTROL["tracking_reference"]
    )
    actual = control_pass_criterion(target[: completed + 1], INITIAL, CONTROL)
    assert actual["scored_samples"] == 281
    assert actual["within_samples"] == max(0, completed - 39)
    assert actual["within_tolerance_fraction"] == max(0, completed - 39) / 281
    assert actual["terminated"]
    assert not actual["met"]


def test_all_twelve_successful_trials_cannot_promote_solver_or_change_old_gate():
    plan, rows, saved = report_fixture()
    result = q.report(plan, rows, saved)
    assert result["no_fit"] and result["new_policy_trials"]
    assert not result["maintained_solver_promoted"]
    assert not result["learner_changed"]
    assert not result["historical_precision_qualification_passed"]
    assert result["policy_decision"] == plan["policy_decision"]
    assert len(result["trials"]) == 12
    for arm in ARMS:
        item = result["per_arm"][arm]
        assert item["trials"] == item["completed_trials"] == 4
        assert item["scored_samples"] == item["within_samples"] == 1124
        assert item["within_tolerance_fraction"] == 1.0
        assert item["application_adequate"]
        assert item["solver_reliable"]
        assert item["meets_application_and_reliability"]


def test_primary_differences_match_seed_not_run_order_or_native_scores():
    plan, rows, saved = report_fixture()
    result = q.report(plan, rows, saved)
    assert [item["initial_state_seed"] for item in result["paired_differences"]] == [
        106,
        107,
        108,
        109,
    ]
    for pair in result["paired_differences"]:
        differences = pair["comparisons"]["float64_minus_float32"]
        assert set(differences) == {
            "within_tolerance_fraction",
            "lateral_rmse_m",
            "altitude_rmse_m",
            "backend_failure_solves",
            "nonfinite_evaluation_solves",
            "fallback_count",
            "terminated_trials",
        }
        assert differences["within_tolerance_fraction"] == 0.0
        assert differences["lateral_rmse_m"] == pytest.approx(-0.1)
        assert differences["altitude_rmse_m"] == pytest.approx(-0.05)
        assert not any(
            "objective" in name or "residual" in name for name in differences
        )


def test_measured_cost_keeps_pg4_evaluations_unavailable_and_extra_audits_separate():
    plan, rows, saved = report_fixture()
    result = q.report(plan, rows, saved)
    pg4, fp32, fp64 = (result["per_arm"][arm]["cost"] for arm in ARMS)
    assert not pg4["objective_evaluation_count_available"]
    assert pg4["new_objective_evaluations"] is None
    assert pg4["seed_objective_evaluations"] is None
    assert pg4["audit_objective_evaluations"] is None
    assert pg4["accepted_iterations"] == 4 * 318 * 4
    assert (
        fp32["new_objective_evaluations"]
        == fp64["new_objective_evaluations"]
        == (4 * 318 * 5)
    )
    assert fp32["seed_objective_evaluations"] == 4 * 318 * 2
    assert fp64["seed_objective_evaluations"] == 4 * 318
    assert fp64["seed_selection_float32_objective_evaluations"] == 4 * (1 + 317 * 2)
    assert fp64["extra_float64_audit_calls"] == 4 * 318 * 2
    assert fp32["extra_float64_audit_calls"] == pg4["extra_float64_audit_calls"] == 0
    for cost, duration, above in ((pg4, 0.02, 0), (fp32, 0.08, 1272), (fp64, 0.04, 0)):
        assert cost["outer_solve_seconds"]["count"] == 1272
        assert cost["outer_solve_seconds"]["median"] == duration
        assert cost["outer_solve_seconds"]["p95"] == duration
        assert cost["outer_solve_seconds"]["maximum"] == duration
        assert cost["solves_over_sample_interval"] == above
        assert cost["prewarm_seconds"] == 2.0


@pytest.mark.parametrize("completed", [0, 100, 319])
def test_pooled_report_keeps_all1124_samples_when_one_trial_terminates(completed):
    plan, rows, saved = report_fixture()
    row = next(row for row in rows if row["arm"] == ARMS[2])
    failed_prefix(row, saved, completed)
    result = q.report(plan, rows, saved)
    arm = result["per_arm"][ARMS[2]]
    assert arm["scored_samples"] == 1124
    assert arm["within_samples"] == 3 * 281 + max(0, completed - 39)
    assert arm["within_tolerance_fraction"] == arm["within_samples"] / 1124
    assert arm["completed_trials"] == 3
    assert not arm["application_adequate"]
    assert not arm["meets_application_and_reliability"]
    pair = result["paired_differences"][0]["comparisons"]["float64_minus_float32"]
    assert pair["lateral_rmse_m"] is None
    assert pair["altitude_rmse_m"] is None
    assert pair["within_tolerance_fraction"] == max(0, completed - 39) / 281 - 1.0


@pytest.mark.parametrize(
    "field", ["within_samples", "scored_samples", "met", "lateral_rmse_m"]
)
def test_task_summary_is_recomputed_from_saved_states(field):
    plan, rows, saved = report_fixture()
    criterion = rows[0]["pass_criterion"]
    criterion[field] = not criterion[field] if field == "met" else criterion[field] + 1
    with pytest.raises((ValueError, AssertionError)):
        q.report(plan, rows, saved)


@pytest.mark.parametrize("flag", ["backend_failure", "nonfinite_evaluation"])
def test_finite_plan_backend_failure_keeps_task_gain_but_fails_reliability(flag):
    plan, rows, saved = report_fixture()
    row = next(row for row in rows if row["arm"] == ARMS[2])
    name = row["directory"] + "/records.json"
    records = json.loads(saved[name])
    records[0]["work"][flag] = True
    saved[name] = json_bytes(records)
    result = q.report(plan, rows, saved)
    arm = result["per_arm"][ARMS[2]]
    assert arm["application_adequate"]
    assert arm["within_samples"] == 1124
    assert arm["fallback_count"] == 0
    assert not arm["solver_reliable"]
    assert not arm["meets_application_and_reliability"]
    field = (
        "backend_failure_solves"
        if flag == "backend_failure"
        else "nonfinite_evaluation_solves"
    )
    assert arm[field] == 1
    assert arm["native_residual_at_threshold_solves"] == 1272
    assert arm["native_converged_solves"] == 1271
    assert result["paired_differences"][0]["comparisons"]["float64_minus_float32"][
        "lateral_rmse_m"
    ] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "seed", "directory", "field"]
)
def test_report_rejects_incomplete_or_forged_trial_roster(defect):
    plan, rows, saved = report_fixture()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif defect == "seed":
        rows[0]["initial_state_seed"] = 101
    elif defect == "directory":
        rows[0]["directory"] = "trial-0/other"
    else:
        rows[0]["unfrozen"] = True
    with pytest.raises((ValueError, AssertionError)):
        q.report(plan, rows, saved)


@pytest.mark.parametrize("defect", ["negative", "nan", "shape", "deadline", "prewarm"])
def test_invalid_timing_is_rejected_without_turning_time_into_a_promotion_gate(defect):
    plan, rows, saved = report_fixture()
    path = rows[0]["directory"] + "/timing.npz"
    timing = q.arrays(saved[path])
    if defect == "negative":
        timing["solve_times_s"][2] = -0.1
    elif defect == "nan":
        timing["tick_times_s"][2] = np.nan
    elif defect == "shape":
        timing["solve_times_s"] = timing["solve_times_s"][:-1]
    elif defect == "deadline":
        timing["deadline_assessed"][2] = True
    else:
        prewarm = json.loads(saved["prewarm.json"])
        prewarm[0]["elapsed_seconds"] = -0.1
        saved["prewarm.json"] = json_bytes(prewarm)
    saved[path] = npz_bytes(**timing)
    with pytest.raises((ValueError, AssertionError)):
        q.report(plan, rows, saved)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_record",
        "origin",
        "work_count",
        "work_score",
        "gradient",
        "precision",
        "audit_role",
    ],
)
def test_record_schema_rejects_unreplayable_or_mislabelled_solver_evidence(defect):
    _, rows, saved = report_fixture()
    row = next(row for row in rows if row["arm"] == ARMS[2])
    prefix = row["directory"] + "/"
    tracking = q.arrays(saved[prefix + "tracking.npz"])
    diagnostics = q.arrays(saved[prefix + "oracle.npz"])
    records = json.loads(saved[prefix + "records.json"])
    if defect == "missing_record":
        records.pop()
    elif defect == "origin":
        records[0]["origin"] = 3
    elif defect == "work_count":
        records[0]["work"]["new_objective_evaluations"] = 1025
    elif defect == "work_score":
        records[0]["work"]["independent_objective"] = 10.0
    elif defect == "gradient":
        records[0]["work"]["independent_gradient"][0][0] = 0.1
    elif defect == "precision":
        records[0]["precision"]["jaxpr"]["float32_arithmetic"] = 1
    else:
        records[0]["audits"]["baseline"] = records[0]["audits"].pop("lifted_seed64")
    with pytest.raises((ValueError, AssertionError)):
        q.validate_records(row["arm"], tracking, diagnostics, records)


def test_float64_work_audit_keeps_float64_residual_near_command_bound():
    _, rows, saved = report_fixture()
    row = next(row for row in rows if row["arm"] == ARMS[2])
    prefix = row["directory"] + "/"
    tracking = q.arrays(saved[prefix + "tracking.npz"])
    diagnostics = q.arrays(saved[prefix + "oracle.npz"])
    records = json.loads(saved[prefix + "records.json"])
    blocks = np.full((5, 3), 0.99999998, dtype=np.float64)
    gradient = np.full((5, 3), 0.00199999, dtype=np.float64)
    residual = float(np.max(np.abs(blocks - np.clip(blocks - gradient, -1, 1))))
    records[0]["work"].update(
        returned_canonical_blocks=blocks.tolist(),
        independent_gradient=gradient.tolist(),
        independent_projected_gradient_inf_norm=residual,
    )
    records[0]["audits"]["returned_plan64"].update(
        blocks=blocks.tolist(), gradient=gradient.tolist(), residual=residual
    )
    diagnostics["projected_gradient_inf_norm"][0] = residual
    assert q.validate_records(row["arm"], tracking, diagnostics, records) == 318


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """Public verification with independent deterministic replay doubles."""
    plan, rows, saved = report_fixture()
    manifest = (ROOT / "docs/harness/control-v5.json").read_bytes()
    inputs = {"control/manifest.json": manifest}
    plan["input_paths"] = {name: name for name in inputs}
    plan["input_sha256"] = {name: q.digest(raw) for name, raw in inputs.items()}
    raw = json_bytes(plan)
    reference_rows, reference_saved = copy.deepcopy(rows), copy.deepcopy(saved)
    by_directory = {row["directory"]: row for row in reference_rows}
    made, physical_calls, optimizer_calls = [], [], []

    class ReplayArm:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.row = reference_rows[len(made) % 12]
            self.name = self.row["arm"]
            made.append(self.row["directory"])

        def summary(self):
            return {}

    def physical(_plan, _context, row, proposed):
        physical_calls.append(row["directory"])
        prefix = row["directory"] + "/"
        original = q.arrays(reference_saved[prefix + "tracking.npz"])
        changed = q.arrays(proposed[prefix + "tracking.npz"])
        q.exact_arrays(changed, original)
        q.exact(row, by_directory[row["directory"]])
        return dict(state_difference=0.0)

    def optimizer(arm, _manifest, _tracking, diagnostics, records):
        prefix = arm.row["directory"] + "/"
        optimizer_calls.append(arm.row["directory"])
        q.exact_arrays(diagnostics, q.arrays(reference_saved[prefix + "oracle.npz"]))
        q.exact(records, json.loads(reference_saved[prefix + "records.json"]))
        return dict(
            verified_optimizer_solves=318,
            verified_oracle_forecasts=318,
            solver_statuses=arm.row["solver_statuses"],
        )

    monkeypatch.setattr(q, "frozen_plan", lambda: (plan, raw))
    monkeypatch.setattr(q, "check_environment", lambda _: {})
    monkeypatch.setattr(
        q,
        "_context",
        lambda _: SimpleNamespace(
            manifest=CONTROL,
            model=None,
            learned=None,
            initial_state=INITIAL,
            initial_command=COMMAND,
        ),
    )
    monkeypatch.setattr(q, "CascadeEquations", lambda _: None)
    monkeypatch.setattr(q, "TrackingOracleArm", ReplayArm)
    monkeypatch.setattr(q, "_replay_trial", physical)
    monkeypatch.setattr(q, "verify_solves", optimizer)
    saved["run.json"] = json_bytes(dict(no_fit=True, new_policy_trials=True))
    saved["environment.json"] = b"{}"
    saved["results.json"] = json_bytes(rows)
    saved["report.json"] = json_bytes(q.report(plan, rows, saved))
    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_bytes(raw)
    for name, data in inputs.items():
        path = output / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in q.artifact_names(plan):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(saved[name])
    q.write_json(
        output / "files.json",
        {name: q.digest(saved[name]) for name in q.artifact_names(plan)},
    )
    verified = q.verify(output)
    assert verified["verified_trials"] == 12
    assert verified["verified_optimizer_solves"] == 12 * 318
    assert len(physical_calls) == len(optimizer_calls) == 12
    return plan, rows, saved, output, physical_calls, optimizer_calls


@pytest.mark.parametrize(
    "defect", ["command", "full_gradient", "work", "qualification"]
)
def test_forged_artifacts_rejected_after_rehash_and_recomputed_dependent_summaries(
    saved_run, defect
):
    plan, rows, saved, output, physical_calls, optimizer_calls = saved_run
    row = next(row for row in rows if row["arm"] == ARMS[2])
    prefix = row["directory"] + "/"
    records = json.loads(saved[prefix + "records.json"])
    diagnostics = q.arrays(saved[prefix + "oracle.npz"])
    if defect == "command":
        tracking = q.arrays(saved[prefix + "tracking.npz"])
        tracking["commands"][2, 0] += 0.001
        saved[prefix + "tracking.npz"] = npz_bytes(**tracking)
        row["files"]["tracking.npz"] = q.digest(saved[prefix + "tracking.npz"])
    elif defect == "full_gradient":
        work = records[0]["work"]
        work["independent_gradient"][0][0] = 0.0001
        work["independent_projected_gradient_inf_norm"] = 0.0001
        audit = records[0]["audits"]["returned_plan64"]
        audit["gradient"][0][0] = audit["residual"] = 0.0001
        diagnostics["projected_gradient_inf_norm"][0] = 0.0001
    elif defect == "work":
        records[0]["work"]["new_objective_evaluations"] += 1
    saved[prefix + "records.json"] = json_bytes(records)
    saved[prefix + "oracle.npz"] = npz_bytes(**diagnostics)
    saved[prefix + "trial.json"] = json_bytes(row)
    saved["results.json"] = json_bytes(rows)
    derived = q.report(plan, rows, saved)
    if defect == "qualification":
        derived["maintained_solver_promoted"] = True
    saved["report.json"] = json_bytes(derived)
    for name in q.artifact_names(plan):
        (output / name).write_bytes(saved[name])
    q.write_json(
        output / "files.json",
        {name: q.digest(saved[name]) for name in q.artifact_names(plan)},
    )
    before = len(physical_calls) + len(optimizer_calls)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
    if defect != "qualification":
        assert len(physical_calls) + len(optimizer_calls) > before


@pytest.mark.parametrize("defect", ["input", "extra_file", "manifest"])
def test_public_verify_rejects_forged_inputs_and_roster_before_replay(
    saved_run, defect
):
    _, _, _, output, physical_calls, optimizer_calls = saved_run
    if defect == "input":
        (output / "inputs/control/manifest.json").write_bytes(b"changed")
    elif defect == "extra_file":
        (output / "unfrozen.json").write_bytes(b"{}")
    else:
        with (output / "manifest.json").open("ab") as stream:
            stream.write(b" ")
    before = len(physical_calls) + len(optimizer_calls)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
    assert len(physical_calls) + len(optimizer_calls) == before
