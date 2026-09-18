"""Frozen four-case diagnosis retains its inputs and rejects forged evidence."""

import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import test_solver_budget as budget_tests
from test_solver_trace import analytic_record

from glassbox.control.plan import SolverPolicy
from glassbox.experimental import solver_termination as q


def test_frozen_protocol_is_diagnostic_and_keeps_every_source_and_input():
    plan, raw = q.frozen_plan()
    assert q.digest(raw) == q.PLAN_SHA256
    assert len(plan["input_sha256"]) == 77
    assert len(plan["source_sha256"]) == 86
    assert plan["selection"]["cases"] == [
        [102, 134],
        [102, 298],
        [104, 186],
        [104, 298],
    ]
    assert plan["no_fit"]
    assert not plan["new_policy_trials"]
    assert not plan["new_solver_candidate"]
    assert plan["finite_difference"]["step_base"] == 2
    assert plan["finite_difference"]["step_exponents"] == [-4, -8, -12, -16, -20, -24]
    assert plan["invariants"]["backend_options"] == dict(
        maxiter=64, maxls=16, maxcor=10, ftol=0.0, gtol=0.002, maxfun=1024
    )


def test_changed_frozen_manifest_is_rejected(tmp_path, monkeypatch):
    _, raw = q.frozen_plan()
    path = tmp_path / "plan.json"
    path.write_bytes(raw + b" ")
    monkeypatch.setattr(q, "PLAN_PATH", path)
    with pytest.raises(ValueError):
        q.frozen_plan()


def test_inherited_source_tampering_is_rejected(tmp_path, monkeypatch):
    plan, _ = q.frozen_plan()
    for name in plan["source_sha256"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((q.ROOT / name).read_bytes())
    (tmp_path / next(iter(plan["source_sha256"]))).write_text("# altered\n")
    monkeypatch.setattr(q, "ROOT", Path(tmp_path))
    with pytest.raises(ValueError):
        q.frozen_plan()


def test_three_prefixes_recover_the_original_task_snapshot():
    task = {
        "manifest.json": b"task",
        "inputs/control/manifest.json": b"control",
        "trial-1/oracle_task_scales/tracking.npz": b"tracking",
    }
    nested = {"inputs/inputs/inputs/" + key: value for key, value in task.items()}
    nested.update(
        {
            "manifest.json": b"first-order",
            "inputs/manifest.json": b"quasi-newton",
            "inputs/inputs/manifest.json": b"budget",
            "trial-1/paired.npz": b"current results",
        }
    )
    before = dict(nested)
    assert q.task_inputs(nested) == task
    assert nested == before


@pytest.mark.parametrize("repetition", [0, 1, 2])
def test_seed_remap_preserves_only_the_selected_original_trial(repetition):
    common = {
        "manifest.json": b"task manifest",
        "inputs/control/generic.npz": b"original model",
        "inputs/historical/trial-0/tracking.npz": b"historical trial",
    }
    task = dict(common)
    for i in (0, 1, 2, 10):
        task[f"trial-{i}/oracle_task_scales/trial.json"] = f"seed-{i}".encode()
        task[f"trial-{i}/oracle_task_scales/tracking.npz"] = f"commands-{i}".encode()
    original = dict(task)
    remapped = q.remap_seed_inputs(task, repetition)
    assert remapped == common | {
        "trial-0/oracle_task_scales/trial.json": f"seed-{repetition}".encode(),
        "trial-0/oracle_task_scales/tracking.npz": f"commands-{repetition}".encode(),
    }
    assert task == original


def test_missing_original_seed_is_not_replaced_by_a_different_trial():
    with pytest.raises(ValueError):
        q.remap_seed_inputs({"trial-0/trial.json": b"wrong seed"}, 1)


def test_reconstruction_uses_original_seed_indices_and_identical_solver_inputs(
    monkeypatch,
):
    plan, _ = q.frozen_plan()
    historical = {
        "selection": {"seeds": [101, 102, 104, 105], "origins": [2, 134, 186, 298]}
    }
    inputs = {
        "manifest.json": json.dumps(historical).encode(),
        "work.json": json.dumps(
            [
                dict(seed=seed, origin=origin, backend_status=2)
                for seed in historical["selection"]["seeds"]
                for origin in historical["selection"]["origins"]
            ]
        ).encode(),
        "inputs/inputs/inputs/manifest.json": b"original task manifest",
        "inputs/inputs/inputs/inputs/control/manifest.json": b"original controls",
    }
    for i, seed in enumerate(historical["selection"]["seeds"]):
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, origins=historical["selection"]["origins"], repetition=i
        )
        inputs[f"trial-{i}/paired.npz"] = buffer.getvalue()
        inputs[f"inputs/inputs/inputs/trial-{i}/oracle_task_scales/trial.json"] = str(
            seed
        ).encode()
    calls, history, remaps, baseline_checks = [], [], [], []

    class Solver:
        kind = "original"

        def __init__(self, model, policy):
            self.model, self.policy = model, policy
            self.last_work = dict(backend_status=2)
            self.last_trace, self.last_directions = {}, {}

        def solve(self, *args, **kwargs):
            calls.append((self.kind, self.model, self.policy, args, kwargs))
            value = budget_tests.result()
            value.kind = self.kind
            return value

    class Untraced(Solver):
        kind = "untraced"

    class Traced(Solver):
        kind = "traced"

    def replay(local_plan, local_inputs, *, _solve_pair):
        seed = local_plan["selection"]["seeds"][0]
        origins = local_plan["selection"]["origins"]
        remaps.append((seed, origins, local_inputs))
        assert (
            local_inputs["trial-0/oracle_task_scales/trial.json"] == str(seed).encode()
        )
        assert local_inputs["inputs/control/manifest.json"] == b"original controls"
        assert len([k for k in local_inputs if k.startswith("trial-")]) == 1
        policy = SolverPolicy(horizon_steps=5, block_count=5, maximum_iterations=4)
        for _ in origins:
            model, state, reference, command, warm = (object() for _ in range(5))
            _solve_pair(
                model,
                policy,
                state,
                reference,
                command,
                warm_start=warm,
                baseline_check=lambda result: baseline_checks.append(result.kind),
            )
        return [dict(origins=np.asarray(origins))], {}

    monkeypatch.setattr(q, "BoundedShootingSolver", Solver)
    monkeypatch.setattr(q.previous, "FirstOrderSolver", Untraced)
    monkeypatch.setattr(q, "TracedFirstOrderSolver", Traced)
    monkeypatch.setattr(q.shared, "validate_work", lambda *_: None)
    monkeypatch.setattr(
        q.shared,
        "historical_parity",
        lambda result, trial, row: history.append(
            (result.kind, int(trial["repetition"]), row)
        ),
    )
    monkeypatch.setattr(q.previous, "historical_work_parity", q.exact)
    monkeypatch.setattr(q.base, "probe", replay)
    monkeypatch.setattr(
        q, "report_from_records", lambda _, records: {"cases": len(records)}
    )
    trials, records, report = q.probe(plan, inputs)
    assert report == {"cases": 4}
    assert [trial["origins"].tolist() for trial in trials] == [[134, 298], [186, 298]]
    assert [[row["seed"], row["origin"]] for row in records] == plan["selection"][
        "cases"
    ]
    assert baseline_checks == ["original"] * 4
    assert [(s, origins) for s, origins, _ in remaps] == [
        (102, [134, 298]),
        (104, [186, 298]),
    ]
    assert history == [
        (kind, repetition, row)
        for repetition, row in [(1, 1), (1, 3), (2, 2), (2, 3)]
        for kind in ("untraced", "traced")
    ]
    for i in range(0, len(calls), 3):
        original, untraced, traced = calls[i : i + 3]
        assert [entry[0] for entry in (original, untraced, traced)] == [
            "original",
            "untraced",
            "traced",
        ]
        assert [
            entry[2].maximum_iterations for entry in (original, untraced, traced)
        ] == [4, 64, 64]
        for other in (untraced, traced):
            assert other[1] is original[1]
            assert all(a is b for a, b in zip(other[3], original[3], strict=True))
            assert other[4]["warm_start"] is original[4]["warm_start"]


def test_readout_preserves_fixed_direction_counts_and_no_promotion_claim():
    plan, _ = q.frozen_plan()
    record = analytic_record()
    records = [
        copy.deepcopy(record) | dict(seed=s, origin=o)
        for s, o in plan["selection"]["cases"]
    ]
    report = q.report_from_records(plan, records)
    assert not report["new_solver_candidate"]
    assert not report["new_policy_trials"]
    assert (
        report["diagnostic_function_calls"]
        == 4 * record["directions"]["diagnostic_function_calls"]
    )
    for row in report["cases"]:
        assert [x["exponent"] for x in row["directional_checks_by_step"]] == plan[
            "finite_difference"
        ]["step_exponents"]
        assert all(
            sum(x["statuses"].values()) == 31 for x in row["directional_checks_by_step"]
        )
    with pytest.raises(ValueError):
        q.report_from_records(plan, records[::-1])


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    plan, raw = q.frozen_plan()
    record = analytic_record()
    records = [
        copy.deepcopy(record) | dict(seed=s, origin=o)
        for s, o in plan["selection"]["cases"]
    ]
    trials = [
        dict(
            origins=np.asarray(origins, dtype=np.int64),
            commands=np.zeros((2, 2, 5, 3), dtype=np.float32),
            states=np.zeros((2, 2, 6, 13), dtype=np.float32),
        )
        for origins in ([134, 298], [186, 298])
    ]
    inputs = {"parent.json": b"pinned synthetic parent"}
    plan = plan | {"input_sha256": {k: q.digest(v) for k, v in inputs.items()}}
    result = q.report_from_records(plan, records)
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
    assert q.verify(output)["verified_cases"] == 4
    return plan, trials, records, result, output, calls


@pytest.mark.parametrize("field", ["trace", "direction_gradient", "work", "report"])
def test_verifier_rejects_forged_records_with_new_hashes_and_summaries(
    saved_run, field
):
    plan, _, records, result, output, calls = saved_run
    changed = copy.deepcopy(records)
    if field == "trace":
        changed[0]["trace"]["requests"][0]["host_blocks"][0][0] += 1e-10
    elif field == "direction_gradient":
        row = next(
            p for p in changed[0]["directions"]["probes"] if p["gradient"] is not None
        )
        row["gradient"][0][0] += 0.01
    elif field == "work":
        changed[0]["work"]["new_objective_evaluations"] += 1
        changed[0]["trace"]["retained_work"] = copy.deepcopy(changed[0]["work"])
    summary = q.report_from_records(plan, changed)
    if field == "report":
        summary["new_solver_candidate"] = True
    q.write_json(output / "records.json", changed)
    q.write_json(output / "report.json", summary)
    hashes = json.loads((output / "files.json").read_text())
    for name in ("records.json", "report.json"):
        hashes[name] = q.digest((output / name).read_bytes())
    q.write_json(output / "files.json", hashes)
    before = len(calls)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
    if field != "report":
        assert len(calls) == before + 1  # Defeats hashes and recomputed summaries.
    assert result["new_solver_candidate"] is False


@pytest.mark.parametrize(
    "alteration", ["command", "dtype", "extra_array", "extra_file", "input"]
)
def test_verifier_rejects_changed_array_schema_roster_or_inputs(saved_run, alteration):
    _, trials, _, _, output, _ = saved_run
    hashes = json.loads((output / "files.json").read_text())
    if alteration in {"command", "dtype", "extra_array"}:
        values = copy.deepcopy(trials[0])
        if alteration == "command":
            values["commands"][0, 0, 0, 0] = np.float32(1e-10)
        elif alteration == "dtype":
            values["commands"] = values["commands"].astype(np.float64)
        else:
            values["unexpected"] = np.zeros(1)
        np.savez_compressed(output / "trial-0/paired.npz", **values)
        hashes["trial-0/paired.npz"] = q.digest(
            (output / "trial-0/paired.npz").read_bytes()
        )
        q.write_json(output / "files.json", hashes)
    elif alteration == "extra_file":
        (output / "extra.json").write_text("{}")
    else:
        (output / "inputs/parent.json").write_bytes(b"changed parent")
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
