"""Toy data/source orchestration tests: no simulation, fitting or forecasts."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import independent_training_data as data


def protocol():
    return data.read_protocol()


def test_complete_literal_rosters_derive_from_roles_and_exclude_prior_cohorts():
    p = protocol()
    before = deepcopy(p)
    roster = data.validate_roster(p)
    original = data._dependency(p, "original_flight")
    indexed = {r["id"]: r for r in original["recordings"]}
    assert len(roster["added_training"]) == 144
    assert len(roster["confirmation"]) == 168
    for row in roster["added_training"]:
        old = indexed[row["source_condition_parent"]]
        assert row["seed"] - old["seed"] == 20000000
        assert row["cell"] == old["cell"]
    for kind in ("training", "confirmation"):
        view = data.physical_view(p, kind)
        assert view["cells"] == original["cells"]
        assert view["generation"] == original["generation"]
        assert view["evaluation"] == original["evaluation"]
    assert p == before
    view = data.physical_view(p, "training")
    assert all(
        len(view["planned_automatic_roles"][s]["training"]) == 72
        for s in data.SIMULATORS
    )
    assert all(
        view["planned_automatic_roles"][s]["development"] == [] for s in data.SIMULATORS
    )


@pytest.mark.parametrize("defect", ["drop", "reorder", "development", "old_seed"])
def test_roster_rejects_coherent_literal_changes(monkeypatch, defect):
    p = protocol()
    original_dependency = data._dependency
    roster = deepcopy(original_dependency(p, "roster"))
    if defect == "drop":
        roster["added_training"].pop()
    elif defect == "reorder":
        roster["added_training"][:2] = roster["added_training"][:2][::-1]
    elif defect == "development":
        roles = roster["original_roles"]["crazyflow"]
        roles["training"][0], roles["development"][0] = (
            roles["development"][0],
            roles["training"][0],
        )
    else:
        roster["confirmation"][0]["seed"] -= 1000000
    monkeypatch.setattr(
        data,
        "_dependency",
        lambda p, name: roster if name == "roster" else original_dependency(p, name),
    )
    with pytest.raises(ValueError):
        data.validate_roster(p)


@pytest.mark.parametrize("dt,minimum", [(0.01, 75), (0.05, 15)])
def test_support_keeps_all_failure_reasons_without_shorter_window_fallback(dt, minimum):
    rows = [
        dict(
            id=str(i),
            metadata={},
            validity=dict(valid_initial=initial, valid_transitions=n, reason=reason),
        )
        for i, (initial, n, reason) in enumerate(
            [
                (True, minimum, "later_failure"),
                (True, minimum - 1, "early_failure"),
                (False, 0, "setup_failure"),
            ]
        )
    ]
    result = data.support(rows, dt)
    assert result["planned"] == ["0", "1", "2"]
    assert result["admitted"] == ["0"]
    assert [r["failure_reason"] for r in result["unavailable"]] == [
        "early_failure",
        "setup_failure",
    ]
    assert not result["all_planned_usable"]


def candidate_anchors(tmp_path):
    anchors = {}
    for simulator in data.SIMULATORS:
        directory = tmp_path / simulator
        directory.mkdir()
        data.write(directory / "details.json", {"no_fit_fixture": True})
        data.write(
            directory / "run.json",
            dict(
                format="glassbox-independent-training-candidate-v1",
                simulator=simulator,
                protocol_sha256=data.PROTOCOL_SHA256,
                implementation_sha256="binding",
                status="complete"
                if simulator == "crazyflow"
                else "preparation_unavailable",
                files=data.inventory(directory),
            ),
        )
        anchors[simulator] = {
            "path": str(directory / "run.json"),
            "sha256": data.digest(directory / "run.json"),
        }
    return anchors


def test_confirmation_requires_both_terminal_authenticated_candidate_outcomes(tmp_path):
    anchors = candidate_anchors(tmp_path)
    result = data.candidate_prerequisites(anchors)
    assert result["cascade"]["status"] == "preparation_unavailable"
    with pytest.raises(ValueError, match="both candidate"):
        data.candidate_prerequisites({"crazyflow": anchors["crazyflow"]})
    (tmp_path / "cascade/details.json").write_text("{}")
    with pytest.raises(ValueError, match="payload inventory"):
        data.candidate_prerequisites(anchors)


def toy_arrays(entry, rule, dt=0.01):
    steps, start, block = round(3 / dt), round(0.75 / dt), round(0.1 / dt)
    material = json.dumps(
        dict(namespace=rule["id"], parent=entry["id"], seed=entry["seed"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    hashed = hashlib.sha256(material.encode()).digest()
    seed = int.from_bytes(hashed, "big")
    signs = (
        2
        * np.random.Generator(np.random.PCG64(seed)).integers(
            0, 2, size=(23, 1), dtype=np.int64
        )
        - 1
    )
    indices = np.full(steps, -1, np.int64)
    indices[start:] = np.arange(steps - start) // block
    assigned = np.zeros((steps, 1))
    assigned[start:] = 0.05 * signs[indices[start:]]
    pilot = np.full((steps, 1), 0.98)
    commands = pilot.copy()
    commands[start:] = np.clip(pilot[start:] + assigned[start:], 0, 1)
    clipped = (pilot + assigned > 1) | (pilot + assigned < 0)
    clipped[:start] = False
    states = np.zeros((steps + 1, 13))
    states[:, 2] = 1.0
    states[:, 6] = 1.0
    arrays = dict(
        states=states,
        full_states=states.copy(),
        commands=commands,
        time_s=np.arange(steps + 1) * dt,
        excitation_pilot_commands=pilot,
        excitation_assigned_delta=assigned,
        excitation_block_index=indices,
        excitation_realized_delta=commands - pilot,
        excitation_clipping_residual=commands - (pilot + assigned),
        excitation_clipped=clipped,
    )
    detail = dict(
        id=rule["id"],
        role="training",
        rng="numpy.PCG64",
        seed_material=material,
        seed_sha256=hashed.hex(),
        seed_integer=seed,
        signs=signs.tolist(),
        start_s=0.75,
        block_s=0.1,
        range_fraction=0.05,
        dt_s=dt,
        steps=steps,
        start_index=start,
        block_steps=block,
        blocks=23,
        lower=[0.0],
        upper=[1.0],
    )
    return arrays, {"command_excitation": detail}


def toy_training(tmp_path, monkeypatch):
    p = protocol()
    view = data.physical_view(p, "training")
    entry = view["recordings"][0]
    view["recordings"] = [entry]
    monkeypatch.setattr(data, "physical_view", lambda p, kind: deepcopy(view))
    directory = tmp_path / "data"
    directory.mkdir()
    arrays, metadata = toy_arrays(entry, view["collection_intervention"])
    record = dict(
        entry,
        prefix="parents/" + entry["id"],
        metadata=metadata,
        validity=dict(
            valid_initial=True, valid_transitions=300, failure_index=None, reason=None
        ),
    )
    configuration = {
        "dt_s": 0.01,
        "spec": {
            "vehicle": {"configuration_id": "toy"},
            "channels": [
                {"kind": "control", "name": "u", "minimum": 0.0, "maximum": 1.0}
            ],
        },
    }
    for relative, value in (
        ("resolved-physical.json", view),
        ("planned-records.json", [entry]),
        ("records.json", [record]),
        ("queries.json", []),
        ("configuration.json", configuration),
        (record["prefix"] + ".json", record),
        (
            "recording-contract.json",
            data._recording_contract(configuration),
        ),
    ):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data.write(path, value)
    np.savez(directory / (record["prefix"] + ".npz"), **arrays)
    path = directory / ("recordings/" + entry["id"] + ".npz")
    path.parent.mkdir(parents=True)
    np.savez(path, **data._recording_values(arrays, 300))
    seal = dict(
        format=data.FORMAT,
        protocol_sha256=data.PROTOCOL_SHA256,
        implementation_sha256="binding",
        simulator="crazyflow",
        kind="training",
        parents=1,
        queries=0,
        arrays=data._array_count(directory),
        candidate_outcomes={},
        support=data.support([record], 0.01),
        files=data.inventory(directory, "seal.json"),
    )
    data.write(directory / "seal.json", seal)
    return directory, p


def reseal(directory):
    seal = data.read(directory / "seal.json")
    seal["files"] = data.inventory(directory, "seal.json")
    (directory / "seal.json").unlink()
    data.write(directory / "seal.json", seal)
    return data.digest(directory / "seal.json")


def test_saved_training_adapter_uses_issued_commands_and_no_excitation_features(
    tmp_path, monkeypatch
):
    directory, p = toy_training(tmp_path, monkeypatch)
    sha = data.digest(directory / "seal.json")
    data.verify_data(directory, sha, protocol=p)
    collection = data.load_added_recordings(directory, sha)
    assert len(collection.segments) == 1
    segment = collection.segments[0]
    assert segment.excitation is None
    assert segment.states.shape == (301, 15)
    assert segment.inputs.shape == (300, 1)
    assert np.any(segment.inputs != 0.98)
    with pytest.raises(ValueError, match="signal/timing contract"):
        data.load_added_recordings(directory, sha, {"different": True})


@pytest.mark.parametrize(
    "defect",
    [
        "command",
        "assigned",
        "pilot",
        "bounds",
        "extra_channel",
        "observations",
        "missing_parent",
    ],
)
def test_semantically_resealed_training_defects_reject(tmp_path, monkeypatch, defect):
    directory, p = toy_training(tmp_path, monkeypatch)
    row = data.read(directory / "records.json")[0]
    if defect in ("command", "assigned", "pilot"):
        path = directory / (row["prefix"] + ".npz")
        values = data._load_npz(path)
        key = {
            "command": "commands",
            "assigned": "excitation_assigned_delta",
            "pilot": "excitation_pilot_commands",
        }[defect]
        values[key][90, 0] += 0.02
        np.savez(path, **values)
    elif defect == "bounds":
        path = directory / "configuration.json"
        value = data.read(path)
        value["spec"]["channels"][0]["maximum"] = 2.0
        path.write_text(json.dumps(value))
    elif defect in ("extra_channel", "observations"):
        path = directory / ("recordings/" + row["id"] + ".npz")
        values = data._load_npz(path)
        if defect == "extra_channel":
            values["assigned"] = np.ones((300, 1))
        else:
            values["states"][0, 0] += 0.1
        np.savez(path, **values)
    else:
        (directory / "records.json").write_text("[]")
    with pytest.raises(ValueError):
        data.verify_data(directory, reseal(directory), protocol=p)


def test_zero_usable_raises_only_after_complete_roster_and_support_validation(
    tmp_path, monkeypatch
):
    directory, _ = toy_training(tmp_path, monkeypatch)
    rows = data.read(directory / "records.json")
    rows[0]["validity"].update(
        valid_transitions=74, failure_index=75, reason="nonfinite_state"
    )
    raw_path = directory / (rows[0]["prefix"] + ".npz")
    raw = data._load_npz(raw_path)
    raw["states"][75:] = np.nan
    raw["full_states"][75:] = np.nan
    np.savez(raw_path, **raw)
    for path in (directory / "records.json", directory / (rows[0]["prefix"] + ".json")):
        path.write_text(json.dumps(rows if path.name == "records.json" else rows[0]))
    seal = data.read(directory / "seal.json")
    seal["support"] = data.support(rows, 0.01)
    (directory / "seal.json").write_text(json.dumps(seal))
    sha = reseal(directory)
    with pytest.raises(data.PreparationUnavailable) as caught:
        data.load_added_recordings(directory, sha)
    assert caught.value.details["unavailable"][0]["valid_transitions"] == 74


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_supervisor_retains_failure_prefix_and_never_retries(
    tmp_path, monkeypatch, failure
):
    p = protocol()
    monkeypatch.setattr(
        data,
        "authenticate",
        lambda *args: (
            p,
            {"oracle_root": "/pinned/old", "interpreter": "/pinned/python"},
        ),
    )
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("partial child evidence\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 14400)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(data.subprocess, "run", execute)
    output = tmp_path / "attempt1"
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        data.collect_training(
            "crazyflow",
            output,
            binding_path=tmp_path / "binding",
            binding_sha256="binding",
        )
    result = data.read(output / "run.json")
    assert result["status"] == (
        "failed" if failure == "exit" else "hard_timeout_incomplete"
    )
    assert result["files"] == data.inventory(output)
    assert len(calls) == 1
    assert "--request-sha256" in calls[0][0]
    assert (
        calls[0][1]["env"]["PYTHONPATH"]
        == "/pinned/old/src:/private/tmp/glassbox-cascade-e8f6ba6/src"
    )
    assert "partial child evidence" in (output / "worker.log").read_text()
    with pytest.raises(FileExistsError):
        data.collect_training(
            "crazyflow",
            output,
            binding_path=tmp_path / "binding",
            binding_sha256="binding",
        )
    assert len(calls) == 1


def test_worker_request_hash_rejects_before_imports_or_numerical_work(
    tmp_path, monkeypatch
):
    request = tmp_path / "request.json"
    data.write(request, {})
    monkeypatch.setattr(
        data, "_worker_identity", lambda _: pytest.fail("identity must not execute")
    )
    with pytest.raises(ValueError, match="external worker request"):
        data._worker(request, "0" * 64)


def test_confirmation_query_plan_includes_all_slots_even_failed_parents():
    p = data.physical_view(protocol(), "confirmation")
    first = p["recordings"][0]
    row = dict(first, validity=dict(valid_initial=False, valid_transitions=0))
    config = {"dt_s": 0.01, "spec": {"channels": [{"kind": "control"}] * 4}}
    queries = data._confirmation_queries([row], p, config)
    assert len(queries) == 26
    assert sum(q["kind"] == "factual" for q in queries) == 10
    assert sum(q["kind"] == "response" for q in queries) == 16
    assert all(q["history_eligible"] is False for q in queries)
    assert queries[0]["id"] == "factual-0050"
    assert queries[-1]["id"] == "response-0200-3-+1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_sha256", "wrong"),
        ("implementation_sha256", "wrong"),
        ("status", "incomplete"),
    ],
)
def test_candidate_prerequisites_reject_wrong_stage_association(tmp_path, field, value):
    anchors = candidate_anchors(tmp_path)
    path = Path(anchors["cascade"]["path"])
    report = data.read(path)
    report[field] = value
    path.write_text(json.dumps(report))
    anchors["cascade"]["sha256"] = data.digest(path)
    with pytest.raises(ValueError):
        data.candidate_prerequisites(anchors, binding_sha256="binding")


def test_reported_array_count_recomputed_without_loading_array_values(
    tmp_path, monkeypatch
):
    directory, p = toy_training(tmp_path, monkeypatch)
    seal = data.read(directory / "seal.json")
    assert seal["arrays"] == 12
    seal["arrays"] += 1
    (directory / "seal.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="data array count"):
        data.verify_data(directory, data.digest(directory / "seal.json"), protocol=p)
