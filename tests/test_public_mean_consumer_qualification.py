"""Bounded consumer adjudication/provenance tests; no fitted model or trial."""

import copy
import json

import numpy as np
import pytest

from glassbox.experimental import public_mean_consumer_qualification as qualification


def _binding():
    return dict(
        protocol_sha256="p",
        oracle_root="/old-oracle",
        oracle_commit="o",
        oracle_source_sha256={"oracle.py": "a"},
        dart_root="/dart",
        consumer_source_sha256={"consumer.py": "b"},
        consumer_distribution_versions={"glassbox": "v4"},
        consumer_exporter_sha256="e",
        consumer_prefreeze_inventory_sha256="i",
        interpreter="/usr/bin/python",
        interpreter_sha256="x",
        runtime={"numpy": "v"},
        machine="arm64",
        public_root="/original",
        implementation_commit="old",
        public_source_sha256={"src/public.py": "u", "tests/old.py": "t"},
    )


def test_successor_may_extend_sources_without_relabelling_original():
    old = _binding()
    new = copy.deepcopy(old)
    new.update(public_root="/evaluator", implementation_commit="new")
    new["public_source_sha256"]["src/qualification.py"] = "q"
    result = qualification.associate_bindings(old, new)
    assert result["original_root"] == "/original"
    assert result["evaluator_root"] == "/evaluator"
    assert result["unchanged_original_sources"] == 2
    assert result["added_sources"] == ["src/qualification.py"]
    assert result["consumer_replay_uses_original_root"]


@pytest.mark.parametrize(
    "change",
    ["changed_old_source", "missing_old_test", "consumer", "runtime", "oracle"],
)
def test_successor_refuses_changes_to_any_old_bound_source_or_runtime(change):
    old = _binding()
    new = copy.deepcopy(old)
    if change == "changed_old_source":
        new["public_source_sha256"]["src/public.py"] = "changed"
    elif change == "missing_old_test":
        del new["public_source_sha256"]["tests/old.py"]
    elif change == "consumer":
        new["consumer_source_sha256"]["consumer.py"] = "changed"
    elif change == "runtime":
        new["runtime"]["numpy"] = "changed"
    else:
        new["oracle_root"] = "/different"
    with pytest.raises(ValueError):
        qualification.associate_bindings(old, new)


def _packet_and_cases():
    packet = {"models": [], "recordings": [], "queries": []}
    for identity in ("crazyflow", "cascade"):
        model = dict(
            id=identity,
            path=identity + "-model.npz",
            sha256=identity + "model",
            fingerprint="a" * 64,
            history_steps=2,
            horizon_steps=3,
            contract={
                "state_channels": ["x [m]", "v [m/s]"],
                "input_channels": ["u [1]"],
                "dt_s": 0.05,
            },
        )
        recording = dict(
            id=identity + "-recordings",
            model_id=identity,
            path=identity + "-recordings.npz",
            sha256=identity + "records",
            parents=[],
        )
        packet["models"].append(model)
        packet["recordings"].append(recording)
        for i in range(8):
            parent = f"{identity}-parent-{i}"
            recording["parents"].append(
                dict(
                    recording_id=parent,
                    segments=[
                        dict(
                            segment_id="whole", start_row=0, state_rows=10, input_rows=9
                        )
                    ],
                    source_parent_sha256="parent",
                    source_record_entry_sha256="entry",
                )
            )
            kind = "factual" if i % 2 == 0 else "response"
            packet["queries"].append(
                dict(
                    model_id=identity,
                    recordings_id=recording["id"],
                    parent=parent,
                    id=f"query-{i}",
                    segment_id="whole",
                    source_origin=3,
                    kind=kind,
                    scope="primary" if i < 4 else "heading_shift",
                    dt_s=0.05,
                    horizon_steps=3,
                    branches=["factual"]
                    if kind == "factual"
                    else ["intervened", "factual"],
                    path=f"{identity}-{i}.npz",
                    sha256="query",
                    source_query_sha256="source",
                )
            )
    packet["queries"].sort(key=lambda q: (q["model_id"], q["parent"], q["id"]))
    cases = []
    for q in packet["queries"]:
        model = next(m for m in packet["models"] if m["id"] == q["model_id"])
        recording = next(
            r for r in packet["recordings"] if r["id"] == q["recordings_id"]
        )
        parent = next(
            p for p in recording["parents"] if p["recording_id"] == q["parent"]
        )
        for branch in q["branches"]:
            cases.append(
                dict(
                    id="/".join((q["model_id"], q["parent"], q["id"], branch)),
                    source="public_archive",
                    model_sha256=model["sha256"],
                    model_fingerprint=model["fingerprint"],
                    dt_s=0.05,
                    history_steps=2,
                    horizon=3,
                    maximum_horizon=3,
                    source_identity=dict(
                        query=copy.deepcopy(q),
                        model=copy.deepcopy(model),
                        branch=branch,
                        future_array_key="factual_inputs"
                        if q["kind"] == "response" and branch == "factual"
                        else "future_inputs",
                        consumer_packet_manifest_sha256="packet",
                        implementation_manifest_sha256="binding",
                        data_seal_sha256=q["model_id"] + "seal",
                        recording_parent=copy.deepcopy(parent),
                        recordings_archive={
                            k: v for k, v in recording.items() if k != "parents"
                        },
                    ),
                )
            )
    return packet, cases


def _mapped(packet, cases):
    return qualification.map_numeric_cases(
        packet,
        cases,
        input_sha256="packet",
        old_binding_sha256="binding",
        data_seals={s: s + "seal" for s in ("crazyflow", "cascade")},
    )


def test_numeric_mapping_keeps_all24_declared_branches_including_paired_factual():
    packet, cases = _packet_and_cases()
    mapped = _mapped(packet, [{"source": "artificial"}, *cases])
    assert len(mapped) == 24
    assert sum(future == "factual_inputs" for _, _, _, future in mapped) == 8
    assert len({case["id"] for case, _, _, _ in mapped}) == 24


@pytest.mark.parametrize(
    "change",
    ["missing", "order", "branch", "packet", "binding", "model", "data", "future"],
)
def test_numeric_mapping_cannot_reselect_or_misroute_consumer_input(change):
    packet, cases = _packet_and_cases()
    if change == "missing":
        cases.pop()
    elif change == "order":
        cases.reverse()
    elif change == "branch":
        cases[0]["source_identity"]["branch"] = "intervened"
    elif change == "packet":
        cases[0]["source_identity"]["consumer_packet_manifest_sha256"] = "other"
    elif change == "binding":
        cases[0]["source_identity"]["implementation_manifest_sha256"] = "other"
    elif change == "model":
        cases[0]["model_sha256"] = "other"
    elif change == "data":
        cases[0]["source_identity"]["data_seal_sha256"] = "other"
    else:
        cases[0]["source_identity"]["future_array_key"] = "factual_inputs"
    with pytest.raises(ValueError):
        _mapped(packet, cases)


def _means():
    physical = np.arange(6, dtype=np.float32).reshape(3, 2)
    derivative = np.ones((3, 2), dtype=np.float32) * 0.25
    actual = {
        "eager": physical.copy(),
        "compiled": physical.copy(),
        "jvp": derivative.copy(),
    }
    public = {
        "actual__singles": physical[None],
        "actual__jit_singles": physical[None],
        "actual__first_command_jvp": derivative,
    }
    oracle = {
        "quantized_inputs__singles": physical.astype(np.float64)[None],
        "quantized_inputs__jit_singles": physical.astype(np.float64)[None],
        "quantized_inputs__first_command_jvp": derivative.astype(np.float64),
    }
    return actual, public, oracle


def test_saved_unbatched_same_path_and_native_tangent_oracle_are_separate_checks():
    actual, public, oracle = _means()
    assert all(
        c["passed"]
        for c in qualification.mean_checks(
            actual, public, oracle, np.zeros(2), np.array([2.0, 0.1])
        ).values()
    )
    oracle["quantized_inputs__first_command_jvp"][0, 0] += 0.2
    checks = qualification.mean_checks(
        actual, public, oracle, np.zeros(2), np.array([2.0, 0.1])
    )
    assert not checks["jvp_oracle"]["passed"]
    assert checks["eager_oracle"]["passed"] and checks["jvp_public"]["passed"]
    actual["eager"][0, 0] += 0.01
    with pytest.raises(ValueError, match="same-path"):
        qualification.mean_checks(actual, public, oracle, np.zeros(2), np.ones(2))


@pytest.mark.parametrize("case", qualification.NEGATIVES)
def test_each_negative_changes_one_intended_semantic_boundary_and_preserves_original_manifest_until_reseal(
    tmp_path, case
):
    packet, _ = _packet_and_cases()
    for q in packet["queries"]:
        arrays = dict(
            past_states=np.zeros((3, 2)),
            past_inputs=np.zeros((2, 1)),
            future_inputs=np.zeros((3, 1)),
        )
        if q["kind"] == "response":
            arrays["factual_inputs"] = np.zeros((3, 1))
        np.savez_compressed(tmp_path / q["path"], **arrays)
        q["sha256"] = qualification._sha(tmp_path / q["path"])
    path = tmp_path / "INPUT.json"
    path.write_text(json.dumps(packet))
    original = path.read_bytes()
    arrays_before = {
        q["path"]: qualification._sha(tmp_path / q["path"]) for q in packet["queries"]
    }
    result = qualification.mutate_packet(tmp_path, case)
    assert path.read_bytes() == original
    assert result["original_manifest_sha256"] == qualification._sha(path)
    changed = result["packet"]
    if case == "wrong_fingerprint":
        assert changed["models"][0]["fingerprint"] != packet["models"][0]["fingerprint"]
    elif case == "swapped_channels":
        assert changed["models"][0]["contract"]["state_channels"] == list(
            reversed(packet["models"][0]["contract"]["state_channels"])
        )
    elif case == "changed_dt":
        assert changed["queries"][0]["dt_s"] == 0.1
    elif case == "history_boundary":
        assert changed["queries"][0]["source_origin"] == 1
    else:
        q = next(q for q in changed["queries"] if q["kind"] == "factual")
        assert (
            q["sha256"]
            == qualification._sha(tmp_path / q["path"])
            != arrays_before[q["path"]]
        )
        assert (
            qualification._arrays(tmp_path / q["path"])["future_inputs"][0, 0] == 0.125
        )
        assert "recording bytes" in result["expected_semantic_error"]
    changed_payloads = [
        name
        for name, sha in arrays_before.items()
        if qualification._sha(tmp_path / name) != sha
    ]
    assert len(changed_payloads) == (1 if case == "future_command" else 0)


def test_packet_anchor_rejected_before_reading_archive_or_forecast(tmp_path):
    (tmp_path / "INPUT.json").write_text("{}")
    with pytest.raises(ValueError, match="external consumer input anchor"):
        qualification.packet_manifest(tmp_path / "INPUT.json", "0" * 64)


def test_consumer_qualification_refuses_output_inside_original_evidence(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        qualification.run(
            tmp_path / "inside",
            old_binding_path="unused",
            old_binding_sha256="unused",
            evaluator_binding_path="unused",
            evaluator_binding_sha256="unused",
            input_manifest=tmp_path / "INPUT.json",
            input_sha256="unused",
            consumer_directory=tmp_path,
            numeric_manifest=tmp_path / "numeric.json",
            numeric_manifest_sha256="unused",
            numeric_directory=tmp_path,
            numeric_run_sha256="unused",
            consumer_result_sha256="unused",
            data_roots={},
            data_seals={},
        )
