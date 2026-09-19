"""Saved-only finalizer semantics; mocked data/consumer, no model forecasts."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "docs/diagnostics/public-mean-corrected-finalization.py"
SPEC = importlib.util.spec_from_file_location("corrected_finalization", DRIVER)
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_optimized_python_rejected():
    p = subprocess.run(
        [sys.executable, "-O", str(DRIVER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode and "without optimization" in p.stderr


@pytest.mark.parametrize(
    "mutation", [None, "path", "seal", "scientific", "extra", "missing"]
)
def test_old_wrapper_is_validated_not_silently_discarded(mutation):
    actual = {"comparison": {"pass": True, "limit": 0.9}, "coverage_is_separate": True}
    stages = {
        "crazyflow": {"path": "old-cf", "seal_sha256": "cf"},
        "cascade": {"path": "old-c", "seal_sha256": "c"},
    }
    saved = copy.deepcopy(dict(actual, input_stages=stages))
    if mutation == "path":
        saved["input_stages"]["crazyflow"]["path"] = "new-cf"
    elif mutation == "seal":
        saved["input_stages"]["cascade"]["seal_sha256"] = "wrong"
    elif mutation == "scientific":
        saved["comparison"]["limit"] = 1.1
    elif mutation == "extra":
        saved["ignored_scientific_flag"] = False
    elif mutation == "missing":
        del saved["coverage_is_separate"]
    if mutation:
        with pytest.raises(AssertionError):
            subject.compare_decision(actual, saved, stages)
    else:
        result = subject.compare_decision(actual, saved, stages)
        assert result["original_input_stage_association_exact"]
        assert result["all_scientific_decision_fields_exact"]
        assert saved["input_stages"] == stages


def test_prior_corrected_source_cannot_change():
    previous = dict(
        format="f",
        protocol_sha256="p",
        runtime={},
        machine="arm",
        interpreter_sha256="python",
        oracle_commit="old",
        oracle_source_sha256={},
        consumer_source_sha256={},
        implementation_commit="first",
        public_source_sha256={"src/core.py": "same"},
    )
    current = copy.deepcopy(previous)
    current["implementation_commit"] = "new"
    current["public_source_sha256"]["tests/new.py"] = "added"
    assert (
        subject.source_continuity(previous, current)["unchanged_prior_bound_sources"]
        == 1
    )
    current["public_source_sha256"]["src/core.py"] = "changed"
    with pytest.raises(AssertionError):
        subject.source_continuity(previous, current)


def test_saved_reduction_precedes_only_missing_consumer(monkeypatch, tmp_path):
    from glassbox.experimental import public_mean_physics as physics

    root = tmp_path / "prefix"
    out = tmp_path / "new"
    source = tmp_path / "source"
    policy = {
        "failed_prefix": {"sha256": "failed-external"},
        "prefix_binding": {"sha256": "old-inference"},
    }
    physical_policy = {
        "original_fit_binding_sha256": "old-fit",
        "simulators": {},
        "expected_counts": {},
    }
    current = {"public_root": str(source)}
    put(source / "docs/harness/public-mean-qualification-v1.json", {})
    rows, queries, reports, inputs = {}, {}, {}, {}
    for sim in ("crazyflow", "cascade"):
        original = tmp_path / "original" / sim
        fit = tmp_path / "fit" / sim
        fit.mkdir(parents=True)
        (fit / "model.npz").write_bytes(sim.encode())
        rows[sim] = [{"query": sim, "value": 1}]
        queries[sim] = [{"id": sim}]
        physical_policy["simulators"][sim] = {
            "evaluation_root": str(original),
            "evaluation_sha256": sim + "-seal",
        }
        reduced = dict(rows=rows[sim], directions=[], summary={"same": True})
        for name, value in reduced.items():
            put(original / (name + ".json"), value)
            put(root / sim / (name + ".json"), value)
        inputs[sim] = dict(
            reference_root="reference",
            data_root=sim,
            data_sha256=sim + "-data",
            flight_fit_root=str(fit),
            flight_fit_sha256=sim + "-fit",
        )
        reports[sim] = {
            role: {"model_sha256": subject.sha(fit / "model.npz")}
            for role in ("public32", "public64")
        }
    decision = {"scientific": "identical"}
    stages = {
        sim: {"path": p["evaluation_root"], "seal_sha256": p["evaluation_sha256"]}
        for sim, p in physical_policy["simulators"].items()
    }
    put(tmp_path / "old-decision.json", dict(decision, input_stages=stages))
    physical_policy["old_decision"] = {
        "path": str(tmp_path / "old-decision.json"),
        "sha256": subject.sha(tmp_path / "old-decision.json"),
    }
    put(root / "seal.json", {"files": {"proof": "pinned"}})
    calls = []

    def score(data, prefix, sim, p):
        calls.append(("saved_score", sim))
        return dict(rows=rows[sim], directions=[], summary={"same": True})

    def reduce(actual, p, q, *, queries):
        assert actual == rows and set(queries) == set(rows)
        calls.append(("decision",))
        return decision

    def consumer(*args):
        assert [x[0] for x in calls] == ["saved_score", "saved_score", "decision"]
        calls.append(("consumer",))
        return {"exact_arrays": 168}

    pe = SimpleNamespace(
        REFERENCE_SHA256="old-reference",
        sealed=lambda path, *a: {
            "implementation_manifest_sha256": "old-fit",
            "status": "complete",
            "simulator": Path(path).name,
        },
        resolved=lambda *a: {},
        score=score,
        _planned=lambda data, sim, p: ([], queries[sim], {}),
        reduce_decision=reduce,
        inventory=lambda p: {"proof": "pinned"},
    )
    bridge = SimpleNamespace(
        old_inputs=lambda pe, p, sim: (
            Path(physical_policy["simulators"][sim]["evaluation_root"]),
            inputs[sim],
        ),
        replay_consumer=consumer,
        check_counts=lambda *a: {"metric_rows": 2},
    )
    args = SimpleNamespace(
        output=out, binding_sha256="final-binding", policy_sha256="frozen"
    )
    monkeypatch.setattr(
        subject,
        "authenticate",
        lambda args: (
            policy,
            current,
            {},
            "source-exact",
            bridge,
            args,
            physical_policy,
            pe,
        ),
    )
    monkeypatch.setattr(subject, "prefix_evidence", lambda *a: (root, reports, 99))
    monkeypatch.setattr(
        physics,
        "verify_data",
        lambda path, sha: {"implementation_sha256": "old-fit", "simulator": path},
    )
    result = subject.run(args)
    assert result["physical_and_consumer_regression_passed"]
    assert (
        result["new_flight_forecasts"]
        == result["new_development_forecasts"]
        == result["new_fits"]
        == 0
    )
    assert calls[-1] == ("consumer",) and len(calls) == 4
    assert subject.read(out / "decision.json") == decision
    assert (
        subject.read(out / "decision-comparison.json")["original_input_stages"]
        == stages
    )
