"""Bounded source/array/control-flow checks; no fitted model or simulation."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import public_mean_physical_evaluation as physical

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "docs/diagnostics/public-mean-corrected-physical-replay.py"
SPEC = importlib.util.spec_from_file_location("corrected_physical_replay", DRIVER)
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_optimized_python_rejected_before_import_or_execution():
    result = subprocess.run(
        [sys.executable, "-O", str(DRIVER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Integrity assertions require Python without optimization" in result.stderr


def binding_pair():
    old = dict(
        runtime={"python": "pinned"},
        machine="arm64",
        interpreter_sha256="python",
        oracle_commit="oracle",
        oracle_source_sha256={"old": "same"},
        consumer_source_sha256={"consumer": "same"},
        implementation_commit="old",
        public_source_sha256={
            "src/glassbox/_sequence_model.py": "old-core",
            "src/glassbox/learner.py": "same",
            "tests/test_old.py": "test",
        },
    )
    current = copy.deepcopy(old)
    current["implementation_commit"] = "current"
    current["public_source_sha256"]["src/glassbox/_sequence_model.py"] = "new-core"
    policy = dict(
        corrected_core_sha256="new-core",
        allowed_common_source_differences={
            "src/glassbox/_sequence_model.py": {
                "old": "old-core",
                "current": "new-core",
            }
        },
    )
    return old, current, policy


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "unlisted_source",
        "wrong_core",
        "runtime",
        "deleted_source",
        "concealed_old_change",
    ],
)
def test_source_bridge_accepts_only_exact_declared_pairs(mutation):
    old, current, policy = binding_pair()
    if mutation == "unlisted_source":
        current["public_source_sha256"]["src/glassbox/learner.py"] = "changed"
    elif mutation == "wrong_core":
        current["public_source_sha256"]["src/glassbox/_sequence_model.py"] = (
            "third-core"
        )
    elif mutation == "runtime":
        current["runtime"]["python"] = "other"
    elif mutation == "deleted_source":
        del current["public_source_sha256"]["src/glassbox/learner.py"]
    elif mutation == "concealed_old_change":
        old["public_source_sha256"]["src/glassbox/_sequence_model.py"] = "other-old"
    if mutation is None:
        result = subject.source_association(old, current, policy)
        assert result["old_commit"] == "old" and result["current_commit"] == "current"
    else:
        with pytest.raises(AssertionError):
            subject.source_association(old, current, policy)


@pytest.mark.parametrize(
    "field",
    [
        None,
        "binding_sha256",
        "data_sha256",
        "flight_fit_sha256",
        "data_binding_sha256",
        "flight_binding_sha256",
    ],
)
def test_original_associations_cannot_be_replaced(field, tmp_path):
    inputs = dict(
        binding_sha256="old",
        data_sha256="data",
        flight_fit_sha256="fit",
        data_binding_sha256="producer",
        flight_binding_sha256="producer",
        simulator="crazyflow",
        replay=False,
    )
    if field:
        inputs[field] = "substitute"
    seal = dict(files={}, status="complete", simulator="crazyflow", inputs=inputs)
    pe = SimpleNamespace(sealed=lambda *a: seal, inventory=lambda *a: {})
    policy = dict(
        old_binding={"sha256": "old"},
        original_fit_binding_sha256="producer",
        simulators={
            "crazyflow": dict(
                evaluation_root=str(tmp_path),
                evaluation_sha256="external-old",
                data_sha256="data",
                flight_fit_sha256="fit",
            )
        },
    )
    if field:
        with pytest.raises(AssertionError):
            subject.old_inputs(pe, policy, "crazyflow")
    else:
        _, result = subject.old_inputs(pe, policy, "crazyflow")
        assert result["binding_sha256"] == "old"
        assert "replay" not in result and "simulator" not in result


def test_original_validator_receives_only_original_binding_no_forecast(
    monkeypatch, tmp_path
):
    calls = []
    pe = SimpleNamespace(
        __file__=str(tmp_path / "old/src/physical.py"),
        verify=lambda *a, **kw: calls.append(kw) or {"fits": 0, "simulations": 0},
    )
    inputs = {"binding_path": "old.json", "binding_sha256": "old-sha"}
    policy = {"simulators": {"crazyflow": {"evaluation_sha256": "saved-evaluation"}}}
    monkeypatch.setattr(
        subject,
        "authenticate",
        lambda args: (policy, {}, {"public_root": str(tmp_path / "old")}, pe),
    )
    monkeypatch.setattr(
        subject, "old_inputs", lambda *a: (tmp_path / "old-evidence", inputs)
    )
    subject.worker(
        SimpleNamespace(
            phase="validate-old", simulator="crazyflow", output=tmp_path / "result"
        )
    )
    assert calls == [dict(expected_sha256="saved-evaluation", **inputs)]
    assert "replay_output" not in calls[0]


def worker_fixture(monkeypatch, tmp_path):
    import jax

    from glassbox import LearnedDynamics

    previous, output, fit, data = [
        tmp_path / name for name in ("old", "new", "fit", "data")
    ]
    for path in (previous / "public64", output, fit, data):
        path.mkdir(parents=True)
    (fit / "model.npz").write_bytes(b"not-loaded-mocked-model")
    query = dict(
        id="response",
        kind="response",
        parent="parent",
        history_eligible=True,
        path="query.npz",
    )
    values = dict(
        past_states=np.zeros((2, 15)),
        past_inputs=np.zeros((1, 1)),
        future_inputs=np.ones((2, 1)),
        factual_inputs=np.zeros((2, 1)),
    )
    np.savez_compressed(data / "query.npz", **values)
    arm = "public_v4_float64"
    model = SimpleNamespace(
        fingerprint=lambda: "same-revision",
        envelope=lambda: np.ones((2, 15)),
        predict=lambda x, u, f: np.ones((len(f), 15), np.float64),
    )
    monkeypatch.setattr(LearnedDynamics, "load", lambda *a: model)
    monkeypatch.setattr(jax, "jit", lambda f: f)
    calibration = dict(
        prediction=np.ones((2, 15)),
        absolute_residual=np.ones((2, 15)),
        half_width=np.ones((2, 15)),
    )
    np.savez_compressed(previous / "public64/envelopes.npz", **{arm: model.envelope()})
    np.savez_compressed(previous / "public64/calibration.npz", **calibration)
    prediction_path = previous / "public64" / physical.query_path(query)
    prediction_path.parent.mkdir(parents=True)
    predictions = {arm: np.ones((2, 15)), arm + "_factual": np.ones((2, 15))}
    np.savez_compressed(prediction_path, **predictions)
    old_result = dict(
        model_sha256={arm: subject.digest(fit / "model.npz")},
        counters={arm: dict(branch_queries=1, factual_queries=1, inferred_prefixes=2)},
        calibration_reconstruction={"quantile_rank": 232},
    )
    write(previous / "public64/result.json", old_result)
    current = {"public_root": str(tmp_path / "source")}
    write(
        Path(current["public_root"]) / "docs/harness/public-mean-qualification-v1.json",
        {},
    )
    policy = dict(
        original_fit_binding_sha256="old-fit-binding",
        simulators={"crazyflow": {"evaluation_sha256": "old-evaluation"}},
    )
    pe = SimpleNamespace(
        _imports=lambda *a: {"current": "source"},
        sealed=lambda *a: {"implementation_manifest_sha256": "old-fit-binding"},
        read=subject.read,
        arrays=physical.arrays,
        equal=physical.equal,
        resolved=lambda *a: {},
        _planned=lambda *a: ([], [query], {}),
        ROLES=physical.ROLES,
        calibration_evidence=lambda model: (calibration, {"quantile_rank": 232}),
        under=physical.under,
        query_path=physical.query_path,
        predict_query=physical.predict_query,
        finite_command_prefix=physical.finite_command_prefix,
    )
    inputs = dict(
        data_root=str(data),
        data_sha256="old-data",
        flight_fit_root=str(fit),
        flight_fit_sha256="old-fit",
        reference_root="reference",
    )
    monkeypatch.setattr(subject, "authenticate", lambda args: (policy, current, {}, pe))
    monkeypatch.setattr(subject, "old_inputs", lambda *a: (previous, inputs))
    args = SimpleNamespace(
        phase="public64",
        simulator="crazyflow",
        output=output,
        binding_sha256="new-binding",
    )
    return args, pe, previous, prediction_path, predictions, calibration


@pytest.mark.parametrize(
    "mutation", [None, "missing_branch", "changed_calibration", "missing_query"]
)
def test_worker_requires_full_branch_and_calibration_identity(
    monkeypatch, tmp_path, mutation
):
    import jax

    args, pe, previous, prediction_path, predictions, calibration = worker_fixture(
        monkeypatch, tmp_path
    )
    if mutation == "missing_branch":
        predictions.pop("public_v4_float64_factual")
        np.savez_compressed(prediction_path, **predictions)
    elif mutation == "changed_calibration":
        changed = {k: v.copy() for k, v in calibration.items()}
        changed["half_width"][0, 0] += 0.001
        np.savez_compressed(previous / "public64/calibration.npz", **changed)
    elif mutation == "missing_query":
        pe._planned = lambda *a: ([], [], {})
    with jax.enable_x64(True):
        if mutation:
            with pytest.raises((AssertionError, physical.EvaluationError)):
                subject.worker(args)
        else:
            subject.worker(args)
            result = subject.read(args.output / "result.json")
            assert result["exact_prediction_arrays"] == 2
            assert result["exact_arrays"] == 6
            assert result["old_fit_binding_sha256"] == "old-fit-binding"
            assert result["new_inference_binding_sha256"] == "new-binding"
            assert (
                result["fits"] == result["initializers"] == result["simulations"] == 0
            )


def counts_fixture():
    report = dict(
        queries=2,
        exact_prediction_arrays=3,
        counters={"inferred_prefixes": 3},
        fits=0,
        initializers=0,
        simulations=0,
    )
    reports = {
        sim: {phase: copy.deepcopy(report) for phase in ("public32", "public64")}
        for sim in ("crazyflow", "cascade")
    }
    rows = {sim: [1, 2] for sim in reports}
    consumer = dict(exact_arrays=168, counts={"means": 24, "responses": 8})
    expected = dict(
        metric_rows=4,
        consumer_arrays=168,
        consumer_means=24,
        consumer_responses=8,
        query_slots_per_precision=4,
        prediction_arrays_per_precision=6,
        inferred_prefixes_per_precision=6,
        new_fits=0,
        new_initializers=0,
        new_simulations=0,
        original_fitting_operations=32,
    )
    return reports, rows, consumer, expected


@pytest.mark.parametrize(
    "mutation", [None, "query", "prefix", "fit", "rows", "consumer"]
)
def test_frozen_counts_reject_omission_or_new_work(mutation):
    reports, rows, consumer, expected = counts_fixture()
    if mutation == "query":
        reports["cascade"]["public32"]["queries"] -= 1
    elif mutation == "prefix":
        reports["cascade"]["public64"]["counters"]["inferred_prefixes"] -= 1
    elif mutation == "fit":
        reports["cascade"]["public64"]["fits"] = 1
    elif mutation == "rows":
        rows["cascade"].pop()
    elif mutation == "consumer":
        consumer["exact_arrays"] -= 1
    if mutation:
        with pytest.raises(AssertionError):
            subject.check_counts(reports, rows, consumer, expected)
    else:
        assert (
            subject.check_counts(reports, rows, consumer, expected)["metric_rows"] == 4
        )


def test_run_passes_complete_query_roster_and_original_worker(monkeypatch, tmp_path):
    reports, rows, consumer, expected = counts_fixture()
    old, current, policy = binding_pair()
    current["public_root"] = str(tmp_path / "current")
    policy.update(
        simulators={sim: {} for sim in reports},
        expected_counts=expected,
        old_binding={"sha256": "old-binding"},
        original_fit_binding_sha256="fit-binding",
    )
    write(
        Path(current["public_root"]) / "docs/harness/public-mean-qualification-v1.json",
        {},
    )
    decision = {"unchanged": "decision"}
    write(tmp_path / "decision.json", decision)
    policy["old_decision"] = {
        "path": str(tmp_path / "decision.json"),
        "sha256": subject.digest(tmp_path / "decision.json"),
    }
    roots = {}
    for sim in reports:
        previous = tmp_path / "original" / sim
        (previous / "historical").mkdir(parents=True)
        (previous / "historical/source.json").write_text("original-source")
        for name, value in dict(rows=rows[sim], directions=[], summary={}).items():
            write(previous / (name + ".json"), value)
        roots[sim] = previous
    calls = []

    def launch(args, phase, sim, output, execution, binding):
        calls.append((phase, sim, binding))
        if phase != "validate-old":
            write(output / "result.json", reports[sim][phase])

    planned = {sim: [{"id": sim + "-first"}, {"id": sim + "-last"}] for sim in reports}
    reduced = []

    def reduce_decision(actual, p, q, *, queries):
        assert actual == rows and queries == planned
        reduced.append(True)
        return decision

    pe = SimpleNamespace(
        resolved=lambda *a: {},
        score=lambda data, base, sim, p: dict(
            rows=rows[sim], directions=[], summary={}
        ),
        _planned=lambda data, sim, p: ([], planned[sim], {}),
        reduce_decision=reduce_decision,
    )
    monkeypatch.setattr(
        subject, "authenticate", lambda args: (policy, current, old, pe)
    )
    monkeypatch.setattr(
        subject,
        "old_inputs",
        lambda pe, p, sim: (
            roots[sim],
            dict(reference_root="reference", data_root="data"),
        ),
    )
    monkeypatch.setattr(subject, "launch", launch)
    monkeypatch.setattr(subject, "replay_consumer", lambda *a: consumer)
    args = SimpleNamespace(
        output=tmp_path / "result", binding_sha256="new-binding", policy_sha256="policy"
    )
    result = subject.run(args)
    assert len(calls) == 6 and len(reduced) == 1
    assert all(
        binding == old if phase == "validate-old" else binding == current
        for phase, sim, binding in calls
    )
    assert result["original_fitting_operations"] == 32 and result["fits"] == 0
    assert result["physical_decision_exact"] and not result["adoption_assessed"]


@pytest.mark.parametrize("mutation", [None, "runtime", "jvp"])
def test_consumer_reuses_old_packet_but_attests_current_runtime(
    monkeypatch, tmp_path, mutation
):
    from glassbox.experimental import (
        public_mean_consumer_qualification as consumer_helper,
    )
    from glassbox.experimental import public_mean_implementation as implementation

    packet, original, output = (
        tmp_path / name for name in ("packet", "original", "output")
    )
    packet.mkdir()
    original.mkdir()
    write(packet / "INPUT.json", {"qualification": "original-exporter"})
    values = {
        f"array_{index}": np.array([index], dtype=np.float32) for index in range(168)
    }
    np.savez_compressed(original / "arrays.npz", **values)
    report = dict(
        input_manifest_sha256=subject.digest(packet / "INPUT.json"),
        arrays_file="arrays.npz",
        arrays_sha256=subject.digest(original / "arrays.npz"),
        runtime={"identity": "old-runtime"},
        counts={"means": 24, "responses": 8},
        qualification={"commit": "old-fit"},
        means=[{"branch": "same"}],
    )
    write(original / "result.json", report)
    anchors = dict(
        manifest=str(packet / "INPUT.json"),
        manifest_sha256=report["input_manifest_sha256"],
        original_directory=str(original),
        result_sha256=subject.digest(original / "result.json"),
    )
    calls = []
    current = {"interpreter": "new-python", "public_root": "new-root"}

    def preflight(path, sha, destination):
        calls.append(("preflight", path, sha))
        return {"identity": "new-runtime"}

    def launch(command, binding, stem):
        calls.append(("launch", command, binding))
        assert binding == current
        assert command[command.index("--manifest") + 1] == str(packet / "INPUT.json")
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        arrays = {k: v.copy() for k, v in values.items()}
        if mutation == "jvp":
            arrays["array_0"][0] += 1
        np.savez_compressed(destination / "arrays.npz", **arrays)
        fresh = copy.deepcopy(report)
        fresh["arrays_sha256"] = subject.digest(destination / "arrays.npz")
        fresh["runtime"] = {
            "identity": "wrong-runtime" if mutation == "runtime" else "new-runtime"
        }
        write(destination / "result.json", fresh)

    monkeypatch.setattr(implementation, "consumer_preflight", preflight)
    monkeypatch.setattr(consumer_helper, "_launch", launch)
    args = SimpleNamespace(
        binding="new-binding.json", binding_sha256="new-binding-sha", output=output
    )
    if mutation:
        with pytest.raises((AssertionError, physical.EvaluationError)):
            subject.replay_consumer(args, {"consumer": anchors}, current, physical)
    else:
        result = subject.replay_consumer(args, {"consumer": anchors}, current, physical)
        assert (
            result["exact_arrays"] == 168
            and result["original_packet_qualification_preserved"]
        )
        assert calls[0] == ("preflight", "new-binding.json", "new-binding-sha")
