"""No-fit saved-port contracts; execute only after implementation is committed."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox import learner
from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox.experimental import public_mean_saved_port as port
from glassbox.recordings import SequenceCollection, SequenceSegment


def _forbid(*args, **kwargs):
    raise AssertionError("a saved port must not fit, initialize, calibrate or forecast")


@pytest.fixture
def no_fitted_calls(monkeypatch):
    from glassbox import _sequence_model as core

    for module, names in (
        (learner, ("fit", "_train", "fit_sequence_model", "_calibrate", "_measure")),
        (
            core,
            (
                "fit_sequence_model",
                "initialize_sequence_model",
                "_initialize_affine",
                "_rollout",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, _forbid)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def research(tmp_path, monkeypatch, no_fitted_calls):
    """Real cache geometry and independently fabricated arrays, no fitted model."""
    segments = []
    t = np.arange(30, dtype=np.float64)
    for i in range(96):
        x = np.stack(
            (np.sin(t / 7 + i / 13), np.cos(t / 9 + i / 11), t / 17 + i / 19), axis=-1
        )
        u = np.stack((np.sin(t[:-1] / 5 + i), np.cos(t[:-1] / 6 + i)), axis=-1)
        segments.append(SequenceSegment(f"record-{i:03d}", "valid-prefix", x, u, 0.25))
    collection = SequenceCollection(
        tuple(segments),
        configuration_id="saved-port-toy",
        state_channels=("x:1", "y:1", "z:1"),
        input_channels=("a:1", "b:1"),
    )
    contract, seen = (
        learner._contract(collection),
        learner._recording_content(collection),
    )
    names = sorted(seen, key=learner._priority)
    roles = {"development": names[:24], "training": names[24:]}
    train = learner._extract(collection, roles["training"], 1536)
    development = learner._extract(collection, roles["development"], 256)
    prefix = learner._extract(collection, roles["training"], 384)
    norms = port.training_norms(train.batch)
    f, d, m = len(norms["feature_scale"]), 3, 2
    initial = {
        "param_linear": np.zeros((f, d)),
        "param_bias": np.full(d, 0.02),
        "param_w1": np.arange(f * 32, dtype=np.float64).reshape(f, 32) / 10000,
        "param_b1": np.zeros(32),
        "param_w2": np.zeros((32, d)),
        "param_memory": np.zeros((f, 8)),
        "param_memory_bias": np.zeros(8),
        "param_interaction": np.zeros((d * m, d)),
        "param_autonomous": np.zeros((6, d)),
        **{"norm_" + k: v for k, v in norms.items()},
    }
    selected = {k: v.copy() for k, v in initial.items()}
    selected["param_bias"] *= 0.8
    hold = np.repeat(train.batch.past_states[:, -1:], 1, axis=1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - train.batch.future_states) ** 2, axis=0)),
        0.01 * norms["state_scale"],
    )
    prediction = train.batch.future_states + 0.01
    e0 = np.mean(((prediction - train.batch.future_states) / scale) ** 2, axis=(0, 1))
    raw = 1 / np.maximum(e0, 0.0001)
    w = raw / raw.mean()
    objective = {
        **initial,
        "initial_training_prediction": prediction,
        "initial_channel_mse": e0,
        "raw_channel_weights": raw,
        "channel_weights": w,
        "fixed_weights": w,
        "normalization": scale,
        "weight_floor": np.asarray(0.0001),
        "weight_normalizer": np.asarray(raw.mean()),
    }

    def wm(windows, *, preparation=False):
        return {
            **port._window_metadata(windows),
            "excitation_declared": False,
            **({"dt_s": 0.25} if preparation else {}),
        }

    def wa(windows, role):
        return {f"{role}_{k}": v for k, v in port.window_arrays(windows).items()}

    root = tmp_path / "research"
    base = root / "crazyflow"
    checkpoint = base / "candidate/checkpoints"
    trajectory = checkpoint / "weighted/anchored/trajectory"
    trajectory.mkdir(parents=True)
    (base / "baseline").mkdir()
    baseline_meta = {
        "seen": seen,
        "contract": contract,
        "windows": {"train": wm(prefix), "development": wm(development)},
    }
    baseline_arrays = {**wa(prefix, "train"), **wa(development, "development")}
    save_arrays(base / "baseline/model.npz", baseline_meta, baseline_arrays)
    data_seal = {
        key: "a" * 64
        for key in (
            "common_data_seal_sha256",
            "excitation_seal_sha256",
            "excitation_reuse_sha256",
            "reference_reuse_sha256",
        )
    }
    provenance = {
        "public_reference_fingerprint": array_fingerprint(
            baseline_meta, baseline_arrays
        ),
        "data_seal": data_seal,
        "recording_content_fingerprints": seen,
        "roles": roles,
        "ridge": 15.36,
        "delay": 1,
        "training_windows_fingerprint": port.window_fingerprint(train),
        "development_windows_fingerprint": port.window_fingerprint(development),
        "reference_training_windows_fingerprint": port.window_fingerprint(prefix),
        "training_excitation_fingerprint": port.excitation_fingerprint(train),
        "development_excitation_fingerprint": port.excitation_fingerprint(development),
        "training_norms_fingerprint": array_fingerprint({}, norms),
        "error_scale_fingerprint": array_fingerprint({}, {"error_scale": scale}),
    }
    envelope = np.full((1, 3), 0.15)
    report = {
        "recipe": {"id": "research"},
        "preparation": provenance,
        "optimization": {"selected_step": 1000},
        "development_errors": {"source": "unchanged"},
        "evidence_limits": ["historical"],
        "envelope": {
            "half_width": envelope.tolist(),
            "calibrated_on": "development",
            "calibration_windows": 256,
            "quantile_rank": 232,
        },
    }
    meta = {
        "report": report,
        "contract": contract,
        "model": {"dt_s": 0.25, "history_steps": 2, "delay_steps": 1},
        "windows": {"train": wm(train), "development": wm(development)},
    }
    save_arrays(
        base / "candidate/model.npz",
        meta,
        {
            **selected,
            **wa(train, "train"),
            **wa(development, "development"),
            "envelope_half_width": envelope,
        },
    )
    preparation = {
        "contract": contract,
        "ridge": 15.36,
        "delay": 1,
        "provenance": provenance,
        "windows": {
            "train": wm(train, preparation=True),
            "development": wm(development, preparation=True),
        },
    }
    save_arrays(
        base / "candidate/preparation.npz",
        preparation,
        {
            "error_scale": scale,
            **{"norm_" + k: v for k, v in norms.items()},
            **wa(train, "train"),
            **wa(development, "development"),
        },
    )
    _write_json(base / "candidate/report.json", report)
    save_arrays(checkpoint / "initializer-return.npz", {}, initial)
    save_arrays(checkpoint / "weighted/objective-witness.npz", {}, objective)
    save_arrays(trajectory / "step-0000.npz", {}, {**initial, "error_scale": scale})
    save_arrays(trajectory / "step-1000.npz", {}, {**selected, "error_scale": scale})
    _write_json(base / "excitation/roles.json", roles)
    _write_json(base / "excitation/seal.json", {})
    manifest = {
        "files": {
            str(p.relative_to(root)): port.digest(p)
            for p in root.rglob("*")
            if p.is_file()
        }
    }
    manifest["files"].update(
        {
            "crazyflow/" + p: "a" * 64
            for p in (
                "data/seal.json",
                "excitation/seal.json",
                "reuse.json",
                "reference-reuse.json",
            )
        }
    )
    monkeypatch.setattr(port, "_collection", lambda *args: (collection, roles))
    return root, manifest


def test_complete_preparation_and_public_fixture_need_no_fitted_calls(
    research, tmp_path
):
    root, manifest = research
    prepared = port._prepare(root, "crazyflow", manifest)
    assert len(prepared.train.keys) == 1536
    assert len(prepared.development.keys) == 256
    assert len(prepared.seen) == 96
    assert not prepared.train.excitation_declared
    assert not prepared.initial_arrays["param_linear"].flags.writeable
    assert not prepared.error_scale.flags.writeable
    fixture = port.public_fixture(prepared)
    path = tmp_path / "public.npz"
    fixture.save(path)
    loaded = learner.LearnedDynamics.load(path)
    assert loaded.fingerprint() == fixture.fingerprint()
    port.equal_windows(loaded._train, prepared.train, "train")
    assert loaded._seen == prepared.seen
    np.testing.assert_array_equal(loaded.envelope(), prepared.envelope)
    assert loaded.report["envelope"] == prepared.source_report["envelope"]
    assert loaded.report["saved_research_port"]["fitting_operations"] == 0
    assert (
        "optimization" not in loaded.report
    )  # old optimizer is not relabeled as a new run


@pytest.mark.parametrize(
    "attack",
    [
        "train",
        "development",
        "origin",
        "norm",
        "hold",
        "initial",
        "selected",
        "weight",
        "ledger",
        "role",
        "sidecar",
        "envelope",
    ],
)
def test_coherently_resealed_scientific_substitutions_reject(research, attack):
    root, manifest = research
    base = root / "crazyflow"
    relative = "candidate/model.npz"
    if attack in {"norm", "hold"}:
        relative = "candidate/preparation.npz"
    elif attack == "initial":
        relative = "candidate/checkpoints/initializer-return.npz"
    elif attack == "selected":
        relative = "candidate/checkpoints/weighted/anchored/trajectory/step-1000.npz"
    elif attack == "weight":
        relative = "candidate/checkpoints/weighted/objective-witness.npz"
    elif attack == "ledger":
        relative = "baseline/model.npz"
    path = base / relative
    meta, arrays = load_arrays(path)
    if attack in {"train", "development"}:
        arrays[attack + "_future_states"][0, 0, 0] += 1
    elif attack == "origin":
        meta["windows"]["train"]["source_origins"][0] += 1
    elif attack == "norm":
        arrays["norm_state_scale"][0] *= 2
    elif attack == "hold":
        arrays["error_scale"][0, 0] *= 2
    elif attack in {"initial", "selected"}:
        arrays["param_bias"][0] += 1
    elif attack == "weight":
        arrays["fixed_weights"][0] *= 2
    elif attack == "ledger":
        meta["seen"][next(iter(meta["seen"]))] = "b" * 64
    elif attack == "role":
        meta["report"]["preparation"]["roles"]["training"].reverse()
        _write_json(base / "candidate/report.json", meta["report"])
    elif attack == "sidecar":
        meta["windows"]["train"]["excitation_declared"] = True
        arrays["train_past_excitation"] = np.zeros_like(arrays["train_past_inputs"])
        arrays["train_future_excitation"] = np.zeros_like(arrays["train_future_inputs"])
    elif attack == "envelope":
        arrays["envelope_half_width"][0, 0] *= 2
    save_arrays(path, meta, arrays)
    with pytest.raises(ValueError):
        port._prepare(root, "crazyflow", manifest)


def test_window_equal_checks_declared_zero_sidecars(research):
    root, manifest = research
    prepared = port._prepare(root, "crazyflow", manifest)
    declared = replace(
        prepared.train,
        past_excitation=np.zeros_like(prepared.train.batch.past_inputs),
        future_excitation=np.zeros_like(prepared.train.batch.future_inputs),
    )
    with pytest.raises(ValueError, match="excitation"):
        port.equal_windows(declared, prepared.train, "role")


def test_inventory_external_anchor_and_payload_roster(tmp_path):
    (tmp_path / "payload").write_text("trusted")
    _write_json(
        tmp_path / "manifest.json",
        {"files": {"payload": port.digest(tmp_path / "payload")}},
    )
    anchor = port.digest(tmp_path / "manifest.json")
    port._inventory(tmp_path, "manifest.json", anchor)
    with pytest.raises(ValueError, match="external"):
        port._inventory(tmp_path, "manifest.json", "0" * 64)
    (tmp_path / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="roster"):
        port._inventory(tmp_path, "manifest.json", anchor)
    (tmp_path / "extra").unlink()
    (tmp_path / "payload").write_text("altered")
    with pytest.raises(ValueError, match="changed"):
        port._inventory(tmp_path, "manifest.json", anchor)


def test_only_named_postseal_reports_can_be_ignored(tmp_path):
    (tmp_path / "payload").write_text("science")
    _write_json(
        tmp_path / "manifest.json",
        {"files": {"payload": port.digest(tmp_path / "payload")}},
    )
    anchor = port.digest(tmp_path / "manifest.json")
    _write_json(tmp_path / "crazyflow/replay.json", {"untrusted": "not used"})
    port._inventory(
        tmp_path,
        "manifest.json",
        anchor,
        ignored_auxiliary=port.UNTRUSTED_REFERENCE_REPORTS,
    )
    with pytest.raises(ValueError, match="roster"):
        port._inventory(tmp_path, "manifest.json", anchor)
    _write_json(tmp_path / "crazyflow/extra.json", {})
    with pytest.raises(ValueError, match="roster"):
        port._inventory(
            tmp_path,
            "manifest.json",
            anchor,
            ignored_auxiliary=port.UNTRUSTED_REFERENCE_REPORTS,
        )


def test_failure_preserves_exclusive_intent_and_sealed_partial_outcome(
    tmp_path, monkeypatch
):
    from glassbox.experimental import public_mean_implementation as implementation

    monkeypatch.setattr(
        implementation,
        "verify",
        lambda *args: {"public_root": str(Path(port.__file__).resolve().parents[3])},
    )
    monkeypatch.setattr(port, "authenticate_reference", lambda *a, **kw: {})

    def fail(*args):
        raise ValueError("preparation mismatch")

    monkeypatch.setattr(port, "_prepare", fail)
    out = tmp_path / "failed"
    kwargs = dict(
        reference_root="unused",
        implementation_manifest="bound.json",
        implementation_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="preparation mismatch"):
        port.run(out, **kwargs)
    assert not (out / "manifest.json").exists()
    failure = port._inventory(out, "failure.json", port.digest(out / "failure.json"))
    assert failure["outcome"]["completed_simulators"] == []
    assert failure["outcome"]["active_simulator"] == "crazyflow"
    assert failure["outcome"]["status"] == "failed"
    assert port._json(out / "attempt.json")["fitting_operations"] == 0
    with pytest.raises(FileExistsError):
        port.run(out, **kwargs)


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a//b", "./payload", "a\\b"]
)
def test_inventory_refuses_unsafe_paths(tmp_path, path):
    with pytest.raises(ValueError, match="relative"):
        port._under(tmp_path, path)


def test_public_fixture_keeps_historical_report_immutable(research):
    root, manifest = research
    prepared = port._prepare(root, "crazyflow", manifest)
    before = copy.deepcopy(prepared.source_report)
    model = port.public_fixture(prepared)
    report = model.report
    report["envelope"]["half_width"][0][0] = 100
    assert prepared.source_report == before
    assert model.report["envelope"] == before["envelope"]


def test_arithmetic_training_norms_do_not_draw_or_solve(research, monkeypatch):
    root, manifest = research
    monkeypatch.setattr(np.linalg, "solve", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    prepared = port._prepare(root, "crazyflow", manifest)
    assert set(prepared.norms) == {
        "state_mean",
        "state_scale",
        "input_mean",
        "input_scale",
        "feature_scale",
        "delta_scale",
        "interaction_scale",
        "autonomous_scale",
    }


def test_run_and_replay_bind_saved_model_report_and_recordings(
    research, monkeypatch, tmp_path
):
    from glassbox.experimental import public_mean_implementation as implementation

    root, manifest = research
    prepared = port._prepare(root, "crazyflow", manifest)
    monkeypatch.setattr(port, "SIMULATORS", ("crazyflow",))
    monkeypatch.setattr(port, "authenticate_reference", lambda *a, **kw: manifest)
    monkeypatch.setattr(port, "_prepare", lambda *a: prepared)
    monkeypatch.setattr(
        implementation,
        "verify",
        lambda *a: {"public_root": str(Path(port.__file__).resolve().parents[3])},
    )
    out = tmp_path / "port"
    kwargs = dict(
        reference_root=root,
        implementation_manifest="bound.json",
        implementation_sha256="c" * 64,
    )
    result = port.run(out, **kwargs)
    assert port.verify(
        out, expected_manifest_sha256=result["manifest_sha256"], **kwargs
    )["passed"]
    # Coherently reseal the outer manifest; fresh reconstruction still authenticates source report.
    (out / "crazyflow/source-report.json").write_text("{}\n")
    saved = port._json(out / "manifest.json")
    saved["files"]["crazyflow/source-report.json"] = port.digest(
        out / "crazyflow/source-report.json"
    )
    _write_json(out / "manifest.json", saved)
    with pytest.raises(ValueError, match="research report"):
        port.replay(
            out, expected_manifest_sha256=port.digest(out / "manifest.json"), **kwargs
        )
