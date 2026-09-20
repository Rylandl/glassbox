"""Bounded analytic/mocked Dart seams; no physical trial or scientific fit."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental import shared_vehicle_dart as dart


def test_queries_retain_each_supported_horizon_without_future_history():
    rows = dart.diagnostic_queries(301)
    assert {h for h, _ in rows} == {5, 15, 25, 120}
    assert len(rows) == 36
    assert all(origin >= 50 and origin + h < 301 for h, origin in rows)
    assert [o for h, o in rows if h == 120] == [50, 75, 100, 125, 150, 175]


@pytest.fixture
def dart_planning(monkeypatch):
    monkeypatch.syspath_prepend("/Users/ryland/autonomy/dart/src")
    from crazydart import mission, planning

    return planning, mission


class AnalyticCore:
    """A differentiable proper-rotation mean; no initialization or fitting."""

    def __init__(self, angle=0.0, improper=False):
        self.angle, self.improper = angle, improper

    def rollout(self, past, issued, future):
        del past, issued
        angle = self.angle + 0.01 * jnp.cumsum(future[:, 0])
        c, s = jnp.cos(angle), jnp.sin(angle)
        z = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        rows = jnp.stack((c, z, s, z, one, z, -s, z, c), axis=1)
        if self.improper:
            rows = rows.at[:, 0].set(-1)
        v = jnp.stack((0.1 * jnp.cumsum(future[:, 0]), z, z), axis=1)
        w = jnp.stack((z, 0.01 * future[:, 0], z), axis=1)
        return jnp.concatenate((v, w, rows), axis=1)


def initial():
    state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    past = jnp.tile(jnp.concatenate((jnp.zeros(6), jnp.eye(3).reshape(9))), (51, 1))
    return state, past, jnp.zeros((50, 4))


@pytest.mark.parametrize("angle", [0.0, 0.7, 2.4])
@pytest.mark.parametrize("intrinsic", [False, True])
def test_full_original_mpc_reverse_gradient_accepts_history_pytree(
    dart_planning, angle, intrinsic
):
    planning, mission = dart_planning
    model = SimpleNamespace(_model=AnalyticCore(angle))
    rollout = dart.physical_rollout(model, intrinsic=intrinsic)
    planner = planning.DirectPlanner(
        rollout, mission.Target(), steps=120, block_steps=3
    )
    flat = jnp.full((160,), 0.6)
    value, gradient = planner.value_gradient(flat, initial(), jnp.asarray(120))
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1e-4
    predicted = np.asarray(
        rollout(initial(), jnp.repeat(flat.reshape(-1, 4), 3, axis=0))
    )
    assert predicted.shape == (121, 13)
    np.testing.assert_allclose(
        np.linalg.norm(predicted[:, 6:10], axis=1), 1.0, atol=2e-7
    )


def test_intrinsic_bridge_never_calls_polar_projection(monkeypatch):
    from glassbox.experimental import learned_plan

    def forbidden(*a, **k):
        raise AssertionError("unnecessary intrinsic projection")

    monkeypatch.setattr(learned_plan, "states_from_observed", forbidden)
    y = dart.physical_rollout(SimpleNamespace(_model=AnalyticCore()), intrinsic=True)(
        initial(), jnp.zeros((3, 4))
    )
    assert np.isfinite(y).all()


def test_improper_v4_rotation_is_visible_failure():
    result = dart.physical_rollout(SimpleNamespace(_model=AnalyticCore(improper=True)))(
        initial(), jnp.zeros((3, 4))
    )
    assert not np.isfinite(result).all()


def test_live_history_never_uses_hidden_plant_coordinates():
    states = np.zeros((55, 17))
    states[:, 2] = 1
    states[:, 6] = 1
    issued = np.arange(54 * 4, dtype=float).reshape(54, 4)
    a = dart.live_initial("shared_vehicle", None, states, issued)
    states[:, 13:] = 12345
    b = dart.live_initial("shared_vehicle", None, states, issued)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    assert a[1].shape == (51, 15) and a[2].shape == (50, 4)
    seen = []
    structured = SimpleNamespace(
        initial_latent_state=lambda u: seen.append(np.asarray(u)) or jnp.zeros(9)
    )
    c = dart.live_initial("structured_causal_history", structured, states, issued)
    np.testing.assert_array_equal(seen[0], issued)
    assert c[1].shape == (9,)


def tiny_preparation():
    from glassbox import learner
    from glassbox.experimental.shared_vehicle import STATE_CHANNELS
    from glassbox.recordings import SequenceCollection, SequenceSegment

    rows = []
    for i in range(3):
        t = np.arange(19) * 0.1
        x = np.zeros((19, 15))
        x[:, 0] = 0.1 * t + 0.01 * i
        x[:, 1] = 0.01 * np.sin(t + i)
        x[:, 6:] = np.eye(3).reshape(9)
        u = np.stack((np.sin(t[:-1] + i), np.cos(t[:-1] * 0.5 + i)), axis=1)
        rows.append(SequenceSegment(str(i), "whole", x, u, 0.1))
    collection = SequenceCollection(
        tuple(rows),
        configuration_id="analytic",
        state_channels=STATE_CHANNELS,
        input_channels=("a [1]", "b [1]"),
    )
    return (
        learner._extract(collection, ("0", "1"), 12),
        learner._extract(collection, ("2",), 6),
        learner._contract(collection),
        learner._recording_content(collection),
    )


def test_actual_tiny_v4_fit_capture_and_fingerprint_api(tmp_path, monkeypatch):
    from glassbox import learner
    from glassbox.experimental import shared_vehicle_experiment as exp

    prepared = tiny_preparation()
    monkeypatch.setattr(dart, "prepare", lambda p: prepared)
    original = learner.fit_sequence_model

    def zero(*args, **kwargs):
        kwargs["steps"] = 0
        return original(*args, **kwargs)

    monkeypatch.setattr(learner, "fit_sequence_model", zero)
    capture = exp.FitCapture
    monkeypatch.setattr(
        exp, "FitCapture", lambda path: capture(path, _expected_steps=0)
    )
    result = dart.fit_worker(tmp_path, {}, {"stage": "fit-v4"})
    assert result["status"] == "complete" and len(result["fingerprint"]) == 64
    assert result["work"]["initializer_returns"] == 1
    assert result["work"]["ridge_solve_returns"] == 2
    assert result["calibration_calls"] == 1
    assert result["work"]["checkpoints"] == 1
    loaded = learner.LearnedDynamics.load(tmp_path / "model.npz")
    dart.check_preparation(loaded, prepared)
    exp.verify_capture(tmp_path / "capture", loaded, replay=True, _expected_steps=0)
    from glassbox.experimental import shared_vehicle

    original_candidate = shared_vehicle.fit_sequence
    monkeypatch.setattr(
        shared_vehicle,
        "fit_sequence",
        lambda *a, **k: original_candidate(*a, **k, _steps=0),
    )
    monkeypatch.setattr(dart, "load_model", lambda *a, **k: loaded)
    monkeypatch.setattr(dart, "verify_stage", lambda *a, **k: {"status": "complete"})
    original_verify = exp.verify_capture
    monkeypatch.setattr(
        exp,
        "verify_capture",
        lambda *a, **k: original_verify(*a, **{**k, "_expected_steps": 0}),
    )
    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    result = dart.fit_worker(
        candidate_path,
        {},
        {
            "stage": "fit-shared",
            "binding_sha256": "unit",
            "inputs": {"v4": {"path": str(tmp_path), "sha256": "unit"}},
        },
    )
    assert result["status"] == "complete" and result["work"]["checkpoints"] == 1
    assert result["calibration_calls"] == 1
    for path in (tmp_path, candidate_path):
        phases = [
            json.loads(line)["phase"]
            for line in (path / "capture/work.jsonl").read_text().splitlines()
        ]
        assert phases.count("calibrated") == 1
    with (
        np.load(tmp_path / "capture/weights.npz") as old,
        np.load(candidate_path / "capture/weights.npz") as new,
    ):
        for key in ("normalization", "channel_weights"):
            np.testing.assert_array_equal(old[key], new[key])
    candidate = shared_vehicle.SharedVehicleDynamics.load(candidate_path / "model.npz")
    exp.verify_capture(
        candidate_path / "capture", candidate, replay=True, _expected_steps=0
    )


def fake_common(tmp_path):
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return SimpleNamespace(
        ROOT=tmp_path, digest=digest, read=lambda p: json.loads(Path(p).read_text())
    )


def test_supervisor_retains_timeout_prefix_without_retry(tmp_path, monkeypatch):
    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    protocol = {"dart": {"root": str(tmp_path)}}
    binding = {"interpreter": sys.executable, "implementation_commit": "abc"}
    monkeypatch.setattr(dart, "authenticate", lambda context: (protocol, binding))
    calls = []

    def timed_out(command, **kwargs):
        calls.append(command)
        output = Path(command[-1]).parent
        (output / "actual-prefix.txt").write_text("started")
        raise subprocess.TimeoutExpired(command, 14400)

    monkeypatch.setattr(dart.subprocess, "run", timed_out)
    result = dart.run_stage(
        "fit-v4",
        tmp_path / "stage",
        protocol_path=tmp_path / "p",
        protocol_sha256="p",
        binding_path=tmp_path / "b",
        binding_sha256="b",
    )
    assert result["status"] == "hard_timeout_incomplete" and len(calls) == 1
    assert "actual-prefix.txt" in result["files"]
    assert c.read(tmp_path / "stage/exit.json")["timed_out"]
    with pytest.raises(FileExistsError):
        dart.run_stage(
            "fit-v4",
            tmp_path / "stage",
            protocol_path=tmp_path / "p",
            protocol_sha256="p",
            binding_path=tmp_path / "b",
            binding_sha256="b",
        )


def test_final_integrity_failure_cannot_qualify_completed_work(tmp_path, monkeypatch):
    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    count = 0

    def auth(context):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("source changed")
        return {"dart": {"root": str(tmp_path)}}, {
            "interpreter": sys.executable,
            "implementation_commit": "abc",
        }

    monkeypatch.setattr(dart, "authenticate", auth)

    def done(command, **kwargs):
        dart.write(
            Path(command[-1]).parent / "outcome.json", {"status": "complete", "fits": 1}
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dart.subprocess, "run", done)
    result = dart.run_stage(
        "fit-v4",
        tmp_path / "stage",
        protocol_path=tmp_path / "p",
        protocol_sha256="p",
        binding_path=tmp_path / "b",
        binding_sha256="b",
    )
    assert result["status"] == "failed"
    assert c.read(tmp_path / "stage/outcome.json")["fits"] == 1
    assert "final-integrity-error.json" in result["files"]


def test_external_seal_and_payload_roster_required(tmp_path, monkeypatch):
    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "x").write_text("a")
    dart.write(directory / "request.json", {"stage": "fit-v4"})
    dart.write(
        directory / "run.json",
        {
            "format": dart.FORMAT,
            "files": dart.inventory(directory),
            "stage": "fit-v4",
            "binding_sha256": "b",
            "request_sha256": c.digest(directory / "request.json"),
        },
    )
    digest = c.digest(directory / "run.json")
    assert dart.verify_stage(directory, digest, stage="fit-v4", binding_sha256="b")
    (directory / "x").write_text("b")
    with pytest.raises(ValueError, match="inventory"):
        dart.verify_stage(directory, digest)
    with pytest.raises(ValueError, match="external"):
        dart.verify_stage(directory, "0" * 64)


def test_actual_vehicle_core_composes_with_original_120_step_mpc(dart_planning):
    from dataclasses import replace

    from test_shared_vehicle_core import constant_model

    planning, mission = dart_planning
    core = constant_model(commands=4, dt=0.01, history=50, delay=10)
    params = {key: value.copy() for key, value in core.params.items()}
    params["bias"][2] = 9.80665
    params["linear"][9, 0] = 0.5
    params["linear"][10, 4] = 0.2
    core = replace(core, params=params)
    planner = planning.DirectPlanner(
        dart.physical_rollout(SimpleNamespace(_model=core), intrinsic=True),
        mission.Target(),
        steps=120,
        block_steps=3,
    )
    value, gradient = planner.value_gradient(
        jnp.full(160, 0.6), initial(), jnp.asarray(120)
    )
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1e-4


def test_ordinary_roles_and_public_window_selection_without_prefix_padding(
    tmp_path, monkeypatch
):
    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    protocol = {"dart": {"files": {}}}
    channels = [
        {
            "name": f"u{i}",
            "kind": "control",
            "minimum": 0.1,
            "maximum": 1.0,
            "unit": "1",
            "semantic": "normalized_command",
            "role": "DO_NOT_USE_AS_PHYSICS",
        }
        for i in range(4)
    ]
    for index, name in enumerate((*dart.TRAIN, *dart.DEVELOPMENT)):
        t = np.arange(301) * 0.01
        state = np.zeros((301, 13))
        state[:, 2] = 1
        state[:, 6] = 1
        state[:, 3] = 0.1 * t + index * 0.01
        commands = np.stack(
            [0.5 + 0.1 * np.sin(t[:-1] * (j + 1) + index) for j in range(4)], axis=1
        )
        path = tmp_path / (name + ".npz")
        np.savez(
            path,
            time_s=t,
            states=state,
            controls=commands,
            control_prefix=np.full((50, 4), 999.0),
            spec_json=json.dumps(
                {
                    "state_schema": "rigid_body_13_nwu_flu_wxyz_v1",
                    "channels": channels,
                    "vehicle": {"family": "invalid_do_not_use"},
                }
            ),
            labels_json=json.dumps({"split": name.split("-")[0]}),
        )
        protocol["dart"]["files"][f"artifacts/demo/telemetry/{name}.npz"] = {
            "path": str(path),
            "sha256": c.digest(path),
        }
    train, dev, contract, seen = dart.prepare(protocol)
    assert len(train.keys) == 1536 and len(dev.keys) == 256
    assert {k.recording_id for k in train.keys} == set(dart.TRAIN)
    assert {k.recording_id for k in dev.keys} == set(dart.DEVELOPMENT)
    assert min(train.source_origins) >= 50 and min(dev.source_origins) >= 50
    assert max(train.batch.past_inputs.flat) < 1
    assert "DO_NOT_USE" not in json.dumps(
        contract
    ) and "invalid_do_not_use" not in json.dumps(contract)
    assert set(seen) == set((*dart.TRAIN, *dart.DEVELOPMENT))


def test_pair_rejects_different_valid_weighting_control(tmp_path, monkeypatch):
    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    current = tmp_path / "candidate"
    current.mkdir()
    dart.write(
        current / "request.json",
        {"inputs": {"v4": {"path": "old-v4", "sha256": "old"}}},
    )
    monkeypatch.setattr(dart, "verify_stage", lambda *a, **k: {"status": "complete"})
    with pytest.raises(ValueError, match="different Dart v4"):
        dart.check_pair(
            {
                "candidate": {"path": str(current), "sha256": "candidate"},
                "v4": {"path": "new-v4", "sha256": "new"},
            },
            {"binding_sha256": "bound"},
        )


def test_pair_rejects_coherently_saved_different_objective_weights(
    tmp_path, monkeypatch
):
    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    paths = {key: tmp_path / key for key in ("v4", "candidate")}
    for path in paths.values():
        (path / "capture").mkdir(parents=True)
    anchors = {key: {"path": str(path), "sha256": key} for key, path in paths.items()}
    dart.write(paths["candidate"] / "request.json", {"inputs": {"v4": anchors["v4"]}})
    for key, path in paths.items():
        np.savez(
            path / "capture/weights.npz",
            normalization=np.ones((25, 15)),
            channel_weights=np.full(15, 1.0 if key == "v4" else 2.0),
        )
    monkeypatch.setattr(dart, "verify_stage", lambda *a, **k: {"status": "complete"})
    with pytest.raises(ValueError, match="shared objective weights"):
        dart.check_pair(anchors, {"binding_sha256": "bound"})


def test_imported_crazyflow_package_root_is_bound_before_plant_construction(
    tmp_path, monkeypatch
):
    from types import ModuleType

    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    modules = {}
    for name in ("__init__", "plant", "planning", "mission"):
        path = tmp_path / f"src/crazydart/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n")
        key = "crazydart" if name == "__init__" else "crazydart." + name
        module = ModuleType(key)
        module.__file__ = str(path)
        modules[key] = module
        monkeypatch.setitem(sys.modules, key, module)
    for name in ("plant", "planning", "mission"):
        setattr(modules["crazydart"], name, modules["crazydart." + name])
    cf = ModuleType("crazyflow")
    cf.__file__ = str(tmp_path / "expected-crazyflow/__init__.py")
    monkeypatch.setitem(sys.modules, "crazyflow", cf)
    binding = {
        "dart_source_sha256": {
            name: c.digest(tmp_path / name) for name in dart.DART_SOURCES
        },
        "simulator_sources": {
            "crazyflow": {"package_root": str(tmp_path / "expected-crazyflow")}
        },
    }
    protocol = {"dart": {"root": str(tmp_path)}}
    assert dart.dart_modules(protocol, binding)[2] is modules["crazydart.plant"]
    cf.__file__ = str(tmp_path / "different-crazyflow/__init__.py")
    with pytest.raises(ValueError, match="Crazyflow package root"):
        dart.dart_modules(protocol, binding)


def failed_weighting_control(tmp_path, monkeypatch, *, with_weights):
    from glassbox.experimental import shared_vehicle_experiment as exp

    prepared = tiny_preparation()
    source = tmp_path / "source"
    source.mkdir()
    dart.save_preparation(source, prepared)
    capture = exp.FitCapture(source / "capture")
    if with_weights:
        capture(
            {
                "phase": "weights",
                "normalization": np.ones(prepared[0].batch.future_states.shape[1:]),
                "channel_weights": np.ones(15),
            }
        )
    capture.finish(None)
    monkeypatch.setattr(dart, "verify_stage", lambda *a, **k: {"status": "fit_failed"})
    monkeypatch.setattr(dart, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(dart, "prepare", lambda p: prepared)
    request = {
        "stage": "fit-shared",
        "binding_sha256": "bound",
        "inputs": {"v4": {"path": str(source), "sha256": "source"}},
    }
    return source, prepared, request


def test_failed_v4_actual_weights_can_reach_one_candidate_attempt(
    tmp_path, monkeypatch
):
    from glassbox.experimental import shared_vehicle

    source, prepared, request = failed_weighting_control(
        tmp_path, monkeypatch, with_weights=True
    )
    calls = []

    def attempted(*args, **kwargs):
        calls.append((args, kwargs))
        raise ArithmeticError("synthetic candidate failure after entry")

    monkeypatch.setattr(shared_vehicle, "_train", attempted)
    output = tmp_path / "candidate"
    output.mkdir()
    with pytest.raises(ArithmeticError, match="after entry"):
        dart.fit_worker(output, {}, request)
    assert len(calls) == 1 and calls[0][0] == prepared
    expected = dart._arrays(source / "capture/weights.npz")
    np.testing.assert_array_equal(calls[0][1]["error_scale"], expected["normalization"])
    np.testing.assert_array_equal(
        calls[0][1]["channel_weights"], expected["channel_weights"]
    )
    assert (output / "capture/work-summary.json").exists()


def test_failed_v4_before_weights_makes_zero_attempt_unavailable(tmp_path, monkeypatch):
    from glassbox.experimental import shared_vehicle

    _, _, request = failed_weighting_control(tmp_path, monkeypatch, with_weights=False)

    def forbidden(*a, **k):
        raise AssertionError("candidate must not initialize or fit")

    monkeypatch.setattr(shared_vehicle, "_train", forbidden)
    output = tmp_path / "candidate"
    output.mkdir()
    result = dart.fit_worker(output, {}, request)
    assert (
        result["status"] == "unavailable"
        and result["fits"] == result["initializers"] == 0
    )
    assert (output / "preparation.npz").exists() and not (output / "capture").exists()


def test_weighting_control_rejects_timeout_or_changed_preparation(
    tmp_path, monkeypatch
):
    source, prepared, request = failed_weighting_control(
        tmp_path, monkeypatch, with_weights=True
    )
    monkeypatch.setattr(
        dart, "verify_stage", lambda *a, **k: {"status": "hard_timeout_incomplete"}
    )
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        dart.weighting_control(request["inputs"]["v4"], prepared, request)
    monkeypatch.setattr(dart, "verify_stage", lambda *a, **k: {"status": "fit_failed"})
    arrays = dart._arrays(source / "preparation.npz")
    key = next(
        k
        for k in arrays
        if k != "metadata" and np.issubdtype(arrays[k].dtype, np.floating)
    )
    arrays[key].flat[0] += 1.0
    np.savez(source / "preparation.npz", **arrays)
    with pytest.raises(ValueError, match="weighting control preparation"):
        dart.weighting_control(request["inputs"]["v4"], prepared, request)


def test_worker_request_sha_checked_before_any_worker_code(tmp_path, monkeypatch):
    c = fake_common(tmp_path)
    c.PROTOCOL_PATH, c.PROTOCOL_SHA256 = tmp_path / "p", "p"
    monkeypatch.setattr(dart, "common", lambda: c)
    path = tmp_path / "request.json"
    dart.write(path, {"stage": "fit-v4"})
    monkeypatch.setattr(
        dart, "worker", lambda request: pytest.fail("worker must not start")
    )
    with pytest.raises(ValueError, match="request SHA"):
        dart.main(["--worker", str(path), "--request-sha256", "0" * 64])


def test_supervisor_preserves_dispatch_hash_when_request_changes(tmp_path, monkeypatch):
    c = fake_common(tmp_path)
    monkeypatch.setattr(dart, "common", lambda: c)
    monkeypatch.setattr(
        dart,
        "authenticate",
        lambda context: (
            {"dart": {"root": str(tmp_path)}},
            {"interpreter": sys.executable, "implementation_commit": "abc"},
        ),
    )

    def changed(command, **kwargs):
        path = Path(command[-1])
        expected = command[command.index("--request-sha256") + 1]
        assert c.digest(path) == expected
        path.write_text(path.read_text() + " ")
        dart.write(path.parent / "outcome.json", {"status": "complete", "fits": 1})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dart.subprocess, "run", changed)
    result = dart.run_stage(
        "fit-v4",
        tmp_path / "stage",
        protocol_path=tmp_path / "p",
        protocol_sha256="p",
        binding_path=tmp_path / "b",
        binding_sha256="b",
    )
    command = c.read(tmp_path / "stage/command.json")["command"]
    assert result["status"] == "failed"
    assert result["request_sha256"] == command[command.index("--request-sha256") + 1]
    assert result["request_sha256"] != c.digest(tmp_path / "stage/request.json")
    assert c.read(tmp_path / "stage/outcome.json")["fits"] == 1
