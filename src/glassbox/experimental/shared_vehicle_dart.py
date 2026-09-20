"""Frozen ordinary-recording fits and one matched nominal Dart task per arm.

This research driver never supplies simulator actuator state to a predictor.
Each numerical stage runs once in an exclusive, bounded child. Saved replay
executes recorded commands and selected means, never a fitter or optimizer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

MODULE = "glassbox.experimental.shared_vehicle_dart"
FORMAT = "glassbox-shared-vehicle-dart-stage-v1"
ARMS = ("shared_vehicle", "v4_research_extrapolation", "structured_causal_history")
STAGES = ("fit-v4", "fit-shared", "forecast", "control", "replay")
CORRECTION_MANIFEST = "docs/harness/shared-vehicle-dart-runtime-correction-v1.json"
CORRECTION_FILES = {
    "src/glassbox/experimental/shared_vehicle_dart.py",
    "tests/test_shared_vehicle_dart.py",
    CORRECTION_MANIFEST,
}
TRAIN = tuple(f"train-{i:02d}" for i in range(8))
DEVELOPMENT = ("calibration-08", "calibration-09")
TEST = ("test-10", "test-11")
DART_SOURCES = tuple(
    f"src/crazydart/{name}.py" for name in ("__init__", "plant", "planning", "mission")
)
HISTORY, HORIZON, REPLAN = 50, 120, 3
DT = 0.01


def common():
    from . import shared_vehicle_data

    return shared_vehicle_data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def inventory(directory):
    root = Path(directory)
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "stage payload symlink")
        if path.is_file() and path != root / "run.json":
            result[str(path.relative_to(root))] = common().digest(path)
    return result


def verify_stage(directory, expected_sha256, *, stage=None, binding_sha256=None):
    root = Path(directory).resolve()
    require(
        common().digest(root / "run.json") == expected_sha256,
        "external Dart stage SHA differs",
    )
    run = common().read(root / "run.json")
    require(run.get("format") == FORMAT, "Dart stage format differs")
    require(run.get("files") == inventory(root), "Dart stage payload inventory differs")
    require(
        run.get("request_sha256") == common().digest(root / "request.json"),
        "Dart stage request differs from dispatch anchor",
    )
    if stage is not None:
        require(run.get("stage") == stage, "Dart stage identity differs")
    if binding_sha256 is not None:
        require(
            run.get("binding_sha256") == binding_sha256,
            "Dart stage source binding differs",
        )
    return run


def authenticate(context):
    p, binding = common().authenticate(
        context["protocol_path"],
        context["protocol_sha256"],
        context["binding_path"],
        context["binding_sha256"],
    )
    root = Path(p["dart"]["root"]).resolve()
    for name, row in p["dart"]["files"].items():
        require(
            Path(row["path"]).resolve() == root / name, "Dart artifact path differs"
        )
        require(
            common().digest(row["path"]) == row["sha256"],
            "frozen Dart artifact differs: " + name,
        )
    source_map = binding.get("dart_source_sha256", {})
    require(set(DART_SOURCES).issubset(source_map), "Dart source binding is incomplete")
    for name in DART_SOURCES:
        require(
            common().digest(root / name) == source_map[name],
            "Dart source differs: " + name,
        )
    return p, binding


def inherited_fit_manifest(context):
    """Authenticate the single pinned fit pair without broadening stage trust."""
    _, current = authenticate(context)
    manifest_path = common().ROOT / CORRECTION_MANIFEST
    require(
        current["current"]["files"].get(CORRECTION_MANIFEST)
        == common().digest(manifest_path),
        "correction manifest is not source-bound",
    )
    manifest = common().read(manifest_path)
    require(
        manifest["format"] == "glassbox-shared-vehicle-dart-runtime-correction-v1"
        and manifest["protocol_sha256"] == context["protocol_sha256"]
        and set(manifest["fits"]) == {"fit-v4", "fit-shared"}
        and set(manifest["allowed_changed_files"]) == CORRECTION_FILES,
        "correction manifest identity differs",
    )
    predecessor = common().read(common().anchor(manifest["predecessor_binding"]))
    require(
        predecessor["format"] == current["format"]
        and predecessor["implementation_commit"] == manifest["predecessor_commit"]
        and predecessor["current"]["commit"] == manifest["predecessor_commit"],
        "predecessor source identity differs",
    )
    for key in (
        "protocol_sha256",
        "runtime",
        "simulator_sources",
        "dart_source_sha256",
        "dart_root",
        "baseline",
        "oracle",
        "prior_binding",
        "interpreter",
        "oracle_root",
    ):
        require(
            predecessor[key] == current[key], "predecessor environment differs: " + key
        )
    source = predecessor["current"]
    require(
        common()._source_identity(source["root"], source["commit"], source["files"])
        == source,
        "predecessor checkout differs",
    )
    before, after = source["files"], current["current"]["files"]
    changed = {
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    }
    require(changed <= CORRECTION_FILES, "correction changes numerical source")
    return manifest, predecessor


def verify_fit_stage(entry, stage, context):
    """Accept the exact authenticated predecessor pair; this worker cannot fit."""
    require(stage in ("fit-v4", "fit-shared"), "inheritance is restricted to fits")
    run = verify_stage(entry["path"], entry["sha256"], stage=stage)
    manifest, predecessor = inherited_fit_manifest(context)
    require(entry == manifest["fits"][stage], "fit is not the pinned predecessor stage")
    require(
        run.get("status") == "complete"
        and run.get("binding_sha256") == manifest["predecessor_binding"]["sha256"]
        and run.get("protocol_sha256") == context["protocol_sha256"]
        and run.get("implementation_commit") == manifest["predecessor_commit"],
        "inherited fit provenance differs",
    )
    request = common().read(Path(entry["path"]) / "request.json")
    require(
        request["stage"] == stage
        and request["binding_path"] == manifest["predecessor_binding"]["path"]
        and request["binding_sha256"] == manifest["predecessor_binding"]["sha256"]
        and request["protocol_path"] == predecessor["inputs"]["protocol"]["path"]
        and request["protocol_sha256"] == context["protocol_sha256"],
        "inherited fit request provenance differs",
    )
    return run


def dart_modules(protocol, binding):
    """Authenticate import locations; importing cannot create a plant or trial."""
    root = Path(protocol["dart"]["root"]).resolve()
    sys.path.insert(0, str(root / "src"))
    import crazydart
    import crazyflow
    from crazydart import mission, planning, plant

    require(
        Path(crazyflow.__file__).resolve().parent
        == Path(binding["simulator_sources"]["crazyflow"]["package_root"]).resolve(),
        "Dart imported Crazyflow package root differs",
    )

    for module in (crazydart, mission, planning, plant):
        file = Path(module.__file__).resolve()
        relative = str(file.relative_to(root))
        require(relative in DART_SOURCES, "unexpected Dart imported source")
        require(
            common().digest(file) == binding["dart_source_sha256"][relative],
            "Dart imported source bytes differ",
        )
    return mission, planning, plant


def read_recording(protocol, name):
    """Decode observations/issued commands only; ignore vehicle-family metadata."""
    import numpy as np

    key = f"artifacts/demo/telemetry/{name}.npz"
    row = protocol["dart"]["files"][key]
    require(common().digest(row["path"]) == row["sha256"], "recording bytes differ")
    with np.load(row["path"], allow_pickle=False) as archive:
        values = {
            k: np.array(archive[k], copy=True)
            for k in ("time_s", "states", "controls", "control_prefix")
        }
        spec = json.loads(str(archive["spec_json"]))
        labels = json.loads(str(archive["labels_json"]))
    require(
        spec["state_schema"] == "rigid_body_13_nwu_flu_wxyz_v1",
        "Dart state semantics differ",
    )
    require(
        values["states"].shape == (301, 13) and values["controls"].shape == (300, 4),
        "Dart recording support differs",
    )
    require(
        all(np.isfinite(v).all() for v in values.values()),
        "nonfinite ordinary recording",
    )
    np.testing.assert_allclose(np.diff(values["time_s"]), DT, rtol=0, atol=1e-9)
    require(labels["split"] == name.split("-")[0], "recording role differs")
    channels = [c for c in spec["channels"] if c["kind"] == "control"]
    require(
        len(channels) == 4 and len({c["name"] for c in channels}) == 4,
        "command identities differ",
    )
    require(
        all(
            c["unit"] == "1" and c["semantic"] == "normalized_command" for c in channels
        ),
        "command units differ",
    )
    # Names establish ordering only. Roles, family, geometry and plant parameters
    # are deliberately absent from the numerical recording contract.
    values["channels"] = tuple(
        json.dumps(
            {k: c[k] for k in ("name", "unit", "semantic", "minimum", "maximum")},
            sort_keys=True,
        )
        for c in channels
    )
    values["minimum"] = np.array([c["minimum"] for c in channels])
    values["maximum"] = np.array([c["maximum"] for c in channels])
    return values


def observed(states):
    import jax
    import numpy as np

    from .learned_plan import observed_from_state

    with jax.enable_x64(True):
        return np.asarray(jax.vmap(observed_from_state)(states))


def prepare(protocol):
    import numpy as np

    from glassbox import learner
    from glassbox.recordings import SequenceCollection, SequenceSegment

    from .learned_plan import OBSERVED_CHANNELS

    recordings = [read_recording(protocol, name) for name in (*TRAIN, *DEVELOPMENT)]
    require(
        all(x["channels"] == recordings[0]["channels"] for x in recordings),
        "recording command order differs",
    )
    collection = SequenceCollection(
        tuple(
            SequenceSegment(name, "whole", observed(x["states"]), x["controls"], DT)
            for name, x in zip((*TRAIN, *DEVELOPMENT), recordings, strict=True)
        ),
        configuration_id="dart-ordinary-recordings-v1",
        state_channels=OBSERVED_CHANNELS,
        input_channels=recordings[0]["channels"],
    )
    train, development = (
        learner._extract(collection, TRAIN, 1536),
        learner._extract(collection, DEVELOPMENT, 256),
    )
    require(
        len(train.keys) == 1536 and len(development.keys) == 256,
        "Dart cache budget/support differs",
    )
    require(
        {k.recording_id for k in train.keys} == set(TRAIN)
        and {k.recording_id for k in development.keys} == set(DEVELOPMENT),
        "Dart cache roles differ",
    )
    require(np.isfinite(train.batch.future_states).all(), "Dart cache is nonfinite")
    return (
        train,
        development,
        learner._contract(collection),
        learner._recording_content(collection),
    )


def save_preparation(output, prepared):
    import numpy as np

    from .public_mean_flight_fit import _cache_payload

    metadata, arrays = _cache_payload(*prepared)
    np.savez_compressed(
        Path(output) / "preparation.npz",
        metadata=json.dumps(metadata, sort_keys=True),
        **arrays,
    )


def check_preparation(model, prepared):
    import numpy as np

    from .public_mean_flight_fit import _cache_payload

    left, la = _cache_payload(*prepared)
    right, ra = _cache_payload(
        model._train, model._development, model.contract, model._seen
    )
    require(
        left == right and set(la) == set(ra), "Dart selected cache metadata differs"
    )
    for name in la:
        require(
            np.array_equal(la[name], ra[name]),
            "Dart selected cache arrays differ: " + name,
        )


def load_model(entry, arm, context):
    run = verify_fit_stage(
        entry,
        "fit-v4" if arm == "v4_research_extrapolation" else "fit-shared",
        context,
    )
    if run["status"] != "complete":
        require(
            run["status"] == "fit_failed"
            or (arm == "shared_vehicle" and run["status"] == "unavailable"),
            "nonterminal or integrity-failed Dart fit",
        )
        return None
    if arm == "v4_research_extrapolation":
        from glassbox.learner import LearnedDynamics

        return LearnedDynamics.load(Path(entry["path"]) / "model.npz")
    from .shared_vehicle import SharedVehicleDynamics

    return SharedVehicleDynamics.load(Path(entry["path"]) / "model.npz")


def weighting_control(source, prepared, context):
    """Use actual v4 weights even if a later numerical fit step failed."""
    import numpy as np

    from .public_mean_flight_fit import _cache_payload
    from .shared_vehicle_experiment import verify_capture

    root = Path(source["path"])
    run = verify_fit_stage(source, "fit-v4", context)
    require(
        run["status"] in ("complete", "fit_failed"),
        "weighting control is incomplete or corrupt",
    )
    metadata, values = _cache_payload(*prepared)
    saved = _arrays(root / "preparation.npz")
    require(
        json.loads(str(saved.pop("metadata"))) == metadata,
        "weighting control preparation metadata differs",
    )
    _same_arrays(saved, values, "weighting control preparation")
    model = load_model(source, "v4_research_extrapolation", context)
    if model is not None:
        check_preparation(model, prepared)
    verify_capture(root / "capture", model)
    path = root / "capture/weights.npz"
    if not path.exists():
        require(run["status"] == "fit_failed", "completed control lacks weights")
        return None
    arrays = _arrays(path)
    scale, weights = arrays["normalization"], arrays["channel_weights"]
    horizon, dimensions = prepared[0].batch.future_states.shape[1:]
    require(
        scale.shape == (horizon, dimensions)
        and weights.shape == (dimensions,)
        and scale.dtype == np.dtype("float64")
        and weights.dtype == np.dtype("float64")
        and np.isfinite(scale).all()
        and np.isfinite(weights).all()
        and (scale > 0).all()
        and (weights > 0).all(),
        "actual weighting control arrays are invalid",
    )
    return scale, weights


def check_pair(inputs, context):
    """The architecture candidate must use this exact newly fitted v4 control."""
    candidate, v4 = inputs["candidate"], inputs["v4"]
    candidate_run = verify_fit_stage(candidate, "fit-shared", context)
    verify_fit_stage(v4, "fit-v4", context)
    request = common().read(Path(candidate["path"]) / "request.json")
    require(
        request["inputs"]["v4"] == v4,
        "candidate belongs to a different Dart v4 weighting control",
    )
    weight_path = Path(candidate["path"]) / "capture/weights.npz"
    if weight_path.exists():
        current, reference = (
            _arrays(weight_path),
            _arrays(Path(v4["path"]) / "capture/weights.npz"),
        )
        _same_arrays(
            {k: current[k] for k in ("normalization", "channel_weights")},
            {k: reference[k] for k in ("normalization", "channel_weights")},
            "Dart shared objective weights",
        )
    else:
        require(
            candidate_run["status"] in ("fit_failed", "unavailable"),
            "completed candidate lacks actual weight witness",
        )


def fit_worker(output, protocol, request):
    import numpy as np

    from glassbox import _sequence_model, learner

    from . import shared_vehicle
    from .shared_vehicle_experiment import FitCapture

    prepared = prepare(protocol)
    save_preparation(output, prepared)
    objective = None
    if request["stage"] == "fit-shared":
        objective = weighting_control(request["inputs"]["v4"], prepared, request)
        if objective is None:
            return {
                "status": "unavailable",
                "fits": 0,
                "initializers": 0,
                "reason": "Numerically failed v4 control did not reach actual objective weights.",
                "public_promotion": False,
            }
    capture = FitCapture(output / "capture")
    start = time.monotonic()
    calibration_times = []
    original_calibrate = learner._calibrate

    def calibrate(*args, **kwargs):
        began = time.monotonic()
        result = original_calibrate(*args, **kwargs)
        calibration_times.append(time.monotonic() - began)
        if request["stage"] == "fit-v4":
            capture({"phase": "calibrated", "calls": len(calibration_times)})
        return result

    try:
        with patch.object(learner, "_calibrate", calibrate):
            if request["stage"] == "fit-v4":
                original_initializer = _sequence_model.initialize_sequence_model
                original_solve = np.linalg.solve

                def initialize(*args, **kwargs):
                    capture({"phase": "initializer_entered"})
                    return original_initializer(*args, **kwargs)

                def solve(*args, **kwargs):
                    value = original_solve(*args, **kwargs)
                    capture({"phase": "ridge_solve"})
                    return value

                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch.object(_sequence_model, "_observe_attempt", capture)
                    )
                    stack.enter_context(
                        patch.object(
                            _sequence_model, "initialize_sequence_model", initialize
                        )
                    )
                    stack.enter_context(patch.object(np.linalg, "solve", solve))
                    model = learner._train(*prepared)
            else:
                scale, weights = objective
                model = shared_vehicle._train(
                    *prepared,
                    error_scale=scale,
                    channel_weights=weights,
                    observer=capture,
                )
    except BaseException:
        capture.finish(None)
        raise
    elapsed = time.monotonic() - start
    check_preparation(model, prepared)
    model.save(output / "model.npz")
    write(output / "report.json", model.report)
    evidence = capture.finish(model)
    write(output / "fit-evidence.json", evidence)
    return {
        "status": "complete",
        "fits": 1,
        "fingerprint": model.fingerprint(),
        "fit_and_calibration_wall_s": elapsed,
        "calibration_wall_s": sum(calibration_times),
        "calibration_calls": len(calibration_times),
        "work": evidence,
        "public_promotion": False,
    }


def physical_rollout(model, *, intrinsic=False):
    """Long means are research extrapolation; no envelope is extrapolated."""
    import jax
    import jax.numpy as jnp

    from glassbox.core.geometry import rotation_to_quaternion

    from .learned_plan import states_from_observed

    def rollout(initial, commands):
        state, past, issued = initial
        predicted = model._model.rollout(past, issued, commands)
        rotations = predicted[:, 6:15].reshape((-1, 3, 3))
        proper = jnp.all(jnp.linalg.det(rotations) > 0) & jnp.all(
            jnp.isfinite(predicted)
        )
        if intrinsic:
            velocity = predicted[:, :3]
            speeds = jnp.concatenate((state[None, 3:6], velocity))
            positions = state[:3] + DT * jnp.cumsum(
                0.5 * (speeds[:-1] + speeds[1:]), axis=0
            )
            quaternions = jax.vmap(rotation_to_quaternion)(rotations)
            future = jnp.concatenate(
                (positions, velocity, quaternions, predicted[:, 3:6]), axis=1
            )
            states = jnp.concatenate((state[None], future))
        else:
            states = states_from_observed(state, predicted, DT)
        return jnp.where(proper, states, jnp.full_like(states, jnp.nan))

    return rollout


def structured_rollout(model):
    import jax
    import jax.numpy as jnp

    def rollout(initial, commands):
        state, latent = initial

        def step(carry, command):
            following = model.transition(*carry, command)
            return following, following[0]

        _, future = jax.lax.scan(step, (state, latent), commands)
        return jnp.concatenate((state[None], future))

    return rollout


def target_from_protocol(protocol, mission):
    t = protocol["dart"]["target"]
    return mission.Target(
        center=t["center_m"],
        normal=t["normal"],
        contact_offset=t["body_contact_offset_m"],
        radius_m=t["radius_m_max"],
        angle_tolerance_deg=t["axis_error_deg_max"],
        minimum_normal_speed_m_s=t["normal_speed_m_s_range"][0],
        maximum_normal_speed_m_s=t["normal_speed_m_s_range"][1],
        maximum_tangent_speed_m_s=t["tangential_speed_m_s_max"],
    )


def prelude(plant, protocol):
    import jax.numpy as jnp
    import numpy as np

    commands = read_recording(protocol, "train-00")["control_prefix"]
    require(
        commands.shape == (HISTORY, 4) and np.all(commands == 0.65195631980896),
        "prelude command tape differs",
    )
    state = plant.initial_state()
    np.testing.assert_array_equal(
        np.asarray(state[:13]), protocol["dart"]["initial_observation"]
    )
    states = [np.asarray(state)]
    for command in commands:
        state = plant.step(state, jnp.asarray(command))
        states.append(np.asarray(state))
    require(np.isfinite(states).all(), "nonfinite shared prelude")
    return np.asarray(states), commands


def live_initial(arm, model, states, commands):
    import jax.numpy as jnp

    if arm == "structured_causal_history":
        return jnp.asarray(states[-1][:13]), model.initial_latent_state(
            jnp.asarray(commands)
        )
    require(
        len(states) == len(commands) + 1 and len(commands) >= HISTORY,
        "causal history support differs",
    )
    return (
        jnp.asarray(states[-1][:13]),
        jnp.asarray(observed(states[-HISTORY - 1 :, :13])),
        jnp.asarray(commands[-HISTORY:]),
    )


def _arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def _save(path, **arrays):
    import numpy as np

    require(not Path(path).exists(), "array evidence already exists")
    np.savez_compressed(path, **arrays)


def control_worker(output, protocol, binding, request):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from glassbox.belief.belief import DynamicsBelief

    check_pair(request["inputs"], request)
    mission, planning, plants = dart_modules(protocol, binding)
    target = target_from_protocol(protocol, mission)
    plant = plants.CrazyflowPlant()
    prefix_states, prefix_commands = prelude(plant, protocol)
    _save(output / "prelude.npz", states=prefix_states, commands=prefix_commands)
    tape = _arrays(
        protocol["dart"]["files"]["artifacts/demo/learned_prediction.npz"]["path"]
    )["commands"]
    require(tape.shape == (HORIZON, 4), "original seed horizon differs")
    bounds = read_recording(protocol, "train-00")
    require(
        np.all(bounds["minimum"] == bounds["minimum"][0])
        and np.all(bounds["maximum"] == 1),
        "original command bounds differ",
    )
    models = {
        "v4_research_extrapolation": load_model(
            request["inputs"]["v4"], "v4_research_extrapolation", request
        ),
        "shared_vehicle": load_model(
            request["inputs"]["candidate"], "shared_vehicle", request
        ),
        "structured_causal_history": DynamicsBelief.load(
            protocol["dart"]["files"]["artifacts/demo/fit/belief.json"]["path"]
        ).model,
    }
    results = {}
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        model = models[arm]
        if model is None:
            results[arm] = {"status": "unavailable", "hit": False}
            write(directory / "result.json", results[arm])
            continue
        states, commands = [np.array(prefix_states[-1])], []
        full_states = np.array(prefix_states)
        issued = np.array(prefix_commands)
        rollout = (
            structured_rollout(model)
            if arm == "structured_causal_history"
            else physical_rollout(model, intrinsic=arm == "shared_vehicle")
        )
        planner = planning.DirectPlanner(
            rollout,
            target,
            steps=HORIZON,
            block_steps=REPLAN,
            minimum=float(bounds["minimum"][0]),
            maximum=1.0,
        )
        plan = np.array(tape)
        diagnostics = []
        status = "complete"
        error = None
        try:
            for step in range(0, HORIZON, REPLAN):
                initial = live_initial(arm, model, full_states, issued)
                _save(
                    directory / f"solve-{step:04d}-inputs.npz",
                    **(
                        {
                            "state": np.asarray(initial[0]),
                            "latent": np.asarray(initial[1]),
                        }
                        if arm == "structured_causal_history"
                        else {
                            "state": np.asarray(initial[0]),
                            "past_states": np.asarray(initial[1]),
                            "past_inputs": np.asarray(initial[2]),
                        }
                    ),
                    seed=plan,
                    active_steps=np.asarray(HORIZON - step),
                )
                actual_kernel = planner.value_gradient
                calls = 0
                with (directory / f"solve-{step:04d}-gradients.jsonl").open(
                    "x"
                ) as gradient_log:

                    def witnessed_gradient(*args, _kernel=actual_kernel):
                        nonlocal calls
                        value, gradient = _kernel(*args)
                        calls += 1
                        scalar, array = float(value), np.asarray(gradient)
                        finite = bool(np.isfinite(scalar) and np.isfinite(array).all())
                        gradient_log.write(
                            json.dumps(
                                {
                                    "call": calls,
                                    "finite": finite,
                                    "objective": scalar
                                    if np.isfinite(scalar)
                                    else str(scalar),
                                    "gradient_norm": float(np.linalg.norm(array))
                                    if finite
                                    else None,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        gradient_log.flush()
                        return value, gradient

                    planner.value_gradient = witnessed_gradient
                    try:
                        plan, predicted, diagnostic = planner.solve(
                            initial, plan, maxiter=100, active_steps=HORIZON - step
                        )
                    finally:
                        planner.value_gradient = actual_kernel
                diagnostic["actual_gradient_calls"] = calls
                _save(
                    directory / f"solve-{step:04d}-outputs.npz",
                    commands=plan,
                    states=predicted,
                )
                diagnostics.append({"step": step, **diagnostic})
                write(directory / f"solve-{step:04d}.json", diagnostics[-1])
                for command in plan[: min(REPLAN, HORIZON - step)]:
                    state = plant.step(jnp.asarray(states[-1]), jnp.asarray(command))
                    require(np.isfinite(state).all(), "nonfinite plant state")
                    states.append(np.asarray(state))
                    commands.append(command.copy())
                    full_states = np.concatenate((full_states, np.asarray(state)[None]))
                    issued = np.concatenate((issued, command[None]))
                    _save(
                        directory / f"executed-{len(commands):04d}.npz",
                        state=np.asarray(state),
                        command=command,
                    )
                    if float(mission.signed_distance(state, target)) <= 0:
                        break
                if float(mission.signed_distance(states[-1], target)) <= 0:
                    break
                plan = np.concatenate(
                    (plan[REPLAN:], np.repeat(plan[-1:], REPLAN, axis=0))
                )
        except Exception as exc:
            status = "failed"
            error = {"type": type(exc).__name__, "message": str(exc)}
            (directory / "error.txt").write_text(traceback.format_exc())
        _save(
            directory / "trajectory.npz",
            states=np.asarray(states),
            commands=np.asarray(commands).reshape((-1, 4)),
            time_s=np.arange(len(states)) * DT,
        )
        contact = (
            mission.score_contact(
                np.asarray(states), np.arange(len(states)) * DT, target
            )
            if len(states) > 1
            else {"hit": False, "contact": False, "reason": "no_executed_interval"}
        )
        results[arm] = {
            "status": status,
            "error": error,
            "hit": bool(status == "complete" and contact["hit"]),
            "contact": contact,
            "solves": len(diagnostics),
            "converged_solves": sum(d["converged"] for d in diagnostics),
            "diagnostics": diagnostics,
            "executed_intervals": len(commands),
            "ideal_physical_state_observation": True,
            "simulator_actuator_information_supplied": False,
            "real_time_claim": False,
        }
        write(directory / "result.json", results[arm])
        jax.clear_caches()
    return {
        "status": "complete",
        "fits": 0,
        "optimizer_trials_per_available_arm": 1,
        "prelude_intervals": 50,
        "deadline_after_prelude_s": 1.2,
        "arms": results,
        "public_promotion": False,
    }


def diagnostic_queries(length):
    return [
        (h, origin)
        for h in (5, 15, 25, 120)
        for origin in range(HISTORY, length - h, 25)
    ]


def forecast_worker(output, protocol, request):
    import jax
    import jax.numpy as jnp
    import numpy as np

    check_pair(request["inputs"], request)
    rows = []
    for arm, key in (
        ("v4_research_extrapolation", "v4"),
        ("shared_vehicle", "candidate"),
    ):
        model = load_model(request["inputs"][key], arm, request)
        if model is None:
            rows.append({"arm": arm, "status": "unavailable"})
            continue
        for name in TEST:
            rec = read_recording(protocol, name)
            x = observed(rec["states"])
            for h, origin in diagnostic_queries(len(x)):
                past = x[origin - HISTORY : origin + 1]
                up = rec["controls"][origin - HISTORY : origin]
                future = rec["controls"][origin : origin + h]
                target = x[origin + 1 : origin + h + 1]
                key = f"{arm}-{name}-{origin:04d}-{h:03d}"
                row = {
                    "arm": arm,
                    "recording": name,
                    "origin": origin,
                    "horizon_steps": h,
                    "horizon_s": h * DT,
                    "status": "failed",
                }
                _save(
                    output / f"{key}-input.npz",
                    past_states=past,
                    past_inputs=up,
                    future_inputs=future,
                    target=target,
                )
                try:
                    kernel = jax.jit(model._model.rollout)
                    px, pu, uf = map(jnp.asarray, (past, up, future))
                    prediction = np.asarray(kernel(px, pu, uf))
                    tangent = jnp.zeros_like(uf).at[0, 0].set(1)
                    jvp = np.asarray(
                        jax.jvp(
                            lambda u, _kernel=kernel, _x=px, _p=pu: _kernel(_x, _p, u),
                            (uf,),
                            (tangent,),
                        )[1]
                    )
                    require(
                        prediction.shape == target.shape
                        and np.isfinite(prediction).all()
                        and np.isfinite(jvp).all(),
                        "nonfinite forecast or command JVP",
                    )
                    _save(
                        output / f"{key}-output.npz",
                        prediction=prediction,
                        first_command_jvp=jvp,
                    )
                    row.update(
                        status="complete",
                        endpoint_rmse={
                            g: float(
                                np.sqrt(
                                    np.mean((prediction[-1, s] - target[-1, s]) ** 2)
                                )
                            )
                            for g, s in (
                                ("velocity_m_s", slice(0, 3)),
                                ("body_rate_rad_s", slice(3, 6)),
                                ("rotation_entries", slice(6, 15)),
                            )
                        },
                    )
                except Exception as exc:
                    row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                rows.append(row)
                write(output / f"{key}.json", row)
    return {
        "status": "complete",
        "fits": 0,
        "scope": "Descriptive held-out ordinary-recording forecasts and computational JVP finiteness; no physical derivative or long-horizon envelope qualification.",
        "rows": rows,
    }


def _same_arrays(actual, expected, label):
    import numpy as np

    require(set(actual) == set(expected), label + ": array roster")
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        require(
            a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b),
            label + ": " + key,
        )


def replay_fit(output, source, context, protocol):
    from .shared_vehicle_experiment import verify_capture

    stage = verify_stage(
        source["path"], source["sha256"], binding_sha256=context["binding_sha256"]
    )
    arm = (
        "v4_research_extrapolation" if stage["stage"] == "fit-v4" else "shared_vehicle"
    )
    if arm == "shared_vehicle":
        saved_request = common().read(Path(source["path"]) / "request.json")
        check_pair({"candidate": source, "v4": saved_request["inputs"]["v4"]}, context)
    prepared = prepare(protocol)
    save_preparation(output, prepared)
    _same_arrays(
        _arrays(output / "preparation.npz"),
        _arrays(Path(source["path"]) / "preparation.npz"),
        "replayed Dart preparation",
    )
    if arm == "shared_vehicle":
        objective = weighting_control(saved_request["inputs"]["v4"], prepared, context)
        if stage["status"] == "unavailable":
            original = Path(source["path"])
            require(
                objective is None
                and not (original / "capture").exists()
                and not (original / "model.npz").exists(),
                "unavailable candidate performed numerical work",
            )
            outcome = common().read(original / "outcome.json")
            require(
                outcome["status"] == "unavailable"
                and outcome["fits"] == 0
                and outcome["initializers"] == 0,
                "unavailable candidate count",
            )
            return {
                "status": "complete",
                "fits": 0,
                "initializers": 0,
                "selected_model_available": False,
                "reason": outcome["reason"],
                "exact_preparation": True,
            }
        require(objective is not None, "candidate used unavailable v4 weights")
    model = load_model(source, arm, context)
    if model is None:
        evidence = verify_capture(Path(source["path"]) / "capture", None)
        return {
            "status": "complete",
            "fits": 0,
            "initializers": 0,
            "selected_model_available": False,
            "failure_prefix": evidence,
        }
    check_preparation(model, prepared)
    evidence = verify_capture(Path(source["path"]) / "capture", model, replay=True)
    return {
        "status": "complete",
        "exact": True,
        "fits": 0,
        "initializers": 0,
        "scope": "Reconstructed ordinary cache, saved selection/work, checkpoint forecasts and calibration; no fitting or optimizer.",
        "evidence": evidence,
    }


def replay_forecast(output, source, context, protocol):
    """Replay the declared query roster with the same selected models, no fits."""
    original = Path(source["path"])
    saved = common().read(original / "request.json")
    result = forecast_worker(output, protocol, {**context, "inputs": saved["inputs"]})
    actual = common().read(original / "outcome.json")
    require(result == actual, "saved forecast diagnostic result differs")
    arrays = 0
    for path in sorted(original.glob("*.npz")):
        _same_arrays(
            _arrays(output / path.name), _arrays(path), "saved forecast replay"
        )
        arrays += len(_arrays(path))
    return {
        "status": "complete",
        "exact": True,
        "fits": 0,
        "initializers": 0,
        "arrays": arrays,
        "scope": "Exact saved query inputs, selected forecasts, command JVPs and physical diagnostic reduction; no optimizer or selection.",
    }


def replay_control(output, source, context, protocol, binding):
    import jax.numpy as jnp
    import numpy as np

    from glassbox.belief.belief import DynamicsBelief

    original = Path(source["path"])
    saved = common().read(original / "request.json")
    check_pair(saved["inputs"], context)
    mission, _, plants = dart_modules(protocol, binding)
    plant = plants.CrazyflowPlant()
    target = target_from_protocol(protocol, mission)
    prefix, issued = prelude(plant, protocol)
    _same_arrays(
        {"states": prefix, "commands": issued},
        _arrays(original / "prelude.npz"),
        "actual common prelude",
    )
    _save(output / "prelude.npz", states=prefix, commands=issued)
    models = {
        "v4_research_extrapolation": load_model(
            saved["inputs"]["v4"], "v4_research_extrapolation", context
        ),
        "shared_vehicle": load_model(
            saved["inputs"]["candidate"], "shared_vehicle", context
        ),
        "structured_causal_history": DynamicsBelief.load(
            protocol["dart"]["files"]["artifacts/demo/fit/belief.json"]["path"]
        ).model,
    }
    counts = {}
    for arm in ARMS:
        directory = original / arm
        result = common().read(directory / "result.json")
        model = models[arm]
        if result["status"] == "unavailable":
            require(model is None, "available model labelled unavailable")
            counts[arm] = {"available": False}
            continue
        trajectory = _arrays(directory / "trajectory.npz")
        commands = trajectory["commands"]
        states = [prefix[-1]]
        for command in commands:
            states.append(
                np.asarray(plant.step(jnp.asarray(states[-1]), jnp.asarray(command)))
            )
        state_array = np.asarray(states)
        _same_arrays(
            {
                "states": state_array,
                "commands": commands,
                "time_s": np.arange(len(states)) * DT,
            },
            trajectory,
            "recorded command plant replay",
        )
        _save(output / f"{arm}-trajectory.npz", states=state_array, commands=commands)
        full_states = np.concatenate((prefix, state_array[1:]))
        all_commands = np.concatenate((issued, commands))
        rollout = (
            structured_rollout(model)
            if arm == "structured_causal_history"
            else physical_rollout(model, intrinsic=arm == "shared_vehicle")
        )
        mean_arrays = 0
        seed_path = protocol["dart"]["files"]["artifacts/demo/learned_prediction.npz"][
            "path"
        ]
        expected_seed = _arrays(seed_path)["commands"]
        for ipath in sorted(directory.glob("solve-*-inputs.npz")):
            step = int(ipath.name.split("-")[1])
            row = _arrays(ipath)
            _same_arrays(
                {"seed": row["seed"]}, {"seed": expected_seed}, "optimizer warm start"
            )
            require(
                step % REPLAN == 0
                and step <= len(commands)
                and int(row["active_steps"]) == HORIZON - step,
                "saved optimizer origin differs",
            )
            initial = live_initial(
                arm,
                model,
                full_states[: HISTORY + step + 1],
                all_commands[: HISTORY + step],
            )
            expected = (
                {"state": np.asarray(initial[0]), "latent": np.asarray(initial[1])}
                if arm == "structured_causal_history"
                else {
                    "state": np.asarray(initial[0]),
                    "past_states": np.asarray(initial[1]),
                    "past_inputs": np.asarray(initial[2]),
                }
            )
            _same_arrays(
                expected, {k: row[k] for k in expected}, "causal optimizer inputs"
            )
            opath = directory / ipath.name.replace("inputs", "outputs")
            if opath.exists():
                out = _arrays(opath)
                predicted = np.asarray(rollout(initial, jnp.asarray(out["commands"])))
                _same_arrays(
                    {"states": predicted},
                    {"states": out["states"]},
                    "selected planned mean",
                )
                executed = min(REPLAN, len(commands) - step)
                require(
                    np.array_equal(
                        commands[step : step + executed], out["commands"][:executed]
                    ),
                    "executed optimizer command prefix differs",
                )
                expected_seed = np.concatenate(
                    (
                        out["commands"][REPLAN:],
                        np.repeat(out["commands"][-1:], REPLAN, axis=0),
                    )
                )
                mean_arrays += 1
            else:
                require(
                    result["status"] == "failed",
                    "complete trial lacks selected solve output",
                )
        contact = (
            mission.score_contact(state_array, np.arange(len(states)) * DT, target)
            if len(states) > 1
            else {"hit": False, "contact": False, "reason": "no_executed_interval"}
        )
        require(contact == result["contact"], "first-contact score differs")
        counts[arm] = {
            "available": True,
            "physical_states": len(states),
            "replayed_planned_means": mean_arrays,
            "contact": contact,
            "original_trial_status": result["status"],
        }
    return {
        "status": "complete",
        "exact": True,
        "fits": 0,
        "initializers": 0,
        "optimizer_calls": 0,
        "scope": "Actual prelude, recorded commands, reconstructed causal optimizer inputs, saved selected means and first-contact score. Optimizer is not rerun; original failures and nonconvergence remain unchanged.",
        "arms": counts,
    }


def worker(request):
    require(not request["stage"].startswith("fit-"), "correction forbids new fits")
    output = Path(request["output"])
    p, b = authenticate(request)
    import jax

    require(not jax.config.jax_enable_x64, "Dart workers require ambient float32")
    stage = request["stage"]
    if stage.startswith("fit-"):
        outcome = fit_worker(output, p, request)
    elif stage == "forecast":
        outcome = forecast_worker(output, p, request)
    elif stage == "control":
        outcome = control_worker(output, p, b, request)
    else:
        source = request["inputs"]["source"]
        original = verify_stage(
            source["path"], source["sha256"], binding_sha256=request["binding_sha256"]
        )
        require(
            original["status"] == "complete"
            or (
                original["stage"].startswith("fit-")
                and original["status"] in ("fit_failed", "unavailable")
            ),
            "replay requires terminal saved stage",
        )
        if original["stage"].startswith("fit-"):
            outcome = replay_fit(output, source, request, p)
        elif original["stage"] == "forecast":
            outcome = replay_forecast(output, source, request, p)
        elif original["stage"] == "control":
            outcome = replay_control(output, source, request, p, b)
        else:
            raise ValueError("cannot replay a replay stage")
    require(not jax.config.jax_enable_x64, "Dart stage changed ambient precision")
    authenticate(request)
    write(output / "outcome.json", outcome)
    return outcome


def run_stage(
    stage,
    output,
    *,
    protocol_path,
    protocol_sha256,
    binding_path,
    binding_sha256,
    inputs=None,
):
    """Supervise exactly one stage; retain a bounded failed prefix without retry."""
    require(stage in STAGES, "unknown Dart stage")
    require(not stage.startswith("fit-"), "correction forbids new fits")
    context = dict(
        protocol_path=str(Path(protocol_path).resolve()),
        protocol_sha256=protocol_sha256,
        binding_path=str(Path(binding_path).resolve()),
        binding_sha256=binding_sha256,
    )
    p, b = authenticate(context)
    inputs = {} if inputs is None else inputs
    required = {
        "fit-v4": set(),
        "fit-shared": {"v4"},
        "forecast": {"v4", "candidate"},
        "control": {"v4", "candidate"},
        "replay": {"source"},
    }[stage]
    require(set(inputs) == required, "Dart prerequisite roster differs")
    for key, entry in inputs.items():
        expected = (
            None if key == "source" else "fit-v4" if key == "v4" else "fit-shared"
        )
        run = (
            verify_fit_stage(entry, expected, context)
            if expected is not None
            else verify_stage(
                entry["path"],
                entry["sha256"],
                binding_sha256=binding_sha256,
            )
        )
        require(
            run["status"] in ("complete", "fit_failed")
            or (run["stage"] == "fit-shared" and run["status"] == "unavailable"),
            "Dart prerequisite is incomplete or corrupt",
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = {**context, "stage": stage, "output": str(output), "inputs": inputs}
    write(output / "request.json", request)
    request_sha256 = common().digest(output / "request.json")
    env = os.environ.copy()
    env["SCIPY_ARRAY_API"] = "1"
    env["JAX_ENABLE_X64"] = "0"
    env["PYTHONPATH"] = (
        str(common().ROOT / "src") + os.pathsep + str(Path(p["dart"]["root"]) / "src")
    )
    command = [
        b["interpreter"],
        "-m",
        MODULE,
        "--request-sha256",
        request_sha256,
        "--worker",
        str(output / "request.json"),
    ]
    write(
        output / "command.json",
        {
            "command": command,
            "cwd": str(common().ROOT),
            "environment": {
                "SCIPY_ARRAY_API": "1",
                "JAX_ENABLE_X64": "0",
                "PYTHONPATH": env["PYTHONPATH"],
            },
            "timeout_s": 14400,
            "automatic_retry": False,
        },
    )
    start = time.monotonic()
    timeout = False
    with (output / "worker.log").open("x") as handle:
        try:
            process = subprocess.run(
                command,
                cwd=common().ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=False,
            )
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = None
            timeout = True
    elapsed = time.monotonic() - start
    write(
        output / "exit.json",
        {"returncode": code, "timed_out": timeout, "elapsed_s": elapsed},
    )
    status = "hard_timeout_incomplete" if timeout else "failed"
    if (output / "outcome.json").exists():
        outcome = common().read(output / "outcome.json")
        if not timeout:
            status = outcome["status"]
    if code not in (0, None) and status == "complete":
        status = "failed"
    try:
        require(
            common().digest(output / "request.json") == request_sha256,
            "Dart request changed after dispatch",
        )
        authenticate(context)
    except Exception as exc:
        status = "failed"
        write(
            output / "final-integrity-error.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
    result = dict(
        format=FORMAT,
        stage=stage,
        status=status,
        protocol_sha256=protocol_sha256,
        binding_sha256=binding_sha256,
        implementation_commit=b["implementation_commit"],
        request_sha256=request_sha256,
        elapsed_s=elapsed,
        returncode=code,
        files=inventory(output),
    )
    write(output / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path, default=common().PROTOCOL_PATH)
    parser.add_argument("--protocol-sha256", default=common().PROTOCOL_SHA256)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--binding-sha256")
    parser.add_argument(
        "--inputs", type=Path, help="JSON mapping prerequisite name to {path,sha256}"
    )
    args = parser.parse_args(argv)
    if args.worker:
        require(
            args.request_sha256 is not None
            and common().digest(args.worker) == args.request_sha256,
            "Dart worker request SHA differs",
        )
        request = common().read(args.worker)
        try:
            result = worker(request)
        except Exception as exc:
            # Only explicit numerical fitter errors are scientifically unavailable.
            # Preparation, integrity and other programming failures remain failed.
            status = "failed"
            if request["stage"].startswith("fit-"):
                from glassbox._sequence_model import SequenceFitError

                if isinstance(exc, SequenceFitError) and "time" not in str(exc).lower():
                    status = "fit_failed"
            output = Path(request["output"])
            (output / "error.txt").write_text(traceback.format_exc())
            if not (output / "outcome.json").exists():
                write(
                    output / "outcome.json",
                    {
                        "status": status,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "scope": "Actual partial evidence retained; no retry.",
                    },
                )
            return 1
        return 0 if result["status"] == "complete" else 1
    require(
        args.stage and args.output and args.binding and args.binding_sha256,
        "stage/output/binding required",
    )
    result = run_stage(
        args.stage,
        args.output,
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        binding_path=args.binding,
        binding_sha256=args.binding_sha256,
        inputs=None if args.inputs is None else common().read(args.inputs),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
