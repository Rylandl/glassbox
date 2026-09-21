"""Replay the adopted baseline through public APIs; never fit, solve or simulate.

The baseline directory is a lossless consolidation of previously qualified
observations, predictions and actual planner callbacks. An externally pinned
manifest authenticates every input and expected output before any model call.
Dart objective replay additionally requires its unchanged external source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def exact(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and np.array_equal(actual, expected, equal_nan=True),
        f"{label}: dtype, shape or values differ",
    )


def verify_integrity(root, manifest_sha256):
    """Verify external authority before reading any numerical expectation."""
    root = Path(root)
    require(digest(root / "manifest.json") == manifest_sha256, "manifest hash differs")
    manifest = read(root / "manifest.json")
    require(manifest["format"] == "glassbox-portable-baseline-v1", "baseline format")
    require(bool(manifest["files"]), "empty baseline inventory")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "unsafe inventory path",
        )
        require(digest(root / relative) == expected, f"archive hash differs: {name}")
    required = {
        "models/dart.npz",
        "models/crazyflow.npz",
        "models/cascade.npz",
        "source-models/dart.npz",
        "source-models/crazyflow.npz",
        "source-models/cascade.npz",
        "dart/spec.json",
        "dart/gradient.npz",
        "dart/selected.npz",
        "dart/native.npz",
        "dart/initials.npz",
        "dart/prelude.npz",
        "dart/trajectory.npz",
        "dart/seed.npz",
        "dart/trial.json",
        "flights/spec.json",
        "provenance/dart-qualification-qualification.json",
    }
    require(required <= set(manifest["files"]), "incomplete baseline inventory")
    spec = read(root / "flights/spec.json")
    for cohort in spec["cohorts"]:
        require(
            {f"flights/{cohort}.json", f"flights/{cohort}.npz"}
            <= set(manifest["files"]),
            "missing flight cohort",
        )
    return manifest


def runtime_check(expected):
    import jax

    require(not jax.config.x64_enabled, "baseline requires JAX default32")
    require(jax.default_backend() == expected["backend"], "backend differs")
    require(platform.machine() == expected["runtime"]["machine"], "machine differs")
    require(
        platform.python_version() == expected["runtime"]["python"], "Python differs"
    )
    for name, version in expected["runtime"]["versions"].items():
        require(importlib.metadata.version(name) == version, f"{name} version differs")


def verify_models(root):
    from glassbox import LearnedDynamics

    models, identities = {}, {}
    for name in ("dart", "crazyflow", "cascade"):
        path = root / "models" / f"{name}.npz"
        public = arrays(path)
        original = arrays(root / "source-models" / f"{name}.npz")
        keys = {key for key in original if key.startswith(("param_", "norm_"))}
        require(
            keys == {key for key in public if key.startswith(("param_", "norm_"))},
            "model array roster",
        )
        for key in keys:
            exact(public[key], original[key], f"{name}/{key}")
        before = json.loads(str(original["metadata"]))["model"]
        after = json.loads(str(public["metadata"]))["model"]
        require(before == after, "core metadata changed")
        model = LearnedDynamics.load(path)
        # Numeric tampering must fail the public loader as well as the file seal.
        changed = {key: value.copy() for key, value in public.items()}
        key = sorted(keys)[0]
        changed[key].flat[0] += 0.125
        with tempfile.TemporaryDirectory(prefix="glassbox-corruption-") as directory:
            altered = Path(directory) / "altered.npz"
            np.savez_compressed(altered, **changed)
            try:
                LearnedDynamics.load(altered)
            except ValueError:
                pass
            else:
                raise ValueError("public loader accepted a numerically altered model")
        identities[name] = dict(
            fingerprint=model.fingerprint(),
            core_arrays_exact=len(keys),
            corruption_rejected=True,
        )
        models[name] = model
    return models, identities


def rotation(quaternion):
    q = np.asarray(quaternion)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def observed(states):
    states = np.asarray(states)
    return np.concatenate(
        (states[:, 3:6], states[:, 10:13], rotation(states[:, 6:10]).reshape(-1, 9)),
        axis=1,
    )


def kinematics(state, target):
    state = np.asarray(state, dtype=np.float64)
    r = rotation(state[6:10])
    offset = np.asarray(target["body_contact_offset_m"], dtype=np.float64)
    return (
        state[:3] + r @ offset,
        r[:, 2],
        state[3:6] + r @ np.cross(state[10:13], offset),
    )


def interpolate(a, b, fraction):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    state = (1 - fraction) * a + fraction * b
    qa, qb = a[6:10] / np.linalg.norm(a[6:10]), b[6:10] / np.linalg.norm(b[6:10])
    dot = float(qa @ qb)
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = (1 - fraction) * qa + fraction * qb
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        q = (
            np.sin((1 - fraction) * angle) * qa + np.sin(fraction * angle) * qb
        ) / np.sin(angle)
    state[6:10] = q / np.linalg.norm(q)
    return state


def contact_score(states, times, target):
    """First crossing and original task thresholds, computed independently."""
    states, times = np.asarray(states), np.asarray(times)
    require(
        states.ndim == 2
        and states.shape[1] >= 13
        and len(states) >= 1
        and times.shape == (len(states),)
        and np.isfinite(states).all()
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0)
        and np.all(np.linalg.norm(states[:, 6:10], axis=1) >= 1e-9),
        "contact trajectory contract",
    )
    if len(states) == 1:
        return {"hit": False, "contact": False, "reason": "no_executed_interval"}
    normal = np.asarray(target["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    center = np.asarray(target["center_m"], dtype=np.float64)

    def distance(state):
        return float((kinematics(state, target)[0] - center) @ normal)

    distances = np.array([distance(state) for state in states])
    if distances[0] <= 0:
        return {
            "hit": False,
            "contact": False,
            "reason": "initial_contact_point_not_in_front",
        }
    indices = np.flatnonzero((distances[:-1] > 0) & (distances[1:] <= 0))
    if not len(indices):
        return {
            "hit": False,
            "contact": False,
            "reason": "no_plane_crossing",
            "minimum_plane_distance_m": float(distances.min()),
        }
    i = int(indices[0])
    lo, hi = 0.0, 1.0
    for _ in range(28):
        mid = (lo + hi) / 2
        if distance(interpolate(states[i], states[i + 1], mid)) > 0:
            lo = mid
        else:
            hi = mid
    fraction = (lo + hi) / 2
    state = interpolate(states[i], states[i + 1], fraction)
    point, axis, velocity = kinematics(state, target)
    error = point - center
    miss = float(np.linalg.norm(error - normal * (error @ normal)))
    angle = float(np.degrees(np.arccos(np.clip(-axis @ normal, -1, 1))))
    speed = float(-velocity @ normal)
    tangent = float(np.linalg.norm(velocity + speed * normal))
    checks = {
        "bullseye": miss <= target["radius_m_max"],
        "axis_alignment": angle <= target["axis_error_deg_max"],
        "normal_speed": target["normal_speed_m_s_range"][0]
        <= speed
        <= target["normal_speed_m_s_range"][1],
        "tangent_speed": tangent <= target["tangential_speed_m_s_max"],
    }
    return {
        "hit": bool(all(checks.values())),
        "contact": True,
        "checks": checks,
        "time_s": float(times[i] + fraction * (times[i + 1] - times[i])),
        "interval": i,
        "fraction": fraction,
        "miss_distance_m": miss,
        "axis_error_deg": angle,
        "normal_speed_m_s": speed,
        "tangent_speed_m_s": tangent,
        "contact_position_m": point.tolist(),
        "contact_state": state[:13].tolist(),
    }


def compare_contact(actual, expected, protocol):
    require(set(actual) == set(expected), "contact field roster")
    budget = protocol["contact_comparison"]
    for key, wanted in expected.items():
        if key in budget["exact"]:
            require(actual[key] == wanted, f"contact {key}")
            continue
        if key == "axis_error_deg":
            atol = budget["angle_atol_deg"]
        elif key == "fraction":
            atol = budget["fraction_atol"]
        elif key == "time_s":
            atol = budget["time_atol_s"]
        elif key.endswith("speed_m_s"):
            atol = budget["speed_atol_m_s"]
        elif key == "contact_state":
            atol = budget["state_atol"]
        else:
            atol = budget["length_atol_m"]
        require(
            np.allclose(
                actual[key], wanted, rtol=budget["rtol"], atol=atol, equal_nan=False
            ),
            f"contact {key}",
        )


def flight_replay(root, models):
    import jax

    spec = read(root / "flights/spec.json")
    require(
        spec["queries"] == 8064 and spec["prediction_arrays"] == 12768,
        "frozen flight counts differ",
    )
    runtime_check(spec["runtime"])
    kernels = {name: jax.jit(model.predict) for name, model in models.items()}
    total_arrays, actual_calls, total_queries = 0, 0, 0
    cohorts = {}
    seen = set()
    for cohort, contract in spec["cohorts"].items():
        roster = read(root / "flights" / f"{cohort}.json")
        data = arrays(root / "flights" / f"{cohort}.npz")
        require(len(roster) == contract["queries"], "cohort roster count")
        responses = [i for i, q in enumerate(roster) if q["kind"] == "response"]
        exact(
            data["response_indices"],
            np.asarray(responses, dtype=np.int64),
            "response array association",
        )
        require(len(responses) == contract["response_arrays"], "response count")
        require(
            sum(not q["history_eligible"] for q in roster)
            == contract["history_ineligible"],
            "history eligibility count",
        )
        require(
            data["predicted"].shape[0] == len(roster)
            and data["predicted_factual"].shape[0] == len(responses),
            "saved prediction count",
        )
        response_index = {row: index for index, row in enumerate(responses)}
        count, called, missing = 0, 0, 0
        for i, q in enumerate(roster):
            require(q["uid"] not in seen, "duplicate query")
            seen.add(q["uid"])
            require(
                q["cohort"] == cohort and q["simulator"] == contract["simulator"],
                "query cohort identity",
            )
            for suffix in ["", "_factual"] if q["kind"] == "response" else [""]:
                command = data["factual_inputs" if suffix else "future_inputs"][i]
                bad = np.flatnonzero(~np.isfinite(command).all(axis=1))
                length = (
                    (int(bad[0]) if len(bad) else len(command))
                    if q["history_eligible"]
                    else 0
                )
                expected = (
                    data["predicted_factual"][response_index[i]]
                    if suffix
                    else data["predicted"][i]
                )
                result = np.full(data["target"][i].shape, np.nan, dtype=np.float32)
                outcome = q["outcomes"][suffix]
                require(
                    outcome["called"] == bool(length) and outcome["error"] is None,
                    "saved eligibility/error differs",
                )
                if length:
                    result[:length] = np.asarray(
                        kernels[q["simulator"]](
                            data["past_states"][i],
                            data["past_inputs"][i],
                            command[:length],
                        )
                    )
                    called += 1
                else:
                    missing += 1
                exact(result, expected, q["uid"] + suffix)
                count += 1
        cohorts[cohort] = dict(
            queries=len(roster),
            arrays_exact=count,
            prediction_calls=called,
            ineligible_arrays=missing,
        )
        total_queries += len(roster)
        total_arrays += count
        actual_calls += called
        print(json.dumps({"cohort": cohort, **cohorts[cohort]}), flush=True)
    require(total_queries == 8064 and total_arrays == 12768, "flight closure count")
    return dict(
        queries=total_queries,
        arrays_exact=total_arrays,
        prediction_calls=actual_calls,
        cohorts=cohorts,
    )


def dart_arrays(root):
    """Check causality and trajectory evidence without any scientific calls."""
    d = {
        name: arrays(root / "dart" / f"{name}.npz")
        for name in (
            "initials",
            "gradient",
            "selected",
            "native",
            "trajectory",
            "prelude",
        )
    }
    spec = read(root / "dart/spec.json")
    require(
        spec["counts"] == dict(gradient=4144, selected=40, native=120),
        "Dart frozen counts differ",
    )
    for kind, count in spec["counts"].items():
        require(
            all(value.shape[0] == count for value in d[kind].values()),
            f"{kind} callback count",
        )
    require(
        all(value.shape[0] == 40 for value in d["initials"].values()),
        "initial history count",
    )
    calls = np.concatenate([d[k]["call"] for k in ("gradient", "selected", "native")])
    exact(np.sort(calls), np.arange(1, 4305, dtype=np.int64), "call roster")
    exact(
        d["selected"]["origin"],
        np.arange(0, 120, 3, dtype=np.int64),
        "selected origins",
    )
    trajectory, prelude = d["trajectory"], d["prelude"]
    require(
        trajectory["states"].shape == (121, 17)
        and trajectory["commands"].shape == (120, 4)
        and prelude["states"].shape == (51, 17)
        and prelude["commands"].shape == (50, 4),
        "trajectory shape",
    )
    exact(prelude["states"][-1], trajectory["states"][0], "prelude continuity")
    exact(d["native"]["inputs_state"], trajectory["states"][:-1], "native inputs")
    exact(d["native"]["result_state"], trajectory["states"][1:], "native outputs")
    exact(
        d["native"]["inputs_command"],
        trajectory["commands"].astype(np.float32),
        "native executed commands",
    )
    exact(trajectory["time_s"], np.arange(121) * 0.01, "native time grid")
    for index, origin in enumerate(d["selected"]["origin"]):
        exact(
            d["selected"]["inputs_commands"][index, :3],
            trajectory["commands"][origin : origin + 3].astype(np.float32),
            "selected/executed command prefix",
        )
    states = np.concatenate((prelude["states"], trajectory["states"][1:]))
    commands = np.concatenate((prelude["commands"], trajectory["commands"]))
    for index, origin in enumerate(range(0, 120, 3)):
        exact(d["initials"]["state"][index], states[origin + 50, :13], "causal state")
        exact(
            d["initials"]["past_inputs"][index],
            commands[origin : origin + 50].astype(np.float32),
            "causal issued history",
        )
        expected = observed(states[origin : origin + 51, :13]).astype(np.float32)
        actual = d["initials"]["past_states"][index]
        exact(actual[:, :6], expected[:, :6], "causal observed motion")
        require(
            np.allclose(
                actual[:, 6:],
                expected[:, 6:],
                rtol=8 * np.finfo(np.float32).eps,
                atol=8 * np.finfo(np.float32).eps,
            ),
            "independent canonical rotation roundoff",
        )
    exact(
        d["gradient"]["inputs_commands"],
        np.repeat(d["gradient"]["inputs_flat"].reshape(4144, 40, 4), 3, axis=1),
        "gradient command expansion",
    )
    exact(
        d["gradient"]["inputs_active_steps"],
        (120 - d["gradient"]["origin"]).astype(np.int32),
        "active objective steps",
    )
    require(
        all(
            np.isfinite(value).all()
            for kind in ("gradient", "selected", "native")
            for key, value in d[kind].items()
            if key.startswith("result_")
        ),
        "nonfinite original returned values",
    )
    score = contact_score(trajectory["states"], trajectory["time_s"], spec["target"])
    qualification = read(root / "provenance/dart-qualification-qualification.json")
    require(
        qualification["diagnostic_evidence_qualified"] is True
        and qualification["readout"]["task_success"] is True,
        "original qualification differs",
    )
    compare_contact(score, qualification["readout"]["contact"], spec)
    return d, spec, score


def dart_replay(root, model, dart_root=None):
    import jax.numpy as jnp

    from glassbox.core.geometry import motion_rollout

    data, spec, score = dart_arrays(root)
    runtime_check(spec["runtime"])

    def rollout(initial, commands):
        return motion_rollout(model, *initial, commands)

    def initial(origin):
        require(origin % 3 == 0 and 0 <= origin < 120, "callback origin")
        index = origin // 3
        return tuple(
            jnp.asarray(data["initials"][key][index])
            for key in ("state", "past_states", "past_inputs")
        )

    for i, origin in enumerate(data["selected"]["origin"]):
        exact(
            np.asarray(
                rollout(
                    initial(origin), jnp.asarray(data["selected"]["inputs_commands"][i])
                )
            ),
            data["selected"]["result_states"][i],
            f"selected mean {origin}",
        )
    result = dict(
        selected_means_exact=40,
        gradients_exact=0,
        optimizer_calls=0,
        native_calls=0,
        contact=score,
        objective_replay="not requested; supply --dart-root",
    )
    if dart_root is None:
        return result
    dart_root = Path(dart_root).resolve()
    for relative, expected in spec["source_sha256"].items():
        require(
            digest(dart_root / relative) == expected, f"Dart source differs: {relative}"
        )
    sys.path.insert(0, str(dart_root / "src"))
    mission = importlib.import_module("crazydart.mission")
    planning = importlib.import_module("crazydart.planning")
    for module in (mission, planning):
        relative = "src/" + module.__name__.replace(".", "/") + ".py"
        require(
            Path(module.__file__).resolve() == (dart_root / relative).resolve(),
            "Dart imported from a different tree",
        )
    t = spec["target"]
    target = mission.Target(
        center=tuple(t["center_m"]),
        normal=tuple(t["normal"]),
        contact_offset=tuple(t["body_contact_offset_m"]),
        radius_m=t["radius_m_max"],
        angle_tolerance_deg=t["axis_error_deg_max"],
        maximum_tangent_speed_m_s=t["tangential_speed_m_s_max"],
        minimum_normal_speed_m_s=t["normal_speed_m_s_range"][0],
        maximum_normal_speed_m_s=t["normal_speed_m_s_range"][1],
    )
    c = spec["controller"]
    planner = planning.DirectPlanner(
        rollout,
        target,
        steps=c["steps"],
        block_steps=c["block_steps"],
        minimum=c["minimum"],
        maximum=c["maximum"],
    )
    gradients = data["gradient"]
    for i, origin in enumerate(gradients["origin"]):
        value, grad = planner.value_gradient(
            jnp.asarray(gradients["inputs_flat"][i]),
            initial(origin),
            jnp.asarray(gradients["inputs_active_steps"][i]),
        )
        exact(value, gradients["result_objective"][i], f"objective {i + 1}")
        exact(grad, gradients["result_gradient"][i], f"gradient {i + 1}")
        if (i + 1) % 1024 == 0:
            print(json.dumps({"Dart gradients exact": i + 1}), flush=True)
    canonical = mission.score_contact(
        data["trajectory"]["states"], data["trajectory"]["time_s"], target
    )
    require(
        canonical == read(root / "dart/trial.json")["contact"],
        "canonical Dart contact differs",
    )
    result.update(
        gradients_exact=4144,
        objective_replay="exact unchanged external objective",
        canonical_contact_exact=True,
    )
    return result


def verify(root, manifest_sha256, dart_root=None):
    root = Path(root).resolve()
    manifest = verify_integrity(root, manifest_sha256)
    models, identities = verify_models(root)
    flights = flight_replay(root, models)
    dart = dart_replay(root, models["dart"], dart_root)
    return dict(
        format="glassbox-baseline-replay-v1",
        manifest_sha256=manifest_sha256,
        files_verified=len(manifest["files"]),
        models=identities,
        flights=flights,
        dart=dart,
        fits=0,
        optimizer_calls=0,
        native_calls=0,
        complete=dart["gradients_exact"] == 4144,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--dart-root", type=Path)
    args = parser.parse_args()
    authority = args.manifest_sha256
    if authority is None:
        reference = Path(__file__).resolve().parents[1] / "docs/baseline.json"
        authority = read(reference)["manifest_sha256"]
    print(
        json.dumps(
            verify(args.baseline, authority, args.dart_root),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
