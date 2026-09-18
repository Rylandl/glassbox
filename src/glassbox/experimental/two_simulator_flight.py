"""Frozen two-simulator baseline; truth generation is separate from model fitting.

Run with the pinned Dart interpreter, SCIPY_ARRAY_API=1, JAX_ENABLE_X64=1,
and PYTHONPATH containing this checkout and the clean Cascade archive. Every
stage writes an immutable seal. Replay regenerates physics and model forecasts,
without fitting. No simulator state reaches the generic fit/predict boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.data import Trajectory, TrajectorySpec
from glassbox.experimental.learned_plan import OBSERVED_CHANNELS, observed_from_state
from glassbox.recordings import SequenceCollection, SequenceSegment

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/two-simulator-flight-v1.json"
OBSERVE = jax.jit(jax.vmap(observed_from_state))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def save_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def array_equal(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label}: array roster changed")
    for name, value in actual.items():
        wanted = expected[name]
        if (
            value.shape != wanted.shape
            or value.dtype != wanted.dtype
            or value.tobytes() != wanted.tobytes()
        ):
            raise ValueError(f"{label}/{name}: fresh array bytes differ")


def freeze_files(directory, name, details):
    directory = Path(directory)
    path = directory / name
    if path.exists():
        raise FileExistsError(path)
    files = {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    write_json(path, dict(details, files=files))
    return digest(path)


def verify_files(directory, name):
    directory = Path(directory)
    seal = read_json(directory / name)
    for relative, expected in seal["files"].items():
        if digest(directory / relative) != expected:
            raise ValueError(f"altered artifact: {relative}")
    return seal


def check_sources(protocol):
    sources_path = ROOT / protocol["sources"]["path"]
    if digest(sources_path) != protocol["sources"]["sha256"]:
        raise ValueError("source inventory changed")
    pins = read_json(sources_path)
    for relative, expected in pins["glassbox_source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"base source changed: {relative}")
    for relative, expected in pins["cascade"]["all_files"].items():
        if digest(Path(pins["cascade"]["unpacked_root"]) / relative) != expected:
            raise ValueError(f"Cascade source changed: {relative}")
    cf = pins["crazyflow"]
    for relative, record in cf["official_source_roster"]["files"].items():
        expected = record["sha256"] if isinstance(record, dict) else record
        # The frozen inventory uses paths relative to the official checkout.
        if digest(Path(cf["source_collection"]["path"]) / relative) != expected:
            raise ValueError(f"Crazyflow source changed: {relative}")
        if digest(Path(cf["installed_package"]["path"]).parent / relative) != expected:
            raise ValueError(f"installed Crazyflow source changed: {relative}")
    for name, root in (
        ("crazyflow", Path(cf["installed_package"]["path"])),
        ("cascade", Path(pins["cascade"]["unpacked_root"]) / "src/cascade"),
    ):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != root.resolve():
            raise ValueError(f"{name} imported from an unpinned location")
    if not jax.config.x64_enabled or jax.default_backend() != "cpu":
        raise ValueError("benchmark requires float64 CPU JAX")
    runtime = protocol["runtime"]
    if os.environ.get("SCIPY_ARRAY_API") != runtime["SCIPY_ARRAY_API"]:
        raise ValueError("SCIPY_ARRAY_API differs from frozen runtime")
    if platform.python_version() != runtime["python_version"]:
        raise ValueError("Python runtime differs from frozen version")
    versions = {
        n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "numpy", "scipy")
    }
    if any(versions[n] != runtime[n] for n in versions):
        raise ValueError("numeric runtime differs from frozen versions")
    return {
        "versions": versions,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "x64": True,
        "SCIPY_ARRAY_API": os.environ["SCIPY_ARRAY_API"],
        "protocol_sha256": digest(PROTOCOL),
        "sources_sha256": digest(sources_path),
        "implementation": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted(
                (ROOT / "src/glassbox/experimental").glob("two_simulator_*.py")
            )
        },
    }


def fixture_for(simulator, protocol):
    return importlib.import_module(
        f"glassbox.experimental.two_simulator_{simulator}"
    ).Fixture(protocol)


def validity(arrays):
    """Count complete transitions before the first invalid observed boundary."""
    x, hidden, u = arrays["states"], arrays["full_states"], arrays["commands"]
    for i in range(len(x)):
        reason = None
        if not np.isfinite(hidden[i]).all() or not np.isfinite(x[i]).all():
            reason = "nonfinite_state"
        elif x[i, 2] < 0.5:
            reason = "altitude_below_0.5m"
        elif abs(np.linalg.norm(x[i, 6:10]) - 1) > 1e-6:
            reason = "quaternion_norm"
        if reason:
            return {
                "valid_transitions": max(0, i - 1),
                "valid_initial": i > 0,
                "failure_index": i,
                "reason": reason,
            }
        if i < len(u) and not np.isfinite(u[i]).all():
            return {
                "valid_transitions": i,
                "valid_initial": True,
                "failure_index": i,
                "reason": "nonfinite_command",
            }
    return {
        "valid_transitions": len(u),
        "valid_initial": True,
        "failure_index": None,
        "reason": None,
    }


def achieved(arrays):
    """Describe valid conditions separately from finite post-failure diagnostics.

    Wind is paired by original row index; missing/nonfinite wind supplies no
    airspeed evidence. Attitude uses normalized WXYZ quaternions, while the
    frozen validity rule still excludes nonunit states from the valid prefix.
    """
    states = np.asarray(arrays["states"], dtype=np.float64)
    hidden = np.asarray(arrays["full_states"])
    wind = np.asarray(arrays.get("wind", np.full((len(states), 3), np.nan)))
    if wind.shape != (len(states), 3):
        raise ValueError("achieved conditions need state-aligned three-column wind")
    status = validity(arrays)
    prefix_count = status["valid_transitions"] + 1 if status["valid_initial"] else 0

    def extent(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        return [float(values.min()), float(values.max())] if values.size else None

    def scope(indices):
        finite = np.isfinite(states[indices]).all(axis=1)
        rows = indices[finite]
        x, aligned_wind = states[rows], wind[rows]
        finite_wind = np.isfinite(aligned_wind).all(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            ground_speed = np.hypot.reduce(x[:, 3:6], axis=1)
            air_velocity = x[:, 3:6] - aligned_wind
            airspeed = np.hypot.reduce(air_velocity[finite_wind], axis=1)
            rate_norm = np.hypot.reduce(x[:, 10:13], axis=1)
            qnorm = np.hypot.reduce(x[:, 6:10], axis=1)
        attitude_valid = np.isfinite(qnorm) & (qnorm > 0)
        w, qx, qy, qz = (x[attitude_valid, 6:10] / qnorm[attitude_valid, None]).T
        # Columns are body axes in world NWU. Only the entries needed below.
        r00, r10, r20 = (
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy + w * qz),
            2 * (qx * qz - w * qy),
        )
        r21, r22 = 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)
        horizontal = np.hypot(r00, r10)
        heading_bank_valid = horizontal > 1e-12
        heading = np.degrees(
            np.arctan2(r10[heading_bank_valid], r00[heading_bank_valid])
        )
        heading = (heading + 180) % 360 - 180
        pitch = np.degrees(np.arctan2(r20, horizontal))
        bank = np.degrees(np.arctan2(r21[heading_bank_valid], r22[heading_bank_valid]))
        forward = np.column_stack((r00, r10, r20))
        with np.errstate(over="ignore", invalid="ignore"):
            body_forward = np.sum(air_velocity[attitude_valid] * forward, axis=1)
        body_forward = body_forward[finite_wind[attitude_valid]]
        circular_arc = None
        if heading.size:
            angles = np.sort(heading % 360)
            gaps = np.diff(np.r_[angles, angles[0] + 360])
            gap = int(np.argmax(gaps))
            start = float((angles[(gap + 1) % len(angles)] + 180) % 360 - 180)
            width = float(360 - gaps[gap])
            circular_arc = {"start_deg": start, "width_deg": width}
        return {
            "requested_boundaries": len(indices),
            "finite_boundaries": len(rows),
            "finite_full_state_boundaries": int(
                np.isfinite(hidden[rows]).all(axis=1).sum()
            ),
            "finite_wind_boundaries": int(finite_wind.sum()),
            "attitude_boundaries": int(attitude_valid.sum()),
            "heading_bank_boundaries": int(heading_bank_valid.sum()),
            "ground_speed_boundaries": int(np.isfinite(ground_speed).sum()),
            "airspeed_boundaries": int(np.isfinite(airspeed).sum()),
            "body_forward_airspeed_boundaries": int(np.isfinite(body_forward).sum()),
            "ground_speed_m_s": extent(ground_speed),
            "airspeed_m_s": extent(airspeed),
            "body_forward_airspeed_m_s": extent(body_forward),
            "heading_wrapped_deg": extent(heading),
            "heading_circular_arc": circular_arc,
            "pitch_deg": extent(pitch),
            "bank_deg": extent(bank),
            "body_rate_rad_s": {
                axis: extent(x[:, 10 + index]) for index, axis in enumerate("xyz")
            },
            "max_body_rate_rad_s": extent(rate_norm)[1] if extent(rate_norm) else None,
            "altitude_m": extent(x[:, 2]),
            "climb_m_s": extent(x[:, 5]),
        }

    return {
        "validity": status,
        "angle_convention": "Heading: body-forward NWU yaw in [-180,180); circular arc runs counterclockwise from start through width. Pitch: body-forward elevation, nose-up positive. Bank: FLU atan2(R21,R22). Heading/bank omitted when forward horizontal projection <=1e-12.",
        "diagnostic_scope": "All finite canonical rows, including hidden-state failure and post-failure rows; this is not valid operating-condition coverage.",
        "valid_prefix": scope(np.arange(prefix_count)),
        "all_finite_diagnostic": scope(np.arange(len(states))),
    }


def roles_for(names):
    names = sorted(
        names,
        key=lambda s: hashlib.sha256(
            json.dumps(s, sort_keys=True).encode()
        ).hexdigest(),
    )
    if len(names) < 2:
        return {"development": [], "training": names}
    n = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    return {"development": names[:n], "training": names[n:]}


def query_roster(entry, protocol, dt, width):
    queries = [
        {
            "id": f"factual-{round(t / dt):04d}",
            "kind": "factual",
            "origin": round(t / dt),
        }
        for t in protocol["evaluation"]["factual_origins_s"]
    ]
    for t in protocol["evaluation"]["responses"]["origins_s"]:
        origin = round(t / dt)
        for channel in range(width):
            for sign in (-1, 1):
                queries.append(
                    {
                        "id": f"response-{origin:04d}-{channel}-{sign:+d}",
                        "kind": "response",
                        "origin": origin,
                        "channel": channel,
                        "sign": sign,
                    }
                )
    return [
        dict(q, parent=entry["id"], cell=entry["cell"], simulator=entry["simulator"])
        for q in queries
    ]


def make_queries(fixture, parent, entry, cell, protocol):
    """Persist every planned slot, including failures, before fitting."""
    dt, width = fixture.dt, len(fixture.lower)
    history, horizon = round(0.5 / dt), round(0.25 / dt)
    queries = query_roster(entry, protocol, dt, width)
    output, truth_branches = {}, {}
    parent_valid = (
        validity(parent)
        if parent is not None
        else {"valid_initial": False, "valid_transitions": 0}
    )
    observed = np.asarray(OBSERVE(parent["states"])) if parent is not None else None
    base_branches = {}
    for query in queries:
        origin, qid = query["origin"], query["id"]
        q = {
            "past_states": np.full((history + 1, 15), np.nan),
            "past_inputs": np.full((history, width), np.nan),
            "future_inputs": np.full((horizon, width), np.nan),
            "factual_inputs": np.full((horizon, width), np.nan),
            "target": np.full((horizon, 15), np.nan),
            "factual_target": np.full((horizon, 15), np.nan),
            "valid": np.zeros(horizon, dtype=bool),
        }
        has_history = (
            parent_valid["valid_initial"]
            and origin <= parent_valid["valid_transitions"]
        )
        query["history_eligible"] = bool(has_history)
        if has_history:
            q["past_states"] = observed[origin - history : origin + 1].copy()
            q["past_inputs"] = parent["commands"][origin - history : origin].copy()
            q["factual_inputs"] = parent["commands"][origin : origin + horizon].copy()
            q["future_inputs"] = q["factual_inputs"].copy()
            q["factual_target"] = observed[origin + 1 : origin + horizon + 1].copy()
            q["target"] = q["factual_target"].copy()
            q["valid"] = (
                np.arange(1, horizon + 1) + origin <= parent_valid["valid_transitions"]
            )
            if query["kind"] == "response":
                bad = np.flatnonzero(~np.isfinite(q["factual_inputs"]).all(axis=1))
                length = int(bad[0]) if len(bad) else horizon
                if length and origin not in base_branches:
                    base, detail = fixture.branch(
                        parent["full_states"][origin],
                        q["factual_inputs"][:length],
                        origin,
                        cell,
                    )
                    for key in ("states", "full_states"):
                        array_equal(
                            {key: base[key]},
                            {key: parent[key][origin : origin + length + 1]},
                            f"{entry['id']}/{origin}/factual-clone",
                        )
                    base_branches[origin] = base
                    truth_branches[f"base-{origin:04d}"] = (base, detail)
                channel, sign = query["channel"], query["sign"]
                bound = fixture.lower[channel] if sign < 0 else fixture.upper[channel]
                q["future_inputs"][:, channel] += 0.1 * (
                    bound - q["future_inputs"][:, channel]
                )
                q["target"] = np.full((horizon, 15), np.nan)
                if length:
                    branch, detail = fixture.branch(
                        parent["full_states"][origin],
                        q["future_inputs"][:length],
                        origin,
                        cell,
                    )
                    truth_branches[qid] = (branch, detail)
                    q["target"][:length] = np.asarray(OBSERVE(branch["states"]))[1:]
                    branch_valid = validity(branch)
                    q["valid"] &= branch_valid["valid_initial"] & (
                        np.arange(1, horizon + 1) <= branch_valid["valid_transitions"]
                    )
                else:
                    q["valid"][:] = False
            q["command_delta"] = q["future_inputs"] - q["factual_inputs"]
        else:
            q["command_delta"] = np.full((horizon, width), np.nan)
        output[qid] = q
    return queries, output, truth_branches


def generate(simulator, output):
    protocol = read_json(PROTOCOL)
    runtime = check_sources(protocol)
    fixture = fixture_for(simulator, protocol)
    directory = Path(output) / simulator / "data"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "configuration.json",
        dict(fixture.config(), spec=fixture.spec.to_dict()),
    )
    cells = {c["id"]: c for c in protocol["cells"][simulator]}
    records, all_queries = [], []
    start = time.perf_counter()
    for entry in protocol["recordings"]:
        if entry["simulator"] != simulator:
            continue
        cell = cells[entry["cell"]]
        prefix = Path("parents") / entry["id"]
        print(f"generate {entry['id']}", flush=True)
        try:
            arrays, metadata = fixture.generate(entry, cell)
        except Exception as error:
            if type(error).__name__ != "FixtureSetupError":
                raise
            arrays = None
            metadata = {"setup_failure": str(error), "exception": type(error).__name__}
        record = dict(entry, cell_facts=cell, prefix=str(prefix), metadata=metadata)
        if arrays is not None:
            save_arrays(directory / prefix.with_suffix(".npz"), arrays)
            record.update(validity=validity(arrays), achieved=achieved(arrays))
        else:
            record["validity"] = {
                "valid_transitions": 0,
                "valid_initial": False,
                "failure_index": 0,
                "reason": "setup_failure",
            }
        write_json(directory / prefix.with_suffix(".json"), record)
        records.append(record)
        if entry["role"] == "test":
            queries, query_arrays, branches = make_queries(
                fixture, arrays, entry, cell, protocol
            )
            for query in queries:
                query["scope"] = cell["group"]
                query["path"] = str(
                    Path("queries") / entry["id"] / (query["id"] + ".npz")
                )
                save_arrays(directory / query["path"], query_arrays[query["id"]])
            for name, (values, detail) in branches.items():
                path = Path("branches") / entry["id"] / name
                save_arrays(directory / path.with_suffix(".npz"), values)
                write_json(directory / path.with_suffix(".json"), detail)
            all_queries.extend(queries)
    minimum = round(0.75 / fixture.dt)
    admitted = [
        r["id"]
        for r in records
        if r["role"] == "calibration_pool"
        and r["validity"]["valid_initial"]
        and r["validity"]["valid_transitions"] >= minimum
    ]
    roles = roles_for(admitted)
    write_json(directory / "records.json", records)
    write_json(directory / "queries.json", all_queries)
    write_json(directory / "roles.json", roles)
    seal = freeze_files(
        directory,
        "seal.json",
        {
            "runtime": runtime,
            "simulator": simulator,
            "wall_time_s": time.perf_counter() - start,
            "planned_records": len(records),
            "admitted": admitted,
            "roles": roles,
            "query_count": len(all_queries),
        },
    )
    print(f"sealed {simulator}: {seal}", flush=True)


def collection_and_trajectories(directory):
    seal = verify_files(directory, "seal.json")
    config = read_json(directory / "configuration.json")
    spec = TrajectorySpec.from_dict(config["spec"])
    records = read_json(directory / "records.json")
    trajectories, segments = [], []
    for record in records:
        if record["id"] not in seal["admitted"]:
            continue
        arrays = load_arrays(directory / (record["prefix"] + ".npz"))
        n = record["validity"]["valid_transitions"]
        dt = float(arrays["time_s"][1] - arrays["time_s"][0])
        observed = np.asarray(OBSERVE(arrays["states"][: n + 1]))
        segments.append(
            SequenceSegment(
                record["id"], "valid-prefix", observed, arrays["commands"][:n], dt
            )
        )
        role = (
            "development"
            if record["id"] in seal["roles"]["development"]
            else "training"
        )
        trajectories.append(
            Trajectory(
                time_s=arrays["time_s"][: n + 1],
                states=arrays["states"][: n + 1],
                controls=arrays["commands"][:n],
                spec=spec,
                control_prefix=arrays["control_prefix"],
                labels={"source_group": record["id"], "generic_role": role},
                provenance={
                    "source_flight_id": record["id"],
                    "prefix": "declared equilibrium initialization prior",
                },
            )
        )
    collection = SequenceCollection(
        tuple(segments),
        configuration_id=spec.vehicle.configuration_id,
        state_channels=OBSERVED_CHANNELS,
        input_channels=tuple(
            json.dumps(c.to_dict(), sort_keys=True) for c in spec.controls
        ),
    )
    return collection, trajectories


def fit_arm(simulator, arm, output):
    from glassbox import fit as generic_fit
    from glassbox.fitting import FitSpec, Holdout
    from glassbox.fitting import fit as structured_fit
    from glassbox.learner import RECIPE

    protocol = read_json(PROTOCOL)
    runtime = check_sources(protocol)
    # Both full simulator datasets must be sealed before the first fit.
    for name in protocol["cells"]:
        verify_files(Path(output) / name / "data", "seal.json")
    directory = Path(output) / simulator / arm
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "start.json",
        {
            "runtime": runtime,
            "arm": arm,
            "simulator": simulator,
            "data_seal": digest(Path(output) / simulator / "data/seal.json"),
        },
    )
    data_seal = read_json(Path(output) / simulator / "data/seal.json")
    start = time.perf_counter()
    if len(data_seal["admitted"]) < 2:
        write_json(
            directory / "outcome.json",
            {
                "status": "fit_failure",
                "reason": "fewer_than_two_admitted_recordings",
                "model_available": False,
            },
        )
        freeze_files(
            directory, "seal.json", {"wall_time_s": time.perf_counter() - start}
        )
        return
    collection, trajectories = collection_and_trajectories(
        Path(output) / simulator / "data"
    )
    print(f"fit {simulator}/{arm}", flush=True)
    if arm == "generic":
        if RECIPE != protocol["fitting"]["generic_recipe"]:
            raise ValueError("generic recipe differs from protocol")
        model = generic_fit(collection)
        model.save(directory / "model.npz")
        report = model.report
        write_json(directory / "contract.json", model.contract)
    else:
        spec = FitSpec(
            holdout=Holdout.by_label("generic_role", ("development",)),
            horizons_s=(0.05, 0.15, 0.25),
            evaluation_horizons_s=(0.05, 0.15, 0.25),
            steps=600,
            parameter_evidence=False,
        )
        if (
            json.loads(json.dumps(asdict(spec)))
            != protocol["fitting"]["structured_spec"]
        ):
            raise ValueError("structured fit spec differs from protocol")
        result = structured_fit(trajectories, spec)
        result.belief.save(directory / "model.json")
        report = result.report
    write_json(directory / "report.json", report)
    write_json(
        directory / "outcome.json", {"status": "complete", "model_available": True}
    )
    freeze_files(directory, "seal.json", {"wall_time_s": time.perf_counter() - start})
    print(f"fit complete {simulator}/{arm}", flush=True)


class Predictors:
    def __init__(self, root, simulator):
        from glassbox.belief.belief import DynamicsBelief
        from glassbox.learner import LearnedDynamics

        self.root = Path(root) / simulator
        for arm in ("generic", "structured"):
            verify_files(self.root / arm, "seal.json")
        self.generic = (
            LearnedDynamics.load(self.root / "generic/model.npz")
            if read_json(self.root / "generic/outcome.json")["model_available"]
            else None
        )
        self.structured = (
            DynamicsBelief.load(self.root / "structured/model.json").model
            if read_json(self.root / "structured/outcome.json")["model_available"]
            else None
        )
        self.gpredict = (
            jax.jit(self.generic.predict) if self.generic is not None else None
        )
        model = self.structured

        @jax.jit
        def latents(states, commands, prefix):
            initial = model.initial_latent_state(prefix)

            def step(latent, row):
                _, following = model.transition(row[0], latent, row[1])
                return following, following

            _, rest = jax.lax.scan(step, initial, (states[:-1], commands))
            return jnp.concatenate((initial[None], rest))

        self.latents = latents

        @jax.jit
        def spredict(state, latent, commands):
            def step(carry, command):
                following = model.transition(*carry, command)
                return following, observed_from_state(following[0])

            return jax.lax.scan(step, (state, latent), commands)[1]

        self.spredict = spredict
        self.parent_cache = {}

    def predict(self, query, arrays, record):
        h = len(arrays["future_inputs"])
        result = {
            arm: np.full((h, 15), np.nan) for arm in ("generic", "structured", "hold")
        }
        if not query["history_eligible"]:
            return result
        parent_id = record["id"]
        if parent_id not in self.parent_cache:
            parent = load_arrays(self.root / "data" / (record["prefix"] + ".npz"))
            n = record["validity"]["valid_transitions"]
            latent = (
                np.asarray(
                    self.latents(
                        parent["states"][: n + 1],
                        parent["commands"][:n],
                        parent["control_prefix"],
                    )
                )
                if self.structured is not None
                else None
            )
            self.parent_cache[parent_id] = (parent, latent)
        parent, latent = self.parent_cache[parent_id]
        finite = np.isfinite(arrays["future_inputs"]).all(axis=1)
        bad = np.flatnonzero(~finite)
        length = int(bad[0]) if len(bad) else h
        if not length:
            return result
        commands = arrays["future_inputs"][:length]
        if self.generic is not None:
            result["generic"][:length] = np.asarray(
                self.gpredict(arrays["past_states"], arrays["past_inputs"], commands)
            )
        if self.structured is not None:
            result["structured"][:length] = np.asarray(
                self.spredict(
                    parent["states"][query["origin"]], latent[query["origin"]], commands
                )
            )
        result["hold"][:length] = arrays["past_states"][-1]
        return result


def evaluate(simulator, output, *, replay=False):
    from glassbox.experimental.two_simulator_metrics import (
        aggregate,
        score_forecast,
        score_response_directions,
    )

    protocol = read_json(PROTOCOL)
    check_sources(protocol)
    base = Path(output) / simulator
    verify_files(base / "data", "seal.json")
    directory = base / "evaluation"
    if not replay:
        directory.mkdir(parents=True, exist_ok=False)
    else:
        verify_files(directory, "seal.json")
    check_links(output, simulator, evaluation=replay)
    predictors = Predictors(output, simulator)
    records = {r["id"]: r for r in read_json(base / "data/records.json")}
    rows, caches, directions, lower_pairs = [], {}, [], {}
    start = time.perf_counter()
    dt = protocol["generation"][simulator]["dt_s"]
    for query in read_json(base / "data/queries.json"):
        arrays = load_arrays(base / "data" / query["path"])
        prediction = predictors.predict(query, arrays, records[query["parent"]])
        result = dict(prediction)
        target = arrays["target"]
        if query["kind"] == "response":
            key = (query["parent"], query["origin"])
            if key not in caches:
                factual_arrays = dict(arrays, future_inputs=arrays["factual_inputs"])
                caches[key] = predictors.predict(
                    query, factual_arrays, records[query["parent"]]
                )
            for arm in prediction:
                result[arm + "_factual"] = caches[key][arm]
                prediction[arm] = prediction[arm] - caches[key][arm]
                result[arm + "_response"] = prediction[arm]
            target = target - arrays["factual_target"]
            pair_key = (query["parent"], query["origin"], query["channel"])
            if query["sign"] < 0:
                lower_pairs[pair_key] = (prediction, target, arrays["valid"])
            else:
                lower_prediction, lower_target, lower_valid = lower_pairs.pop(pair_key)
                for arm in prediction:
                    scored = score_response_directions(
                        np.stack((lower_prediction[arm], prediction[arm])),
                        np.stack((lower_target, target)),
                        np.stack((lower_valid, arrays["valid"])),
                        dt_s=dt,
                        horizons_s=protocol["evaluation"]["horizons_s"],
                        thresholds=protocol["evaluation"]["responses"][
                            "weak_endpoint_thresholds"
                        ],
                    )
                    directions.extend(
                        dict(
                            row,
                            simulator=simulator,
                            scope=query["scope"],
                            cell=query["cell"],
                            parent=query["parent"],
                            origin=query["origin"],
                            channel=query["channel"],
                            arm=arm,
                        )
                        for row in scored
                    )
        path = directory / (query["parent"] + "/" + query["id"] + ".npz")
        if replay:
            array_equal(result, load_arrays(path), str(path))
        else:
            save_arrays(path, result)
        for arm, predicted in prediction.items():
            envelope = (
                predictors.generic.envelope()
                if arm == "generic"
                and query["kind"] == "factual"
                and predictors.generic is not None
                else None
            )
            scores = score_forecast(
                predicted,
                target,
                arrays["valid"],
                dt_s=dt,
                horizons_s=protocol["evaluation"]["horizons_s"],
                envelope=envelope,
                rotation_geometry=query["kind"] == "factual",
            )
            rows.extend(
                dict(
                    score,
                    simulator=simulator,
                    scope=query["scope"],
                    cell=query["cell"],
                    parent=query["parent"],
                    query=query["id"],
                    origin=query["origin"],
                    channel=query.get("channel"),
                    sign=query.get("sign"),
                    kind=query["kind"],
                    arm=arm,
                )
                for score in scores
            )
    nonweak = {
        (
            row["parent"],
            row["origin"],
            row["channel"],
            row["horizon_s"],
            row["group"],
        ): row["pair_nonweak"]
        for row in directions
    }
    for row in rows:
        if row["kind"] == "response":
            row["pair_nonweak"] = nonweak[
                (
                    row["parent"],
                    row["origin"],
                    row["channel"],
                    row["horizon_s"],
                    row["group"],
                )
            ]
    summary = aggregate(rows)
    if replay:
        if directions != read_json(directory / "directions.json"):
            raise ValueError("fresh response directions differ")
        if summary != read_json(directory / "summary.json") or rows != read_json(
            directory / "rows.json"
        ):
            raise ValueError("fresh metrics differ from saved scores")
        return {
            "prediction_queries": len(read_json(base / "data/queries.json")),
            "metric_rows": len(rows),
            "exact": True,
        }
    write_json(directory / "directions.json", directions)
    write_json(directory / "rows.json", rows)
    write_json(directory / "summary.json", summary)
    freeze_files(
        directory,
        "seal.json",
        {
            "data_seal": digest(base / "data/seal.json"),
            "models": {
                a: digest(base / a / "seal.json") for a in ("generic", "structured")
            },
            "wall_time_s": time.perf_counter() - start,
        },
    )
    print(f"evaluation complete {simulator}: {len(rows)} metric rows", flush=True)


def replay_data(simulator, output):
    protocol = read_json(PROTOCOL)
    runtime = check_sources(protocol)
    directory = Path(output) / simulator / "data"
    seal = verify_files(directory, "seal.json")
    if seal["runtime"] != runtime:
        raise ValueError("runtime/source identity differs from generation")
    fixture = fixture_for(simulator, protocol)
    if dict(fixture.config(), spec=fixture.spec.to_dict()) != read_json(
        directory / "configuration.json"
    ):
        raise ValueError("fixture configuration changed")
    records = read_json(directory / "records.json")
    original_queries = read_json(directory / "queries.json")
    all_queries, arrays_count = [], 0
    entries = [e for e in protocol["recordings"] if e["simulator"] == simulator]
    if len(records) != len(entries):
        raise ValueError("record roster changed")
    for entry, record in zip(entries, records):
        if record != read_json(directory / (record["prefix"] + ".json")):
            raise ValueError("per-parent record mirror differs")
        for key in entry:
            if entry[key] != record[key]:
                raise ValueError("record identity changed")
        cell = next(c for c in protocol["cells"][simulator] if c["id"] == entry["cell"])
        try:
            arrays, detail = fixture.generate(entry, cell)
        except Exception as error:
            if type(error).__name__ != "FixtureSetupError":
                raise
            arrays = None
            detail = {"setup_failure": str(error), "exception": type(error).__name__}
        if detail != record["metadata"] or cell != record["cell_facts"]:
            raise ValueError("fresh physical setup differs")
        if arrays is not None:
            array_equal(
                arrays,
                load_arrays(directory / (record["prefix"] + ".npz")),
                record["id"],
            )
            arrays_count += len(arrays)
            if (
                validity(arrays) != record["validity"]
                or achieved(arrays) != record["achieved"]
            ):
                raise ValueError("physical admission/coverage changed")
        elif record["validity"] != {
            "valid_transitions": 0,
            "valid_initial": False,
            "failure_index": 0,
            "reason": "setup_failure",
        }:
            raise ValueError("setup failure admission changed")
        if entry["role"] == "test":
            queries, values, branches = make_queries(
                fixture, arrays, entry, cell, protocol
            )
            for query in queries:
                query["scope"] = cell["group"]
                query["path"] = str(
                    Path("queries") / entry["id"] / (query["id"] + ".npz")
                )
                array_equal(
                    values[query["id"]],
                    load_arrays(directory / query["path"]),
                    query["path"],
                )
                arrays_count += len(values[query["id"]])
            for name, (values, detail) in branches.items():
                path = Path("branches") / entry["id"] / name
                array_equal(
                    values, load_arrays(directory / path.with_suffix(".npz")), str(path)
                )
                if detail != read_json(directory / path.with_suffix(".json")):
                    raise ValueError("branch metadata changed")
                arrays_count += len(values)
            all_queries.extend(queries)
        print(f"replay physics {entry['id']}", flush=True)
    if all_queries != original_queries:
        raise ValueError("query roster changed")
    admitted = [
        r["id"]
        for r in records
        if r["role"] == "calibration_pool"
        and r["validity"]["valid_initial"]
        and r["validity"]["valid_transitions"] >= round(0.75 / fixture.dt)
    ]
    if (
        admitted != seal["admitted"]
        or roles_for(admitted) != seal["roles"]
        or seal["roles"] != read_json(directory / "roles.json")
    ):
        raise ValueError("admission/split changed")
    return {
        "parents": len(records),
        "arrays": arrays_count,
        "queries": len(all_queries),
        "exact": True,
    }


def check_links(output, simulator, *, evaluation=False):
    """A local seal is insufficient: stages must bind the same data and sources."""
    base = Path(output) / simulator
    runtime = check_sources(read_json(PROTOCOL))
    data = verify_files(base / "data", "seal.json")
    if data["runtime"] != runtime:
        raise ValueError("data runtime/source identity differs")
    for arm in ("generic", "structured"):
        verify_files(base / arm, "seal.json")
        start = read_json(base / arm / "start.json")
        if start != {
            "runtime": runtime,
            "arm": arm,
            "simulator": simulator,
            "data_seal": digest(base / "data/seal.json"),
        }:
            raise ValueError("fit data/runtime provenance differs")
    if evaluation:
        seal = verify_files(base / "evaluation", "seal.json")
        if seal["data_seal"] != digest(base / "data/seal.json") or seal["models"] != {
            arm: digest(base / arm / "seal.json") for arm in ("generic", "structured")
        }:
            raise ValueError("evaluation data/model provenance differs")


def finalize(output):
    for simulator in ("crazyflow", "cascade"):
        check_links(output, simulator, evaluation=True)
    sha = freeze_files(
        output,
        "run.json",
        {
            "protocol_sha256": digest(PROTOCOL),
            "scope": "Frozen descriptive two-simulator prediction baseline",
        },
    )
    print(f"trusted bundle SHA256: {sha}", flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    if not expected_sha256 or digest(Path(output) / "run.json") != expected_sha256:
        raise ValueError("bundle differs from externally trusted SHA256")
    verify_files(output, "run.json")
    for simulator in ("crazyflow", "cascade"):
        check_links(output, simulator, evaluation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "fit", "evaluate", "finalize", "replay")
    )
    parser.add_argument("simulator", choices=("crazyflow", "cascade"))
    parser.add_argument("--arm", choices=("generic", "structured"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.stage == "generate":
        generate(args.simulator, args.output)
    elif args.stage == "fit":
        if args.arm is None:
            parser.error("--arm required for fit")
        fit_arm(args.simulator, args.arm, args.output)
    elif args.stage == "evaluate":
        evaluate(args.simulator, args.output)
    elif args.stage == "finalize":
        finalize(args.output)
    else:
        verify_bundle(args.output, args.expected_bundle_sha)
        result = {
            "physics": replay_data(args.simulator, args.output),
            "predictions": evaluate(args.simulator, args.output, replay=True),
        }
        write_json(args.output / args.simulator / "replay.json", result)


if __name__ == "__main__":
    main()
