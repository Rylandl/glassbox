"""Frozen training-only excitation around unchanged simulator fixture kernels.

The caller supplies the private historical flight module and frozen protocol.
Neither simulator globals nor learner inputs are extended by this adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

EXTRA_ARRAYS = (
    "excitation_pilot_commands",
    "excitation_block_index",
    "excitation_assigned_delta",
    "excitation_realized_delta",
    "excitation_clipped",
    "excitation_clipping_residual",
)
BOUNDARY_ARRAYS = {"time_s", "states", "full_states", "actuator_outputs", "wind"}


def _roles(simulator, p):
    roles = p["planned_automatic_roles"][simulator]
    if set(roles) != {"training", "development"}:
        raise ValueError("excitation requires explicit training/development roles")
    counts = p["counts"]
    expected = {
        "training": counts["excited_training_parents_per_simulator"],
        "development": counts["unchanged_development_copies_per_simulator"],
    }
    ids = roles["training"] + roles["development"]
    if len(ids) != len(set(ids)) or any(
        len(roles[key]) != count for key, count in expected.items()
    ):
        raise ValueError("frozen excitation roles are missing or duplicated")
    return roles


def _schedule(dt, steps, lower, upper, entry, p):
    rule = p["collection_intervention"]
    start = round(rule["start_s"] / dt)
    block = round(rule["block_s"] / dt)
    if (
        start < 0
        or block < 1
        or start >= steps
        or not np.isclose(start * dt, rule["start_s"], rtol=0, atol=1e-12)
        or not np.isclose(block * dt, rule["block_s"], rtol=0, atol=1e-12)
    ):
        raise ValueError("excitation schedule must contain integral sample intervals")
    count = (steps - start + block - 1) // block
    if count != 23:
        raise ValueError("frozen excitation requires exactly 23 blocks")
    lower, upper = (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    if (
        lower.ndim != 1
        or lower.shape != upper.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("excitation requires finite ordered command bounds")
    material = json.dumps(
        {"namespace": rule["id"], "parent": entry["id"], "seed": entry["seed"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    seed = int.from_bytes(digest, "big", signed=False)
    bits = np.random.Generator(np.random.PCG64(seed)).integers(
        0, 2, size=(count, len(lower)), dtype=np.int64
    )
    signs = 2 * bits - 1
    indices = np.full(steps, -1, dtype=np.int64)
    indices[start:] = (np.arange(start, steps, dtype=np.int64) - start) // block
    assigned = np.zeros((steps, len(lower)), dtype=np.float64)
    assigned[start:] = rule["range_fraction"] * (upper - lower) * signs[indices[start:]]
    detail = {
        "id": rule["id"],
        "role": "training",
        "rng": "numpy.PCG64",
        "seed_material": material,
        "seed_sha256": digest.hex(),
        "seed_integer": seed,
        "signs": signs.tolist(),
        "start_s": rule["start_s"],
        "block_s": rule["block_s"],
        "range_fraction": rule["range_fraction"],
        "dt_s": dt,
        "steps": steps,
        "start_index": start,
        "block_steps": block,
        "blocks": count,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }
    return detail, indices, assigned


def _issued(base, assigned, start, lower, upper):
    # The prefix is copied directly, preserving even signed-zero command bytes.
    commands = base.copy()
    commands[start:] = np.clip(base[start:] + assigned[start:], lower, upper)
    requested = base + assigned
    clipped = (requested < lower) | (requested > upper)
    clipped[:start] = False
    return commands, clipped


def excite(fixture, entry, cell, p):
    """Generate one training parent using actual perturbed issued commands."""
    roles = _roles(entry["simulator"], p)
    if entry["id"] not in roles["training"] or entry["role"] != "calibration_pool":
        raise ValueError("excitation is restricted to the frozen training roster")
    detail, indices, assigned = _schedule(
        fixture.dt, fixture.steps, fixture.lower, fixture.upper, entry, p
    )
    kernel = fixture._advance
    bases, actual = [], []

    def advance(state, command, index, wind):
        original_index = index
        index = int(index)
        if index != len(actual) or index >= fixture.steps:
            raise ValueError("excitation requires consecutive parent intervals")
        base = np.asarray(command, dtype=np.float64)
        if base.shape != np.asarray(fixture.lower).shape:
            raise ValueError("pilot command width changed")
        issued = (
            base.copy()
            if index < detail["start_index"]
            else np.clip(base + assigned[index], fixture.lower, fixture.upper)
        )
        bases.append(base.copy())
        actual.append(issued.copy())
        # Preserve the original clock argument type as well as its value.
        return kernel(state, jnp.asarray(issued), original_index, wind)

    fixture._advance = advance
    try:
        arrays, metadata = fixture.generate(entry, cell)
    finally:
        fixture._advance = kernel
    if len(actual) != fixture.steps:
        raise ValueError("excited parent did not return the full scheduled tape")
    base, commands = np.asarray(bases), np.asarray(actual)
    if (
        arrays["commands"].shape != base.shape
        or arrays["commands"].dtype != base.dtype
        or arrays["commands"].tobytes() != base.tobytes()
    ):
        raise ValueError("fixture did not record the intercepted pilot command tape")
    expected, clipped = _issued(
        base, assigned, detail["start_index"], fixture.lower, fixture.upper
    )
    if commands.tobytes() != expected.tobytes():
        raise ValueError("issued command tape differs from frozen excitation")
    arrays = dict(arrays)
    arrays.update(
        {
            "commands": commands,
            "excitation_pilot_commands": base,
            "excitation_block_index": indices,
            "excitation_assigned_delta": assigned,
            "excitation_realized_delta": commands - base,
            "excitation_clipped": clipped,
            "excitation_clipping_residual": commands - (base + assigned),
        }
    )
    return arrays, dict(metadata, command_excitation=detail)


def _common(simulator, output, flight, p):
    directory = Path(output) / simulator / "data"
    seal = flight.verify_files(directory, "seal.json")
    roles = _roles(simulator, p)
    entries = [
        e
        for e in p["recordings"]
        if e["simulator"] == simulator and e["role"] == "calibration_pool"
    ]
    expected_ids = [e["id"] for e in entries]
    if (
        set(expected_ids) != set(roles["training"] + roles["development"])
        or len(expected_ids) != len(set(expected_ids))
        or seal["admitted"] != expected_ids
        or seal["roles"] != roles
        or flight.read_json(directory / "roles.json") != roles
        or flight.roles_for(expected_ids) != roles
    ):
        raise ValueError("original calibration admission or frozen roles differ")
    records = [
        r
        for r in flight.read_json(directory / "records.json")
        if r["role"] == "calibration_pool"
    ]
    if [r["id"] for r in records] != expected_ids:
        raise ValueError("original calibration record roster differs")
    for entry, record in zip(entries, records, strict=True):
        if any(
            record[k] != value for k, value in entry.items()
        ) or record != flight.read_json(directory / (record["prefix"] + ".json")):
            raise ValueError("original calibration entry or record mirror differs")
    return directory, seal, entries, records, roles


def _check_parent(new, old, record, original, config, entry, flight, p):
    channels = [c for c in config["spec"]["channels"] if c["kind"] == "control"]
    detail, indices, assigned = _schedule(
        float(config["dt_s"]),
        len(new["commands"]),
        np.asarray([c["minimum"] for c in channels]),
        np.asarray([c["maximum"] for c in channels]),
        entry,
        p,
    )
    if record["metadata"].get("command_excitation") != detail:
        raise ValueError("excitation random tape or schedule metadata changed")
    metadata = dict(record["metadata"])
    del metadata["command_excitation"]
    if metadata != original["metadata"]:
        raise ValueError("excitation altered original initialization metadata")
    if set(new) != set(old) | set(EXTRA_ARRAYS):
        raise ValueError("excitation array roster differs")
    start = detail["start_index"]
    for key, value in old.items():
        if new[key].shape != value.shape or new[key].dtype != value.dtype:
            raise ValueError("excitation changed a parent array shape or dtype")
        count = start + int(key in BOUNDARY_ARRAYS)
        left, right = (
            (new[key], value)
            if key == "control_prefix"
            else (new[key][:count], value[:count])
        )
        flight.array_equal({key: left}, {key: right}, entry["id"] + "/prefix")
    base = new["excitation_pilot_commands"]
    if base.shape != new["commands"].shape or base.dtype != np.dtype("float64"):
        raise ValueError("excitation pilot tape shape or dtype changed")
    issued, clipped = _issued(base, assigned, start, detail["lower"], detail["upper"])
    expected = {
        "commands": issued,
        "excitation_block_index": indices,
        "excitation_assigned_delta": assigned,
        "excitation_realized_delta": issued - base,
        "excitation_clipped": clipped,
        "excitation_clipping_residual": issued - (base + assigned),
    }
    flight.array_equal({key: new[key] for key in expected}, expected, entry["id"])


def _compare(simulator, output, flight, p, *, sealed):
    common, common_seal, entries, originals, roles = _common(
        simulator, output, flight, p
    )
    directory = Path(output) / simulator / "excitation"
    seal = flight.verify_files(directory, "seal.json") if sealed else None
    if (directory / "configuration.json").read_bytes() != (
        common / "configuration.json"
    ).read_bytes():
        raise ValueError("excitation changed physical configuration")
    config = flight.read_json(directory / "configuration.json")
    records = flight.read_json(directory / "records.json")
    if (
        len(records) != len(entries)
        or flight.read_json(directory / "roles.json") != roles
    ):
        raise ValueError("excitation record roster or roles changed")
    array_count, prefix_arrays, copied = 0, 0, 0
    admitted = []
    for entry, record, original in zip(entries, records, originals, strict=True):
        if (
            any(record.get(k) != value for k, value in entry.items())
            or record["cell_facts"] != original["cell_facts"]
            or record["prefix"] != original["prefix"]
            or record != flight.read_json(directory / (record["prefix"] + ".json"))
        ):
            raise ValueError("excited calibration entry or record mirror changed")
        path = record["prefix"] + ".npz"
        arrays, old = (
            flight.load_arrays(directory / path),
            flight.load_arrays(common / path),
        )
        array_count += len(arrays)
        if entry["id"] in roles["development"]:
            if (
                record != original
                or (directory / path).read_bytes() != (common / path).read_bytes()
                or (directory / (record["prefix"] + ".json")).read_bytes()
                != (common / (record["prefix"] + ".json")).read_bytes()
            ):
                raise ValueError("development parent was changed instead of copied")
            copied += 1
        else:
            _check_parent(arrays, old, record, original, config, entry, flight, p)
            prefix_arrays += len(old)
        if (
            flight.validity(arrays) != record["validity"]
            or flight.achieved(arrays) != record["achieved"]
        ):
            raise ValueError("excitation validity or support diagnostics changed")
        valid = record["validity"]
        if valid["valid_initial"] and valid["valid_transitions"] >= round(
            0.75 / config["dt_s"]
        ):
            admitted.append(entry["id"])
    if admitted != common_seal["admitted"] or flight.roles_for(admitted) != roles:
        raise ValueError("excitation changed admission or automatic roles")
    if sealed and (
        seal["admitted"] != admitted
        or seal["roles"] != roles
        or seal["common_data_seal"] != flight.digest(common / "seal.json")
        or seal["runtime"] != common_seal["runtime"]
        or seal["simulator"] != simulator
        or seal["planned_records"] != len(entries)
    ):
        raise ValueError("excitation seal provenance or role links changed")
    return {
        "common_data_seal": flight.digest(common / "seal.json"),
        "calibration_parents": len(entries),
        "excited_training_parents": len(entries) - copied,
        "copied_development_parents": copied,
        "arrays": array_count,
        "exact_prefix_arrays": prefix_arrays,
        "roles_and_admission_exact": True,
        "development_files_exact": True,
        "assigned_and_realized_excitation_exact": True,
    }


def compare_excitation(simulator, output, flight, p):
    """Check sealed data links, fixed roles, unchanged prefixes and random tapes."""
    comparison = _compare(simulator, output, flight, p, sealed=True)
    seal = flight.read_json(Path(output) / simulator / "excitation/seal.json")
    if seal["comparison"] != comparison:
        raise ValueError("excitation saved comparison differs")
    return comparison


def generate_excitation(simulator, output, flight, p, runtime):
    common, common_seal, entries, originals, roles = _common(
        simulator, output, flight, p
    )
    if common_seal["runtime"] != runtime:
        raise ValueError("excitation runtime differs from common data")
    fixture = flight.fixture_for(simulator, p)
    if dict(fixture.config(), spec=fixture.spec.to_dict()) != flight.read_json(
        common / "configuration.json"
    ):
        raise ValueError("excitation fixture configuration differs from common data")
    directory = Path(output) / simulator / "excitation"
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(common / "configuration.json", directory / "configuration.json")
    start = time.perf_counter()
    records = []
    for entry, original in zip(entries, originals, strict=True):
        prefix = original["prefix"]
        (directory / prefix).parent.mkdir(parents=True, exist_ok=True)
        if entry["id"] in roles["development"]:
            record = copy.deepcopy(original)
            for suffix in (".npz", ".json"):
                shutil.copyfile(
                    common / (prefix + suffix), directory / (prefix + suffix)
                )
        else:
            print(f"excite {entry['id']}", flush=True)
            arrays, metadata = excite(fixture, entry, original["cell_facts"], p)
            record = dict(
                entry,
                cell_facts=original["cell_facts"],
                prefix=prefix,
                metadata=metadata,
                validity=flight.validity(arrays),
                achieved=flight.achieved(arrays),
            )
            flight.save_arrays(directory / (prefix + ".npz"), arrays)
            flight.write_json(directory / (prefix + ".json"), record)
        records.append(record)
    flight.write_json(directory / "records.json", records)
    flight.write_json(directory / "roles.json", roles)
    comparison = _compare(simulator, output, flight, p, sealed=False)
    flight.freeze_files(
        directory,
        "seal.json",
        {
            "runtime": runtime,
            "simulator": simulator,
            "wall_time_s": time.perf_counter() - start,
            "planned_records": len(records),
            "admitted": common_seal["admitted"],
            "roles": roles,
            "common_data_seal": flight.digest(common / "seal.json"),
            "comparison": comparison,
        },
    )
    return compare_excitation(simulator, output, flight, p)


def replay_excitation(simulator, output, flight, p, runtime):
    comparison = compare_excitation(simulator, output, flight, p)
    common, _, entries, originals, roles = _common(simulator, output, flight, p)
    directory = Path(output) / simulator / "excitation"
    if flight.verify_files(directory, "seal.json")["runtime"] != runtime:
        raise ValueError("excitation replay runtime/source identity changed")
    fixture = flight.fixture_for(simulator, p)
    if dict(fixture.config(), spec=fixture.spec.to_dict()) != flight.read_json(
        common / "configuration.json"
    ):
        raise ValueError("excitation replay physical configuration changed")
    records = flight.read_json(directory / "records.json")
    arrays_count, regenerated = 0, 0
    for entry, record, original in zip(entries, records, originals, strict=True):
        path = record["prefix"] + ".npz"
        if entry["id"] in roles["development"]:
            arrays = flight.load_arrays(common / path)
            metadata = original["metadata"]
        else:
            arrays, metadata = excite(fixture, entry, record["cell_facts"], p)
            regenerated += 1
        if metadata != record["metadata"]:
            raise ValueError("fresh excitation metadata changed")
        flight.array_equal(arrays, flight.load_arrays(directory / path), entry["id"])
        arrays_count += len(arrays)
        print(f"replay excitation {entry['id']}", flush=True)
    return dict(
        comparison,
        regenerated_training_parents=regenerated,
        replay_arrays=arrays_count,
        exact=True,
    )
