"""Independent array/accounting checks for the frozen JSBSim onboarding audit.

These checks do not import the simulator adapter. They establish internal evidence
consistency; authenticating a coherently rewritten trajectory additionally requires
the runner's fresh simulator replay against pinned sources and assets.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

ARRAY_NAMES = (
    "observations", "native_observations", "time_s", "commands",
    "immediate_readbacks", "end_readbacks", "native_attempted", "native_completed",
    "failed_native_observation", "failed_observation", "failed_time_s",
)
GROUPS = {
    "velocity_m_s": slice(0, 3),
    "body_rate_rad_s": slice(3, 6),
    "rotation_entries": slice(6, 15),
    "altitude_m": slice(15, 16),
}
SURFACES = ["fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm", "fcs/rudder-cmd-norm"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def bitwise_equal(left, right):
    """Exact array evidence equality, including signed zero and NaN payloads."""
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    )


def expected_branches(command_count):
    rows = [{"label": "factual", "command_index": None, "direction": None}]
    for index in range(command_count):
        for direction in ("lower", "upper"):
            rows.append({"label": f"c{index:03d}_{direction}",
                         "command_index": index, "direction": direction})
    return rows


def expected_tape(protocol, entry, initial, bounds):
    """Recreate the prescribed PCG64 tape without using generator implementation."""
    seed = int.from_bytes(
        hashlib.sha256((protocol["id"] + ":" + entry["id"]).encode()).digest()[:8],
        "little",
    )
    random = np.random.Generator(np.random.PCG64(seed))
    length = protocol["recording"]["transitions"]
    rows = []
    for _ in range(0, length, 5):
        offsets = random.uniform(-1, 1, len(initial))
        row = np.clip(initial + 0.05 * (bounds[:, 1] - bounds[:, 0]) * offsets,
                      bounds[:, 0], bounds[:, 1])
        rows.extend([row] * 5)
    return np.asarray(rows[:length], dtype=np.float64)


def _mapped(native):
    mapped = native.copy()
    mapped[:, :3] *= 0.3048
    mapped[:, 6:15] = native[:, 6:15].reshape(-1, 3, 3).transpose(0, 2, 1).reshape(-1, 9)
    mapped[:, 15] *= 0.3048
    return mapped


def _contract(metadata, protocol):
    names = metadata.get("command_names", [])
    raw_bounds = metadata.get("bounds", [])
    if not names:
        _require(raw_bounds == [], "unknown command contract has bounds")
        return None, np.empty((0, 2), dtype=np.float64)
    count = metadata.get("diagnostics", {}).get("engine_count")
    _require(type(count) is int and count >= 0, "missing/invalid engine count")
    expected = SURFACES + [
        "fcs/throttle-cmd-norm" + (f"[{i}]" if i else "") for i in range(count)
    ]
    _require(names == expected, "command names/order differ from frozen contract")
    bounds = np.asarray(raw_bounds, dtype=np.float64)
    expected_bounds = np.asarray(
        [protocol["recording"]["bounds"]["surfaces"]] * 3
        + [protocol["recording"]["bounds"]["throttle"]] * count, dtype=np.float64,
    )
    _require(bitwise_equal(bounds, expected_bounds), "command bounds differ")
    return len(names), bounds


def _arrays_for(label, arrays):
    keys = {f"{label}__{name}" for name in ARRAY_NAMES}
    _require(keys.issubset(arrays), f"missing arrays for {label}")
    result = {name: np.asarray(arrays[f"{label}__{name}"]) for name in ARRAY_NAMES}
    for name, value in result.items():
        _require(value.dtype == np.dtype("float64"), f"{label}/{name} must be float64")
    return result


def _validate_run(label, metadata, values, protocol, planned_length, inherited_contract=None):
    status = metadata.get("status")
    _require(isinstance(status, str) and status not in ("", "unattempted"),
             f"invalid attempted run status: {label}")
    count, bounds = _contract(metadata, protocol)
    width = count or 0
    command_count, command_bounds = count, bounds
    if count is None and inherited_contract is not None and values["commands"].shape[-1:] != (0,):
        command_count, command_bounds = inherited_contract
    command_width = command_count or 0
    observed, raw, times = (values[k] for k in ("observations", "native_observations", "time_s"))
    _require(observed.ndim == 2 and observed.shape[1] == 16, "observation shape")
    _require(raw.shape == observed.shape and times.shape == (len(observed),), "native/time shape")
    _require(np.isfinite(observed).all() and np.isfinite(raw).all()
             and np.isfinite(times).all(), "nonfinite valid observation prefix")
    _require(bitwise_equal(observed, _mapped(raw)), "native observation unit/frame conversion differs")
    rotations = observed[:, 6:15].reshape(-1, 3, 3)
    if len(observed):
        orthogonality = np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
        determinant_error = np.max(np.abs(np.linalg.det(rotations) - 1))
        _require(orthogonality <= 1e-8 and determinant_error <= 1e-8, "invalid saved rotation")
        _require(np.allclose(np.diff(times), protocol["recording"]["dt_s"], rtol=0, atol=1e-12),
                 "observation interval differs from frozen timing")
    completed_intervals = max(0, len(observed) - 1)
    _require(metadata.get("completed_intervals") == completed_intervals, "completed interval count differs")
    tape = values["commands"]
    _require(tape.ndim == 2 and tape.shape[1] == command_width, "command tape shape")
    _require(len(tape) in (0, planned_length), "intended command tape is truncated")
    _require(np.isfinite(tape).all(), "nonfinite intended commands")
    if command_count is not None:
        _require(np.all(tape >= command_bounds[:, 0]) and np.all(tape <= command_bounds[:, 1]), "command bounds violated")
    _require(completed_intervals <= len(tape), "observations extend beyond command tape")
    immediate, end = values["immediate_readbacks"], values["end_readbacks"]
    substeps = protocol["recording"]["substeps"]
    _require(immediate.ndim == 3 and immediate.shape[1:] == (substeps, width)
             and end.shape == immediate.shape, "readback shape")
    intervals = len(immediate)
    _require(completed_intervals <= intervals <= min(completed_intervals + 1, len(tape)),
             "attempted interval count inconsistent with observed prefix")
    attempted, completed = values["native_attempted"], values["native_completed"]
    _require(attempted.shape == completed.shape == (intervals, substeps), "native mask shape")
    _require(np.isin(attempted, [0., 1.]).all() and np.isin(completed, [0., 1.]).all()
             and np.all(completed <= attempted), "invalid native masks")
    for name, mask in (("attempted", attempted), ("completed", completed)):
        _require(np.all(np.diff(mask.ravel()) <= 0), f"nonchronological native {name} mask")
    _require(np.all(completed[:completed_intervals] == 1), "observed interval missing native completion")
    # The final read/write can fail before run() is invoked, hence masks do not
    # assert that all unattempted readbacks must be NaN.
    if intervals:
        native_position = int(np.sum(attempted))
        for readback in (immediate, end):
            flat = readback.reshape(-1, width)
            _require(np.isnan(flat[native_position + 1:]).all(), "readbacks after unattempted native step")
        _require(np.isnan(end.reshape(-1, width)[native_position:]).all(),
                 "end readback exists without native run attempt")
    failed_raw, failed_obs, failed_time = (values[k] for k in (
        "failed_native_observation", "failed_observation", "failed_time_s"))
    _require(failed_raw.shape in ((0, 16), (1, 16))
             and failed_obs.shape == failed_raw.shape
             and failed_time.shape == (len(failed_raw),), "failed observation shape")
    if len(failed_raw):
        _require(metadata.get("stage") in ("initial_observation", "observation"),
                 "failed observation has wrong failure stage")
    if status == "completed":
        _require(count is not None and len(observed) == planned_length + 1
                 and intervals == planned_length and len(tape) == planned_length,
                 "completed run is incomplete")
        _require(np.all(attempted == 1) and np.all(completed == 1), "completed run has missing native steps")
        # Frozen failures concern observations and the command baseline. A native
        # command getter returning nonfinite data is retained/countable evidence;
        # it does not silently add a new numerical qualification criterion.
        _require(not len(failed_raw) and metadata.get("failed_interval") is None
                 and metadata.get("failed_substep") is None, "completed run carries failure")
    elif status == "running" and intervals == completed_intervals:
        _require(np.all(attempted == 1) and np.all(completed == 1), "running boundary lacks native steps")
        if intervals:
            _require(metadata.get("failed_interval") == completed_intervals - 1
                     and metadata.get("failed_substep") == substeps - 1
                     and metadata.get("stage") == "observation", "running boundary breadcrumb differs")
    elif intervals:
        _require(metadata.get("failed_interval") == completed_intervals,
                 "failed interval does not follow completed prefix")
        substep = metadata.get("failed_substep")
        _require(type(substep) is int and 0 <= substep < substeps, "invalid failed native substep")
        if metadata.get("stage") == "observation":
            _require(substep == substeps - 1 and np.all(completed[-1] == 1),
                     "failed boundary observation lacks native steps")
        else:
            _require(np.all(attempted[-1, substep + 1:] == 0), "native attempts after failed substep")
    if len(tape) and len(observed):
        initial = np.asarray(metadata.get("diagnostics", {}).get("initial_commands"), dtype=np.float64)
        _require(initial.shape == (width,) and np.isfinite(initial).all()
                 and np.all(initial >= bounds[:, 0]) and np.all(initial <= bounds[:, 1]),
                 "invalid baseline for generated commands")
    intended = tape[:intervals, None, :]
    result = {
        "attempted_native_substeps": int(attempted.sum()),
        "completed_native_substeps": int(completed.sum()),
    }
    for name, data in (("immediate", immediate), ("end", end)):
        finite = np.isfinite(data)
        result[f"{name}_overwrites"] = int(np.sum(finite & (data != intended))) if intervals else 0
        result[f"{name}_nonfinite_attempted_readbacks"] = int(
            np.sum((~finite) & (attempted[..., None] == 1))
        )
    return result


def _counts(labels, runs, known=True):
    if not known:
        return {"planned": None, "attempted": sum(runs[x]["status"] != "unattempted" for x in labels),
                "completed": sum(runs[x]["status"] == "completed" for x in labels), "unattempted": None}
    attempted = sum(runs[x]["status"] != "unattempted" for x in labels)
    return {"planned": len(labels), "attempted": attempted,
            "completed": sum(runs[x]["status"] == "completed" for x in labels),
            "unattempted": len(labels) - attempted}


def _same_prefix(left, right, transitions):
    return all(bitwise_equal(left[name][:transitions + 1], right[name][:transitions + 1])
               for name in ("observations", "native_observations", "time_s")) and all(
        bitwise_equal(left[name][:transitions], right[name][:transitions])
        for name in ("commands", "immediate_readbacks", "end_readbacks", "native_attempted", "native_completed")
    )


def validate_case(case: dict, arrays: dict[str, np.ndarray], protocol: dict, entry: dict) -> dict:
    """Return independently computed outcomes; numerical/replay failure is data.

    Malformed contracts, rosters, commands, units or array accounting raise
    ValueError. A valid simulator failure or unequal fresh prefix is retained as
    an outcome and never turned into an inactive command or a successful case.
    """
    _require(case.get("id") == entry["id"], "candidate identity differs")
    execution = case.get("execution_status")
    if execution is not None:
        _require(execution in ("timeout", "worker_failure"), "unknown execution outcome")
        _require(case.get("timeout_s") == protocol["recording"]["case_timeout_s"], "timeout budget differs")
        if execution == "worker_failure":
            _require(type(case.get("returncode")) is int and case["returncode"] < 0,
                     "worker_failure must identify a native process signal")
        else:
            _require(case.get("returncode") is None, "timeout cannot have successful process return")
    runs = case.get("runs", {})
    if execution is not None and not arrays:
        _require(case.get("branches") == [] and (
            runs == {} or runs == {"parent": {"status": "running", "stage": "construct"}}),
            "missing evidence for noninitial execution checkpoint")
        return {"outcome": execution, "command_count": None, "activity": [],
                "active_commands": 0, "weak_commands": 0, "unavailable_commands": None,
                "run_counts": {"planned": None, "attempted": len(runs), "completed": 0, "unattempted": None},
                "branch_counts": {"planned": None, "attempted": 0, "completed": 0, "unattempted": None},
                "parent_replay_equal": None, "factual_equal": None,
                "prefixes_equal": {}, "readbacks": {}}
    _require("parent" in runs, "missing parent run")
    count, bounds = _contract(runs["parent"], protocol)
    roster = expected_branches(count or 0)
    _require(case.get("branches") == roster, "branch roster or mapping differs")
    labels = ["parent", "replay", *(branch["label"] for branch in roster)]
    _require(set(runs) == set(labels), "run roster differs")
    attempted_labels = [label for label in labels if runs[label].get("status") != "unattempted"]
    _require("parent" in attempted_labels, "parent cannot be unattempted")
    running = [label for label in labels if runs[label].get("status") == "running"]
    _require(not running or (execution is not None and len(running) == 1), "nonterminal run without execution failure")
    if execution is not None:
        _require(attempted_labels == labels[:len(attempted_labels)], "nonchronological interrupted run roster")
        _require(not running or running == attempted_labels[-1:], "running run is not last attempted")
    metadata_only = [label for label in attempted_labels if not any(key.startswith(label + "__") for key in arrays)]
    for label in metadata_only:
        _require(execution is not None and runs[label] == {"status": "running", "stage": "construct"},
                 "attempted run lacks arrays")
    saved_labels = [label for label in attempted_labels if label not in metadata_only]
    expected_keys = {f"{label}__{name}" for label in saved_labels for name in ARRAY_NAMES}
    _require(set(arrays) == expected_keys, "unexpected/missing array roster")
    values, readbacks = {}, {}
    length = protocol["recording"]["transitions"]
    origin, horizon = protocol["replay"]["branch_origin"], protocol["replay"]["horizon_steps"]
    for label in saved_labels:
        data = _arrays_for(label, arrays)
        values[label] = data
        readbacks[label] = _validate_run(label, runs[label], data, protocol,
                                         length if label in ("parent", "replay") else origin + horizon,
                                         inherited_contract=(count, bounds) if label != "parent" else None)
        if label != "parent" and runs[label].get("command_names"):
            _require(runs[label]["command_names"] == runs["parent"].get("command_names"),
                     "fresh run command contract differs")
    parent_complete = runs["parent"]["status"] == "completed"
    if not parent_complete:
        _require(attempted_labels == ["parent"], "branches attempted after incomplete parent")
    elif execution is None:
        _require(len(attempted_labels) == len(labels), "complete parent has unattempted planned runs")
    tape = values["parent"]["commands"]
    if len(tape):
        initial = np.asarray(runs["parent"]["diagnostics"]["initial_commands"], dtype=np.float64)
        _require(bitwise_equal(tape, expected_tape(protocol, entry, initial, bounds)),
                 "parent deterministic command tape differs")
    for label in saved_labels:
        if label == "parent" or not len(values[label]["commands"]):
            continue
        expected = tape.copy() if label == "replay" else tape[:origin + horizon].copy()
        branch = next((row for row in roster if row["label"] == label), None)
        if branch is not None and branch["command_index"] is not None:
            index = branch["command_index"]
            bound = bounds[index, int(branch["direction"] == "upper")]
            expected[origin:, index] += 0.1 * (bound - expected[origin:, index])
        _require(bitwise_equal(values[label]["commands"], expected), f"{label} intended tape differs")
    def is_complete(label):
        return runs[label]["status"] == "completed"
    replay_equal = (_same_prefix(values["parent"], values["replay"], length)
                    if parent_complete and is_complete("replay") else None)
    factual_equal = (_same_prefix(values["parent"], values["factual"], origin + horizon)
                     if parent_complete and is_complete("factual") else None)
    prefixes = {branch["label"]: (
        _same_prefix(values["parent"], values[branch["label"]], origin)
        if parent_complete and is_complete(branch["label"]) else None
    ) for branch in roster[1:]}
    activity = []
    for index in range(count or 0):
        lower, upper = f"c{index:03d}_lower", f"c{index:03d}_upper"
        available = execution is None and factual_equal is True and prefixes[lower] is True and prefixes[upper] is True
        row = {"command_index": index, "name": runs["parent"]["command_names"][index],
               "available": available, "active": None, "lower": None, "upper": None,
               "lower_signed_deltas": None, "upper_signed_deltas": None}
        if available:
            active = False
            factual = values["factual"]["observations"][origin + 1:origin + horizon + 1]
            for direction, label in (("lower", lower), ("upper", upper)):
                delta = values[label]["observations"][origin + 1:origin + horizon + 1] - factual
                magnitudes = {group: np.max(np.abs(delta[:, columns]), axis=1).tolist()
                              for group, columns in GROUPS.items()}
                row[direction] = magnitudes
                row[f"{direction}_signed_deltas"] = delta.tolist()
                active |= any(max(magnitudes[group]) > protocol["replay"]["activity_thresholds"][group]
                              for group in GROUPS)
            row["active"] = bool(active)
        activity.append(row)
    if execution is not None:
        outcome = execution
    elif not parent_complete:
        outcome = "parent_failure" if len(tape) else "setup_failure"
    elif replay_equal is not True or factual_equal is False or any(x is False for x in prefixes.values()):
        outcome = "replay_failure"
    elif not all(is_complete(label) for label in labels):
        outcome = "branch_failure"
    else:
        outcome = "completed"
    return {
        "outcome": outcome, "command_count": count,
        "run_counts": _counts(labels, runs, count is not None),
        "branch_counts": _counts([row["label"] for row in roster], runs, count is not None),
        "parent_replay_equal": replay_equal, "factual_equal": factual_equal,
        "prefixes_equal": prefixes, "activity": activity,
        "active_commands": sum(row["active"] is True for row in activity),
        "weak_commands": sum(row["active"] is False for row in activity),
        "unavailable_commands": sum(not row["available"] for row in activity) if count is not None else None,
        "readbacks": readbacks,
    }


def summarize(cases: list[dict], inventory: dict) -> dict:
    """Preserve every candidate, directory, root template and unavailable count."""
    ids = [entry["id"] for entry in inventory["entries"]]
    _require(len(ids) == len(set(ids)), "duplicate frozen inventory candidate")
    _require(len(cases) == len(ids) and {case["id"] for case in cases} == set(ids),
             "candidate inventory accounting differs")
    by_id = {case["id"]: case for case in cases}
    rows = []
    for entry_id in ids:
        case = by_id[entry_id]
        computed = case.get("computed")
        _require(isinstance(computed, dict), "missing computed case accounting")
        rows.append({"id": entry_id, **{name: computed[name] for name in (
            "outcome", "command_count", "active_commands", "weak_commands", "unavailable_commands",
            "run_counts", "branch_counts")}})
    directory_names = inventory["directories"]
    _require(len(directory_names) == len(set(directory_names)), "duplicate directory inventory")
    directories = []
    for name in directory_names:
        children = [row for row in rows if "/" in row["id"] and row["id"].split("/", 1)[0] == name]
        directories.append({"directory": name, "candidate_count": len(children),
                            "status": "has_candidates" if children else "no_config",
                            "candidates": [row["id"] for row in children],
                            "outcome_counts": dict(sorted(Counter(row["outcome"] for row in children).items()))})
    root_rows = [row for row in rows if "/" not in row["id"]]
    _require(sum(row["candidate_count"] for row in directories) + len(root_rows) == len(rows),
             "candidate outside directory/root accounting")
    result = {"candidate_count": len(rows), "directory_count": len(directories),
              "outcome_counts": dict(sorted(Counter(row["outcome"] for row in rows).items())),
              "candidates": rows, "directories": directories, "root_candidates": root_rows,
              "command_activity": {
                  "known_commands": sum(row["command_count"] or 0 for row in rows),
                  "unknown_command_count_cases": sum(row["command_count"] is None for row in rows),
                  "active": sum(row["active_commands"] for row in rows),
                  "weak": sum(row["weak_commands"] for row in rows),
                  "unavailable_known_commands": sum(row["unavailable_commands"] or 0 for row in rows),
              }}
    for key in ("run_counts", "branch_counts"):
        result[key] = {name: sum(row[key][name] or 0 for row in rows)
                       for name in ("planned", "attempted", "completed", "unattempted")}
        result[key]["unknown_planned_cases"] = sum(row[key]["planned"] is None for row in rows)
    return result
