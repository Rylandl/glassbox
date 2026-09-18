"""Independent accounting and horizon metrics for frozen JSBSim excitation.

Core evidence rules are inherited unchanged. Diagnostic telemetry never gates
physical response availability. Fresh pinned simulator replay remains necessary
to authenticate internally consistent, coherently rewritten measurements.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from . import jsbsim_onboarding_audit as old

EXTRA_ARRAYS = ("telemetry", "telemetry_valid", "failed_telemetry", "failed_telemetry_valid") + tuple(
    f"startup_{phase}_{field}"
    for phase in ("pre", "post")
    for field in ("telemetry", "telemetry_valid", "native_observation", "observation",
                  "observation_valid", "commands", "commands_valid", "time_s", "time_valid")
)


def _require(value, message):
    old._require(value, message)


def requested_telemetry(protocol, engines):
    names = list(protocol["telemetry"]["global_properties"])
    for index in range(engines):
        suffix = f"[{index}]" if index else ""
        names.extend(f"propulsion/engine{suffix}/{field}" for field in protocol["telemetry"]["engine_suffixes"])
        names.extend(f"fcs/{field}{suffix}" for field in protocol["telemetry"]["fcs_per_engine"])
    return names


def common_tape(protocol, inherited, entry, reference_case, reference_arrays):
    """Extend the historical random stream using only its saved baseline."""
    parent = reference_case["runs"]["parent"]
    count, bounds = old._contract(parent, inherited)
    baseline = parent.get("diagnostics", {}).get("initial_commands")
    if baseline is None or count is None:
        return None
    initial = np.asarray(baseline, dtype=np.float64)
    if (initial.shape != (count,) or not np.isfinite(initial).all()
            or np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1])):
        return None
    expanded = deepcopy(inherited)
    expanded["recording"]["transitions"] = protocol["recording"]["transitions"]
    tape = old.expected_tape(expanded, entry, initial, bounds)
    saved = reference_arrays.get("parent__commands")
    if saved is not None and len(saved):
        _require(old.bitwise_equal(tape[:len(saved)], saved), "historical command tape differs")
    return tape


def _validity(values, mask, shape, name, *, diagnostic_nan=True):
    _require(values.shape == mask.shape == shape, f"{name} shape")
    _require(np.isin(mask, [0., 1.]).all(), f"{name} validity mask")
    _require(np.array_equal(mask == 1, np.isfinite(values)), f"{name} finite/valid mismatch")
    if diagnostic_nan:
        _require(np.isnan(values[mask == 0]).all(), f"{name} invalid values must be NaN")


def _telemetry(metadata, data, protocol, arm):
    telemetry, startup = metadata["telemetry"], metadata["startup"]
    names, missing = telemetry["names"], telemetry["missing"]
    _require(len(set(names)) == len(names) and len(set(missing)) == len(missing), "duplicate telemetry path")
    engines = metadata.get("diagnostics", {}).get("engine_count", 0)
    requested = requested_telemetry(protocol, engines)
    if names or missing or startup["native_run_ic_result"] is True:
        _require(set(names).isdisjoint(missing) and set(names) | set(missing) == set(requested),
                 "telemetry path accounting differs")
        _require(names == [x for x in requested if x in names]
                 and missing == [x for x in requested if x in missing], "telemetry path order differs")
    n, width = len(data["observations"]), len(names)
    failed_rows = len(data["failed_telemetry"])
    _require(failed_rows <= len(data["failed_observation"]), "failed telemetry has no captured failed core boundary")
    for prefix, rows, errors in (("", n, telemetry["errors"]),
                                 ("failed_", failed_rows, telemetry["failed_errors"])):
        values, mask = data[prefix + "telemetry"], data[prefix + "telemetry_valid"]
        _validity(values, mask, (rows, width), prefix + "telemetry")
        _require(len(errors) == rows and all(isinstance(row, dict) for row in errors), "telemetry error row accounting")
        for index, row in enumerate(errors):
            expected_errors = {names[j] for j in range(width) if not mask[index, j]}
            _require(set(row) == expected_errors and all(isinstance(v, str) and v for v in row.values()),
                     "telemetry error/validity accounting differs")
    _require(startup["arm"] == arm, "startup arm differs")
    for key in ("native_run_ic_attempted_count", "native_run_ic_returned_count", "attempted_count", "returned_count"):
        _require(type(startup[key]) is int and startup[key] in (0, 1), "startup call count")
    _require(startup["native_run_ic_returned_count"] <= startup["native_run_ic_attempted_count"]
             and startup["returned_count"] <= startup["attempted_count"], "startup return count")
    _require(startup["native_run_ic_result"] is None or type(startup["native_run_ic_result"]) is bool, "native IC result")
    if startup["native_run_ic_result"] is not None:
        _require(startup["native_run_ic_returned_count"] == 1, "native IC result lacks return")
    if arm == "as_shipped":
        _require(startup["attempted_count"] == startup["returned_count"] == 0, "as-shipped attempted bootstrap")
    if startup["attempted_count"]:
        _require(startup["native_run_ic_result"] is True and arm == "engine_bootstrap", "bootstrap before native IC")
    if startup["phase"] == "ready" and arm == "engine_bootstrap":
        _require(startup["attempted_count"] == startup["returned_count"] == 1, "missing completed bootstrap")
    _require(startup["phase"] in ("uninitialized", "native_run_ic", "pre_startup", "bootstrap", "post_startup", "ready"),
             "unknown startup phase")
    commands = len(metadata.get("command_names", []))
    for phase in ("pre", "post"):
        prefix = f"startup_{phase}_"
        rows = len(data[prefix + "observation"])
        _require(rows in (0, 1), "startup snapshot row count")
        _require(data[prefix + "native_observation"].shape == (rows, 16), "startup native observation shape")
        for field, shape in (("telemetry", (rows, width)), ("observation", (rows, 16)),
                             ("commands", (rows, commands)), ("time", (rows,))):
            value_key = "time_s" if field == "time" else field
            _validity(data[prefix + value_key], data[prefix + field + "_valid"], shape,
                      prefix + field, diagnostic_nan=field == "telemetry")
        if rows and np.isfinite(data[prefix + "native_observation"]).all():
            _require(old.bitwise_equal(old._mapped(data[prefix + "native_observation"]), data[prefix + "observation"]),
                     "startup unit/frame conversion differs")
    if startup["phase"] == "ready":
        _require(len(data["startup_pre_observation"]) == len(data["startup_post_observation"]) == 1,
                 "ready startup snapshots missing")
    return {"present_paths": len(names), "missing_paths": len(missing),
            "valid_values": int(data["telemetry_valid"].sum()),
            "unavailable_values": int(data["telemetry_valid"].size - data["telemetry_valid"].sum()),
            "startup_attempted": startup["attempted_count"], "startup_returned": startup["returned_count"],
            "startup_error": startup["error"]}


def _prefix_steps(left, right):
    """Largest byte-equal complete core prefix; -1 means x0 differs/unavailable."""
    if not len(left["observations"]) or not len(right["observations"]):
        return -1
    maximum = min(len(left["observations"]), len(right["observations"])) - 1
    if not old._same_prefix(left, right, 0):
        return -1
    low, high = 0, maximum + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if old._same_prefix(left, right, middle):
            low = middle
        else:
            high = middle
    return low


def _telemetry_parity(left, right, left_meta, right_meta, steps):
    left_names, right_names = left_meta["telemetry"]["names"], right_meta["telemetry"]["names"]
    common = [name for name in left_names if name in right_names]
    compared = unequal = unavailable = 0
    rows = min(steps + 1, len(left["telemetry"]), len(right["telemetry"]))
    for name in common:
        i, j = left_names.index(name), right_names.index(name)
        valid = (left["telemetry_valid"][:rows, i] == 1) & (right["telemetry_valid"][:rows, j] == 1)
        for row in np.flatnonzero(valid):
            compared += 1
            unequal += not old.bitwise_equal(left["telemetry"][row:row + 1, i], right["telemetry"][row:row + 1, j])
        unavailable += int(rows - valid.sum())
    return {"common_paths": common, "compared_values": compared, "unequal_values": unequal,
            "unavailable_common_values": unavailable, "equal_where_valid": unequal == 0,
            "observed_through_steps": rows - 1}


def _historical(case, values, reference_case, reference_arrays):
    if case["arm"] != "as_shipped":
        return {}
    parity = {}
    for label, data in values.items():
        if reference_case["runs"].get(label, {}).get("status") == "unattempted":
            continue
        key = label + "__observations"
        if key not in reference_arrays or not len(reference_arrays[key]) or not len(data["observations"]):
            continue
        reference = {name: reference_arrays[f"{label}__{name}"] for name in old.ARRAY_NAMES}
        steps = min(len(reference["observations"]), len(data["observations"])) - 1
        equal = old._same_prefix(data, reference, steps)
        _require(equal, f"historical as-shipped core prefix differs: {label}")
        parity[label] = {"compared_steps": steps, "equal": equal}
    return parity


def _first_detection(magnitudes, threshold, observed, maximum, dt):
    crossed = np.flatnonzero(np.asarray(magnitudes) > threshold)
    return {"time_s": float((crossed[0] + 1) * dt) if len(crossed) else None,
            "observed_through_s": float(observed * dt),
            "right_censored": bool(not len(crossed) and observed < maximum)}


def _response(index, name, values, runs, origin, maximum, matched_steps, prefix_steps, execution, thresholds, dt):
    lower, upper = f"c{index:03d}_lower", f"c{index:03d}_upper"
    observed = 0
    if (execution is None and all(label in values for label in ("factual", lower, upper))
            and prefix_steps.get(lower, -1) >= origin and prefix_steps.get(upper, -1) >= origin):
        observed = max(0, min(maximum, matched_steps - origin,
                             len(values[lower]["observations"]) - origin - 1,
                             len(values[upper]["observations"]) - origin - 1))
    row = {"command_index": index, "name": name, "observed_through_steps": observed,
           "observed_through_s": float(observed * dt), "lower_signed_deltas": [], "upper_signed_deltas": [],
           "lower": {group: [] for group in old.GROUPS}, "upper": {group: [] for group in old.GROUPS},
           "first_detection": {}, "telemetry": {"lower": {}, "upper": {}}}
    if observed:
        factual = values["factual"]["observations"][origin + 1:origin + observed + 1]
        for direction, label in (("lower", lower), ("upper", upper)):
            delta = values[label]["observations"][origin + 1:origin + observed + 1] - factual
            _require(np.isfinite(delta).all(), "finite observation subtraction overflow")
            row[f"{direction}_signed_deltas"] = delta.tolist()
            row[direction] = {group: np.max(np.abs(delta[:, columns]), axis=1).tolist()
                              for group, columns in old.GROUPS.items()}
            factual_names, names = runs["factual"]["telemetry"]["names"], runs[label]["telemetry"]["names"]
            for path in factual_names:
                if path not in names:
                    continue
                i, j = factual_names.index(path), names.index(path)
                sl = slice(origin + 1, origin + observed + 1)
                valid = (values["factual"]["telemetry_valid"][sl, i] == 1) & (values[label]["telemetry_valid"][sl, j] == 1)
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = values[label]["telemetry"][sl, j] - values["factual"]["telemetry"][sl, i]
                valid &= np.isfinite(difference)
                row["telemetry"][direction][path] = {
                    "signed_delta": [float(x) if good else None for x, good in zip(difference, valid, strict=True)],
                    "valid": valid.tolist(),
                }
    for group in old.GROUPS:
        combined = np.maximum(row["lower"][group], row["upper"][group])
        row["first_detection"][group] = _first_detection(combined, thresholds[group], observed, maximum, dt)
    times = [record["time_s"] for record in row["first_detection"].values() if record["time_s"] is not None]
    row["first_detection_any"] = {"time_s": min(times) if times else None,
                                  "observed_through_s": float(observed * dt),
                                  "right_censored": not times and observed < maximum}
    return row


def _horizon(command, steps, thresholds):
    available = command["observed_through_steps"] >= steps
    result = {"command_index": command["command_index"], "name": command["name"], "available": available,
              "cumulative_active": None, "endpoint_active": None,
              "lower_endpoint": None, "upper_endpoint": None}
    if available:
        cumulative = endpoint = False
        for direction in ("lower", "upper"):
            result[f"{direction}_endpoint"] = {group: command[direction][group][steps - 1] for group in old.GROUPS}
            cumulative |= any(max(command[direction][group][:steps]) > thresholds[group] for group in old.GROUPS)
            endpoint |= any(command[direction][group][steps - 1] > thresholds[group] for group in old.GROUPS)
        result["cumulative_active"], result["endpoint_active"] = bool(cumulative), bool(endpoint)
    return result


def _empty(case, reference_case, protocol, inherited):
    count, _ = old._contract(reference_case["runs"]["parent"], inherited)
    roster = old.expected_branches(count or 0)
    labels = ["parent", "replay", *(row["label"] for row in roster)]
    runs = case.get("runs", {})
    if runs:
        _require(set(runs) == set(labels) and case.get("branches") == roster,
                 "initial execution roster differs from reference")
        _require(runs["parent"] in ({"status": "unattempted"}, {"status": "running", "stage": "construct"})
                 and all(runs[label] == {"status": "unattempted"} for label in labels[1:]),
                 "missing noninitial execution evidence")
    else:
        _require(case.get("branches") in ([], roster), "initial execution branch roster differs")
        runs = {label: {"status": "unattempted"} for label in labels}
    names = reference_case["runs"]["parent"].get("command_names", [])
    activities = [{"command_index": i, "name": name, "available": False, "cumulative_active": None,
                   "endpoint_active": None, "lower_endpoint": None, "upper_endpoint": None}
                  for i, name in enumerate(names)]
    return {"outcome": case["execution_status"], "command_count": count, "commands": [],
            "command_contract_source": "reference" if count is not None else "unknown",
            "old_completed_cohort": reference_case["computed"]["outcome"] == "completed",
            "run_counts": old._counts(labels, runs, count is not None),
            "branch_counts": old._counts([row["label"] for row in roster], runs, count is not None),
            "readbacks": {},
            "historical_prefixes": {}, "telemetry_diagnostics": {}, "telemetry_parity": {},
            "core_equal_prefix_steps": {}, "horizon_results": {
                str(h): {"activity": activities, "available": 0, "unavailable_known": count or 0,
                         "cumulative_active": 0, "endpoint_active": 0, "weak": 0}
                for h in protocol["recording"]["horizons_steps"]}}


def validate_arm(case, arrays, protocol, inherited, entry, reference_case, reference_arrays):
    """Validate one arm and recompute censored response/diagnostic evidence."""
    _require(case.get("id") == entry["id"] == reference_case["id"], "candidate identity differs")
    _require(case.get("arm") in protocol["arms"], "unknown arm")
    execution = case.get("execution_status")
    if execution is not None:
        _require(execution in ("timeout", "worker_failure"), "unknown execution outcome")
        if execution == "timeout":
            _require(case.get("returncode") is None, "timeout has process return")
        else:
            _require(type(case.get("returncode")) is int and case["returncode"] < 0, "worker failure requires native signal")
        _require(case.get("timeout_s") == 180, "execution budget differs")
    runs = case.get("runs", {})
    if execution is not None and not arrays:
        return _empty(case, reference_case, protocol, inherited)
    _require("parent" in runs, "missing parent")
    count, bounds = old._contract(runs["parent"], inherited)
    reference_count, reference_bounds = old._contract(reference_case["runs"]["parent"], inherited)
    contract_source = "current" if count is not None else "reference" if reference_count is not None else "unknown"
    if count is not None and reference_count is not None:
        _require(count == reference_count and old.bitwise_equal(bounds, reference_bounds), "reference command contract differs")
    if count is None:
        count, bounds = reference_count, reference_bounds
    roster = old.expected_branches(count or 0)
    _require(case.get("branches") == roster, "branch roster/mapping differs")
    labels = ["parent", "replay", *(row["label"] for row in roster)]
    _require(set(runs) == set(labels), "run roster differs")
    attempted = [label for label in labels if runs[label].get("status") != "unattempted"]
    running = [label for label in labels if runs[label].get("status") == "running"]
    _require(attempted and attempted[0] == "parent", "parent not attempted")
    _require(not running or (execution is not None and running == attempted[-1:]), "invalid nonterminal run")
    metadata_only = [label for label in attempted if not any(key.startswith(label + "__") for key in arrays)]
    for label in metadata_only:
        _require(execution is not None and runs[label] == {"status": "running", "stage": "construct"}, "attempted run lacks arrays")
    saved = [label for label in attempted if label not in metadata_only]
    _require(set(arrays) == {f"{label}__{name}" for label in saved for name in (*old.ARRAY_NAMES, *EXTRA_ARRAYS)},
             "array roster differs")
    core_protocol = deepcopy(inherited)
    length, origin = protocol["recording"]["transitions"], protocol["recording"]["origin"]
    maximum = max(protocol["recording"]["horizons_steps"])
    core_protocol["recording"]["transitions"] = length
    core_protocol["replay"]["horizon_steps"] = maximum
    values, readbacks, telemetry = {}, {}, {}
    tape = common_tape(protocol, inherited, entry, reference_case, reference_arrays)
    for label in saved:
        data = {name: np.asarray(arrays[f"{label}__{name}"]) for name in (*old.ARRAY_NAMES, *EXTRA_ARRAYS)}
        _require(all(value.dtype == np.dtype("float64") for value in data.values()), "non-float64 evidence array")
        values[label] = data
        readbacks[label] = old._validate_run(label, runs[label], data, core_protocol, length,
                                            inherited_contract=(reference_count, reference_bounds))
        telemetry[label] = _telemetry(runs[label], data, protocol, case["arm"])
        if len(data["commands"]):
            _require(tape is not None, "invented command baseline")
            expected = tape.copy()
            branch = next((row for row in roster if row["label"] == label), None)
            if branch is not None and branch["command_index"] is not None:
                index = branch["command_index"]
                bound = reference_bounds[index, int(branch["direction"] == "upper")]
                expected[origin:, index] += .1 * (bound - expected[origin:, index])
            _require(old.bitwise_equal(expected, data["commands"]), f"{label} common intended tape differs")
    _require("parent" in values, "parent snapshot missing")
    admitted = len(values["parent"]["observations"]) > origin
    expected_attempts = ["parent"] + (["replay"] if tape is not None else []) + ([row["label"] for row in roster] if admitted else [])
    if execution is None:
        _require(attempted == expected_attempts, "branch/replay admission differs")
    else:
        _require(attempted == expected_attempts[:len(attempted)], "interrupted run order differs")
    historical = _historical(case, values, reference_case, reference_arrays)
    prefixes, telemetry_parity = {}, {}
    for label in saved:
        if label == "parent":
            continue
        prefixes[label] = _prefix_steps(values["parent"], values[label])
        parity_steps = min(len(values["parent"]["observations"]), len(values[label]["observations"])) - 1
        if label not in ("replay", "factual"):
            parity_steps = min(parity_steps, origin)
        telemetry_parity[label] = _telemetry_parity(values["parent"], values[label], runs["parent"], runs[label], max(0, parity_steps))
    matched_steps = min(prefixes.get("replay", -1), prefixes.get("factual", -1))
    thresholds = inherited["replay"]["activity_thresholds"]
    dt = inherited["recording"]["dt_s"]
    names = runs["parent"].get("command_names") or reference_case["runs"]["parent"].get("command_names", [])
    commands = [_response(i, name, values, runs, origin, maximum, matched_steps, prefixes, execution, thresholds, dt)
                for i, name in enumerate(names)]
    horizons = {}
    for steps in protocol["recording"]["horizons_steps"]:
        activity = [_horizon(command, steps, thresholds) for command in commands]
        available = sum(row["available"] for row in activity)
        horizons[str(steps)] = {"activity": activity, "available": available,
                               "unavailable_known": len(activity) - available,
                               "cumulative_active": sum(row["cumulative_active"] is True for row in activity),
                               "endpoint_active": sum(row["endpoint_active"] is True for row in activity),
                               "weak": sum(row["cumulative_active"] is False for row in activity)}
    if execution is not None:
        outcome = execution
    elif not len(values["parent"]["observations"]):
        outcome = "setup_failure"
    elif any(prefixes.get(label, -1) < min(len(values["parent"]["observations"]), len(values[label]["observations"])) - 1
             for label in ("replay", "factual") if label in values):
        outcome = "replay_failure"
    elif any(prefixes.get(row["label"], -1) < min(origin, len(values["parent"]["observations"]) - 1,
                                                len(values[row["label"]]["observations"]) - 1)
             for row in roster[1:] if row["label"] in values):
        outcome = "replay_failure"
    elif runs["parent"]["status"] != "completed":
        outcome = "parent_failure"
    elif any(runs[label]["status"] != "completed" for label in labels):
        outcome = "branch_failure"
    else:
        outcome = "completed"
    return {"outcome": outcome, "command_count": count, "command_contract_source": contract_source,
            "old_completed_cohort": reference_case["computed"]["outcome"] == "completed",
            "run_counts": old._counts(labels, runs, count is not None),
            "branch_counts": old._counts([row["label"] for row in roster], runs, count is not None),
            "readbacks": readbacks, "telemetry_diagnostics": telemetry, "telemetry_parity": telemetry_parity,
            "historical_prefixes": historical, "core_equal_prefix_steps": prefixes,
            "commands": commands, "horizon_results": horizons}


def _aggregate(rows, protocol):
    result = {"case_count": len(rows), "outcome_counts": dict(sorted(Counter(row["outcome"] for row in rows).items())),
              "known_commands": sum(row["command_count"] or 0 for row in rows),
              "unknown_command_count_cases": sum(row["command_count"] is None for row in rows), "horizons": {}}
    for key in ("run_counts", "branch_counts"):
        known = [row for row in rows if row[key]["planned"] is not None]
        unknown = [row for row in rows if row[key]["planned"] is None]
        result[key] = {"known_plan": {name: sum(row[key][name] for row in known)
                                    for name in ("planned", "attempted", "completed", "unattempted")},
                       "unknown_plan_cases": len(unknown),
                       "unknown_plan_attempted": sum(row[key]["attempted"] for row in unknown),
                       "unknown_plan_completed": sum(row[key]["completed"] for row in unknown)}
    for h in protocol["recording"]["horizons_steps"]:
        result["horizons"][str(h)] = {name: sum(row["horizon_results"][str(h)][name] for row in rows)
                                      for name in ("available", "unavailable_known", "cumulative_active", "endpoint_active", "weak")}
    return result


def _paired(cases, protocol):
    result = {}
    for h in protocol["recording"]["horizons_steps"]:
        paired = {}
        for metric in ("cumulative_active", "endpoint_active"):
            transitions = {f"{a}_to_{b}": 0 for a in ("active", "weak", "unavailable") for b in ("active", "weak", "unavailable")}
            unknown = 0
            for left, right in cases:
                left_rows = left["horizon_results"][str(h)]["activity"]
                right_rows = right["horizon_results"][str(h)]["activity"]
                if left["command_count"] is None or right["command_count"] is None:
                    unknown += 1
                width = max(len(left_rows), len(right_rows))
                for index in range(width):
                    def state(rows, index=index, metric=metric):
                        if index >= len(rows) or not rows[index]["available"]:
                            return "unavailable"
                        return "active" if rows[index][metric] else "weak"
                    transitions[f"{state(left_rows)}_to_{state(right_rows)}"] += 1
            matched = sum(transitions[f"{a}_to_{b}"] for a in ("active", "weak") for b in ("active", "weak"))
            paired[metric] = {"transitions": transitions, "matched_available": matched,
                              "gain": transitions["weak_to_active"], "loss": transitions["active_to_weak"],
                              "unchanged_active": transitions["active_to_active"], "unchanged_weak": transitions["weak_to_weak"],
                              "unpaired_known_slots": sum(transitions.values()) - matched,
                              "cases_with_unknown_command_count": unknown}
        result[str(h)] = paired
    return result


def summarize(cases, inventory, protocol):
    """Two-arm accounting, fixed historical cohort, and full availability table."""
    ids = [entry["id"] for entry in inventory["entries"]]
    arms = protocol["arms"]
    expected = {(entry_id, arm) for entry_id in ids for arm in arms}
    _require(len(ids) == len(set(ids)) and len(cases) == len(expected)
             and {(case["id"], case["arm"]) for case in cases} == expected, "candidate/arm inventory accounting differs")
    by_key = {(case["id"], case["arm"]): case["computed"] for case in cases}
    cohort = []
    for entry_id in ids:
        values = [by_key[(entry_id, arm)] for arm in arms]
        _require(len({row["old_completed_cohort"] for row in values}) == 1, "historical cohort differs across arms")
        if values[0]["old_completed_cohort"]:
            cohort.append(entry_id)
    results = {"candidate_count": len(ids), "arm_case_count": len(cases),
               "directory_count": len(inventory["directories"]),
               "directories": [{"directory": directory,
                                "candidates": [entry_id for entry_id in ids if entry_id.startswith(directory + "/")]}
                               for directory in inventory["directories"]],
               "root_candidates": [entry_id for entry_id in ids if "/" not in entry_id],
               "old_completed_cohort": cohort}
    for scope, selected in (("all_candidates", ids), ("old_completed", cohort)):
        results[scope] = {"arms": {arm: _aggregate([by_key[(entry_id, arm)] for entry_id in selected], protocol) for arm in arms},
                          "paired_horizons": _paired([(by_key[(entry_id, arms[0])], by_key[(entry_id, arms[1])]) for entry_id in selected], protocol)}
    return results
