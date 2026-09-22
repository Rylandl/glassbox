"""Run or verify the frozen causal stream experiment; verification never fits."""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
from collect_throw import (
    authenticate,
    binding,
    check_source,
    checkpoint,
    seal,
)
from run_dart import ROOT, Journal, write
from scipy.spatial.transform import Rotation
from verify_baseline import arrays, digest, exact, observed, read, require

PROTOCOL = ROOT / "docs/harness/online-fit-v8.json"

COUNTERS = (
    "observations",
    "optimizer_steps",
    "gradient_calls",
    "objective_calls",
    "accepted_proposals",
    "cg_iterations",
    "curvature_calls",
    "conditioning_calls",
)
BACKTRACKING_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def proposal_decision(record):
    """Replay a saved scalar decision with NumPy only, never a learner call."""
    require(
        isinstance(record, dict)
        and set(record)
        == {
            "current_loss",
            "conditioning_finite",
            "direction_finite",
            "trust_shrink",
            "trials",
            "selected_alpha",
            "trial_evaluations",
        },
        "proposal evidence schema differs",
    )

    def numeric(value):
        return type(value) in (float, int) and np.isfinite(value)

    def optional(value):
        return value is None or numeric(value)

    current, shrink = record["current_loss"], record["trust_shrink"]
    require(
        optional(current)
        and optional(shrink)
        and (current is None or current >= 0)
        and type(record["conditioning_finite"]) is bool
        and type(record["direction_finite"]) is bool
        and (shrink is None or 0 <= shrink <= 1),
        "proposal header differs",
    )
    require(
        not record["direction_finite"] or (numeric(current) and numeric(shrink)),
        "finite proposal header differs",
    )
    trials = record["trials"]
    require(
        isinstance(trials, list)
        and 1 <= len(trials) <= 5
        and type(record["trial_evaluations"]) is int
        and record["trial_evaluations"] == len(trials)
        and numeric(record["selected_alpha"]),
        "proposal trial count differs",
    )
    accepted, gain = False, -np.inf
    for index, trial in enumerate(trials):
        require(
            isinstance(trial, dict)
            and set(trial) == {"alpha", "loss", "predicted_reduction", "finite"}
            and numeric(trial["alpha"])
            and trial["alpha"] == BACKTRACKING_ALPHAS[index]
            and optional(trial["loss"])
            and (trial["loss"] is None or trial["loss"] >= 0)
            and optional(trial["predicted_reduction"])
            and type(trial["finite"]) is bool,
            "proposal trial evidence differs",
        )
        loss, predicted = trial["loss"], trial["predicted_reduction"]
        require(
            not trial["finite"] or (numeric(loss) and numeric(predicted)),
            "finite trial evidence differs",
        )
        eligible = (
            record["conditioning_finite"]
            and record["direction_finite"]
            and trial["finite"]
            and predicted > 0
        )
        if eligible:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                gain = float(np.float64(current - loss) / predicted)
        else:
            gain = -np.inf
        accepted = bool(eligible and loss < current and gain >= 0.1)
        require(not accepted or index == len(trials) - 1, "trial followed acceptance")
    require(
        (accepted and record["selected_alpha"] == trials[-1]["alpha"])
        or (not accepted and len(trials) == 5 and record["selected_alpha"] == 0),
        "first acceptable proposal selection differs",
    )
    return dict(accepted=accepted, gain=gain, trial_evaluations=len(trials))


def proposal_transition(before, after):
    """Validate one saved observation's work, acceptance and damping decision."""
    result = proposal_decision(after["last_proposal"])
    require(
        after["objective_calls"] - before["objective_calls"]
        == 1 + result["trial_evaluations"]
        and after["accepted_proposals"] - before["accepted_proposals"]
        == int(result["accepted"]),
        "backtracking observation accounting differs",
    )
    damping = before["damping"]
    if not result["accepted"] or result["gain"] < 0.25:
        damping *= 4.0
    elif result["gain"] > 0.75:
        damping *= 0.5
    require(
        after["damping"] == float(np.clip(damping, 1e-8, 1e8)),
        "backtracking damping differs",
    )
    return result


def work_summary(data):
    mask = data["assimilated"]
    trials, alpha = data["trial_evaluations"][mask], data["selected_alpha"][mask]
    count = int(mask.sum())
    return dict(
        observations=count,
        scheduled_cg_iterations=16 * count,
        gradient_calls=count,
        current_objective_calls=count,
        trial_objective_calls=int(trials.sum()),
        total_objective_calls=count + int(trials.sum()),
        trial_count_histogram={str(n): int((trials == n).sum()) for n in range(1, 6)},
        selected_alpha_histogram={
            str(value): int((alpha == value).sum())
            for value in (0.0, *BACKTRACKING_ALPHAS)
        },
    )


def fingerprint(model):
    value = model.fingerprint
    return value() if callable(value) else value


def kinematic(state, dt):
    result = np.array(state, dtype=np.float64, copy=True)
    result[6:] = (
        state[6:].reshape(3, 3) @ Rotation.from_rotvec(state[3:6] * dt).as_matrix()
    ).ravel()
    return result


def residuals(prediction, truth):
    error = prediction.astype(np.float64) - truth
    angles = np.full(len(truth), np.nan)
    finite = np.isfinite(prediction).all(axis=1) & np.isfinite(truth).all(axis=1)
    if finite.any():
        relative = np.swapaxes(
            truth[finite, 6:].reshape(-1, 3, 3), -1, -2
        ) @ prediction[finite, 6:].reshape(-1, 3, 3)
        angles[finite] = Rotation.from_matrix(relative).magnitude()
    return error, angles


def metrics(prediction, truth):
    if (
        not len(truth)
        or not np.isfinite(prediction).all()
        or not np.isfinite(truth).all()
    ):
        return None
    error, angles = residuals(prediction, truth)
    return dict(
        velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        body_rate_rmse_rad_s=float(
            np.sqrt(np.mean(np.sum(error[:, 3:6] ** 2, axis=1)))
        ),
        orientation_rmse_rad=float(np.sqrt(np.mean(angles**2))),
    )


def latency(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    warm = values[1:]
    warm = warm[np.isfinite(warm)]
    return dict(
        count=len(finite),
        first_s=float(values[0]) if len(values) and np.isfinite(values[0]) else None,
        warm_median_s=float(np.median(warm)) if len(warm) else None,
        warm_p95_s=float(np.quantile(warm, 0.95)) if len(warm) else None,
        warm_max_s=float(np.max(warm)) if len(warm) else None,
    )


def ratio(a, b):
    return {key: a[key] / max(b[key], 1e-9) for key in a} if a and b else None


def summarize(data, info, reference=None, protocol=None, stream=None):
    summaries = {
        arm: metrics(data[arm], data["truth"])
        for arm in ("candidate", "frozen", "kinematic")
    }
    if reference is not None:
        summaries["reference"] = metrics(reference["candidate"], data["truth"])
    result = dict(
        id=info["id"],
        family=info["family"],
        expected=len(data["origin"]),
        predicted=int(data["predicted"].sum()),
        assimilated=int(data["assimilated"].sum()),
        metrics=summaries,
        ratios=ratio(summaries["candidate"], summaries["frozen"]),
        kinematic_ratios=ratio(summaries["candidate"], summaries["kinematic"]),
        latency={
            name: latency(data[name + "_s"])
            for name in ("predict", "frozen_predict", "observe")
        },
        complete=bool(
            len(data["origin"])
            and data["predicted"].all()
            and data["assimilated"].all()
            and all(summaries.values())
            and info["status"] == "complete"
            and info.get("collection", {}).get("status", "complete") == "complete"
        ),
        collection=info.get("collection"),
        prefix=info["prefix"],
        startup_s={
            name: info.get(name + "_startup_s") for name in ("candidate", "frozen")
        },
    )
    if reference is not None:
        result["reference_ratios"] = ratio(
            summaries["candidate"], summaries["reference"]
        )
    if info.get("change_at_s") is not None:
        change = info["change_at_s"]
        result["adaptation"] = {}
        for label, mask in (
            (
                "first_second",
                (data["time_s"] >= change) & (data["time_s"] < change + 1),
            ),
            ("after_first_second", data["time_s"] >= change + 1),
        ):
            part = {
                arm: metrics(data[arm][mask], data["truth"][mask])
                for arm in ("candidate", "frozen", "kinematic")
            }
            if reference is not None:
                part["reference"] = metrics(
                    reference["candidate"][mask], data["truth"][mask]
                )
            result["adaptation"][label] = dict(
                count=int(mask.sum()),
                metrics=part,
                ratios=ratio(part["candidate"], part["frozen"]),
            )
            if reference is not None:
                result["adaptation"][label]["reference_ratios"] = ratio(
                    part["candidate"], part["reference"]
                )
            result["complete"] &= bool(mask.any())
    support = data["motion_support_ratio"]
    result["support"] = dict(
        observed_rows=int(np.isfinite(support).all(axis=1).sum()),
        outside_bootstrap_motion_support=int(np.any(support > 1, axis=1).sum()),
        maximum_motion_ratio=float(np.nanmax(support))
        if np.isfinite(support).any()
        else None,
        maximum_abs_command_z=float(np.nanmax(np.abs(data["command_z"])))
        if np.isfinite(data["command_z"]).any()
        else None,
    )
    if protocol is not None and protocol["id"] in (
        "online-fit-v5",
        "online-fit-v6",
        "online-fit-v7",
        "online-fit-v8",
    ):
        require(stream is not None, "prospective diagnostics require observed origins")
        origins = observed(stream["states"][data["origin"]])
        forecasts = {arm: data[arm] for arm in ("candidate", "frozen", "kinematic")}
        if reference is not None:
            forecasts["reference"] = reference["candidate"]
        result["diagnostics"] = {
            arm: forecast_diagnostics(prediction, data["truth"], origins, info["dt_s"])
            for arm, prediction in forecasts.items()
        }
        a, b = (
            result["diagnostics"]["candidate"],
            result["diagnostics"].get("reference"),
        )
        result["robustness_ratios"] = (
            dict(
                velocity_tail=a["tail"]["velocity_m_s"]["upper_decile_rmse"]
                / max(b["tail"]["velocity_m_s"]["upper_decile_rmse"], 1e-9),
                rate_tail=a["tail"]["body_rate_rad_s"]["upper_decile_rmse"]
                / max(b["tail"]["body_rate_rad_s"]["upper_decile_rmse"], 1e-9),
                orientation=result["reference_ratios"]["orientation_rmse_rad"],
                rotation_rate=a["rotation_rate"]["truth_relative_rmse_rad_s"]
                / max(b["rotation_rate"]["truth_relative_rmse_rad_s"], 1e-9),
            )
            if a is not None and b is not None
            else None
        )
    if protocol is not None and protocol["id"] in (
        "online-fit-v6",
        "online-fit-v7",
        "online-fit-v8",
    ):
        result["endpoint_objectives"] = info.get("endpoint_objectives")
    if protocol is not None and protocol["id"] in ("online-fit-v7", "online-fit-v8"):
        result["causal_captures"] = info.get("causal_captures", [])
    if protocol is not None and protocol["id"] == "online-fit-v8":
        result["work"] = work_summary(data)
    return result


def forecast_diagnostics(prediction, truth, origins, dt_s):
    """Saved-data tail and sampled rotation/rate diagnostics; no model calls."""
    if not len(truth) or not all(
        np.isfinite(a).all() for a in (prediction, truth, origins)
    ):
        return None
    residual, angle = residuals(prediction, truth)
    tail = {}
    for name, errors in (
        ("velocity_m_s", np.linalg.norm(residual[:, :3], axis=-1)),
        ("body_rate_rad_s", np.linalg.norm(residual[:, 3:6], axis=-1)),
        ("orientation_rad", angle),
    ):
        descending = np.sort(errors)[::-1]
        count = int(np.ceil(0.1 * len(errors)))
        sse = float(np.sum(errors**2))
        tail[name] = dict(
            median=float(np.quantile(errors, 0.5, method="linear")),
            p95=float(np.quantile(errors, 0.95, method="linear")),
            p99=float(np.quantile(errors, 0.99, method="linear")),
            maximum=float(descending[0]),
            largest_five_squared_error_share=float(np.sum(descending[:5] ** 2) / sse)
            if sse
            else 0.0,
            upper_decile_count=count,
            upper_decile_rmse=float(np.sqrt(np.mean(descending[:count] ** 2))),
        )

    def defect(next_states):
        relative = origins[:, 6:].reshape(-1, 3, 3).transpose(0, 2, 1) @ next_states[
            :, 6:
        ].reshape(-1, 3, 3)
        return Rotation.from_matrix(relative).as_rotvec() / dt_s - 0.5 * (
            origins[:, 3:6] + next_states[:, 3:6]
        )

    actual, predicted = defect(truth), defect(prediction)
    return dict(
        count=len(truth),
        tail=tail,
        rotation_rate=dict(
            truth_relative_rmse_rad_s=float(
                np.sqrt(np.mean(np.sum((predicted - actual) ** 2, axis=-1)))
            ),
            predicted_defect_rmse_rad_s=float(
                np.sqrt(np.mean(np.sum(predicted**2, axis=-1)))
            ),
            truth_defect_rmse_rad_s=float(np.sqrt(np.mean(np.sum(actual**2, axis=-1)))),
        ),
    )


def geometric(values):
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return (
        0.0
        if any(value == 0 for value in values)
        else float(np.exp(np.mean(np.log(values))))
    )


def comparison_aggregate(cases, ratio_key):
    primary = ("velocity_rmse_m_s", "body_rate_rmse_rad_s")
    families = {}
    for family in ("quad", "fixedwing"):
        chosen = [case for case in cases if case["family"] == family]
        per_metric = {
            key: geometric(
                [case[ratio_key][key] if case[ratio_key] else None for case in chosen]
            )
            if chosen
            else None
            for key in primary
        }
        families[family] = dict(
            metric_ratios=per_metric, aggregate=geometric(list(per_metric.values()))
        )
    return dict(
        families=families,
        aggregate_ratio=geometric([row["aggregate"] for row in families.values()]),
    )


def aggregate(cases, protocol=None):
    protocol = read(PROTOCOL) if protocol is None else protocol
    paired = protocol["id"] in (
        "online-fit-v3",
        "online-fit-v4",
        "online-fit-v5",
        "online-fit-v6",
        "online-fit-v7",
        "online-fit-v8",
    )
    comparison = comparison_aggregate(cases, "reference_ratios" if paired else "ratios")
    total, families = comparison["aggregate_ratio"], comparison["families"]
    expected = {case["id"]: "quad" for case in protocol["streams"]["quad"]["cases"]}
    expected.update(
        {
            f"fixedwing-{80 + i}": "fixedwing"
            for i in range(len(protocol["streams"]["fixedwing"]["recordings"]))
        }
    )
    complete = (
        len(cases) == len(expected)
        and {case["id"]: case["family"] for case in cases} == expected
        and all(case["complete"] for case in cases)
    )
    result = dict(
        **comparison,
        all_cases_complete=complete,
        accuracy_passed=bool(
            complete
            and total is not None
            and total <= 0.8
            and all(row["aggregate"] < 1 for row in families.values())
        ),
        real_time_target_passed=bool(
            complete
            and all(
                row["latency"]["observe"]["warm_p95_s"] is not None
                and row["latency"]["observe"]["warm_p95_s"] <= row["prefix"]["dt_s"]
                for row in cases
            )
        ),
    )
    if paired:
        result.update(
            primary_comparison="candidate / saved "
            + protocol["comparison"]["reference"]["protocol_id"],
            frozen_comparison=comparison_aggregate(cases, "ratios"),
        )
    if protocol["id"] in (
        "online-fit-v5",
        "online-fit-v6",
        "online-fit-v7",
        "online-fit-v8",
    ):
        robustness = {}
        for family in ("quad", "fixedwing"):
            selected = [case for case in cases if case["family"] == family]
            values = {
                key: geometric(
                    [
                        case["robustness_ratios"][key]
                        if case.get("robustness_ratios")
                        else None
                        for case in selected
                    ]
                )
                if selected
                else None
                for key in (
                    "velocity_tail",
                    "rate_tail",
                    "orientation",
                    "rotation_rate",
                )
            }
            robustness[family] = dict(
                values,
                upper_decile=geometric([values["velocity_tail"], values["rate_tail"]]),
            )
        ratios = {
            key: geometric([row[key] for row in robustness.values()])
            for key in ("upper_decile", "orientation", "rotation_rate")
        }
        result["robustness"] = dict(families=robustness, aggregate_ratios=ratios)
        result["robustness_passed"] = bool(
            complete
            and all(value is not None for value in ratios.values())
            and ratios["upper_decile"] < 1
            and ratios["orientation"] <= 1
            and ratios["rotation_rate"] < 1
        )
    if protocol["id"] == "online-fit-v8":
        scalar = (
            "observations",
            "scheduled_cg_iterations",
            "gradient_calls",
            "current_objective_calls",
            "trial_objective_calls",
            "total_objective_calls",
        )
        result["work"] = {
            key: sum(case["work"][key] for case in cases) for key in scalar
        }
        for key, bins in (
            ("trial_count_histogram", [str(n) for n in range(1, 6)]),
            ("selected_alpha_histogram", [str(a) for a in (0.0, *BACKTRACKING_ALPHAS)]),
        ):
            result["work"][key] = {
                value: sum(case["work"][key][value] for case in cases) for value in bins
            }
    return result


def counter_names(protocol):
    if protocol["id"] == "online-fit-v1":
        return COUNTERS[:5]
    return (
        COUNTERS
        if protocol["id"]
        in (
            "online-fit-v3",
            "online-fit-v4",
            "online-fit-v5",
            "online-fit-v6",
            "online-fit-v7",
            "online-fit-v8",
        )
        else COUNTERS[:7]
    )


def dynamic_normalizers(protocol):
    allowed = protocol["candidate"].get("dynamic_normalizers", [])
    require(
        allowed
        == (
            ["feature_scale", "quadratic_scale", "output_scale"]
            if protocol["id"]
            in (
                "online-fit-v3",
                "online-fit-v4",
                "online-fit-v5",
                "online-fit-v6",
                "online-fit-v7",
                "online-fit-v8",
            )
            else []
        ),
        "dynamic normalization contract differs",
    )
    return allowed


def check_normalizers(initial, current, protocol, previous=None, accepted=True):
    """Only the frozen allowlist may grow, and only with an accepted proposal."""
    allowed = dynamic_normalizers(protocol)
    require(set(initial) == set(current), "normalization inventory differs")
    previous = initial if previous is None else previous
    for key, value in current.items():
        if key == "feature_scale" and key in allowed:
            exact(value[-8:], initial[key][-8:], "fixed hidden feature normalization")
        if key not in allowed or not accepted:
            exact(value, previous[key], "fixed normalization " + key)
        else:
            require(
                value.dtype == initial[key].dtype
                and value.shape == initial[key].shape
                and np.isfinite(value).all()
                and np.all(value > 0)
                and np.all(value >= previous[key]),
                "nonfinite or decreasing dynamic normalization " + key,
            )


def paired_exact(actual, expected, label):
    exact(actual, expected, label)
    require(
        np.asarray(actual).tobytes() == np.asarray(expected).tobytes(),
        label + ": bytes differ",
    )


def paired_inputs(stream, reference):
    require(set(stream) == set(reference), "reference input inventory differs")
    for key in stream:
        paired_exact(stream[key], reference[key], "reference input " + key)


def paired_case(data, info, reference, reference_info):
    for key in (
        "id",
        "family",
        "opaque_id",
        "dt_s",
        "begin",
        "first",
        "ordered_commands",
    ):
        require(
            info[key] == reference_info[key], "reference case identity differs: " + key
        )
    for key in ("origin", "time_s"):
        paired_exact(data[key], reference[key], "reference " + key)
    for key, mask in (
        ("truth", data["revealed"]),
        ("frozen", data["predicted"]),
        ("kinematic", data["predicted"]),
    ):
        paired_exact(data[key][mask], reference[key][mask], "reference " + key)


def paired_initial(initial, reference):
    for prefix, values in (("param_", initial.params), ("norm_", initial.norms)):
        for key, value in values.items():
            paired_exact(
                value, reference[prefix + key], "reference initialization " + key
            )
    require(
        {key for key in reference if key.startswith(("param_", "norm_"))}
        == {"param_" + key for key in initial.params}
        | {"norm_" + key for key in initial.norms},
        "reference initial core inventory differs",
    )


def reference_contract(reference, protocol):
    wanted = protocol["comparison"]["reference"]
    require(
        wanted["protocol_id"]
        == {
            "online-fit-v3": "online-fit-v2",
            "online-fit-v4": "online-fit-v2",
            "online-fit-v5": "online-fit-v4",
            "online-fit-v6": "online-fit-v4",
            "online-fit-v7": "online-fit-v6",
            "online-fit-v8": "online-fit-v6",
        }.get(protocol["id"]),
        "reference protocol contract differs",
    )
    authenticate(reference, wanted["manifest_sha256"])
    require(
        read(reference / "protocol.json")["id"] == wanted["protocol_id"],
        "reference protocol differs",
    )
    result = verify(reference, wanted["manifest_sha256"])
    require(
        read(reference / "summary.json")["all_cases_complete"],
        "reference cases incomplete",
    )
    return result


def make_data(rows, times, commands, protocol=None):
    n, m = len(rows), commands.shape[1]
    data = dict(
        origin=rows,
        time_s=times[rows],
        candidate=np.full((n, 15), np.nan, dtype=np.float32),
        frozen=np.full((n, 15), np.nan, dtype=np.float32),
        kinematic=np.full((n, 15), np.nan),
        truth=np.full((n, 15), np.nan),
        predicted=np.zeros(n, bool),
        revealed=np.zeros(n, bool),
        assimilated=np.zeros(n, bool),
        model_before=np.full(n, "", dtype="U64"),
        model_after=np.full(n, "", dtype="U64"),
        motion_support_ratio=np.full((n, 6), np.nan),
        command_z=np.full((n, m), np.nan),
    )
    data.update(
        {
            name + "_s": np.full(n, np.nan)
            for name in ("predict", "frozen_predict", "observe")
        }
    )
    data.update(
        {
            name + "_" + when: np.full(n, -1, np.int64)
            for name in COUNTERS
            for when in ("before", "after")
        }
    )
    if protocol is not None and protocol["id"] == "online-fit-v8":
        data.update(
            trial_evaluations=np.full(n, -1, np.int64),
            selected_alpha=np.full(n, np.nan, np.float64),
        )
    return data


def capture_origins(protocol, case_id):
    if protocol["id"] not in ("online-fit-v7", "online-fit-v8"):
        return []
    chosen = protocol["causal_captures"]["origins"].get(case_id, [])
    require(chosen == sorted(set(chosen)), "capture declaration differs")
    return chosen


def verify_causal_captures(case, data, stream, info, protocol):
    """Authenticate causal snapshot arrays against the tape; never load a model."""
    from glassbox._learner_arrays import array_fingerprint, load_arrays

    expected = capture_origins(protocol, info["id"])
    found = sorted(int(path.name) for path in case.iterdir() if path.is_dir())
    require(set(found) <= set(expected), "unexpected causal capture")
    if info["status"] == "complete":
        require(found == expected, "missing successful causal capture")
    reports = {
        row["index"]: row["report"]
        for row in (
            json.loads(line)
            for line in (case / "events.jsonl").read_text().splitlines()
        )
        if row["phase"] == "predicted"
    }
    initial = None
    completed = []
    for origin in found:
        selected = np.flatnonzero(data["origin"] == origin)
        require(len(selected) == 1, "capture outside scored origins")
        index = int(selected[0])
        path = case / str(origin)
        if not (path / "session.npz").exists():
            require(info["status"] == "failed", "missing captured session")
            continue
        values = session_arrays(
            path / "session.npz",
            info,
            origin - info["first"],
            data["model_before"][index],
            protocol,
            reports.get(origin),
        )
        meta, _ = load_arrays(path / "session.npz")
        for name in counter_names(protocol):
            count = (
                origin - info["first"]
                if name == "observations"
                else meta["counts"][name]
            )
            require(
                count == data[name + "_before"][index], "capture before counter differs"
            )
        if initial is None:
            initial = load_arrays(case / "initial-online.npz")
        first_meta, first_arrays = initial
        for name in ("contract", "initial_cursor", "initial_count", "horizon"):
            require(meta[name] == first_meta[name], "capture initial identity differs")
        for name in first_arrays:
            if name == "scale" or name.startswith("bootstrap_"):
                paired_exact(
                    values[name], first_arrays[name], "capture fixed cache " + name
                )
        h, horizon = meta["model"]["history_steps"], meta["horizon"]
        states, commands = observed(stream["states"]), stream["commands"]
        paired_exact(
            values["tail_states"],
            states[origin - h - horizon : origin + 1],
            "capture causal state tail",
        )
        paired_exact(
            values["tail_inputs"],
            commands[origin - h - horizon : origin],
            "capture causal command tail",
        )
        count = min(32, origin - info["first"])
        starts = range(origin - horizon + 1 - count, origin - horizon + 1)
        windows = (
            dict(
                past_states=np.stack([states[k - h : k + 1] for k in starts]),
                past_inputs=np.stack([commands[k - h : k] for k in starts]),
                future_inputs=np.stack([commands[k : k + horizon] for k in starts]),
                future_states=np.stack(
                    [states[k + 1 : k + horizon + 1] for k in starts]
                ),
            )
            if count
            else {
                name: first_arrays["bootstrap_" + name][:0]
                for name in (
                    "past_states",
                    "past_inputs",
                    "future_inputs",
                    "future_states",
                )
            }
        )
        for name, wanted in windows.items():
            paired_exact(
                values["recent_" + name], wanted, "capture recent cache " + name
            )
        if not all((path / name).exists() for name in ("context.npz", "capture.json")):
            require(info["status"] == "failed", "missing successful capture context")
            continue
        context = arrays(path / "context.npz")
        wanted = dict(
            origin=np.asarray(origin, dtype=np.int64),
            past_states=states[origin - h : origin + 1],
            past_inputs=commands[origin - h : origin],
            command=commands[origin],
            truth=data["truth"][index],
            recorded_prediction=data["candidate"][index],
        )
        require(set(context) == set(wanted), "capture context inventory differs")
        require(
            data["predicted"][index] and data["revealed"][index],
            "capture precedes prediction or revelation",
        )
        for name, value in wanted.items():
            paired_exact(context[name], value, "capture context " + name)
        paired_exact(context["truth"], states[origin + 1], "capture revealed truth")
        require(
            read(path / "capture.json")
            == dict(
                origin=origin,
                time_s=float(stream["time_s"][origin]),
                session_fingerprint=array_fingerprint(meta, values),
                model_fingerprint=data["model_before"][index],
                report=reports[origin],
            ),
            "capture full identity differs",
        )
        completed.append(origin)
    require(
        completed == info["causal_captures"], "capture completion inventory differs"
    )
    if info["status"] == "complete":
        require(completed == expected, "incomplete successful causal captures")
    return completed


def evaluate_case(source, output, info, protocol=None, reference=None):
    import jax

    from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

    protocol = read(PROTOCOL) if protocol is None else protocol
    counters = counter_names(protocol)
    output.mkdir()
    write(output / "attempt.json", info)
    dt, first, begin = info["dt_s"], info["first"], info["begin"]
    stream = arrays(source)
    x, u, times = stream["states"], stream["commands"], stream["time_s"]
    rows = np.arange(first, max(first, len(u)), dtype=np.int64)
    data = make_data(rows, times, u, protocol)
    info = dict(
        info,
        prefix=dict(start_row=begin, next_command_row=first, dt_s=dt),
        status="complete",
    )
    journal, online, frozen = Journal(output), None, None
    captures = capture_origins(protocol, info["id"])
    if protocol["id"] in ("online-fit-v7", "online-fit-v8"):
        info["causal_captures"] = []

    def save():
        for arm in ("candidate", "frozen", "kinematic"):
            data[arm + "_residual"], data[arm + "_orientation_error_rad"] = residuals(
                data[arm], data["truth"]
            )
        checkpoint(output / "predictions.npz", **data)

    try:
        require(
            x.shape == (len(u) + 1, 13) and times.shape == (len(x),),
            "unaligned input tape",
        )
        require(
            np.allclose(times, np.arange(len(x)) * dt, rtol=0, atol=1e-10),
            "input sample grid",
        )
        require(len(u) > first, "stream cannot reach the fixed startup prefix")
        require(set(captures) <= set(rows), "capture outside scored origins")
        require(np.isfinite(x).all() and np.isfinite(u).all(), "nonfinite input tape")
        history = observed(x[begin : first + 1])
        issued = u[begin:first].copy()
        prefix = SequenceCollection(
            (SequenceSegment(info["opaque_id"], "prefix", history, issued, dt, begin),),
            info["opaque_id"],
            STATE_CHANNELS,
            tuple(info["ordered_commands"]),
        )
        init = time.perf_counter()
        online = OnlineFit(prefix)
        jax.block_until_ready(online.model.params)
        info["candidate_startup_s"] = time.perf_counter() - init
        init = time.perf_counter()
        frozen = OnlineFit(prefix)
        jax.block_until_ready(frozen.model.params)
        info["frozen_startup_s"] = time.perf_counter() - init
        require(online.cursor == frozen.cursor == first, "initial cursor differs")
        initial = online.model
        require(
            fingerprint(initial) == fingerprint(frozen.model),
            "cold initial models differ",
        )
        info["initial_model_fingerprint"] = fingerprint(initial)
        if reference is not None:
            paired_initial(initial, arrays(reference / "initial-online.npz"))
            require(
                info["initial_model_fingerprint"]
                == read(reference / "case.json")["initial_model_fingerprint"],
                "reference initial fingerprint differs",
            )
        for key in initial.params:
            exact(
                initial.params[key],
                frozen.model.params[key],
                "initial parameter " + key,
            )
        for key in initial.norms:
            exact(
                initial.norms[key],
                frozen.model.norms[key],
                "initial normalization " + key,
            )
        online.save(output / "initial-online.npz")
        frozen.save(output / "initial-frozen.npz")
        np.savez_compressed(output / "normalization.npz", **initial.norms)
        info["prefix"].update(
            command_span=np.ptp(issued, axis=0).tolist(),
            centered_command_rank=int(
                np.linalg.matrix_rank(issued - issued.mean(axis=0))
            ),
        )
        target_commands = issued[-round(0.25 / dt) :]
        info["prefix"].update(
            initialization_command_span=np.ptp(target_commands, axis=0).tolist(),
            initialization_centered_command_rank=int(
                np.linalg.matrix_rank(target_commands - target_commands.mean(axis=0))
            ),
        )
        h = initial.history_steps
        history, issued = history[-h - 1 :], issued[-h:]
        previous_norms = initial.norms
        for i, k in enumerate(rows):
            require(online.cursor == k, "causal cursor differs")
            before = online.report
            data["model_before"][i] = fingerprint(online.model)
            for key in counters:
                data[key + "_before"][i] = before[key]
            if k in captures:
                capture = output / str(int(k))
                capture.mkdir()
                session_identity = online.fingerprint()
                online.save(capture / "session.npz")
                require(
                    online.fingerprint() == session_identity,
                    "capture saving mutated session",
                )
            for arm, learner, latency_name in (
                ("candidate", online, "predict"),
                ("frozen", frozen, "frozen_predict"),
            ):
                started = time.perf_counter()
                prediction = np.asarray(learner.predict(history, issued, u[k : k + 1]))
                data[latency_name + "_s"][i] = time.perf_counter() - started
                require(
                    prediction.dtype == np.float32 and prediction.shape == (1, 15),
                    "prediction contract differs",
                )
                data[arm][i] = prediction[0]
            data["kinematic"][i] = kinematic(history[-1], dt)
            journal.event(
                dict(
                    phase="predicted",
                    index=int(k),
                    model=data["model_before"][i],
                    report=before,
                ),
                dict(
                    candidate=data["candidate"][i],
                    frozen=data["frozen"][i],
                    kinematic=data["kinematic"][i],
                    command=u[k],
                ),
            )
            require(
                np.isfinite(data["candidate"][i]).all()
                and np.isfinite(data["frozen"][i]).all(),
                "nonfinite prediction",
            )
            data["predicted"][i] = True
            # The next observation is first decoded and delivered after durable prediction.
            truth = observed(x[k + 1 : k + 2])[0]
            data["truth"][i] = truth
            journal.event(dict(phase="revealed", index=int(k)), dict(truth=truth))
            data["revealed"][i] = True
            if k in captures:
                require(
                    online.fingerprint() == session_identity,
                    "capture prediction mutated session",
                )
                checkpoint(
                    capture / "context.npz",
                    origin=np.asarray(k, dtype=np.int64),
                    past_states=history.copy(),
                    past_inputs=issued.copy(),
                    command=u[k].copy(),
                    truth=truth,
                    recorded_prediction=data["candidate"][i].copy(),
                )
                write(
                    capture / "capture.json",
                    dict(
                        origin=int(k),
                        time_s=float(times[k]),
                        session_fingerprint=session_identity,
                        model_fingerprint=data["model_before"][i],
                        report=before,
                    ),
                )
                info["causal_captures"].append(int(k))
            started = time.perf_counter()
            online.observe(int(k), u[k], truth)
            updated = online.model
            jax.block_until_ready(updated.params)
            data["observe_s"][i] = time.perf_counter() - started
            require(online.cursor == k + 1, "observation not assimilated once")
            require(
                all(np.isfinite(v).all() for v in updated.params.values()),
                "nonfinite parameters",
            )
            check_normalizers(
                initial.norms,
                updated.norms,
                protocol,
                previous_norms,
                online.report["accepted_proposals"] > before["accepted_proposals"],
            )
            previous_norms = updated.norms
            data["assimilated"][i] = True
            data["model_after"][i] = fingerprint(updated)
            for key in counters:
                data[key + "_after"][i] = online.report[key]
            if protocol["id"] == "online-fit-v8":
                proposal_transition(before, online.report)
                decision = online.report["last_proposal"]
                data["trial_evaluations"][i] = decision["trial_evaluations"]
                data["selected_alpha"][i] = decision["selected_alpha"]
            body = np.r_[truth[6:].reshape(3, 3).T @ truth[:3], truth[3:6]]
            norm = initial.norms
            data["motion_support_ratio"][i] = np.abs(
                (body - norm["body_mean"][:6]) / norm["body_scale"][:6]
            ) / (norm["motion_bound_scale"] / 4)
            data["command_z"][i] = (u[k] - norm["input_mean"]) / norm["input_scale"]
            journal.event(
                dict(
                    phase="assimilated",
                    index=int(k),
                    model=data["model_after"][i],
                    report=online.report,
                ),
                {
                    "norm_" + key: updated.norms[key]
                    for key in dynamic_normalizers(protocol)
                },
            )
            history = np.concatenate((history[1:], truth[None]))
            issued = np.concatenate((issued[1:], u[k : k + 1]))
            if (i + 1) % 25 == 0:
                save()
            if (i + 1) % 100 == 0:
                print(
                    json.dumps(
                        dict(case=info["id"], observations=i + 1, total=len(rows))
                    ),
                    flush=True,
                )
        if protocol["id"] in ("online-fit-v7", "online-fit-v8"):
            require(
                info["causal_captures"] == captures, "missing successful causal capture"
            )
        require(
            fingerprint(frozen.model) == info["initial_model_fingerprint"],
            "frozen comparator changed",
        )
    except Exception as error:
        info.update(status="failed", error=repr(error))
    finally:
        journal.close()
        save()
        for name, learner in (("online", online), ("frozen", frozen)):
            if learner is not None:
                try:
                    learner.save(output / ("final-" + name + ".npz"))
                    info[name + "_final_report"] = learner.report
                except Exception as error:
                    info.update(status="failed", final_save_error=repr(error))
        if (
            protocol["id"] in ("online-fit-v6", "online-fit-v7", "online-fit-v8")
            and online is not None
        ):
            try:
                info["endpoint_objectives"] = save_endpoint_diagnostics(output)
            except Exception as error:
                info.update(status="failed", endpoint_diagnostic_error=repr(error))
        write(output / "case.json", info)
    reference_data = (
        arrays(reference / "predictions.npz") if reference is not None else None
    )
    if reference is not None:
        paired_case(data, info, reference_data, read(reference / "case.json"))
    result = summarize(data, info, reference_data, protocol, stream)
    write(output / "summary.json", result)
    return result


def endpoint_objective(meta, values, predictions):
    """Independent NumPy arithmetic from one saved cache/model endpoint."""
    norms = {key[5:]: value for key, value in values.items() if key.startswith("norm_")}
    require(
        meta["format"]
        in (
            "glassbox-online-fit-v6",
            "glassbox-online-fit-v7",
            "glassbox-online-fit-v8",
        ),
        "unsupported endpoint session format",
    )
    envelope = meta["format"] == "glassbox-online-fit-v7"
    role_losses, motion_squares, command_squares, command_maxima, counts = (
        [],
        [],
        [],
        [],
        {},
    )
    dt, delay = meta["model"]["dt_s"], meta["model"]["delay_steps"]
    require(
        set(predictions) == {"bootstrap", "recent"},
        "endpoint prediction inventory differs",
    )
    for role in ("bootstrap", "recent"):
        past, up, future, truth = (
            values[role + "_" + key]
            for key in ("past_states", "past_inputs", "future_inputs", "future_states")
        )
        prediction = predictions[role]
        require(
            prediction.shape == truth.shape
            and prediction.dtype == np.float64
            and np.isfinite(prediction).all(),
            "endpoint cache prediction contract differs",
        )
        counts[role] = len(past)
        if not len(past):
            continue
        residual = (prediction - truth) / values["scale"]
        groups = np.stack(
            [
                np.linalg.norm(residual[..., a:b], axis=-1)
                for a, b in ((0, 3), (3, 6), (6, 15))
            ],
            axis=-1,
        )
        role_losses.append(
            float(np.mean(np.where(groups <= 1, 0.5 * groups**2, groups - 0.5)))
        )
        if envelope:
            commands = np.concatenate((up, future), axis=1)
            command_maxima.append(
                np.max(
                    np.abs((commands - norms["input_mean"]) / norms["input_scale"]),
                    axis=(0, 1),
                )
            )
            continue
        states = np.concatenate((past, truth[:, :-1]), axis=1)[:, delay:]
        commands = np.concatenate((up, future), axis=1)[:, delay:]
        rotation = states[..., 6:].reshape(*states.shape[:-1], 3, 3)
        body = np.concatenate(
            (
                np.einsum("...ji,...j->...i", rotation, states[..., :3]),
                states[..., 3:6],
            ),
            axis=-1,
        )
        motion = (body - norms["body_mean"][:6]) / norms["body_scale"][:6]
        support = norms["motion_bound_scale"] / 4
        excess = np.maximum(np.abs(motion) - support, 0)
        bounded = np.where(
            np.abs(motion) <= support,
            motion,
            np.sign(motion) * (support + 3 * support * np.tanh(excess / (3 * support))),
        )
        motion_squares.append(np.mean(bounded**2, axis=(0, 1)))
        command_squares.append(
            np.mean(
                ((commands - norms["input_mean"]) / norms["input_scale"]) ** 2,
                axis=(0, 1),
            )
        )
    require(bool(role_losses), "endpoint has no measured cache")
    if envelope:
        issued = np.maximum(1.0, np.max(command_maxima, axis=0))
        motion = norms["motion_bound_scale"]
    else:
        issued = np.maximum(1.0, np.sqrt(np.mean(command_squares, axis=0)))
        motion = np.maximum(1.0, np.sqrt(np.mean(motion_squares, axis=0)))
    domain = np.r_[
        motion,
        1 / norms["body_scale"][6:9],
        issued,
        issued,
    ]
    left, right = np.triu_indices(len(domain))
    factors = np.where(left == right, 2.0, np.sqrt(2.0))
    physical = (
        values["param_quadratic"]
        * norms["output_scale"][None, :]
        / norms["quadratic_scale"][:, None]
    )
    scaled = (
        factors[:, None]
        * domain[left, None]
        * domain[right, None]
        * dt
        * physical
        / values["scale"][0, :6]
    )
    prior = float(0.01 / 4 * np.sum(scaled**2))
    data_loss = float(np.mean(role_losses))
    require(np.isfinite([prior, data_loss]).all(), "nonfinite endpoint objective")
    return dict(
        cache_windows=counts,
        data_loss=data_loss,
        prior_loss=prior,
        combined_loss=data_loss + prior,
        domain=domain.tolist(),
    )


def save_endpoint_diagnostics(output):
    """Read-only rollouts after all measured case timings; no new learner state."""
    import jax

    from glassbox import OnlineFit
    from glassbox._learner_arrays import load_arrays

    saved, result = {}, {}
    for endpoint in ("initial", "final"):
        path = output / (endpoint + "-online.npz")
        session = OnlineFit.load(path)
        before = session.fingerprint()
        meta, values = load_arrays(path)
        predicted = {}
        with jax.enable_x64(True):
            for role in ("bootstrap", "recent"):
                truth = values[role + "_future_states"]
                predicted[role] = (
                    np.asarray(
                        session.model.rollout(
                            *(
                                values[role + "_" + key]
                                for key in (
                                    "past_states",
                                    "past_inputs",
                                    "future_inputs",
                                )
                            )
                        )
                    )
                    if len(truth)
                    else np.empty_like(truth, dtype=np.float64)
                )
                saved[endpoint + "_" + role] = predicted[role]
        require(session.fingerprint() == before, "endpoint diagnostic mutated session")
        result[endpoint] = endpoint_objective(meta, values, predicted)
    checkpoint(output / "endpoint-predictions.npz", **saved)
    write(output / "endpoint-objectives.json", result)
    return result


def verify_endpoint_diagnostics(output):
    """Recompute losses and physical prior with zero model or optimizer calls."""
    from glassbox._learner_arrays import load_arrays

    predicted = arrays(output / "endpoint-predictions.npz")
    require(
        set(predicted)
        == {
            endpoint + "_" + role
            for endpoint in ("initial", "final")
            for role in ("bootstrap", "recent")
        },
        "endpoint prediction inventory differs",
    )
    result = {}
    for endpoint in ("initial", "final"):
        meta, values = load_arrays(output / (endpoint + "-online.npz"))
        result[endpoint] = endpoint_objective(
            meta,
            values,
            {
                role: predicted[endpoint + "_" + role]
                for role in ("bootstrap", "recent")
            },
        )
    require(
        result == read(output / "endpoint-objectives.json"),
        "endpoint objective differs",
    )
    return result


def run(collection, authority, output, protocol=PROTOCOL, reference=None):
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "attempt.json",
        dict(
            collection=str(collection),
            authority=authority,
            protocol=str(protocol),
            reference=str(reference),
        ),
    )
    try:
        p = read(protocol)
        require(
            p["id"] == "online-fit-v8",
            "unsupported candidate protocol for current learner",
        )
        require(
            reference is not None,
            "current candidate requires the authority-pinned reference pack",
        )
        reference_contract(reference, p)
        shutil.copytree(reference, output / "reference")
        reference = output / "reference"
        reference_contract(reference, p)
        shutil.copyfile(protocol, output / "protocol.json")
        bound = binding(protocol)
        require(
            Path(importlib.util.find_spec("glassbox").origin).resolve()
            == ROOT / "src/glassbox/__init__.py",
            "evaluate with the current bound Glassbox checkout",
        )
        authenticate(collection, authority)
        collection_contract(collection, authority, p, digest(protocol))
        bound["collection_manifest_sha256"] = authority
        bound["reference_manifest_sha256"] = p["comparison"]["reference"][
            "manifest_sha256"
        ]
        write(output / "binding.json", bound)
        inputs = output / "inputs"
        inputs.mkdir()
        cases = []
        for row in p["streams"]["quad"]["cases"]:
            case = dict(
                id=row["id"],
                family="quad",
                dt_s=0.01,
                begin=50,
                first=125,
                collection=read(collection / row["id"] / "report.json"),
                change_at_s=row.get("change_at_s"),
            )
            case["ordered_commands"] = case["collection"]["ordered_commands"]
            tape = arrays(collection / row["id"] / "stream.npz")
            checkpoint(inputs / (case["id"] + ".npz"), **tape)
            cases.append(case)
        for index, row in enumerate(p["streams"]["fixedwing"]["recordings"]):
            source = Path(row["path"])
            require(
                digest(source) == row["sha256"], "reserved fixed-wing recording changed"
            )
            data = arrays(source)
            channels = json.loads(str(data["spec_json"]))["channels"]
            ordered = [f"{c['name']} [{c['unit']}]" for c in channels]
            require(
                ordered == p["streams"]["fixedwing"]["ordered_commands"],
                "fixed-wing command order differs",
            )
            case = dict(
                id=f"fixedwing-{80 + index}",
                family="fixedwing",
                dt_s=0.05,
                begin=0,
                first=15,
                ordered_commands=ordered,
                source_sha256=row["sha256"],
            )
            checkpoint(
                inputs / (case["id"] + ".npz"),
                states=data["states"],
                commands=data["controls"],
                time_s=data["time_s"],
            )
            cases.append(case)
        for case in cases:
            paired_inputs(
                arrays(inputs / (case["id"] + ".npz")),
                arrays(reference / "inputs" / (case["id"] + ".npz")),
            )
        results = []
        for i, case in enumerate(cases):
            case["opaque_id"] = f"stream-{i:02d}"
            print(json.dumps(dict(case=case["id"], status="starting")), flush=True)
            results.append(
                evaluate_case(
                    inputs / (case["id"] + ".npz"),
                    output / case["id"],
                    case,
                    p,
                    reference / case["id"],
                )
            )
            print(json.dumps(dict(case=case["id"], result=results[-1])), flush=True)
        reference_contract(reference, p)
        check_source(bound)
        write(
            output / "summary.json",
            dict(
                cases=results,
                **aggregate(results, p),
                orientation_metric="Rotation.from_matrix(R_true.T @ R_pred).magnitude; proper-rotation projection",
                timing="synchronized calls; durable journal/checkpoint overhead excluded from per-call latency",
                candidate_closed_loop_trials=0,
            ),
        )
    except BaseException as error:
        write(output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        authority = seal(output, "glassbox-online-evaluation-v1")
        print(
            json.dumps(dict(output=str(output), manifest_sha256=authority)), flush=True
        )


def collection_contract(collection, authority, protocol, protocol_sha256):
    require(
        authority == protocol.get("collection_manifest_sha256", authority),
        "collection manifest differs from frozen protocol",
    )
    require(
        read(collection / "binding.json")["protocol_sha256"]
        == protocol.get("collection_protocol_sha256", protocol_sha256),
        "collection protocol differs",
    )


def accounting(report, count, protocol):
    proposals = protocol["candidate"]["proposals_per_observation"]
    require(
        report["observations"] == count
        and report["optimizer_steps"] == report["gradient_calls"] == proposals * count,
        "proposal accounting differs",
    )
    if protocol["id"] in (
        "online-fit-v2",
        "online-fit-v3",
        "online-fit-v4",
        "online-fit-v5",
        "online-fit-v6",
        "online-fit-v7",
    ):
        require(
            report["cg_iterations"] == report["curvature_calls"] == 4 * count
            and report["objective_calls"] == 2 * count
            and 1e-8 <= report["damping"] <= 1e8,
            "curvature accounting differs",
        )
    if protocol["id"] == "online-fit-v8":
        require(
            protocol["candidate"]["cg_iterations"] == 16
            and tuple(protocol["candidate"]["backtracking_alphas"])
            == BACKTRACKING_ALPHAS
            and report["cg_iterations"] == report["curvature_calls"] == 16 * count
            and type(report["objective_calls"]) is int
            and type(report["accepted_proposals"]) is int
            and 0 <= report["accepted_proposals"] <= count
            and 2 * count <= report["objective_calls"] <= 6 * count
            and 1e-8 <= report["damping"] <= 1e8,
            "backtracking curvature accounting differs",
        )
        if count:
            last = proposal_decision(report["last_proposal"])
            previous_calls = report["objective_calls"] - 1 - last["trial_evaluations"]
            require(
                2 * (count - 1) <= previous_calls <= 6 * (count - 1)
                and 0
                <= report["accepted_proposals"] - int(last["accepted"])
                <= count - 1,
                "last proposal objective accounting differs",
            )
        else:
            require(
                report["last_proposal"] is None and report["damping"] == 1.0,
                "initial proposal evidence differs",
            )

    if protocol["id"] in (
        "online-fit-v3",
        "online-fit-v4",
        "online-fit-v5",
        "online-fit-v6",
        "online-fit-v7",
        "online-fit-v8",
    ):
        require(
            report["conditioning_calls"] == count, "conditioning accounting differs"
        )


def session_arrays(path, info, count, endpoint, protocol, report=None):
    # Archive evidence is independent of the current optimizer/session loader.
    from glassbox._learner_arrays import array_fingerprint, load_arrays

    meta, values = load_arrays(path)
    core = {k: v for k, v in values.items() if k.startswith(("param_", "norm_"))}
    require(
        meta["format"] == "glassbox-" + protocol["id"]
        and meta["initial_cursor"] == info["first"]
        and meta["cursor"] == info["first"] + count
        and meta["recipe"]["proposals"]
        == protocol["candidate"]["proposals_per_observation"]
        and array_fingerprint(meta["model"], core) == endpoint,
        "session endpoint differs",
    )
    accounting(
        dict(
            meta["counts"],
            observations=count,
            damping=meta.get("damping"),
            last_proposal=meta.get("last_proposal"),
        ),
        count,
        protocol,
    )
    require(
        all(np.isfinite(value).all() for value in values.values()),
        "nonfinite saved session",
    )
    require(
        0 <= meta["counts"]["accepted_proposals"] <= meta["counts"]["optimizer_steps"],
        "accepted proposal count differs",
    )
    if report is not None:
        require(
            all(report[key] == value for key, value in meta["counts"].items()),
            "saved report counters differ",
        )
        if "damping" in meta:
            require(report["damping"] == meta["damping"], "saved damping differs")
        if protocol["id"] == "online-fit-v8":
            require(
                report["last_proposal"] == meta["last_proposal"],
                "saved proposal evidence differs",
            )
    if protocol["id"] == "online-fit-v8":
        require(
            meta["recipe"]["cg_iterations"] == 16
            and tuple(meta["recipe"]["backtracking"]["scales"]) == BACKTRACKING_ALPHAS,
            "saved backtracking recipe differs",
        )
    return values


def verify_journal(case, data, stream, info, protocol=None):
    protocol = read(PROTOCOL) if protocol is None else protocol
    counters = counter_names(protocol)
    dynamic = dynamic_normalizers(protocol)
    normalizers = {}
    if dynamic and (case / "normalization.npz").exists():
        normalizers = arrays(case / "normalization.npz")
    elif dynamic:
        require(
            info["status"] == "failed" and not data["assimilated"].any(),
            "missing normalization archive",
        )
    backtracking = protocol["id"] == "online-fit-v8"
    if backtracking:
        n = len(data["origin"])
        require(
            data["trial_evaluations"].shape == data["selected_alpha"].shape == (n,)
            and data["trial_evaluations"].dtype == np.int64
            and data["selected_alpha"].dtype == np.float64
            and np.all(data["trial_evaluations"][~data["assimilated"]] == -1)
            and np.isnan(data["selected_alpha"][~data["assimilated"]]).all(),
            "backtracking row inventory differs",
        )
    previous_accepted, previous_proposal = 0, None
    previous_report = None
    before = None
    position, phase, offset = 0, "predicted", 0
    with (
        (case / "arrays.bin").open("rb") as binary,
        (case / "events.jsonl").open() as events,
    ):
        for line in events:
            event = json.loads(line)
            require(
                position < len(data["origin"])
                and event["index"] == data["origin"][position]
                and event["phase"] == phase,
                "pre-assimilation journal order differs",
            )
            values = {}
            for key, wanted in event["arrays"].items():
                require(wanted == offset, "journal array offsets differ")
                binary.seek(offset)
                values[key] = np.load(binary, allow_pickle=False)
                offset = binary.tell()
            if phase == "predicted":
                for arm in ("candidate", "frozen", "kinematic"):
                    exact(values[arm], data[arm][position], "journal prediction")
                exact(
                    values["command"],
                    stream["commands"][event["index"]],
                    "journal issued command",
                )
                require(
                    event["model"] == data["model_before"][position],
                    "pre-assimilation fingerprint differs",
                )
                require(
                    event["model"]
                    == (
                        data["model_after"][position - 1]
                        if position
                        else info["initial_model_fingerprint"]
                    ),
                    "model revision chain differs",
                )
                for key in counters:
                    require(
                        event["report"][key] == data[key + "_before"][position],
                        "before counter differs",
                    )
                require(
                    event["report"]["cursor"] == event["index"],
                    "before cursor/counters differ",
                )
                accounting(event["report"], position, protocol)
                if backtracking:
                    require(
                        event["report"]["last_proposal"] == previous_proposal,
                        "proposal evidence chain differs",
                    )
                    if previous_report is not None:
                        require(
                            all(
                                event["report"][key] == previous_report[key]
                                for key in (*counters, "damping", "cursor")
                            ),
                            "backtracking report chain differs",
                        )
                    before = event["report"]
                    previous_report = before
                phase = "revealed"
            elif phase == "revealed":
                exact(
                    values["truth"], data["truth"][position], "journal revealed truth"
                )
                require(data["revealed"][position], "missing revelation flag")
                phase = "assimilated"
            else:
                require(
                    set(values) == {"norm_" + key for key in dynamic}
                    and data["assimilated"][position],
                    "unexpected assimilation payload",
                )
                current = dict(
                    normalizers, **{key: values["norm_" + key] for key in dynamic}
                )
                accepted = event["report"]["accepted_proposals"]
                if dynamic:
                    require(
                        accepted - previous_accepted in (0, 1),
                        "acceptance accounting differs",
                    )
                check_normalizers(
                    normalizers,
                    current,
                    protocol,
                    accepted=accepted > previous_accepted,
                )
                normalizers, previous_accepted = current, accepted
                require(
                    event["model"] == data["model_after"][position],
                    "updated fingerprint differs",
                )
                for key in counters:
                    require(
                        event["report"][key] == data[key + "_after"][position],
                        "after counter differs",
                    )
                require(
                    event["report"]["cursor"] == event["index"] + 1,
                    "assimilation accounting differs",
                )
                accounting(event["report"], position + 1, protocol)
                if backtracking:
                    result = proposal_transition(before, event["report"])
                    previous_proposal = event["report"]["last_proposal"]
                    require(
                        data["trial_evaluations"][position]
                        == result["trial_evaluations"]
                        and data["selected_alpha"][position]
                        == previous_proposal["selected_alpha"],
                        "saved backtracking row differs",
                    )
                    require(
                        result["accepted"]
                        or data["model_before"][position]
                        == data["model_after"][position],
                        "rejected proposal changed model",
                    )
                    previous_report = event["report"]
                phase, position = "predicted", position + 1
        require(
            offset == (case / "arrays.bin").stat().st_size, "unreferenced journal bytes"
        )
    if dynamic and (case / "final-online.npz").exists():
        final = arrays(case / "final-online.npz")
        for key in dynamic:
            exact(final["norm_" + key], normalizers[key], "journal final normalization")
    require(
        position == int(data["assimilated"].sum()), "journal assimilation count differs"
    )
    if backtracking and previous_report is not None:
        require(
            all(
                info["online_final_report"][key] == previous_report[key]
                for key in (*counters, "damping", "cursor", "last_proposal")
            ),
            "final backtracking report differs",
        )
    if info["status"] == "complete":
        require(
            phase == "predicted" and position == len(data["origin"]),
            "incomplete successful journal",
        )


def verify(output, authority):
    authenticate(output, authority)
    protocol = read(output / "protocol.json")
    require(
        digest(output / "protocol.json")
        == read(output / "binding.json")["protocol_sha256"],
        "saved protocol differs",
    )
    reference = None
    if protocol["id"] in (
        "online-fit-v3",
        "online-fit-v4",
        "online-fit-v5",
        "online-fit-v6",
        "online-fit-v7",
        "online-fit-v8",
    ):
        reference = output / "reference"
        reference_contract(reference, protocol)
        require(
            read(output / "binding.json")["reference_manifest_sha256"]
            == protocol["comparison"]["reference"]["manifest_sha256"],
            "bound reference authority differs",
        )
    report, results = read(output / "summary.json"), []
    for wanted in report["cases"]:
        case = output / wanted["id"]
        info, data = read(case / "case.json"), arrays(case / "predictions.npz")
        stream = arrays(output / "inputs" / (wanted["id"] + ".npz"))
        reference_data = None
        if reference is not None:
            paired_inputs(
                stream, arrays(reference / "inputs" / (wanted["id"] + ".npz"))
            )
            reference_case = reference / wanted["id"]
            reference_data = arrays(reference_case / "predictions.npz")
            paired_case(data, info, reference_data, read(reference_case / "case.json"))
        exact(
            data["origin"],
            np.arange(
                info["first"],
                max(info["first"], len(stream["commands"])),
                dtype=np.int64,
            ),
            "scored origins",
        )
        exact(data["time_s"], stream["time_s"][data["origin"]], "scored times")
        for i in np.flatnonzero(data["revealed"]):
            k = data["origin"][i]
            exact(
                data["truth"][i],
                observed(stream["states"][k + 1 : k + 2])[0],
                "saved causal truth",
            )
            exact(
                data["kinematic"][i],
                kinematic(observed(stream["states"][k : k + 1])[0], info["dt_s"]),
                "kinematic diagnostic",
            )
        verify_journal(case, data, stream, info, protocol)
        if (case / "initial-online.npz").exists():
            initial, frozen = (
                session_arrays(
                    case / ("initial-" + name + ".npz"),
                    info,
                    0,
                    info["initial_model_fingerprint"],
                    protocol,
                )
                for name in ("online", "frozen")
            )
            for key in initial:
                if key != "metadata":
                    exact(initial[key], frozen[key], "independent initialization")
            if protocol["id"] in (
                "online-fit-v3",
                "online-fit-v4",
                "online-fit-v5",
                "online-fit-v6",
                "online-fit-v7",
                "online-fit-v8",
            ):
                norm = arrays(case / "normalization.npz")
                for key, value in norm.items():
                    exact(
                        value, initial["norm_" + key], "initial normalization archive"
                    )
            if reference is not None:
                previous = arrays(reference_case / "initial-online.npz")
                for key in initial:
                    if key.startswith(("param_", "norm_")):
                        paired_exact(
                            initial[key], previous[key], "reference initial core"
                        )
            for name in ("online", "frozen"):
                path = case / ("final-" + name + ".npz")
                if not path.exists():
                    require(
                        info["status"] == "failed", "missing final successful session"
                    )
                    continue
                count = int(data["assimilated"].sum()) if name == "online" else 0
                endpoint = (
                    data["model_after"][count - 1]
                    if count
                    else info["initial_model_fingerprint"]
                )
                final = session_arrays(
                    path, info, count, endpoint, protocol, info[name + "_final_report"]
                )
                exact(initial["scale"], final["scale"], "fixed loss scale")
                for key in final:
                    if key != "metadata":
                        require(
                            np.isfinite(final[key]).all(), "nonfinite saved session"
                        )
                    if (
                        key.startswith("norm_")
                        and key[5:] not in dynamic_normalizers(protocol)
                    ) or (name == "frozen" and key != "metadata"):
                        exact(
                            initial[key],
                            final[key],
                            "fixed normalization/frozen session",
                        )
        if protocol["id"] in ("online-fit-v6", "online-fit-v7", "online-fit-v8"):
            if info.get("endpoint_objectives") is not None:
                require(
                    verify_endpoint_diagnostics(case) == info["endpoint_objectives"],
                    "case endpoint objective differs",
                )
            else:
                require(
                    info["status"] == "failed",
                    "missing successful endpoint diagnostics",
                )
        if protocol["id"] in ("online-fit-v7", "online-fit-v8"):
            verify_causal_captures(case, data, stream, info, protocol)
        for arm in ("candidate", "frozen", "kinematic"):
            error, angles = residuals(data[arm], data["truth"])
            exact(data[arm + "_residual"], error, "component residual")
            exact(data[arm + "_orientation_error_rad"], angles, "orientation residual")
        result = summarize(data, info, reference_data, protocol, stream)
        require(result == wanted == read(case / "summary.json"), "case metrics differ")
        results.append(result)
    require(
        all(
            report[key] == value for key, value in aggregate(results, protocol).items()
        ),
        "aggregate differs",
    )
    if (
        protocol["id"] in ("online-fit-v7", "online-fit-v8")
        and report["all_cases_complete"]
    ):
        require(
            sum(len(row["causal_captures"]) for row in results) == 23,
            "successful evaluation requires all23 captures",
        )
    return dict(
        verified=True,
        manifest_sha256=authority,
        cases=len(results),
        fits=0,
        model_calls=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("collection", type=Path)
    execute.add_argument("--collection-sha256", required=True)
    execute.add_argument("--protocol", type=Path, default=PROTOCOL)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--reference", type=Path, required=True)
    audit = commands.add_parser("verify")
    audit.add_argument("output", type=Path)
    audit.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.action == "run":
        run(
            args.collection.resolve(),
            args.collection_sha256,
            args.output.resolve(),
            args.protocol.resolve(),
            args.reference.resolve(),
        )
    else:
        print(
            json.dumps(verify(args.output.resolve(), args.manifest_sha256)), flush=True
        )
