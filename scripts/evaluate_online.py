"""Run or verify the frozen causal stream experiment; verification never fits."""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
from collect_throw import (
    PROTOCOL,
    authenticate,
    binding,
    check_source,
    checkpoint,
    seal,
)
from run_dart import ROOT, Journal, write
from scipy.spatial.transform import Rotation
from verify_baseline import arrays, digest, exact, observed, read, require

COUNTERS = (
    "observations",
    "optimizer_steps",
    "gradient_calls",
    "objective_calls",
    "accepted_proposals",
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


def summarize(data, info):
    summaries = {
        arm: metrics(data[arm], data["truth"])
        for arm in ("candidate", "frozen", "kinematic")
    }
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
            result["adaptation"][label] = dict(
                count=int(mask.sum()),
                metrics=part,
                ratios=ratio(part["candidate"], part["frozen"]),
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
    return result


def geometric(values):
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return (
        0.0
        if any(value == 0 for value in values)
        else float(np.exp(np.mean(np.log(values))))
    )


def aggregate(cases, protocol=None):
    primary = ("velocity_rmse_m_s", "body_rate_rmse_rad_s")
    families = {}
    for family in ("quad", "fixedwing"):
        chosen = [case for case in cases if case["family"] == family]
        per_metric = {
            key: geometric(
                [case["ratios"][key] if case["ratios"] else None for case in chosen]
            )
            if chosen
            else None
            for key in primary
        }
        families[family] = dict(
            metric_ratios=per_metric, aggregate=geometric(list(per_metric.values()))
        )
    total = geometric([row["aggregate"] for row in families.values()])
    protocol = read(PROTOCOL) if protocol is None else protocol
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
    return dict(
        families=families,
        aggregate_ratio=total,
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


def make_data(rows, times, commands):
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
    return data


def evaluate_case(source, output, info):
    import jax

    from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

    output.mkdir()
    write(output / "attempt.json", info)
    dt, first, begin = info["dt_s"], info["first"], info["begin"]
    stream = arrays(source)
    x, u, times = stream["states"], stream["commands"], stream["time_s"]
    rows = np.arange(first, max(first, len(u)), dtype=np.int64)
    data = make_data(rows, times, u)
    info = dict(
        info,
        prefix=dict(start_row=begin, next_command_row=first, dt_s=dt),
        status="complete",
    )
    journal, online, frozen = Journal(output), None, None

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
        for i, k in enumerate(rows):
            require(online.cursor == k, "causal cursor differs")
            before = online.report
            data["model_before"][i] = fingerprint(online.model)
            for key in COUNTERS:
                data[key + "_before"][i] = before[key]
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
            for key in initial.norms:
                exact(
                    updated.norms[key], initial.norms[key], "fixed normalization " + key
                )
            data["assimilated"][i] = True
            data["model_after"][i] = fingerprint(updated)
            for key in COUNTERS:
                data[key + "_after"][i] = online.report[key]
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
                {},
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
        write(output / "case.json", info)
    result = summarize(data, info)
    write(output / "summary.json", result)
    return result


def run(collection, authority, output, protocol=PROTOCOL):
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "attempt.json",
        dict(collection=str(collection), authority=authority, protocol=str(protocol)),
    )
    try:
        p = read(protocol)
        shutil.copyfile(protocol, output / "protocol.json")
        bound = binding(protocol)
        require(
            Path(importlib.util.find_spec("glassbox").origin).resolve()
            == ROOT / "src/glassbox/__init__.py",
            "evaluate with the current bound Glassbox checkout",
        )
        authenticate(collection, authority)
        require(
            read(collection / "binding.json")["protocol_sha256"] == digest(protocol),
            "collection protocol differs",
        )
        bound["collection_manifest_sha256"] = authority
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
        results = []
        for i, case in enumerate(cases):
            case["opaque_id"] = f"stream-{i:02d}"
            print(json.dumps(dict(case=case["id"], status="starting")), flush=True)
            results.append(
                evaluate_case(inputs / (case["id"] + ".npz"), output / case["id"], case)
            )
            print(json.dumps(dict(case=case["id"], result=results[-1])), flush=True)
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


def verify_journal(case, data, stream, info):
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
                for key in COUNTERS:
                    require(
                        event["report"][key] == data[key + "_before"][position],
                        "before counter differs",
                    )
                require(
                    event["report"]["cursor"] == event["index"]
                    and event["report"]["observations"] == position
                    and event["report"]["optimizer_steps"]
                    == event["report"]["gradient_calls"]
                    == 4 * position,
                    "before cursor/counters differ",
                )
                phase = "revealed"
            elif phase == "revealed":
                exact(
                    values["truth"], data["truth"][position], "journal revealed truth"
                )
                require(data["revealed"][position], "missing revelation flag")
                phase = "assimilated"
            else:
                require(
                    not values and data["assimilated"][position],
                    "unexpected assimilation payload",
                )
                require(
                    event["model"] == data["model_after"][position],
                    "updated fingerprint differs",
                )
                for key in COUNTERS:
                    require(
                        event["report"][key] == data[key + "_after"][position],
                        "after counter differs",
                    )
                require(
                    event["report"]["cursor"] == event["index"] + 1
                    and event["report"]["observations"] == position + 1
                    and event["report"]["optimizer_steps"]
                    == event["report"]["gradient_calls"]
                    == 4 * (position + 1),
                    "assimilation accounting differs",
                )
                phase, position = "predicted", position + 1
        require(
            offset == (case / "arrays.bin").stat().st_size, "unreferenced journal bytes"
        )
    require(
        position == int(data["assimilated"].sum()), "journal assimilation count differs"
    )
    if info["status"] == "complete":
        require(
            phase == "predicted" and position == len(data["origin"]),
            "incomplete successful journal",
        )


def verify(output, authority):
    from glassbox import OnlineFit

    authenticate(output, authority)
    protocol = read(output / "protocol.json")
    require(
        digest(output / "protocol.json")
        == read(output / "binding.json")["protocol_sha256"],
        "saved protocol differs",
    )
    report, results = read(output / "summary.json"), []
    for wanted in report["cases"]:
        case = output / wanted["id"]
        info, data = read(case / "case.json"), arrays(case / "predictions.npz")
        stream = arrays(output / "inputs" / (wanted["id"] + ".npz"))
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
        verify_journal(case, data, stream, info)
        if (case / "initial-online.npz").exists():
            initial_session = OnlineFit.load(case / "initial-online.npz")
            require(
                initial_session.cursor == info["first"]
                and initial_session.report["observations"] == 0
                and fingerprint(initial_session.model)
                == info["initial_model_fingerprint"],
                "initial session endpoint differs",
            )
            initial, frozen = (
                arrays(case / "initial-online.npz"),
                arrays(case / "initial-frozen.npz"),
            )
            for key in initial:
                if key != "metadata":
                    exact(initial[key], frozen[key], "independent initialization")
            for name in ("online", "frozen"):
                path = case / ("final-" + name + ".npz")
                if not path.exists():
                    require(
                        info["status"] == "failed", "missing final successful session"
                    )
                    continue
                session = OnlineFit.load(path)
                count = int(data["assimilated"].sum()) if name == "online" else 0
                endpoint = (
                    data["model_after"][count - 1]
                    if count
                    else info["initial_model_fingerprint"]
                )
                require(
                    session.cursor == info["first"] + count
                    and session.report["observations"] == count
                    and fingerprint(session.model) == endpoint,
                    "final session endpoint differs",
                )
                final = arrays(case / ("final-" + name + ".npz"))
                for key in final:
                    if key != "metadata":
                        require(
                            np.isfinite(final[key]).all(), "nonfinite saved session"
                        )
                    if key.startswith("norm_") or (
                        name == "frozen" and key != "metadata"
                    ):
                        exact(
                            initial[key],
                            final[key],
                            "fixed normalization/frozen session",
                        )
        for arm in ("candidate", "frozen", "kinematic"):
            error, angles = residuals(data[arm], data["truth"])
            exact(data[arm + "_residual"], error, "component residual")
            exact(data[arm + "_orientation_error_rad"], angles, "orientation residual")
        result = summarize(data, info)
        require(result == wanted == read(case / "summary.json"), "case metrics differ")
        results.append(result)
    require(
        all(
            report[key] == value for key, value in aggregate(results, protocol).items()
        ),
        "aggregate differs",
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
        )
    else:
        print(
            json.dumps(verify(args.output.resolve(), args.manifest_sha256)), flush=True
        )
