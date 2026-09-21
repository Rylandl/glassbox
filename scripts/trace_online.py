"""Capture and verify the frozen, unchanged online fitter's causal failure states."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import COUNTERS, fingerprint, metrics
from run_dart import ROOT, clean, write
from scipy.stats import rankdata
from verify_baseline import arrays, digest, observed, read, require

from glassbox import OnlineFit

PROTOCOL = ROOT / "docs/harness/online-causal-trace-v1.json"


def paired_exact(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes() == expected.tobytes(),
        label + ": dtype, shape or bytes differ",
    )


def selected_origins(data, case):
    rows = data["origin"]
    errors = data["candidate"] - data["truth"]
    v, w = (np.linalg.norm(errors[:, a:b], axis=1) for a, b in ((0, 3), (3, 6)))
    anchors = dict(velocity=int(rows[np.argmax(v)]), rate=int(rows[np.argmax(w)]))
    score = rankdata(v, method="average") + rankdata(w, method="average")
    middle, controls = np.median(score), {}
    for name, anchor in anchors.items():
        eligible = [
            i
            for i, row in enumerate(rows)
            if abs(row - anchor) <= 10
            and all(abs(row - center) > 1 for center in anchors.values())
        ]
        require(bool(eligible), "no eligible ordinary origin")
        chosen = min(
            eligible,
            key=lambda i: (abs(score[i] - middle), abs(rows[i] - anchor), rows[i]),
        )
        controls[name] = int(rows[chosen])
    chosen = {k + d for k in anchors.values() for d in (-1, 0, 1)}
    chosen |= {k + d for k in controls.values() for d in (-1, 0)}
    require(
        anchors == case["anchors"] and controls == case["controls"],
        "frozen selection differs",
    )
    require(sorted(chosen) == case["captures"], "capture inventory differs")
    require(chosen <= set(rows), "capture outside scored origins")
    return sorted(chosen)


def check_position(session, data, index, when):
    require(
        fingerprint(session.model) == data["model_" + when][index],
        "causal model " + when + " differs",
    )
    for key in COUNTERS:
        require(
            session.report[key] == data[key + "_" + when][index],
            "causal counter " + key + " differs",
        )


def event_reports(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        (event["index"], event["phase"]): event["report"]
        for event in events
        if "report" in event
    }


def validate_cache(session, tape, initial):
    """Reconstruct retained causal observations, rather than trust internal agreement."""
    for name in ("_contract", "_initial_cursor", "_initial_count", "_horizon"):
        require(
            getattr(session, name) == getattr(initial, name), "initial identity differs"
        )
    paired_exact(session._scale, initial._scale, "fixed bootstrap loss scale")
    for name in initial._bootstrap:
        paired_exact(
            session._bootstrap[name], initial._bootstrap[name], "bootstrap " + name
        )
    p, h, row = session.model.history_steps, session._horizon, session.cursor
    states, commands = observed(tape["states"]), tape["commands"]
    paired_exact(
        session._states, states[row - p - h : row + 1], "complete causal state tail"
    )
    paired_exact(
        session._inputs, commands[row - p - h : row], "complete causal command tail"
    )
    count = min(32, row - initial.cursor)
    origins = range(row - h + 1 - count, row - h + 1)
    if count:
        expected = dict(
            past_states=np.stack([states[k - p : k + 1] for k in origins]),
            past_inputs=np.stack([commands[k - p : k] for k in origins]),
            future_inputs=np.stack([commands[k : k + h] for k in origins]),
            future_states=np.stack([states[k + 1 : k + h + 1] for k in origins]),
        )
    else:
        expected = {name: values[:0] for name, values in initial._bootstrap.items()}
    for name in expected:
        paired_exact(
            session._recent[name], expected[name], "causal recent cache " + name
        )


def validate_capture(session, context, data, info):
    origin = int(context["origin"])
    found = np.flatnonzero(data["origin"] == origin)
    require(len(found) == 1, "capture origin differs")
    i = int(found[0])
    require(session.cursor == origin, "capture is not before assimilation")
    require(
        session.report["observations"] == origin - info["first"],
        "capture observation count differs",
    )
    check_position(session, data, i, "before")
    h = session.model.history_steps
    require(
        context["past_states"].shape == (h + 1, 15), "capture history shape differs"
    )
    require(
        context["past_inputs"].shape == (h, len(context["command"])),
        "capture command history shape differs",
    )
    paired_exact(session._states[-h - 1 :], context["past_states"], "causal state tail")
    paired_exact(session._inputs[-h:], context["past_inputs"], "causal command tail")
    paired_exact(context["truth"], data["truth"][i], "captured truth")
    paired_exact(
        context["recorded_prediction"],
        data["candidate"][i],
        "captured original prediction",
    )


def exact_prediction(prediction, wanted):
    require(
        prediction.dtype == np.float32 and prediction.shape == (1, 15),
        "prediction contract differs",
    )
    paired_exact(prediction[0], wanted, "causal forecast")


def replay_case(reference, output, case):
    output.mkdir()
    original = reference / case["id"]
    data, info = arrays(original / "predictions.npz"), read(original / "case.json")
    selected = selected_origins(data, case)
    reports = event_reports(original / "events.jsonl")
    tape = arrays(reference / "inputs" / (case["id"] + ".npz"))
    session = OnlineFit.load(original / "initial-online.npz")
    h, first = session.model.history_steps, info["first"]
    history = observed(tape["states"][first - h : first + 1])
    issued = tape["commands"][first - h : first].copy()
    actual = {
        k: np.empty_like(data[k])
        for k in ("origin", "candidate", "model_before", "model_after")
    }
    actual.update(
        {
            k + "_" + when: np.empty_like(data[k + "_" + when])
            for k in COUNTERS
            for when in ("before", "after")
        }
    )
    captures = []
    for i, row in enumerate(data["origin"]):
        k = int(row)
        require(session.cursor == k, "causal cursor differs")
        check_position(session, data, i, "before")
        require(session.report == reports[k, "predicted"], "full before report differs")
        before = session.fingerprint()
        selected_path = output / str(k)
        if k in selected:
            selected_path.mkdir()
            session.save(selected_path / "session.npz")
        prediction = np.asarray(
            session.predict(history, issued, tape["commands"][k : k + 1])
        )
        exact_prediction(prediction, data["candidate"][i])
        require(session.fingerprint() == before, "prediction or saving mutated session")
        actual["origin"][i], actual["candidate"][i] = k, prediction[0]
        actual["model_before"][i] = fingerprint(session.model)
        for key in COUNTERS:
            actual[key + "_before"][i] = session.report[key]
        # Reveal the target only after the unchanged public prediction was checked.
        truth = observed(tape["states"][k + 1 : k + 2])[0]
        paired_exact(truth, data["truth"][i], "revealed target")
        if k in selected:
            context = dict(
                origin=np.asarray(k, dtype=np.int64),
                past_states=history.copy(),
                past_inputs=issued.copy(),
                command=tape["commands"][k].copy(),
                truth=truth,
                recorded_prediction=data["candidate"][i],
            )
            validate_capture(session, context, data, info)
            checkpoint(selected_path / "context.npz", **context)
            write(
                selected_path / "capture.json",
                dict(
                    origin=k,
                    time_s=float(tape["time_s"][k]),
                    session_fingerprint=before,
                    model_fingerprint=fingerprint(session.model),
                    report=session.report,
                ),
            )
            captures.append(k)
        session.observe(k, tape["commands"][k], truth)
        check_position(session, data, i, "after")
        require(
            session.report == reports[k, "assimilated"], "full after report differs"
        )
        actual["model_after"][i] = fingerprint(session.model)
        for key in COUNTERS:
            actual[key + "_after"][i] = session.report[key]
        history = np.concatenate((history[1:], truth[None]))
        issued = np.concatenate((issued[1:], tape["commands"][k : k + 1]))
    require(captures == selected, "missing selected captures")
    require(
        session.fingerprint()
        == OnlineFit.load(original / "final-online.npz").fingerprint(),
        "final session differs",
    )
    checkpoint(output / "replay.npz", **actual)
    result = dict(
        id=case["id"],
        transitions=len(data["origin"]),
        captures=captures,
        final_session_fingerprint=session.fingerprint(),
    )
    write(output / "replay.json", result)
    return result


def snapshot_diagnostics(path):
    from _online_trace_diagnostics import diagnose_snapshot

    session = OnlineFit.load(path / "session.npz")
    c = arrays(path / "context.npz")
    before = session.fingerprint()
    result, values = diagnose_snapshot(
        session,
        c["past_states"],
        c["past_inputs"],
        c["command"],
        c["truth"],
        c["recorded_prediction"],
    )
    require(session.fingerprint() == before, "diagnostic mutated session")
    return result, values


def update_comparison(path, case, data):
    result, saved = {}, {}
    for name in ("anchors", "controls"):
        result[name] = {}
        for metric, row in case[name].items():
            c = arrays(path / str(row) / "context.npz")
            predictions = []
            for origin in (row - 1, row):
                session = OnlineFit.load(path / str(origin) / "session.npz")
                predictions.append(
                    np.asarray(
                        session.predict(
                            c["past_states"], c["past_inputs"], c["command"][None]
                        )
                    )[0]
                )
            previous, current = predictions
            exact_prediction(current[None], c["recorded_prediction"])
            delta = current.astype(float) - previous
            i = int(np.flatnonzero(data["origin"] == row - 1)[0])
            result[name][metric] = dict(
                origin=row,
                previous_update_accepted=bool(
                    data["accepted_proposals_after"][i]
                    > data["accepted_proposals_before"][i]
                ),
                previous_model_error=metrics(previous[None], c["truth"][None]),
                actual_model_error=metrics(current[None], c["truth"][None]),
                parameter_update_prediction_change=dict(
                    velocity_m_s=float(np.linalg.norm(delta[:3])),
                    rate_rad_s=float(np.linalg.norm(delta[3:6])),
                ),
            )
            saved[name + "_" + metric] = np.stack(predictions)
    return result, saved


def run(reference, output, protocol=PROTOCOL):
    reference, output, protocol = (
        reference.resolve(),
        output.resolve(),
        protocol.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    p, bound = read(protocol), binding(protocol)
    write(output / "attempt.json", dict(protocol=p["id"]))
    try:
        require(p["id"] == "online-causal-trace-v1", "unsupported trace protocol")
        authority = p["reference"]["manifest_sha256"]
        authenticate(reference, authority)
        from evaluate_online import verify as verify_reference

        verify_reference(reference, authority)
        original_binding = read(reference / "binding.json")
        require(
            bound["runtime"] == original_binding["runtime"],
            "runtime differs from causal reference",
        )
        require(
            Path(importlib.util.find_spec("glassbox").origin).resolve()
            == ROOT / "src/glassbox/__init__.py",
            "use bound learner checkout",
        )
        require(
            {
                name: value
                for name, value in bound["source"]["files"].items()
                if name.startswith("src/")
            }
            == {
                name: value
                for name, value in original_binding["source"]["files"].items()
                if name.startswith("src/")
            },
            "learner source inventory differs",
        )
        shutil.copytree(reference, output / "reference")
        shutil.copyfile(protocol, output / "protocol.json")
        write(output / "binding.json", bound)
        replay = []
        for case in p["cases"]:
            print("replaying " + case["id"], flush=True)
            replay.append(replay_case(output / "reference", output / case["id"], case))
        for case in p["cases"]:
            path = output / case["id"]
            for row in case["captures"]:
                print(f"diagnosing {case['id']} origin {row}", flush=True)
                summary, values = snapshot_diagnostics(path / str(row))
                checkpoint(path / str(row) / "diagnostics.npz", **values)
                write(path / str(row) / "diagnostics.json", summary)
            summary, values = update_comparison(
                path,
                case,
                arrays(output / "reference" / case["id"] / "predictions.npz"),
            )
            write(path / "updates.json", summary)
            checkpoint(path / "updates.npz", **values)
        check_source(bound)
        write(
            output / "summary.json",
            dict(
                complete=True,
                cases=replay,
                transitions=sum(c["transitions"] for c in replay),
                captures=sum(len(c["captures"]) for c in replay),
                learner_changed=False,
                accuracy_adoption=False,
            ),
        )
    except Exception as error:
        write(output / "failure.json", dict(error=repr(error), complete=False))
        raise
    finally:
        authority = seal(output, "glassbox-online-causal-trace-v1")
        print(dict(output=str(output), manifest_sha256=authority), flush=True)


def verify(output, authority):
    authenticate(output, authority)
    require(not (output / "failure.json").exists(), "trace attempt failed")
    p = read(output / "protocol.json")
    require(p["id"] == "online-causal-trace-v1", "unsupported trace protocol")
    require(
        digest(output / "protocol.json")
        == read(output / "binding.json")["protocol_sha256"],
        "protocol binding differs",
    )
    reference = output / "reference"
    from evaluate_online import verify as verify_reference

    verify_reference(reference, p["reference"]["manifest_sha256"])
    count, captures = 0, 0
    results = []
    for case in p["cases"]:
        path = output / case["id"]
        info, data = (
            read(reference / case["id"] / "case.json"),
            arrays(reference / case["id"] / "predictions.npz"),
        )
        chosen = selected_origins(data, case)
        reports = event_reports(reference / case["id"] / "events.jsonl")
        initial = OnlineFit.load(reference / case["id"] / "initial-online.npz")
        actual = arrays(path / "replay.npz")
        require(
            set(actual)
            == {"origin", "candidate", "model_before", "model_after"}
            | {key + "_" + when for key in COUNTERS for when in ("before", "after")},
            "replay inventory differs",
        )
        for key in actual:
            paired_exact(actual[key], data[key], "saved replay " + key)
        require(
            sorted(int(d.name) for d in path.iterdir() if d.is_dir()) == chosen,
            "snapshot inventory differs",
        )
        tape = arrays(reference / "inputs" / (case["id"] + ".npz"))
        for row in chosen:
            point = path / str(row)
            session, c = (
                OnlineFit.load(point / "session.npz"),
                arrays(point / "context.npz"),
            )
            validate_capture(session, c, data, info)
            validate_cache(session, tape, initial)
            require(
                session.report == reports[row, "predicted"],
                "captured original report differs",
            )
            require(int(c["origin"]) == row, "snapshot directory differs")
            h = session.model.history_steps
            paired_exact(
                c["past_states"],
                observed(tape["states"][row - h : row + 1]),
                "saved observed context",
            )
            paired_exact(
                c["past_inputs"], tape["commands"][row - h : row], "saved past commands"
            )
            paired_exact(c["command"], tape["commands"][row], "saved issued command")
            identity = read(point / "capture.json")
            require(
                identity
                == dict(
                    origin=row,
                    time_s=float(tape["time_s"][row]),
                    session_fingerprint=session.fingerprint(),
                    model_fingerprint=fingerprint(session.model),
                    report=session.report,
                ),
                "full snapshot identity differs",
            )
            exact_prediction(
                np.asarray(
                    session.predict(
                        c["past_states"], c["past_inputs"], c["command"][None]
                    )
                ),
                c["recorded_prediction"],
            )
            summary, values = snapshot_diagnostics(point)
            require(
                clean(summary) == read(point / "diagnostics.json"),
                "diagnostic summary differs",
            )
            saved = arrays(point / "diagnostics.npz")
            require(set(saved) == set(values), "diagnostic array inventory differs")
            for name, value in values.items():
                paired_exact(np.asarray(value), saved[name], "diagnostic " + name)
            captures += 1
        summary, values = update_comparison(path, case, data)
        require(
            clean(summary) == read(path / "updates.json"), "update comparison differs"
        )
        saved = arrays(path / "updates.npz")
        require(set(saved) == set(values), "update array inventory differs")
        for name, value in values.items():
            paired_exact(value, saved[name], "update comparison " + name)
        result = dict(
            id=case["id"],
            transitions=len(data["origin"]),
            captures=chosen,
            final_session_fingerprint=OnlineFit.load(
                reference / case["id"] / "final-online.npz"
            ).fingerprint(),
        )
        require(result == read(path / "replay.json"), "replay result differs")
        results.append(result)
        count += len(data["origin"])
    require(
        read(output / "summary.json")
        == dict(
            complete=True,
            cases=results,
            transitions=count,
            captures=captures,
            learner_changed=False,
            accuracy_adoption=False,
        ),
        "trace summary differs",
    )
    return dict(
        verified=True,
        transitions=count,
        captures=captures,
        optimizer_steps=0,
        model_calls="read-only snapshot and diagnostic replays",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("reference", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("output", type=Path)
    verify_parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.reference, args.output, args.protocol)
    else:
        print(verify(args.output, args.manifest_sha256))


if __name__ == "__main__":
    main()
