"""Independent v8/v6 sealed-array audit; no Glassbox/JAX imports or model calls."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def authenticate(path, authority):
    assert sha(path / "manifest.json") == authority, "manifest authority differs"
    files = read(path / "manifest.json")["files"]
    assert set(files) == {
        str(p.relative_to(path))
        for p in path.rglob("*")
        if p.is_file() and p != path / "manifest.json"
    }, "inventory differs"
    for name, wanted in files.items():
        assert sha(path / name) == wanted, "payload differs: " + name
    return len(files)


def same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def close(a, b, label):
    assert np.allclose(a, b, rtol=3e-10, atol=3e-12), (label, a, b)


def fingerprint(metadata, values):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def session(path):
    values = load(path)
    metadata = json.loads(str(values.pop("metadata")))
    identity = metadata.pop("fingerprint")
    assert fingerprint(metadata, values) == identity, "session fingerprint differs"
    assert all(np.isfinite(value).all() for value in values.values())
    core = {
        key: value
        for key, value in values.items()
        if key.startswith(("param_", "norm_"))
    }
    return metadata, values, identity, fingerprint(metadata["model"], core)


def canonical(native):
    return np.c_[native[:, 3:6], native[:, 10:13], raw_rotation(native).reshape(-1, 9)]


def gm(values):
    values = np.asarray(values)
    return 0.0 if np.any(values == 0) else float(np.exp(np.log(values).mean()))


def raw_rotation(native):
    # Independent library conversion from source WXYZ quaternion observations.
    return Rotation.from_quat(native[:, [7, 8, 9, 6]]).as_matrix()


def errors(prediction, truth):
    delta = prediction.astype(float) - truth
    relative = truth[:, 6:].reshape(-1, 3, 3).transpose(0, 2, 1) @ prediction[
        :, 6:
    ].reshape(-1, 3, 3)
    return np.column_stack(
        (
            np.linalg.norm(delta[:, :3], axis=1),
            np.linalg.norm(delta[:, 3:6], axis=1),
            Rotation.from_matrix(relative).magnitude(),
        )
    )


def defect(prediction, origin_native, dt):
    relative = raw_rotation(origin_native).transpose(0, 2, 1) @ prediction[
        :, 6:
    ].reshape(-1, 3, 3)
    return Rotation.from_matrix(relative).as_rotvec() / dt - 0.5 * (
        origin_native[:, 10:13] + prediction[:, 3:6]
    )


def endpoint_components(archive, predictions):
    metadata = json.loads(str(archive["metadata"]))
    dt = metadata["model"]["dt_s"]
    delay = metadata["model"]["delay_steps"]
    norms = {k[5:]: v for k, v in archive.items() if k.startswith("norm_")}
    assert metadata["format"] in ("glassbox-online-fit-v6", "glassbox-online-fit-v8")
    losses, squared_domains, counts = [], [], {}
    for role in ("bootstrap", "recent"):
        past = archive[role + "_past_states"]
        up = archive[role + "_past_inputs"]
        future = archive[role + "_future_inputs"]
        target = archive[role + "_future_states"]
        counts[role] = len(past)
        assert (
            predictions[role].shape == target.shape
            and predictions[role].dtype == np.float64
        )
        if not len(past):
            continue
        residual = (predictions[role] - target) / archive["scale"]
        group_losses = []
        for left, right in ((0, 3), (3, 6), (6, 15)):
            radius = np.sqrt(np.sum(residual[..., left:right] ** 2, axis=-1))
            group_losses.append(np.where(radius < 1, 0.5 * radius**2, radius - 0.5))
        losses.append(float(np.mean(group_losses)))
        window_domains = []
        for states, commands, targets, issued in zip(past, up, target, future):
            motion = []
            command = []
            x = np.concatenate((states, targets[:-1]))
            u = np.concatenate((commands, issued))
            for row in range(delay, len(x)):
                body = np.r_[x[row, 6:].reshape(3, 3).T @ x[row, :3], x[row, 3:6]]
                z = (body - norms["body_mean"][:6]) / norms["body_scale"][:6]
                bound = norms["motion_bound_scale"] / 4
                for axis in range(6):
                    if abs(z[axis]) > bound[axis]:
                        z[axis] = (
                            np.sign(z[axis])
                            * bound[axis]
                            * (1 + 3 * np.tanh((abs(z[axis]) / bound[axis] - 1) / 3))
                        )
                motion.append(z)
                command.append((u[row] - norms["input_mean"]) / norms["input_scale"])
            window_domains.append(
                np.r_[
                    np.mean(np.square(motion), axis=0),
                    np.mean(np.square(command), axis=0),
                ]
            )
        squared_domains.append(np.mean(window_domains, axis=0))
    rms = np.maximum(1, np.sqrt(np.mean(squared_domains, axis=0)))
    domain = np.r_[rms[:6], 1 / norms["body_scale"][6:9], rms[6:], rms[6:]]
    coefficients = (
        archive["param_quadratic"]
        * norms["output_scale"][None, :]
        / norms["quadratic_scale"][:, None]
    )
    hessian = np.zeros((6, len(domain), len(domain)))
    term = 0
    for i in range(len(domain)):
        for j in range(i, len(domain)):
            hessian[:, i, j] = coefficients[term] * (2 if i == j else 1)
            if i != j:
                hessian[:, j, i] = coefficients[term]
            term += 1
    physical = hessian * domain[None, :, None] * domain[None, None, :]
    physical *= (dt / archive["scale"][0, :6])[:, None, None]
    prior = float(0.01 / 4 * np.sum(physical**2))
    return dict(
        data_loss=float(np.mean(losses)),
        prior_loss=prior,
        combined_loss=float(np.mean(losses)) + prior,
        domain=domain.tolist(),
        cache_windows=counts,
    )


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
DYNAMIC = ("feature_scale", "quadratic_scale", "output_scale")
ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def optional_finite(value):
    assert value is None or (type(value) in (int, float) and math.isfinite(value)), (
        "nonfinite diagnostic must be null"
    )
    return value is not None


def proposal_decision(proposal):
    """Check saved scalar decisions, not the model provenance of their values."""
    assert isinstance(proposal, dict) and set(proposal) == {
        "current_loss",
        "conditioning_finite",
        "direction_finite",
        "trust_shrink",
        "trials",
        "selected_alpha",
        "trial_evaluations",
    }, "proposal schema differs"
    current, shrink = proposal["current_loss"], proposal["trust_shrink"]
    current_finite, shrink_finite = optional_finite(current), optional_finite(shrink)
    assert current is None or current >= 0, "negative current objective"
    assert (
        type(proposal["conditioning_finite"]) is bool
        and type(proposal["direction_finite"]) is bool
    )
    assert shrink is None or 0 <= shrink <= 1, "invalid trust shrink"
    if proposal["direction_finite"]:
        assert current_finite and shrink_finite, (
            "finite direction lacks finite objective or trust"
        )
    trials = proposal["trials"]
    assert isinstance(trials, list) and 1 <= len(trials) <= 5, (
        "trial count outside ladder"
    )
    assert type(proposal["trial_evaluations"]) is int and proposal[
        "trial_evaluations"
    ] == len(trials), "trial call count differs"
    passing = []
    gains = []
    for index, trial in enumerate(trials):
        assert set(trial) == {"alpha", "loss", "predicted_reduction", "finite"}, (
            "trial schema differs"
        )
        assert (
            type(trial["alpha"]) in (float, int) and trial["alpha"] == ALPHAS[index]
        ), "trial alpha is not ladder prefix"
        assert type(trial["finite"]) is bool
        loss, predicted = trial["loss"], trial["predicted_reduction"]
        loss_finite, predicted_finite = (
            optional_finite(loss),
            optional_finite(predicted),
        )
        assert loss is None or loss >= 0, "negative trial objective"
        if trial["finite"]:
            assert loss_finite and predicted_finite, "finite trial lacks finite scalars"
        eligible = (
            proposal["conditioning_finite"]
            and proposal["direction_finite"]
            and trial["finite"]
        )
        gain = (current - loss) / predicted if eligible and predicted > 0 else -math.inf
        passing.append(bool(eligible and loss < current and gain >= 0.1))
        gains.append(gain)
    selected = proposal["selected_alpha"]
    assert type(selected) in (int, float) and math.isfinite(selected), (
        "invalid selected alpha"
    )
    accepted = any(passing)
    if accepted:
        assert (
            passing.index(True) == len(trials) - 1 and selected == trials[-1]["alpha"]
        ), "did not stop at first acceptable trial"
    else:
        assert len(trials) == 5 and selected == 0, (
            "rejection must retain all five trials"
        )
    return dict(
        accepted=accepted,
        gain=gains[-1],
        trials=len(trials),
        selected_alpha=float(selected),
    )


def accounting(report, count):
    assert all(type(report[key]) is int for key in COUNTERS), (
        "cumulative count type differs"
    )
    for name, multiplier in dict(
        observations=1,
        optimizer_steps=1,
        gradient_calls=1,
        cg_iterations=16,
        curvature_calls=16,
        conditioning_calls=1,
    ).items():
        assert report[name] == multiplier * count, (name, count)
    assert 0 <= report["accepted_proposals"] <= count
    assert 1e-8 <= report["damping"] <= 1e8
    assert (
        type(report["objective_calls"]) is int
        and 2 * count <= report["objective_calls"] <= 6 * count
    )
    if count:
        decision = proposal_decision(report["last_proposal"])
        assert (
            2 * (count - 1)
            <= report["objective_calls"] - 1 - decision["trials"]
            <= 6 * (count - 1)
        )
        assert (
            int(decision["accepted"])
            <= report["accepted_proposals"]
            <= count - int(not decision["accepted"])
        )
    else:
        assert report["last_proposal"] is None and report["damping"] == 1.0


def verify_transition(before, after):
    """Reconstruct one decision and counter/damping deltas from logged scalars."""
    assert after["observations"] == before["observations"] + 1
    accounting(before, before["observations"])
    accounting(after, after["observations"])
    decision = proposal_decision(after["last_proposal"])
    assert (
        after["objective_calls"] - before["objective_calls"] == 1 + decision["trials"]
    ), "objective delta differs from attempted trials"
    assert after["accepted_proposals"] - before["accepted_proposals"] == int(
        decision["accepted"]
    ), "decision and acceptance count differ"
    damping = before["damping"]
    if not decision["accepted"] or decision["gain"] < 0.25:
        damping *= 4.0
    elif decision["gain"] > 0.75:
        damping *= 0.5
    damping = min(1e8, max(1e-8, damping))
    assert after["damping"] == damping, "one final damping update differs"
    return decision


def check_cache(values, meta, source, initial):
    cursor, first = meta["cursor"], meta["initial_cursor"]
    history, horizon = meta["model"]["history_steps"], meta["horizon"]
    states = canonical(source["states"])
    close(
        values["tail_states"],
        states[cursor - history - horizon : cursor + 1],
        "session causal state tail",
    )
    same(values["tail_inputs"], source["commands"][cursor - history - horizon : cursor])
    for field in ("past_states", "past_inputs", "future_inputs", "future_states"):
        same(values["bootstrap_" + field], initial["bootstrap_" + field])
    close(values["scale"], initial["scale"], "fixed group scales")
    count = min(cursor - first, 32)
    origins = np.arange(cursor - horizon - count + 1, cursor - horizon + 1)
    assert len(values["recent_past_states"]) == count
    for index, origin in enumerate(origins):
        close(
            values["recent_past_states"][index],
            states[origin - history : origin + 1],
            "recent measured history",
        )
        same(
            values["recent_past_inputs"][index],
            source["commands"][origin - history : origin],
        )
        same(
            values["recent_future_inputs"][index],
            source["commands"][origin : origin + horizon],
        )
        close(
            values["recent_future_states"][index],
            states[origin + 1 : origin + horizon + 1],
            "recent completed targets",
        )


def verify_bootstrap(initial, metadata, source):
    history, horizon = metadata["model"]["history_steps"], metadata["horizon"]
    first, count = metadata["initial_cursor"], metadata["initial_count"]
    origins = np.arange(first - count, first - horizon + 1)[-32:]
    assert len(initial["bootstrap_past_states"]) == len(origins)
    states = canonical(source["states"])
    for index, origin in enumerate(origins):
        close(
            initial["bootstrap_past_states"][index],
            states[origin - history : origin + 1],
            "prefix bootstrap states",
        )
        same(
            initial["bootstrap_past_inputs"][index],
            source["commands"][origin - history : origin],
        )
        same(
            initial["bootstrap_future_inputs"][index],
            source["commands"][origin : origin + horizon],
        )
        close(
            initial["bootstrap_future_states"][index],
            states[origin + 1 : origin + horizon + 1],
            "prefix bootstrap targets",
        )
    origin = initial["bootstrap_past_states"][:, -1:]
    target = initial["bootstrap_future_states"]
    scales = []
    for left, right, coefficient in ((0, 3, 1.0), (3, 6, 1.0), (6, 15, 0.5)):
        x = origin[..., left:right]
        change = np.sqrt(
            coefficient
            * np.mean(np.sum((x - target[..., left:right]) ** 2, axis=-1), axis=0)
        )
        spread = np.sqrt(
            coefficient * np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=-1))
        )
        scales.append(
            np.repeat(
                (
                    np.maximum(change, 0.01 * max(float(spread), 1e-4))
                    / np.sqrt(coefficient)
                )[:, None],
                right - left,
                axis=1,
            )
        )
    close(initial["scale"], np.concatenate(scales, axis=1), "bootstrap group scales")


def journal(case, predictions, source, info, captures):
    before_norms = {}
    before_reports = {}
    initial = load(case / "normalization.npz")
    norms = {key: initial[key] for key in DYNAMIC}
    accepted = 0
    previous = None
    decisions = []
    count = len(predictions["origin"])
    with (
        (case / "arrays.bin").open("rb") as binary,
        (case / "events.jsonl").open() as events,
    ):
        event_count = 0
        for row, origin in enumerate(predictions["origin"]):
            for phase in ("predicted", "revealed", "assimilated"):
                line = next(events, None)
                assert line is not None, "incomplete causal journal"
                event = json.loads(line)
                assert event["phase"] == phase and event["index"] == origin
                values = {}
                for key, offset in event["arrays"].items():
                    assert offset == binary.tell(), "journal offset differs"
                    values[key] = np.load(binary, allow_pickle=False)
                if phase == "revealed":
                    assert set(values) == {"truth"}
                    same(values["truth"], predictions["truth"][row])
                else:
                    when = "before" if phase == "predicted" else "after"
                    report = event["report"]
                    accounting(report, row + (when == "after"))
                    assert report["cursor"] == origin + (when == "after")
                    for key in COUNTERS:
                        assert report[key] == predictions[key + "_" + when][row]
                    assert event["model"] == predictions["model_" + when][row]
                    if phase == "predicted":
                        if previous is not None:
                            assert report == previous, (
                                "pre-prediction report differs from prior assimilation"
                            )
                        before = report
                        assert set(values) == {
                            "candidate",
                            "frozen",
                            "kinematic",
                            "command",
                        }
                        for arm in ("candidate", "frozen", "kinematic"):
                            same(values[arm], predictions[arm][row])
                        same(values["command"], source["commands"][origin])
                        if int(origin) in captures:
                            before_norms[int(origin)] = {
                                key: value.copy() for key, value in norms.items()
                            }
                            before_reports[int(origin)] = report
                    else:
                        decision = verify_transition(before, report)
                        assert (
                            predictions["trial_evaluations"][row] == decision["trials"]
                        ), "derived trial count differs"
                        assert (
                            predictions["selected_alpha"][row]
                            == decision["selected_alpha"]
                        ), "derived selected alpha differs"
                        decisions.append(decision)
                        previous = report
                        assert set(values) == {"norm_" + key for key in DYNAMIC}
                        following = report["accepted_proposals"]
                        assert following - accepted in (0, 1)
                        for key in DYNAMIC:
                            changed = values["norm_" + key]
                            assert np.isfinite(changed).all() and np.all(changed > 0)
                            if following == accepted:
                                same(changed, norms[key])
                                assert (
                                    predictions["model_after"][row]
                                    == predictions["model_before"][row]
                                )
                            else:
                                assert np.all(changed >= norms[key])
                            if key == "feature_scale":
                                same(changed[-8:], initial[key][-8:])
                            norms[key] = changed
                        accepted = following
                event_count += 1
        assert next(events, None) is None, "surplus journal event"
        assert binary.read() == b"", "unreferenced journal bytes"
    assert event_count == 3 * count
    final = load(case / "final-online.npz")
    for key in DYNAMIC:
        same(norms[key], final["norm_" + key])
    assert previous == info["online_final_report"], "final journal report differs"
    trial_count = sum(row["trials"] for row in decisions)
    work = dict(
        observations=count,
        scheduled_cg_iterations=16 * count,
        gradient_calls=count,
        current_objective_calls=count,
        trial_objective_calls=trial_count,
        total_objective_calls=count + trial_count,
        trial_count_histogram={
            str(n): sum(row["trials"] == n for row in decisions) for n in range(1, 6)
        },
        selected_alpha_histogram={
            str(alpha): sum(row["selected_alpha"] == alpha for row in decisions)
            for alpha in (0.0,) + ALPHAS
        },
    )
    return before_norms, before_reports, event_count, work


def verify_captures(
    case, predictions, source, info, protocol, initial, before_norms, before_reports
):
    wanted = protocol["causal_captures"]["origins"].get(info["id"], [])
    actual = sorted(int(path.parent.name) for path in case.glob("*/session.npz"))
    assert actual == wanted == info["causal_captures"]
    evidence = []
    for origin in wanted:
        path = case / str(origin)
        assert {f.name for f in path.iterdir()} == {
            "session.npz",
            "context.npz",
            "capture.json",
        }
        capture, context = read(path / "capture.json"), load(path / "context.npz")
        meta, values, identity, core = session(path / "session.npz")
        row = int(np.flatnonzero(predictions["origin"] == origin)[0])
        assert meta["format"] == "glassbox-online-fit-v8"
        assert meta["cursor"] == origin and meta["initial_cursor"] == info["first"]
        assert capture["origin"] == int(context["origin"]) == origin
        assert capture["time_s"] == source["time_s"][origin]
        assert identity == capture["session_fingerprint"]
        assert core == capture["model_fingerprint"] == predictions["model_before"][row]
        accounting(
            dict(
                meta["counts"],
                observations=row,
                damping=meta["damping"],
                last_proposal=meta["last_proposal"],
            ),
            row,
        )
        accounting(capture["report"], row)
        assert capture["report"] == before_reports[origin], (
            "capture report differs from causal journal"
        )
        assert meta["last_proposal"] == capture["report"]["last_proposal"], (
            "capture decision differs"
        )
        assert capture["report"]["cursor"] == origin
        assert capture["report"]["damping"] == meta["damping"]
        assert capture["report"]["recipe"] == meta["recipe"]
        for key in COUNTERS:
            assert capture["report"][key] == predictions[key + "_before"][row]
            if key != "observations":
                assert meta["counts"][key] == predictions[key + "_before"][row]
        history = meta["model"]["history_steps"]
        close(
            context["past_states"],
            canonical(source["states"][origin - history : origin + 1]),
            "capture actual history",
        )
        same(context["past_inputs"], source["commands"][origin - history : origin])
        same(context["command"], source["commands"][origin])
        same(context["recorded_prediction"], predictions["candidate"][row])
        assert context["recorded_prediction"].dtype == np.float32
        same(context["truth"], predictions["truth"][row])
        check_cache(values, meta, source, initial)
        for key in DYNAMIC:
            same(values["norm_" + key], before_norms[origin][key])
        for key in initial:
            if key.startswith("norm_") and key[5:] not in DYNAMIC:
                same(values[key], initial[key])
        evidence.append(
            dict(
                origin=origin,
                session_fingerprint=identity,
                model_fingerprint=core,
                observations=row,
            )
        )
    return evidence


def audit(root, authority):
    payloads = authenticate(root, authority)
    ref = root / "reference"
    ref_authority = "43014bce12460b91fc034e127ec4c9249322dc134ce18dbed4b7c6795791983c"
    assert sha(ref / "manifest.json") == ref_authority
    report = read(root / "summary.json")
    protocol = read(root / "protocol.json")
    assert protocol["id"] == "online-fit-v8"
    assert protocol["comparison"]["reference"]["manifest_sha256"] == ref_authority
    assert report["primary_comparison"] == "candidate / saved online-fit-v6"
    binding = read(root / "binding.json")
    assert sha(root / "protocol.json") == binding["protocol_sha256"]
    assert binding["reference_manifest_sha256"] == ref_authority
    nested = []
    location = root
    while (location / "reference").is_dir():
        p = read(location / "protocol.json")
        reference = p["comparison"]["reference"]
        location = location / "reference"
        nested.append(
            dict(
                protocol=reference["protocol_id"],
                authority=reference["manifest_sha256"],
                payloads=authenticate(location, reference["manifest_sha256"]),
            )
        )
        assert read(location / "protocol.json")["id"] == reference["protocol_id"]
    assert [item["protocol"] for item in nested] == [
        "online-fit-v6",
        "online-fit-v4",
        "online-fit-v2",
    ]
    names = [
        "quad-arm-115",
        "quad-arm-125",
        "quad-arm-135",
        "quad-change",
        "fixedwing-80",
        "fixedwing-81",
    ]
    assert sorted(c["id"] for c in report["cases"]) == sorted(names)
    metrics = ("velocity_rmse_m_s", "body_rate_rmse_rad_s", "orientation_rmse_rad")
    rows = {}
    total = 0
    capture_count = 0
    journal_events = 0
    for name in names:
        p = load(root / name / "predictions.npz")
        r = load(ref / name / "predictions.npz")
        source = load(root / "inputs" / (name + ".npz"))
        original = load(ref / "inputs" / (name + ".npz"))
        assert set(source) == set(original)
        for key in source:
            same(source[key], original[key])
        info = read(root / name / "case.json")
        reference_info = read(ref / name / "case.json")
        for key in (
            "id",
            "family",
            "opaque_id",
            "dt_s",
            "begin",
            "first",
            "ordered_commands",
        ):
            assert info[key] == reference_info[key], "paired case identity differs"
        saved = read(root / name / "summary.json")
        assert saved == next(c for c in report["cases"] if c["id"] == name)
        n = len(p["origin"])
        total += n
        assert np.array_equal(
            p["origin"], np.arange(info["first"], len(source["commands"]))
        )
        same(p["time_s"], source["time_s"][p["origin"]])
        for key in ("origin", "time_s", "truth", "frozen", "kinematic"):
            same(p[key], r[key])
        assert all(p[key].all() for key in ("predicted", "revealed", "assimilated"))
        assert p["candidate"].dtype == p["frozen"].dtype == np.float32
        assert all(
            np.isfinite(p[key]).all()
            for key in ("candidate", "truth", "frozen", "kinematic")
        )
        assert np.array_equal(p["observations_before"], np.arange(n))
        assert np.array_equal(p["observations_after"], np.arange(1, n + 1))
        assert np.array_equal(p["model_before"][1:], p["model_after"][:-1])
        assert saved["complete"] and info["status"] == "complete"
        initial = load(root / name / "initial-online.npz")
        ref_initial = load(ref / name / "initial-online.npz")
        assert {key for key in initial if key.startswith(("param_", "norm_"))} == {
            key for key in ref_initial if key.startswith(("param_", "norm_"))
        }
        for key in initial:
            if key.startswith(("param_", "norm_")) or key == "scale":
                same(initial[key], ref_initial[key])
        captures = protocol["causal_captures"]["origins"].get(name, [])
        before_norms, before_reports, event_count, work = journal(
            root / name, p, source, info, captures
        )
        assert work == saved["work"], "saved work summary differs"
        journal_events += event_count
        initial_meta, initial_values, _, initial_core = session(
            root / name / "initial-online.npz"
        )
        assert initial_meta["format"] == "glassbox-online-fit-v8"
        assert initial_core == info["initial_model_fingerprint"] == p["model_before"][0]
        verify_bootstrap(initial_values, initial_meta, source)
        snapshot_evidence = verify_captures(
            root / name,
            p,
            source,
            info,
            protocol,
            initial_values,
            before_norms,
            before_reports,
        )
        assert saved["causal_captures"] == captures
        capture_count += len(snapshot_evidence)
        for when in ("initial", "final"):
            for arm in ("online", "frozen"):
                meta, values, _, core = session(
                    root / name / (when + "-" + arm + ".npz")
                )
                count = n if when == "final" and arm == "online" else 0
                assert (
                    meta["initial_cursor"] == info["first"]
                    and meta["cursor"] == info["first"] + count
                )
                assert core == (p["model_after"][-1] if count else initial_core)
                accounting(
                    dict(
                        meta["counts"],
                        observations=count,
                        damping=meta["damping"],
                        last_proposal=meta["last_proposal"],
                    ),
                    count,
                )
                if when == "final":
                    final_report = info[arm + "_final_report"]
                    assert meta["last_proposal"] == final_report["last_proposal"]
                    assert meta["damping"] == final_report["damping"]
                    assert all(
                        final_report[key] == value
                        for key, value in meta["counts"].items()
                    )
                check_cache(values, meta, source, initial_values)
                if arm == "frozen" or when == "initial":
                    for key in initial_values:
                        same(values[key], initial_values[key])
                else:
                    for key in initial_values:
                        if key.startswith("norm_") and key[5:] not in DYNAMIC:
                            same(values[key], initial_values[key])
        target_native = source["states"][p["origin"] + 1]
        close(p["truth"][:, :3], target_native[:, 3:6], "source velocity")
        close(p["truth"][:, 3:6], target_native[:, 10:13], "source rate")
        close(
            p["truth"][:, 6:].reshape(-1, 3, 3),
            raw_rotation(target_native),
            "source rotation",
        )
        norms = {
            key[5:]: value for key, value in initial.items() if key.startswith("norm_")
        }
        body = np.c_[
            np.einsum(
                "nji,nj->ni", p["truth"][:, 6:].reshape(-1, 3, 3), p["truth"][:, :3]
            ),
            p["truth"][:, 3:6],
        ]
        support = np.abs((body - norms["body_mean"][:6]) / norms["body_scale"][:6]) / (
            norms["motion_bound_scale"] / 4
        )
        command_z = (source["commands"][p["origin"]] - norms["input_mean"]) / norms[
            "input_scale"
        ]
        close(p["motion_support_ratio"], support, "observed motion support")
        close(p["command_z"], command_z, "issued command support")
        assert saved["support"]["observed_rows"] == n
        assert saved["support"]["outside_bootstrap_motion_support"] == int(
            np.any(support > 1, axis=1).sum()
        )
        close(
            saved["support"]["maximum_motion_ratio"],
            support.max(),
            "maximum motion support",
        )
        close(
            saved["support"]["maximum_abs_command_z"],
            np.abs(command_z).max(),
            "maximum command support",
        )
        assert saved["prefix"] == info["prefix"]
        prefix = source["commands"][info["begin"] : info["first"]]
        for prefix_key, commands in (
            ("", prefix),
            ("initialization_", prefix[-round(0.25 / info["dt_s"]) :]),
        ):
            close(
                saved["prefix"][prefix_key + "command_span"],
                np.ptp(commands, axis=0),
                "prefix command span",
            )
            assert saved["prefix"][prefix_key + "centered_command_rank"] == int(
                np.linalg.matrix_rank(commands - commands.mean(axis=0))
            )
        for arm in ("candidate", "frozen"):
            assert saved["startup_s"][arm] == info[arm + "_startup_s"]
            assert (
                math.isfinite(saved["startup_s"][arm]) and saved["startup_s"][arm] >= 0
            )
        origin = source["states"][p["origin"]]
        canonical_origin = canonical(origin)
        held = canonical_origin.copy()
        held[:, 6:] = (
            canonical_origin[:, 6:].reshape(-1, 3, 3)
            @ Rotation.from_rotvec(info["dt_s"] * canonical_origin[:, 3:6]).as_matrix()
        ).reshape(-1, 9)
        close(p["kinematic"], held, "independent kinematic forecasts")
        all_errors = {
            arm: errors(prediction, p["truth"])
            for arm, prediction in [
                ("candidate", p["candidate"]),
                ("reference", r["candidate"]),
                ("frozen", p["frozen"]),
                ("kinematic", p["kinematic"]),
            ]
        }
        for arm in ("candidate", "frozen", "kinematic"):
            same(p[arm + "_residual"], p[arm].astype(float) - p["truth"])
            close(
                p[arm + "_orientation_error_rad"],
                all_errors[arm][:, 2],
                "saved SO3 errors",
            )
        rmse = {arm: np.sqrt(np.mean(e**2, axis=0)) for arm, e in all_errors.items()}
        tail = {
            arm: np.sqrt(
                np.mean(np.sort(e, axis=0)[-math.ceil(0.1 * n) :] ** 2, axis=0)
            )
            for arm, e in all_errors.items()
        }
        truth_defect = defect(p["truth"], origin, info["dt_s"])
        defect_rms = {}
        for arm, prediction in [
            ("candidate", p["candidate"]),
            ("reference", r["candidate"]),
            ("frozen", p["frozen"]),
            ("kinematic", p["kinematic"]),
        ]:
            for i, key in enumerate(metrics):
                close(rmse[arm][i], saved["metrics"][arm][key], arm + " " + key)
            d = defect(prediction, origin, info["dt_s"])
            defect_rms[arm] = float(
                np.sqrt(np.mean(np.sum((d - truth_defect) ** 2, axis=1)))
            )
            diag = saved["diagnostics"][arm]
            assert diag["count"] == n
            close(
                defect_rms[arm],
                diag["rotation_rate"]["truth_relative_rmse_rad_s"],
                "defect",
            )
            close(
                np.sqrt(np.mean(np.sum(d * d, axis=1))),
                diag["rotation_rate"]["predicted_defect_rmse_rad_s"],
                "raw predicted defect",
            )
            close(
                np.sqrt(np.mean(np.sum(truth_defect * truth_defect, axis=1))),
                diag["rotation_rate"]["truth_defect_rmse_rad_s"],
                "raw truth defect",
            )
            for i, key in enumerate(
                ("velocity_m_s", "body_rate_rad_s", "orientation_rad")
            ):
                tail_info = diag["tail"][key]
                values = all_errors[arm][:, i]
                for quantile, label in ((0.5, "median"), (0.95, "p95"), (0.99, "p99")):
                    close(
                        np.quantile(values, quantile), tail_info[label], "tail " + label
                    )
                sse = np.sum(values**2)
                close(
                    np.sum(np.sort(values)[-5:] ** 2) / sse if sse else 0.0,
                    tail_info["largest_five_squared_error_share"],
                    "tail concentration",
                )
                assert tail_info["upper_decile_count"] == math.ceil(0.1 * n)
                close(tail[arm][i], diag["tail"][key]["upper_decile_rmse"], "tail")
                close(
                    np.max(all_errors[arm][:, i]), diag["tail"][key]["maximum"], "max"
                )
        ratio = rmse["candidate"] / np.maximum(rmse["reference"], 1e-9)
        tail_ratio = tail["candidate"] / np.maximum(tail["reference"], 1e-9)
        defect_ratio = defect_rms["candidate"] / max(defect_rms["reference"], 1e-9)
        for index, key in enumerate(metrics):
            close(ratio[index], saved["reference_ratios"][key], "reference ratio")
            close(
                rmse["candidate"][index] / max(rmse["frozen"][index], 1e-9),
                saved["ratios"][key],
                "frozen ratio",
            )
            close(
                rmse["candidate"][index] / max(rmse["kinematic"][index], 1e-9),
                saved["kinematic_ratios"][key],
                "kinematic ratio",
            )
        for key, value in dict(
            velocity_tail=tail_ratio[0],
            rate_tail=tail_ratio[1],
            orientation=ratio[2],
            rotation_rate=defect_ratio,
        ).items():
            close(value, saved["robustness_ratios"][key], "case robustness ratio")
        for action in ("predict", "frozen_predict", "observe"):
            timings = p[action + "_s"]
            latency = saved["latency"][action]
            assert (
                np.isfinite(timings).all()
                and np.all(timings >= 0)
                and latency["count"] == n
            )
            for key, value in dict(
                first_s=timings[0],
                warm_median_s=np.median(timings[1:]),
                warm_p95_s=np.quantile(timings[1:], 0.95),
                warm_max_s=np.max(timings[1:]),
            ).items():
                close(value, latency[key], "latency " + key)
        if info.get("change_at_s") is not None:
            change = info["change_at_s"]
            for label, mask in [
                ("first_second", (p["time_s"] >= change) & (p["time_s"] < change + 1)),
                ("after_first_second", p["time_s"] >= change + 1),
            ]:
                part = saved["adaptation"][label]
                assert int(mask.sum()) == part["count"] and mask.any()
                for arm, values in all_errors.items():
                    measures = np.sqrt(np.mean(values[mask] ** 2, axis=0))
                    for index, key in enumerate(metrics):
                        close(
                            measures[index],
                            part["metrics"][arm][key],
                            "adaptation " + key,
                        )
                for key in metrics:
                    for field, comparator in (
                        ("ratios", "frozen"),
                        ("reference_ratios", "reference"),
                    ):
                        close(
                            part[field][key],
                            part["metrics"]["candidate"][key]
                            / max(part["metrics"][comparator][key], 1e-9),
                            "adaptation ratio",
                        )
        endpoints = {}
        for version, location in (("v8", root), ("v6", ref)):
            endpoint_predictions = load(location / name / "endpoint-predictions.npz")
            endpoint_report = read(location / name / "endpoint-objectives.json")
            endpoints[version] = {}
            for when in ("initial", "final"):
                archive = load(location / name / (when + "-online.npz"))
                assert (
                    json.loads(str(archive["metadata"]))["format"]
                    == "glassbox-online-fit-" + version
                )
                session(location / name / (when + "-online.npz"))
                components = endpoint_components(
                    archive,
                    {
                        role: endpoint_predictions[when + "_" + role]
                        for role in ("bootstrap", "recent")
                    },
                )
                for key in ("data_loss", "prior_loss", "combined_loss", "domain"):
                    close(
                        components[key],
                        endpoint_report[when][key],
                        version + " " + when + " " + key,
                    )
                assert (
                    components["cache_windows"]
                    == endpoint_report[when]["cache_windows"]
                )
                endpoints[version][when] = {
                    key: value for key, value in components.items() if key != "domain"
                }
        rows[name] = dict(
            family=saved["family"],
            count=n,
            work=work,
            causal_captures=snapshot_evidence,
            rmse={k: v.tolist() for k, v in rmse.items()},
            ratios=ratio.tolist(),
            tail_ratios=tail_ratio.tolist(),
            defect_ratio=defect_ratio,
            endpoints=endpoints,
            maximums={
                arm: [
                    dict(
                        error=float(e[:, i].max()),
                        origin=int(p["origin"][e[:, i].argmax()]),
                        time_s=float(p["time_s"][e[:, i].argmax()]),
                    )
                    for i in range(3)
                ]
                for arm, e in all_errors.items()
            },
            latency=saved["latency"]["observe"],
        )
    assert total == 3137 and capture_count == 23 and journal_events == 3 * total
    work = {}
    for key, value in next(iter(rows.values()))["work"].items():
        if isinstance(value, dict):
            work[key] = {
                bucket: sum(row["work"][key][bucket] for row in rows.values())
                for bucket in value
            }
        else:
            work[key] = sum(row["work"][key] for row in rows.values())
    assert work == report["work"], "aggregate work summary differs"
    families = {}
    for family in ("quad", "fixedwing"):
        selected = [c for c in rows.values() if c["family"] == family]
        families[family] = dict(
            primary=gm([v for c in selected for v in c["ratios"][:2]]),
            upper_decile=gm([v for c in selected for v in c["tail_ratios"][:2]]),
            orientation=gm([c["ratios"][2] for c in selected]),
            rotation_rate=gm([c["defect_ratio"] for c in selected]),
        )
        close(
            families[family]["primary"],
            report["families"][family]["aggregate"],
            "family primary",
        )
        for index, key in enumerate(metrics[:2]):
            close(
                gm([c["ratios"][index] for c in selected]),
                report["families"][family]["metric_ratios"][key],
                "family per-metric",
            )
        for key in ("upper_decile", "orientation", "rotation_rate"):
            close(
                families[family][key],
                report["robustness"]["families"][family][key],
                "family robustness",
            )
    aggregate = {
        key: gm([c[key] for c in families.values()])
        for key in ("primary", "upper_decile", "orientation", "rotation_rate")
    }
    close(aggregate["primary"], report["aggregate_ratio"], "aggregate")
    for key in ("upper_decile", "orientation", "rotation_rate"):
        close(
            aggregate[key],
            report["robustness"]["aggregate_ratios"][key],
            "aggregate robustness",
        )
    accuracy = aggregate["primary"] <= 0.8 and all(
        c["primary"] < 1 for c in families.values()
    )
    robustness = (
        aggregate["upper_decile"] < 1
        and aggregate["orientation"] <= 1
        and aggregate["rotation_rate"] < 1
    )
    assert (
        accuracy == report["accuracy_passed"]
        and robustness == report["robustness_passed"]
    )
    assert report["all_cases_complete"]
    assert report["real_time_target_passed"] == all(
        row["latency"]["warm_p95_s"] <= read(root / name / "case.json")["dt_s"]
        for name, row in rows.items()
    )
    return dict(
        audit_passed=True,
        manifest_sha256=authority,
        authenticated_payloads=payloads,
        nested_references=nested,
        rows=total,
        causal_captures=capture_count,
        journal_events=journal_events,
        work=work,
        model_calls=0,
        optimizer_steps=0,
        method="Independent NumPy/SciPy arithmetic and archive/journal fingerprints, paired outcomes and metrics, v8/v6 measured RMS endpoint Hessian penalties, 23 pre-prediction archives, measured-cache identities, first-acceptable scalar decisions, objective counts and damping/rollback. No Glassbox/JAX imports.",
        limit="Authenticates saved causal ordering and arithmetic; per-trial model loss values are source-bound recorded evidence, not independently regenerated objectives. Does not rerun optimizer or regenerate predictions. Known tapes are not blind generalization.",
        metric_order=list(metrics),
        families=families,
        aggregate=aggregate,
        accuracy_passed=accuracy,
        robustness_passed=robustness,
        cases=rows,
    )


def self_test():
    """Analytic endpoint fixture distinguishes both formats and pre-delay inputs."""
    result = {}
    for version in ("v6", "v8"):
        channels = 11
        terms = channels * (channels + 1) // 2
        values = dict(
            metadata=np.asarray(
                json.dumps(
                    dict(
                        format="glassbox-online-fit-" + version,
                        model=dict(dt_s=0.05, delay_steps=1),
                    )
                )
            ),
            norm_body_mean=np.zeros(9),
            norm_body_scale=np.ones(9),
            norm_motion_bound_scale=np.full(6, 8.0),
            norm_input_mean=np.zeros(1),
            norm_input_scale=np.ones(1),
            norm_output_scale=np.ones(6),
            norm_quadratic_scale=np.ones(terms),
            param_quadratic=np.zeros((terms, 6)),
            scale=np.ones((1, 15)),
        )
        left, right = np.triu_indices(channels)
        values["param_quadratic"][(left == 0) & (right == 0), 0] = 1.0
        values["param_quadratic"][(left == 0) & (right == 9), 0] = 2.0
        past = np.zeros((1, 3, 15))
        past[..., 6:] = np.eye(3).ravel()
        past[..., 0] = 2.0
        up = np.array([[[9.0], [3.0]]])
        future = np.array([[[3.0]]])
        target = past[:, -1:]
        predictions = {}
        for role, count in (("bootstrap", 1), ("recent", 0)):
            for key, value in zip(
                ("past_states", "past_inputs", "future_inputs", "future_states"),
                (past, up, future, target),
            ):
                values[role + "_" + key] = value[:count]
            predictions[role] = target[:count].copy()
        measured = endpoint_components(values, predictions)
        motion, command = 2.0, 3.0
        expected = (
            0.01
            / 4
            * ((2 * motion**2 * 0.05) ** 2 + 2 * (motion * command * 0.05 * 2) ** 2)
        )
        close(measured["domain"][0], motion, "fixture motion")
        close(
            measured["domain"][9:], np.full(2, command), "fixture all issued commands"
        )
        close(measured["prior_loss"], expected, "fixture closed-form Hessian prior")
        assert measured["data_loss"] == 0.0
        result[version] = measured["prior_loss"]
    return dict(
        analytic_fixture_passed=True,
        prior_losses=result,
        model_calls=0,
        optimizer_steps=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("authority", nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else audit(args.root, args.authority)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
