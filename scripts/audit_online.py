"""Audit historical proposals without applying updates; verify arrays or rederive actions."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import COUNTERS, paired_exact
from run_dart import ROOT, clean, write
from verify_baseline import arrays, digest, observed, read, require

PROTOCOL = ROOT / "docs/harness/online-proposal-audit-v1.json"
FIELDS = ("past_states", "past_inputs", "future_inputs", "future_states")
LINKS = {
    "fixedwing-80": [(30, 31), (35, 36)],
    "fixedwing-81": [(63, 64), (149, 150), (156, 157)],
}


def windows(states, inputs, origins, history, horizon):
    return dict(
        zip(
            FIELDS,
            (
                np.stack([states[k - history : k + 1] for k in origins]),
                np.stack([inputs[k - history : k] for k in origins]),
                np.stack([inputs[k : k + horizon] for k in origins]),
                np.stack([states[k + 1 : k + horizon + 1] for k in origins]),
            ),
        )
    )


def post_cache(meta, values, context):
    """Independent NumPy reconstruction, also used by zero-model verification."""
    states = np.concatenate(
        (values["tail_states"][1:], np.asarray(context["truth"], dtype=float)[None])
    )
    inputs = np.concatenate(
        (values["tail_inputs"][1:], np.asarray(context["command"], dtype=float)[None])
    )
    history, horizon = meta["model"]["history_steps"], meta["horizon"]
    new = windows(states, inputs, [history], history, horizon)
    recent = {
        name: np.concatenate((values["recent_" + name], new[name]))[-32:]
        for name in FIELDS
    }
    blocks, weights = [], []
    for role in ("bootstrap", "recent"):
        role_data = (
            {name: values[role + "_" + name] for name in FIELDS}
            if role == "bootstrap"
            else recent
        )
        count = len(role_data["past_states"])
        require(0 < count <= 32, "invalid actual cache size")
        blocks.append(
            {name: value[np.arange(32) % count] for name, value in role_data.items()}
        )
        weights.append(np.where(np.arange(32) < count, 0.5 / count, 0.0))
    return dict(
        states=states,
        inputs=inputs,
        recent=recent,
        data=tuple(
            np.concatenate((blocks[0][name], blocks[1][name])) for name in FIELDS
        ),
        weights=np.concatenate(weights),
    )


def prepare_update(module, session, context):
    """Reconstruct only the observed cache and original conditioning; no proposal."""
    import jax
    import jax.numpy as jnp

    before = session.fingerprint()
    meta, values = session._metadata(), session._arrays()
    require(
        int(context.get("origin", session.cursor)) == session.cursor,
        "wrong update origin",
    )
    cache = post_cache(meta, values, context)
    model = session.model
    original_new = module._windows(
        cache["states"],
        cache["inputs"],
        [model.history_steps],
        model.history_steps,
        session._horizon,
    )
    recent = {
        name: np.concatenate((session._recent[name], original_new[name]))[-32:]
        for name in FIELDS
    }
    original_data, original_weights = module._full_cache(session._bootstrap, recent)
    for name, actual, expected in zip(FIELDS, original_data, cache["data"]):
        paired_exact(actual, expected, "reconstructed cache " + name)
    paired_exact(original_weights, cache["weights"], "reconstructed role weights")
    with jax.enable_x64(True):
        params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
        params, norms, finite = module._recondition(
            params,
            norms,
            tuple(map(jnp.asarray, original_data)),
            jnp.asarray(original_weights),
            delay=model.delay_steps,
            dt_s=model.dt_s,
        )
        result = dict(
            cache,
            params=jax.tree.map(np.asarray, params),
            norms=jax.tree.map(np.asarray, norms),
            scale=np.asarray(session._scale).copy(),
            damping=np.asarray(session._damping, dtype=float),
            delay=model.delay_steps,
            dt_s=model.dt_s,
            conditioning_finite=bool(finite),
        )
    require(session.fingerprint() == before, "conditioning mutated original session")
    return result


def after_update(meta, values, prepared, production):
    """Original scalar decision and copied after-state; never instantiate a learner."""
    from glassbox._learner_arrays import array_fingerprint

    proposal, current, trial, predicted, proposal_finite = production
    finite = (
        bool(prepared["conditioning_finite"])
        and bool(proposal_finite)
        and all(
            np.isfinite(value).all()
            for value in [
                *proposal.values(),
                np.asarray(current),
                np.asarray(trial),
                np.asarray(predicted),
            ]
        )
    )
    current, trial, predicted = map(float, (current, trial, predicted))
    gain = (current - trial) / predicted if finite and predicted > 0 else -np.inf
    accepted = bool(finite and trial < current and gain >= 0.1)
    damping = meta["damping"]
    if not accepted or gain < 0.25:
        damping *= 4.0
    elif gain > 0.75:
        damping *= 0.5
    damping = float(np.clip(damping, 1e-8, 1e8))
    after_meta, after_values = (
        copy.deepcopy(meta),
        {name: np.array(value, copy=True) for name, value in values.items()},
    )
    if accepted:
        after_values.update(
            {"param_" + name: np.asarray(value) for name, value in proposal.items()}
        )
        after_values.update(
            {
                "norm_" + name: np.asarray(value)
                for name, value in prepared["norms"].items()
            }
        )
    after_values.update(tail_states=prepared["states"], tail_inputs=prepared["inputs"])
    after_values.update(
        {"recent_" + name: value for name, value in prepared["recent"].items()}
    )
    for name, increment in dict(
        conditioning_calls=1,
        optimizer_steps=1,
        gradient_calls=1,
        objective_calls=2,
        cg_iterations=4,
        curvature_calls=4,
        accepted_proposals=int(accepted),
    ).items():
        after_meta["counts"][name] += increment
    after_meta["cursor"] += 1
    after_meta["damping"] = damping
    core = {
        name: value
        for name, value in after_values.items()
        if name.startswith(("param_", "norm_"))
    }
    summary = dict(
        accepted=accepted,
        finite=finite,
        conditioning_finite=bool(prepared["conditioning_finite"]),
        current=current,
        trial=trial,
        predicted=predicted,
        gain=gain,
        damping_before=meta["damping"],
        damping_after=damping,
        core_after=array_fingerprint(meta["model"], core),
        session_after=array_fingerprint(after_meta, after_values),
        session_before=array_fingerprint(meta, values),
    )
    return after_meta, after_values, summary


def reconstruct_update(module, session, context):
    """One exact historical production proposal, no observe or session mutation."""
    import jax
    import jax.numpy as jnp

    before = session.fingerprint()
    prepared = prepare_update(module, session, context)
    with jax.enable_x64(True):
        production = module._proposal(
            prepared["params"],
            prepared["norms"],
            prepared["data"],
            prepared["scale"],
            prepared["weights"],
            jnp.asarray(session._damping),
            delay=prepared["delay"],
            dt_s=prepared["dt_s"],
        )
        production = jax.tree.map(np.asarray, production)
    meta, values, summary = after_update(
        session._metadata(), session._arrays(), prepared, production
    )
    require(session.fingerprint() == before, "proposal reconstruction mutated session")
    return dict(
        prepared=prepared,
        production=production,
        after_metadata=meta,
        after_arrays=values,
        summary=summary,
    )


def payload_inventory(protocol):
    result = {}
    for version in protocol["references"]:
        eval_names = ["binding.json", "protocol.json"]
        capture_names = ["binding.json"]
        for case, origins in protocol["origins"].items():
            eval_names.append(f"inputs/{case}.npz")
            eval_names.extend(
                f"{case}/{name}"
                for name in (
                    "case.json",
                    "predictions.npz",
                    "events.jsonl",
                    "initial-online.npz",
                )
            )
            capture_names.extend(
                f"{case}/{origin}/{name}"
                for origin in origins
                for name in ("session.npz", "context.npz", "capture.json")
            )
        for role, names in (("evaluation", eval_names), ("captures", capture_names)):
            for name in names:
                result[f"evidence/{version}/{role}/{name}"] = dict(
                    version=version, role=role, original_path=name
                )
    return result


def check_package_sources(protocol, output):
    """Current loader dependencies must be the adopted, authenticated fifteen files."""
    bound = read(output / "evidence/v6/evaluation/binding.json")
    expected = {
        name: value
        for name, value in bound["source"]["files"].items()
        if name.startswith("src/")
    }
    actual = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (ROOT / "src").rglob("*.py")
    }
    require(
        len(expected) == 15 and actual == expected,
        "current package source inventory differs",
    )
    require(
        Path(importlib.util.find_spec("glassbox").origin).resolve()
        == ROOT / "src/glassbox/__init__.py",
        "use bound package checkout",
    )
    for version, reference in protocol["references"].items():
        source = read(output / f"evidence/{version}/evaluation/binding.json")["source"]
        require(
            source["commit"] == reference["source_commit"],
            "historical source commit differs",
        )
        other = {
            name: value
            for name, value in source["files"].items()
            if name.startswith("src/") and name != "src/glassbox/online.py"
        }
        require(
            other
            == {
                name: value
                for name, value in expected.items()
                if name != "src/glassbox/online.py"
            },
            "historical shared source inventory differs",
        )


def historical_module(output, protocol, version):
    check_package_sources(protocol, output)
    path = output / f"sources/{version}-online.py"
    reference = protocol["references"][version]
    bound = read(output / f"evidence/{version}/evaluation/binding.json")
    require(
        digest(path)
        == reference["online_source_sha256"]
        == bound["source"]["files"]["src/glassbox/online.py"],
        "historical online source differs",
    )
    name = "glassbox._proposal_audit_" + version
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(
        module._FORMAT == "glassbox-online-fit-" + version,
        "historical archive format differs",
    )
    return module


def copy_evidence(output, protocol, git):
    mapping = payload_inventory(protocol)
    for version, reference in protocol["references"].items():
        for role in ("evaluation", "captures"):
            source = Path(reference[role])
            authority = reference[
                "evaluation_manifest_sha256"
                if role == "evaluation"
                else "capture_manifest_sha256"
            ]
            authenticate(source, authority)
            manifest = read(source / "manifest.json")
            destination = output / f"authorities/{version}-{role}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / "manifest.json", destination)
            for name, row in mapping.items():
                if row["version"] == version and row["role"] == role:
                    wanted = manifest["files"][row["original_path"]]
                    destination = output / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / row["original_path"], destination)
                    require(digest(destination) == wanted, "copied evidence differs")
                    row["sha256"] = wanted
        bound = read(output / f"evidence/{version}/evaluation/binding.json")
        require(
            bound["source"]["commit"] == reference["source_commit"],
            "historical source commit differs",
        )
        source_bytes = subprocess.check_output(
            [
                git,
                "-C",
                str(ROOT),
                "show",
                reference["source_commit"] + ":src/glassbox/online.py",
            ]
        )
        require(
            hashlib.sha256(source_bytes).hexdigest()
            == reference["online_source_sha256"]
            == bound["source"]["files"]["src/glassbox/online.py"],
            "archived source hash differs",
        )
        destination = output / f"sources/{version}-online.py"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(source_bytes)
    check_package_sources(protocol, output)
    source = read(output / "evidence/v6/evaluation/binding.json")["source"]["files"]
    for name in source:
        if name.startswith("src/") and name != "src/glassbox/online.py":
            destination = output / "sources/shared" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    write(output / "copied-payloads.json", mapping)


def verify_evidence(output, protocol):
    wanted = payload_inventory(protocol)
    mapping = read(output / "copied-payloads.json")
    require(set(mapping) == set(wanted), "copied payload inventory differs")
    for version, reference in protocol["references"].items():
        for role in ("evaluation", "captures"):
            path = output / f"authorities/{version}-{role}.json"
            authority = reference[
                "evaluation_manifest_sha256"
                if role == "evaluation"
                else "capture_manifest_sha256"
            ]
            require(digest(path) == authority, "copied authority differs")
            manifest = read(path)
            for name, row in wanted.items():
                if row["version"] == version and row["role"] == role:
                    require(
                        mapping[name]
                        == dict(row, sha256=manifest["files"][row["original_path"]]),
                        "copied payload mapping differs",
                    )
                    require(
                        digest(output / name) == mapping[name]["sha256"],
                        "copied payload differs",
                    )
        bound = read(output / f"evidence/{version}/evaluation/binding.json")
        require(
            bound["source"]["commit"] == reference["source_commit"],
            "historical source commit differs",
        )
        require(
            read(output / f"evidence/{version}/evaluation/protocol.json")["id"]
            == "online-fit-" + version,
            "historical evaluation format differs",
        )
        require(
            digest(output / f"sources/{version}-online.py")
            == reference["online_source_sha256"]
            == bound["source"]["files"]["src/glassbox/online.py"],
            "historical source bytes differ",
        )
        for name, sha in bound["source"]["files"].items():
            if name.startswith("src/") and name != "src/glassbox/online.py":
                require(
                    digest(output / "sources/shared" / name) == sha,
                    "shared source bytes differ",
                )
        require(
            read(output / "binding.json")["runtime"]
            == bound["runtime"]
            == read(output / f"evidence/{version}/captures/binding.json")["runtime"],
            "historical runtime differs",
        )


def original_point(output, protocol, version, case, origin):
    """Authenticate all pre-update identities using only archive arrays and tape."""
    from glassbox._learner_arrays import array_fingerprint, load_arrays

    evaluation = output / f"evidence/{version}/evaluation"
    point = output / f"evidence/{version}/captures/{case}/{origin}"
    meta, values = load_arrays(point / "session.npz")
    context, identity = arrays(point / "context.npz"), read(point / "capture.json")
    data, info = (
        arrays(evaluation / case / "predictions.npz"),
        read(evaluation / case / "case.json"),
    )
    stream = arrays(evaluation / f"inputs/{case}.npz")
    events = {
        (event["index"], event["phase"]): event
        for event in (
            json.loads(line)
            for line in (evaluation / case / "events.jsonl").read_text().splitlines()
        )
        if "report" in event
    }
    found = np.flatnonzero(data["origin"] == origin)
    require(len(found) == 1, "missing scored origin")
    index = int(found[0])
    before, after = (
        events[origin, "predicted"]["report"],
        events[origin, "assimilated"]["report"],
    )
    require(
        events[origin, "predicted"]["model"] == data["model_before"][index]
        and events[origin, "assimilated"]["model"] == data["model_after"][index],
        "journal model identity differs",
    )
    core = {
        name: value
        for name, value in values.items()
        if name.startswith(("param_", "norm_"))
    }
    core_identity = array_fingerprint(meta["model"], core)
    require(
        meta["format"] == "glassbox-online-fit-" + version
        and meta["cursor"] == origin
        and meta["initial_cursor"] == info["first"],
        "capture causal stage/version differs",
    )
    require(
        identity
        == dict(
            origin=origin,
            time_s=float(stream["time_s"][origin]),
            session_fingerprint=array_fingerprint(meta, values),
            model_fingerprint=core_identity,
            report=before,
        ),
        "capture identity differs",
    )
    require(
        core_identity == data["model_before"][index],
        "capture core differs",
    )
    for name in COUNTERS:
        value = (
            origin - info["first"] if name == "observations" else meta["counts"][name]
        )
        require(
            value == before[name] == data[name + "_before"][index],
            "capture counter differs",
        )
        require(after[name] == data[name + "_after"][index], "after counter differs")
    require(meta["damping"] == before["damping"], "before damping differs")
    require(
        all(data[key][index] for key in ("predicted", "revealed", "assimilated")),
        "incomplete historical transition",
    )
    states, commands = observed(stream["states"]), stream["commands"]
    h, horizon = meta["model"]["history_steps"], meta["horizon"]
    wanted = dict(
        origin=np.asarray(origin, dtype=np.int64),
        past_states=states[origin - h : origin + 1],
        past_inputs=commands[origin - h : origin],
        command=commands[origin],
        truth=states[origin + 1],
        recorded_prediction=data["candidate"][index],
    )
    require(set(wanted) == set(context), "context inventory differs")
    for name, value in wanted.items():
        paired_exact(context[name], value, "context " + name)
    paired_exact(context["truth"], data["truth"][index], "recorded truth")
    paired_exact(
        values["tail_states"],
        states[origin - h - horizon : origin + 1],
        "causal state tail",
    )
    paired_exact(
        values["tail_inputs"],
        commands[origin - h - horizon : origin],
        "causal input tail",
    )
    first_meta, first = load_arrays(evaluation / case / "initial-online.npz")
    for name in ("contract", "initial_cursor", "initial_count", "horizon"):
        require(meta[name] == first_meta[name], "initial session identity differs")
    for name in first:
        if name == "scale" or name.startswith("bootstrap_"):
            paired_exact(values[name], first[name], "fixed cache " + name)
    count = min(32, origin - info["first"])
    recent = (
        windows(
            states,
            commands,
            range(origin - horizon + 1 - count, origin - horizon + 1),
            h,
            horizon,
        )
        if count
        else {name: first["bootstrap_" + name][:0] for name in FIELDS}
    )
    for name in FIELDS:
        paired_exact(values["recent_" + name], recent[name], "causal recent " + name)
    return dict(
        meta=meta,
        values=values,
        context=context,
        before=before,
        after=after,
        after_core=str(data["model_after"][index]),
        point=point,
    )


def check_after(original, reconstruction, output, protocol, version, case, origin):
    summary, meta = reconstruction["summary"], reconstruction["after_metadata"]
    require(
        summary["core_after"] == original["after_core"], "historical after core differs"
    )
    report = dict(
        original["before"],
        **meta["counts"],
        cursor=meta["cursor"],
        observations=meta["cursor"] - meta["initial_cursor"],
        recent_windows=len(reconstruction["after_arrays"]["recent_past_states"]),
        tail_rows=len(reconstruction["after_arrays"]["tail_states"]),
        damping=meta["damping"],
    )
    require(report == original["after"], "historical after report differs")
    adjacent = origin + 1 in protocol["origins"][case]
    if adjacent:
        following = read(
            output / f"evidence/{version}/captures/{case}/{origin + 1}/capture.json"
        )
        require(
            summary["session_after"] == following["session_fingerprint"],
            "adjacent full session differs",
        )
    return adjacent


def save_point(path, result, diagnostic_summary, diagnostic_arrays):
    path.mkdir(parents=True)
    p = result["prepared"]
    checkpoint(
        path / "prepared.npz",
        **{"param_" + k: v for k, v in p["params"].items()},
        **{"norm_" + k: v for k, v in p["norms"].items()},
        **{"data_" + k: v for k, v in zip(FIELDS, p["data"])},
        weights=p["weights"],
        scale=p["scale"],
        damping=p["damping"],
        conditioning_finite=np.asarray(p["conditioning_finite"]),
    )
    proposal, current, trial, predicted, finite = result["production"]
    checkpoint(
        path / "production.npz",
        **{"param_" + k: v for k, v in proposal.items()},
        current=current,
        trial=trial,
        predicted=predicted,
        finite=finite,
    )
    write(path / "reconstruction.json", result["summary"])
    write(path / "diagnostics.json", diagnostic_summary)
    checkpoint(path / "diagnostics.npz", **diagnostic_arrays)


def saved_prepared(path, original):
    saved = arrays(path / "prepared.npz")
    cache = post_cache(original["meta"], original["values"], original["context"])
    params = {
        name[6:]: value for name, value in saved.items() if name.startswith("param_")
    }
    norms = {
        name[5:]: value for name, value in saved.items() if name.startswith("norm_")
    }
    expected = (
        {"param_" + name for name in params}
        | {"norm_" + name for name in norms}
        | {"data_" + name for name in FIELDS}
        | {"weights", "scale", "damping", "conditioning_finite"}
    )
    require(set(saved) == expected, "prepared array inventory differs")
    require(
        set(params)
        == {name[6:] for name in original["values"] if name.startswith("param_")},
        "prepared parameter inventory differs",
    )
    require(
        set(norms)
        == {name[5:] for name in original["values"] if name.startswith("norm_")},
        "prepared normalizer inventory differs",
    )
    for name, value in zip(FIELDS, cache["data"]):
        paired_exact(saved["data_" + name], value, "post-reveal cache " + name)
    paired_exact(saved["weights"], cache["weights"], "post-reveal weights")
    paired_exact(saved["scale"], original["values"]["scale"], "fixed loss scale")
    paired_exact(
        saved["damping"], np.asarray(original["meta"]["damping"]), "original damping"
    )
    return dict(
        cache,
        params=params,
        norms=norms,
        scale=saved["scale"],
        damping=saved["damping"],
        delay=original["meta"]["model"]["delay_steps"],
        dt_s=original["meta"]["model"]["dt_s"],
        conditioning_finite=bool(saved["conditioning_finite"]),
    )


def check_diagnostic_binding(prepared, production, summary, values):
    """Bind self-consistent numerical diagnostics to the historical proposal inputs."""
    theta, proposed, schema, start = [], [], [], 0
    for name in sorted(prepared["params"]):
        value = np.asarray(prepared["params"][name])
        stop = start + value.size
        schema.append(
            dict(path=f"[{name!r}]", shape=list(value.shape), start=start, stop=stop)
        )
        theta.append(value.ravel())
        require(name in production[0], "production parameter inventory differs")
        proposed.append(np.asarray(production[0][name]).ravel())
        start = stop
    require(
        set(production[0]) == set(prepared["params"]),
        "production parameter inventory differs",
    )
    require(summary["parameter_schema"] == schema, "parameter schema differs")
    paired_exact(
        values["theta"], np.concatenate(theta), "diagnostic conditioned parameters"
    )
    paired_exact(
        values["production_theta"],
        np.concatenate(proposed),
        "diagnostic production parameters",
    )
    for name, actual in zip(
        ("current", "trial", "predicted", "finite"), production[1:]
    ):
        paired_exact(
            values["production_" + name],
            np.asarray(actual),
            "diagnostic production " + name,
        )
    paired_exact(values["damping"], prepared["damping"], "diagnostic damping")
    require(
        values["raw"].shape == prepared["data"][3].shape,
        "diagnostic residual shape differs",
    )
    require(
        np.allclose(
            values["weight"],
            prepared["weights"][:, None, None] / (prepared["data"][3].shape[1] * 3),
            atol=1e-10,
            rtol=1e-8,
        ),
        "diagnostic role weighting differs",
    )


def check_conditioned_chart(prepared, original):
    old_p = {
        name[6:]: value
        for name, value in original["values"].items()
        if name.startswith("param_")
    }
    old_n = {
        name[5:]: value
        for name, value in original["values"].items()
        if name.startswith("norm_")
    }
    params, norms = prepared["params"], prepared["norms"]
    if not prepared["conditioning_finite"]:
        for name in params:
            paired_exact(params[name], old_p[name], "invalid conditioning parameters")
        for name in norms:
            paired_exact(norms[name], old_n[name], "invalid conditioning normalizers")
        return
    for name in norms:
        if name in ("feature_scale", "quadratic_scale", "output_scale"):
            require(
                np.isfinite(norms[name]).all() and np.all(norms[name] >= old_n[name]),
                "conditioned scale decreases or is nonfinite",
            )
        else:
            paired_exact(norms[name], old_n[name], "immutable normalizer")
    paired_exact(
        norms["feature_scale"][-8:], old_n["feature_scale"][-8:], "fixed hidden scales"
    )
    sf = norms["feature_scale"] / old_n["feature_scale"]
    sq = norms["quadratic_scale"] / old_n["quadratic_scale"]
    so = old_n["output_scale"] / norms["output_scale"]
    expected = dict(old_p)
    expected.update(
        linear=sf[:, None] * old_p["linear"] * so[None, :],
        quadratic=sq[:, None] * old_p["quadratic"] * so[None, :],
        bias=old_p["bias"] * so,
        w2=old_p["w2"] * so[None, :],
        w1=sf[:, None] * old_p["w1"],
        memory=sf[:, None] * old_p["memory"],
    )
    for name, value in expected.items():
        require(
            np.allclose(params[name], value, rtol=1e-8, atol=1e-10),
            "conditioned coordinate compensation differs",
        )


def probe_arguments(prepared):
    return tuple(
        prepared[name]
        for name in ("params", "norms", "data", "scale", "weights", "damping")
    ), dict(delay=prepared["delay"], dt_s=prepared["dt_s"])


def links(output, protocol):
    result = []
    for version in protocol["references"]:
        for case, pairs in LINKS.items():
            for producer, anchor in pairs:
                outcome = read(
                    output / f"{version}/{case}/{producer}/reconstruction.json"
                )
                target = read(
                    output / f"evidence/{version}/captures/{case}/{anchor}/capture.json"
                )
                require(
                    outcome["core_after"] == target["model_fingerprint"],
                    "anchor-producing link differs",
                )
                result.append(
                    dict(
                        version=version,
                        case=case,
                        update_origin=producer,
                        prediction_origin=anchor,
                        model_fingerprint=outcome["core_after"],
                    )
                )
    return result


def verify_points(output, protocol, *, recompute=False):
    from _online_proposal_verification import verify_arrays

    count = adjacent = 0
    rows = []
    modules = (
        {
            version: historical_module(output, protocol, version)
            for version in protocol["references"]
        }
        if recompute
        else {}
    )
    for version in protocol["references"]:
        for case, origins in protocol["origins"].items():
            for origin in origins:
                original = original_point(output, protocol, version, case, origin)
                path = output / f"{version}/{case}/{origin}"
                prepared = saved_prepared(path, original)
                saved = arrays(path / "production.npz")
                production = (
                    {
                        name[6:]: value
                        for name, value in saved.items()
                        if name.startswith("param_")
                    },
                    *(
                        saved[name]
                        for name in ("current", "trial", "predicted", "finite")
                    ),
                )
                check_conditioned_chart(prepared, original)
                meta, values, summary = after_update(
                    original["meta"], original["values"], prepared, production
                )
                require(
                    clean(summary) == read(path / "reconstruction.json"),
                    "reconstruction summary differs",
                )
                adjacent += check_after(
                    original,
                    dict(after_metadata=meta, after_arrays=values, summary=summary),
                    output,
                    protocol,
                    version,
                    case,
                    origin,
                )
                diag_summary, diag_arrays = (
                    read(path / "diagnostics.json"),
                    arrays(path / "diagnostics.npz"),
                )
                check_diagnostic_binding(
                    prepared, production, diag_summary, diag_arrays
                )
                verified = verify_arrays(diag_summary, diag_arrays)
                if recompute:
                    from _online_proposal_diagnostics import recompute_actions

                    module = modules[version]
                    session = module.OnlineFit.load(original["point"] / "session.npz")
                    before = session.fingerprint()
                    derived = prepare_update(module, session, original["context"])
                    for group in ("params", "norms"):
                        for name, value in derived[group].items():
                            require(
                                np.allclose(
                                    value, prepared[group][name], atol=1e-10, rtol=1e-8
                                ),
                                "recomputed conditioning differs",
                            )
                    require(
                        derived["conditioning_finite"]
                        == prepared["conditioning_finite"],
                        "recomputed conditioning flag differs",
                    )
                    args, kwargs = probe_arguments(derived)
                    verified = recompute_actions(
                        module,
                        *args,
                        **kwargs,
                        summary=diag_summary,
                        arrays=diag_arrays,
                    )
                    require(
                        session.fingerprint() == before,
                        "derivative recomputation mutated session",
                    )
                rows.append(
                    dict(version=version, case=case, origin=origin, result=verified)
                )
                count += 1
    return dict(
        snapshots=count,
        adjacent_full_session_checks=adjacent,
        links=links(output, protocol),
        rows=rows,
    )


def verify(output, authority, *, recompute=False, git="git"):
    authenticate(output, authority)
    require(not (output / "failure.json").exists(), "audit attempt failed")
    protocol = read(output / "protocol.json")
    require(protocol["id"] == "online-proposal-audit-v1", "unsupported audit protocol")
    require(
        digest(output / "protocol.json")
        == read(output / "binding.json")["protocol_sha256"],
        "protocol binding differs",
    )
    verify_evidence(output, protocol)
    check_source(read(output / "binding.json"))
    if recompute:
        import collect_throw

        collect_throw.GIT = git
        current = binding(PROTOCOL)
        require(
            current["runtime"] == read(output / "binding.json")["runtime"],
            "recomputation runtime differs",
        )
    result = verify_points(output, protocol, recompute=recompute)
    summary = read(output / "summary.json")
    require(
        summary["complete"]
        and result["snapshots"] == summary["snapshots"] == 46
        and result["adjacent_full_session_checks"]
        == summary["adjacent_full_session_checks"]
        == 28
        and result["links"] == summary["links"],
        "audit coverage differs",
    )
    return dict(
        verified=True,
        mode="recompute" if recompute else "verify",
        **result,
        observe_calls=0,
        initialization_calls=0,
        production_proposal_calls=0,
        diagnostic_pcg_calls=0,
        conditioning_calls=result["snapshots"] if recompute else 0,
        model_calls="counted per snapshot in rows" if recompute else 0,
    )


def run(output, protocol=PROTOCOL, git="git"):
    import collect_throw
    from _online_proposal_diagnostics import diagnose_proposal

    output, protocol = output.resolve(), protocol.resolve()
    p = read(protocol)
    require(p["id"] == "online-proposal-audit-v1", "unsupported audit protocol")
    for reference in p["references"].values():
        for name in ("evaluation", "captures"):
            source = Path(reference[name]).resolve()
            require(
                not output.is_relative_to(source) and not source.is_relative_to(output),
                "output overlaps source evidence",
            )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "attempt.json", dict(protocol=str(protocol)))
    try:
        collect_throw.GIT = git
        bound = binding(protocol)
        shutil.copyfile(protocol, output / "protocol.json")
        write(output / "binding.json", bound)
        copy_evidence(output, p, git)
        verify_evidence(output, p)
        count = adjacent = 0
        for version in p["references"]:
            module = historical_module(output, p, version)
            for case, origins in p["origins"].items():
                for origin in origins:
                    print(f"auditing {version} {case} origin {origin}", flush=True)
                    original = original_point(output, p, version, case, origin)
                    session = module.OnlineFit.load(original["point"] / "session.npz")
                    before = session.fingerprint()
                    result = reconstruct_update(module, session, original["context"])
                    adjacent += check_after(
                        original, result, output, p, version, case, origin
                    )
                    args, kwargs = probe_arguments(result["prepared"])
                    summary, values = diagnose_proposal(
                        module, *args, **kwargs, production=result["production"]
                    )
                    require(session.fingerprint() == before, "audit mutated session")
                    save_point(
                        output / f"{version}/{case}/{origin}", result, summary, values
                    )
                    count += 1
        check_source(bound)
        write(
            output / "summary.json",
            dict(
                complete=True,
                snapshots=count,
                adjacent_full_session_checks=adjacent,
                links=links(output, p),
                production_proposal_calls=count,
                conditioning_calls=count,
                observe_calls=0,
                initialization_calls=0,
                diagnostic_pcg_calls=count,
            ),
        )
    except BaseException as error:
        write(output / "failure.json", dict(complete=False, error=repr(error)))
        raise
    finally:
        authority = seal(output, "glassbox-online-proposal-audit-v1")
        print(dict(output=str(output), manifest_sha256=authority), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--protocol", type=Path, default=PROTOCOL)
    execute.add_argument("--git", default="git")
    for name in ("verify", "recompute"):
        command = sub.add_parser(name)
        command.add_argument("output", type=Path)
        command.add_argument("--manifest-sha256", required=True)
        if name == "recompute":
            command.add_argument("--git", default="git")
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.protocol, args.git)
    else:
        print(
            clean(
                verify(
                    args.output,
                    args.manifest_sha256,
                    recompute=args.command == "recompute",
                    git=getattr(args, "git", "git"),
                )
            )
        )


if __name__ == "__main__":
    main()
