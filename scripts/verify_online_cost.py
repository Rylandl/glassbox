"""Independent sealed-array v8 cost-profile audit; never executes the learner."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from verify_online_v8 import (
    accounting,
    authenticate,
    canonical,
    check_cache,
    fingerprint,
    load,
    proposal_decision,
    read,
    same,
    session,
    sha,
    verify_bootstrap,
    verify_transition,
)

RTOL = 2e-8
ATOL = 2e-10
FIELDS = ("past_states", "past_inputs", "future_inputs", "future_states")
SCOPES = (
    "conditioning",
    "prior",
    "residual",
    "linearization",
    "setup",
    "solve",
    "trust",
    "proposal",
    "instrumented",
    "cached_jvp",
    "cached_vjp",
    "cached_curvature",
)
PUBLIC = ("public_observe", "public_snapshot", "public_combined")
STAGES = {
    "setup": (
        "flat",
        "prior",
        "preconditioner",
        "raw",
        "weight",
        "root_weight",
        "irls",
        "gradient",
    ),
    "solve": ("delta", "finite"),
    "trust": (
        "delta",
        "linearized",
        "radius",
        "shrink",
        "linear_decrease",
        "quadratic_cost",
        "current_loss",
        "direction_finite",
    ),
}
EVIDENCE = (
    "direction_finite",
    "trust_shrink",
    "trial_evaluations",
    "trial_losses",
    "trial_predicted",
    "trial_finite",
    "selected_alpha",
)
NATIVE = ("current_loss", "trial_loss", "predicted_reduction", "finite")


def numeric_same(actual, expected, label):
    """Frozen tolerance with exact shape/dtype/finite masks and discrete values."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    assert np.array_equal(np.isfinite(actual), np.isfinite(expected)), label
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), label
    assert np.array_equal(np.isposinf(actual), np.isposinf(expected)), label
    assert np.array_equal(np.isneginf(actual), np.isneginf(expected)), label
    if actual.dtype.kind in "biu":
        assert np.array_equal(actual, expected), label
        return 0.0
    finite = np.isfinite(actual)
    scaled = np.abs(actual[finite] - expected[finite]) / (
        ATOL + RTOL * np.abs(expected[finite])
    )
    maximum = float(np.max(scaled, initial=0.0))
    assert maximum <= 1, (label, maximum)
    return maximum


def statistics(values):
    values = np.asarray(values)
    assert values.shape == (21,) and values.dtype == np.float64
    assert np.isfinite(values).all() and np.all(values >= 0)
    return dict(
        count=21,
        median_s=float(np.median(values)),
        p95_s=float(np.quantile(values, 0.95)),
        max_s=float(np.max(values)),
        min_s=float(np.min(values)),
    )


def verify_timings(timings, saved, genuine):
    wanted = set(SCOPES) | (set(PUBLIC) if genuine else set())
    assert set(timings) == wanted, "timing scope roster differs"
    assert set(saved) == wanted, "timing statistic roster differs"
    for name in sorted(wanted):
        times = timings[name]
        assert times.shape == (25,) and times.dtype == np.float64, name
        assert np.isfinite(times).all() and np.all(times >= 0), name
        assert saved[name] == statistics(times[4:]), name
    if genuine:
        # These alone are nested timestamps from a single call and are additive.
        np.testing.assert_allclose(
            timings["public_combined"],
            timings["public_observe"] + timings["public_snapshot"],
            rtol=1e-12,
            atol=2e-12,
        )
    return len(wanted) * 25


def expected_roster(protocol, parent):
    result = []
    for case in protocol["points"]["case_order"]:
        for kind in ("initial", "final"):
            relative = f"{case}/{kind}-online.npz"
            meta, _, _, _ = session(parent / relative)
            result.append(
                dict(
                    case=case,
                    family=read(parent / case / "case.json")["family"],
                    kind=kind,
                    origin=meta["cursor"],
                    session=relative,
                    id=f"{case}/{kind}",
                )
            )
        for origin in protocol["points"]["capture_origins"].get(case, []):
            result.append(
                dict(
                    case=case,
                    family=read(parent / case / "case.json")["family"],
                    kind="capture",
                    origin=origin,
                    session=f"{case}/{origin}/session.npz",
                    id=f"{case}/capture-{origin}",
                )
            )
    assert len(result) == 35
    return result


def events_by_origin(case):
    events = {}
    for line in (case / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        key = (event["index"], event["phase"])
        assert key not in events
        events[key] = event
    return events


def prepared_cache(meta, values, source, genuine):
    """Rebuild observe's bounded window without invoking production code."""
    recent = {key: values["recent_" + key] for key in FIELDS}
    if genuine:
        cursor, history, horizon = (
            meta["cursor"],
            meta["model"]["history_steps"],
            meta["horizon"],
        )
        states = np.concatenate(
            (values["tail_states"][1:], canonical(source["states"][[cursor + 1]]))
        )
        commands = np.concatenate(
            (values["tail_inputs"][1:], source["commands"][[cursor]])
        )
        new = dict(
            past_states=states[: history + 1],
            past_inputs=commands[:history],
            future_inputs=commands[history : history + horizon],
            future_states=states[history + 1 : history + horizon + 1],
        )
        recent = {
            key: np.concatenate((recent[key], new[key][None]))[-32:] for key in FIELDS
        }
    bootstrap = {key: values["bootstrap_" + key] for key in FIELDS}
    blocks, weights = [], []
    for role in (bootstrap, recent):
        count = len(role["past_states"])
        assert 1 <= count <= 32
        indexes = np.arange(32) % count
        blocks.append({key: role[key][indexes] for key in FIELDS})
        weights.append(np.where(np.arange(32) < count, 0.5 / count, 0.0))
    return (
        {key: np.concatenate((blocks[0][key], blocks[1][key])) for key in FIELDS},
        np.concatenate(weights),
    )


def verify_prepared(prepared, meta, values, source, genuine):
    wanted = {"scale", "weights", "damping", "conditioning_finite", "p", "c"}
    wanted |= {"data/" + name for name in FIELDS}
    for role, archive_prefix in (("params", "param_"), ("norms", "norm_")):
        names = {
            key[len(archive_prefix) :]
            for key in values
            if key.startswith(archive_prefix)
        }
        wanted |= {
            f"{prefix}{role}/{name}" for prefix in ("", "raw_") for name in names
        }
        for name in names:
            same(prepared[f"raw_{role}/{name}"], values[archive_prefix + name])
    assert set(prepared) == wanted, "prepared input roster differs"
    same(prepared["scale"], values["scale"])
    assert (
        prepared["damping"].shape == ()
        and float(prepared["damping"]) == meta["damping"]
    )
    assert (
        prepared["conditioning_finite"].shape == ()
        and prepared["conditioning_finite"].dtype == bool
    )
    data, weights = prepared_cache(meta, values, source, genuine)
    same(prepared["weights"], weights)
    for key in FIELDS:
        if "states" in key:
            np.testing.assert_allclose(
                prepared["data/" + key], data[key], rtol=3e-10, atol=3e-12
            )
        else:
            same(prepared["data/" + key], data[key])


def subset(arrays, prefix):
    return {
        key[len(prefix) :]: value
        for key, value in arrays.items()
        if key.startswith(prefix)
    }


def compare_group(arrays, left, right):
    a, b = subset(arrays, left), subset(arrays, right)
    assert a and set(a) == set(b), (left, right, "array roster differs")
    errors = []
    for key in a:
        if key.endswith("selected_alpha"):
            same(a[key], b[key])
            errors.append(0.0)
        else:
            errors.append(numeric_same(a[key], b[key], left + key))
    return max(errors)


def verify_native_decision(arrays, conditioning_finite):
    get = lambda name: arrays["native/" + name]  # noqa: E731
    for name in (*NATIVE, *["evidence/" + key for key in EVIDENCE]):
        value = get(name)
        vector = name in {
            "evidence/trial_losses",
            "evidence/trial_predicted",
            "evidence/trial_finite",
        }
        assert value.shape == ((5,) if vector else ()), (name, "shape differs")
        if name in {"finite", "evidence/direction_finite", "evidence/trial_finite"}:
            assert value.dtype == bool, (name, "boolean dtype differs")
        elif name == "evidence/trial_evaluations":
            assert value.dtype.kind == "i", (name, "integer dtype differs")
        else:
            assert value.dtype == np.float64, (name, "float dtype differs")
    count = int(get("evidence/trial_evaluations"))
    assert 1 <= count <= 5
    losses, predictions, finite = (
        get("evidence/" + name)
        for name in ("trial_losses", "trial_predicted", "trial_finite")
    )
    assert losses.shape == predictions.shape == finite.shape == (5,)
    assert finite.dtype == bool
    assert np.isnan(losses[count:]).all() and np.isnan(predictions[count:]).all()
    assert not finite[count:].any()
    optional = lambda x: float(x) if np.isfinite(x) else None  # noqa: E731
    report = dict(
        current_loss=optional(get("current_loss")),
        conditioning_finite=bool(conditioning_finite),
        direction_finite=bool(get("evidence/direction_finite")),
        trust_shrink=optional(get("evidence/trust_shrink")),
        trial_evaluations=count,
        selected_alpha=float(get("evidence/selected_alpha")),
        trials=[
            dict(
                alpha=alpha,
                loss=optional(losses[index]),
                predicted_reduction=optional(predictions[index]),
                finite=bool(finite[index]),
            )
            for index, alpha in enumerate((1.0, 0.5, 0.25, 0.125, 0.0625)[:count])
        ],
    )
    proposal_decision(report)
    assert bool(get("finite")) == (
        bool(conditioning_finite)
        and report["direction_finite"]
        and bool(finite[count - 1])
    )
    numeric_same(get("trial_loss"), losses[count - 1], "last trial loss")
    numeric_same(get("predicted_reduction"), predictions[count - 1], "last prediction")
    return report


def verify_components(arrays, prepared):
    expected_keys = {
        "prior",
        "residual",
        "linearization/raw",
        "linearization/tape_hashes",
        "linearization/tape_nbytes",
        "cached_jvp",
        "cached_vjp",
        "cached_curvature",
        "conditioning/finite",
    }
    for role in ("params", "norms"):
        expected_keys |= {
            "conditioning/" + role + "/" + key for key in subset(prepared, role + "/")
        }
    native_keys = set(NATIVE) | {"evidence/" + key for key in EVIDENCE}
    native_keys |= {"params/" + key for key in subset(prepared, "params/")}
    for prefix in ("native/", "instrumented/native/"):
        expected_keys |= {prefix + key for key in native_keys}
    for stage, fields in STAGES.items():
        for prefix in (stage + "/", "instrumented/" + stage + "/"):
            expected_keys |= {prefix + key for key in fields}
    assert set(arrays) == expected_keys, "complete component output roster differs"
    for stage, fields in STAGES.items():
        for prefix in (stage + "/", "instrumented/" + stage + "/"):
            assert set(subset(arrays, prefix)) == set(fields), (
                prefix,
                "stage roster differs",
            )
    native_names = set(NATIVE) | {"evidence/" + key for key in EVIDENCE}
    native_names |= {"params/" + key for key in subset(prepared, "params/")}
    assert set(subset(arrays, "native/")) == native_names, (
        "native output roster differs"
    )
    errors = {
        "instrumented_native": compare_group(arrays, "instrumented/native/", "native/")
    }
    for stage in ("setup", "solve", "trust"):
        errors[stage] = compare_group(
            arrays, stage + "/", "instrumented/" + stage + "/"
        )
    for role in ("params", "norms"):
        conditioned = subset(arrays, "conditioning/" + role + "/")
        expected = subset(prepared, role + "/")
        assert set(conditioned) == set(expected) and conditioned
        for name in conditioned:
            same(conditioned[name], expected[name])
    same(arrays["conditioning/finite"], prepared["conditioning_finite"])
    errors["linearization_raw"] = numeric_same(
        arrays["linearization/raw"],
        arrays["instrumented/setup/raw"],
        "linearization residual",
    )
    errors["residual"] = numeric_same(
        arrays["residual"], arrays["instrumented/setup/raw"], "raw residual"
    )
    errors["prior"] = numeric_same(
        arrays["prior"], arrays["instrumented/setup/prior"], "prior diagonal"
    )
    setup = subset(arrays, "setup/")
    errors["initial_direction"] = numeric_same(
        prepared["p"],
        -setup["gradient"] / setup["preconditioner"],
        "cached action direction",
    )
    errors["cached_c"] = numeric_same(
        prepared["c"],
        setup["weight"] * setup["irls"] * arrays["cached_jvp"],
        "cached adjoint cotangent",
    )
    lhs = np.asarray(np.vdot(arrays["cached_jvp"], prepared["c"]))
    rhs = np.asarray(np.vdot(prepared["p"], arrays["cached_vjp"]))
    errors["adjoint"] = numeric_same(lhs, rhs, "cached adjoint identity")
    errors["curvature"] = numeric_same(
        arrays["cached_curvature"],
        arrays["cached_vjp"] + setup["preconditioner"] * prepared["p"],
        "cached curvature arithmetic",
    )
    report = verify_native_decision(arrays, prepared["conditioning_finite"])
    return errors, report


def verify_replays(records, before, after):
    assert len(records) == 25, "replay count differs"
    verify_transition(before["report"], after["report"])
    for index, record in enumerate(records):
        assert record["index"] == index
        assert record["phase"] == (
            "qualification" if index == 0 else "warmup" if index < 4 else "timed"
        )
        assert record["model"] == after["model"], "replayed model differs"
        assert record["report"] == after["report"], "replayed report differs"
        accounting(record["report"], record["report"]["observations"])
    return len(records)


def prefix_ast(source, scope):
    """Independently construct the permitted source transformation, without exec."""
    tree = ast.parse(source)
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_proposal"
    )
    function = copy.deepcopy(original)
    function.name = "_profile_" + scope + "_kernel"
    function.body = []
    endpoints = {}
    for index, statement in enumerate(original.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = ast.unparse(statement.targets[0]).strip("()")
            if name == "gradient":
                endpoints[index] = "setup"
            elif name == "direction_finite":
                endpoints[index] = "trust"
            elif name == "delta, _, _, _, finite":
                assert ast.unparse(statement.value.func) == "jax.lax.fori_loop"
                assert [
                    ast.literal_eval(item) for item in statement.value.args[:2]
                ] == [0, 16]
                endpoints[index] = "solve"
    assert tuple(endpoints.values()) == tuple(STAGES)
    for index, statement in enumerate(original.body):
        statement = copy.deepcopy(statement)
        if isinstance(statement, ast.Return):
            assert scope == "instrumented"
            statement.value = ast.Tuple(
                elts=[
                    statement.value,
                    ast.Dict(
                        keys=[ast.Constant(stage) for stage in STAGES],
                        values=[
                            ast.Name(id="_profile_" + stage, ctx=ast.Load())
                            for stage in STAGES
                        ],
                    ),
                ],
                ctx=ast.Load(),
            )
        function.body.append(statement)
        if index in endpoints:
            stage = endpoints[index]
            checkpoint = ast.Assign(
                targets=[ast.Name(id="_profile_" + stage, ctx=ast.Store())],
                value=ast.Dict(
                    keys=[ast.Constant(name) for name in STAGES[stage]],
                    values=[
                        ast.Name(id=name, ctx=ast.Load()) for name in STAGES[stage]
                    ],
                ),
            )
            function.body.append(checkpoint)
            if scope == stage:
                function.body.append(
                    ast.Return(value=ast.Name(id="_profile_" + stage, ctx=ast.Load()))
                )
                break
    return ast.dump(ast.Module(body=[function], type_ignores=[]))


def verify_generated(root, production, helpers):
    """Validate exact production-derived prefixes; authenticate recorded helpers."""
    spec = read(root / "generated.json")
    files = spec.get("files", spec)
    wanted = {
        *STAGES,
        "instrumented",
        "linearization",
        "cached_jvp",
        "cached_vjp",
        "cached_curvature",
    }
    assert {Path(name).stem for name in files} == wanted
    helper_ast = ast.parse(helpers)
    helper_functions = {
        node.name: node for node in helper_ast.body if isinstance(node, ast.FunctionDef)
    }
    for name, wanted_hash in files.items():
        path = root / "generated" / name
        assert sha(path) == wanted_hash, "generated source hash differs"
        scope = path.stem
        saved_ast = ast.dump(ast.parse(path.read_text()))
        if scope in {*STAGES, "instrumented"}:
            expected = prefix_ast(production, scope)
        else:
            expected = ast.dump(
                ast.Module(body=[helper_functions["_" + scope]], type_ignores=[])
            )
        assert saved_ast == expected, "generated arithmetic differs: " + scope
    return len(files)


def session_report(meta, values):
    return dict(
        recipe=meta["recipe"],
        cursor=meta["cursor"],
        observations=meta["cursor"] - meta["initial_cursor"],
        initialized_transition_count=meta["initial_count"],
        initializers=1,
        ridge_solves=2,
        bootstrap_windows=len(values["bootstrap_past_states"]),
        recent_windows=len(values["recent_past_states"]),
        tail_rows=len(values["tail_states"]),
        history_steps=meta["model"]["history_steps"],
        training_horizon_steps=meta["horizon"],
        **meta["counts"],
        damping=meta["damping"],
        last_proposal=meta["last_proposal"],
        envelope=dict(available=False),
    )


def replay_session(meta, values, context, after, outputs, prepared):
    """Independent session bookkeeping from authentic inputs and saved outputs."""
    meta, values = (
        copy.deepcopy(meta),
        {key: value.copy() for key, value in values.items()},
    )
    states = np.concatenate((values["tail_states"][1:], context["truth"][None]))
    commands = np.concatenate((values["tail_inputs"][1:], context["command"][None]))
    history, horizon = meta["model"]["history_steps"], meta["horizon"]
    new = dict(
        past_states=states[: history + 1],
        past_inputs=commands[:history],
        future_inputs=commands[history : history + horizon],
        future_states=states[history + 1 : history + horizon + 1],
    )
    for field in FIELDS:
        values["recent_" + field] = np.concatenate(
            (values["recent_" + field], new[field][None])
        )[-32:]
    values["tail_states"], values["tail_inputs"] = states, commands
    report = after["report"]
    if proposal_decision(report["last_proposal"])["accepted"]:
        for name, value in subset(outputs, "native/params/").items():
            values["param_" + name] = value
        for name, value in subset(prepared, "norms/").items():
            values["norm_" + name] = value
    meta["cursor"] += 1
    for key in meta["counts"]:
        meta["counts"][key] = report[key]
    meta["damping"], meta["last_proposal"] = report["damping"], report["last_proposal"]
    assert session_report(meta, values) == report, "replayed session report differs"
    model = fingerprint(
        meta["model"],
        {
            key: value
            for key, value in values.items()
            if key.startswith(("param_", "norm_"))
        },
    )
    assert model == after["model"], (
        "saved native component does not reproduce parent model"
    )
    return fingerprint(meta, values)


def verify_tape(arrays, details):
    tape = details["linearization_tape"]
    hashes, sizes = (
        arrays["linearization/tape_hashes"],
        arrays["linearization/tape_nbytes"],
    )
    assert len(tape) > 0 and hashes.shape == sizes.shape == (len(tape),)
    assert hashes.dtype == np.dtype("U64") and sizes.dtype == np.int64
    descriptions = []
    for index, item in enumerate(tape):
        assert set(item) == {"shape", "dtype", "sha256", "nbytes", "weak_type"}
        assert type(item["weak_type"]) is bool
        assert isinstance(item["shape"], list) and all(
            type(n) is int and n >= 0 for n in item["shape"]
        )
        dtype = np.dtype(item["dtype"])
        assert dtype.kind in "fibu", "unsupported retained tape leaf dtype"
        assert (
            item["nbytes"]
            == int(np.prod(item["shape"], dtype=np.int64)) * dtype.itemsize
            == sizes[index]
        )
        assert item["sha256"] == hashes[index] and len(item["sha256"]) == 64
        int(item["sha256"], 16)
        descriptions.append(
            dict(shape=item["shape"], dtype=str(dtype), weak_type=item["weak_type"])
        )
    return descriptions


def array_signature(values):
    return [
        dict(shape=list(value.shape), dtype=str(value.dtype), weak_type=False)
        for value in values
    ]


def expected_signature(scope, prepared, outputs, details, tape):
    params = [value for _, value in sorted(subset(prepared, "params/").items())]
    norms = [value for _, value in sorted(subset(prepared, "norms/").items())]
    raw_params = [value for _, value in sorted(subset(prepared, "raw_params/").items())]
    raw_norms = [value for _, value in sorted(subset(prepared, "raw_norms/").items())]
    data = [prepared["data/" + key] for key in FIELDS]
    static = (
        details["static"]
        if scope not in {"cached_jvp", "cached_vjp", "cached_curvature"}
        else {}
    )
    keywords = [np.asarray(static[key]) for key in sorted(static)]
    if scope == "conditioning":
        values = raw_params + raw_norms + data + [prepared["weights"]] + keywords
    elif scope == "prior":
        values = (
            params + norms + data + [prepared["scale"], prepared["weights"]] + keywords
        )
    elif scope in {"residual", "linearization"}:
        values = params + norms + data + [prepared["scale"]] + keywords
    elif scope in {"setup", "solve", "trust", "proposal", "instrumented"}:
        values = (
            params
            + norms
            + data
            + [
                prepared[key]
                for key in ("scale", "weights", "damping", "conditioning_finite")
            ]
            + keywords
        )
    elif scope == "cached_jvp":
        values = [prepared["p"]]
    elif scope == "cached_vjp":
        values = [prepared["p"], prepared["c"]]
    else:
        values = [prepared["p"]] + [
            outputs["setup/" + key] for key in ("weight", "irls", "preconditioner")
        ]
    leaves = array_signature(values)
    if scope in {"setup", "solve", "trust", "proposal", "instrumented"}:
        leaves[-4]["weak_type"] = (
            True  # Production damping is a weak Python-float array.
        )
    if scope.startswith("cached_"):
        leaves = tape + leaves
    return dict(scope=scope, leaves=leaves, static=static)


def verify_signatures(prepared, outputs, details, seen):
    assert set(details["signatures"]) == set(SCOPES)
    tape = verify_tape(outputs, details)
    for scope in SCOPES:
        description = expected_signature(scope, prepared, outputs, details, tape)
        signature = details["signatures"][scope]
        assert signature["description"] == description, (
            scope,
            "dynamic input signature differs",
        )
        digest = hashlib.sha256(
            json.dumps(description, sort_keys=True).encode()
        ).hexdigest()
        assert signature["sha256"] == digest
        assert type(signature["first_seen"]) is bool and signature["first_seen"] == (
            digest not in seen
        )
        seen.add(digest)


def verify_output_hashes(outputs, details):
    assert set(details["output_hashes"]) == set(SCOPES)
    used = set()
    for scope in SCOPES:
        name = "native" if scope == "proposal" else scope
        if name in outputs:
            values = {"": outputs[name]}
            used.add(name)
        else:
            values = subset(outputs, name + "/")
            assert values, scope
            used.update(name + "/" + key for key in values)
        identity = fingerprint({}, values)
        assert details["output_hashes"][scope] == [identity] * 25, (
            scope,
            "repeated output identity differs",
        )
    assert used == set(outputs), "unexpected unqualified output arrays"


def verify_qualification(details, errors, outputs):
    expected = {
        "native": dict(
            arrays=len(subset(outputs, "native/")),
            maximum_scaled_error=errors["instrumented_native"],
        ),
        **{
            stage: dict(arrays=len(STAGES[stage]), maximum_scaled_error=errors[stage])
            for stage in STAGES
        },
        "linearization": dict(
            arrays=1, maximum_scaled_error=errors["linearization_raw"]
        ),
        "adjoint": dict(arrays=1, maximum_scaled_error=errors["adjoint"]),
        "curvature": dict(arrays=1, maximum_scaled_error=errors["curvature"]),
        "prior": dict(arrays=1, maximum_scaled_error=errors["prior"]),
        "residual": dict(arrays=1, maximum_scaled_error=errors["residual"]),
        "direction": dict(arrays=1, maximum_scaled_error=errors["initial_direction"]),
        "cotangent": dict(arrays=1, maximum_scaled_error=errors["cached_c"]),
    }
    assert details["qualification"] == expected, "saved numerical qualification differs"


def recompute_summary(points, protocol):
    assert len(points) == 35 and all(point["status"] == "complete" for point in points)
    groups = {}
    for family in ("quad", "fixedwing"):
        for kind in ("initial", "final", "capture"):
            chosen = [
                point
                for point in points
                if point["family"] == family and point["kind"] == kind
            ]
            if not chosen:
                continue
            scopes = {}
            for scope in chosen[0]["statistics"]:
                medians = np.array(
                    [point["statistics"][scope]["median_s"] for point in chosen]
                )
                scopes[scope] = dict(
                    points=len(chosen),
                    median_of_point_medians_s=float(np.median(medians)),
                    min_point_median_s=float(np.min(medians)),
                    max_point_median_s=float(np.max(medians)),
                )
            groups[family + "/" + kind] = scopes
    ratios = {}
    for point in points:
        stat = point["statistics"]
        pairs = [
            ("solve/setup", "solve", "setup"),
            ("trust/solve", "trust", "solve"),
            ("instrumented/native", "instrumented", "proposal"),
        ]
        if point["kind"] != "final":
            pairs.append(("native/public", "proposal", "public_combined"))
        ratios[point["id"]] = {
            label: stat[top]["median_s"] / stat[bottom]["median_s"]
            for label, top, bottom in pairs
        }
    return dict(
        format="glassbox-online-cost-profile-v1",
        protocol_id=protocol["id"],
        status="complete",
        attempted_points=35,
        completed_points=35,
        point_statuses={point["id"]: "complete" for point in points},
        native_observe_calls=725,
        initialization_calls=0,
        component_dispatches=10500,
        groups=groups,
        diagnostic_ratios=ratios,
        timing_claim="Repeated fixed inputs; component scopes are nonadditive and this is not real-time trajectory qualification.",
    )


def audit(root, authority, *, parent=None, source_root=None):
    root = Path(root)
    source_root = Path(source_root) if source_root is not None else None
    payloads = authenticate(root, authority)
    protocol = read(root / "protocol.json")
    assert protocol["id"] == "online-cost-profile-v1"
    assert protocol["measurement"]["scopes"] == list(SCOPES)
    assert [
        protocol["measurement"][key]
        for key in ("qualification_calls", "warmup_repetitions", "timed_repetitions")
    ] == [1, 3, 21]
    assert {
        key: protocol["qualification"]["component_numerics"][key]
        for key in ("rtol", "atol")
    } == dict(rtol=RTOL, atol=ATOL)
    parent_info = read(root / "parent.json")
    assert parent_info["authority"] == protocol["parent"]["manifest_sha256"]
    parent = Path(parent) if parent is not None else Path(parent_info["path"])
    parent_payloads = authenticate(parent, parent_info["authority"])
    assert parent_payloads == 400
    assert parent_info["manifest_files"] == read(parent / "manifest.json")["files"]
    assert read(parent / "protocol.json")["id"] == protocol["parent"]["protocol_id"]
    parent_binding = read(parent / "binding.json")
    assert (
        parent_binding["source"]["commit"]
        == protocol["parent"]["scientific_source_commit"]
    )
    bound, preserved = read(root / "binding.json"), read(root / "preservation.json")
    assert bound["protocol_sha256"] == sha(root / "protocol.json")
    assert bound["runtime"] == protocol["measurement"]["runtime"]
    package = protocol["parent"]["package_files"]
    assert len(package) == 15
    if source_root is not None:
        assert {
            str(path.relative_to(source_root))
            for path in (source_root / "src/glassbox").rglob("*.py")
        } == set(package)
    assert bound["source_hashes_before"] == preserved["source_hashes_after"] == package
    for name, wanted in package.items():
        assert (
            parent_binding["source"]["files"][name]
            == wanted
            == bound["source"]["files"][name]
        )
        if source_root is not None:
            assert sha(source_root / name) == wanted
    for name, wanted in bound["source"]["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        if source_root is not None:
            assert sha(source_root / name) == wanted, (
                "bound scientific source differs: " + name
            )
    assert (
        preserved["parent_authority_after"] == parent_info["authority"]
        and preserved["parent_payloads_after"] == parent_payloads
    )
    assert sha(root / "source/online.py") == package["src/glassbox/online.py"]
    assert (
        sha(root / "source/_online_cost_kernels.py")
        == bound["source"]["files"]["scripts/_online_cost_kernels.py"]
    )
    generated = verify_generated(
        root,
        (root / "source/online.py").read_text(),
        (root / "source/_online_cost_kernels.py").read_text(),
    )
    roster = expected_roster(protocol, parent)
    assert read(root / "roster.json") == roster
    assert {
        str(path.parent.relative_to(root / "points"))
        for path in (root / "points").rglob("point.json")
    } == {point["id"] for point in roster}
    summaries, seen, replay_count, timing_count = [], set(), 0, 0
    cache = {}
    for point in roster:
        location = root / "points" / point["id"]
        details = read(location / "point.json")
        assert all(details[key] == value for key, value in point.items()), point["id"]
        assert details["status"] == "complete"
        case = point["case"]
        if case not in cache:
            source = load(parent / "inputs" / (case + ".npz"))
            initial_meta, initial_values, _, _ = session(
                parent / case / "initial-online.npz"
            )
            verify_bootstrap(initial_values, initial_meta, source)
            cache[case] = source, initial_values, events_by_origin(parent / case)
        source, initial, events = cache[case]
        meta, values, identity, model = session(location / "session.npz")
        assert (
            sha(location / "session.npz")
            == sha(parent / point["session"])
            == details["source_session_sha256"]
        )
        assert (
            details["session_fingerprint_before"]
            == details["session_fingerprint_after"]
            == identity
        )
        assert meta["cursor"] == point["origin"]
        assert details["static"] == dict(
            delay=meta["model"]["delay_steps"], dt_s=meta["model"]["dt_s"]
        )
        check_cache(values, meta, source, initial)
        genuine = point["kind"] != "final"
        if genuine:
            context = load(location / "context.npz")
            assert set(context) == {"origin", "command", "truth", "native_truth"}
            same(context["origin"], np.asarray(point["origin"], dtype=np.int64))
            same(context["command"], source["commands"][point["origin"]])
            same(context["native_truth"], source["states"][point["origin"] + 1])
            np.testing.assert_allclose(
                context["truth"],
                canonical(context["native_truth"][None])[0],
                rtol=3e-10,
                atol=3e-12,
            )
            before, after = (
                events[(point["origin"], phase)]
                for phase in ("predicted", "assimilated")
            )
            assert (
                session_report(meta, values) == before["report"]
                and model == before["model"]
            )
            assert details["parent_expected"] == dict(
                model=after["model"], report=after["report"]
            )
            with (parent / case / "arrays.bin").open("rb") as stream:
                for event, key in (
                    (before, "command"),
                    (events[(point["origin"], "revealed")], "truth"),
                ):
                    stream.seek(event["arrays"][key])
                    same(context[key], np.load(stream, allow_pickle=False))
            if point["kind"] == "capture":
                capture = read(parent / case / str(point["origin"]) / "capture.json")
                assert (
                    capture["session_fingerprint"] == identity
                    and capture["model_fingerprint"] == model
                )
                assert capture["report"] == before["report"]
                captured = load(parent / case / str(point["origin"]) / "context.npz")
                for key in ("origin", "command", "truth"):
                    same(context[key], captured[key])
        else:
            assert (
                not (location / "context.npz").exists()
                and "parent_expected" not in details
            )
        prepared, outputs = (
            load(location / "prepared.npz"),
            load(location / "outputs.npz"),
        )
        verify_prepared(prepared, meta, values, source, genuine)
        errors, proposal_report = verify_components(outputs, prepared)
        verify_qualification(details, errors, outputs)
        verify_signatures(prepared, outputs, details, seen)
        verify_output_hashes(outputs, details)
        timings = load(location / "timings.npz")
        timing_count += verify_timings(timings, details["statistics"], genuine)
        assert details["first_call_s"] == {
            name: float(times[0]) for name, times in timings.items()
        }
        assert details["component_dispatches"] == 300
        records = read(location / "replays.json")
        if genuine:
            assert proposal_report == after["report"]["last_proposal"], (
                "native saved scalar proposal differs from replay"
            )
            after_identity = replay_session(
                meta, values, context, after, outputs, prepared
            )
            replay_count += verify_replays(records, before, after)
            assert details["native_observe_calls"] == 25
            for record in records:
                assert record["before_session_fingerprint"] == identity
                assert record["after_session_fingerprint"] == after_identity
                assert record["matches_parent"] is True
        else:
            assert records == [] and details["native_observe_calls"] == 0
        summaries.append(details)
    assert replay_count == 725 and timing_count == 12675
    assert read(root / "summary.json") == recompute_summary(summaries, protocol), (
        "summary differs from raw samples"
    )
    return dict(
        status="complete",
        payloads=payloads,
        parent_payloads=parent_payloads,
        points=35,
        genuine_points=29,
        retained_cache_probes=6,
        exact_replays=replay_count,
        component_dispatches=10500,
        raw_timing_samples=timing_count,
        generated_sources=generated,
        unchanged_package_files=15,
        learner_calls=0,
        limitation="Audits saved outputs and source; does not independently execute dynamics or derivative kernels.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.root,
                args.manifest_sha256,
                parent=args.parent,
                source_root=args.source_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
