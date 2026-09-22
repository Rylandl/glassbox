"""Frozen saved-session cost profile; never change the adopted online learner.

Component prefixes are diagnostic compiler scopes, not additive production costs.
Only disposable, authentic recorded transitions call the public observer.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
from collect_throw import authenticate, binding, check_source, seal
from run_dart import ROOT, write
from verify_baseline import arrays, digest, observed, read, require
from verify_online_v8 import fingerprint

PROTOCOL = ROOT / "docs/harness/online-cost-profile-v1.json"
FIELDS = ("past_states", "past_inputs", "future_inputs", "future_states")
NATIVE_FIELDS = (
    "params",
    "current_loss",
    "trial_loss",
    "predicted_reduction",
    "finite",
    "evidence",
)


def flatten(value, prefix=""):
    """Readable, deterministic array paths for plain dict/tuple outputs."""
    if isinstance(value, dict):
        result = {}
        for key, child in sorted(value.items()):
            result.update(flatten(child, prefix + str(key) + "/"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(flatten(child, prefix + str(index) + "/"))
        return result
    return {prefix.rstrip("/"): np.asarray(value)}


def compare(actual, expected, *, rtol, atol):
    """Qualify saved arrays, enforcing discrete values and every nonfinite mask."""
    require(set(actual) == set(expected), "component array roster differs")
    maximum = 0.0
    for key in actual:
        value, reference = np.asarray(actual[key]), np.asarray(expected[key])
        require(
            value.shape == reference.shape and value.dtype == reference.dtype,
            "component shape/dtype differs: " + key,
        )
        if value.dtype.kind not in "fc" or key.endswith("selected_alpha"):
            require(
                np.array_equal(value, reference), "discrete component differs: " + key
            )
            continue
        require(
            np.array_equal(np.isnan(value), np.isnan(reference))
            and np.array_equal(np.isposinf(value), np.isposinf(reference))
            and np.array_equal(np.isneginf(value), np.isneginf(reference)),
            "component finite masks differ: " + key,
        )
        finite = np.isfinite(reference)
        if np.any(finite):
            scaled = np.abs(value[finite] - reference[finite]) / (
                atol + rtol * np.abs(reference[finite])
            )
            maximum = max(maximum, float(np.max(scaled)))
            require(np.all(scaled <= 1), "component tolerance exceeded: " + key)
    return dict(arrays=len(actual), maximum_scaled_error=maximum)


def statistics(values):
    values = np.asarray(values, dtype=float)
    require(
        values.ndim == 1
        and len(values) > 0
        and np.isfinite(values).all()
        and np.all(values >= 0),
        "invalid timing samples",
    )
    return dict(
        count=len(values),
        median_s=float(np.median(values)),
        p95_s=float(np.quantile(values, 0.95)),
        max_s=float(np.max(values)),
        min_s=float(np.min(values)),
    )


def roster(protocol, parent):
    points = []
    for case in protocol["points"]["case_order"]:
        info = read(parent / case / "case.json")
        for kind, filename in (
            ("initial", "initial-online.npz"),
            ("final", "final-online.npz"),
        ):
            meta = json.loads(str(arrays(parent / case / filename)["metadata"]))
            points.append(
                dict(
                    id=f"{case}/{kind}",
                    case=case,
                    family=info["family"],
                    kind=kind,
                    origin=meta["cursor"],
                    session=f"{case}/{filename}",
                )
            )
        for origin in protocol["points"]["capture_origins"].get(case, []):
            points.append(
                dict(
                    id=f"{case}/capture-{origin}",
                    case=case,
                    family=info["family"],
                    kind="capture",
                    origin=origin,
                    session=f"{case}/{origin}/session.npz",
                )
            )
    return points


def prepare_cache(meta, values, context=None):
    """Pure bounded cache assembly, independently auditable from saved arrays."""
    roles = {
        role: {name: values[role + "_" + name].copy() for name in FIELDS}
        for role in ("bootstrap", "recent")
    }
    if context is not None:
        require(int(context["origin"]) == meta["cursor"], "context cursor differs")
        states = np.concatenate((values["tail_states"][1:], context["truth"][None]))
        inputs = np.concatenate((values["tail_inputs"][1:], context["command"][None]))
        history, horizon = meta["model"]["history_steps"], meta["horizon"]
        new = dict(
            zip(
                FIELDS,
                (
                    states[: history + 1][None],
                    inputs[:history][None],
                    inputs[history : history + horizon][None],
                    states[history + 1 : history + horizon + 1][None],
                ),
            )
        )
        roles["recent"] = {
            name: np.concatenate((roles["recent"][name], new[name]))[-32:]
            for name in FIELDS
        }
    blocks, weights = [], []
    for role in ("bootstrap", "recent"):
        data = roles[role]
        count = len(data["past_states"])
        require(0 < count <= 32, "invalid real cache count")
        indices = np.arange(32) % count
        blocks.append({name: value[indices] for name, value in data.items()})
        weights.append(np.where(np.arange(32) < count, 0.5 / count, 0.0))
    return tuple(
        np.concatenate((blocks[0][name], blocks[1][name])) for name in FIELDS
    ), np.concatenate(weights)


def _synchronize(value):
    import jax

    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _timed(function, args=(), keywords=None):
    keywords = {} if keywords is None else keywords
    _synchronize((args, keywords))
    start = time.perf_counter()
    output = function(*args, **keywords)
    _synchronize(output)
    return output, time.perf_counter() - start


def _plain(scope, output):
    if scope == "conditioning":
        return dict(zip(("params", "norms", "finite"), output))
    if scope == "proposal":
        return dict(zip(NATIVE_FIELDS, output))
    if scope == "instrumented":
        native, stages = output
        return dict(native=dict(zip(NATIVE_FIELDS, native)), **stages)
    if scope == "linearization":
        import jax

        raw, push = output
        leaves = [np.asarray(value) for value in jax.tree.leaves(push)]
        return dict(
            raw=raw,
            tape_hashes=np.asarray(
                [fingerprint({}, {"value": leaf}) for leaf in leaves], dtype="U64"
            ),
            tape_nbytes=np.asarray([leaf.nbytes for leaf in leaves], dtype=np.int64),
        )
    return output


def _signature(scope, args, keywords):
    import jax

    leaves, structure = jax.tree.flatten((args, keywords))
    description = dict(
        scope=scope,
        leaves=[
            dict(
                shape=list(np.shape(value)),
                dtype=str(np.asarray(value).dtype),
                weak_type=bool(getattr(value, "weak_type", False)),
            )
            for value in leaves
        ],
        # Static values participate in jit specialization.
        static={key: keywords[key] for key in ("delay", "dt_s") if key in keywords},
    )
    # Function identities in Partial treedefs are process-specific; array signatures
    # and scope/static identity establish the honest first-scope/signature label.
    del structure
    code = json.dumps(description, sort_keys=True).encode()
    return dict(sha256=hashlib.sha256(code).hexdigest(), description=description)


def _save_npz(path, values):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def _public_replay(session_path, context, expected, index, phase):
    import jax

    from glassbox.online import OnlineFit

    clone = OnlineFit.load(session_path)
    before = clone.fingerprint()
    start = time.perf_counter()
    clone.observe(int(context["origin"]), context["command"], context["truth"])
    observed_at = time.perf_counter()
    model = clone.model
    jax.block_until_ready(model.params)
    finished = time.perf_counter()
    identity = model.fingerprint
    identity = identity() if callable(identity) else identity
    record = dict(
        index=index,
        phase=phase,
        before_session_fingerprint=before,
        after_session_fingerprint=clone.fingerprint(),
        model=identity,
        report=clone.report,
    )
    record["matches_parent"] = (
        identity == expected["model"] and clone.report == expected["report"]
    )
    return record, (observed_at - start, finished - observed_at, finished - start)


def _qualify(outputs, prepared, protocol):
    tolerance = protocol["qualification"]["component_numerics"]
    settings = {key: tolerance[key] for key in ("rtol", "atol")}
    result = {
        "native": compare(
            flatten(outputs["instrumented"]["native"]),
            flatten(outputs["proposal"]),
            **settings,
        )
    }
    for stage in ("setup", "solve", "trust"):
        result[stage] = compare(
            flatten(outputs[stage]), flatten(outputs["instrumented"][stage]), **settings
        )
    setup = outputs["setup"]
    result["linearization"] = compare(
        {"raw": np.asarray(outputs["linearization"]["raw"])},
        {"raw": np.asarray(outputs["instrumented"]["setup"]["raw"])},
        **settings,
    )
    for name in ("prior", "residual"):
        field = "raw" if name == "residual" else name
        result[name] = compare(
            {field: np.asarray(outputs[name])},
            {field: np.asarray(outputs["instrumented"]["setup"][field])},
            **settings,
        )
    p, c = prepared["p"], prepared["c"]
    result["direction"] = compare(
        {"p": p},
        {"p": -np.asarray(setup["gradient"]) / np.asarray(setup["preconditioner"])},
        **settings,
    )
    jp, vjp = np.asarray(outputs["cached_jvp"]), np.asarray(outputs["cached_vjp"])
    result["cotangent"] = compare(
        {"c": c},
        {"c": np.asarray(setup["weight"]) * np.asarray(setup["irls"]) * jp},
        **settings,
    )
    result["adjoint"] = compare(
        {"inner_product": np.asarray(np.vdot(jp, c))},
        {"inner_product": np.asarray(np.vdot(p, vjp))},
        **settings,
    )
    result["curvature"] = compare(
        {"value": np.asarray(outputs["cached_curvature"])},
        {"value": vjp + np.asarray(setup["preconditioner"]) * p},
        **settings,
    )
    return result


def profile_point(point, parent, output, protocol, kernels, seen):
    import jax
    import jax.numpy as jnp

    import glassbox.online as online

    output.mkdir(parents=True)
    source = parent / point["session"]
    session, before = None, None
    details = dict(
        point,
        status="running",
        session_fingerprint_before=before,
        session_fingerprint_after=None,
        source_session_sha256=None,
        signatures={},
        output_hashes={},
        first_call_semantics="first scope/signature dispatch; not guaranteed cold compilation",
        linearization_tape=[],
        tape_bytes_semantics="Sum of logical leaf payload bytes; not peak or unique memory.",
        qualification={},
        statistics={},
    )
    timings, replay_records, recorded, prepared, signatures = (
        {},
        [],
        {},
        {},
        details["signatures"],
    )
    context, expected = None, None
    try:
        shutil.copyfile(source, output / "session.npz")
        saved = arrays(source)
        meta = json.loads(str(saved.pop("metadata")))
        session = online.OnlineFit.load(source)
        before = session.fingerprint()
        details["session_fingerprint_before"] = before
        details["source_session_sha256"] = digest(source)
        require(before == meta["fingerprint"], "session fingerprint differs")
        if point["kind"] != "final":
            tape = arrays(parent / "inputs" / (point["case"] + ".npz"))
            origin = point["origin"]
            context = dict(
                origin=np.asarray(origin, dtype=np.int64),
                command=tape["commands"][origin].copy(),
                truth=observed(tape["states"][origin + 1 : origin + 2])[0],
                native_truth=tape["states"][origin + 1].copy(),
            )
            if point["kind"] == "capture":
                captured = arrays(parent / point["case"] / str(origin) / "context.npz")
                for key in ("origin", "command", "truth"):
                    require(
                        np.array_equal(context[key], captured[key]),
                        "capture context differs",
                    )
                capture = read(parent / point["case"] / str(origin) / "capture.json")
                require(
                    capture["session_fingerprint"] == before, "capture session differs"
                )
            events = [
                json.loads(line)
                for line in (parent / point["case"] / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            predecessor = next(
                event
                for event in events
                if event["phase"] == "predicted" and event["index"] == origin
            )
            expected = next(
                event
                for event in events
                if event["phase"] == "assimilated" and event["index"] == origin
            )
            require(session.report == predecessor["report"], "before report differs")
            identity = session.model.fingerprint
            identity = identity() if callable(identity) else identity
            require(identity == predecessor["model"], "before core-model differs")
            details["parent_expected"] = dict(
                model=expected["model"], report=expected["report"]
            )
            _save_npz(output / "context.npz", context)
        data, weights = prepare_cache(meta, saved, context)
        prepared = dict(
            raw_params={
                key[6:]: value
                for key, value in saved.items()
                if key.startswith("param_")
            },
            raw_norms={
                key[5:]: value
                for key, value in saved.items()
                if key.startswith("norm_")
            },
            data=dict(zip(FIELDS, data)),
            weights=weights,
            scale=saved["scale"],
            damping=np.asarray(meta["damping"]),
        )
        delay, dt_s = meta["model"]["delay_steps"], meta["model"]["dt_s"]
        static = dict(delay=delay, dt_s=dt_s)
        details["static"] = static
        calls = {}

        def public(index, phase):
            if context is None:
                return
            with jax.enable_x64(False):
                record, times = _public_replay(source, context, expected, index, phase)
            replay_records.append(record)
            for name, elapsed in zip(
                ("public_observe", "public_snapshot", "public_combined"), times
            ):
                timings.setdefault(name, []).append(elapsed)
            require(
                record["before_session_fingerprint"] == before,
                "replay clone differs from saved session",
            )
            require(
                record["matches_parent"],
                "native replay differs from original assimilation",
            )

        def dispatch(name, qualification=False):
            function, args, keywords = calls[name]
            if qualification:
                signature = _signature(name, args, keywords)
                signature["first_seen"] = signature["sha256"] not in seen
                seen.add(signature["sha256"])
                signatures[name] = signature
            result, elapsed = _timed(function, args, keywords)
            timings.setdefault(name, []).append(elapsed)
            plain = _plain(name, result)
            value_hash = fingerprint({}, flatten(plain))
            details["output_hashes"].setdefault(name, []).append(value_hash)
            if qualification:
                recorded[name] = plain
            else:
                require(
                    value_hash == details["output_hashes"][name][0],
                    "repeated component output differs: " + name,
                )
            return result

        public(0, "qualification")
        with jax.enable_x64(True):
            raw_params, raw_norms = jax.tree.map(
                jnp.asarray, (prepared["raw_params"], prepared["raw_norms"])
            )
            device_data = tuple(jnp.asarray(value) for value in data)
            device_weights, scale = map(jnp.asarray, (weights, prepared["scale"]))
            damping = jnp.asarray(float(meta["damping"]))
            calls["conditioning"] = (
                online._recondition,
                (raw_params, raw_norms, device_data, device_weights),
                static,
            )
            params, norms, conditioning_finite = dispatch("conditioning", True)
            prepared.update(
                params=jax.tree.map(np.asarray, params),
                norms=jax.tree.map(np.asarray, norms),
                conditioning_finite=np.asarray(conditioning_finite),
            )
            proposal_args = (params, norms, device_data, scale, device_weights, damping)
            proposal_keywords = dict(static, conditioning_finite=conditioning_finite)
            calls["prior"] = (
                partial(jax.jit, static_argnames=("delay", "dt_s"))(
                    online._curvature_diagonal
                ),
                (params, norms, device_data, scale, device_weights),
                static,
            )
            calls["residual"] = (
                partial(jax.jit, static_argnames=("delay", "dt_s"))(online._residual),
                (params, norms, device_data, scale),
                static,
            )
            calls["linearization"] = (
                kernels["linearization"],
                (params, norms, device_data, scale),
                static,
            )
            dispatch("prior", True)
            dispatch("residual", True)
            _, push = dispatch("linearization", True)
            for leaf in jax.tree.leaves(push):
                weak_type = bool(getattr(leaf, "weak_type", False))
                leaf = np.asarray(leaf)
                details["linearization_tape"].append(
                    dict(
                        shape=list(leaf.shape),
                        dtype=leaf.dtype.str,
                        weak_type=weak_type,
                        nbytes=leaf.nbytes,
                        sha256=fingerprint({}, {"value": leaf}),
                    )
                )
            for name in ("setup", "solve", "trust", "proposal", "instrumented"):
                calls[name] = (
                    (online._proposal if name == "proposal" else kernels[name]),
                    proposal_args,
                    proposal_keywords,
                )
                dispatch(name, True)
            setup = recorded["setup"]
            p = -jnp.asarray(setup["gradient"]) / jnp.asarray(setup["preconditioner"])
            calls["cached_jvp"] = (kernels["cached_jvp"], (push, p), {})
            jp = dispatch("cached_jvp", True)
            weight, irls, preconditioner = (
                jnp.asarray(setup[key]) for key in ("weight", "irls", "preconditioner")
            )
            c = weight * irls * jp
            template = jnp.zeros_like(p)
            calls["cached_vjp"] = (kernels["cached_vjp"], (push, template, c), {})
            calls["cached_curvature"] = (
                kernels["cached_curvature"],
                (push, p, weight, irls, preconditioner),
                {},
            )
            dispatch("cached_vjp", True)
            dispatch("cached_curvature", True)
            prepared.update(p=np.asarray(p), c=np.asarray(c))
            details["qualification"] = _qualify(recorded, prepared, protocol)
            scopes = protocol["measurement"]["scopes"]
            warm = protocol["measurement"]["warmup_repetitions"]
            repetitions = warm + protocol["measurement"]["timed_repetitions"]
            for repetition in range(repetitions):
                public(repetition + 1, "warmup" if repetition < warm else "timed")
                offset = repetition % len(scopes)
                for name in scopes[offset:] + scopes[:offset]:
                    dispatch(name)
        require(session.fingerprint() == before, "component probes mutated session")
        details["status"] = "complete"
    except Exception as error:
        details["status"] = "failed"
        details["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        details["session_fingerprint_after"] = (
            None if session is None else session.fingerprint()
        )
        details["native_observe_calls"] = len(replay_records)
        details["component_dispatches"] = sum(
            len(timings.get(name, [])) for name in protocol["measurement"]["scopes"]
        )
        first_timed = 1 + protocol["measurement"]["warmup_repetitions"]
        details["first_call_s"] = {
            key: value[0] for key, value in timings.items() if value
        }
        details["statistics"] = {
            key: statistics(value[first_timed:])
            for key, value in timings.items()
            if len(value) > first_timed
        }
        if recorded:
            renamed = {
                ("native" if name == "proposal" else name): value
                for name, value in recorded.items()
            }
            _save_npz(output / "outputs.npz", flatten(renamed))
        if prepared:
            _save_npz(output / "prepared.npz", flatten(prepared))
        _save_npz(
            output / "timings.npz",
            {key: np.asarray(value) for key, value in timings.items()},
        )
        write(output / "replays.json", replay_records)
        write(output / "point.json", details)
    require(
        before == details["session_fingerprint_after"],
        "component probes mutated session",
    )
    return details


def summary(points, protocol):
    completed = [point for point in points if point["status"] == "complete"]
    groups = {}
    for family in ("quad", "fixedwing"):
        for kind in ("initial", "final", "capture"):
            selected = [
                point
                for point in completed
                if point["family"] == family and point["kind"] == kind
            ]
            if not selected:
                continue
            scopes = {}
            for name in selected[0]["statistics"]:
                medians = [point["statistics"][name]["median_s"] for point in selected]
                scopes[name] = dict(
                    points=len(medians),
                    median_of_point_medians_s=float(np.median(medians)),
                    min_point_median_s=float(min(medians)),
                    max_point_median_s=float(max(medians)),
                )
            groups[family + "/" + kind] = scopes
    ratios = {}
    for point in completed:
        stat = point["statistics"]
        ratios[point["id"]] = {
            label: stat[top]["median_s"] / stat[bottom]["median_s"]
            for label, top, bottom in (
                ("solve/setup", "solve", "setup"),
                ("trust/solve", "trust", "solve"),
                ("instrumented/native", "instrumented", "proposal"),
            )
        }
        if "public_combined" in stat:
            ratios[point["id"]]["native/public"] = (
                stat["proposal"]["median_s"] / stat["public_combined"]["median_s"]
            )
    return dict(
        format="glassbox-online-cost-profile-v1",
        protocol_id=protocol["id"],
        status="complete" if len(completed) == 35 else "failed",
        attempted_points=len(points),
        completed_points=len(completed),
        point_statuses={point["id"]: point["status"] for point in points},
        native_observe_calls=sum(point["native_observe_calls"] for point in points),
        initialization_calls=0,
        component_dispatches=sum(point["component_dispatches"] for point in points),
        groups=groups,
        diagnostic_ratios=ratios,
        timing_claim="Repeated fixed inputs; component scopes are nonadditive and this is not real-time trajectory qualification.",
    )


def run(parent, output, protocol_path=PROTOCOL):
    import _online_cost_kernels as diagnostic
    import jax

    protocol = read(protocol_path)
    require(protocol["id"] == "online-cost-profile-v1", "profile protocol differs")
    output.mkdir(parents=True, exist_ok=False)
    points, result = [], None

    def source_hashes():
        return {
            name: digest(ROOT / name) for name in protocol["parent"]["package_files"]
        }

    try:
        shutil.copyfile(protocol_path, output / "protocol.json")
        write(
            output / "attempt.json",
            dict(parent=str(parent.resolve()), protocol=str(protocol_path.resolve())),
        )
        authority = protocol["parent"]["manifest_sha256"]
        inventory = authenticate(parent, authority)
        require(len(inventory["files"]) == 400, "parent payload count differs")
        require(
            read(parent / "protocol.json")["id"] == "online-fit-v8",
            "parent protocol differs",
        )
        require(
            read(parent / "binding.json")["source"]["commit"]
            == protocol["parent"]["scientific_source_commit"],
            "parent scientific source differs",
        )
        before = source_hashes()
        require(
            before == protocol["parent"]["package_files"],
            "adopted package source differs",
        )
        bound = binding(protocol_path)
        require(
            bound["runtime"] == protocol["measurement"]["runtime"],
            "profile runtime differs",
        )
        bound["source_hashes_before"] = before
        bound["execution_settings"] = dict(
            environment={
                key: os.environ[key]
                for key in (
                    "JAX_ENABLE_X64",
                    "JAX_PLATFORMS",
                    "JAX_PLATFORM_NAME",
                    "JAX_COMPILATION_CACHE_DIR",
                    "JAX_ENABLE_COMPILATION_CACHE",
                    "JAX_DEFAULT_MATMUL_PRECISION",
                    "XLA_FLAGS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
                if key in os.environ
            },
            compilation_cache_dir=jax.config.jax_compilation_cache_dir,
            enable_compilation_cache=jax.config.jax_enable_compilation_cache,
            default_matmul_precision=jax.config.jax_default_matmul_precision,
        )
        write(output / "binding.json", bound)
        (output / "source").mkdir()
        shutil.copyfile(ROOT / "src/glassbox/online.py", output / "source/online.py")
        shutil.copyfile(
            ROOT / "scripts/_online_cost_kernels.py",
            output / "source/_online_cost_kernels.py",
        )
        write(
            output / "parent.json",
            dict(
                path=str(parent.resolve()),
                authority=authority,
                manifest_files=inventory["files"],
            ),
        )
        planned = roster(protocol, parent)
        require(
            len(planned) == 35 and sum(p["kind"] != "final" for p in planned) == 29,
            "point roster differs",
        )
        write(output / "roster.json", planned)
        (output / "generated").mkdir()
        for name, source in diagnostic.generated_sources().items():
            (output / "generated" / (name + ".py")).write_text(source)
        write(
            output / "generated.json",
            {p.name: digest(p) for p in sorted((output / "generated").iterdir())},
        )
        kernels, seen = diagnostic.make_kernels(), set()
        for point in planned:
            print(json.dumps(dict(point=point["id"], status="profiling")), flush=True)
            location = output / "points" / point["id"]
            try:
                value = profile_point(point, parent, location, protocol, kernels, seen)
            except Exception:
                if (location / "point.json").exists():
                    points.append(read(location / "point.json"))
                raise
            points.append(value)
            print(json.dumps(dict(point=point["id"], status="complete")), flush=True)
        require(source_hashes() == before, "package source changed during profile")
        check_source(bound)
        authenticate(parent, authority)
        write(
            output / "preservation.json",
            dict(
                source_hashes_after=source_hashes(),
                parent_authority_after=digest(parent / "manifest.json"),
                parent_payloads_after=len(inventory["files"]),
            ),
        )
        result = summary(points, protocol)
        require(
            result["native_observe_calls"] == 725
            and result["component_dispatches"] == 10500,
            "profile work count differs",
        )
    except Exception as error:
        result = summary(points, protocol)
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        write(output / "summary.json", result)
        authority = seal(output, "glassbox-online-cost-profile-v1")
        print(json.dumps(dict(manifest_sha256=authority, summary=result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    result = run(args.parent, args.output, args.protocol)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
