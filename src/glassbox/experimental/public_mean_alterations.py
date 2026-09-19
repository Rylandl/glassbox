"""Twelve frozen, actual-bundle alterations; no fitting or simulation.

The separately anchored layout supplies locations and declared mirrors, never
validator implementations or acceptance criteria. Each direct validator executes
in a fresh process. Only PM02/05/08/11 perform new model inference; PM09 uses the
frozen paired-bootstrap RNG. The final bundle is copied with copy-on-write,
never hardlinks. Failed copies and every subprocess transcript are retained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

FORMAT = "glassbox-public-mean-alterations-v1"
PROTOCOL_SHA256 = "d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99"
CHECKS = dict(
    zip(
        (f"PM{i:02d}" for i in range(1, 13)),
        (
            "public_archive_integrity",
            "retained_mean_parity",
            "training_cache_reconstruction",
            "initial_role_identity",
            "calibration_reduction",
            "query_reconstruction",
            "truth_roster_and_masks",
            "saved_prediction_replay",
            "decision_reduction",
            "worker_source_identity",
            "consumer_execution_identity",
            "revision_chain_identity",
        ),
        strict=True,
    )
)
RELATIVE = "src/glassbox/experimental/public_mean_alterations.py"
HARD_TIMEOUT_S = 14400


class CheckFailure(ValueError):
    def __init__(self, check_id, detail):
        super().__init__(detail)
        self.check_id = check_id


def require(value, check_id, detail):
    if not value:
        raise CheckFailure(check_id, detail)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def under(root, relative):
    root = Path(root).resolve()
    path = root / relative
    require(
        not Path(relative).is_absolute() and path.resolve().is_relative_to(root),
        "audit_prerequisite",
        "relative bundle path escapes its root",
    )
    require(not path.is_symlink(), "audit_prerequisite", "symlink payload")
    return path


def inventory(root, excluding=()):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), "audit_prerequisite", "symlink in bundle")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in excluding:
                result[name] = digest(path)
    return result


def anchor(root, manifest, expected):
    path = under(root, manifest)
    require(digest(path) == expected, "external_anchor", "root manifest SHA differs")
    value = read(path)
    require(
        value["files"] == inventory(root, (manifest,)),
        "external_anchor",
        "complete externally anchored payload inventory differs",
    )
    return value


def pointer(value, keys):
    for key in keys:
        value = value[key]
    return value


def assign(value, keys, replacement):
    require(bool(keys), "audit_prerequisite", "empty mirror pointer")
    pointer(value, keys[:-1])[keys[-1]] = replacement


def archive(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["metadata"])), {
            key: np.array(z[key], copy=True) for key in z.files if key != "metadata"
        }


def rewrite_archive(path, metadata, values, *, coherent=True):
    import numpy as np

    from glassbox._learner_arrays import array_fingerprint

    metadata = copy.deepcopy(metadata)
    if coherent:
        metadata.pop("fingerprint", None)
        metadata["fingerprint"] = array_fingerprint(metadata, values)
    np.savez_compressed(path, metadata=json.dumps(metadata), **values)
    return metadata


def array_equal(actual, expected, check_id):
    import numpy as np

    require(set(actual) == set(expected), check_id, "array roster")
    for name in actual:
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        require(
            a.dtype == b.dtype
            and a.shape == b.shape
            and a.tobytes(order="C") == b.tobytes(order="C"),
            check_id,
            name,
        )


def npz(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        return {key: np.array(z[key], copy=True) for key in z.files}


def physical(binding):
    # This standalone source is also used inside the historical process. Its
    # lazy Glassbox imports resolve only against that process's pinned checkout.
    path = (
        Path(binding["public_root"])
        / "src/glassbox/experimental/public_mean_physical_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("_alteration_physical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(request):
    from glassbox.experimental.public_mean_saved_port import prepare_reference

    return prepare_reference(
        request["reference_root"],
        "crazyflow",
        expected_bundle_sha256=request["reference_sha256"],
    )


def case_role(case):
    if case in ("PM06", "PM07", "PM10"):
        return "historical"
    return "public32" if case in ("PM08", "PM11") else "public64"


def location(root, layout, name):
    return under(root, layout["paths"][name])


def role_for(evaluator, arm):
    roles = [role for role, arms in evaluator.ROLES.items() if arm in arms]
    require(len(roles) == 1, "audit_prerequisite", "unique arm execution role")
    return roles[0]


def target_query(root, layout, *, complete):
    import numpy as np

    data = location(root, layout, "crazyflow_data")
    for query in sorted(
        read(data / "queries.json"), key=lambda q: (q["parent"], q["id"])
    ):
        if query["kind"] != "factual":
            continue
        values = npz(data / query["path"])
        if complete:
            okay = query["history_eligible"] and all(
                np.isfinite(values[key]).all()
                for key in ("past_states", "past_inputs", "future_inputs")
            )
        else:
            okay = query["scope"] == "primary" and bool(np.all(values["valid"]))
        if okay:
            return query
    raise CheckFailure("audit_prerequisite", "frozen query target does not exist")


def translate(call, *, source_id, target_id):
    """Only translate the helper's declared semantic failure, never prerequisites."""
    try:
        return call()
    except Exception as error:
        if getattr(error, "check_id", None) == source_id:
            raise CheckFailure(target_id, str(error)) from error
        raise


def normalized_forecast_check(actual, expected, norms):
    """The frozen forecast64 tolerance is in authenticated normalized units."""
    import numpy as np

    from glassbox.experimental.public_v4_numerics import TOLERANCES, element_check

    mean = np.asarray(norms["state_mean"], dtype=np.float64)
    scale = np.asarray(norms["state_scale"], dtype=np.float64)
    return element_check(
        (np.asarray(actual, dtype=np.float64) - mean) / scale,
        (np.asarray(expected, dtype=np.float64) - mean) / scale,
        TOLERANCES["forecast64"],
    )


def _parity(root, layout, request):
    import jax
    import numpy as np

    from glassbox import LearnedDynamics
    from glassbox.experimental.public_v4_numerics import INPUTS

    prepared = reference(request)
    model = LearnedDynamics.load(location(root, layout, "port_model"))
    array_equal(model._model.arrays(), prepared.selected_arrays, "retained_mean_parity")
    packet = location(root, layout, "numeric_inputs")
    manifest = read(packet / "manifest.json")
    old = location(root, layout, "numeric_oracle")
    results = read(old / "result.json")
    require(
        results["mode"] == "oracle64" and results["fits"] == 0,
        "audit_prerequisite",
        "historical parity evidence role",
    )
    output_rows = {row["id"]: row for row in results["cases"]}
    selected = layout["parity_case_ids"]
    require(
        len(selected) == 12 and len(set(selected)) == 12,
        "audit_prerequisite",
        "exact twelve Crazyflow actual-input parity cases",
    )
    cases = {case["id"]: case for case in manifest["cases"]}
    expected_cases = [
        case["id"]
        for case in manifest["cases"]
        if case["source"] == "public_archive"
        and case["source_identity"]["query"]["model_id"] == "crazyflow"
    ]
    require(
        selected == expected_cases,
        "audit_prerequisite",
        "all frozen Crazyflow parity inputs in original order",
    )
    binding = request["binding"]
    for item in results["imported_sources"].values():
        relative = item["relative_path"]
        require(
            binding["oracle_source_sha256"].get(relative)
            == item["sha256"]
            == digest(Path(binding["oracle_root"]) / relative),
            "audit_prerequisite",
            "historical numerical oracle source role",
        )
    checks = []
    predict = jax.jit(model.predict)
    for name in selected:
        case = cases[name]
        require(
            case["source"] == "public_archive",
            "audit_prerequisite",
            "actual input case",
        )
        values = npz(packet / case["path"])
        science = {k: v for k, v in values.items() if k.startswith(("param_", "norm_"))}
        array_equal(science, prepared.selected_arrays, "audit_prerequisite")
        row = output_rows[name]
        require(
            digest(old / row["path"]) == row["sha256"]
            and digest(packet / case["path"]) == case["sha256"],
            "audit_prerequisite",
            "parity input/output hashes",
        )
        expected = npz(old / row["path"])["original__eager"]
        actual = np.asarray(predict(*(values[k] for k in INPUTS)))
        result = normalized_forecast_check(actual, expected, prepared.norms)
        require(result["passed"], "retained_mean_parity", name)
        checks.append(dict(id=name, **result))
    return dict(
        scientific_arrays=17, forecasts=checks, failed_fd_gate_not_reinterpreted=True
    )


def _cache(root, layout, request, roles_only):
    prepared = reference(request)
    meta, values = archive(location(root, layout, "public_model"))
    check = "initial_role_identity" if roles_only else "training_cache_reconstruction"
    from glassbox.experimental.public_mean_saved_port import window_arrays

    require(meta["seen"] == prepared.seen, check, "content ledger")
    require(
        meta["contract"] == prepared.contract
        and meta["model"]["dt_s"] == prepared.train.batch.dt_s,
        check,
        "contract/timing",
    )
    for role, expected in (
        ("train", prepared.train),
        ("development", prepared.development),
    ):
        actual = meta["windows"][role]
        expected_keys = [
            dict(recording_id=k.recording_id, segment_id=k.segment_id, origin=k.origin)
            for k in expected.keys
        ]
        require(
            actual["keys"] == expected_keys
            and actual["source_origins"] == list(expected.source_origins),
            check,
            role + " identities/origins",
        )
        identities = sorted({k["recording_id"] for k in actual["keys"]})
        wanted = prepared.roles["training" if role == "train" else "development"]
        require(identities == sorted(wanted), check, role + " role membership")
        require(
            set(meta["report"]["training" if role == "train" else "development"])
            == set(wanted),
            check,
            role + " report identity mirror",
        )
        if not roles_only:
            expected_arrays = window_arrays(expected)
            actual_arrays = {k: values[role + "_" + k] for k in expected_arrays}
            array_equal(actual_arrays, expected_arrays, check)
            require(
                actual["excitation_declared"] is expected.excitation_declared,
                check,
                "excitation declaration",
            )
    return dict(training_windows=1536, development_windows=256, roles=prepared.roles)


def _revision(root, layout):
    from glassbox import LearnedDynamics
    from glassbox.experimental import public_mean_lifecycle as lifecycle

    m0 = LearnedDynamics.load(location(root, layout, "lifecycle_m0"))
    m1 = LearnedDynamics.load(location(root, layout, "lifecycle_m1"))
    check = "revision_chain_identity"
    require(
        m1.report["previous_revision"] == m0.fingerprint(),
        check,
        "actual predecessor fingerprint",
    )
    for role in ("train", "development"):
        windows = getattr(m1, "_" + role)
        keys, arrays = lifecycle._expected_windows("M1", role)
        require(
            [(k.recording_id, k.origin) for k in windows.keys] == keys
            and windows.source_origins == tuple(o for _, o in keys),
            check,
            role + " merged ordered origins",
        )
        array_equal({k: getattr(windows.batch, k) for k in arrays}, arrays, check)
    require(m1._seen == lifecycle._expected_ledger("M1"), check, "content-ledger union")
    from glassbox.experimental.public_mean_saved_port import window_arrays

    array_equal(window_arrays(m1._development), window_arrays(m0._development), check)
    return dict(
        predecessor=m0.fingerprint(),
        updated=m1.fingerprint(),
        immutable_development=True,
    )


def _decision(root, layout, request, evaluator):
    p = read(location(root, layout, "resolved_protocol"))
    qualification = read(
        Path(request["binding"]["public_root"])
        / "docs/harness/public-mean-qualification-v1.json"
    )
    all_rows, queries = {}, {}
    for sim in ("crazyflow", "cascade"):
        data, evaluation = (
            location(root, layout, sim + "_" + key) for key in ("data", "evaluation")
        )
        scored = evaluator.score(data, evaluation, sim, p)
        for name, value in scored.items():
            require(
                value == read(evaluation / (name + ".json")),
                "decision_reduction",
                sim + " raw-array reduction " + name,
            )
        all_rows[sim], queries[sim] = scored["rows"], read(data / "queries.json")
    physical_result = evaluator.reduce_decision(
        all_rows, p, qualification, queries=queries
    )
    gates = {}
    for name, evidence in layout["gates"].items():
        path = under(root, evidence["path"])
        # Evidence files and pointers were independently anchored before copying;
        # none is allowed to derive its Boolean from the final adoption record.
        require(
            evidence["path"] != layout["paths"]["decision"],
            "audit_prerequisite",
            "circular gate provenance",
        )
        require(
            digest(path) == evidence["sha256"],
            "audit_prerequisite",
            "qualification gate anchor",
        )
        gates[name] = pointer(read(path), evidence["pointer"])
    result = evaluator.promotion(physical_result, gates, qualification)
    saved = read(location(root, layout, "decision"))
    require(
        pointer(saved, layout["decision_pointer"]) == result,
        "decision_reduction",
        "frozen promotion and all named evidence gates",
    )
    return result


def _source(root, layout, request):
    import importlib

    binding = request["binding"]
    result = read(
        location(root, layout, "crazyflow_evaluation") / "historical/result.json"
    )
    sources = result["imported_sources"]
    actual = {}
    for name in sources:
        require(
            name == "glassbox" or name.startswith("glassbox."),
            "audit_prerequisite",
            "module namespace",
        )
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        expected_root = Path(binding["oracle_root"]).resolve()
        require(
            path.is_relative_to(expected_root),
            "audit_prerequisite",
            "wrong actual oracle import",
        )
        relative = path.relative_to(expected_root).as_posix()
        require(
            binding["oracle_source_sha256"].get(relative) == digest(path),
            "audit_prerequisite",
            "actual oracle pin",
        )
        actual[name] = dict(path=str(path), sha256=digest(path))
    require(
        sources == actual,
        "worker_source_identity",
        "recorded versus fresh historical imported paths/bytes",
    )
    return dict(imported_sources=actual)


def _consumer(root, layout, request):
    binding = read(location(root, layout, "consumer_binding"))
    destination = Path(request["scratch"]) / "consumer-fresh"
    # Standalone bootstrap below imports no experimental/controller package.
    command = [
        binding["interpreter"],
        str(Path(__file__).resolve()),
        "consumer-worker",
        "--binding",
        str(location(root, layout, "consumer_binding")),
        "--expected-binding-sha256",
        digest(location(root, layout, "consumer_binding")),
        "--manifest",
        str(location(root, layout, "consumer_manifest")),
        "--expected-manifest-sha256",
        digest(location(root, layout, "consumer_manifest")),
        "--output",
        str(destination),
    ]
    environment = dict(os.environ)
    environment.pop("JAX_ENABLE_X64", None)
    environment["SCIPY_ARRAY_API"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        str(Path(binding[k]) / "src") for k in ("public_root", "dart_root")
    )
    execute(
        command,
        Path(request["scratch"]) / "consumer-execution",
        environment,
        binding["dart_root"],
    )
    fresh, saved = (
        read(destination / "result.json"),
        read(location(root, layout, "consumer_result")),
    )
    check = "consumer_execution_identity"
    wanted = binding["consumer_source_sha256"]["src/crazydart/glassbox_forecast.py"]
    actual = fresh["runtime"]["imports"]["crazydart.glassbox_forecast"]
    require(
        actual["sha256"] == wanted and digest(actual["path"]) == wanted,
        "audit_prerequisite",
        "fresh actual consumer source",
    )
    require(
        saved["runtime"] == fresh["runtime"],
        check,
        "recorded actual consumer runtime/source identity",
    )
    for key in saved:
        if key != "arrays_sha256":
            require(saved[key] == fresh[key], check, "fresh consumer output " + key)
    array_equal(
        npz(location(root, layout, "consumer_result").parent / saved["arrays_file"]),
        npz(destination / fresh["arrays_file"]),
        check,
    )
    return dict(counts=fresh["counts"], blocked_imports=True, runtime=fresh["runtime"])


def validate(case, root, layout, request):
    import numpy as np

    check = CHECKS[case]
    if case == "PM01":
        from glassbox import LearnedDynamics

        try:
            LearnedDynamics.load(location(root, layout, "public_model"))
        except ValueError as error:
            # PM01 is intentionally the one raw-loader integrity case.
            raise CheckFailure(check, str(error)) from error
        return dict(public_load=True)
    if case == "PM02":
        return _parity(root, layout, request)
    if case in ("PM03", "PM04"):
        return _cache(root, layout, request, case == "PM04")
    if case == "PM12":
        return _revision(root, layout)
    if case == "PM10":
        return _source(root, layout, request)
    if case == "PM11":
        return _consumer(root, layout, request)
    evaluator = physical(request["binding"])
    if case == "PM09":
        return _decision(root, layout, request, evaluator)
    if case == "PM05":
        from glassbox import LearnedDynamics

        _, evidence = translate(
            lambda: evaluator.calibration_evidence(
                LearnedDynamics.load(location(root, layout, "public_model"))
            ),
            source_id="calibration_envelope_reconstruction",
            target_id=check,
        )
        return evidence
    data = location(root, layout, "crazyflow_data")
    p = read(location(root, layout, "resolved_protocol"))
    if case in ("PM06", "PM07"):
        return translate(
            lambda: evaluator.reconstruct_truth(data, "crazyflow", p),
            source_id="truth_roster_and_masks",
            target_id=check,
        )
    if case == "PM08":
        import jax

        from glassbox import LearnedDynamics

        query = request["targets"]["primary_factual"]
        values = npz(data / query["path"])
        model = LearnedDynamics.load(location(root, layout, "public_model"))
        predicted = evaluator.predict_query(
            jax.jit(model.predict), query, values, dtype=np.float32
        )
        evaluation = location(root, layout, "crazyflow_evaluation")
        path = (
            evaluation / role_for(evaluator, "public_v4") / evaluator.query_path(query)
        )
        array_equal(
            {"public_v4": npz(path)["public_v4"]}, {"public_v4": predicted}, check
        )
        scored = evaluator.score(data, evaluation, "crazyflow", p)
        for name, value in scored.items():
            require(
                value == read(evaluation / (name + ".json")),
                check,
                "physical reduction " + name,
            )
        return dict(query=query["id"], exact_replay=True)
    raise CheckFailure("audit_prerequisite", "undeclared case")


def _swap(value, left, right):
    if isinstance(value, dict):
        return {
            right if k == left else left if k == right else k: _swap(v, left, right)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_swap(v, left, right) for v in value]
    return right if value == left else left if value == right else value


def mutate(case, root, layout, request):
    import numpy as np

    details, changed = {}, []
    if case in ("PM01", "PM02", "PM03", "PM04", "PM05", "PM12"):
        name = {"PM02": "port_model", "PM12": "lifecycle_m1"}.get(case, "public_model")
        path = location(root, layout, name)
        meta, values = archive(path)
        before = digest(path)
        if case == "PM01":
            values["param_linear"].view(np.uint8).reshape(-1)[0] ^= 1
        elif case == "PM02":
            values["param_autonomous"][0, 0] += 0.125
        elif case == "PM03":
            values["train_future_inputs"][384, 0, 0] += 0.01
        elif case == "PM04":
            left = min(k["recording_id"] for k in meta["windows"]["train"]["keys"])
            right = min(
                k["recording_id"] for k in meta["windows"]["development"]["keys"]
            )
            require(
                left != right, "audit_prerequisite", "disjoint original role identities"
            )
            meta["windows"] = _swap(meta["windows"], left, right)
            for key in ("training", "development", "development_errors"):
                meta["report"][key] = _swap(meta["report"][key], left, right)
            details.update(training_identity=left, development_identity=right)
        elif case == "PM05":
            values["envelope_half_width"][0, 0] += 0.001
            meta["report"]["envelope"]["half_width"] = values[
                "envelope_half_width"
            ].tolist()
        elif case == "PM12":
            require(
                meta["report"]["previous_revision"] != "0" * 64,
                "audit_prerequisite",
                "nonzero predecessor",
            )
            meta["report"]["previous_revision"] = "0" * 64
        meta = rewrite_archive(path, meta, values, coherent=case != "PM01")
        details.update(
            model_metadata=meta,
            model_sha256=digest(path),
            model_fingerprint=meta["fingerprint"],
        )
        require(
            digest(path) != before,
            "audit_prerequisite",
            "mutation changed no archive bytes",
        )
        changed.append(path.relative_to(root).as_posix())
    elif case in ("PM06", "PM07", "PM08"):
        evaluator = physical(request["binding"])
        data = location(root, layout, "crazyflow_data")
        evaluation = location(root, layout, "crazyflow_evaluation")
        query = request["targets"][
            "complete_factual" if case == "PM06" else "primary_factual"
        ]
        details["query"] = query
        if case == "PM06":
            path = data / query["path"]
            values = npz(path)
            values["future_inputs"][0, 0] += 0.01
            np.savez_compressed(path, **values)
            changed.append(path.relative_to(root).as_posix())
        elif case == "PM08":
            path = (
                evaluation
                / role_for(evaluator, "public_v4")
                / evaluator.query_path(query)
            )
            values = npz(path)
            require(
                np.isfinite(values["public_v4"][0, 0]),
                "audit_prerequisite",
                "finite selected prediction target",
            )
            values["public_v4"][0, 0] += 0.125
            np.savez_compressed(path, **values)
            changed.append(path.relative_to(root).as_posix())
            scored = evaluator.score(
                data,
                evaluation,
                "crazyflow",
                read(location(root, layout, "resolved_protocol")),
            )
            for name, value in scored.items():
                path = evaluation / (name + ".json")
                write(path, value)
                changed.append(path.relative_to(root).as_posix())
        else:
            from glassbox.experimental.two_simulator_metrics import aggregate

            identity = (query["parent"], query["id"])
            data_seal = read(data / "seal.json")
            details["remaining_data_arrays"] = data_seal["arrays"] - len(
                npz(data / query["path"])
            )
            details["remaining_data_queries"] = data_seal["queries"] - 1
            data_seal["queries"] = details["remaining_data_queries"]
            data_seal["arrays"] = details["remaining_data_arrays"]
            write(data / "seal.json", data_seal)
            changed.append((data / "seal.json").relative_to(root).as_posix())
            rows = read(evaluation / "rows.json")
            rows = [r for r in rows if (r["parent"], r["query"]) != identity]
            directions = read(evaluation / "directions.json")
            remaining = [
                q
                for q in read(data / "queries.json")
                if (q["parent"], q["id"]) != identity
            ]
            for path, value in (
                (data / "queries.json", remaining),
                (evaluation / "rows.json", rows),
                (evaluation / "directions.json", directions),
                (evaluation / "summary.json", aggregate(rows)),
            ):
                write(path, value)
                changed.append(path.relative_to(root).as_posix())
            for role, arms in evaluator.ROLES.items():
                path = evaluation / role / evaluator.query_path(query)
                path.unlink()
                changed.append(path.relative_to(root).as_posix())
                result_path = evaluation / role / "result.json"
                result = read(result_path)
                for arm in arms:
                    result["counters"][arm]["branch_queries"] -= 1
                    result["counters"][arm]["inferred_prefixes"] -= 1
                write(result_path, result)
                changed.append(result_path.relative_to(root).as_posix())
            path = data / query["path"]
            path.unlink()
            changed.append(path.relative_to(root).as_posix())
    else:
        if case == "PM09":
            path = location(root, layout, "decision")
            value = read(path)
            keys = layout["decision_pointer"] + ["public_mean_adopted"]
            old = pointer(value, keys)
            require(type(old) is bool, "audit_prerequisite", "Boolean adoption target")
            assign(value, keys, not old)
        elif case == "PM10":
            path = (
                location(root, layout, "crazyflow_evaluation")
                / "historical/result.json"
            )
            value = read(path)
            keys = ["imported_sources", "glassbox.learner", "sha256"]
            replacement = request["binding"]["public_source_sha256"][
                "src/glassbox/learner.py"
            ]
            require(
                pointer(value, keys) != replacement,
                "audit_prerequisite",
                "distinct learner sources",
            )
            assign(value, keys, replacement)
        elif case == "PM11":
            path = location(root, layout, "consumer_result")
            value = read(path)
            keys = ["runtime", "imports", "crazydart.glassbox_forecast", "sha256"]
            require(
                pointer(value, keys) != "0" * 64,
                "audit_prerequisite",
                "nonzero consumer source",
            )
            assign(value, keys, "0" * 64)
        else:
            raise CheckFailure("audit_prerequisite", "unknown mutation")
        write(path, value)
        changed.append(path.relative_to(root).as_posix())
        details["target_pointer"] = keys
    return details, changed


def repair(root, layout, case, details):
    """Only layout-declared mirrors and bottom-up file inventories are rewritten."""
    changed = []
    for mirror in layout["mirrors"][case]:
        target = under(root, mirror["path"])
        value = read(target)
        source = mirror["source"]
        if source["kind"] == "detail":
            replacement = pointer(details, source["pointer"])
        elif source["kind"] == "json":
            replacement = pointer(read(under(root, source["path"])), source["pointer"])
        else:
            raise CheckFailure("audit_prerequisite", "unknown declared mirror kind")
        assign(value, mirror["pointer"], replacement)
        write(target, value)
        changed.append(mirror["path"])
    # The layout is anchored separately, and explicitly orders dependency links
    # and nested inventories before their ancestors. No recursive guessing.
    for node in layout["reseal_order"]:
        path = under(root, node["path"])
        value = read(path)
        if node["kind"] == "hash_links":
            for link in node["links"]:
                assign(value, link["pointer"], digest(under(root, link["payload"])))
        elif node["kind"] == "inventory":
            base = under(root, node["root"])
            assign(value, node["pointer"], inventory(base, node["excluding"]))
        else:
            raise CheckFailure("audit_prerequisite", "unknown seal operation")
        write(path, value)
        changed.append(node["path"])
    verify_internal(root, layout)
    return changed


def verify_internal(root, layout):
    for node in layout["reseal_order"]:
        value = read(under(root, node["path"]))
        if node["kind"] == "hash_links":
            for link in node["links"]:
                require(
                    pointer(value, link["pointer"])
                    == digest(under(root, link["payload"])),
                    "internal_seals",
                    "repaired payload hash",
                )
        else:
            require(
                pointer(value, node["pointer"])
                == inventory(under(root, node["root"]), node["excluding"]),
                "internal_seals",
                "repaired complete inventory",
            )
    from glassbox._learner_arrays import load_arrays

    for name in ("public_model", "port_model", "lifecycle_m0", "lifecycle_m1"):
        load_arrays(location(root, layout, name))
    return True


def execute(command, directory, environment, cwd):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write(
        directory / "command.json",
        dict(
            command=command,
            cwd=str(cwd),
            environment={
                k: environment.get(k)
                for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
            },
        ),
    )
    started = time.monotonic()
    status = dict(returncode=None, timeout=False)
    try:
        with (
            (directory / "stdout.log").open("w") as out,
            (directory / "stderr.log").open("w") as err,
        ):
            result = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=out,
                stderr=err,
                timeout=HARD_TIMEOUT_S,
                check=False,
            )
        status["returncode"] = result.returncode
    except subprocess.TimeoutExpired:
        status["timeout"] = True
    finally:
        status["wall_time_s"] = time.monotonic() - started
        write(directory / "exit.json", status)
    require(
        status["returncode"] == 0 and not status["timeout"],
        "audit_prerequisite",
        "subprocess failed; retained logs",
    )


def semantic(case, root, request, directory):
    request = dict(request, case=case, bundle=str(root), scratch=str(directory))
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "request.json", request)
    binding = request["binding"]
    role = case_role(case)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(
        Path(binding["oracle_root" if role == "historical" else "public_root"]) / "src"
    )
    environment["SCIPY_ARRAY_API"] = "1"
    if role == "public32":
        environment.pop("JAX_ENABLE_X64", None)
    else:
        environment["JAX_ENABLE_X64"] = "1"
    execute(
        [
            binding["interpreter"],
            str(Path(__file__).resolve()),
            "validate",
            str(directory / "request.json"),
        ],
        directory / "execution",
        environment,
        binding["public_root"],
    )
    return read(directory / "result.json")


def clone(source, destination, evidence):
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), "audit_prerequisite", "fresh case directory")
    command = ["/bin/cp", "-cR", str(source), str(destination)]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result = dict(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        strategy="copy-on-write",
    )
    if completed.returncode:
        # An unsupported clone operation may leave a partial destination. It is
        # inside the fresh audit-owned case directory; preserve command failure.
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, symlinks=False)
        result["strategy"] = "ordinary-copy-after-recorded-clone-failure"
    for path in source.rglob("*"):
        if path.is_file():
            other = destination / path.relative_to(source)
            require(
                path.stat().st_ino != other.stat().st_ino
                or path.stat().st_dev != other.stat().st_dev,
                "audit_prerequisite",
                "hardlinked case payload",
            )
    result["wall_time_s"] = time.monotonic() - started
    write(evidence, result)
    return result


def preflight(bundle, layout, binding, reference_root):
    """Authenticate every stage association before selecting or altering targets."""
    required = {
        "public_model",
        "port_model",
        "lifecycle_m0",
        "lifecycle_m1",
        "decision",
        "resolved_protocol",
        "crazyflow_data",
        "cascade_data",
        "crazyflow_evaluation",
        "cascade_evaluation",
        "numeric_inputs",
        "numeric_oracle",
        "consumer_manifest",
        "consumer_result",
        "consumer_binding",
    }
    require(
        set(layout["paths"]) == required,
        "audit_prerequisite",
        "exact logical path roster",
    )
    for name in required:
        require(
            location(bundle, layout, name).exists(),
            "audit_prerequisite",
            "missing " + name,
        )
    require(
        layout["anchors"] and layout["stage_bindings"],
        "audit_prerequisite",
        "explicit stage anchors/bindings",
    )
    for item in layout["anchors"]:
        require(
            digest(under(bundle, item["path"])) == item["sha256"],
            "audit_prerequisite",
            "external stage anchor",
        )
    anchor_paths = {item["path"] for item in layout["anchors"]}
    require(
        layout["paths"]["consumer_binding"] in anchor_paths,
        "audit_prerequisite",
        "consumer binding external anchor",
    )
    for item in layout["stage_bindings"]:
        path = under(bundle, item["path"])
        require(
            digest(path) == item["sha256"],
            "audit_prerequisite",
            "original stage binding SHA",
        )
        old = read(path)
        require(
            old["protocol_sha256"] == PROTOCOL_SHA256
            and old["runtime"] == binding["runtime"]
            and old["interpreter_sha256"] == binding["interpreter_sha256"]
            and old["oracle_source_sha256"] == binding["oracle_source_sha256"],
            "audit_prerequisite",
            "stage protocol/runtime/historical source pins",
        )
        require(
            bool(item["unchanged_sources"]),
            "audit_prerequisite",
            "explicit continuity scope",
        )
        for relative in item["unchanged_sources"]:
            require(
                old["public_source_sha256"].get(relative)
                == binding["public_source_sha256"].get(relative)
                == digest(Path(binding["public_root"]) / relative),
                "audit_prerequisite",
                "reused stage source continuity " + relative,
            )
        # The original binding remains associated with the original stage's own
        # recorded commit. No stage is relabeled as the current implementation.
        for association in item["associations"]:
            recorded = pointer(
                read(under(bundle, association["path"])), association["pointer"]
            )
            require(
                recorded == item["sha256"],
                "audit_prerequisite",
                "stage original binding association",
            )
    consumer = read(location(bundle, layout, "consumer_binding"))
    require(
        consumer["consumer_source_sha256"] == binding["consumer_source_sha256"],
        "audit_prerequisite",
        "consumer source continuity",
    )
    for relative in (
        "src/glassbox/__init__.py",
        "src/glassbox/learner.py",
        "src/glassbox/_sequence_model.py",
        "src/glassbox/_learner_arrays.py",
        "src/glassbox/recordings.py",
        "src/glassbox/io/__init__.py",
        "src/glassbox/io/recordings.py",
    ):
        require(
            consumer["public_source_sha256"][relative]
            == binding["public_source_sha256"][relative],
            "audit_prerequisite",
            "consumer public dependency continuity",
        )
    evaluator = physical(binding)
    qualification = read(
        Path(binding["public_root"]) / "docs/harness/public-mean-qualification-v1.json"
    )
    require(
        read(location(bundle, layout, "resolved_protocol"))
        == evaluator.resolved(reference_root, qualification, binding["public_root"]),
        "audit_prerequisite",
        "exact inherited/fresh evaluation resolution",
    )
    require(
        set(layout["gates"])
        == set(qualification["public_mean_promotion"]["all_required"]),
        "audit_prerequisite",
        "complete independently grounded gate roster",
    )
    for item in layout["gates"].values():
        require(
            digest(under(bundle, item["path"])) == item["sha256"]
            and type(pointer(read(under(bundle, item["path"])), item["pointer"]))
            is bool,
            "audit_prerequisite",
            "anchored Boolean gate evidence",
        )
    require(
        layout["reseal_order"][-1]
        == dict(
            kind="inventory",
            path=layout["root_manifest"],
            root=".",
            pointer=["files"],
            excluding=[layout["root_manifest"]],
        ),
        "audit_prerequisite",
        "last repair seals entire qualification bundle",
    )


def worker_identity(request):
    import importlib.metadata

    import jax

    binding = request["binding"]
    role = case_role(request["case"])
    require(
        digest(__file__) == binding["public_source_sha256"][RELATIVE],
        "audit_prerequisite",
        "actual audit worker source",
    )
    require(
        Path(sys.executable).resolve() == Path(binding["interpreter"]).resolve()
        and digest(Path(sys.executable).resolve()) == binding["interpreter_sha256"]
        and jax.default_backend() == "cpu"
        and bool(jax.config.x64_enabled) == (role != "public32"),
        "audit_prerequisite",
        "actual audit runtime/precision",
    )
    for name in ("jax", "jaxlib", "numpy", "scipy"):
        require(
            importlib.metadata.version(name) == binding["runtime"][name],
            "audit_prerequisite",
            "actual runtime version",
        )
    root = Path(
        binding["oracle_root" if role == "historical" else "public_root"]
    ).resolve()
    pins = binding[
        "oracle_source_sha256" if role == "historical" else "public_source_sha256"
    ]
    for name, module in tuple(sys.modules.items()):
        if (name == "glassbox" or name.startswith("glassbox.")) and getattr(
            module, "__file__", None
        ):
            path = Path(module.__file__).resolve()
            require(
                path.is_relative_to(root)
                and pins.get(path.relative_to(root).as_posix()) == digest(path),
                "audit_prerequisite",
                "actual imported source role " + name,
            )


def run(
    bundle,
    output,
    *,
    expected_bundle_sha256,
    layout_path,
    expected_layout_sha256,
    binding_path,
    expected_binding_sha256,
    reference_root,
    reference_sha256,
):
    from glassbox.experimental.public_mean_implementation import verify

    binding = verify(binding_path, expected_binding_sha256)
    require(
        Path(__file__).resolve() == Path(binding["public_root"]) / RELATIVE
        and binding["public_source_sha256"].get(RELATIVE) == digest(__file__),
        "audit_prerequisite",
        "committed audit source",
    )
    require(
        digest(layout_path) == expected_layout_sha256,
        "audit_prerequisite",
        "external layout SHA",
    )
    layout = read(layout_path)
    require(
        layout["format"] == FORMAT and list(layout["mirrors"]) == list(CHECKS),
        "audit_prerequisite",
        "exact ordered twelve-case layout",
    )
    require(
        layout["protocol_sha256"] == PROTOCOL_SHA256,
        "audit_prerequisite",
        "frozen protocol",
    )
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    require(
        not output.is_relative_to(bundle) and not bundle.is_relative_to(output),
        "audit_prerequisite",
        "separate immutable source and audit output",
    )
    original = anchor(bundle, layout["root_manifest"], expected_bundle_sha256)
    verify_internal(bundle, layout)
    preflight(bundle, layout, binding, reference_root)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(layout_path, output / "layout.json")
    shutil.copy2(__file__, output / "executed-audit-source.py")
    request = dict(
        layout=layout,
        binding=binding,
        reference_root=str(Path(reference_root).resolve()),
        reference_sha256=reference_sha256,
        original_bundle=str(bundle),
        original_files=original["files"],
        targets={
            "complete_factual": target_query(bundle, layout, complete=True),
            "primary_factual": target_query(bundle, layout, complete=False),
        },
    )
    write(
        output / "intent.json",
        dict(
            format=FORMAT,
            bundle=str(bundle),
            bundle_sha256=expected_bundle_sha256,
            layout_sha256=expected_layout_sha256,
            implementation_sha256=expected_binding_sha256,
            script_sha256=digest(__file__),
            cases=CHECKS,
            logical_bytes=sum(
                p.stat().st_size for p in bundle.rglob("*") if p.is_file()
            ),
            scope="Twelve fixed actual-bundle mutations; fresh semantic workers; no fits, gradients, solves or simulators.",
        ),
    )
    result = dict(
        format=FORMAT, status="failed", clean={}, cases=[], original_reverified=False
    )
    try:
        for case in CHECKS:
            clean = semantic(case, bundle, request, output / (case + "-clean"))
            require(
                clean["status"] == "passed",
                "audit_prerequisite",
                case + " clean direct validator",
            )
            result["clean"][case] = clean
            write(output / "result.json", result)
        for case, check in CHECKS.items():
            directory = output / case
            directory.mkdir()
            copied = directory / "bundle"
            clone(bundle, copied, directory / "copy.json")
            anchor(copied, layout["root_manifest"], expected_bundle_sha256)
            details, changed = mutate(case, copied, layout, request)
            # Full metadata is useful for mirror repair, not duplicated in report.
            evidence = {k: v for k, v in details.items() if k != "model_metadata"}
            write(
                directory / "mutation.json",
                dict(details=evidence, directly_changed=changed),
            )
            repaired = [] if case == "PM01" else repair(copied, layout, case, details)
            write(
                directory / "repair.json",
                dict(repaired=repaired, coherent_internal_seals=case != "PM01"),
            )
            current = inventory(copied, (layout["root_manifest"],))
            difference = {
                k
                for k in set(original["files"]) | set(current)
                if original["files"].get(k) != current.get(k)
            }
            require(
                difference <= set(changed + repaired),
                "audit_prerequisite",
                "undeclared mutation/mirror payload",
            )
            rejected = False
            try:
                anchor(copied, layout["root_manifest"], expected_bundle_sha256)
            except CheckFailure as error:
                require(
                    error.check_id == "external_anchor",
                    "audit_prerequisite",
                    "wrong integrity prerequisite",
                )
                rejected = True
            require(
                rejected,
                "audit_prerequisite",
                "unchanged external anchor accepted altered bundle",
            )
            semantic_result = semantic(case, copied, request, directory / "semantic")
            require(
                semantic_result["status"] == "rejected"
                and semantic_result["check_id"] == check,
                "audit_prerequisite",
                case + " wrong semantic outcome",
            )
            row = dict(
                id=case,
                check_id=check,
                mutation=evidence,
                changed_payloads=sorted(difference),
                coherent_internal_seals=case != "PM01",
                external_anchor_rejected=True,
                semantic=semantic_result,
                cleanup="pending",
            )
            result["cases"].append(row)
            write(output / "result.json", result)
            anchor(bundle, layout["root_manifest"], expected_bundle_sha256)
            shutil.rmtree(copied)
            row["cleanup"] = "removed disposable copy after passed challenge"
            write(output / "result.json", result)
        anchor(bundle, layout["root_manifest"], expected_bundle_sha256)
        result.update(
            status="passed",
            original_reverified=True,
            raw_integrity_cases=1,
            coherent_semantic_cases=11,
        )
    except Exception as error:
        result["error"] = dict(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        try:
            anchor(bundle, layout["root_manifest"], expected_bundle_sha256)
            result["original_reverified"] = True
        except Exception as error:
            result["original_reverification_error"] = dict(
                type=type(error).__name__, message=str(error)
            )
            result["status"] = "failed"
        write(output / "result.json", result)
    require(
        result["status"] == "passed",
        "audit_prerequisite",
        "original integrity failed at completion",
    )
    return result


def consumer_worker(argv):
    import importlib.abc

    allowed = {
        "glassbox",
        "glassbox.learner",
        "glassbox._sequence_model",
        "glassbox._learner_arrays",
        "glassbox.recordings",
        "glassbox.io",
        "glassbox.io.recordings",
    }

    class Boundary(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("glassbox.") and fullname not in allowed:
                raise ImportError("consumer forbids nonpublic module: " + fullname)
            return None

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    args, consumer_argv = parser.parse_known_args(argv)
    require(
        digest(args.binding) == args.expected_binding_sha256,
        "audit_prerequisite",
        "consumer original binding anchor",
    )
    binding = read(args.binding)
    sys.meta_path.insert(0, Boundary())
    from crazydart.glassbox_forecast import main as consumer_main
    from crazydart.glassbox_forecast import observed_runtime

    actual = observed_runtime()
    require(
        actual["interpreter_sha256"] == binding["interpreter_sha256"]
        and Path(actual["interpreter"]).resolve()
        == Path(binding["interpreter"]).resolve()
        and actual["jax_enable_x64"] is False,
        "audit_prerequisite",
        "consumer fresh runtime before forecasting",
    )
    for name, value in actual["imports"].items():
        if name in allowed:
            root = Path(binding["public_root"]).resolve()
            pins = binding["public_source_sha256"]
        elif name in ("crazydart", "crazydart.glassbox_forecast"):
            root = Path(binding["dart_root"]).resolve()
            pins = binding["consumer_source_sha256"]
        else:
            raise CheckFailure(
                "audit_prerequisite", "unexpected actual consumer import"
            )
        path = Path(value["path"]).resolve()
        require(
            path.is_relative_to(root)
            and pins.get(path.relative_to(root).as_posix())
            == value["sha256"]
            == digest(path),
            "audit_prerequisite",
            "consumer imported path/source before forecasting",
        )
    consumer_main(consumer_argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "consumer-worker":
        consumer_worker(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("validate")
    worker.add_argument("request")
    runner = sub.add_parser("run")
    runner.add_argument("bundle")
    runner.add_argument("output")
    for key in (
        "expected-bundle-sha256",
        "layout-path",
        "expected-layout-sha256",
        "binding-path",
        "expected-binding-sha256",
        "reference-root",
        "reference-sha256",
    ):
        runner.add_argument("--" + key, required=True)
    args = vars(parser.parse_args(argv))
    if args.pop("command") == "run":
        print(json.dumps(run(**args), sort_keys=True))
        return
    request = read(args["request"])
    try:
        worker_identity(request)
        evidence = validate(
            request["case"], Path(request["bundle"]), request["layout"], request
        )
        result = dict(
            status="passed", check_id=CHECKS[request["case"]], evidence=evidence
        )
    except CheckFailure as error:
        result = dict(status="rejected", check_id=error.check_id, detail=str(error))
    except Exception as error:
        result = dict(
            status="unexpected_error",
            type=type(error).__name__,
            detail=str(error),
            traceback=traceback.format_exc(),
        )
    try:
        worker_identity(request)
    except Exception as error:
        result = dict(
            status="unexpected_error",
            type=type(error).__name__,
            detail=str(error),
            traceback=traceback.format_exc(),
        )
    write(Path(request["scratch"]) / "result.json", result)


if __name__ == "__main__":
    main()
