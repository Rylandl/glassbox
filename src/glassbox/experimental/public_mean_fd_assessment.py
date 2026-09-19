"""Prospective no-fit finite-difference assessment.

Commit this file and the policy before preparing inputs or running trials.
Public and historical numerical implementations run in separate subprocesses.
The existing exactly-24 prepare helper is untouched; numerical run accepts41.

Request fields (external SHA required at CLI): binding, binding_sha256,
policy_path, assessment_protocol_sha256, relative_source_path,
original_manifest, original_manifest_sha256, original_run, original_run_sha256,
data={crazyflow|cascade: {root, seal_sha256}}. Run/reduce additionally require
manifest, manifest_sha256. The caller supplies run_sha256 for saved verification.
`prepare` returns a new manifest SHA; anchor it externally before `run`.
Old numerical protocol_sha256 remains the inherited v1 arithmetic contract;
assessment_protocol_sha256 is separate. This code never reports adoption.
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

import numpy as np

STEPS = (2.0**-7, 2.0**-8, 2.0**-9, 2.0**-10)
INPUTS = ("past_states", "past_inputs", "future_inputs")
SCOPES = {"primary", "heading_shift", "maneuver_shift", "speed_shift", "wind_shift"}
MODES = ("public32", "oracle64")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {
            k: np.array(z[k], copy=True) for k in (z.files if keys is None else keys)
        }


def under(root, relative):
    root = Path(root).resolve()
    parts = Path(relative).parts
    require(
        not Path(relative).is_absolute()
        and parts
        and not any(p in (".", "..") for p in parts),
        "relative path",
    )
    path = root
    for part in parts:
        path /= part
        require(not path.is_symlink(), "symlink payload")
    require(
        path.is_file() and path.resolve().is_relative_to(root),
        "missing/escaped payload",
    )
    return path


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def input_key(values):
    h = hashlib.sha256()
    for name in INPUTS:
        a = np.asarray(values[name])
        h.update(
            json.dumps(
                [name, a.dtype.str, list(a.shape)], separators=(",", ":")
            ).encode()
        )
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def helper(binding, name):
    relative = "src/glassbox/experimental/" + name + ".py"
    path = under(binding["public_root"], relative)
    require(sha(path) == binding["public_source_sha256"][relative], "helper source")
    spec = importlib.util.spec_from_file_location("_fd_assessment_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate(request):
    require(
        sha(request["binding"]) == request["binding_sha256"],
        "external implementation binding",
    )
    b = read(request["binding"])
    path = under(b["public_root"], request["relative_source_path"])
    require(
        path.resolve() == Path(__file__).resolve()
        and sha(path) == b["public_source_sha256"][request["relative_source_path"]],
        "assessment source",
    )
    require(
        sha(request["policy_path"]) == request["assessment_protocol_sha256"],
        "separate frozen assessment policy",
    )
    n = helper(b, "public_v4_numerics")
    f = helper(b, "public_mean_fd_diagnosis")
    for prefix, commit in (
        ("public", "implementation_commit"),
        ("oracle", "oracle_commit"),
    ):
        root = Path(b[prefix + "_root"])
        require(
            n._git(root, "rev-parse", "HEAD") == b[commit]
            and not n._git(root, "status", "--porcelain", "--untracked-files=normal"),
            "source revision/cleanliness",
        )
        n._sources(root, b[prefix + "_source_sha256"])
    require(
        Path(sys.executable).resolve() == Path(b["interpreter"]).resolve()
        and sha(Path(sys.executable).resolve()) == b["interpreter_sha256"],
        "interpreter",
    )
    original_path = Path(request["original_manifest"])
    require(
        sha(original_path) == request["original_manifest_sha256"],
        "original input anchor",
    )
    old = read(original_path)
    require(
        len(old["cases"]) == 33
        and sum(c["source"] == "public_archive" for c in old["cases"]) == 24,
        "original33/24 roster",
    )
    for c in old["cases"]:
        require(
            sha(under(original_path.parent, c["path"])) == c["sha256"],
            "original case bytes",
        )
        if c["model_path"] is not None:
            require(
                sha(under(original_path.parent, c["model_path"])) == c["model_sha256"],
                "original model bytes",
            )
    old_run = Path(request["original_run"])
    require(
        sha(old_run / "run.json") == request["original_run_sha256"],
        "original numeric run",
    )
    require(
        read(old_run / "run.json")["input_manifest_sha256"]
        == request["original_manifest_sha256"],
        "original run/input link",
    )
    f.inventory(old_run, read(old_run / "run.json")["files"])
    require(
        read(old_run / "result.json")["passed"] is False,
        "historical failed verdict remains false",
    )
    return b, n, f, old


def select_additions(request, old):
    """Input-only selector; never load targets, masks, predictions or residuals."""
    original_root = Path(request["original_manifest"]).parent
    actual = [c for c in old["cases"] if c["source"] == "public_archive"]
    old_keys = {
        input_key(
            {k: v[0] for k, v in arrays(original_root / c["path"], INPUTS).items()}
        )
        for c in actual
    }
    old_ids = {
        (
            c["source_identity"]["query"]["model_id"],
            c["source_identity"]["query"]["parent"],
            c["source_identity"]["query"]["id"],
        )
        for c in actual
    }
    require(set(request["data"]) == {"crazyflow", "cascade"}, "two source cohorts")
    additions = []
    for sim in sorted(request["data"]):
        source = request["data"][sim]
        root = Path(source["root"])
        require(
            sha(root / "seal.json") == source["seal_sha256"],
            "external physical data seal",
        )
        seal = read(root / "seal.json")
        for relative, expected in seal["files"].items():
            require(sha(under(root, relative)) == expected, "sealed source payload")
        template = next(
            c for c in actual if c["source_identity"]["query"]["model_id"] == sim
        )
        require(
            template["source_identity"]["data_seal_sha256"] == source["seal_sha256"],
            "same physical cohort",
        )
        values = arrays(original_root / template["path"])
        shapes = {k: values[k].shape[1:] for k in INPUTS}
        choices, identifiers = [], set()
        for row in read(root / "queries.json"):
            require(
                row["scope"] in SCOPES and row["kind"] in ("factual", "response"),
                "query taxonomy",
            )
            identity = (row["parent"], row["id"])
            require(identity not in identifiers, "duplicate source query identity")
            identifiers.add(identity)
            require(
                type(row["history_eligible"]) is bool, "explicit history eligibility"
            )
            if (
                row["kind"] != "factual"
                or not row["history_eligible"]
                or (sim, *identity) in old_ids
            ):
                continue
            path = under(root, row["path"])
            x = arrays(path, INPUTS)
            if any(
                x[k].shape != shapes[k] or not np.isfinite(x[k]).all() for k in INPUTS
            ):
                continue
            require(
                all(
                    x[k].dtype.str == "<f8" and x[k].flags.c_contiguous for k in INPUTS
                ),
                "original input layout",
            )
            key = input_key(x)
            if key not in old_keys:
                choices.append((identity, key, row, x))
        # Global simulator dedup BEFORE partition; first sorted identity wins.
        retained, seen = [], set()
        for item in sorted(choices, key=lambda item: item[0]):
            if item[1] not in seen:
                retained.append(item)
                seen.add(item[1])
        selected = []
        for scope in ("primary", "shifted"):
            group = [
                r
                for r in retained
                if ("primary" if r[2]["scope"] == "primary" else "shifted") == scope
            ]
            require(
                len(group) >= 2,
                "insufficient confirmation inputs: " + sim + "/" + scope,
            )
            selected.extend((group[0], group[-1]))
        require(
            len({item[1] for item in selected}) == 4, "four unique confirmation triples"
        )
        records = {r["id"]: r for r in read(root / "records.json")}
        for _identity, key, row, x in sorted(selected, key=lambda item: item[0]):
            record = records[row["parent"]]
            parent_path = under(root, record["prefix"] + ".npz")
            meta = {
                k: copy.deepcopy(v)
                for k, v in template.items()
                if k not in ("path", "sha256", "source_identity")
            }
            meta["id"] = (
                sim + "/" + row["parent"] + "/" + row["id"] + "/factual-confirmation"
            )
            meta["source_identity"] = {
                "role": "confirmation",
                "model_id": sim,
                "query": {
                    "model_id": sim,
                    "parent": row["parent"],
                    "id": row["id"],
                    "kind": "factual",
                    "scope": row["scope"],
                    "source_origin": row["origin"],
                    "segment_id": "valid-prefix",
                    "segment_start_row": 0,
                },
                "data_seal_sha256": source["seal_sha256"],
                "source_query_path": row["path"],
                "source_query_sha256": sha(under(root, row["path"])),
                "source_parent_sha256": sha(parent_path),
                "source_record_sha256": hashlib.sha256(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "input_triple_sha256": key,
            }
            supplied = {k: v for k, v in values.items() if k not in INPUTS}
            supplied.update({k: x[k][None] for k in INPUTS})
            additions.append((meta, supplied))
    require(len(additions) == 8, "eight confirmation cases")
    return additions


def prepare(request, output):
    b, n, _, old = authenticate(request)
    added = select_additions(request, old)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = copy.deepcopy(old["cases"])
    origin = Path(request["original_manifest"]).parent
    for c in old["cases"]:
        shutil.copyfile(origin / c["path"], output / c["path"])
        if c["model_path"] is not None and not (output / c["model_path"]).exists():
            shutil.copyfile(origin / c["model_path"], output / c["model_path"])
    for meta, values in added:
        n._check_arrays(meta, values)
        path = output / f"case-{len(cases):03d}.npz"
        np.savez_compressed(path, **values)
        cases.append({**meta, "path": path.name, "sha256": sha(path)})
    provenance = {
        k: copy.deepcopy(b[k])
        for k in (
            "implementation_commit",
            "protocol_sha256",
            "runtime",
            "public_source_sha256",
            "oracle_source_sha256",
        )
    }
    provenance.update(
        implementation_manifest_sha256=request["binding_sha256"],
        assessment_protocol_sha256=request["assessment_protocol_sha256"],
        original_input_manifest_sha256=request["original_manifest_sha256"],
    )
    manifest = {
        "format": old["format"],
        "provenance": provenance,
        "cases": cases,
        "assessment": {
            "protocol_sha256": request["assessment_protocol_sha256"],
            "regression_ids": [c["id"] for c in old["cases"]],
            "confirmation_ids": [c["id"] for c in cases[33:]],
            "original_v1_qualification_passed": False,
            "selector_source_sha256": sha(__file__),
        },
    }
    n.write_json(output / "manifest.json", manifest)
    return {
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": sha(output / "manifest.json"),
        "regression_cases": 33,
        "confirmation_cases": 8,
    }


def packet(request, *, reconstruct=False):
    b, n, f, old = authenticate(request)
    path = Path(request["manifest"])
    require(
        sha(path) == request["manifest_sha256"], "external assessment input manifest"
    )
    manifest = read(path)
    require(
        manifest["cases"][:33] == old["cases"] and len(manifest["cases"]) == 41,
        "unchanged regression prefix plus8",
    )
    require(
        manifest["assessment"]["original_v1_qualification_passed"] is False
        and manifest["assessment"]["protocol_sha256"]
        == request["assessment_protocol_sha256"],
        "separate assessment policy",
    )
    require(
        manifest["assessment"]["regression_ids"] == [c["id"] for c in old["cases"]]
        and manifest["assessment"]["confirmation_ids"]
        == [c["id"] for c in manifest["cases"][33:]],
        "case roles",
    )
    for key in (
        "implementation_commit",
        "protocol_sha256",
        "runtime",
        "public_source_sha256",
        "oracle_source_sha256",
    ):
        require(manifest["provenance"][key] == b[key], "execution provenance " + key)
    require(
        manifest["provenance"]["assessment_protocol_sha256"]
        == request["assessment_protocol_sha256"]
        and manifest["provenance"]["implementation_manifest_sha256"]
        == request["binding_sha256"],
        "assessment provenance",
    )
    require(len({c["id"] for c in manifest["cases"]}) == 41, "unique case IDs")
    for c in manifest["cases"]:
        require(sha(under(path.parent, c["path"])) == c["sha256"], "case bytes")
        n._check_arrays(c, arrays(path.parent / c["path"]))
        if c["model_path"] is not None:
            require(
                sha(under(path.parent, c["model_path"])) == c["model_sha256"],
                "saved model",
            )
    if reconstruct:
        for (meta, supplied), c in zip(
            select_additions(request, old), manifest["cases"][33:], strict=True
        ):
            require(
                {k: v for k, v in c.items() if k not in ("path", "sha256")} == meta,
                "replayed input-only selection",
            )
            stored = arrays(path.parent / c["path"])
            require(
                set(stored) == set(supplied)
                and all(same(stored[k], supplied[k]) for k in stored),
                "replayed selected arrays",
            )
    return b, n, f, manifest


def geometry(data):
    x = tuple(np.array(data[k][0], copy=True) for k in INPUTS)
    h, d = x[2].shape[0], x[0].shape[1]
    c = np.sin(np.arange(1, h * d + 1)).reshape(h, d) / (h * d)
    directions = []
    for i, value in enumerate(x):
        v = np.sin(np.arange(1, value.size + 1, dtype=np.float64)).reshape(value.shape)
        v /= np.sqrt(np.sum(v**2))
        v *= data["norm_state_scale" if i == 0 else "norm_input_scale"]
        directions.append(v)
    return x, tuple(directions), c


def worker(request, mode, output):
    b, n, _f, manifest = packet(request)
    require(mode in MODES, "worker mode")
    prefix = "public" if mode == "public32" else "oracle"
    root = Path(b[prefix + "_root"])
    import jax
    import jax.numpy as jnp

    require(
        bool(jax.config.x64_enabled) == (mode == "oracle64"), "fixed worker precision"
    )
    runtime = n._runtime()
    require(
        runtime["backend"] == "cpu"
        and all(runtime[k] == v for k, v in b["runtime"].items()),
        "worker runtime",
    )
    if mode == "oracle64":
        from glassbox.experimental.state_quadratic_model import (
            KIND,
            QuadraticSequenceModel,
        )

        cls = QuadraticSequenceModel
    else:
        from glassbox import LearnedDynamics
        from glassbox._sequence_model import KIND, SequenceModel

        cls = SequenceModel
    n._imported_sources(root, b[prefix + "_source_sha256"])
    env = dict(os.environ)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, case in enumerate(manifest["cases"]):
        data = arrays(Path(request["manifest"]).parent / case["path"])
        model = cls(
            KIND,
            case["dt_s"],
            case["history_steps"],
            n._tree(data, "param_", n.PARAMETERS),
            n._tree(data, "norm_", n.NORMS),
            case["delay_steps"],
        )
        if mode == "public32" and case["source"] == "public_archive":
            model = LearnedDynamics.load(
                Path(request["manifest"]).parent / case["model_path"]
            )
            require(
                model.fingerprint() == case["model_fingerprint"],
                "public model fingerprint",
            )
            predict = model.predict
        else:
            predict = model.rollout
        x64, v64, c64 = geometry(data)
        dtype = jnp.float64 if mode == "oracle64" else jnp.float32
        x = tuple(jnp.asarray(v, dtype=dtype) for v in x64)
        directions = tuple(jnp.asarray(v, dtype=dtype) for v in v64)
        c, mu, sx = (
            jnp.asarray(v, dtype=dtype)
            for v in (c64, data["norm_state_mean"], data["norm_state_scale"])
        )
        values = {
            **{f"x{i}": np.asarray(v) for i, v in enumerate(x)},
            **{f"v{i}": np.asarray(v) for i, v in enumerate(directions)},
            "c": np.asarray(c),
            "mu": np.asarray(mu),
            "sx": np.asarray(sx),
        }
        saved32 = None
        if mode == "oracle64":
            r32 = read(Path(request["public32_output"]) / "result.json")
            require(
                [r["id"] for r in r32["cases"]] == [q["id"] for q in manifest["cases"]],
                "complete32 roster",
            )
            row32 = r32["cases"][index]
            path32 = Path(request["public32_output"]) / row32["path"]
            require(sha(path32) == row32["sha256"], "achieved32 endpoint evidence")
            saved32 = arrays(path32)

        def objective(*given, c=c, predict=predict, mu=mu, sx=sx):
            return jnp.sum(c * (predict(*given) - mu) / sx)

        for i in range(3):
            tangents = tuple(
                directions[i] if j == i else jnp.zeros_like(a) for j, a in enumerate(x)
            )
            if mode == "oracle64":
                values[f"ad{i}"] = np.asarray(jax.jvp(objective, x, tangents)[1])
            for k, h in enumerate(STEPS):
                for sign, label in ((1, "plus"), (-1, "minus")):
                    key = f"d{i}_h{k}_{label}"
                    ideal = tuple(
                        a + h * v if sign == 1 else a - h * v
                        for a, v in zip(x, tangents, strict=True)
                    )
                    if mode == "public32":
                        variants = (("", ideal),)
                    else:
                        achieved = tuple(
                            jnp.asarray(saved32[key + "_input"], jnp.float64)
                            if j == i
                            else jnp.asarray(
                                np.asarray(a, np.float32).astype(np.float64)
                            )
                            for j, a in enumerate(x)
                        )
                        variants = (("_nominal", ideal), ("_achieved", achieved))
                    for suffix, supplied in variants:
                        y = predict(*supplied)
                        values[key + suffix + "_input"] = np.asarray(supplied[i])
                        values[key + suffix + "_prediction"] = np.asarray(y)
                        values[key + suffix + "_scalar"] = np.asarray(
                            jnp.sum(c * (y - mu) / sx)
                        )
                if mode == "public32":
                    key = f"d{i}_h{k}"
                    values[key + "_fd"] = np.asarray(
                        (
                            jnp.asarray(values[key + "_plus_scalar"])
                            - jnp.asarray(values[key + "_minus_scalar"])
                        )
                        / (2 * h)
                    )
        path = output / f"case-{index:03d}.npz"
        np.savez_compressed(path, **values)
        rows.append(
            {
                "id": case["id"],
                "path": path.name,
                "sha256": sha(path),
                "input_sha256": case["sha256"],
            }
        )
        n.write_json(
            output / "progress.json", {"completed_cases": len(rows), "fits": 0}
        )
        require(
            dict(os.environ) == env
            and bool(jax.config.x64_enabled) == (mode == "oracle64"),
            "caller state unchanged",
        )
        n._imported_sources(root, b[prefix + "_source_sha256"])
        jax.clear_caches()
    n.write_json(
        output / "result.json",
        {
            "mode": mode,
            "steps": list(STEPS),
            "cases": rows,
            "fits": 0,
            "runtime": runtime,
            "imported_sources": n._imported_sources(root, b[prefix + "_source_sha256"]),
        },
    )


def grid_keys(mode):
    keys = {*(f"x{i}" for i in range(3)), *(f"v{i}" for i in range(3)), "c", "mu", "sx"}
    if mode == "oracle64":
        keys.update(f"ad{i}" for i in range(3))
    for i in range(3):
        for k in range(4):
            if mode == "public32":
                keys.add(f"d{i}_h{k}_fd")
            for sign in ("plus", "minus"):
                for variant in (
                    ("",) if mode == "public32" else ("_nominal", "_achieved")
                ):
                    keys.update(
                        f"d{i}_h{k}_{sign}{variant}_{part}"
                        for part in ("input", "prediction", "scalar")
                    )
    return keys


def grid_rows(output, mode, manifest):
    report = read(Path(output) / mode / "result.json")
    require(
        report["mode"] == mode
        and report["fits"] == 0
        and report["steps"] == list(STEPS),
        "grid worker identity",
    )
    require(
        [r["id"] for r in report["cases"]] == [c["id"] for c in manifest["cases"]],
        "complete grid roster",
    )
    return report["cases"]


def reduce_arrays(request, output):
    """Only stored-array/NumPy reduction: no prediction, JAX import or fitting."""
    _, n, _, manifest = packet(request, reconstruct=True)
    output = Path(output)
    numeric = output / "numerical"
    n.verify_saved(
        request["manifest"],
        numeric,
        expected_manifest_sha256=request["manifest_sha256"],
        expected_run_sha256=sha(numeric / "run.json"),
    )
    ordinary = n.reduce_saved(request["manifest"], numeric)
    require(
        "results" in ordinary and len(ordinary["results"]) == 41,
        "complete ordinary arithmetic evidence",
    )
    indexes = {mode: grid_rows(output, mode, manifest) for mode in MODES}
    numeric_rows = {
        mode: read(numeric / mode / "result.json")["cases"] for mode in MODES
    }
    historical_rows = {
        r["id"]: r
        for r in read(Path(request["original_run"]) / "result.json")["results"]
    }
    results = []
    for index, (case, normal) in enumerate(
        zip(manifest["cases"], ordinary["results"], strict=True)
    ):
        require(normal["id"] == case["id"], "ordinary case identity")
        data = arrays(Path(request["manifest"]).parent / case["path"])
        x, v, c = geometry(data)
        mu, sx = data["norm_state_mean"], data["norm_state_scale"]
        saved = {}
        for mode in MODES:
            row = indexes[mode][index]
            path = output / mode / row["path"]
            require(
                row["input_sha256"] == case["sha256"] and sha(path) == row["sha256"],
                "grid evidence bytes",
            )
            saved[mode] = arrays(path)
            require(set(saved[mode]) == grid_keys(mode), "complete grid array roster")
            dtype = np.dtype("float32" if mode == "public32" else "float64")
            require(all(a.dtype == dtype for a in saved[mode].values()), "grid dtypes")
            for j in range(3):
                require(
                    same(saved[mode][f"x{j}"], x[j].astype(dtype))
                    and same(saved[mode][f"v{j}"], v[j].astype(dtype)),
                    "input/direction witness",
                )
            for key, expected in (("c", c), ("mu", mu), ("sx", sx)):
                require(
                    same(saved[mode][key], expected.astype(dtype)),
                    "scalar definition witness",
                )
        a, b = saved["public32"], saved["oracle64"]
        regular = {
            mode: arrays(numeric / mode / numeric_rows[mode][index]["path"])
            for mode in MODES
        }
        raw_fd = {
            name: row
            for name, row in normal["checks"].items()
            if "/finite_difference/" in name
        }
        non_fd = {
            name: row
            for name, row in normal["checks"].items()
            if "/finite_difference/" not in name
            and (case["stress"] is None or name.endswith(("/causality", "/dtype")))
        }
        required = case["stress"] is None
        grid, convergence = [], []
        for i in range(3):
            require(
                same(a[f"v{i}"], regular["public32"][f"actual__direction_{i}"])
                and same(b[f"v{i}"], regular["oracle64"][f"original__direction_{i}"]),
                "ordinary direction linkage",
            )
            require(
                same(a[f"d{i}_h0_fd"], regular["public32"][f"actual__fd_{i}"]),
                "unchanged raw h32 quotient",
            )
            derivative = float(b[f"ad{i}"])
            contracted = float(
                np.sum(c * regular["oracle64"][f"original__jvp_{i}"] / sx)
            )
            ad_link = n.element_check(
                derivative, contracted, n.TOLERANCES["derivative64"]
            )
            quotients = []
            for k, h in enumerate(STEPS):
                endpoint_checks, response_predictions, response_references, bounds = (
                    [],
                    [],
                    [],
                    [],
                )
                scalars = []
                resolved = True
                for sign, label in ((1, "plus"), (-1, "minus")):
                    key = f"d{i}_h{k}_{label}"
                    expected32 = (
                        x[i].astype(np.float32)
                        + np.float32(h) * v[i].astype(np.float32)
                        if sign == 1
                        else x[i].astype(np.float32)
                        - np.float32(h) * v[i].astype(np.float32)
                    )
                    nominal = x[i] + h * v[i] if sign == 1 else x[i] - h * v[i]
                    require(
                        same(a[key + "_input"], expected32)
                        and same(b[key + "_nominal_input"], nominal)
                        and same(
                            b[key + "_achieved_input"], expected32.astype(np.float64)
                        ),
                        "exact nominal/achieved coordinates",
                    )
                    resolved &= bool(np.any(expected32 != x[i].astype(np.float32)))
                    py = a[key + "_prediction"].astype(np.float64)
                    oy = b[key + "_achieved_prediction"]
                    nominal_y = b[key + "_nominal_prediction"]
                    require(
                        py.shape
                        == oy.shape
                        == nominal_y.shape
                        == (case["horizon"], len(sx)),
                        "forecast shape",
                    )
                    endpoint_checks.append(
                        n.element_check(
                            (py - mu) / sx, (oy - mu) / sx, n.TOLERANCES["forecast32"]
                        )
                    )
                    for label64, forecast in (("nominal", nominal_y), ("achieved", oy)):
                        scalar = b[key + "_" + label64 + "_scalar"]
                        require(scalar.shape == (), "scalar shape")
                        check = n.element_check(
                            scalar,
                            np.sum(c * (forecast - mu) / sx),
                            n.TOLERANCES["forecast64"],
                        )
                        if required:
                            require(
                                check["passed"], "saved64 scalar/forecast reduction"
                            )
                    scalars.append(float(b[key + "_nominal_scalar"]))
                    response_predictions.append(float(np.sum(c * (py - mu) / sx)))
                    response_references.append(float(np.sum(c * (oy - mu) / sx)))
                    bounds.append(1e-5 + 1e-4 * np.abs((oy - mu) / sx))
                q = (scalars[0] - scalars[1]) / (2 * h)
                quotients.append(q)
                response = (response_predictions[0] - response_predictions[1]) / (2 * h)
                reference = (response_references[0] - response_references[1]) / (2 * h)
                allowance = float(np.sum(np.abs(c) * (bounds[0] + bounds[1])) / (2 * h))
                finite = bool(
                    np.isfinite([q, response, reference, allowance, derivative]).all()
                )
                passed = (
                    finite
                    and resolved
                    and all(check["passed"] for check in endpoint_checks)
                    and abs(response - reference) <= allowance
                )
                grid.append(
                    {
                        "direction": INPUTS[i],
                        "step": h,
                        "required": required,
                        "passed": bool(passed),
                        "nominal64_quotient": q,
                        "achieved32_scalar_response": response,
                        "achieved_endpoint_oracle_response": reference,
                        "propagated_endpoint_allowance": allowance,
                        "endpoint_checks": endpoint_checks,
                        "resolved": resolved,
                        "raw_float32_quotient": float(a[f"d{i}_h{k}_fd"]),
                    }
                )
            richardson = [(4 * quotients[j + 1] - quotients[j]) / 3 for j in range(3)]
            allowance = 2e-8 + 2e-5 * abs(derivative)
            ad_error = abs(richardson[2] - derivative)
            adjacent_error = abs(richardson[2] - richardson[1])
            passed = bool(
                np.isfinite([*quotients, *richardson, derivative]).all()
                and ad_link["passed"]
                and ad_error <= allowance
                and adjacent_error <= allowance
            )
            convergence.append(
                {
                    "direction": INPUTS[i],
                    "required": required,
                    "passed": passed,
                    "oracle_AD": derivative,
                    "oracle_AD_link": ad_link,
                    "central_quotients": quotients,
                    "richardson_estimates": richardson,
                    "selected_estimate_index": 2,
                    "AD_absolute_error": ad_error,
                    "finest_adjacent_absolute_difference": adjacent_error,
                    "allowance": allowance,
                }
            )
        passed = all(row["passed"] for row in non_fd.values()) and (
            not required or all(row["passed"] for row in [*grid, *convergence])
        )
        results.append(
            {
                "id": case["id"],
                "role": "regression" if index < 33 else "confirmation",
                "stress": case["stress"],
                "passed": bool(passed),
                "non_FD_checks": non_fd,
                "fresh_raw_FD_flags": raw_fd,
                "historical_original_FD_flags": {
                    name: row
                    for name, row in historical_rows.get(case["id"], {})
                    .get("checks", {})
                    .items()
                    if "/finite_difference/" in name
                },
                "grid": grid,
                "convergence": convergence,
            }
        )
    return {
        "format": "public-mean-fd-assessment-v1",
        "passed": all(r["passed"] for r in results),
        "regression_passed": all(r["passed"] for r in results[:33]),
        "confirmation_passed": all(r["passed"] for r in results[33:]),
        "regression_cases": 33,
        "confirmation_cases": 8,
        "mandatory_cases": 37,
        "diagnostic_stress_cases": 4,
        "grid_cells": 492,
        "steps": list(STEPS),
        "results": results,
        "original_v1_qualification_passed": False,
        "ordinary_41case_raw_gate_passed": ordinary["passed"],
        "assessment_protocol_sha256": request["assessment_protocol_sha256"],
        "inherited_arithmetic_protocol_sha256": manifest["provenance"][
            "protocol_sha256"
        ],
        "physical_derivative_fidelity_qualified": False,
        "public_promotion": False,
        "fits": 0,
    }


def run(request, output):
    b, n, _, _ = packet(request, reconstruct=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    outcome = {
        "status": "failed",
        "request": request,
        "source_sha256": sha(__file__),
        "fits": 0,
        "stages": [],
    }
    try:
        # Full ordinary suite is retained; its old aggregate FD gate is NOT the new verdict.
        normal = n.run(
            request["manifest"],
            output / "numerical",
            expected_manifest_sha256=request["manifest_sha256"],
            public_root=b["public_root"],
            oracle_root=b["oracle_root"],
            python=b["interpreter"],
        )
        require(
            all(s["returncode"] == 0 for s in read(output / "numerical/stages.json")),
            "ordinary worker execution failure",
        )
        outcome["ordinary_result"] = normal
        for mode in MODES:
            task = {**request, "public32_output": str(output / "public32")}
            path = output / (mode + "-request.json")
            n.write_json(path, task)
            env = dict(
                os.environ,
                SCIPY_ARRAY_API="1",
                JAX_PLATFORMS="cpu",
                PYTHONPATH=str(
                    Path(b["public_root" if mode == "public32" else "oracle_root"])
                    / "src"
                ),
            )
            env.pop("JAX_ENABLE_X64", None)
            if mode == "oracle64":
                env["JAX_ENABLE_X64"] = "1"
            command = [
                b["interpreter"],
                str(Path(__file__).resolve()),
                "worker",
                "--worker-mode",
                mode,
                "--request",
                str(path),
                "--expected-request-sha256",
                sha(path),
                "--output",
                str(output / mode),
            ]
            n.write_json(
                output / (mode + "-command.json"),
                {
                    "argv": command,
                    "cwd": b["public_root"],
                    "environment": {
                        k: env.get(k)
                        for k in (
                            "PYTHONPATH",
                            "SCIPY_ARRAY_API",
                            "JAX_ENABLE_X64",
                            "JAX_PLATFORMS",
                        )
                    },
                    "timeout_s": 7200,
                },
            )
            start = time.monotonic()
            with (output / (mode + ".log")).open("x") as stream:
                try:
                    p = subprocess.run(
                        command,
                        cwd=b["public_root"],
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=7200,
                        check=False,
                    )
                    code, timeout = p.returncode, False
                except subprocess.TimeoutExpired:
                    code, timeout = None, True
            stage = {
                "mode": mode,
                "returncode": code,
                "timed_out": timeout,
                "elapsed_s": time.monotonic() - start,
                "log_sha256": sha(output / (mode + ".log")),
            }
            outcome["stages"].append(stage)
            n.write_json(output / (mode + "-exit.json"), stage)
            require(code == 0, "grid worker failed: no automatic retry")
        assessment = reduce_arrays(request, output)
        n.write_json(output / "assessment.json", assessment)
        packet(request, reconstruct=True)
        outcome.update(status="complete", assessment_passed=assessment["passed"])
    except BaseException as error:
        outcome.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        outcome["files"] = {
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        n.write_json(output / "run.json", outcome)
    return {
        "status": outcome["status"],
        "assessment_passed": outcome["assessment_passed"],
        "run_sha256": sha(output / "run.json"),
        "fits": 0,
    }


def verify_saved(request, output, expected_run_sha256):
    _, _n, _, _ = packet(request, reconstruct=True)
    output = Path(output)
    require(
        sha(output / "run.json") == expected_run_sha256,
        "external assessment run anchor",
    )
    seal = read(output / "run.json")
    require(
        seal["status"] == "complete"
        and seal["request"] == request
        and seal["source_sha256"] == sha(__file__),
        "assessment run identity",
    )
    found = {
        str(p.relative_to(output))
        for p in output.rglob("*")
        if p.is_file() and p != output / "run.json"
    }
    require(found == set(seal["files"]), "complete assessment inventory")
    for relative, expected in seal["files"].items():
        require(
            sha(under(output, relative)) == expected, "assessment payload integrity"
        )
    result = reduce_arrays(request, output)

    # Normalizes only explicit diagnostic nonfinite values, as the inherited writer does.
    def safe(v):
        if isinstance(v, dict):
            return {k: safe(x) for k, x in v.items()}
        if isinstance(v, list):
            return [safe(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    require(
        safe(result) == read(output / "assessment.json"), "fresh saved-array reduction"
    )
    return result


def replay(request, previous, output, expected_run_sha256):
    verify_saved(request, previous, expected_run_sha256)
    result = run(request, output)
    count = 0
    # The inherited numeric replay exits early on its old FD verdict; compare explicitly.
    for relative in (
        "numerical/public32",
        "numerical/public64",
        "numerical/oracle64",
        "public32",
        "oracle64",
    ):
        old, new = Path(previous) / relative, Path(output) / relative
        left, right = (
            read(old / "result.json")["cases"],
            read(new / "result.json")["cases"],
        )
        require([r["id"] for r in left] == [r["id"] for r in right], "replay roster")
        for a, b in zip(left, right, strict=True):
            x, y = arrays(old / a["path"]), arrays(new / b["path"])
            require(
                set(x) == set(y) and all(same(x[k], y[k]) for k in x),
                "exact replayed arrays",
            )
            count += len(x)
    require(
        read(Path(previous) / "assessment.json")
        == read(Path(output) / "assessment.json"),
        "identical assessment verdict/reductions",
    )
    return {**result, "exact_replayed_arrays": count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "verify", "replay", "worker"))
    p.add_argument("--request", required=True)
    p.add_argument("--expected-request-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--worker-mode", choices=MODES)
    p.add_argument("--expected-run-sha256")
    p.add_argument("--previous")
    args = p.parse_args()
    require(
        sha(args.request) == args.expected_request_sha256, "external request anchor"
    )
    request = read(args.request)
    if args.action == "prepare":
        result = prepare(request, args.output)
    elif args.action == "run":
        result = run(request, args.output)
    elif args.action == "verify":
        result = verify_saved(request, args.output, args.expected_run_sha256)
    elif args.action == "replay":
        result = replay(request, args.previous, args.output, args.expected_run_sha256)
    else:
        worker(request, args.worker_mode, args.output)
        result = {"worker": args.worker_mode, "status": "complete", "fits": 0}
    print(
        json.dumps({k: v for k, v in result.items() if k != "results"}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
