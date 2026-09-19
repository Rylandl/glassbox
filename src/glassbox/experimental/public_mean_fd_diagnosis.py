"""Prospective fixed-step diagnostic only; never replaces qualification gates.

Standalone so an old-source worker need not import new Glassbox source. Execute
only after this file/request are reviewed, committed and externally anchored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

STEPS = (2.0**-7, 2.0**-8, 2.0**-9, 2.0**-10)
INPUTS = ("past_states", "past_inputs", "future_inputs")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    require(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label
    )


def q32(value):
    return np.asarray(value, np.float32).astype(np.float64)


def inventory(root, entries):
    for relative, expected in entries.items():
        path = (Path(root) / relative).resolve()
        require(
            path.is_relative_to(Path(root).resolve()) and sha(path) == expected,
            "source/payload " + relative,
        )


def imported(root, expected):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == "glassbox" or name.startswith("glassbox."):
            path = Path(module.__file__).resolve()
            require(
                path.is_relative_to(Path(root) / "src/glassbox"),
                "escaped source " + name,
            )
            relative = str(path.relative_to(root))
            require(expected.get(relative) == sha(path), "unbound source " + relative)
            result[name] = {"relative_path": relative, "sha256": sha(path)}
    return result


def authenticate(request):
    require(
        sha(__file__) == request["diagnostic_source_sha256"], "diagnostic source SHA"
    )
    diagnostic_root = Path(request["diagnostic_root"]).resolve()
    require(
        Path(__file__).resolve()
        == diagnostic_root / request["diagnostic_relative_path"],
        "diagnostic source location",
    )
    require(
        subprocess.check_output(
            ["git", "-C", str(diagnostic_root), "rev-parse", "HEAD"], text=True
        ).strip()
        == request["diagnostic_commit"],
        "diagnostic committed revision",
    )
    require(
        not subprocess.check_output(
            [
                "git",
                "-C",
                str(diagnostic_root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
        ).strip(),
        "diagnostic clean checkout",
    )
    require(
        sha(request["binding"]) == request["binding_sha256"],
        "external original binding",
    )
    binding = read(request["binding"])
    require(
        sha(request["manifest"]) == request["manifest_sha256"],
        "external numeric inputs",
    )
    require(
        sha(Path(request["original"]) / "run.json") == request["run_sha256"],
        "external original run",
    )
    run = read(Path(request["original"]) / "run.json")
    require(
        run["input_manifest_sha256"] == request["manifest_sha256"], "run/input identity"
    )
    inventory(request["original"], run["files"])
    manifest = read(request["manifest"])
    input_root = Path(request["manifest"]).parent
    for case in manifest["cases"]:
        inventory(input_root, {case["path"]: case["sha256"]})
        if case["model_path"] is not None:
            inventory(input_root, {case["model_path"]: case["model_sha256"]})
    original_rows = read(Path(request["original"]) / "public32/result.json")["cases"]
    require(
        [row["id"] for row in original_rows]
        == [case["id"] for case in manifest["cases"]],
        "exact original numerical worker roster",
    )
    for key in (
        "implementation_commit",
        "protocol_sha256",
        "public_source_sha256",
        "oracle_source_sha256",
        "runtime",
    ):
        require(
            manifest["provenance"][key] == binding[key],
            "original input provenance " + key,
        )
    require(
        manifest["provenance"]["implementation_manifest_sha256"]
        == request["binding_sha256"],
        "input binding",
    )
    require(
        Path(sys.executable).resolve() == Path(binding["interpreter"]).resolve(),
        "interpreter path",
    )
    require(
        sha(Path(sys.executable).resolve()) == binding["interpreter_sha256"],
        "interpreter SHA",
    )
    for prefix, commit_key in (
        ("public", "implementation_commit"),
        ("oracle", "oracle_commit"),
    ):
        root = binding[prefix + "_root"]
        inventory(root, binding[prefix + "_source_sha256"])
        head = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
        clean = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=normal"],
            text=True,
        ).strip()
        require(
            head == binding[commit_key] and not clean,
            "original source revision/cleanliness",
        )
    cases = [case for case in manifest["cases"] if case["source"] == "public_archive"]
    require(
        len(cases) == 24 and len({c["id"] for c in cases}) == 24,
        "all24 fixed actual cases",
    )
    require(
        sum(c["source_identity"]["query"]["model_id"] == "cascade" for c in cases)
        == 12,
        "fixed simulator counts",
    )
    return binding, cases


def worker_rows(directory, mode, cases):
    """Require the whole ordered worker result before consuming any arrays."""
    directory = Path(directory)
    report = read(directory / "result.json")
    rows = report["cases"]
    require(
        report["mode"] == mode
        and report["steps"] == list(STEPS)
        and report["fits"] == report["updates"] == 0
        and report["qualification_gate_changed"] is False,
        "diagnostic worker identity/scope",
    )
    require(
        [row["id"] for row in rows] == [case["id"] for case in cases],
        "exact ordered24 diagnostic worker roster",
    )
    for index, (row, case) in enumerate(zip(rows, cases, strict=True)):
        require(
            row["path"] == f"case-{index:03d}.npz"
            and row["original_case_sha256"] == case["sha256"],
            "diagnostic worker payload identity",
        )
        inventory(directory, {row["path"]: row["sha256"]})
    return {row["id"]: row for row in rows}


def worker(request, mode, output):
    binding, cases = authenticate(request)
    require(mode in ("public32", "oracle64"), "worker mode")
    prefix = "public" if mode == "public32" else "oracle"
    root = Path(binding[prefix + "_root"]).resolve()
    source_map = binding[prefix + "_source_sha256"]
    import jax
    import jax.numpy as jnp

    x64 = mode == "oracle64"
    require(
        bool(jax.config.x64_enabled) == x64 and jax.default_backend() == "cpu",
        "worker precision/backend",
    )
    for name in ("jax", "jaxlib", "numpy", "scipy"):
        require(
            importlib.metadata.version(name) == binding["runtime"][name],
            "runtime " + name,
        )
    if x64:
        from glassbox.experimental.state_quadratic_model import (
            KIND,
            QuadraticSequenceModel,
        )
    else:
        from glassbox import LearnedDynamics
    imported(root, source_map)
    before = dict(os.environ)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old_rows = {
        row["id"]: row
        for row in read(Path(request["original"]) / "public32/result.json")["cases"]
    }
    new32 = (
        {} if not x64 else worker_rows(request["public32_output"], "public32", cases)
    )
    rows = []
    for index, case in enumerate(cases):
        source = Path(request["manifest"]).parent / case["path"]
        require(sha(source) == case["sha256"], "input payload")
        data = arrays(source)
        old = arrays(
            Path(request["original"]) / "public32" / old_rows[case["id"]]["path"]
        )
        # Both modes receive the SAME Q32 caller values, even in oracle64.
        x = tuple(
            jnp.asarray(q32(data[name][0]), dtype=jnp.float64 if x64 else jnp.float32)
            for name in INPUTS
        )
        directions = tuple(
            jnp.asarray(old["actual__direction_" + str(i)], dtype=x[0].dtype)
            for i in range(3)
        )
        h, d = data["future_inputs"].shape[-2], data["past_states"].shape[-1]
        coefficients = jnp.asarray(
            q32(np.sin(np.arange(1, h * d + 1)).reshape(h, d) / (h * d)),
            dtype=x[0].dtype,
        )
        mu = jnp.asarray(q32(data["norm_state_mean"]), dtype=x[0].dtype)
        sx = jnp.asarray(q32(data["norm_state_scale"]), dtype=x[0].dtype)
        if x64:
            model = QuadraticSequenceModel(
                KIND,
                case["dt_s"],
                case["history_steps"],
                {k[6:]: q32(v) for k, v in data.items() if k.startswith("param_")},
                {k[5:]: q32(v) for k, v in data.items() if k.startswith("norm_")},
                case["delay_steps"],
            )
            predict = model.rollout
            saved32 = arrays(
                Path(request["public32_output"]) / new32[case["id"]]["path"]
            )
            require(
                sha(Path(request["public32_output"]) / new32[case["id"]]["path"])
                == new32[case["id"]]["sha256"],
                "new32 endpoint evidence",
            )
        else:
            model_path = Path(request["manifest"]).parent / case["model_path"]
            require(sha(model_path) == case["model_sha256"], "original saved model")
            model = LearnedDynamics.load(model_path)
            require(
                model.fingerprint() == case["model_fingerprint"],
                "original model fingerprint",
            )
            predict = model.predict

        def objective(
            *values, coefficients=coefficients, predict=predict, mu=mu, sx=sx
        ):
            return jnp.sum(coefficients * (predict(*values) - mu) / sx)

        values = {f"input_{i}": np.asarray(value) for i, value in enumerate(x)}
        for i in range(3):
            values["direction_" + str(i)] = np.asarray(directions[i])
            tangents = tuple(
                directions[i] if j == i else jnp.zeros_like(value)
                for j, value in enumerate(x)
            )
            if x64:
                values["ad_" + str(i)] = np.asarray(jax.jvp(objective, x, tangents)[1])
            for k, step in enumerate(STEPS):
                for sign, label in ((1, "plus"), (-1, "minus")):
                    key = f"d{i}_h{k}_{label}"
                    endpoint = tuple(
                        value + step * direction
                        if sign == 1
                        else value - step * direction
                        for value, direction in zip(x, tangents, strict=True)
                    )
                    if not x64:
                        values[key + "_input"] = np.asarray(endpoint[i])
                        prediction = predict(*endpoint)
                        values[key + "_prediction"] = np.asarray(prediction)
                        values[key + "_scalar"] = np.asarray(
                            jnp.sum(coefficients * (prediction - mu) / sx)
                        )
                    else:
                        for variant, supplied in (
                            ("ideal", endpoint),
                            (
                                "achieved",
                                tuple(
                                    jnp.asarray(saved32[key + "_input"], jnp.float64)
                                    if j == i
                                    else value
                                    for j, value in enumerate(x)
                                ),
                            ),
                        ):
                            prediction = predict(*supplied)
                            values[key + "_" + variant + "_prediction"] = np.asarray(
                                prediction
                            )
                            values[key + "_" + variant + "_scalar"] = np.asarray(
                                jnp.sum(coefficients * (prediction - mu) / sx)
                            )
                if not x64:
                    values[f"d{i}_h{k}_fd"] = np.asarray(
                        (
                            jnp.asarray(values[f"d{i}_h{k}_plus_scalar"])
                            - jnp.asarray(values[f"d{i}_h{k}_minus_scalar"])
                        )
                        / (2 * step)
                    )
                    if k == 0:
                        same(
                            values[f"d{i}_h{k}_fd"],
                            old["actual__fd_" + str(i)],
                            "unchanged frozen h32 result",
                        )
        require(
            all(np.isfinite(v).all() for v in values.values()),
            "nonfinite diagnostic value",
        )
        path = output / f"case-{index:03d}.npz"
        np.savez_compressed(path, **values)
        rows.append(
            {
                "id": case["id"],
                "path": path.name,
                "sha256": sha(path),
                "original_case_sha256": case["sha256"],
            }
        )
        require(
            dict(os.environ) == before and bool(jax.config.x64_enabled) == x64,
            "changed caller environment",
        )
        imported(root, source_map)
        jax.clear_caches()
    write(
        output / "result.json",
        {
            "mode": mode,
            "cases": rows,
            "imports": imported(root, source_map),
            "steps": list(STEPS),
            "fits": 0,
            "updates": 0,
            "qualification_gate_changed": False,
        },
    )


def reduce(request, output):
    output = Path(output)
    _, cases = authenticate(request)
    maps = {m: worker_rows(output / m, m, cases) for m in ("public32", "oracle64")}
    original_rows = {
        r["id"]: r
        for r in read(Path(request["original"]) / "public32/result.json")["cases"]
    }
    rows = []
    for case in cases:
        data = arrays(Path(request["manifest"]).parent / case["path"])
        a, b = (
            arrays(output / m / maps[m][case["id"]]["path"])
            for m in ("public32", "oracle64")
        )
        old = arrays(
            Path(request["original"]) / "public32" / original_rows[case["id"]]["path"]
        )
        h, d = data["future_inputs"].shape[-2], data["past_states"].shape[-1]
        c = np.sin(np.arange(1, h * d + 1)).reshape(h, d) / (h * d)
        qc = q32(c)
        mu = q32(data["norm_state_mean"])
        sx = q32(data["norm_state_scale"])
        for i in range(3):
            # The original gate used the dimensionless JVP contraction below.
            ad32 = float(
                np.sum(
                    c
                    * old["actual__jvp_" + str(i)].astype(np.float64)
                    / data["norm_state_scale"]
                )
            )
            ad64 = float(b["ad_" + str(i)])
            for k, step in enumerate(STEPS):
                prefix = f"d{i}_h{k}_"
                fd32 = float(a[prefix + "fd"])
                ideal = float(
                    (b[prefix + "plus_ideal_scalar"] - b[prefix + "minus_ideal_scalar"])
                    / (2 * step)
                )
                achieved = float(
                    (
                        b[prefix + "plus_achieved_scalar"]
                        - b[prefix + "minus_achieved_scalar"]
                    )
                    / (2 * step)
                )
                l32 = {s: float(a[prefix + s + "_scalar"]) for s in ("plus", "minus")}
                post64 = {
                    s: float(
                        np.sum(
                            qc
                            * (a[prefix + s + "_prediction"].astype(np.float64) - mu)
                            / sx
                        )
                    )
                    for s in ("plus", "minus")
                }
                forecast32_fd = (post64["plus"] - post64["minus"]) / (2 * step)
                scalar32_fd = (l32["plus"] - l32["minus"]) / (2 * step)
                terms = {
                    "finite_step_truncation": ideal - ad64,
                    "achieved_input_rounding": achieved - ideal,
                    "forecast_arithmetic_and_output": forecast32_fd - achieved,
                    "scalar_objective_arithmetic": scalar32_fd - forecast32_fd,
                    "final_difference_arithmetic": fd32 - scalar32_fd,
                    "ad_arithmetic_and_original_gate_contraction": ad64 - ad32,
                }
                require(
                    abs(sum(terms.values()) - (fd32 - ad32))
                    <= 1e-12 * (1 + abs(fd32 - ad32)),
                    "diagnostic telescope",
                )
                rows.append(
                    {
                        "id": case["id"],
                        "direction": INPUTS[i],
                        "step": step,
                        "original_gate_ad32": ad32,
                        "rounded_model_ad64": ad64,
                        "fd32": fd32,
                        "fd64_ideal": ideal,
                        "fd64_achieved": achieved,
                        "fd32_minus_original_gate_ad": fd32 - ad32,
                        "terms": terms,
                        "scalar32_plus": l32["plus"],
                        "scalar32_minus": l32["minus"],
                    }
                )
    return {
        "scope": "Prospective fixed-grid diagnosis, all24 cases/all3 directions/all4 steps; no chosen h, tolerance change, replacement pass flag or model qualification.",
        "rows": rows,
        "cases": 24,
        "directions_per_case": 3,
        "steps": list(STEPS),
        "cells": len(rows),
        "fits": 0,
        "qualification_gate_changed": False,
        "limitation": "Terms compare a float64 historical oracle with Q32 model/input/scalar constants to actual public32. The AD bridge includes constant quantization and the original gate's float64 contraction; no physical Jacobian correctness claim.",
    }


def run(request, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    outcome = {
        "status": "failed",
        "fits": 0,
        "request": request,
        "diagnostic_source_sha256": sha(__file__),
    }
    try:
        binding, _ = authenticate(request)
        # Supervisor validates the complete original binding using unchanged public code.
        from glassbox.experimental import public_mean_implementation

        require(
            Path(public_mean_implementation.__file__).resolve()
            == Path(binding["public_root"])
            / "src/glassbox/experimental/public_mean_implementation.py",
            "supervisor original source import",
        )
        public_mean_implementation.verify(request["binding"], request["binding_sha256"])
        stages = []
        outcome["stages"] = stages
        for mode in ("public32", "oracle64"):
            task = {**request, "public32_output": str(output / "public32")}
            path = output / (mode + "-request.json")
            write(path, task)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(
                Path(binding["public_root" if mode == "public32" else "oracle_root"])
                / "src"
            )
            env.pop("JAX_ENABLE_X64", None)
            if mode == "oracle64":
                env["JAX_ENABLE_X64"] = "1"
            command = [
                binding["interpreter"],
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--request",
                str(path),
                "--expected-request-sha256",
                sha(path),
                "--output",
                str(output / mode),
            ]
            write(
                output / (mode + "-command.json"),
                {
                    "argv": command,
                    "cwd": binding["public_root"],
                    "environment": {
                        k: env.get(k)
                        for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
                    },
                    "timeout_s": 7200,
                },
            )
            start = time.monotonic()
            with (output / (mode + ".log")).open("x") as stream:
                try:
                    result = subprocess.run(
                        command,
                        cwd=binding["public_root"],
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=7200,
                        check=False,
                    )
                    returncode, timed_out = result.returncode, False
                except subprocess.TimeoutExpired:
                    returncode, timed_out = None, True
            stage = {
                "mode": mode,
                "returncode": returncode,
                "timed_out": timed_out,
                "elapsed_s": time.monotonic() - start,
                "log_sha256": sha(output / (mode + ".log")),
                "persisted_prefix_files": {
                    str(p.relative_to(output / mode)): sha(p)
                    for p in sorted((output / mode).rglob("*"))
                    if p.is_file()
                },
                "complete_worker_manifest_present": (
                    output / mode / "result.json"
                ).is_file(),
                "unpersisted_work_unknown": returncode != 0,
            }
            stages.append(stage)
            write(output / (mode + "-exit.json"), stage)
            require(
                returncode == 0,
                "diagnostic worker failed or timed out, no automatic retry",
            )
        result = reduce(request, output)
        write(output / "diagnosis.json", result)
        public_mean_implementation.verify(request["binding"], request["binding_sha256"])
        outcome.update(
            status="complete",
            stages=stages,
            diagnosis_sha256=sha(output / "diagnosis.json"),
        )
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
        write(output / "run.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--expected-request-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", choices=("public32", "oracle64"))
    args = parser.parse_args()
    require(
        sha(args.request) == args.expected_request_sha256,
        "external diagnostic request SHA",
    )
    request = read(args.request)
    if args.worker:
        worker(request, args.worker, args.output)
    else:
        run(request, args.output)


if __name__ == "__main__":
    main()
