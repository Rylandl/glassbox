"""Frozen public-v4 arithmetic qualification, without fitting or simulation.

The worker entry point intentionally imports no Glassbox module until it has
authenticated its requested checkout. Historical and maintained implementations
never share a Python process. Artificial coefficients are arithmetic fixtures,
not fabricated fitted-model archives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

PROTOCOL_SHA256 = "d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99"
ORACLE_COMMIT = "8b61830c9c353dbb25edc6e63a76886d0ce9b9d3"
PARAMETERS = (
    "linear",
    "interaction",
    "autonomous",
    "bias",
    "w1",
    "b1",
    "w2",
    "memory",
    "memory_bias",
)
NORMS = (
    "state_mean",
    "state_scale",
    "input_mean",
    "input_scale",
    "feature_scale",
    "delta_scale",
    "interaction_scale",
    "autonomous_scale",
)
INPUTS = ("past_states", "past_inputs", "future_inputs")
TOLERANCES = {
    "forecast64": (1e-12, 1e-10),
    "paths32": (2e-6, 2e-5),
    "forecast32": (1e-5, 1e-4),
    "derivative64": (1e-11, 1e-9),
    "derivative32": (2e-6, 2e-3),
    "duality64": (1e-11, 1e-9),
    "duality32": (2e-6, 2e-4),
    "difference64": (2e-8, 2e-5),
    "difference32": (2e-5, 1e-2),
}
FIXTURES = (
    ("scalar-coarse", 1, 1, 0.25, 1, 2, 1),
    ("rectangular-input", 2, 3, 0.05, 2, 10, 5),
    ("rectangular-state", 5, 2, 0.02, 5, 25, 12),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {k: safe(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(v) for v in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    Path(path).write_text(
        json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_json(path):
    return json.loads(Path(path).read_text())


def _tree(data, prefix, names):
    return {name: np.array(data[prefix + name], copy=True) for name in names}


def _quantize(value):
    return np.asarray(value, dtype=np.float32).astype(np.float64)


def artificial_fixture(spec, *, stress=None):
    """Construct exactly the frozen deterministic arrays; no RNG or initializer."""
    name, d, m, dt, delay, context, horizon = spec
    f, q, a = (delay + 1) * (d + m) + 8, d * m, d * (d + 1) // 2

    def wave(shape, kind="sin"):
        k = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float64).reshape(shape)
        return getattr(np, kind)(k)

    params = {
        "linear": 0.01 * wave((f, d)) / np.sqrt(f),
        "interaction": 0.02 * wave((q, d)) / np.sqrt(q),
        "autonomous": 0.02 * wave((a, d), "cos") / np.sqrt(a),
        "bias": 0.01 * wave((d,), "cos"),
        "w1": 0.2 * wave((f, 32)) / np.sqrt(f),
        "b1": 0.05 * wave((32,)),
        "w2": 0.03 * wave((32, d), "cos") / np.sqrt(32),
        "memory": 0.2 * wave((f, 8), "cos") / np.sqrt(f),
        "memory_bias": 0.02 * wave((8,)),
    }
    params["linear"][:d] -= 0.04 * np.eye(d)
    params["linear"][d : d + m] += (
        0.08 * np.sin(np.arange(m)[:, None] + np.arange(d)[None, :] + 1) / m
    )
    i, j = np.arange(d), np.arange(m)
    norms = {
        "state_mean": (-1.0) ** i * 2 * (i + 1),
        "state_scale": 2.0 ** (i % 3 - 1),
        "input_mean": (-1.0) ** j * 0.2,
        "input_scale": 2.0 ** (j % 3 - 2),
        "feature_scale": np.ones(f),
        "delta_scale": np.full(d, 0.1),
        "interaction_scale": np.ones(q),
        "autonomous_scale": np.ones(a),
    }
    if stress == "offset-2p30":
        norms["state_mean"] += 2.0**30
    elif stress == "small-scale-offset":
        norms["state_mean"] = (-1.0) ** i * 2.0**16
        norms["state_scale"] *= 2.0**-10
    elif stress is not None:
        raise ValueError("unknown prospectively declared stress")
    b = np.arange(3)[:, None, None]
    t = np.arange(-context, 1)[None, :, None]
    k = (i + 1)[None, None, :]
    zx = 0.15 * np.sin(0.37 * t + 0.29 * k + 0.11 * b)
    zx += 0.05 * np.cos(0.13 * t * k + 0.17 * b)

    def commands(times):
        t = np.asarray(times)[None, :, None]
        k = (j + 1)[None, None, :]
        z = 0.20 * np.cos(0.23 * t + 0.31 * k + 0.07 * b)
        z += 0.04 * np.sin(0.11 * t * k)
        return norms["input_mean"] + norms["input_scale"] * z

    arrays = {
        **{"param_" + k: v for k, v in params.items()},
        **{"norm_" + k: v for k, v in norms.items()},
        "past_states": norms["state_mean"] + norms["state_scale"] * zx,
        "past_inputs": commands(np.arange(-context, 0)),
        "future_inputs": commands(np.arange(horizon)),
    }
    metadata = dict(
        fixture=name,
        dt_s=dt,
        history_steps=context,
        delay_steps=delay,
        maximum_horizon=horizon,
        stress=stress,
        source="artificial",
    )
    return metadata, arrays


def _check_arrays(meta, arrays):
    expected = {
        *INPUTS,
        *("param_" + k for k in PARAMETERS),
        *("norm_" + k for k in NORMS),
    }
    if set(arrays) != expected:
        raise ValueError("numeric packet array roster differs")
    if any(
        a.dtype != np.dtype("float64") or not np.isfinite(a).all()
        for a in arrays.values()
    ):
        raise ValueError("numeric packet requires finite float64 source arrays")
    x, up, uf = (arrays[k] for k in INPUTS)
    d, m, c, h = (
        len(arrays["norm_state_mean"]),
        len(arrays["norm_input_mean"]),
        meta["history_steps"],
        meta["horizon"],
    )
    if (
        x.ndim != 3
        or x.shape != (len(x), c + 1, d)
        or up.shape != (len(x), c, m)
        or uf.shape != (len(x), h, m)
    ):
        raise ValueError("numeric packet histories/horizons are inconsistent")
    if not 1 <= h <= meta["maximum_horizon"] or not 1 <= meta["delay_steps"] < c:
        raise ValueError("numeric packet timing is inconsistent")
    if meta["source"] == "public_archive" and (
        len(x) != 1 or h != meta["maximum_horizon"]
    ):
        raise ValueError("actual numerical cases are single maximum-horizon inputs")


def prepare(directory, *, provenance, archive_cases=()):
    """Write frozen artificial cases and optional caller-supplied actual inputs.

    Each archive case supplies id, model_path, the three INPUTS arrays, and a
    source dict containing its externally authenticated query/branch identities.
    The caller selects the exact 24-case lifecycle roster before any inference.
    """
    if provenance["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("wrong numerical qualification protocol")
    archive_cases = tuple(archive_cases)
    if archive_cases and len(archive_cases) != 24:
        raise ValueError("the actual lifecycle roster requires exactly 24 mean inputs")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases, copies = [], {}

    def add(meta, arrays):
        _check_arrays(meta, arrays)
        path = directory / f"case-{len(cases):03d}.npz"
        np.savez_compressed(path, **arrays)
        cases.append({**meta, "path": path.name, "sha256": digest(path)})

    for spec in FIXTURES:
        for stress in (
            (None, "offset-2p30", "small-scale-offset")
            if spec[0] == "rectangular-input"
            else (None,)
        ):
            base, arrays = artificial_fixture(spec, stress=stress)
            for h in sorted({1, base["maximum_horizon"]}):
                add(
                    {
                        **base,
                        "id": f"{spec[0]}/{stress or 'nominal'}/h{h}",
                        "horizon": h,
                        "model_path": None,
                    },
                    {**arrays, "future_inputs": arrays["future_inputs"][:, :h].copy()},
                )
    for case in archive_cases:
        source = Path(case["model_path"])
        sha = digest(source)
        if sha not in copies:
            name = f"model-{sha}.npz"
            shutil.copyfile(source, directory / name)
            copies[sha] = name
        with np.load(source, allow_pickle=False) as archive:
            model_meta = json.loads(str(archive["metadata"]))
            inner = model_meta["model"]
            arrays = {
                **{
                    k: np.array(archive[k], copy=True)
                    for k in archive.files
                    if k.startswith(("param_", "norm_"))
                },
                **{k: np.array(case[k], copy=True) for k in INPUTS},
            }
            maximum = archive["train_future_states"].shape[1]
        for key in INPUTS:
            if arrays[key].ndim == 2:
                arrays[key] = arrays[key][None]
        add(
            dict(
                id=case["id"],
                fixture=None,
                stress=None,
                source="public_archive",
                source_identity=case["source"],
                dt_s=inner["dt_s"],
                history_steps=inner["history_steps"],
                delay_steps=inner["delay_steps"],
                horizon=arrays["future_inputs"].shape[1],
                maximum_horizon=maximum,
                model_path=copies[sha],
                model_sha256=sha,
                model_fingerprint=model_meta["fingerprint"],
            ),
            arrays,
        )
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate numerical case identity")
    manifest = dict(format="public-v4-numerics-v1", provenance=provenance, cases=cases)
    write_json(directory / "manifest.json", manifest)
    return {
        "manifest": str(directory / "manifest.json"),
        "manifest_sha256": digest(directory / "manifest.json"),
        "cases": len(cases),
    }


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _sources(root, expected):
    root = Path(root).resolve()
    if not expected:
        raise ValueError("source inventory is required")
    for relative, sha in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != sha:
            raise ValueError(f"source binding differs: {relative}")


def _imported_sources(root, expected):
    root = Path(root).resolve()
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if name == "glassbox" or name.startswith("glassbox."):
            path = getattr(module, "__file__", None)
            if path is None:
                continue
            path = Path(path).resolve()
            if not path.is_relative_to(root / "src/glassbox"):
                raise ValueError(f"Glassbox module escaped its checkout: {name}")
            rel = str(path.relative_to(root))
            if rel not in expected or digest(path) != expected[rel]:
                raise ValueError(f"unbound imported source: {name}")
            loaded[name] = {"relative_path": rel, "sha256": expected[rel]}
    return loaded


def _runtime():
    import jax

    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        **{
            k: importlib.metadata.version(k)
            for k in ("jax", "jaxlib", "numpy", "scipy")
        },
        "x64": bool(jax.config.x64_enabled),
        "backend": jax.default_backend(),
    }


def _capture(predict, arrays, *, x64):
    """Capture the prospectively declared transformations for one fixed query."""
    import jax
    import jax.numpy as jnp

    dtype = np.float64 if x64 else np.float32
    inputs = tuple(jnp.asarray(arrays[k], dtype=dtype) for k in INPUTS)
    mu = jnp.asarray(arrays["norm_state_mean"], dtype=dtype)
    sx = jnp.asarray(arrays["norm_state_scale"], dtype=dtype)
    compiled = jax.jit(predict)
    result = {
        "eager": np.asarray(predict(*inputs)),
        "jit": np.asarray(compiled(*inputs)),
        "vmap": np.asarray(jax.vmap(predict)(*inputs)),
        "singles": np.stack(
            [
                np.asarray(predict(*(x[b] for x in inputs)))
                for b in range(len(inputs[0]))
            ]
        ),
        "jit_singles": np.stack(
            [
                np.asarray(compiled(*(x[b] for x in inputs)))
                for b in range(len(inputs[0]))
            ]
        ),
    }
    single = tuple(x[0] for x in inputs)
    h, d = inputs[-1].shape[-2], inputs[0].shape[-1]
    coefficients = jnp.asarray(
        np.sin(np.arange(1, h * d + 1)).reshape(h, d) / (h * d), dtype=dtype
    )

    def objective(*xs):
        return jnp.sum(coefficients * (predict(*xs) - mu) / sx)

    def compiled_objective(*xs):
        return jnp.sum(coefficients * (compiled(*xs) - mu) / sx)

    gradients = {
        "grad": jax.grad(objective, argnums=(0, 1, 2))(*single),
        "jit_grad": jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(*single),
        "grad_jit": jax.grad(compiled_objective, argnums=(0, 1, 2))(*single),
    }
    for method, values in gradients.items():
        for i, value in enumerate(values):
            result[f"{method}_{i}"] = np.asarray(value)
    for i, original in enumerate(single):
        v = np.sin(np.arange(1, original.size + 1, dtype=np.float64)).reshape(
            original.shape
        )
        v /= np.sqrt(np.sum(v**2))
        v *= arrays["norm_state_scale" if i == 0 else "norm_input_scale"]
        directions = tuple(
            jnp.asarray(v, dtype=dtype) if j == i else jnp.zeros_like(x)
            for j, x in enumerate(single)
        )
        result[f"direction_{i}"] = np.asarray(directions[i])
        result[f"jvp_{i}"] = np.asarray(jax.jvp(predict, single, directions)[1])
        result[f"jit_jvp_{i}"] = np.asarray(
            jax.jit(lambda *xs, tangents=directions: jax.jvp(predict, xs, tangents)[1])(
                *single
            )
        )
        epsilon = 2.0 ** (-16 if x64 else -7)
        plus = tuple(x + epsilon * v for x, v in zip(single, directions, strict=True))
        minus = tuple(x - epsilon * v for x, v in zip(single, directions, strict=True))
        result[f"fd_{i}"] = np.asarray(
            (objective(*plus) - objective(*minus)) / (2 * epsilon)
        )
        result[f"achieved_plus_{i}"] = np.asarray(plus[i] - single[i])
        result[f"achieved_minus_{i}"] = np.asarray(single[i] - minus[i])
    last = (
        jnp.zeros_like(single[-1])
        .at[-1]
        .set(jnp.asarray(arrays["norm_input_scale"], dtype=dtype))
    )
    result["causal_perturbed"] = np.asarray(
        predict(single[0], single[1], single[2] + last)
    )
    result["causal_jvp"] = np.asarray(
        jax.jvp(
            predict,
            single,
            (jnp.zeros_like(single[0]), jnp.zeros_like(single[1]), last),
        )[1]
    )
    first = jnp.zeros_like(single[-1]).at[0, 0].set(1)
    result["first_command_jvp"] = np.asarray(
        jax.jvp(
            predict,
            single,
            (jnp.zeros_like(single[0]), jnp.zeros_like(single[1]), first),
        )[1]
    )
    return result


def _worker(request_path, output):
    """Run only in a fresh interpreter with one authenticated source root."""
    request = read_json(request_path)
    root, mode = Path(request["root"]).resolve(), request["mode"]
    if mode not in ("public32", "public64", "oracle64"):
        raise ValueError("unknown numerical worker mode")
    _sources(root, request["source_sha256"])
    runtime = _runtime()
    x64 = mode != "public32"
    if runtime["x64"] != x64 or runtime["backend"] != "cpu":
        raise ValueError("numerical worker precision/backend differs")
    for key, value in request["runtime"].items():
        if runtime.get(key) != value:
            raise ValueError(f"numerical worker runtime differs: {key}")
    env_before = dict(os.environ)
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
    _imported_sources(root, request["source_sha256"])
    packet = Path(request["manifest"])
    if digest(packet) != request["manifest_sha256"]:
        raise ValueError("numeric input manifest changed")
    manifest = read_json(packet)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    import jax

    for index, case in enumerate(manifest["cases"]):
        source = packet.parent / case["path"]
        if digest(source) != case["sha256"]:
            raise ValueError("numeric input arrays changed")
        with np.load(source, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        _check_arrays(case, arrays)
        params, norms = (
            _tree(arrays, "param_", PARAMETERS),
            _tree(arrays, "norm_", NORMS),
        )

        def explicit(parameters, normalizations, *, spec=case):
            return cls(
                KIND,
                spec["dt_s"],
                spec["history_steps"],
                parameters,
                normalizations,
                spec["delay_steps"],
            )

        model = explicit(params, norms)
        if mode != "oracle64" and case["source"] == "public_archive":
            archive_path = packet.parent / case["model_path"]
            if digest(archive_path) != case["model_sha256"]:
                raise ValueError("actual public archive changed")
            model = LearnedDynamics.load(archive_path)
            if model.fingerprint() != case["model_fingerprint"]:
                raise ValueError("actual public archive fingerprint differs")
            predict = model.predict
        else:
            predict = model.rollout
        values = {}
        variants = (
            ("original", "quantized_inputs") if mode == "oracle64" else ("actual",)
        )
        for variant in variants:
            supplied = dict(arrays)
            if variant == "quantized_inputs":
                supplied.update({k: _quantize(arrays[k]) for k in INPUTS})
            values.update(
                {
                    f"{variant}__{k}": v
                    for k, v in _capture(predict, supplied, x64=x64).items()
                }
            )
        if mode == "oracle64":
            rounded = explicit(
                {k: _quantize(v) for k, v in params.items()},
                {k: _quantize(v) for k, v in norms.items()},
            )
            values["rounded_parameters__eager"] = np.asarray(
                rounded.rollout(*(_quantize(arrays[k]) for k in INPUTS))
            )
        path = output / f"case-{index:03d}.npz"
        np.savez_compressed(path, **values)
        rows.append({"id": case["id"], "path": path.name, "sha256": digest(path)})
        if bool(jax.config.x64_enabled) != x64 or dict(os.environ) != env_before:
            raise ValueError("numerical execution changed caller precision/environment")
        _imported_sources(root, request["source_sha256"])
        jax.clear_caches()
    result = dict(
        mode=mode,
        runtime=runtime,
        cases=rows,
        imported_sources=_imported_sources(root, request["source_sha256"]),
        fits=0,
    )
    write_json(output / "result.json", result)
    return result


def element_check(actual, reference, tolerance):
    """Elementwise frozen bound, with maxima and worst component retained."""
    actual, reference = (
        np.asarray(actual, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    if actual.shape != reference.shape:
        return dict(
            passed=False,
            reason="shape",
            actual_shape=list(actual.shape),
            reference_shape=list(reference.shape),
        )
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return dict(passed=False, reason="nonfinite")
    atol, rtol = tolerance
    difference = np.abs(actual - reference)
    allowance = atol + rtol * np.abs(reference)
    ratio = difference / allowance
    flat = int(np.argmax(ratio))
    return dict(
        passed=bool(np.all(difference <= allowance)),
        atol=atol,
        rtol=rtol,
        maximum_absolute_error=float(difference.max()),
        maximum_allowance_ratio=float(ratio.max()),
        worst_index=[int(i) for i in np.unravel_index(flat, difference.shape)],
        elements=int(difference.size),
    )


def _variant(values, name):
    return {
        k.split("__", 1)[1]: v for k, v in values.items() if k.startswith(name + "__")
    }


def _capture_keys():
    return {
        "eager",
        "jit",
        "vmap",
        "singles",
        "jit_singles",
        "causal_perturbed",
        "causal_jvp",
        "first_command_jvp",
        *(
            f"{method}_{i}"
            for method in (
                "grad",
                "jit_grad",
                "grad_jit",
                "direction",
                "jvp",
                "jit_jvp",
                "fd",
                "achieved_plus",
                "achieved_minus",
            )
            for i in range(3)
        ),
    }


def _case_reduction(case, arrays, worker_arrays):
    for mode, values in worker_arrays.items():
        variants = (
            ("original", "quantized_inputs") if mode == "oracle64" else ("actual",)
        )
        expected = {
            f"{variant}__{key}" for variant in variants for key in _capture_keys()
        }
        if mode == "oracle64":
            expected.add("rounded_parameters__eager")
        if set(values) != expected:
            raise ValueError("saved transformation array roster differs")
    sx, mu = arrays["norm_state_scale"], arrays["norm_state_mean"]
    old = worker_arrays["oracle64"]
    o0, o1 = _variant(old, "original"), _variant(old, "quantized_inputs")
    p32, p64 = (_variant(worker_arrays[k], "actual") for k in ("public32", "public64"))
    checks = {}

    def check(name, value, reference, tolerance):
        checks[name] = element_check(value, reference, TOLERANCES[tolerance])

    for precision, actual, oracle in (("32", p32, o1), ("64", p64, o0)):
        checks[f"{precision}/finite_transforms"] = {
            "passed": all(np.isfinite(v).all() for v in actual.values())
        }
        for path in ("eager", "jit", "vmap", "singles", "jit_singles"):
            check(
                f"{precision}/oracle/{path}",
                (actual[path] - mu) / sx,
                (oracle[path] - mu) / sx,
                "forecast" + precision,
            )
            check(
                f"{precision}/paths/{path}",
                (actual[path] - mu) / sx,
                (actual["eager"] - mu) / sx,
                "paths32" if precision == "32" else "forecast64",
            )
        h, d = actual["eager"].shape[-2:]
        c = np.sin(np.arange(1, h * d + 1)).reshape(h, d) / (h * d)
        for i in range(3):
            scale = sx if i == 0 else arrays["norm_input_scale"]
            for method in ("grad", "jit_grad", "grad_jit"):
                check(
                    f"{precision}/{method}/{i}",
                    actual[f"{method}_{i}"] * scale,
                    oracle[f"{method}_{i}"] * scale,
                    "derivative" + precision,
                )
            for method in ("jvp", "jit_jvp"):
                check(
                    f"{precision}/{method}/{i}",
                    actual[f"{method}_{i}"] / sx,
                    oracle[f"{method}_{i}"] / sx,
                    "derivative" + precision,
                )
            tangent = float(np.sum(c * actual[f"jvp_{i}"] / sx))
            dual = float(np.sum(actual[f"grad_{i}"] * actual[f"direction_{i}"]))
            check(f"{precision}/duality/{i}", tangent, dual, "duality" + precision)
            check(
                f"{precision}/finite_difference/{i}",
                actual[f"fd_{i}"],
                tangent,
                "difference" + precision,
            )
            checks[f"{precision}/resolved_perturbation/{i}"] = {
                "passed": bool(
                    np.any(actual[f"achieved_plus_{i}"])
                    and np.any(actual[f"achieved_minus_{i}"])
                ),
                "maximum_plus": float(np.max(np.abs(actual[f"achieved_plus_{i}"]))),
                "maximum_minus": float(np.max(np.abs(actual[f"achieved_minus_{i}"]))),
            }
        checks[f"{precision}/causality"] = {
            "passed": bool(
                np.array_equal(
                    actual["causal_perturbed"][:-1], actual["singles"][0, :-1]
                )
                and np.count_nonzero(actual["causal_jvp"][:-1]) == 0
            )
        }
        checks[f"{precision}/dtype"] = {
            "passed": all(
                v.dtype == np.dtype("float" + precision) for v in actual.values()
            )
        }
        if case["source"] == "artificial" and case["stress"] is None:
            effect = float(np.max(np.abs(actual["first_command_jvp"] / sx)))
            checks[f"{precision}/active_command"] = {
                "passed": effect > 1e-4,
                "maximum_normalized_effect": effect,
            }
    y0, y1, y2, y3 = (
        o0["eager"],
        o1["eager"],
        old["rounded_parameters__eager"],
        p32["eager"],
    )
    decomposition = {
        "input": y1 - y0,
        "parameters": y2 - y1,
        "output": _quantize(y2) - y2,
        "arithmetic": y3 - _quantize(y2),
    }
    stress = case["stress"] is not None
    required = {
        name: row
        for name, row in checks.items()
        if not stress or name.endswith(("/causality", "/dtype"))
    }
    return dict(
        id=case["id"],
        source=case["source"],
        stress=case["stress"],
        passed=all(row["passed"] for row in required.values()),
        checks=checks,
        representation={
            "maximum_absolute_terms": {
                k: float(np.max(np.abs(v))) if np.isfinite(v).all() else None
                for k, v in decomposition.items()
            },
            "input_unique_values": {
                k: {
                    "original": len(np.unique(arrays[k])),
                    "quantized": len(np.unique(_quantize(arrays[k]))),
                }
                for k in INPUTS
            },
            "state_input_spacing_over_scale": (
                np.abs(np.spacing(arrays["past_states"].astype(np.float32))).astype(
                    np.float64
                )
                / sx
            )
            .max(axis=(0, 1))
            .tolist(),
            "state_output_spacing_over_scale": (
                np.abs(np.spacing(y3.astype(np.float32))).astype(np.float64) / sx
            )
            .max(axis=(0, 1))
            .tolist(),
            "finite": {
                k: bool(np.isfinite(v).all())
                for k, v in {"Y0": y0, "Y1": y1, "Y2": y2, "Y3": y3}.items()
            },
        },
        scope="Artificial representability stress is diagnostic; ordinary transformations and causality/dtype remain required."
        if stress
        else "Every numerical and structural check is required.",
    )


def reduce_saved(manifest_path, directory):
    """Reduce saved transforms only; no forecast, differentiation or fitting."""
    manifest_path, directory = Path(manifest_path), Path(directory)
    manifest = read_json(manifest_path)
    workers = {
        mode: read_json(directory / mode / "result.json")
        for mode in ("public32", "public64", "oracle64")
    }
    expected = [c["id"] for c in manifest["cases"]]
    if any([c["id"] for c in row["cases"]] != expected for row in workers.values()):
        raise ValueError("numerical worker case roster differs")
    rows = []
    for i, case in enumerate(manifest["cases"]):
        with np.load(
            manifest_path.parent / case["path"], allow_pickle=False
        ) as archive:
            arrays = {k: archive[k] for k in archive.files}
        saved = {}
        for mode, result in workers.items():
            item = result["cases"][i]
            path = directory / mode / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError("saved numerical transform changed")
            with np.load(path, allow_pickle=False) as archive:
                saved[mode] = {k: archive[k] for k in archive.files}
        rows.append(_case_reduction(case, arrays, saved))
    return dict(
        passed=all(r["passed"] for r in rows),
        cases=len(rows),
        results=rows,
        fits=0,
        nominal_artificial_cases=sum(
            c["source"] == "artificial" and c["stress"] is None
            for c in manifest["cases"]
        ),
        diagnostic_stress_cases=sum(c["stress"] is not None for c in manifest["cases"]),
        actual_mean_inputs=sum(
            c["source"] == "public_archive" for c in manifest["cases"]
        ),
    )


def run(
    manifest_path,
    output,
    *,
    expected_manifest_sha256,
    public_root,
    oracle_root,
    python=sys.executable,
    timeout_s=14400,
):
    """Execute isolated workers after the caller binds a clean implementation.

    provenance requires protocol_sha256, implementation_commit, runtime,
    public_source_sha256 and oracle_source_sha256. All are authenticated by the
    externally supplied input-manifest hash; no source pin is learned in a worker.
    """
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if digest(manifest_path) != expected_manifest_sha256:
        raise ValueError("external numerical input anchor differs")
    manifest = read_json(manifest_path)
    p = manifest["provenance"]
    if p["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("wrong numerical qualification protocol")
    public_root, oracle_root = Path(public_root).resolve(), Path(oracle_root).resolve()
    own_relative = str(Path(__file__).resolve().relative_to(public_root))
    if p["public_source_sha256"].get(own_relative) != digest(__file__):
        raise ValueError("numerical runner itself is not source-bound")
    for root, commit, sources in (
        (public_root, p["implementation_commit"], p["public_source_sha256"]),
        (oracle_root, ORACLE_COMMIT, p["oracle_source_sha256"]),
    ):
        if _git(root, "rev-parse", "HEAD") != commit or _git(
            root, "status", "--porcelain", "--untracked-files=no"
        ):
            raise ValueError("numerical work requires the bound clean checkout")
        _sources(root, sources)
    output.mkdir(parents=True, exist_ok=False)
    stages = []
    for mode in ("public32", "public64", "oracle64"):
        root = oracle_root if mode == "oracle64" else public_root
        source = p[
            "oracle_source_sha256" if mode == "oracle64" else "public_source_sha256"
        ]
        request = dict(
            mode=mode,
            root=str(root),
            source_sha256=source,
            runtime=p["runtime"],
            manifest=str(manifest_path),
            manifest_sha256=expected_manifest_sha256,
        )
        request_path = output / f"{mode}-request.json"
        write_json(request_path, request)
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "--worker",
            str(request_path),
            "--output",
            str(output / mode),
        ]
        env = {
            **os.environ,
            "SCIPY_ARRAY_API": "1",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(root / "src"),
        }
        env.pop("JAX_ENABLE_X64", None)
        if mode != "public32":
            env["JAX_ENABLE_X64"] = "1"
        with (output / f"{mode}.log").open("w") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                )
                stage = dict(
                    mode=mode,
                    command=command,
                    returncode=completed.returncode,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                stage = dict(
                    mode=mode, command=command, returncode=None, timed_out=True
                )
        stages.append(stage)
        write_json(output / "stages.json", stages)
    result = (
        reduce_saved(manifest_path, output)
        if all(s["returncode"] == 0 for s in stages)
        else dict(passed=False, reason="worker_failure", stages=stages, fits=0)
    )
    write_json(output / "result.json", result)
    files = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(
        output / "run.json",
        dict(input_manifest_sha256=expected_manifest_sha256, files=files, fits=0),
    )
    return {
        "passed": result["passed"],
        "result_sha256": digest(output / "result.json"),
        "run_sha256": digest(output / "run.json"),
        "fits": 0,
    }


def verify_saved(
    manifest_path, directory, *, expected_manifest_sha256, expected_run_sha256
):
    """Authenticate and independently reduce a completed saved numerical run."""
    manifest_path, directory = Path(manifest_path), Path(directory)
    if (
        digest(manifest_path) != expected_manifest_sha256
        or digest(directory / "run.json") != expected_run_sha256
    ):
        raise ValueError("external numerical evidence anchor differs")
    run_manifest = read_json(directory / "run.json")
    if run_manifest["input_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("numerical run names another input packet")
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "run.json"
    }
    if actual != set(run_manifest["files"]):
        raise ValueError("numerical evidence file roster differs")
    for rel, sha in run_manifest["files"].items():
        if digest(directory / rel) != sha:
            raise ValueError(f"numerical evidence payload changed: {rel}")
    packet = read_json(manifest_path)
    for case in packet["cases"]:
        if digest(manifest_path.parent / case["path"]) != case["sha256"]:
            raise ValueError("numerical source arrays changed")
        if (
            case["model_path"] is not None
            and digest(manifest_path.parent / case["model_path"])
            != case["model_sha256"]
        ):
            raise ValueError("numerical source model changed")
    if any(s["returncode"] != 0 for s in read_json(directory / "stages.json")):
        return dict(passed=False, reason="preserved_worker_failure", fits=0)
    result = reduce_saved(manifest_path, directory)
    # JSON roundtrip applies only the explicit nonfinite diagnostic representation.
    saved = read_json(directory / "result.json")

    def equivalent(a, b):
        if isinstance(a, dict):
            return (
                isinstance(b, dict)
                and set(a) == set(b)
                and all(equivalent(a[k], b[k]) for k in a)
            )
        if isinstance(a, list):
            return (
                isinstance(b, list)
                and len(a) == len(b)
                and all(equivalent(x, y) for x, y in zip(a, b, strict=True))
            )
        if isinstance(a, float) and not np.isfinite(a):
            return b is None
        return bool(a == b)

    if not equivalent(result, saved):
        raise ValueError("numerical result differs from saved transform reductions")
    return {"passed": result["passed"], "cases": result["cases"], "fits": 0}


def replay(manifest_path, directory, output, *, expected_run_sha256, **kwargs):
    """Fresh transformations plus exact saved-array comparison, never a refit."""
    verify_saved(
        manifest_path,
        directory,
        expected_manifest_sha256=kwargs["expected_manifest_sha256"],
        expected_run_sha256=expected_run_sha256,
    )
    result = run(manifest_path, output, **kwargs)
    if not result["passed"]:
        return result
    checked = 0
    for mode in ("public32", "public64", "oracle64"):
        old = read_json(Path(directory) / mode / "result.json")
        new = read_json(Path(output) / mode / "result.json")
        if [x["id"] for x in old["cases"]] != [x["id"] for x in new["cases"]]:
            raise ValueError("fresh numeric replay case roster differs")
        for previous, fresh in zip(old["cases"], new["cases"], strict=True):
            with (
                np.load(
                    Path(directory) / mode / previous["path"], allow_pickle=False
                ) as a,
                np.load(Path(output) / mode / fresh["path"], allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files):
                    raise ValueError("fresh numeric replay array roster differs")
                for key in a.files:
                    if a[key].dtype != b[key].dtype or not np.array_equal(
                        a[key], b[key], equal_nan=True
                    ):
                        raise ValueError(
                            f"fresh numeric replay differs: {mode}/{previous['id']}/{key}"
                        )
                    checked += 1
    return {**result, "exact_replayed_arrays": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _worker(args.worker, args.output)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
