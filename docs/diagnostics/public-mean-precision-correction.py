"""No-fit same-model precision transitions on the original 24 actual cases.

Run each precision order in a fresh process after the correction source is
committed and bound. Original arithmetic qualification keeps its own verdict.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import jax
import numpy as np

from glassbox import LearnedDynamics
from glassbox.experimental.public_mean_implementation import verify
from glassbox.experimental.public_v4_numerics import (
    INPUTS,
    TOLERANCES,
    _imported_sources,
    element_check,
)

BASE = Path("/private/tmp/glassbox-public-mean-qualification/artifacts/2026-09-19")
PACKET = BASE / "public-mean-qualification-v1-actual-numeric-input-attempt1"
ORIGINAL = BASE / "public-mean-qualification-v1-actual-numeric-attempt1"
PACKET_SHA = "5a8f1ebd6007537bb9f9d751cd138c7e0400e9ff7180c5cd2771ef31df132797"
RUN_SHA = "ac86379c6794d5e0962c25b35590b09754f9b09077605b3737497c5c8ff4471e"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: np.array(z[key], copy=True) for key in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order", choices=("32-64-32", "64-32-64"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    args = parser.parse_args()
    binding = verify(args.binding, args.binding_sha256)
    assert Path(__file__).resolve().parents[2] == Path(binding["public_root"])
    imported = _imported_sources(
        Path(binding["public_root"]), binding["public_source_sha256"]
    )
    assert digest(PACKET / "manifest.json") == PACKET_SHA
    assert digest(ORIGINAL / "run.json") == RUN_SHA
    original_run = read(ORIGINAL / "run.json")
    for relative, expected in original_run["files"].items():
        assert digest(ORIGINAL / relative) == expected
    packet = read(PACKET / "manifest.json")
    cases = [
        (i, c) for i, c in enumerate(packet["cases"]) if c["source"] == "public_archive"
    ]
    assert len(cases) == 24
    assert any(
        c["id"] == "cascade/cascade/primary-00/test-11200000/factual-0010/factual"
        for _, c in cases
    )
    args.output.mkdir(parents=True, exist_ok=False)
    ambient = bool(jax.config.x64_enabled)
    result = dict(
        status="failed",
        order=args.order,
        fits=0,
        cases=[],
        binding_sha256=args.binding_sha256,
        packet_sha256=PACKET_SHA,
        original_run_sha256=RUN_SHA,
        script_sha256=digest(__file__),
        original_numerical_qualification_passed=False,
        imported_sources=imported,
    )
    try:
        for index, case in cases:
            assert digest(PACKET / case["path"]) == case["sha256"]
            assert digest(PACKET / case["model_path"]) == case["model_sha256"]
            values = load(PACKET / case["path"])
            model = LearnedDynamics.load(PACKET / case["model_path"])
            assert model.fingerprint() == case["model_fingerprint"]
            fingerprint = model.fingerprint()
            inputs = tuple(values[k] for k in INPUTS)
            predict = jax.jit(model.predict)
            separate = {precision: jax.jit(model.predict) for precision in ("32", "64")}
            saved, checks, repeats = {}, [], {}
            for step, precision in enumerate(args.order.split("-")):
                enabled = precision == "64"
                mode = "public" + precision
                witness = read(ORIGINAL / mode / "result.json")["cases"][index]
                assert witness["id"] == case["id"]
                assert digest(ORIGINAL / mode / witness["path"]) == witness["sha256"]
                old = load(ORIGINAL / mode / witness["path"])
                with jax.enable_x64(enabled):
                    # Exercise the reused compiled wrapper before any eager call
                    # or fresh wrapper can prime the new precision context.
                    paths = {
                        "shared_jit": np.asarray(predict(*inputs)),
                        "separate_jit": np.asarray(separate[precision](*inputs)),
                        "eager": np.asarray(model.predict(*inputs)),
                    }
                    for name, y in paths.items():
                        assert y.dtype == np.dtype("float" + precision)
                        expected = old[
                            "actual__eager" if name == "eager" else "actual__jit"
                        ]
                        mean, scale = (
                            values["norm_state_mean"],
                            values["norm_state_scale"],
                        )
                        check = element_check(
                            (y.astype(np.float64) - mean) / scale,
                            (expected.astype(np.float64) - mean) / scale,
                            TOLERANCES["forecast64" if enabled else "paths32"],
                        )
                        check.update(
                            step=step,
                            precision=precision,
                            path=name,
                            exact=bool(np.array_equal(y, expected)),
                        )
                        assert check["passed"], (case["id"], check)
                        key = precision, name
                        if key in repeats:
                            assert np.array_equal(y, repeats[key])
                        repeats[key] = y
                        saved[f"step{step}_{name}"] = y
                        checks.append(check)
                assert bool(jax.config.x64_enabled) == ambient
            assert model.fingerprint() == fingerprint
            path = args.output / f"case-{index:03d}.npz"
            np.savez_compressed(path, **saved)
            result["cases"].append(
                dict(id=case["id"], path=path.name, sha256=digest(path), checks=checks)
            )
            write(args.output / "result.json", result)
            jax.clear_caches()
        result["status"] = "passed"
    except Exception as error:
        result["error"] = dict(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["ambient_restored"] = bool(jax.config.x64_enabled) == ambient
        result["imported_sources_at_completion"] = _imported_sources(
            Path(binding["public_root"]), binding["public_source_sha256"]
        )
        verify(args.binding, args.binding_sha256)
        write(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result["cases"]),
                "order": args.order,
                "fits": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
