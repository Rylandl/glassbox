"""Freeze observational diagnostics against synthetic witnesses and blind spots."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_model_qualification import collection, encoding, recordings

from glassbox.experimental.default_model import _RECIPE, LearnedDynamics, fit
from glassbox.experimental.sequence_collection import SequenceSegment
from glassbox.experimental.sequence_diagnostics import _DIAGNOSTIC_RECIPE, _run


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fresh_recordings(family, seed, *, evaluation=False):
    rows = []
    for i in range(4 if evaluation else 8):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(evaluation), i, 517])
        )
        if family in ("quadratic_policy", "cubic_policy"):
            states = rng.uniform(-1, 1, (161, 1))
            inputs = states[:-1] ** (2 if family == "quadratic_policy" else 3)
        else:
            assert family == "process_noise"
            inputs = rng.uniform(-1, 1, (160, 1))
            disturbance = 0.1 * rng.normal(size=(160, 1))
            states = np.zeros((161, 1))
            states[0] = rng.uniform(-1, 1)
            for k in range(160):
                states[k + 1] = 0.8 * states[k] + 0.2 * inputs[k] + disturbance[k]
        rows.append(
            SequenceSegment(
                f"{'evaluation' if evaluation else 'fit'}-{i}",
                "whole",
                states,
                inputs,
                0.05,
            )
        )
    return tuple(rows)


def main(args):
    assert jax.config.jax_enable_x64
    plan = json.loads(args.plan.read_text())
    assert (
        plan["learner_recipe"] == _RECIPE
        and plan["diagnostic_recipe"] == _DIAGNOSTIC_RECIPE
    )
    assert plan["data_seeds"] == [101, 202, 303]
    assert plan["cases"] == dict(
        existing=[
            "command-0.0",
            "command-0.02",
            "command-0.25",
            "memory-hidden",
            "encoding-identity",
        ],
        fresh=["quadratic_policy", "cubic_policy", "process_noise"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    write(
        args.output / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64=True,
        ),
    )
    repo = Path(__file__).resolve().parents[1]
    source = [
        *sorted((repo / "src/glassbox").rglob("*.py")),
        Path(__file__),
        repo / "scripts/experiment_model_qualification.py",
        repo / "tests/test_sequence_diagnostics.py",
    ]
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in source:
            z.write(p, str(p.relative_to(repo)))
    write(
        args.output / "sources.json",
        {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source
        },
    )
    summaries = []
    for kind in ("existing", "fresh"):
        for family in plan["cases"][kind]:
            for seed in plan["data_seeds"]:
                name = f"{family}-{seed}"
                directory = args.output / name
                directory.mkdir()
                print(json.dumps(dict(starting=name)), flush=True)
                if kind == "existing":
                    short, variant = family.split("-", 1)
                    source_path = (
                        args.previous / f"{short}-{seed}-{variant}" / "model.npz"
                    )
                    model = LearnedDynamics.load(source_path)
                    rows = recordings(
                        short,
                        seed,
                        evaluation=True,
                        excitation=float(variant) if short == "command" else 0,
                    )
                    supplied = collection(
                        rows,
                        encoding("identity", 2 if short == "encoding" else 1),
                        short,
                    )
                    old = json.loads(source_path.with_name("result.json").read_text())
                    assert old["fingerprint"] == model.fingerprint()
                else:
                    base = fresh_recordings(family, seed)
                    model = fit(collection(base, encoding("identity", 1), family))
                    source_path = directory / "model.npz"
                    model.save(source_path)
                    write(directory / "fit-report.json", model.report)
                    rows = fresh_recordings(family, seed, evaluation=True)
                    supplied = collection(rows, encoding("identity", 1), family)
                before = model.fingerprint()
                report, arrays = _run(model, supplied)
                assert model.fingerprint() == before
                write(directory / "report.json", report)
                np.savez_compressed(directory / "evidence.npz", **arrays)
                result = dict(
                    case=family,
                    seed=seed,
                    model_source=str(source_path.resolve()),
                    model_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    model_fingerprint=before,
                    **report["aggregate"],
                )
                write(directory / "result.json", result)
                summaries.append(result)
                write(args.output / "results.json", summaries)
                print(json.dumps(result), flush=True)
    assert len(summaries) == 24


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument(
        "--previous",
        type=Path,
        default=Path("../artifacts/model-qualification/comparison-01"),
    )
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
