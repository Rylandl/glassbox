"""A fixed-budget investigation of output sharing, history, and optimization.

Only non-Melon recordings are read. Runs 1/2 fit the mean, run 3 is development,
and run 4 is a secondary evaluation. All were used in earlier work, so neither
evaluation is advertised as a pristine final holdout. No error calibration is fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_real_transition import OUTPUT_NAMES, prediction, ridge_prediction, score

from glassbox.core.data import load_trajectory_npz
from glassbox.core.geometry import quaternion_to_rotation_matrices
from glassbox.experimental.outputwise_transition import fit_outputwise_transition_gp
from glassbox.experimental.temporal import forecast_windows
from glassbox.experimental.transition_gp import TransitionSamples, fit_transition_gp
from glassbox.io.nanodrone_reference import BENCHMARK_COMMIT, NanoDroneBenchmarkAdapter

HISTORIES = {"none": (), "20ms": (1, 2), "100ms": (5, 10), "200ms": (10, 20)}
HORIZONS = (1, 10, 25)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_records(corpus):
    records = []
    for path in sorted((corpus / "canonical/train").glob("*.npz")):
        flight = load_trajectory_npz(path)
        if flight.labels["benchmark_split"] != "train":
            raise ValueError("this diagnosis must not read Melon test recordings")
        raw = corpus / "raw" / flight.provenance["benchmark"]["relative_path"]
        inspection = NanoDroneBenchmarkAdapter().inspect(raw)
        records.append(
            {
                "name": path.stem,
                "flight": flight,
                "role": {1: "train", 2: "train", 3: "development", 4: "replication"}[
                    flight.labels["replicate"]
                ],
                "inspection": inspection,
                "states": np.concatenate(
                    (flight.states[:, 3:6], flight.states[:, 10:13]), axis=1
                ),
                "controls": np.sqrt(flight.controls) * 2500,
                "context": quaternion_to_rotation_matrices(
                    flight.states[:, 6:10]
                ).reshape(-1, 9),
            }
        )
    if len(records) != 12:
        raise ValueError("expected twelve pinned non-Melon recordings")
    return records


def assemble(records, role, seed, budget, horizon, lags):
    selected = [r for r in records if r["role"] == role]
    arrays = {
        key: []
        for key in (
            "states",
            "commands",
            "next_states",
            "context",
            "features",
            "ids",
            "groups",
            "anchors",
            "trend",
        )
    }
    usage = []
    if budget is not None and budget % len(selected):
        raise ValueError("training budget must divide evenly among flights")
    for record in selected:
        pool = np.arange(20, len(record["states"]) - max(HORIZONS))
        if budget is None:
            anchors = pool[::10]
        else:
            identity = int(hashlib.sha256(record["name"].encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
            anchors = rng.permutation(pool)[: budget // len(selected)]
        batch = forecast_windows(
            record["states"],
            record["controls"],
            recording_id=record["name"],
            dt_s=0.01,
            horizon_steps=horizon,
            anchors=anchors,
            context=record["context"],
            history_lags=lags,
        )
        for key in ("states", "commands", "next_states", "context", "features"):
            arrays[key].append(getattr(batch.samples, key))
        arrays["ids"].append(np.asarray(batch.sample_ids))
        arrays["groups"].append(np.full(len(anchors), record["name"]))
        arrays["anchors"].append(anchors)
        arrays["trend"].append(
            record["states"][anchors]
            + horizon / 5 * (record["states"][anchors] - record["states"][anchors - 5])
        )
        state_rows = set(anchors) | set(anchors + horizon)
        control_rows = {a + step for a in anchors for step in range(horizon)}
        for lag in lags:
            state_rows.update(anchors - lag)
            control_rows.update(anchors - lag)
        usage.append(
            {
                "recording": record["name"],
                "windows": len(anchors),
                "unique_state_rows": len(state_rows),
                "unique_control_intervals": len(control_rows),
                "control_interval_union_s": len(control_rows) * 0.01,
            }
        )
    return {key: np.concatenate(parts) for key, parts in arrays.items()}, usage


def run(corpus, output, seeds, smoke):
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded diagnosis requires JAX_ENABLE_X64=1")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "scripts/experiment_real_transition.py",
        root / "tests/test_outputwise_transition.py",
        *sorted((root / "src/glassbox/experimental").glob("*.py")),
    ]
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            archive.write(path, path.relative_to(root))
    write_json(
        output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    records = load_records(corpus)
    histories = {k: HISTORIES[k] for k in (("none", "100ms") if smoke else HISTORIES)}
    horizons = (10,) if smoke else HORIZONS
    steps = 4 if smoke else 160
    cases = [
        {"seed": seed, "budget": budget, "history": history, "horizon_steps": horizon}
        for seed in seeds
        for horizon in horizons
        for history in histories
        for budget in (
            (48,) if smoke else ((192, 384) if history == "100ms" else (384,))
        )
    ]
    plan = {
        "seeds": seeds,
        "cases": cases,
        "histories": histories,
        "horizons": horizons,
        "shared_steps": steps,
        "shared_long_steps": steps * 3,
        "outputwise_steps_per_member": steps,
        "restarts": 2,
        "kernel": "rq",
        "mean_mode": "increment",
        "outputs": OUTPUT_NAMES,
        "split": {
            role: [r["name"] for r in records if r["role"] == role]
            for role in ("train", "development", "replication")
        },
        "source_commit": BENCHMARK_COMMIT,
        "smoke": smoke,
        "interpretation": "Factorial diagnostic, not final generalization evidence. No Melon files loaded. Non-Melon recordings were used in prior work.",
        "comparison": "Identical feature information and vector labels per case. Outputwise fits six independent kernels/noises; shared-long gets 3x updates. Compute is recorded, not equalized.",
        "history": "Two causal lags of state/input changes for every nonempty history, with equal feature dimension. Current orientation only; no future state features.",
        "budget": "Selected vector windows, not independent observations or a flight collection time. Smaller training budgets are nested within each flight.",
        "processing": "Offline filtered/retimed real measurements; supplied future measured motor speeds. History effects do not establish physical memory.",
    }
    write_json(output / "plan.json", plan)
    write_json(output / "data-audit.json", [r["inspection"] for r in records])
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "x64": True,
        },
    )
    reports = []
    for case in cases:
        seed, budget, history, horizon = (
            case[k] for k in ("seed", "budget", "history", "horizon_steps")
        )
        folder = output / f"h{horizon}-{history}-n{budget}-seed{seed}"
        folder.mkdir()
        batches, usage = {}, {}
        for role in ("train", "development", "replication"):
            batches[role], usage[role] = assemble(
                records,
                role,
                seed,
                budget if role == "train" else None,
                horizon,
                histories[history],
            )
        train = batches["train"]
        samples = TransitionSamples(
            train["states"],
            train["commands"],
            train["next_states"],
            horizon * 0.01,
            train["context"],
        )
        np.savez_compressed(
            folder / "samples.npz",
            **{
                f"{role}_{key}": values
                for role, batch in batches.items()
                for key, values in batch.items()
            },
        )
        report = {**case, "usage": usage, "methods": {}}
        saved = {}
        for method, fit_steps in (
            ("shared", steps),
            ("shared-long", steps * 3),
            ("outputwise", steps),
        ):
            start = time.monotonic()
            fitter = (
                fit_outputwise_transition_gp
                if method == "outputwise"
                else fit_transition_gp
            )
            model = fitter(
                samples, kernel="rq", steps=fit_steps, restarts=2, mean_mode="increment"
            )
            elapsed = time.monotonic() - start
            model.save(folder / (method if method == "outputwise" else f"{method}.npz"))
            members = model.members if method == "outputwise" else (model,)
            arm = {
                "fit_seconds_including_compile": elapsed,
                "parameter_count": sum(len(m.theta) for m in members),
                "optimizer_updates": fit_steps * 2 * len(members),
                "training_nll_sum": sum(m.fit_report["nll"] for m in members),
                "fingerprint": model.fingerprint(),
                "scores": {},
            }
            for role in ("development", "replication"):
                mean, fvar, ovar = prediction(model, batches[role])
                for key, value in (
                    ("mean", mean),
                    ("function_variance", fvar),
                    ("observation_variance", ovar),
                ):
                    saved[f"{method}_{role}_{key}"] = value
                arm["scores"][role] = score(mean, batches[role])
            report["methods"][method] = arm
        both = {
            key: np.concatenate(
                [batches[role][key] for role in ("development", "replication")]
            )
            for key in ("states", "features")
        }
        linear = ridge_prediction(train, both, folder / "ridge.npz")
        split = len(batches["development"]["states"])
        for role, linear_mean in zip(
            ("development", "replication"), (linear[:split], linear[split:])
        ):
            for method, mean in (
                ("ridge", linear_mean),
                ("hold", batches[role]["states"]),
                ("trend", batches[role]["trend"]),
            ):
                saved[f"{method}_{role}_mean"] = mean
                report["methods"].setdefault(method, {"scores": {}})["scores"][role] = (
                    score(mean, batches[role])
                )
        np.savez_compressed(folder / "predictions.npz", **saved)
        write_json(folder / "report.json", report)
        reports.append(report)
        text = {
            **case,
            "rates_replication": {
                m: round(v["scores"]["replication"]["all"]["rate_rmse_rad_s"], 5)
                for m, v in report["methods"].items()
            },
            "fit_s": round(
                sum(
                    v.get("fit_seconds_including_compile", 0)
                    for v in report["methods"].values()
                ),
                2,
            ),
        }
        print(json.dumps(text), flush=True)
    write_json(output / "summary.json", {"cases": len(reports), "reports": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[60, 61, 62])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.corpus, args.output, args.seeds, args.smoke)
