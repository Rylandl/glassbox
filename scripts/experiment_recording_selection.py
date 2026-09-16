"""Preplanned recording-level selection and fixed-row segment coverage study."""

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from experiment_forecast_representation import cf_windows, fit_case
from prepare_crazyflie_reference import HASHES, decode
from sequence_transfer_data import write_json

from glassbox.experimental.causal_sampling import causal_hold
from glassbox.experimental.forecast_selection import POLICIES, ForecastEvidence
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
    segments_from_mask,
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recording_ids(usage):
    return np.concatenate([np.repeat(u["recording"], u["windows"]) for u in usage])


def recording_scores(prediction, target, ids, groups):
    return {
        r: np.stack(
            [
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                prediction[ids == r][:, :, list(g)]
                                - target[ids == r][:, :, list(g)]
                            )
                            ** 2,
                            -1,
                        ),
                        0,
                    )
                )
                for g in groups
            ],
            -1,
        ).tolist()
        for r in dict.fromkeys(ids.tolist())
    }


def archive(output):
    repo = Path(__file__).resolve().parents[1]
    files = [
        *sorted((repo / "scripts").glob("*.py")),
        *sorted((repo / "src/glassbox/experimental").glob("*.py")),
    ]
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in files:
            z.write(p, p.relative_to(repo))
    write_json(
        output / "sources.json", {str(p.relative_to(repo)): digest(p) for p in files}
    )
    write_json(
        output / "environment.json",
        dict(
            python=sys.version,
            jax=jax.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
            float64=jax.config.jax_enable_x64,
        ),
    )


def selection_study(a):
    rows = []
    for dataset in ("nano", "x8"):
        for seed in (60, 61, 62):
            name = f"fold0-seed{seed}"
            source = a.previous / "ablation-01" / dataset / name
            old = a.original / dataset / name
            out = a.output / "selection" / dataset / name
            out.mkdir(parents=True)
            report = read(source / "report.json")
            usage = read(old / "usage.json")
            ids = {r: recording_ids(usage[r]) for r in ("development", "evaluation")}
            identities = {
                r: {u["recording"] for u in uses} for r, uses in usage.items()
            }
            assert not any(
                identities[x] & identities[y]
                for x, y in (
                    ("train", "development"),
                    ("train", "evaluation"),
                    ("development", "evaluation"),
                )
            )
            with np.load(source / "samples.npz") as z:
                targets = {
                    r: z[f"{r}_future_states"][:, np.array(report["horizons"]) - 1]
                    for r in ids
                }
            with np.load(source / "predictions.npz") as z:
                predictions = {
                    r: {
                        k.split("__")[1]: z[k]
                        for k in z.files
                        if k.startswith(r + "__")
                    }
                    for r in ids
                }
            evidence = ForecastEvidence.from_predictions(
                predictions["development"],
                targets["development"],
                ids["development"],
                scale=report["state_scale"],
                groups=report["groups"],
            )
            assert evidence.choose("pooled")["selected"] == report["selected"]
            metrics = {
                r: {
                    n: recording_scores(p, targets[r], ids[r], report["groups"])
                    for n, p in predictions[r].items()
                }
                for r in ids
            }
            decisions = []
            for excluded in (None, *evidence.recordings):
                records = [r for r in evidence.recordings if r != excluded]
                role = "evaluation" if excluded is None else "development"
                tested = (
                    list(dict.fromkeys(ids[role].tolist()))
                    if excluded is None
                    else [excluded]
                )
                for policy in POLICIES:
                    decision = evidence.choose(policy, recordings=records)
                    selected = decision["selected"]
                    decision.update(
                        excluded=excluded,
                        role=role,
                        tested={r: metrics[role][selected][r] for r in tested},
                        hold={r: metrics[role]["hold"][r] for r in tested},
                    )
                    decisions.append(decision)
                    rows.append(dict(dataset=dataset, seed=seed, **decision))
            write_json(
                out / "report.json",
                dict(
                    dataset=dataset,
                    seed=seed,
                    horizons=report["horizons"],
                    groups=report["groups"],
                    names=evidence.names,
                    recording_counts=dict(
                        zip(evidence.recordings, evidence.counts.tolist())
                    ),
                    decisions=decisions,
                ),
            )
            write_json(out / "recording-scores.json", metrics)
            write_json(
                out / "source-files.json",
                {
                    str(p.resolve()): digest(p)
                    for p in [
                        source / "report.json",
                        source / "samples.npz",
                        source / "predictions.npz",
                        old / "usage.json",
                    ]
                },
            )
            np.savez_compressed(
                out / "evidence.npz",
                mse=evidence.mse,
                counts=evidence.counts,
                group_widths=evidence.group_widths,
            )
            print(
                json.dumps(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        all_development={
                            d["policy"]: d["selected"]
                            for d in decisions
                            if d["excluded"] is None
                        },
                    )
                ),
                flush=True,
            )
    write_json(a.output / "selection-summary.json", rows)


def balanced_lengths(capacities, budget, minimum):
    if (
        budget < minimum * len(capacities)
        or budget > sum(capacities)
        or min(capacities) < minimum
    ):
        raise ValueError("infeasible contiguous-segment row budget")
    allocation = np.full(len(capacities), minimum)
    while allocation.sum() < budget:
        available = np.flatnonzero(allocation < capacities)
        i = available[np.argmin(allocation[available])]
        allocation[i] += 1
    return allocation


def coverage_study(a):
    path = a.previous / "new-corpus/log10"
    assert digest(path) == HASHES["log10"]
    raw = decode(path)["fixedFrequency"]
    t = raw["timestamp_s"]
    values = np.column_stack(
        [raw[f"acc.{c}"] for c in "xyz"]
        + [raw[f"gyro.{c}"] for c in "xyz"]
        + [raw[f"motor.m{i}"] for i in range(1, 5)]
    )
    grid = np.arange(np.ceil(t[0] / 0.02), np.floor(t[-1] / 0.02) + 1) * 0.02
    held = causal_hold(t, values, grid, maximum_age_s=0.01)
    valid = held.valid & (held.values[:, 6:].mean(1) / 65536 > 0.1)
    states = np.column_stack((held.values[:, :3], np.deg2rad(held.values[:, 3:6])))
    inputs = held.values[:-1, 6:] / 65536
    segments = segments_from_mask("log10", states, inputs, valid, dt_s=0.02)
    eligible = [s for s in segments if len(s.states) >= 18]
    assert len(eligible) == 10
    lengths = balanced_lengths(np.array([len(s.states) for s in eligible]), 337, 18)
    fixed = {
        r: cf_windows(a.previous / "cf-complete", r)[0]
        for r in ("development", "evaluation")
    }
    for seed in (60, 61, 62):
        out = a.output / "coverage" / f"seed{seed}"
        out.mkdir(parents=True)
        rng = np.random.default_rng(seed)
        crops = []
        provenance = []
        for s, length in zip(eligible, lengths):
            offset = int(rng.integers(len(s.states) - length + 1))
            start = s.start_row + offset
            crops.append(
                SequenceSegment(
                    s.recording_id,
                    f"crop-{start}-{start + length}",
                    s.states[offset : offset + length],
                    s.inputs[offset : offset + length - 1],
                    s.dt_s,
                    start,
                )
            )
            provenance.append(
                dict(
                    segment=s.segment_id,
                    eligible_start=s.start_row,
                    eligible_stop=s.start_row + len(s.states),
                    crop_start=start,
                    crop_stop=int(start + length),
                )
            )
        collection = SequenceCollection(tuple(crops))
        windows = collection.extract(
            collection.window_keys(history_steps=5, horizon_steps=12),
            history_steps=5,
            horizon_steps=12,
        )
        coverage = windows.coverage()["log10"]
        assert coverage["unique_state_rows"] == 337 and coverage["windows"] == 167
        write_json(
            out / "usage.json",
            dict(
                raw_sha256=HASHES["log10"],
                seed=seed,
                crops=provenance,
                keys=[asdict(k) for k in windows.keys],
                source_origins=windows.source_origins,
                coverage=coverage,
                development="log15 unchanged",
                evaluation="log16 unchanged; already inspected",
            ),
        )
        start_time = time.monotonic()
        report = fit_case(
            out / "fitted",
            {"train": windows.batch, **fixed},
            0.02,
            (1, 5, 12),
            ((0, 1, 2), (3, 4, 5)),
            dict(dataset="crazyflie_fixed_rows", seed=seed),
        )
        scores = report["scores"]["evaluation"]
        print(
            json.dumps(
                dict(
                    seed=seed,
                    coverage=coverage,
                    selected=report["selected"],
                    selected_rmse=scores[report["selected"]]["rmse"][-1],
                    full_rmse=scores[report["selected_families"]["full"]]["rmse"][-1],
                    seconds=round(time.monotonic() - start_time, 2),
                )
            ),
            flush=True,
        )


def main(a):
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded experiments require float64")
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "plan.json", read(a.plan))
    archive(a.output)
    selection_study(a)
    coverage_study(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--plan", type=Path, default=Path("../artifacts/recording-selection/plan.json")
    )
    p.add_argument(
        "--previous", type=Path, default=Path("../artifacts/representation-study")
    )
    p.add_argument(
        "--original",
        type=Path,
        default=Path("../artifacts/forecast-diagnosis/comparison-01"),
    )
    main(p.parse_args())
