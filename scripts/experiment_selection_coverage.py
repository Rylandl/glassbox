"""Vary independent development recordings at a fixed scored-window budget."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_checkpoint_attribution import OBJECTIVES, predictions, select
from experiment_horizon_generalization import generate, write

from glassbox.experimental.sequence_model import SequenceModel


def pool_arrays(plan, data_seed, pool_seed, count):
    dataset = copy.deepcopy(plan["dataset"])
    dataset["rng_salt"] += plan["selection_rng_salt_offset"] + pool_seed
    dataset["evaluation_recordings_per_regime"] = max(plan["recording_counts"])
    supplied, _ = generate(dataset, "near_periodic", data_seed, "matched")
    assert plan["windows"] % count == 0
    per_record = plan["windows"] // count
    index = []
    for i, segment in enumerate(supplied.segments[:count]):
        identity = f"selection-{i}"
        ranked = sorted(
            range(2, len(segment.states) - 5),
            key=lambda t: hashlib.sha256(
                json.dumps([identity, "whole", t], sort_keys=True).encode()
            ).hexdigest(),
        )
        assert len(ranked) >= per_record
        index.extend((identity, segment, t) for t in ranked[:per_record])
    return dict(
        past_states=np.stack([s.states[t - 2 : t + 1] for _, s, t in index]),
        past_inputs=np.stack([s.inputs[t - 2 : t] for _, s, t in index]),
        future_inputs=np.stack([s.inputs[t : t + 5] for _, s, t in index]),
        targets=np.stack([s.states[t + 1 : t + 6] for _, s, t in index]),
        recording_ids=np.array([name for name, _, _ in index]),
        origins=np.array([t for _, _, t in index]),
    )


def main(args):
    assert jax.config.jax_enable_x64
    plan = json.loads(args.plan.read_text())
    assert plan["recording_counts"] == [2, 4, 8, 16]
    assert plan["selection_pool_seeds"] == [101, 202, 303] and plan["windows"] == 256
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
    root = Path(__file__).resolve().parents[1]
    sources = (
        sorted(root.glob("src/**/*.py"))
        + sorted(root.glob("scripts/*.py"))
        + [root / "tests/test_selection_coverage.py"]
    )
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sources:
            z.write(p, str(p.relative_to(root)))
    write(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    result_rows = []
    source_hashes = {}
    for phase, run_names in plan["sources"].items():
        for run_name in run_names:
            run = args.artifacts / run_name
            for source in json.loads((run / "results.json").read_text()):
                if source["family"] != "near_periodic":
                    continue
                name = f"near_periodic-{source['data_seed']}-fit{source['fit_seed']}"
                directory = run / name
                paths = {
                    o: [
                        SequenceModel.load(directory / f"{o}-{step:04d}.npz")
                        for step in range(0, 1001, 100)
                    ]
                    for o in OBJECTIVES
                }
                for p in [*directory.glob("*.npz"), directory / "result.json"]:
                    source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                for pool_seed in plan["selection_pool_seeds"]:
                    output = args.output / f"{name}-pool{pool_seed}"
                    output.mkdir()
                    variants = {}
                    for count in plan["recording_counts"]:
                        arrays = pool_arrays(
                            plan, source["data_seed"], pool_seed, count
                        )
                        forecasts = {
                            o: predictions(models, arrays)
                            for o, models in paths.items()
                        }
                        selectors = {
                            o: select(
                                p,
                                arrays["targets"],
                                np.asarray(source["loss_scales"][o]),
                                arrays["recording_ids"],
                            )
                            for o, p in forecasts.items()
                        }
                        write(output / f"selections-{count}.json", selectors)
                        np.savez_compressed(
                            output / f"pool-{count}.npz", **arrays, **forecasts
                        )
                        # No evaluation values are consulted until selectors are fixed.
                        scores = {
                            regime: {
                                o: report["metrics"][o]["checkpoints"][
                                    selectors[o]["selected_index"]
                                ]
                                for o in OBJECTIVES
                            }
                            for regime, report in source["regimes"].items()
                        }
                        variants[str(count)] = dict(
                            selections=selectors, regimes=scores
                        )
                    baseline = {
                        regime: {
                            o: report["metrics"][o]["checkpoints"][
                                source["selected"][o][o]["selected_index"]
                            ]
                            for o in OBJECTIVES
                        }
                        for regime, report in source["regimes"].items()
                    }
                    row = dict(
                        phase=phase,
                        data_seed=source["data_seed"],
                        fit_seed=source["fit_seed"],
                        pool_seed=pool_seed,
                        source_run=run_name,
                        source_case=name,
                        source_result_sha256=source_hashes[
                            str(directory / "result.json")
                        ],
                        baseline=baseline,
                        variants=variants,
                    )
                    write(output / "result.json", row)
                    result_rows.append(row)
                    write(args.output / "results.json", result_rows)
                    print(
                        json.dumps(
                            dict(
                                completed=output.name,
                                phase=phase,
                                selected_steps={
                                    n: {
                                        o: v["selected_index"] * 100
                                        for o, v in r["selections"].items()
                                    }
                                    for n, r in variants.items()
                                },
                            )
                        ),
                        flush=True,
                    )
    assert len(result_rows) == 45
    write(args.output / "source-artifacts.json", source_hashes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
