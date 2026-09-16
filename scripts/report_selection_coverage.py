"""Replay selection-coverage pools and retain every paired accuracy tradeoff."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from experiment_horizon_generalization import generate
from report_checkpoint_attribution import metric, read, score_vector, write
from report_model_structures import recurrence

from glassbox.experimental.sequence_model import SequenceModel


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan, rows = read(args.run / "plan.json"), read(args.run / "results.json")
    assert len(rows) == 45
    assert sum(r["phase"] == "confirmation" for r in rows) == 9
    with zipfile.ZipFile(args.run / "executed-sources.zip") as z:
        for name, digest in read(args.run / "sources.json").items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    for name, digest in read(args.run / "source-artifacts.json").items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    comparisons, maximum, windows, checkpoint_replays = 0, 0.0, 0, 0

    def check(a, b):
        nonlocal comparisons, maximum
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = max(maximum, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-8)
        comparisons += 1

    detailed = []
    seen = set()
    for row in rows:
        identity = (row["phase"], row["data_seed"], row["fit_seed"], row["pool_seed"])
        assert identity not in seen
        seen.add(identity)
        assert row["pool_seed"] in plan["selection_pool_seeds"]
        assert row["source_run"] in plan["sources"][row["phase"]]
        directory = args.artifacts / row["source_run"] / row["source_case"]
        source = read(directory / "result.json")
        assert (
            row["source_result_sha256"]
            == hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
        )
        assert (row["data_seed"], row["fit_seed"]) == (
            source["data_seed"],
            source["fit_seed"],
        )
        run = args.run / f"{row['source_case']}-pool{row['pool_seed']}"
        assert row == read(run / "result.json")
        dataset = copy.deepcopy(plan["dataset"])
        dataset["rng_salt"] += plan["selection_rng_salt_offset"] + row["pool_seed"]
        dataset["evaluation_recordings_per_regime"] = 16
        supplied, _ = generate(dataset, "near_periodic", row["data_seed"], "matched")
        paths = {
            o: [
                SequenceModel.load(directory / f"{o}-{step:04d}.npz")
                for step in range(0, 1001, 100)
            ]
            for o in ("original", "first_step_floor")
        }
        eval_arrays = {}
        for regime in ("matched", "shifted"):
            with np.load(directory / f"{regime}.npz", allow_pickle=False) as z:
                eval_arrays[regime] = {k: z[k] for k in z.files}
        for count in plan["recording_counts"]:
            variant = row["variants"][str(count)]
            assert variant["selections"] == read(run / f"selections-{count}.json")
            with np.load(run / f"pool-{count}.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            expected = []
            for i, segment in enumerate(supplied.segments[:count]):
                label = f"selection-{i}"
                ranked = [
                    (
                        hashlib.sha256(
                            json.dumps([label, "whole", t], sort_keys=True).encode()
                        ).hexdigest(),
                        t,
                    )
                    for t in range(2, len(segment.states) - 5)
                ]
                for _, t in sorted(ranked)[: 256 // count]:
                    expected.append((label, segment, t))
            assert len(expected) == len(arrays["targets"]) == 256
            assert arrays["recording_ids"].tolist() == [name for name, _, _ in expected]
            check(arrays["origins"], [t for _, _, t in expected])
            for key, attribute, offsets in (
                ("past_states", "states", np.arange(-2, 1)),
                ("past_inputs", "inputs", np.arange(-2, 0)),
                ("future_inputs", "inputs", np.arange(5)),
                ("targets", "states", np.arange(1, 6)),
            ):
                check(
                    arrays[key],
                    np.stack(
                        [
                            getattr(segment, attribute)[t + offsets]
                            for _, segment, t in expected
                        ]
                    ),
                )
            windows += 256
            for objective, path in paths.items():
                selected = variant["selections"][objective]
                predicted = np.stack(
                    [
                        recurrence(
                            m,
                            arrays["past_states"],
                            arrays["past_inputs"],
                            arrays["future_inputs"],
                        )
                        for m in path
                    ]
                )
                check(arrays[objective], predicted)
                checkpoint_replays += len(path)
                scale = np.asarray(source["loss_scales"][objective])
                squared = (predicted - arrays["targets"]) ** 2
                losses = (squared.mean(axis=1) / scale**2).mean(axis=(1, 2))
                check(selected["losses"], losses)
                assert selected["selected_index"] == int(np.argmin(losses))
                for label in sorted(set(arrays["recording_ids"])):
                    mask = arrays["recording_ids"] == label
                    check(
                        selected["recording_losses"][label],
                        (squared[:, mask].mean(axis=1) / scale**2).mean(axis=(1, 2)),
                    )
                    excluded = (squared[:, ~mask].mean(axis=1) / scale**2).mean(
                        axis=(1, 2)
                    )
                    assert selected["leave_one_recording_out_indices"][label] == int(
                        np.argmin(excluded)
                    )
                for regime, ev in eval_arrays.items():
                    assert not set(arrays["recording_ids"]) & set(ev["recording_ids"])
                    chosen = path[selected["selected_index"]]
                    prior_index = source["selected"][objective][objective][
                        "selected_index"
                    ]
                    prior = path[prior_index]
                    actual = metric(
                        recurrence(
                            chosen,
                            ev["past_states"],
                            ev["past_inputs"],
                            ev["future_inputs"],
                        ),
                        ev["targets"],
                        ev["state_scale"],
                    )
                    baseline = metric(
                        recurrence(
                            prior,
                            ev["past_states"],
                            ev["past_inputs"],
                            ev["future_inputs"],
                        ),
                        ev["targets"],
                        ev["state_scale"],
                    )
                    for key in actual:
                        check(variant["regimes"][regime][objective][key], actual[key])
                        check(row["baseline"][regime][objective][key], baseline[key])
                    a, b = (
                        np.asarray(score_vector(baseline)),
                        np.asarray(score_vector(actual)),
                    )
                    detailed.append(
                        dict(
                            phase=row["phase"],
                            data_seed=row["data_seed"],
                            fit_seed=row["fit_seed"],
                            pool_seed=row["pool_seed"],
                            recordings=count,
                            objective=objective,
                            regime=regime,
                            prior_step=prior_index * 100,
                            selected_step=selected["selected_index"] * 100,
                            baseline=a.tolist(),
                            candidate=b.tolist(),
                            absolute_change=(b - a).tolist(),
                            relative_change=(b / a - 1).tolist(),
                        )
                    )
    counts = {}
    for phase in ("known", "confirmation"):
        counts[phase] = {}
        for objective in ("original", "first_step_floor"):
            counts[phase][objective] = {}
            for regime in ("matched", "shifted"):
                counts[phase][objective][regime] = {}
                for count in plan["recording_counts"]:
                    subset = [
                        r
                        for r in detailed
                        if (r["phase"], r["objective"], r["regime"], r["recordings"])
                        == (phase, objective, regime, count)
                    ]
                    entry = {}
                    for name, index in (("first", 0), ("overall", 5)):
                        delta = np.array([r["absolute_change"][index] for r in subset])
                        relative = np.array(
                            [r["relative_change"][index] for r in subset]
                        )
                        entry[name] = dict(
                            improved=int(np.sum(delta < -1e-12)),
                            worse=int(np.sum(delta > 1e-12)),
                            unchanged=int(np.sum(np.abs(delta) <= 1e-12)),
                            relative_change=dict(
                                median=float(np.median(relative)),
                                min=float(relative.min()),
                                max=float(relative.max()),
                            ),
                            absolute_change=dict(
                                median=float(np.median(delta)),
                                min=float(delta.min()),
                                max=float(delta.max()),
                            ),
                        )
                    counts[phase][objective][regime][str(count)] = entry
    # Direct nested comparison distinguishes pool size from replacing the old data.
    identifiers = ("phase", "data_seed", "fit_seed", "pool_seed", "objective", "regime")
    two = {
        tuple(r[k] for k in identifiers): r for r in detailed if r["recordings"] == 2
    }
    nested = []
    for row in detailed:
        if row["recordings"] != 16:
            continue
        reference = two[tuple(row[k] for k in identifiers)]
        a, b = np.asarray(reference["candidate"]), np.asarray(row["candidate"])
        nested.append(
            dict(
                **{k: row[k] for k in identifiers},
                two_recordings=a.tolist(),
                sixteen_recordings=b.tolist(),
                absolute_change=(b - a).tolist(),
                relative_change=(b / a - 1).tolist(),
            )
        )
    write(
        args.output / "summary.json",
        dict(
            counts=counts,
            comparisons=detailed,
            nested_two_to_sixteen=nested,
            score_order=["h1", "h2", "h3", "h4", "h5", "overall"],
            limits=plan["limits"],
        ),
    )
    audit = dict(
        pool_cases=len(rows),
        selected_model_regime_comparisons=len(detailed),
        scored_windows=windows,
        checkpoint_pool_replays=checkpoint_replays,
        numeric_comparisons=comparisons,
        max_absolute_difference=maximum,
        atol=1e-9,
        rtol=1e-8,
        method="Independent pool-window ranking and time indexing, constant per-variant budget, regenerated selection arrays, NumPy replay of all checkpoint paths, all per-recording/leave-one-out choices and paired evaluation scores. Shared generator and source models. Window counts and comparisons reuse paths, pools, and evaluation.",
    )
    write(args.output / "audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
