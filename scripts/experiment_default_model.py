"""Exercise the same opinionated fit/predict/update recipe across existing corpora."""

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
from experiment_transition_diagnosis import load_records
from sequence_transfer_data import load_prepared, write_json

from glassbox.experimental.default_model import _RECIPE, fit
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)
from glassbox.experimental.sequence_model import initialize_sequence_model


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collection(records, configuration, state_names, input_names):
    return SequenceCollection(
        tuple(
            SequenceSegment(r["name"], "prepared", r["states"], r["inputs"], r["dt_s"])
            for r in records
        ),
        configuration_id=configuration,
        state_channels=state_names,
        input_channels=input_names,
    )


def cases(a):
    state_names = tuple(
        [f"velocity_{c} [m/s,NWU]" for c in "xyz"]
        + [f"body_rate_{c} [rad/s,FLU]" for c in "xyz"]
        + [f"rotation_{i}{j} [unitless,FLU-to-NWU]" for i in range(3) for j in range(3)]
    )
    nano = []
    for r in load_records(a.nano):
        path = a.nano / "canonical/train" / f"{r['name']}.npz"
        nano.append(
            dict(
                name=r["name"],
                states=np.column_stack((r["states"], r["context"])),
                inputs=r["controls"],
                dt_s=0.01,
                role="evaluation" if r["role"] == "replication" else "calibration",
                source=str(path.resolve()),
                sha256=sha(path),
            )
        )
    yield (
        "nano",
        nano,
        "idsia_crazyflie_2_1_brushless_flow_v2_ai_deck",
        state_names,
        tuple(f"rotor_{i} recorded speed [rad/s]" for i in range(4)),
    )
    x8 = load_prepared(a.prepared, "x8")
    yield (
        "x8",
        [
            {
                **r,
                "source": str((a.prepared / "x8" / f"{r['name']}.npz").resolve()),
                "sha256": sha(a.prepared / "x8" / f"{r['name']}.npz"),
                "role": "evaluation" if r["role"] == "evaluation" else "calibration",
            }
            for r in x8
        ],
        "ntnu_skywalker_x8_flying_wing",
        state_names,
        (
            "recorded throttle [normalized]",
            "recorded aileron angle [rad]",
            "recorded elevator angle [rad]",
        ),
    )
    arp = load_prepared(a.prepared, "arp")
    for fold in range(4):
        yield (
            f"arp-fold{fold}",
            [
                {
                    **r,
                    "source": str((a.prepared / "arp" / f"{r['name']}.npz").resolve()),
                    "sha256": sha(a.prepared / "arp" / f"{r['name']}.npz"),
                    "role": "evaluation" if i == fold else "calibration",
                }
                for i, r in enumerate(arp)
            ],
            "arp-quad-recording-cohort",
            state_names,
            tuple(
                f"actuator_motors.control[{i}] [normalized command]" for i in range(4)
            ),
        )
    cf = []
    for name in ("log10", "log15", "log16"):
        path = a.cf / f"{name}.npz"
        with np.load(path) as z:
            cf.append(
                dict(
                    name=name,
                    states=z["states"],
                    inputs=z["inputs"],
                    dt_s=0.02,
                    role="evaluation" if name == "log16" else "calibration",
                    source=str(path.resolve()),
                    sha256=sha(path),
                )
            )
    yield (
        "crazyflie-sensor",
        cf,
        "arp-crazyflie-sensor-cohort",
        tuple(
            [f"acc.{c} [g,logger axes]" for c in "xyz"]
            + [f"gyro.{c} [rad/s,logger axes]" for c in "xyz"]
        ),
        tuple(f"motor.m{i} [command/65536]" for i in range(1, 5)),
    )


def grouped_rmse(prediction, target, groups):
    return np.stack(
        [
            np.sqrt(
                np.mean(
                    np.sum(
                        (prediction[:, :, list(g)] - target[:, :, list(g)]) ** 2, -1
                    ),
                    0,
                )
            )
            for g in groups
        ],
        -1,
    )


def run(a):
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded experiments require float64")
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "plan.json", read(a.plan))
    write_json(a.output / "recipe.json", _RECIPE)
    repo = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((repo / "scripts").glob("*.py")),
        *sorted((repo / "src/glassbox").rglob("*.py")),
    ]
    with zipfile.ZipFile(
        a.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sources:
            z.write(p, p.relative_to(repo))
    write_json(
        a.output / "sources.json", {str(p.relative_to(repo)): sha(p) for p in sources}
    )
    write_json(
        a.output / "environment.json",
        dict(
            python=sys.version,
            jax=jax.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
            float64=True,
        ),
    )
    summaries = []
    for case, records, config, states, inputs in cases(a):
        folder = a.output / case
        folder.mkdir()
        calibrated = sorted(
            [r for r in records if r["role"] == "calibration"], key=lambda r: r["name"]
        )
        fresh = calibrated[-1:] if len(calibrated) >= 3 else []
        original = calibrated[:-1] if fresh else calibrated
        evaluated = [r for r in records if r["role"] == "evaluation"]
        write_json(
            folder / "records.json",
            [
                dict(
                    name=r["name"],
                    role="update" if r in fresh else r["role"],
                    source=r["source"],
                    sha256=r["sha256"],
                    rows=len(r["states"]),
                    dt_s=r["dt_s"],
                )
                for r in records
            ],
        )
        np.savez_compressed(
            folder / "records.npz",
            **{
                f"{i}_{k}": r[k]
                for i, r in enumerate(records)
                for k in ("states", "inputs")
            },
        )
        start = time.monotonic()
        model = fit(collection(original, config, states, inputs))
        model.save(folder / "initial.npz")
        initial_seconds = time.monotonic() - start
        models = {"initial": model}
        if fresh:
            start = time.monotonic()
            updated = model.update(collection(fresh, config, states, inputs))
            updated.save(folder / "updated.npz")
            models["updated"] = updated
            update_seconds = time.monotonic() - start
        else:
            update_seconds = None
        eval_collection = collection(evaluated, config, states, inputs)
        windows = eval_collection.extract(
            eval_collection.window_keys(
                history_steps=model.history_steps,
                horizon_steps=model.horizon_steps,
                stride=max(1, int(np.rint(0.1 / model.contract["dt_s"]))),
            ),
            history_steps=model.history_steps,
            horizon_steps=model.horizon_steps,
        )
        b = windows.batch
        truth = b.future_states
        predictions = {
            name: np.asarray(m.predict(b.past_states, b.past_inputs, b.future_inputs))
            for name, m in models.items()
        }
        train = model._train.batch
        init = initialize_sequence_model(
            train,
            kind=_RECIPE["kind"],
            width=_RECIPE["width"],
            seed=_RECIPE["seed"],
            ridge=_RECIPE["ridge_fraction"]
            * len(train.past_states)
            * model.horizon_steps,
        )
        init.save(folder / "initialization.npz")
        predictions["initialization"] = np.asarray(
            init.rollout(b.past_states, b.past_inputs, b.future_inputs)
        )
        predictions["hold"] = np.repeat(b.past_states[:, -1:], model.horizon_steps, 1)
        np.savez_compressed(
            folder / "evaluation.npz",
            **{
                k: getattr(b, k)
                for k in (
                    "past_states",
                    "past_inputs",
                    "future_inputs",
                    "future_states",
                )
            },
            **{f"prediction_{k}": v for k, v in predictions.items()},
        )
        write_json(
            folder / "evaluation-usage.json",
            dict(
                keys=[
                    dict(
                        recording_id=k.recording_id,
                        segment_id=k.segment_id,
                        origin=k.origin,
                    )
                    for k in windows.keys
                ],
                source_origins=windows.source_origins,
                coverage=windows.coverage(),
            ),
        )
        groups = (
            ((0, 1, 2), (3, 4, 5))
            if len(states) == 6
            else ((0, 1, 2), (3, 4, 5), tuple(range(6, 15)))
        )
        report = dict(
            case=case,
            contract=model.contract,
            groups=groups,
            dt_s=model.contract["dt_s"],
            horizon_steps=model.horizon_steps,
            initial_seconds=initial_seconds,
            update_seconds=update_seconds,
            evaluation_windows=len(windows.keys),
            models={
                name: dict(fingerprint=m.fingerprint(), report=m.report)
                for name, m in models.items()
            },
            rmse={
                name: grouped_rmse(p, truth, groups).tolist()
                for name, p in predictions.items()
            },
        )
        write_json(folder / "report.json", report)
        summaries.append(report)
        write_json(a.output / "summary.json", summaries)
        print(
            json.dumps(
                dict(
                    case=case,
                    seconds=round(initial_seconds + (update_seconds or 0), 2),
                    chosen_steps={
                        n: m.report["optimization"]["selected_step"]
                        for n, m in models.items()
                    },
                    final_rmse={n: v[-1][:2] for n, v in report["rmse"].items()},
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--plan", type=Path, default=Path("../artifacts/default-recipe/plan.json")
    )
    p.add_argument(
        "--nano", type=Path, default=Path("../artifacts/real-transition/corpus")
    )
    p.add_argument(
        "--prepared",
        type=Path,
        default=Path("../artifacts/sequence-transfer/prepared-02"),
    )
    p.add_argument(
        "--cf", type=Path, default=Path("../artifacts/representation-study/cf-complete")
    )
    run(p.parse_args())
