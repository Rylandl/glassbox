"""Single-release gate for a dual-control variant: seconds per case.

Fly one or more throw-study cases on a dual-control configuration, with any
config field overridden from the command line, and print the flight summary
and a per-interval table of the controller's decisions.  This is the first
thing to run on a candidate design: a configuration that cannot arrest the
canonical release here has nothing to gain from the seven-case study or the
release ensemble, and the answer arrives in under a minute.

    uv run python scripts/throw_gate.py --variant pass6 --cases canonical
    uv run python scripts/throw_gate.py --variant pass6 --set goal_horizon=declared

Overrides are parsed as JSON where they can be (``--set iteration_count=40``,
``--set clip_action_spread=false``) and as strings otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from glassbox.experimental.dual_control import DUAL_CONTROL_VARIANTS, DualControlConfig
from glassbox.integrations.crazyflow import CrazyflowPlant, CrazyflowPlantConfig
from glassbox.integrations.crazyflow_throw_study import (
    CRAZYFLOW_THROW_STUDY_CASES,
    DUAL_CONTROL_PASS5_MODEL,
    MODEL_ENABLE_DELAY_S,
    ThrowEnsembleConfig,
    _fly_trial,
    build_ensemble_cases,
    tilt_rad,
)


def _parse_override(text: str) -> tuple[str, object]:
    key, value = text.split("=", 1)
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--variant", default="pass6")
    parser.add_argument("--cases", default="canonical")
    parser.add_argument("--set", action="append", default=[], help="key=value override")
    parser.add_argument("--rows", type=int, default=40, help="intervals to tabulate")
    parser.add_argument(
        "--every", type=int, default=1, help="tabulate every n-th interval"
    )
    parser.add_argument(
        "--identifier",
        action="append",
        default=[],
        help="identifier config override, key=value",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=0,
        help="print the n lowest multi-start candidate objectives per row",
    )
    parser.add_argument(
        "--draw",
        type=int,
        default=None,
        help="fly the n-th perturbed release of each case from the ensemble draws",
    )
    args = parser.parse_args()
    overrides = dict(_parse_override(item) for item in args.set)
    identifier_options = dict(_parse_override(item) for item in args.identifier)
    cases = {case.name: case for case in CRAZYFLOW_THROW_STUDY_CASES}
    plant = CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
    merged = (
        dict(DUAL_CONTROL_VARIANTS[args.variant])
        | {"sample_period_s": plant.sample_period_s}
        | overrides
    )
    config = DualControlConfig(**merged)
    from glassbox.experimental.dual_control import DualControlNMPC

    candidate_names = DualControlNMPC(config).candidate_names
    print("config:", args.variant, overrides, "identifier:", identifier_options)
    try:
        for name in args.cases.split(","):
            case = cases[name]
            if args.draw is not None:
                index = [c.name for c in CRAZYFLOW_THROW_STUDY_CASES].index(name)
                draws = build_ensemble_cases(case, index, ThrowEnsembleConfig())
                case = draws[args.draw]
                print(
                    f"  draw {args.draw}: "
                    f"{json.dumps(case.release_perturbation.to_dict())}"
                )
            start = time.perf_counter()
            record, telemetry, _requested, _identification, _trace = _fly_trial(
                case,
                DUAL_CONTROL_PASS5_MODEL,
                plant,
                dual_config=config,
                identifier_options=identifier_options,
            )
            wall = time.perf_counter() - start
            states = telemetry.state_array()
            times = telemetry.timestamp_array()
            enable = round(MODEL_ENABLE_DELAY_S / plant.sample_period_s)
            altitude = states[:, 2]
            tilt = np.asarray([tilt_rad(state) for state in states])
            rate = np.linalg.norm(states[:, 10:13], axis=1)
            speed = np.linalg.norm(states[:, 3:6], axis=1)
            settled = bool(
                (speed[-300:] < 0.1).all()
                and (rate[-300:] < 0.1).all()
                and altitude[-300:].min() > 0.0
            )
            contact = record.floor_contact_step
            outcome = (
                f"FLOOR CONTACT at {times[enable + contact + 1]:.2f} s, trial stopped"
                if contact is not None
                else f"settled hover last 3 s: {settled}"
            )
            print(
                f"\n== {name} wall {wall:.0f}s  min alt {altitude.min():.3f}  "
                f"max tilt after enable {tilt[enable:].max():.2f}  {outcome}"
            )
            print(
                f"  rank4 step {record.command_rank_four_step}  "
                f"first supported {record.first_supported_control_step}"
            )
            print(
                "   k    t   tilt   p     q     r    alt   vz   spd | sel         "
                "cmd                      | track   spread  rate_p  tilt_p  alt_p  it st amp"
            )
            for k, result in enumerate(record.dual_results[: args.rows * args.every]):
                if k % args.every:
                    continue
                i = enable + k
                s = states[i]
                print(
                    f"{k:4d} {times[i]:5.2f} {tilt[i]:5.2f} {s[10]:5.1f} {s[11]:5.1f} "
                    f"{s[12]:5.1f} {altitude[i]:5.2f} {s[5]:5.2f} {speed[i]:5.2f} | "
                    f"{result.selected_candidate:11s} "
                    f"{' '.join(f'{v:.2f}' for v in result.command)} | "
                    f"{result.tracking_cost:8.2e} {result.spread_charge:7.1f} "
                    f"{result.body_rate_penalty:7.1e} {result.tilt_penalty:7.1e} "
                    f"{result.altitude_penalty:6.1e} {result.iterations:2d} "
                    f"{result.status.name[:4]} {result.plan_amplitude:.2f}"
                )
                if args.candidates and len(result.candidate_objectives):
                    order = np.argsort(result.candidate_objectives)[: args.candidates]
                    print(
                        "        candidates: "
                        + "  ".join(
                            f"{candidate_names[i]}={result.candidate_objectives[i]:.3g}"
                            for i in order
                        )
                    )
    finally:
        plant.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
