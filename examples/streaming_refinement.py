"""Replay canonical recordings through score-before-learn model refinement.

With no input paths, generate separate synthetic calibration and replay
recordings for both model families. Supply --belief and --recordings to replay
existing artifacts. No commands are sent to a vehicle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from glassbox import DynamicsBelief, FitSpec, Holdout, fit
from glassbox.core.data import (
    load_trajectory_npz,
    save_trajectory_npz,
    trajectory_content_digest,
    trajectory_segment,
)
from glassbox.core.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.core.synthetic import generate_trajectory
from glassbox.workflows.evaluate import save_report
from glassbox.workflows.refinement import ModelRefiner, ModelRevision

GENERATORS = {
    "multirotor": generate_trajectory,
    "fixedwing": generate_fixed_wing_trajectory,
}


def replay(
    belief: DynamicsBelief,
    recordings: list[Path],
    output: Path,
    *,
    block_steps: int,
    adopt_between_recordings: bool = False,
) -> dict:
    if block_steps < 1:
        raise ValueError("block_steps must be positive")
    output.mkdir(parents=True, exist_ok=True)
    refiner = ModelRefiner(belief)
    # A failed rerun must not leave an old summary pointing to overwritten traces.
    summary_path = output / "summary.json"
    summary_path.unlink(missing_ok=True)
    artifacts: dict[str, str] = {}

    def save_revision(revision: ModelRevision) -> None:
        if revision.revision_id not in artifacts:
            path = Path(f"revision-{len(artifacts):04d}.json")
            revision.belief.save(output / path)
            artifacts[revision.revision_id] = str(path)

    save_revision(refiner.active)
    sources = []
    for recording_index, path in enumerate(recordings):
        recording = load_trajectory_npz(path)
        recording_id = f"recording-{recording_index}"
        sources.append(
            {
                "recording_id": recording_id,
                "path": str(path),
                "content_sha256": trajectory_content_digest(recording),
            }
        )
        for start in range(0, len(recording.controls), block_steps):
            block = trajectory_segment(
                recording, start, min(start + block_steps, len(recording.controls))
            )
            result = refiner.observe(
                block, recording_id=recording_id, start_interval=start
            )
            save_revision(result.candidate_after)
            np.savez_compressed(
                output / f"block-{len(refiner.results) - 1:04d}.npz",
                observed_states=block.states,
                active_states=np.asarray(result.active_score.prediction.states),
                candidate_states=np.asarray(result.candidate_score.prediction.states),
                commands=block.controls,
                preceding_commands=(
                    block.control_prefix
                    if block.control_prefix is not None
                    else np.empty((0, block.controls.shape[1]))
                ),
                exogenous=block.exogenous[:-1],
                time_s=block.time_s,
                active_revision_id=result.active_score.revision.revision_id,
                candidate_revision_id=result.candidate_score.revision.revision_id,
            )
        if adopt_between_recordings and recording_index < len(recordings) - 1:
            # An explicit offline exercise of the adoption interface, with no
            # performance threshold and no hardware-readiness implication.
            # The last block scored candidate_score, not candidate_after.
            refiner.adopt(
                result.candidate_score.revision.revision_id,
                expected_active_revision=result.active_score.revision.revision_id,
                reason="explicit between-recordings replay exercise; no performance gate",
            )
    report = {
        "format_version": 1,
        "purpose": "recorded_telemetry_refinement_walkthrough",
        "session_id": refiner.session_id,
        "family": belief.input_spec.vehicle.family,
        "configuration_id": belief.input_spec.vehicle.configuration_id,
        "block_steps": block_steps,
        "scoring": {
            "protocol": "block_rollout_rmse_excluding_initial_state",
            "commands_and_exogenous_inputs": "observed_during_block",
            "state_initialization": "first_observed_state_of_each_block",
            "actuator_initialization": "estimated_from_full_preceding_command_history",
            "evaluation_before_absorption": True,
            "independent_flight_holdout": False,
            "forecast_error_recalibrated": False,
            "forgets_old_information": False,
            "live_control": False,
        },
        "sources": sources,
        "revision_artifacts": artifacts,
        "blocks": [result.to_dict() for result in refiner.results],
        "adoptions": [adoption.to_dict() for adoption in refiner.adoptions],
        "final_active": refiner.active.to_dict(),
        "final_candidate": refiner.candidate.to_dict(),
    }
    save_report(report, summary_path)
    return report


def synthetic_inputs(family: str, output: Path, *, fit_steps: int):
    output.mkdir(parents=True, exist_ok=True)
    generate = GENERATORS[family]
    calibration = [
        replace(
            generate(seed=seed, duration_s=2.0),
            labels={"source_group": f"calibration-{seed}"},
        )
        for seed in range(3)
    ]
    outcome = fit(
        calibration,
        FitSpec(
            holdout=Holdout.by_group(),
            horizons_s=(0.1, 0.4),
            evaluation_horizons_s=(0.1, 0.4),
            steps=fit_steps,
        ),
    )
    save_report(outcome.report, output / "fit.json")
    outcome.belief.save(output / "initial-belief.json")
    recordings = []
    for seed in (6, 7):
        path = output / f"recording-seed-{seed}.npz"
        save_trajectory_npz(generate(seed=seed, duration_s=2.0), path)
        recordings.append(path)
    return DynamicsBelief.load(output / "initial-belief.json"), recordings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--family", choices=["both", *GENERATORS], default="both")
    source.add_argument("--belief", type=Path)
    parser.add_argument("--recordings", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/refinement"))
    parser.add_argument("--fit-steps", type=int, default=20)
    parser.add_argument("--block-steps", type=int, default=20)
    parser.add_argument("--adopt-between-recordings", action="store_true")
    args = parser.parse_args()
    if args.block_steps < 1 or args.fit_steps < 1:
        parser.error("--block-steps and --fit-steps must be positive")
    if bool(args.belief) != bool(args.recordings):
        parser.error("--belief and --recordings must be supplied together")
    families = GENERATORS if args.family == "both" else [args.family]
    for family in ["recorded"] if args.belief else families:
        output = args.output if args.belief else args.output / family
        print(f"Running {family} refinement replay...", flush=True)
        belief, recordings = (
            (DynamicsBelief.load(args.belief), args.recordings)
            if args.belief
            else synthetic_inputs(family, output, fit_steps=args.fit_steps)
        )
        report = replay(
            belief,
            recordings,
            output,
            block_steps=args.block_steps,
            adopt_between_recordings=args.adopt_between_recordings,
        )
        print(
            json.dumps(
                {
                    "family": report["family"],
                    "blocks": len(report["blocks"]),
                    "absorbed": sum(
                        block["update"]["absorbed"] for block in report["blocks"]
                    ),
                    "adoptions": len(report["adoptions"]),
                    "summary": str(output / "summary.json"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
