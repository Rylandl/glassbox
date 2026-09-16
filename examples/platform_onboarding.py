"""Run the same public onboarding workflow for both supported model families.

The bundled synthetic generators provide interface fixtures, not evidence of
hardware onboarding performance. Only fixture generation branches on family.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from glassbox import DynamicsBelief, FitSpec, Holdout, fit
from glassbox.core.data import trajectory_content_digest, trajectory_segment
from glassbox.core.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.core.synthetic import generate_trajectory
from glassbox.workflows.evaluate import evaluate, save_report

GENERATORS = {
    "multirotor": generate_trajectory,
    "fixedwing": generate_fixed_wing_trajectory,
}
HORIZONS_S = (0.1, 0.4)


def run_family(family: str, output: Path, *, fit_steps: int = 20) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    generate = GENERATORS[family]
    flights = [
        replace(
            generate(seed=seed, duration_s=2.0),
            labels={"source_group": f"recording-{seed}"},
        )
        for seed in range(6)
    ]
    # Fit reserves recording 2 for error calibration. The remaining flights
    # are untouched initial evaluation, fresh update, and updated evaluation.
    calibration = flights[:3]
    test_flight, fresh_telemetry, updated_test_flight = flights[3:]
    outcome = fit(
        calibration,
        FitSpec(
            holdout=Holdout.by_group(),
            horizons_s=HORIZONS_S,
            evaluation_horizons_s=HORIZONS_S,
            steps=fit_steps,
        ),
    )
    save_report(outcome.report, output / "fit.json")
    outcome.belief.save(output / "belief.json")
    belief = DynamicsBelief.load(output / "belief.json")
    initial_evaluation = evaluate(
        belief,
        [test_flight],
        horizons_s=HORIZONS_S,
        report_path=output / "evaluation.json",
    )

    # Replay a short observed command sequence for a prediction check. These
    # are commanded values, not privileged measurements of simulator actuators.
    segment = trajectory_segment(test_flight, 25, 45)
    prediction = belief.rollout(
        segment.states[0],
        segment.controls,
        command_history=segment.control_prefix,
        exogenous=segment.exogenous[:-1],
    )
    np.savez_compressed(
        output / "prediction.npz",
        observed_states=segment.states,
        predicted_states=np.asarray(prediction.states),
        predicted_latent_states=np.asarray(prediction.latent_states),
        commands=segment.controls,
        preceding_commands=segment.control_prefix,
        exogenous=segment.exogenous[:-1],
        time_s=segment.time_s,
    )

    updated, update = belief.absorb(fresh_telemetry)
    updated.save(output / "updated-belief.json")
    updated_evaluation = evaluate(
        updated,
        [updated_test_flight],
        horizons_s=HORIZONS_S,
        report_path=output / "updated-evaluation.json",
    )
    # Comparing old and updated models on the same fresh flight isolates the
    # model change. It does not use that flight to choose an update or refit.
    prior_on_updated_test = evaluate(
        belief,
        [updated_test_flight],
        horizons_s=HORIZONS_S,
        report_path=output / "prior-on-updated-test.json",
    )
    summary = {
        "purpose": "synthetic_interface_walkthrough",
        "family": family,
        "configuration_id": belief.input_spec.vehicle.configuration_id,
        "fit_steps": fit_steps,
        "data_roles": {
            "training": [0, 1],
            "forecast_error_calibration": [2],
            "initial_evaluation": [3],
            "update": [4],
            "updated_evaluation": [5],
        },
        "recording_content_sha256": [trajectory_content_digest(f) for f in flights],
        "assumptions": {
            "state_source": "simulator_truth",
            "actuator_initialization": "estimated_from_preceding_commands",
            "hardware_validation": False,
            "independent_recordings": "separate_synthetic_seeds",
        },
        "prediction": {
            "horizon_s": len(segment.controls) * belief.sample_period_s,
            "finite": bool(np.isfinite(prediction.states).all()),
            "parameter_information_rank": prediction.parameter_information_rank,
            "parameter_information_complete": prediction.parameter_information_complete,
            "forecast_error_available": prediction.forecast_error_available,
            "forecast_error_horizon_supported": prediction.forecast_error_horizon_supported,
            "maximum_validity_utilization": float(
                np.max(prediction.validity_utilization)
            ),
        },
        "initial_evaluation": initial_evaluation["model"]["horizon_rollouts"],
        "update": update.to_dict(),
        "updated_model": {
            "update_count": updated.update_count,
            "parameter_distance_since_measurement": updated.parameter_distance_since_measurement,
            "forecast_error_recalibrated": False,
        },
        "prior_on_updated_test": prior_on_updated_test["model"]["horizon_rollouts"],
        "updated_evaluation": updated_evaluation["model"]["horizon_rollouts"],
    }
    save_report(summary, output / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["both", *GENERATORS], default="both")
    parser.add_argument("--output", type=Path, default=Path("artifacts/onboarding"))
    parser.add_argument("--fit-steps", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.fit_steps < 1:
        parser.error("--fit-steps must be positive")
    families = GENERATORS if arguments.family == "both" else [arguments.family]
    for family in families:
        print(f"Running {family} onboarding walkthrough...", flush=True)
        summary = run_family(
            family, arguments.output / family, fit_steps=arguments.fit_steps
        )
        print(
            json.dumps(
                {
                    "family": family,
                    "finite_prediction": summary["prediction"]["finite"],
                    "update_absorbed": summary["update"]["absorbed"],
                    "summary": str(arguments.output / family / "summary.json"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
