"""Identify a hidden Cascade X8 plant and evaluate ordinary reference tracking.

The calibration pilot uses Cascade's published stabilizer and simulator trim.
Glassbox receives canonical observed states and requested commands only. No
Cascade coefficients, applied actuation, or separation states enter its model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import cascade
import jax
import jax.numpy as jnp
import numpy as np
from cascade.canonical import rigid_body_from_canonical, rigid_body_to_canonical
from cascade.control import (
    GuidanceSetpoint,
    cascade_step,
    initial_cascade_state,
    skywalker_x8_controller,
)
from live_refinement import SCORE_SCALES, TrackingPlant, run_trial

from glassbox.belief.belief import DynamicsBelief
from glassbox.core.data import (
    Trajectory,
    load_trajectory_npz,
    save_trajectory_npz,
    trajectory_content_digest,
)
from glassbox.fitting import FitSpec, Holdout, fit
from glassbox.integrations.cascade import CascadePlant, CascadePlantConfig
from glassbox.io.x8_reference import x8_trajectory_spec
from glassbox.workflows.evaluate import evaluate, save_report

DT_S = 0.05
COMMAND_MINIMUM = np.array([0.0, -0.35, -0.35])
COMMAND_MAXIMUM = np.array([1.0, 0.35, 0.35])


def telemetry_spec():
    base = x8_trajectory_spec(trusted_wind=False)
    return replace(
        base,
        observation_source="simulator_truth",
        channels=tuple(
            replace(
                channel,
                minimum=float(low),
                maximum=float(high),
                semantic="normalized_command"
                if channel.role == "throttle"
                else "surface_angle_command",
            )
            for channel, low, high in zip(
                base.controls, COMMAND_MINIMUM, COMMAND_MAXIMUM
            )
        ),
        vehicle=replace(
            base.vehicle,
            configuration_id="cascade_published_x8_cruise",
            fixed_states={
                **base.vehicle.fixed_states,
                "wind_world_m_s": [0.0, 0.0, 0.0],
            },
        ),
    )


def fixture():
    spec = cascade.skywalker_x8_spec()
    model = spec.to_model()
    trim = cascade.trim_straight_flight(
        model, cascade.StraightFlightCondition(airspeed_m_s=18.0, altitude_m=100.0)
    )
    if not trim.success:
        raise RuntimeError(f"Cascade calibration trim failed: {trim.message}")
    state = np.asarray(rigid_body_to_canonical(trim.state.rigid_body), dtype=float)
    command = np.asarray(cascade.control_to_array(trim.control), dtype=float)
    if np.any(command < COMMAND_MINIMUM) or np.any(command > COMMAND_MAXIMUM):
        raise ValueError("trim lies outside the declared application command box")
    plant = CascadePlant(
        CascadePlantConfig(control_frequency_hz=round(1 / DT_S)), spec=spec, model=model
    )
    return plant, trim, state, command


def collect_recording(plant, trim, initial_state, initial_command, seed, duration_s):
    """A known calibration pilot; its output commands are held for one interval."""
    tuned = skywalker_x8_controller()
    pilot = tuned._replace(
        guidance=tuned.guidance._replace(
            pitch_trim=trim.decision[1], throttle_trim=trim.control.propeller[0]
        ),
        rate_period=1,
        attitude_period=1,
        guidance_period=2,
    )
    environment = cascade.standard_environment()
    pilot_state = initial_cascade_state(pilot, trim.state, trim.control)

    @jax.jit
    def command_for(observed, pilot_state, setpoint):
        # The pilot reads only the rigid-body observation. Other fields are
        # unused by cascade_step; they are not sampled from the running plant.
        observed_aircraft = trim.state._replace(
            rigid_body=rigid_body_from_canonical(observed)
        )
        command, updated = cascade_step(
            pilot, pilot_state, setpoint, observed_aircraft, environment, DT_S
        )
        return cascade.control_to_array(command), updated

    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=3)
    sample = plant.reset(initial_state, applied_control=initial_command)
    states, commands = [sample.state.copy()], []
    for index in range(round(duration_s / DT_S)):
        t = index * DT_S
        ramp = min(1.0, t / 1.0)
        setpoint = GuidanceSetpoint(
            airspeed_m_s=jnp.asarray(18.0 + ramp * 0.4 * np.sin(0.7 * t + phases[0])),
            altitude_m=jnp.asarray(100.0 + ramp * 0.5 * np.sin(0.5 * t + phases[1])),
            heading_rad=jnp.asarray(ramp * 0.025 * np.sin(0.6 * t + phases[2])),
        )
        raw, pilot_state = command_for(sample.state, pilot_state, setpoint)
        excitation = (
            ramp
            * np.array([0.035, 0.025, 0.02])
            * np.sin(np.array([1.3, 2.1, 1.7]) * t + phases)
        )
        command = np.clip(
            np.asarray(raw) + np.r_[0.0, initial_command[1:]] + excitation,
            COMMAND_MINIMUM,
            COMMAND_MAXIMUM,
        )
        sample = plant.step(command)
        if not np.isfinite(sample.state).all():
            raise RuntimeError(
                f"nonfinite calibration observation at seed {seed}, interval {index}"
            )
        states.append(sample.state.copy())
        commands.append(command.copy())
    return Trajectory(
        time_s=np.arange(len(states)) * DT_S,
        states=np.asarray(states),
        controls=np.asarray(commands),
        control_prefix=initial_command[None],
        spec=telemetry_spec(),
        labels={"source_group": f"cascade-calibration-{seed}"},
        provenance={
            "plant": "cascade.skywalker_x8",
            "seed": seed,
            "calibration_pilot": "cascade.control.skywalker_x8_controller with trim feedforward",
            "command_policy": "requested command held for the full sample interval",
            "initial_history_assumption": "reset at constant-command actuator equilibrium",
        },
    )


def calibrate(output, *, fit_steps, recording_duration_s):
    (output / "calibration-summary.json").unlink(missing_ok=True)
    plant, trim, state, command = fixture()
    save_report(
        {
            "cascade": cascade.stamp(plant.spec, plant.model),
            "initial_state": state.tolist(),
            "initial_command": command.tolist(),
            "trim_balance_residual": np.asarray(trim.residual).tolist(),
            "calibration_pilot": "published X8 stabilizer with simulator-derived trim",
            "state_source": "simulator_truth",
            "plant_internals_visible_to_glassbox": False,
            "configuration": telemetry_spec().to_dict(),
            "data_roles": {
                "fit": [0, 1],
                "forecast_error_calibration": [2],
                "final_evaluation": [3],
            },
            "calibration_duration_s": 3 * recording_duration_s,
            "evaluation_duration_s": recording_duration_s,
            "fit_steps": fit_steps,
        },
        output / "platform.json",
    )
    recordings = []
    for seed in range(4):
        flight = collect_recording(
            plant, trim, state, command, seed, recording_duration_s
        )
        save_trajectory_npz(flight, output / f"recording-{seed}.npz")
        recordings.append(flight)
        print(
            json.dumps(
                {
                    "recording": seed,
                    "altitude_range_m": [
                        float(flight.states[:, 2].min()),
                        float(flight.states[:, 2].max()),
                    ],
                    "maximum_rate_rad_s": float(
                        np.linalg.norm(flight.states[:, 10:13], axis=1).max()
                    ),
                }
            ),
            flush=True,
        )
    print(
        "Fitting Glassbox from recordings 0 and 1; measuring errors on 2...", flush=True
    )
    outcome = fit(
        recordings[:3],
        FitSpec(
            holdout=Holdout.by_group(),
            steps=fit_steps,
            horizons_s=(0.1, 0.4),
            evaluation_horizons_s=(0.1, 0.4, 0.8),
        ),
    )
    outcome.belief.save(output / "initial-belief.json")
    save_report(outcome.report, output / "fit.json")
    report = evaluate(
        output / "initial-belief.json",
        [recordings[3]],
        horizons_s=(0.1, 0.4, 0.8),
        report_path=output / "held-out-evaluation.json",
    )
    save_report(
        {
            "recording_content_sha256": [
                trajectory_content_digest(f) for f in recordings
            ],
            "initial_belief_sha256": hashlib.sha256(
                (output / "initial-belief.json").read_bytes()
            ).hexdigest(),
            "held_out_evaluation": report,
        },
        output / "calibration-summary.json",
    )
    return outcome.belief, recordings[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cascade-refinement")
    )
    parser.add_argument("--fit-steps", type=int, default=200)
    parser.add_argument("--recording-duration-s", type=float, default=8.0)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--reuse-calibration", action="store_true")
    args = parser.parse_args()
    if (
        args.fit_steps < 1
        or args.repeats < 1
        or not np.isfinite(args.recording_duration_s)
        or args.recording_duration_s < 2
        or not np.isfinite(args.duration_s)
        or args.duration_s < 6
    ):
        parser.error(
            "positive fit steps, recordings at least 2 s, and tracking at least 6 s required"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.json", "audit.json"):
        (args.output / name).unlink(missing_ok=True)
    archive_sources(args.output)
    if args.reuse_calibration:
        belief, warmup = load_calibration(args.output)
    else:
        belief, warmup = calibrate(
            args.output,
            fit_steps=args.fit_steps,
            recording_duration_s=args.recording_duration_s,
        )
    if not args.calibration_only:
        run_comparison(
            belief,
            warmup,
            args.output,
            duration_s=args.duration_s,
            repeats=args.repeats,
        )


def archive_sources(output):
    root = Path(__file__).resolve().parents[1]
    sources = [
        (root / "src/glassbox", "glassbox/src/glassbox"),
        (Path(cascade.__file__).resolve().parent, "cascade/src/cascade"),
    ]
    files = [
        (Path(__file__), "examples/cascade_refinement.py"),
        (Path(__file__).with_name("live_refinement.py"), "examples/live_refinement.py"),
    ]
    for directory, prefix in sources:
        files.extend(
            (path, f"{prefix}/{path.relative_to(directory)}")
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".py", ".toml"}
        )
    manifest = {}
    for source, relative in files:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    save_report(manifest, output / "source/manifest.json")


def load_calibration(output):
    summary = json.loads((output / "calibration-summary.json").read_text())
    artifact = output / "initial-belief.json"
    if (
        hashlib.sha256(artifact.read_bytes()).hexdigest()
        != summary["initial_belief_sha256"]
    ):
        raise ValueError("calibration belief does not match its completed summary")
    flights = [load_trajectory_npz(output / f"recording-{i}.npz") for i in range(4)]
    if [trajectory_content_digest(f) for f in flights] != summary[
        "recording_content_sha256"
    ]:
        raise ValueError("calibration recordings do not match their completed summary")
    return DynamicsBelief.load(artifact), flights[0]


def tracking_plant(initial_state, initial_command):
    plant = CascadePlant(CascadePlantConfig(control_frequency_hz=round(1 / DT_S)))
    plant.reset(initial_state, applied_control=initial_command)
    plant.step(initial_command)  # Prewarm before the trial's clock starts.
    plant.reset(initial_state, applied_control=initial_command)

    def reference(times):
        states = np.tile(initial_state, (len(times), 1))
        states[:, :3] += times[:, None] * initial_state[3:6]
        states[:, 2] += 0.15 * (1 - np.cos(0.5 * times))
        states[:, 5] += 0.075 * np.sin(0.5 * times)
        return states

    return TrackingPlant(
        initial_state.copy(),
        initial_command.copy(),
        lambda command: plant.step(command).state,
        reference,
        "cascade.skywalker_x8",
    )


def run_comparison(belief, warmup, output, *, duration_s, repeats):
    metadata = json.loads((output / "platform.json").read_text())
    current = cascade.spec_hash(cascade.skywalker_x8_spec())
    if current != metadata["cascade"]["spec_hash"]:
        raise ValueError("Cascade aircraft specification changed since calibration")
    current_commit = cascade.stamp()["git_commit"]
    if current_commit != metadata["cascade"]["git_commit"]:
        raise ValueError("Cascade source revision changed since calibration")
    if telemetry_spec().prediction_spec() != belief.input_spec.prediction_spec():
        raise ValueError(
            "calibration telemetry contract does not match this experiment"
        )
    initial_state = np.asarray(metadata["initial_state"])
    initial_command = np.asarray(metadata["initial_command"])
    reserved = load_trajectory_npz(output / "recording-3.npz")
    comparisons = []
    for repetition in range(repeats):
        trials = {}
        for mode in (
            ("frozen", "adopting") if repetition % 2 == 0 else ("adopting", "frozen")
        ):
            path = output / f"fixedwing-{repetition}" / mode
            print(
                f"Tracking Cascade X8: repetition {repetition}, {mode}...", flush=True
            )
            trials[mode] = run_trial(
                belief,
                lambda: tracking_plant(initial_state, initial_command),
                warmup,
                path,
                duration_s=duration_s,
                adopting=mode == "adopting",
                injected_delay_s=1.0,
                interruption_s=(duration_s * 0.45, duration_s * 0.45 + 0.3),
            )
            report = trials[mode]
            active = report["worker"]["acknowledged_active"]
            artifact = path / f"belief-{active['update_count']:04d}.json"
            evaluate(
                artifact,
                [reserved],
                horizons_s=(0.1, 0.4, 0.8),
                report_path=path / "active-held-out-evaluation.json",
            )
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "tracking_rmse": report["tracking_rmse"],
                        "adoptions": len(report["applied_adoptions"]),
                        "worker_error": report["worker"]["error"],
                    }
                ),
                flush=True,
            )
        comparisons.append(
            {
                "family": "fixedwing",
                "variant": repetition,
                "variant_semantics": "repetition of the same X8 plant",
                "trials": trials,
            }
        )
        save_report(
            {
                "purpose": "cascade_hidden_plant_refinement_experiment",
                "calibration": metadata,
                "gate": {
                    "minimum_relative_prediction_gain": 0.02,
                    "maximum_headline_regression": 0.05,
                    "normalized_metric_scales": SCORE_SCALES,
                    "maximum_score_age_intervals": 100,
                },
                "comparisons": comparisons,
            },
            output / "comparison.json",
        )


if __name__ == "__main__":
    main()
