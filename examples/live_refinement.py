"""Paced tracking while a background worker refines a platform model.

Both families use the same transport, learner, controller handoff and scoring.
Plant parameters are fixed during each trial. This is a synthetic interface
experiment with truth state observations, not a hardware or real-time claim.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox import FitSpec, Holdout, fit
from glassbox.control.fitted import NMPCController, default_solver_policy
from glassbox.control.plan import ReferenceTrajectory
from glassbox.core.data import save_trajectory_npz, trajectory_segment
from glassbox.core.dynamics import (
    control_state_after_history,
    fixed_wing_trim_control,
    hover_control,
    latent_response_time_constants,
    step_with_latent,
    with_response_time_constant,
)
from glassbox.core.fixedwing_synthetic import (
    fixed_wing_trim_state,
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.metrics import state_rmse_metrics
from glassbox.core.synthetic import generate_trajectory, resting_state, true_parameters
from glassbox.workflows.evaluate import save_report
from glassbox.workflows.refinement import ModelRefiner, ModelRevision
from glassbox.workflows.streaming import RefinementWorker, TransitionBuffer

DT_S = 0.05
BLOCK_STEPS = 4
HISTORY_STEPS = 20
GENERATORS = {
    "multirotor": generate_trajectory,
    "fixedwing": generate_fixed_wing_trajectory,
}
SCORE_SCALES = {
    "position_rmse_m": 0.1,
    "velocity_rmse_m_s": 0.5,
    "attitude_rmse_deg": 5.0,
    "angular_velocity_rmse_rad_s": 0.5,
}


@dataclass(frozen=True)
class TrackingPlant:
    """Consumer-side experiment seam: observations, commands and a reference.

    Construct a fresh, prewarmed plant for each trial. The advance callback
    holds a command for DT_S and returns the next observed rigid-body state.
    Internal plant parameters and latent state stay inside that callback.
    """

    initial_state: np.ndarray
    initial_command: np.ndarray
    advance: Callable[[np.ndarray], np.ndarray]
    reference: Callable[[np.ndarray], np.ndarray]
    source: str


def reference_states(family, times):
    states = np.zeros((len(times), 13))
    states[:, 6] = 1.0
    if family == "multirotor":
        states[:, 0] = 0.08 * np.sin(0.7 * times)
        states[:, 3] = 0.056 * np.cos(0.7 * times)
        states[:, 2] = 0.04 * np.sin(0.5 * times)
        states[:, 5] = 0.02 * np.cos(0.5 * times)
    else:
        states[:, 0] = 15.0 * times
        states[:, 3] = 15.0
        states[:, 2] = 0.05 * np.sin(0.5 * times)
        states[:, 5] = 0.025 * np.cos(0.5 * times)
    return states


def trim_command(belief):
    return np.asarray(
        hover_control(belief.params)
        if belief.input_spec.vehicle.family == "multirotor"
        else fixed_wing_trim_control(
            belief.params, 15.0, belief.input_spec.control_roles
        )
    )


def controller_for(revision: ModelRevision):
    # This comparison isolates changes to the fitted mean. Full belief evidence
    # remains in the learner/gate reports. The application explicitly permits
    # partial parameter information; stock controller defaults are unchanged.
    return NMPCController(
        revision.belief.model,
        policy=replace(
            default_solver_policy(revision.belief),
            maximum_iterations=4,
            allow_unresolved_parameters=True,
        ),
    )


def warm_controller(controller, state, history, source_time_s, reference_fn):
    times = source_time_s + np.arange(controller.prediction_steps + 1) * DT_S
    reference = ReferenceTrajectory(reference_fn(times))
    latent = controller.model.initial_latent_state(history)
    result = controller.solve(state, reference, history[-1], latent_state=latent)
    result = controller.solve(
        state, reference, history[-1], latent_state=latent, warm_start=result.warm_start
    )
    np.asarray(result.predicted_states)  # Complete asynchronous compilation/work.


def prediction_gate(result):
    active, candidate = result.active_score.rmse, result.candidate_score.rmse
    if active is None or candidate is None:
        return False
    if (
        not np.isfinite(result.candidate_score.prediction.validity_utilization).all()
        or np.max(result.candidate_score.prediction.validity_utilization) > 1.0
    ):
        return False
    old = np.asarray([active[key] / scale for key, scale in SCORE_SCALES.items()])
    new = np.asarray([candidate[key] / scale for key, scale in SCORE_SCALES.items()])
    return bool(
        np.linalg.norm(new) <= 0.98 * np.linalg.norm(old)
        and np.all(new <= 1.05 * old + 1e-6)
    )


def calibrate(family, variant, output, fit_steps):
    output.mkdir(parents=True, exist_ok=True)
    base = true_parameters() if family == "multirotor" else true_fixed_wing_parameters()
    thrust_scale, lag_scale = ((0.90, 0.80), (1.10, 1.25))[variant]
    plant = with_response_time_constant(
        base._replace(log_thrust_accel=base.log_thrust_accel + jnp.log(thrust_scale)),
        0.08 * lag_scale,
    )
    configuration_id = f"streaming_{family}_{variant}"
    calibration = []
    for seed in range(3):
        flight = GENERATORS[family](seed=seed, duration_s=3.0, dt_s=DT_S, params=plant)
        flight = replace(
            flight,
            spec=replace(
                flight.spec,
                vehicle=replace(flight.spec.vehicle, configuration_id=configuration_id),
            ),
            labels={"source_group": f"calibration-{seed}"},
        )
        calibration.append(flight)
        save_trajectory_npz(flight, output / f"calibration-{seed}.npz")
    outcome = fit(
        calibration,
        FitSpec(
            holdout=Holdout.by_group(),
            steps=fit_steps,
            horizons_s=(0.1, 0.4),
            evaluation_horizons_s=(0.1, 0.4),
        ),
    )
    outcome.belief.save(output / "initial-belief.json")
    save_report(outcome.report, output / "fit.json")
    save_report(
        {
            "family": family,
            "configuration_id": configuration_id,
            "thrust_scale": thrust_scale,
            "actuator_lag_scale": lag_scale,
            "parameters_fixed_during_trial": True,
            "calibration_duration_s": 9.0,
            "state_source": "simulator_truth",
            "calibration_uses_known_stabilizer": True,
        },
        output / "platform.json",
    )
    return outcome.belief, plant, calibration[0]


def synthetic_plant(belief, plant_params):
    family = belief.input_spec.vehicle.family
    state = resting_state() if family == "multirotor" else fixed_wing_trim_state()
    command = trim_command(belief)
    plant_step = jax.jit(
        lambda state, latent, command: step_with_latent(
            plant_params, state, latent, command, DT_S, belief.input_spec.control_roles
        )
    )
    latent = control_state_after_history(
        plant_params,
        jnp.asarray(command)[None],
        DT_S,
        belief.input_spec.control_roles,
    )
    jax.block_until_ready(plant_step(state, latent, command))

    def advance(command):
        nonlocal state, latent
        state, latent = plant_step(state, latent, command)
        state = np.asarray(state)
        return state.copy()

    return TrackingPlant(
        state.copy(),
        command.copy(),
        advance,
        lambda times: reference_states(family, times),
        "glassbox_synthetic",
    )


def run_trial(
    belief,
    plant_factory,
    warmup_flight,
    output,
    *,
    duration_s,
    adopting,
    injected_delay_s,
    interruption_s,
):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").unlink(missing_ok=True)
    family = belief.input_spec.vehicle.family
    steps = round(duration_s / DT_S)
    startup = time.monotonic()
    plant = plant_factory()
    state = plant.initial_state.copy()
    previous_command = plant.initial_command.copy()
    command_history = deque(
        [previous_command.copy()] * HISTORY_STEPS, maxlen=HISTORY_STEPS
    )
    buffer = TransitionBuffer(
        belief.input_spec,
        DT_S,
        recording_id="tracking",
        block_steps=BLOCK_STEPS,
        history_steps=HISTORY_STEPS,
    )
    journal = (output / "events.jsonl").open("w")
    worker = None

    def event(record):
        if record["kind"] == "block":
            revision = worker.refiner.candidate
            path = f"belief-{revision.belief.update_count:04d}.json"
            revision.belief.save(output / path)
            record["candidate_artifact"] = path
        journal.write(json.dumps(record, allow_nan=False) + "\n")
        journal.flush()

    def prepare(revision, block):
        controller = controller_for(revision)
        history = np.concatenate((block.control_prefix, block.controls))[
            -HISTORY_STEPS:
        ]
        warm_controller(
            controller,
            block.states[-1],
            history,
            block.provenance["source_time_end_s"],
            plant.reference,
        )
        return controller

    # Both arms run a learner, so their compute topology and injected faults
    # match. Only the adopting arm prepares/offers replacement controllers.
    worker = RefinementWorker(
        belief,
        history_steps=HISTORY_STEPS,
        block_steps=BLOCK_STEPS,
        queue_capacity=2,
        retained_blocks=2,
        prepare=prepare if adopting else None,
        should_offer=prediction_gate if adopting else None,
        on_event=event,
        delay_s=lambda index: injected_delay_s if index == 0 else 0.0,
    )
    active = worker.initial_revision
    active.belief.save(output / "belief-0000.json")
    controller = controller_for(active)
    warm_controller(
        controller, state, np.asarray(command_history), 0.0, plant.reference
    )
    # Compile the learning block shapes on calibration data in a disposable
    # coordinator. No warmup updates or scores are transferred into the trial.
    scratch = ModelRefiner(
        belief,
        history_steps=HISTORY_STEPS,
        retained_blocks=2,
        propagate_parameter_covariance=False,
    )
    scratch.skip(
        recording_id="warmup",
        start_interval=0,
        stop_interval=HISTORY_STEPS,
        reason="compile shapes on calibration data",
    )
    for i in range(HISTORY_STEPS, HISTORY_STEPS + 2 * BLOCK_STEPS, BLOCK_STEPS):
        segment = trajectory_segment(warmup_flight, i, i + BLOCK_STEPS)
        segment = replace(
            segment, control_prefix=segment.control_prefix[-HISTORY_STEPS:]
        )
        scratch.observe(segment, recording_id="warmup", start_interval=i)
    startup_s = time.monotonic() - startup
    states = [state.copy()]
    commands, revisions, tick_times, lags, solve_times, submit_times = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    adoptions = []
    fallback_count = unavailable_intervals = rejected_blocks = 0
    warm_start = None
    started = time.monotonic()
    worker.start()
    failure = None
    try:
        for index in range(steps):
            scheduled = started + index * DT_S
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            tick = time.monotonic()
            lags.append(max(0.0, tick - scheduled))
            offer = worker.poll_offer()
            if offer is not None:
                applied = (
                    offer.expected_active_revision == active.revision_id
                    and 0 <= index - offer.scored_stop_interval <= 100
                )
                if applied:
                    active, controller = offer.revision, offer.controller
                    warm_start = None
                    adoptions.append(
                        {
                            "interval": index,
                            "revision_id": active.revision_id,
                            "scored_stop_interval": offer.scored_stop_interval,
                        }
                    )
                worker.acknowledge(offer, applied=applied)
            future = (index + np.arange(controller.prediction_steps + 1)) * DT_S
            reference = ReferenceTrajectory(plant.reference(future))
            # Reconstruct from commanded history under this exact model revision.
            latent_estimate = controller.model.initial_latent_state(
                np.asarray(command_history)
            )
            result = controller.solve(
                state,
                reference,
                previous_command,
                latent_state=latent_estimate,
                warm_start=warm_start,
                deadline_s=DT_S,
            )
            command = np.asarray(result.command)
            solve_times.append(result.diagnostics.solve_time_s)
            fallback_count += int(result.used_fallback)
            warm_start = result.warm_start
            next_state = np.asarray(plant.advance(command))
            if not np.isfinite(next_state).all():
                failure = "nonfinite plant state"
                break
            # State observations disappear for this source-time interval; applied
            # commands remain available from the local command log.
            available = not (interruption_s[0] <= index * DT_S < interruption_s[1])
            unavailable_intervals += int(not available)
            begin_submit = time.monotonic()
            block = buffer.push(
                interval=index,
                source_time_s=index * DT_S,
                command=command,
                state=state if available else None,
                next_state=next_state if available else None,
                received_at_s=time.monotonic(),
            )
            if block is not None:
                rejected_blocks += int(not worker.submit(block))
            submit_times.append(time.monotonic() - begin_submit)
            states.append(next_state.copy())
            commands.append(command.copy())
            revisions.append(active.revision_id)
            command_history.append(command.copy())
            previous_command, state = command, next_state
            tick_times.append(time.monotonic() - tick)
    finally:
        control_elapsed_s = time.monotonic() - started
        stopped = worker.close(timeout_s=30.0)
        if stopped:
            journal.close()
    if not stopped:
        raise TimeoutError("refinement worker did not stop within the shutdown budget")
    observed = np.asarray(states)
    applied = np.asarray(commands)
    times = np.arange(len(states)) * DT_S
    reference = plant.reference(times)
    metrics = state_rmse_metrics(observed[1:], reference[1:]) if len(commands) else None
    minimum, maximum = (
        np.asarray(controller.model.command_minimum),
        np.asarray(controller.model.command_maximum),
    )
    bound_violation = (
        max(0.0, float(np.max(minimum - applied)), float(np.max(applied - maximum)))
        if len(commands)
        else 0.0
    )
    np.savez_compressed(
        output / "tracking.npz",
        time_s=times,
        states=observed,
        reference_states=reference,
        commands=applied,
        revision_ids=np.asarray(revisions),
        tick_times_s=tick_times,
        source_clock_lags_s=lags,
        solve_times_s=solve_times,
        telemetry_submit_times_s=submit_times,
    )
    report = {
        "family": family,
        "configuration_id": belief.input_spec.vehicle.configuration_id,
        "mode": "adopting" if adopting else "frozen",
        "completed_intervals": len(commands),
        "requested_intervals": steps,
        "failure": failure,
        "worker": worker.summary(),
        "tracking_rmse": metrics,
        "maximum_command_bound_violation": bound_violation,
        "fallback_count": fallback_count,
        "deadline_misses": int(np.sum(np.asarray(tick_times) > DT_S)),
        "solve_deadline_misses": int(np.sum(np.asarray(solve_times) > DT_S)),
        "maximum_source_clock_lag_s": max(lags, default=0.0),
        "maximum_tick_time_s": max(tick_times, default=0.0),
        "maximum_telemetry_submit_time_s": max(submit_times, default=0.0),
        "startup_s": startup_s,
        "control_elapsed_s": control_elapsed_s,
        "applied_adoptions": adoptions,
        "unavailable_state_intervals": unavailable_intervals,
        "buffer_discarded_intervals": buffer.discarded_intervals,
        "partial_intervals_at_shutdown": buffer.partial_intervals,
        "rejected_blocks": rejected_blocks,
        "initialization": {
            "history_steps": HISTORY_STEPS,
            "initial_model_actuator_memory_fraction": float(
                np.max(
                    np.exp(
                        -HISTORY_STEPS
                        * DT_S
                        / np.asarray(latent_response_time_constants(belief.params))
                    )
                )
            ),
        },
        "assumptions": {
            "plant_source": plant.source,
            "state_source": "simulator_truth",
            "plant_latent_visible_to_controller": False,
            "controller_uses": "fitted_mean_only",
            "allow_unresolved_parameters": True,
            "forecast_error_recalibrated": False,
            "forgetting": False,
            "simulation_clock": "paced_fixed_steps; plant time slows if control misses deadlines",
            "wall_times_are_host_specific": True,
            "adoption_evidence": "one subsequent block; not an independent flight holdout",
        },
    }
    save_report(report, output / "summary.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["both", *GENERATORS], default="both")
    parser.add_argument("--variants", type=int, choices=(1, 2), default=2)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--fit-steps", type=int, default=10)
    parser.add_argument("--learner-delay-s", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/live-refinement")
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.duration_s)
        or args.duration_s < 6
        or args.fit_steps < 1
        or not np.isfinite(args.learner_delay_s)
        or args.learner_delay_s < 0
    ):
        parser.error(
            "duration must be at least 6 seconds, fit steps positive, and learner delay finite/nonnegative"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").unlink(missing_ok=True)
    comparisons = []
    for family in GENERATORS if args.family == "both" else [args.family]:
        for variant in range(args.variants):
            output = args.output / f"{family}-{variant}"
            print(f"Calibrating {family} variant {variant}...", flush=True)
            belief, plant, warmup = calibrate(family, variant, output, args.fit_steps)
            results = {}
            # Alternate order across variants. Each arm has the same learner
            # topology and source-time faults; scheduling remains host-dependent.
            for mode in (
                ("frozen", "adopting") if variant == 0 else ("adopting", "frozen")
            ):
                print(f"Running {family}-{variant} {mode}...", flush=True)
                results[mode] = run_trial(
                    belief,
                    lambda belief=belief, plant=plant: synthetic_plant(belief, plant),
                    warmup,
                    output / mode,
                    duration_s=args.duration_s,
                    adopting=mode == "adopting",
                    injected_delay_s=args.learner_delay_s,
                    interruption_s=(
                        args.duration_s * 0.45,
                        args.duration_s * 0.45 + 0.3,
                    ),
                )
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "steps": results[mode]["completed_intervals"],
                            "adoptions": len(results[mode]["applied_adoptions"]),
                            "worker_error": results[mode]["worker"]["error"],
                        }
                    ),
                    flush=True,
                )
            comparisons.append(
                {"family": family, "variant": variant, "trials": results}
            )
            save_report(
                {
                    "purpose": "synthetic_streaming_interface_experiment",
                    "gate": {
                        "minimum_relative_prediction_gain": 0.02,
                        "maximum_headline_regression": 0.05,
                        "normalized_metric_scales": SCORE_SCALES,
                        "maximum_score_age_intervals": 100,
                    },
                    "comparisons": comparisons,
                },
                args.output / "comparison.json",
            )


if __name__ == "__main__":
    main()
