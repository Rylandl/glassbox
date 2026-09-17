"""Ordinary cruise tracking with a frozen generic learner and matched predictors.

This is an offline research benchmark, not an operational flight controller.
The calibration pilot is platform-specific; the learned recipe is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import cascade
import jax
import jax.numpy as jnp
import numpy as np
from cascade.canonical import rigid_body_from_canonical, rigid_body_to_canonical
from cascade.initialization import (
    control_from_array,
    equilibrate_internal_state,
    zero_state,
)
from cascade.integration import repeat_control, rollout
from cascade_refinement import (
    COMMAND_MAXIMUM,
    COMMAND_MINIMUM,
    DT_S,
    collect_recording,
    fixture,
)

from glassbox import fit
from glassbox._sequence_model import sequence_windows
from glassbox.core.data import save_trajectory_npz
from glassbox.core.geometry import quaternion_to_rotation_batch, state_plus_tangent
from glassbox.recordings import (
    SequenceCollection,
    SequenceSegment,
)

CONTROLLER = dict(
    horizon_steps=5,
    lookahead_s=1.5,
    position_scale_m=0.5,
    forward_speed_scale_m_s=2.0,
    rate_weight=0.1,
    rotation_weight=0.2,
    command_weight=0.03,
    command_change_weight=0.05,
    damping=0.001,
    iterations=3,
    line_search=[1.0, 0.5, 0.25, 0.0],
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def observation(state):
    state = jnp.asarray(state)
    rotation = quaternion_to_rotation_batch(state[..., 6:10])
    return jnp.concatenate(
        (state[..., 3:6], state[..., 10:13], rotation.reshape((*state.shape[:-1], 9))),
        axis=-1,
    )


def reference(times):
    return (
        jnp.stack(
            (18.0 * times, jnp.sin(0.35 * times), 100 + 0.75 * jnp.sin(0.3 * times)), -1
        ),
        jnp.stack(
            (
                jnp.full_like(times, 18.0),
                0.35 * jnp.cos(0.35 * times),
                0.225 * jnp.cos(0.3 * times),
            ),
            -1,
        ),
    )


class Oracle:
    """Known equations and equilibrium reset, advanced by issued commands only."""

    def __init__(self, model):
        self.model = model
        environment = cascade.standard_environment()

        def advance(state, command):
            return rollout(
                model,
                state,
                repeat_control(control_from_array(model, command), 20),
                environment,
                1 / 400,
            )[0]

        def reset(state, command):
            base = zero_state(model)._replace(
                rigid_body=rigid_body_from_canonical(state)
            )
            return equilibrate_internal_state(
                model, base, control_from_array(model, command), environment
            )

        def predict(state, commands):
            def step(s, u):
                updated = advance(s, u)
                return updated, observation(rigid_body_to_canonical(updated.rigid_body))

            return jax.lax.scan(step, state, commands)[1]

        self.advance = jax.jit(advance)
        self.reset = jax.jit(reset)
        self.predict = jax.jit(predict)


def predictor(oracle, learned=None):
    def predict(oracle_state, past_x, past_u, commands, mix, bias):
        def exact(_):
            return oracle.predict(oracle_state, commands)

        def model(_):
            return learned.predict(past_x, past_u, commands)

        def blend(_):
            return (1 - mix) * exact(None) + mix * model(None)

        mean = (
            exact(None)
            if learned is None
            else jax.lax.cond(
                mix == 0,
                exact,
                lambda _: jax.lax.cond(mix == 1, model, blend, None),
                None,
            )
        )
        ramp = jnp.arange(1, len(commands) + 1) / len(commands)
        # Prescribed vertical-velocity output error; identical input Jacobian.
        return mean.at[:, 2].add(bias * ramp)

    return predict


def make_controller(predict, trim_command, trim_observation):
    c = CONTROLLER
    h = c["horizon_steps"]
    low, high = map(jnp.asarray, (COMMAND_MINIMUM, COMMAND_MAXIMUM))
    center, half = (low + high) / 2, (high - low) / 2
    trim_z = (trim_command - center) / half

    def residual(z, oracle_state, past_x, past_u, position, t, previous, mix, bias):
        commands = jnp.broadcast_to(center + half * z, (h, 3))
        prediction = predict(oracle_state, past_x, past_u, commands, mix, bias)
        velocity = prediction[:, :3]
        preceding_velocity = jnp.concatenate((past_x[-1:, :3], velocity[:-1]))
        positions = position + jnp.cumsum(
            (preceding_velocity + velocity) * DT_S / 2, axis=0
        )
        target_p, target_v = reference(t + DT_S * jnp.arange(1, h + 1))
        anticipate = positions[:, 1:3] + c["lookahead_s"] * velocity[:, 1:3]
        desired = target_p[:, 1:3] + c["lookahead_s"] * target_v[:, 1:3]
        tracking = jnp.concatenate(
            (
                (anticipate - desired) / c["position_scale_m"],
                (velocity[:, :1] - target_v[:, :1]) / c["forward_speed_scale_m_s"],
                c["rate_weight"] * prediction[:, 3:6],
                c["rotation_weight"] * (prediction[:, 6:] - trim_observation[6:]),
            ),
            axis=-1,
        ).reshape(-1) / jnp.sqrt(h)
        return jnp.concatenate(
            (
                tracking,
                c["command_weight"] * (z - trim_z),
                c["command_change_weight"] * (z - (previous - center) / half),
            )
        )

    jacobian = jax.jacfwd(residual)

    @jax.jit
    def solve(oracle_state, past_x, past_u, position, t, previous, mix, bias):
        args = (oracle_state, past_x, past_u, position, t, previous, mix, bias)
        initial = jnp.clip((previous - center) / half, -1, 1)

        def cost(z):
            r = residual(z, *args)
            return jnp.sum(r**2)

        def step(_, z):
            r, jac = residual(z, *args), jacobian(z, *args)
            direction = jnp.linalg.solve(
                jac.T @ jac + c["damping"] * jnp.eye(3), jac.T @ r
            )
            moved = jnp.clip(
                z - jnp.asarray(c["line_search"][:-1])[:, None] * direction, -1, 1
            )
            candidates = jnp.concatenate((moved, z[None]), axis=0)
            costs = jax.vmap(cost)(candidates)
            costs = jnp.where(jnp.isfinite(costs), costs, jnp.inf)
            return candidates[jnp.argmin(costs)]

        final = jax.lax.fori_loop(0, c["iterations"], step, initial)
        command = center + half * final
        prediction = predict(
            oracle_state, past_x, past_u, jnp.broadcast_to(command, (h, 3)), mix, bias
        )
        return command, cost(initial), cost(final), prediction

    return solve


def initial_condition(state, seed):
    if seed == 0:
        return np.array(state, copy=True)
    rng = np.random.default_rng(seed)
    offset = np.zeros(12)
    offset[1:3] = rng.uniform(-0.15, 0.15, 2)
    offset[4:6] = rng.uniform(-0.05, 0.05, 2)
    offset[6:9] = rng.uniform(-0.01, 0.01, 3)
    offset[9:12] = rng.uniform(-0.02, 0.02, 3)
    return np.asarray(state_plus_tangent(jnp.asarray(state), jnp.asarray(offset)))


def run_trial(
    plant,
    oracle,
    solve,
    reset_state,
    trim_command,
    *,
    seed,
    mix,
    bias,
    alternating,
    duration_s=16.0,
):
    sample = plant.reset(
        initial_condition(reset_state, seed), applied_control=trim_command
    )
    oracle_state = oracle.reset(sample.state, trim_command)
    past_x, past_u = [np.asarray(observation(sample.state))], []
    for _ in range(2):
        sample = plant.step(trim_command)
        oracle_state = oracle.advance(oracle_state, trim_command)
        past_x.append(np.asarray(observation(sample.state)))
        past_u.append(trim_command.copy())
    states, commands, costs, elapsed, replay_errors = (
        [sample.state.copy()],
        [],
        [],
        [],
        [],
    )
    previous = trim_command.copy()
    predicted = []
    failure = None
    n = round(duration_s / DT_S)
    for k in range(n):
        before = time.perf_counter()
        signed_bias = bias * (-1 if alternating and k % 2 else 1)
        command, pre, post, forecast = solve(
            oracle_state,
            np.array(past_x[-3:]),
            np.array(past_u[-2:]),
            sample.state[:3],
            k * DT_S,
            previous,
            mix,
            signed_bias,
        )
        command = np.asarray(command)
        elapsed.append(time.perf_counter() - before)
        if not np.isfinite(command).all() or not np.isfinite(float(post)):
            failure = "nonfinite controller result"
            break
        sample = plant.step(command)
        oracle_state = oracle.advance(oracle_state, command)
        discrepancy = np.max(
            np.abs(
                sample.state
                - np.asarray(rigid_body_to_canonical(oracle_state.rigid_body))
            )
        )
        replay_errors.append(float(discrepancy))
        states.append(sample.state.copy())
        commands.append(command.copy())
        predicted.append(np.asarray(forecast))
        costs.append([float(pre), float(post)])
        past_x.append(np.asarray(observation(sample.state)))
        past_u.append(command.copy())
        previous = command
        if (
            not np.isfinite(sample.state).all()
            or abs(sample.state[2] - 100) > 20
            or np.linalg.norm(sample.state[3:6]) > 45
            or np.linalg.norm(sample.state[10:13]) > 5
        ):
            failure = "nonfinite or divergent plant state"
            break
    states, commands = np.array(states), np.array(commands)
    times = np.arange(1, n + 1) * DT_S
    target, _ = reference(jnp.asarray(times))
    error = np.full((n, 2), np.inf)
    error[: len(states) - 1] = np.abs(
        states[1:, 1:3] - np.asarray(target[: len(states) - 1, 1:3])
    )
    scored = times >= 2.0
    within = np.all(error <= 0.5, axis=1)
    fraction = float(np.mean(within[scored]))
    actual_observations = (
        np.asarray(observation(states[1:])) if len(commands) else np.empty((0, 15))
    )
    local_error = (
        np.asarray(predicted)[:, 0] - actual_observations
        if len(commands)
        else np.empty((0, 15))
    )
    local_rmse = (
        [
            float(np.sqrt(np.mean(np.sum(local_error[:, group] ** 2, axis=1))))
            for group in (slice(0, 3), slice(3, 6))
        ]
        if len(commands)
        else [None, None]
    )
    local_rmse = [v if v is None or np.isfinite(v) else None for v in local_rmse]
    result = dict(
        seed=seed,
        mix=mix,
        bias_m_s=bias,
        alternating=alternating,
        expected_steps=n,
        completed_steps=len(commands),
        failure=failure,
        within_tolerance_fraction=fraction,
        meets_requirement=failure is None and fraction >= 0.95,
        tracking_rmse_m=[
            float(v) if np.isfinite(v) else None
            for v in np.sqrt(np.mean(error[scored] ** 2, axis=0))
        ],
        maximum_oracle_replay_difference=max(replay_errors, default=0.0)
        if np.isfinite(replay_errors).all()
        else None,
        maximum_solver_cost_increase=max((b - a for a, b in costs), default=0.0),
        solve_seconds=float(sum(elapsed)),
        on_policy_one_step_velocity_rate_rmse=local_rmse,
    )
    arrays = dict(
        states=states,
        commands=commands,
        costs=np.array(costs),
        solve_seconds=np.array(elapsed),
        oracle_replay_errors=np.array(replay_errors),
        errors=error,
        times_s=times,
        constant_command_forecasts=np.asarray(predicted),
        local_observation_errors=local_error,
        initial_state=initial_condition(reset_state, seed),
    )
    return result, arrays


def as_collection(recordings):
    return SequenceCollection(
        tuple(
            SequenceSegment(
                r.labels["source_group"],
                "whole",
                np.asarray(observation(r.states)),
                r.controls,
                DT_S,
            )
            for r in recordings
        ),
        configuration_id="cascade_published_x8_cruise",
        state_channels=tuple(
            [f"velocity_{v} [m/s,NWU]" for v in "xyz"]
            + [f"body_rate_{v} [rad/s,FLU]" for v in "xyz"]
            + [
                f"rotation_{i}{j} [unitless,FLU-to-NWU]"
                for i in range(3)
                for j in range(3)
            ]
        ),
        input_channels=(
            "throttle [normalized command]",
            "aileron [rad,requested surface angle]",
            "elevator [rad,requested surface angle]",
        ),
    )


def arms():
    values = [
        dict(name=name, mix=value, bias=0.0, alternating=False)
        for name, value in (
            ("oracle", 0.0),
            ("mix-025", 0.25),
            ("mix-050", 0.5),
            ("mix-075", 0.75),
            ("learned", 1.0),
        )
    ]
    for bias in (0.1, 0.3, 0.6):
        for alternating in (False, True):
            values.append(
                dict(
                    name=f"{'alternating' if alternating else 'bias'}-{bias:.1f}",
                    mix=0.0,
                    bias=bias,
                    alternating=alternating,
                )
            )
    return values


def forecast_check(oracle, learned, recordings, output):
    predict = jax.jit(jax.vmap(predictor(oracle, learned)))
    reports = []
    for flight in recordings:
        name = flight.labels["source_group"]
        states = np.asarray(observation(flight.states))
        p, h = learned.history_steps, learned.horizon_steps
        origins = np.arange(p, len(states) - h, h)
        b = sequence_windows(
            states,
            flight.controls,
            origins,
            history_steps=p,
            horizon_steps=h,
            dt_s=DT_S,
        )
        initial = oracle.reset(flight.states[0], flight.control_prefix[-1])

        def advance(s, u):
            next_state = oracle.advance(s, u)
            return next_state, next_state

        _, replay = jax.lax.scan(advance, initial, jnp.asarray(flight.controls))
        replayed = np.asarray(rigid_body_to_canonical(replay.rigid_body))
        np.testing.assert_allclose(replayed, flight.states[1:], atol=1e-8, rtol=1e-8)
        start_states = jax.tree.map(
            lambda leaf, origins=origins: leaf[origins - 1], replay
        )
        saved = dict(
            origins=origins,
            past_states=b.past_states,
            past_inputs=b.past_inputs,
            future_inputs=b.future_inputs,
            targets=b.future_states,
        )
        arm_reports = {}
        for arm in arms():
            signs = (
                np.where(origins % 2, -1, 1)
                if arm["alternating"]
                else np.ones(len(origins))
            )
            predicted = np.asarray(
                predict(
                    start_states,
                    b.past_states,
                    b.past_inputs,
                    b.future_inputs,
                    np.full(len(origins), arm["mix"]),
                    arm["bias"] * signs,
                )
            )
            saved[f"prediction_{arm['name']}"] = predicted
            error = predicted - b.future_states
            arm_reports[arm["name"]] = dict(
                per_horizon_velocity_rate_rmse=np.stack(
                    [
                        np.sqrt(np.mean(np.sum(error[:, :, g] ** 2, axis=-1), axis=0))
                        for g in (slice(0, 3), slice(3, 6))
                    ],
                    axis=-1,
                ).tolist(),
                rotation_matrix_defect_p95=float(
                    np.quantile(
                        np.linalg.norm(
                            predicted[:, :, 6:].reshape(-1, 3, 3).transpose(0, 2, 1)
                            @ predicted[:, :, 6:].reshape(-1, 3, 3)
                            - np.eye(3),
                            axis=(1, 2),
                        ),
                        0.95,
                    )
                ),
            )
        np.savez_compressed(output / f"forecast-{name}.npz", **saved)
        reports.append(
            dict(
                recording=name,
                windows=len(origins),
                maximum_oracle_replay_difference=float(
                    np.max(np.abs(replayed - flight.states[1:]))
                ),
                arms=arm_reports,
            )
        )
    write(output / "forecast-errors.json", reports)


def main(a):
    if not jax.config.jax_enable_x64:
        raise ValueError("benchmark requires JAX_ENABLE_X64=1")
    a.output.mkdir(parents=True, exist_ok=False)
    plant, trim, state, command = fixture()
    oracle = Oracle(plant.model)
    write(a.output / "controller.json", CONTROLLER)
    write(
        a.output / "platform.json",
        dict(
            cascade=cascade.stamp(plant.spec, plant.model),
            initial_state=state.tolist(),
            initial_command=command.tolist(),
        ),
    )
    repo = Path(__file__).resolve().parents[1]
    source_files = [
        *sorted((repo / "src/glassbox").rglob("*.py")),
        *sorted((repo / "examples").glob("*.py")),
        *sorted(Path(cascade.__file__).parent.rglob("*.py")),
    ]
    sources = {}
    with zipfile.ZipFile(
        a.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for path in source_files:
            name = (
                f"glassbox/{path.relative_to(repo)}"
                if path.is_relative_to(repo)
                else f"cascade/{path.relative_to(Path(cascade.__file__).parent)}"
            )
            z.write(path, name)
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(a.output / "sources.json", sources)
    if a.develop:
        solve = make_controller(
            predictor(oracle), jnp.asarray(command), observation(state)
        )
        result, arrays = run_trial(
            plant,
            oracle,
            solve,
            state,
            command,
            seed=0,
            mix=0.0,
            bias=0.0,
            alternating=False,
        )
        write(a.output / "development.json", result)
        np.savez_compressed(a.output / "development.npz", **arrays)
        print(json.dumps(result), flush=True)
        return
    frozen = json.loads(a.plan.read_text())
    if frozen["controller"] != CONTROLLER or frozen["arms"] != arms():
        raise ValueError("controller/arms differ from the frozen evaluation protocol")
    write(a.output / "plan.json", frozen)
    recordings = []
    for seed in (0, 1, 2, 3, 80, 81):
        flight = collect_recording(plant, trim, state, command, seed, 12.0)
        save_trajectory_npz(flight, a.output / f"recording-{seed}.npz")
        recordings.append(flight)
        print(
            json.dumps(dict(collected_recording=seed, rows=len(flight.states))),
            flush=True,
        )
    learned = fit(as_collection(recordings[:4]))
    if (
        learned.history_steps != 2
        or learned.horizon_steps != CONTROLLER["horizon_steps"]
    ):
        raise ValueError("fixed learner/controller horizon mismatch")
    learned.save(a.output / "learned.npz")
    write(a.output / "fit-report.json", learned.report)
    print(
        json.dumps(
            dict(
                fit_selected_step=learned.report["optimization"]["selected_step"],
                fingerprint=learned.fingerprint(),
            )
        ),
        flush=True,
    )
    forecast_check(oracle, learned, recordings[4:], a.output)
    print("Reserved-recording forecasts complete", flush=True)
    solve = make_controller(
        predictor(oracle, learned), jnp.asarray(command), observation(state)
    )
    results = []
    all_arms = arms()
    for seed in (101, 102, 103):
        shift = seed - 101
        for arm in all_arms[shift:] + all_arms[:shift]:
            parameters = {k: v for k, v in arm.items() if k != "name"}
            result, arrays = run_trial(
                plant, oracle, solve, state, command, seed=seed, **parameters
            )
            result["arm"] = arm["name"]
            np.savez_compressed(a.output / f"trial-{seed}-{arm['name']}.npz", **arrays)
            results.append(result)
            write(a.output / "tracking-results.json", results)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--develop", action="store_true")
    p.add_argument(
        "--plan",
        type=Path,
        default=Path("../artifacts/cascade-accuracy/evaluation-plan.json"),
    )
    main(p.parse_args())
