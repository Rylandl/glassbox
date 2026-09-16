"""Audit frozen tracking outcomes and diagnose command coverage without refitting."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_model_structures import recurrence

from glassbox.core.data import load_trajectory_npz
from glassbox.core.geometry import quaternion_to_rotation_matrices
from glassbox.experimental.default_model import LearnedDynamics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from cascade_accuracy import (
    COMMAND_MAXIMUM,
    COMMAND_MINIMUM,
    CONTROLLER,
    Oracle,
    arms,
    fixture,
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def obs(states):
    return np.concatenate(
        (
            states[..., 3:6],
            states[..., 10:13],
            quaternion_to_rotation_matrices(states[..., 6:10]).reshape(
                *states.shape[:-1], 9
            ),
        ),
        axis=-1,
    )


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    checks, maximum = 0, 0.0

    def check(x, y):
        nonlocal checks, maximum
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape
        np.testing.assert_array_equal(np.isfinite(x), np.isfinite(y))
        finite = np.isfinite(x)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(x[finite] - y[finite]))))
        np.testing.assert_allclose(x, y, rtol=2e-9, atol=2e-9)
        checks += 1

    with zipfile.ZipFile(a.run / "executed-sources.zip") as z:
        for name, digest in read(a.run / "sources.json").items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    plan = read(a.run / "plan.json")
    assert plan["controller"] == CONTROLLER and plan["arms"] == arms()
    model = LearnedDynamics.load(a.run / "learned.npz")
    assert model.report == read(a.run / "fit-report.json")
    assert model.report["recipe"] == plan["learner_recipe"]
    recordings = {
        f"cascade-calibration-{i}": load_trajectory_npz(a.run / f"recording-{i}.npz")
        for i in (0, 1, 2, 3, 80, 81)
    }
    assert set(model._seen) == {f"cascade-calibration-{i}" for i in range(4)}
    windows_checked = 0
    for cache in (model._train, model._development):
        for i, (key, row) in enumerate(zip(cache.keys, cache.source_origins)):
            flight = recordings[key.recording_id]
            state = obs(flight.states)
            assert row == key.origin and key.segment_id == "whole"
            check(cache.batch.past_states[i], state[row - 2 : row + 1])
            check(cache.batch.past_inputs[i], flight.controls[row - 2 : row])
            check(cache.batch.future_states[i], state[row + 1 : row + 6])
            check(cache.batch.future_inputs[i], flight.controls[row : row + 5])
            windows_checked += 1
    train_u = model._train.batch.future_inputs.reshape(-1, 3)
    train_min, train_max = train_u.min(axis=0), train_u.max(axis=0)
    forecast_errors = {}
    for r in read(a.run / "forecast-errors.json"):
        flight = recordings[r["recording"]]
        with np.load(a.run / f"forecast-{r['recording']}.npz") as z:
            x = obs(flight.states)
            origins = z["origins"]
            check(z["targets"], np.array([x[i + 1 : i + 6] for i in origins]))
            check(z["past_states"], np.array([x[i - 2 : i + 1] for i in origins]))
            check(
                z["past_inputs"],
                np.array([flight.controls[i - 2 : i] for i in origins]),
            )
            check(
                z["future_inputs"],
                np.array([flight.controls[i : i + 5] for i in origins]),
            )
            learned = recurrence(
                model._model, z["past_states"], z["past_inputs"], z["future_inputs"]
            )
            check(learned, z["prediction_learned"])
            check(z["prediction_oracle"], z["targets"])
            for arm in arms():
                n = arm["name"]
                expected = (1 - arm["mix"]) * z["prediction_oracle"] + arm[
                    "mix"
                ] * learned
                signs = (
                    np.where(origins % 2, -1, 1)
                    if arm["alternating"]
                    else np.ones(len(origins))
                )
                expected[:, :, 2] += (
                    arm["bias"] * signs[:, None] * np.arange(1, 6)[None] / 5
                )
                check(expected, z[f"prediction_{n}"])
                e = expected - z["targets"]
                rmse = np.stack(
                    [
                        np.sqrt(np.mean(np.sum(e[:, :, g] ** 2, axis=-1), axis=0))
                        for g in (slice(0, 3), slice(3, 6))
                    ],
                    axis=-1,
                )
                check(rmse, r["arms"][n]["per_horizon_velocity_rate_rmse"])
                rotation = expected[:, :, 6:].reshape(-1, 3, 3)
                defect = np.quantile(
                    np.linalg.norm(
                        rotation.transpose(0, 2, 1) @ rotation - np.eye(3), axis=(1, 2)
                    ),
                    0.95,
                )
                check(defect, r["arms"][n]["rotation_matrix_defect_p95"])
                forecast_errors.setdefault(n, []).append(rmse)
    plant, _trim, _initial, trim_command = fixture()
    oracle = Oracle(plant.model)
    from cascade.canonical import rigid_body_to_canonical

    rows = read(a.run / "tracking-results.json")
    assert len(rows) == 33 and len({(r["seed"], r["arm"]) for r in rows}) == 33
    assert {r["seed"] for r in rows} == set(plan["initial_condition_seeds"])
    diagnostics = []
    initialized = {}
    jacobians = []
    for r in rows:
        with np.load(a.run / f"trial-{r['seed']}-{r['arm']}.npz") as z:
            states, commands, reset = z["states"], z["commands"], z["initial_state"]
            m = len(commands)
            assert m == r["completed_steps"] and len(states) == m + 1
            assert np.isfinite(states).all() and np.isfinite(commands).all()
            assert np.all(commands >= COMMAND_MINIMUM - 1e-12) and np.all(
                commands <= COMMAND_MAXIMUM + 1e-12
            )
            t = np.arange(1, 321) * 0.05
            check(z["times_s"], t)
            error = np.full((320, 2), np.inf)
            error[:m] = np.abs(
                states[1:, 1:3]
                - np.column_stack(
                    (np.sin(0.35 * t[:m]), 100 + 0.75 * np.sin(0.3 * t[:m]))
                )
            )
            check(error, z["errors"])
            scored = t >= 2
            within = np.all(error <= 0.5, axis=1)
            fraction = float(within[scored].mean())
            check(fraction, r["within_tolerance_fraction"])
            assert r["meets_requirement"] == (r["failure"] is None and fraction >= 0.95)
            rmse = np.sqrt(np.mean(error[scored] ** 2, axis=0))
            check(rmse, [np.inf if v is None else v for v in r["tracking_rmse_m"]])
            if m < 320:
                assert r["failure"] is not None
                assert (
                    abs(states[-1, 2] - 100) > 20
                    or np.linalg.norm(states[-1, 3:6]) > 45
                    or np.linalg.norm(states[-1, 10:13]) > 5
                )
            check(
                np.max(z["costs"][:, 1] - z["costs"][:, 0]),
                r["maximum_solver_cost_increase"],
            )
            assert np.max(z["costs"][:, 1] - z["costs"][:, 0]) <= 1e-8
            check(z["solve_seconds"].sum(), r["solve_seconds"])
            internal = oracle.reset(reset, trim_command)
            warm1 = oracle.advance(internal, trim_command)
            warm2 = oracle.advance(warm1, trim_command)
            check(np.asarray(rigid_body_to_canonical(warm2.rigid_body)), states[0])
            full_x = obs(
                np.vstack(
                    (
                        reset,
                        np.asarray(rigid_body_to_canonical(warm1.rigid_body)),
                        states,
                    )
                )
            )
            full_u = np.vstack((trim_command, trim_command, commands))
            past_x = np.array([full_x[k : k + 3] for k in range(m)])
            past_u = np.array([full_u[k : k + 2] for k in range(m)])
            learned = recurrence(
                model._model, past_x, past_u, np.repeat(commands[:, None], 5, axis=1)
            )
            first = (1 - r["mix"]) * obs(states[1:]) + r["mix"] * learned[:, 0]
            signs = (
                np.where(np.arange(m) % 2, -1, 1) if r["alternating"] else np.ones(m)
            )
            first[:, 2] += r["bias_m_s"] * signs / 5
            check(first, z["constant_command_forecasts"][:, 0])
            if r["arm"] == "learned":
                check(learned, z["constant_command_forecasts"])
            local = first - obs(states[1:])
            check(local, z["local_observation_errors"])
            check(
                [
                    np.sqrt(np.mean(np.sum(local[:, g] ** 2, axis=1)))
                    for g in (slice(0, 3), slice(3, 6))
                ],
                r["on_policy_one_step_velocity_rate_rmse"],
            )

            def replay_step(s, u):
                nxt = oracle.advance(s, u)
                return nxt, rigid_body_to_canonical(nxt.rigid_body)

            _, replayed = jax.lax.scan(replay_step, warm2, jnp.asarray(commands))
            check(np.asarray(replayed), states[1:])
            outside = (commands < train_min) | (commands > train_max)
            diagnostics.append(
                dict(
                    seed=r["seed"],
                    arm=r["arm"],
                    completed_steps=m,
                    commands_outside_training_range_fraction=float(
                        np.any(outside, axis=1).mean()
                    ),
                    per_channel_outside_fraction=outside.mean(axis=0).tolist(),
                    first_command=commands[0].tolist(),
                    first_command_outside=np.any(outside[0]).item(),
                    command_box_boundary_fraction=float(
                        np.any(
                            np.isclose(commands, COMMAND_MINIMUM, atol=1e-8)
                            | np.isclose(commands, COMMAND_MAXIMUM, atol=1e-8),
                            axis=1,
                        ).mean()
                    ),
                )
            )
            if r["seed"] not in initialized:
                initialized[r["seed"]] = reset.copy()
                half = jnp.asarray((COMMAND_MAXIMUM - COMMAND_MINIMUM) / 2)

                def exact_at(delta, warm2=warm2, half=half):
                    return oracle.predict(
                        warm2, jnp.broadcast_to(trim_command + half * delta, (5, 3))
                    )

                def learned_at(delta, past_x=past_x, past_u=past_u, half=half):
                    return model.predict(
                        past_x[0],
                        past_u[0],
                        jnp.broadcast_to(trim_command + half * delta, (5, 3)),
                    )

                jt = np.asarray(jax.jacfwd(exact_at)(jnp.zeros(3)))
                jl = np.asarray(jax.jacfwd(learned_at)(jnp.zeros(3)))
                relative = {}
                for step in (0, 4):
                    for label, g in (("velocity", slice(0, 3)), ("rate", slice(3, 6))):
                        relative[f"{step + 1}_step_{label}"] = float(
                            np.linalg.norm(jl[step, g] - jt[step, g])
                            / np.linalg.norm(jt[step, g])
                        )
                jacobians.append(
                    dict(
                        seed=r["seed"],
                        command=trim_command.tolist(),
                        input_half_box=np.asarray(half).tolist(),
                        oracle_jacobian=jt.tolist(),
                        learned_jacobian=jl.tolist(),
                        relative_frobenius_error=relative,
                    )
                )
            else:
                check(reset, initialized[r["seed"]])
    summary = []
    for arm in arms():
        n = arm["name"]
        trials = [r for r in rows if r["arm"] == n]
        summary.append(
            dict(
                **arm,
                passes=sum(r["meets_requirement"] for r in trials),
                trials=3,
                minimum_within_fraction=min(
                    r["within_tolerance_fraction"] for r in trials
                ),
                maximum_within_fraction=max(
                    r["within_tolerance_fraction"] for r in trials
                ),
                forecast_rmse=np.sqrt(
                    np.mean(np.asarray(forecast_errors[n]) ** 2, axis=0)
                ).tolist(),
                tracking_rmse_m=[r["tracking_rmse_m"] for r in trials],
                completed_steps=[r["completed_steps"] for r in trials],
                on_policy_rmse=[
                    r["on_policy_one_step_velocity_rate_rmse"] for r in trials
                ],
            )
        )
    write(a.output / "summary.json", summary)
    write(
        a.output / "command-coverage.json",
        dict(
            training_command_minimum=train_min.tolist(),
            training_command_maximum=train_max.tolist(),
            trials=diagnostics,
            limitation="Marginal training-window command ranges, not joint support or a causal explanation.",
        ),
    )
    write(
        a.output / "initial-command-jacobians.json",
        dict(
            samples=jacobians,
            limitation="Post-hoc diagnostics at a shared initial history and trim command. Oracle knowledge is used only for evaluation; no parameter update or controller tuning.",
        ),
    )
    audit = dict(
        trials=len(rows),
        forecast_recordings=2,
        forecast_arms=22,
        cached_training_development_windows=windows_checked,
        numerical_checks=checks,
        maximum_absolute_difference=maximum,
        source_archives_verified=True,
        all_saved_commands_replayed=True,
        learned_forecasts_replayed_in_numpy=True,
        all_truncated_intervals_count_as_failures=True,
        limitation="Numerical replay and interface checks, not independent physical truth. Optimizer iterations are not independently re-solved; cost descent and actual bounded commands are checked.",
    )
    write(a.output / "audit.json", audit)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {
        "mixed": "#cf4b38",
        "bias": "#9c650d",
        "alternating": "#287caa",
        "oracle": "#242b38",
    }
    for row in summary:
        kind = (
            "oracle"
            if row["name"] == "oracle"
            else "mixed"
            if row["mix"]
            else "alternating"
            if row["alternating"]
            else "bias"
        )
        x = row["forecast_rmse"][-1][0]
        axes[0].plot(
            [x, x],
            np.array([row["minimum_within_fraction"], row["maximum_within_fraction"]])
            * 100,
            color=colors[kind],
            alpha=0.4,
        )
        if kind == "bias":
            axes[0].scatter(
                x,
                100 * row["minimum_within_fraction"],
                edgecolors=colors[kind],
                facecolors="none",
                marker="s",
                s=100,
            )
        else:
            axes[0].scatter(
                x, 100 * row["minimum_within_fraction"], color=colors[kind], s=40
            )
    for k, label in (
        ("oracle", "Oracle"),
        ("mixed", "Oracle/learned mixtures"),
        ("bias", "Persistent velocity bias"),
        ("alternating", "Alternating velocity bias"),
    ):
        if k == "bias":
            axes[0].scatter(
                [], [], edgecolors=colors[k], facecolors="none", marker="s", label=label
            )
        else:
            axes[0].scatter([], [], color=colors[k], label=label)
    axes[0].axhline(95, color="black", ls="--", lw=1)
    axes[0].set(
        xlabel="250 ms velocity RMSE on common reserved data [m/s]",
        ylabel="Time jointly within 0.5 m [%]",
        ylim=(-5, 105),
        title="Worst of three trials; bars show trial range",
    )
    axes[0].legend(fontsize=8, loc="center right")
    for n, label, color in (
        ("oracle", "Oracle", colors["oracle"]),
        ("bias-0.3", "Persistent 0.3 m/s", colors["bias"]),
        ("alternating-0.3", "Alternating 0.3 m/s", colors["alternating"]),
    ):
        with np.load(a.run / f"trial-101-{n}.npz") as z:
            times = np.arange(len(z["states"])) * 0.05
            error = z["states"][:, 2] - (100 + 0.75 * np.sin(0.3 * times))
            axes[1].plot(times, error, label=label, color=color)
    axes[1].axhline(-0.5, color="black", ls="--", lw=1)
    axes[1].axhline(0.5, color="black", ls="--", lw=1)
    axes[1].axvspan(0, 2, color="grey", alpha=0.1)
    axes[1].set(
        xlabel="Time [s]",
        ylabel="Altitude tracking error [m]",
        title="Equal error RMS, different tracking behavior · seed 101",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Cascade cruise tracking · fixed learner and controller · 33 offline trials",
        fontsize=12,
    )
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"accuracy-vs-tracking.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(audit), flush=True)
    print(
        json.dumps(
            [
                dict(
                    arm=r["name"],
                    passes=r["passes"],
                    forecast_250ms=r["forecast_rmse"][-1],
                    minimum_fraction=r["minimum_within_fraction"],
                )
                for r in summary
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", type=Path, default=Path("../artifacts/cascade-accuracy/comparison-01")
    )
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
