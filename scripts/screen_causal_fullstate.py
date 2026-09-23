"""Full-state rollout of the episode-fitted causal actuator model.

Uses the committed frozen benchmark's recordings and public-model forecasts.
Every fit sees only its origin's completed observations and issued commands.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark_online_readout import SPEC, read, source_case
from scipy.spatial.transform import Rotation
from screen_causal_relaxation import fit, fit_inertia, latent_trace
from verify_baseline import observed

PUBLIC = (
    Path(read(SPEC)["sources"]["recordings"]["path"]).parents[1]
    / "readout-public-rate-v1/public-rate-lean-full"
)
GRAVITY = np.array([0.0, 0.0, -9.80665])


def full_state_forecast(start, latent, origin, J, Cf, Ct, dt):
    """Integrate one immutable episode fit with shared actuator and rigid-body state."""
    m = latent.shape[1]
    H = Ct[: 3 * m].reshape(m, 3).T
    B = Ct[3 * m :].reshape(3, 2 + 2 * m)
    v = start[:3].copy()
    w = start[3:6].copy()
    R = start[6:].reshape(3, 3).copy()
    predicted = []

    def angular_derivative(rate, applied, applied_rate):
        torque = (
            B[:, 0]
            + B[:, 1 : 1 + m] @ applied
            + B[:, 1 + m : 1 + 2 * m] @ (applied * applied)
            - B[:, -1] * rate
        )
        torque += (
            H @ applied_rate + np.cross(rate, H @ applied) - np.cross(rate, J @ rate)
        )
        return np.linalg.solve(J, torque)

    def linear_derivative(velocity, orientation, applied):
        body_velocity = orientation.T @ velocity
        speed = np.linalg.norm(body_velocity)
        features = np.r_[
            1.0, applied, applied * applied, body_velocity, body_velocity * speed
        ]
        return GRAVITY + orientation @ (features @ Cf)

    for t in range(origin, len(latent) - 1):
        applied, following = latent[t], latent[t + 1]
        middle = 0.5 * (applied + following)
        applied_rate = (following - applied) / dt
        w_mid = w + 0.5 * dt * angular_derivative(w, applied, applied_rate)
        R_mid = R @ Rotation.from_rotvec(0.5 * dt * w_mid).as_matrix()
        v_mid = v + 0.5 * dt * linear_derivative(v, R, applied)
        v = v + dt * linear_derivative(v_mid, R_mid, middle)
        w = w + dt * angular_derivative(w_mid, middle, applied_rate)
        R = R @ Rotation.from_rotvec(dt * w_mid).as_matrix()
        predicted.append(np.r_[v, w, R.reshape(-1)])
    return np.asarray(predicted)


def row_errors(forecast, control, truth, dt):
    errors = {}
    for ms in (50, 250):
        index = round(ms / 1000 / dt) - 1
        a, b, t = forecast[index], control[index], truth[index]
        errors[ms] = {
            "velocity": (
                float(np.linalg.norm(a[:3] - t[:3])),
                float(np.linalg.norm(b[:3] - t[:3])),
            ),
            "rate": (
                float(np.linalg.norm(a[3:6] - t[3:6])),
                float(np.linalg.norm(b[3:6] - t[3:6])),
            ),
            "orientation": (
                float(
                    Rotation.from_matrix(
                        t[6:].reshape(3, 3).T @ a[6:].reshape(3, 3)
                    ).magnitude()
                ),
                float(
                    Rotation.from_matrix(
                        t[6:].reshape(3, 3).T @ b[6:].reshape(3, 3)
                    ).magnitude()
                ),
            ),
        }
    return errors


def evaluate_case(name, origins=None, suite="smoke", output=None):
    spec = read(SPEC)
    case = source_case(spec, name, suite)
    states, commands, dt = case["states"], case["commands"], case["dt"]
    public = np.load(PUBLIC / f"{name}.npz")
    assert np.array_equal(public["origins"][: len(case["origins"])], case["origins"])
    rows = case["origins"] if origins is None else origins
    result = []
    forecasts = []
    for origin in rows:
        origin = int(origin)
        prefix = states[: origin + 1]
        issued = commands[: origin + case["horizon"]]
        J = fit_inertia(prefix, commands[:origin], dt)
        detail, _, seconds = fit(
            prefix, commands[:origin], case["begin"], origin, dt, J
        )
        _, q, coeff, _, rf, rt, Cf, Ct, _ = detail
        latent = latent_trace(issued, q, coeff, dt)
        forecast = full_state_forecast(states[origin], latent, origin, J, Cf, Ct, dt)
        control = public["forecast"][np.flatnonzero(public["origins"] == origin)[0]]
        truth = states[origin + 1 : origin + case["horizon"] + 1]
        assert forecast.shape == truth.shape == control.shape
        errors = row_errors(forecast, control, truth, dt)
        result.append(
            (
                origin,
                errors,
                seconds,
                np.sqrt(np.mean(rf * rf)),
                np.sqrt(np.mean(rt * rt)),
            )
        )
        forecasts.append(forecast)
    if output is not None:
        np.savez_compressed(
            output / f"{name}.npz", origins=rows, forecast=np.asarray(forecasts)
        )
    return result


def summarize(rows):
    return {
        str(ms): {
            key: [
                float(np.sqrt(np.mean(np.square([row[1][ms][key][i] for row in rows]))))
                for i in range(2)
            ]
            for key in ("velocity", "rate", "orientation")
        }
        for ms in (50, 250)
    }


def verify_saved(name, output):
    case = source_case(read(SPEC), name, "full")
    saved = np.load(output / f"{name}.npz")
    public = np.load(PUBLIC / f"{name}.npz")
    assert np.array_equal(saved["origins"], case["origins"])
    assert np.array_equal(public["origins"], case["origins"])
    assert saved["forecast"].shape == case["truth"].shape
    rows = [
        (
            int(origin),
            row_errors(forecast, control, truth, case["dt"]),
            None,
            None,
            None,
        )
        for origin, forecast, control, truth in zip(
            case["origins"], saved["forecast"], public["forecast"], case["truth"]
        )
    ]
    return summarize(rows)


def highspin_case():
    from screen_causal_relaxation import ROOT

    for arm in ("085", "140"):
        for branch in ("control", "orthogonal"):
            saved = np.load(ROOT / f"quad-arm-{arm}-heldout-{branch}.npz")
            states = observed(saved["states"])
            commands = saved["commands"]
            origin, horizon, dt = 150, 25, 0.01
            J = fit_inertia(states[: origin + 1], commands[:origin], dt)
            detail, _, seconds = fit(
                states[: origin + 1], commands[:origin], 100, origin, dt, J
            )
            _, q, coeff, _, rf, rt, Cf, Ct, _ = detail
            latent = latent_trace(commands[: origin + horizon], q, coeff, dt)
            forecast = full_state_forecast(
                states[origin], latent, origin, J, Cf, Ct, dt
            )
            truth = states[origin + 1 : origin + horizon + 1]
            errors = {}
            for ms in (50, 250):
                index = ms // 10 - 1
                errors[ms] = (
                    float(np.linalg.norm(forecast[index, :3] - truth[index, :3])),
                    float(np.linalg.norm(forecast[index, 3:6] - truth[index, 3:6])),
                    float(
                        Rotation.from_matrix(
                            truth[index, 6:].reshape(3, 3).T
                            @ forecast[index, 6:].reshape(3, 3)
                        ).magnitude()
                    ),
                )
            print(
                "highspin",
                arm,
                branch,
                errors,
                "fit_s",
                round(seconds, 3),
                "force_fit",
                round(float(np.sqrt(np.mean(rf * rf))), 4),
                "torque_fit",
                round(float(np.sqrt(np.mean(rt * rt))), 4),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append")
    parser.add_argument("--highspin", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.full:
        assert args.output is not None
        args.output.mkdir(parents=True, exist_ok=True)
        names = args.name or read(SPEC)["full_cases"]
        if args.verify:
            saved_summary = json.loads((args.output / "summary.json").read_text())
            assert saved_summary == {
                name: verify_saved(name, args.output) for name in names
            }
            print("verified", len(names), "cases without refitting")
        else:
            summary = {}
            for name in names:
                rows = evaluate_case(name, suite="full", output=args.output)
                summary[name] = summarize(rows)
                print(name, len(rows), summary[name], flush=True)
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
        raise SystemExit(0)
    if args.highspin:
        highspin_case()
    for name in args.name or ():
        for origin, errors, seconds, force, torque in evaluate_case(name):
            print(
                name,
                origin,
                errors,
                "fit_s",
                round(seconds, 3),
                "force_fit",
                round(force, 4),
                "torque_fit",
                round(torque, 4),
                flush=True,
            )
