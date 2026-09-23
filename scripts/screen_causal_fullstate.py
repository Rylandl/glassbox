"""Full-state rollout of the episode-fitted causal actuator model.

Uses the committed frozen benchmark's recordings and public-model forecasts.
Every fit sees only its origin's completed observations and issued commands.
"""

import argparse
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
            H @ applied_rate
            + np.cross(rate, H @ applied)
            - np.cross(rate, J @ rate)
        )
        return np.linalg.solve(J, torque)

    def linear_derivative(velocity, orientation, applied):
        body_velocity = orientation.T @ velocity
        speed = np.linalg.norm(body_velocity)
        features = np.r_[1.0, applied, applied * applied, body_velocity,
                         body_velocity * speed]
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


def evaluate_case(name, origins=None):
    spec = read(SPEC)
    case = source_case(spec, name, "smoke")
    states, commands, dt = case["states"], case["commands"], case["dt"]
    public = np.load(PUBLIC / f"{name}.npz")
    assert np.array_equal(public["origins"][:2], case["origins"])
    rows = case["origins"] if origins is None else origins
    result = []
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
        errors = {}
        for ms in (50, 250):
            index = round(ms / 1000 / dt) - 1
            a, b, t = forecast[index], control[index], truth[index]
            errors[ms] = {
                "velocity": (float(np.linalg.norm(a[:3] - t[:3])),
                             float(np.linalg.norm(b[:3] - t[:3]))),
                "rate": (float(np.linalg.norm(a[3:6] - t[3:6])),
                         float(np.linalg.norm(b[3:6] - t[3:6]))),
                "orientation": (
                    float(Rotation.from_matrix(t[6:].reshape(3, 3).T
                                               @ a[6:].reshape(3, 3)).magnitude()),
                    float(Rotation.from_matrix(t[6:].reshape(3, 3).T
                                               @ b[6:].reshape(3, 3)).magnitude()),
                ),
            }
        result.append((origin, errors, seconds, np.sqrt(np.mean(rf * rf)),
                       np.sqrt(np.mean(rt * rt))))
    return result


def highspin_case():
    from screen_causal_relaxation import ROOT

    for arm in ("085", "140"):
        for branch in ("control", "orthogonal"):
            saved = np.load(ROOT / f"quad-arm-{arm}-heldout-{branch}.npz")
            states = observed(saved["states"])
            commands = saved["commands"]
            origin, horizon, dt = 150, 25, 0.01
            J = fit_inertia(states[: origin + 1], commands[:origin], dt)
            detail, _, seconds = fit(states[: origin + 1], commands[:origin],
                                     100, origin, dt, J)
            _, q, coeff, _, rf, rt, Cf, Ct, _ = detail
            latent = latent_trace(commands[: origin + horizon], q, coeff, dt)
            forecast = full_state_forecast(states[origin], latent, origin,
                                           J, Cf, Ct, dt)
            truth = states[origin + 1 : origin + horizon + 1]
            errors = {}
            for ms in (50, 250):
                index = ms // 10 - 1
                errors[ms] = (
                    float(np.linalg.norm(forecast[index, :3] - truth[index, :3])),
                    float(np.linalg.norm(forecast[index, 3:6] - truth[index, 3:6])),
                    float(Rotation.from_matrix(
                        truth[index, 6:].reshape(3, 3).T
                        @ forecast[index, 6:].reshape(3, 3)
                    ).magnitude()),
                )
            print("highspin", arm, branch, errors, "fit_s", round(seconds, 3),
                  "force_fit", round(float(np.sqrt(np.mean(rf * rf))), 4),
                  "torque_fit", round(float(np.sqrt(np.mean(rt * rt))), 4),
                  flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append")
    parser.add_argument("--highspin", action="store_true")
    args = parser.parse_args()
    if args.highspin:
        highspin_case()
    for name in args.name or ():
        for origin, errors, seconds, force, torque in evaluate_case(name):
            print(name, origin, errors, "fit_s", round(seconds, 3),
                  "force_fit", round(force, 4), "torque_fit", round(torque, 4),
                  flush=True)
