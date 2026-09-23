"""Causal actuator-relaxation architecture screen on frozen recordings.

This is a research fitter, separate from the maintained online learner. It fits
only observed states and issued commands from each prefix. Future states are
used solely to score the 250 ms rate forecast.
"""

import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_online_readout import SPEC, read, source_case
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from verify_baseline import observed

ROOT = (
    Path(read(SPEC)["sources"]["recordings"]["path"]).parents[1]
    / "high-spin-excitation-v1"
)
ARTIFACTS = ROOT.parent
jax.config.update("jax_enable_x64", True)
J_BASIS = (
    np.diag([1.0, -1.0, 0.0]),
    np.diag([1.0, 1.0, -2.0]),
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
)


def observed_force(states, dt):
    rotation = states[:-1, 6:].reshape(-1, 3, 3)
    rate = states[:-1, 3:6]
    middle = rotation @ Rotation.from_rotvec(0.5 * dt * rate).as_matrix()
    dv = np.diff(states[:, :3], axis=0) / dt
    force = np.einsum("tji,tj->ti", middle, dv - np.array([0.0, 0.0, -9.80665]))
    body_velocity = np.einsum("tji,tj->ti", rotation, states[:-1, :3])
    return force, body_velocity


def fit_inertia(states, commands, dt):
    """Use low-command free motion for trace-normalized inertia; otherwise I."""
    rate = 0.5 * (states[:-1, 3:6] + states[1:, 3:6])
    derivative = np.diff(states[:, 3:6], axis=0) / dt
    design = np.stack(
        [
            np.array(
                [basis @ d + np.cross(w, basis @ w) for w, d in zip(rate, derivative)]
            )
            for basis in J_BASIS
        ],
        axis=-1,
    ).reshape(-1, 5)
    weight = np.exp(-0.5 * (np.linalg.norm(commands, axis=1) / 0.03) ** 2)
    root = np.repeat(np.sqrt(weight), 3)
    beta = np.linalg.lstsq(
        np.vstack((design * root[:, None], 0.01 * np.eye(5))),
        np.r_[-derivative.ravel() * root, np.zeros(5)],
        rcond=None,
    )[0]
    return np.eye(3) + sum(beta[i] * J_BASIS[i] for i in range(5))


@partial(jax.jit, static_argnames=("dt",))
def _latent_trace_compiled(commands, q, coeff, dt):
    root = jnp.sqrt(q)
    targets = (
        jnp.sign(commands)
        * (jnp.sqrt(jnp.abs(commands) + q) - root)
        / (jnp.sqrt(1 + q) - root)
    )

    def update(state, command):
        rising = command >= state
        c0 = jnp.where(rising, coeff[0], coeff[2])
        c1 = jnp.where(rising, coeff[1], coeff[3])
        h = dt / 4

        def substep(_, current):
            rate = c0 + c1 * (jnp.abs(command) + jnp.abs(current))
            midpoint = command + (current - command) * jnp.exp(-0.5 * h * rate)
            midpoint_rate = c0 + c1 * (jnp.abs(command) + jnp.abs(midpoint))
            return command + (current - command) * jnp.exp(-h * midpoint_rate)

        following = jax.lax.fori_loop(0, 4, substep, state)
        return following, following

    _, following = jax.lax.scan(update, targets[0], targets)
    return jnp.concatenate((targets[0][None, :], following), axis=0)


def latent_trace(commands, q, coeff, dt):
    return np.asarray(
        _latent_trace_compiled(
            jnp.asarray(commands), jnp.asarray(q), jnp.asarray(coeff), dt
        )
    )


def rate_forecast(start_rate, latent, future_start, J, Ct, dt=0.01):
    m = latent.shape[1]
    H = Ct[: 3 * m].reshape(m, 3).T
    B = Ct[3 * m :].reshape(3, 2 + 2 * m)
    w = start_rate.copy()
    out = []
    for t in range(future_start, len(latent) - 1):
        rb = latent[t]
        rn = latent[t + 1]
        rm = 0.5 * (rb + rn)
        dr = (rn - rb) / dt

        def derivative(rate, s, latent_derivative):
            torque = (
                B[:, 0]
                + B[:, 1 : 1 + m] @ s
                + B[:, 1 + m : 1 + 2 * m] @ (s * s)
                - B[:, -1] * rate
            )
            torque += (
                H @ latent_derivative + np.cross(rate, H @ s) - np.cross(rate, J @ rate)
            )
            return np.linalg.solve(J, torque)

        first = derivative(w, rb, dr)
        w = w + dt * derivative(w + 0.5 * dt * first, rm, dr)
        out.append(w.copy())
    return np.asarray(out)


class ReadoutProblem:
    """Cache observed physics while the five response parameters vary."""

    def __init__(self, states, commands, J, begin, end, dt):
        self.m = commands.shape[1]
        self.n = end - begin
        self.begin = begin
        self.end = end
        self.dt = dt
        m, n = self.m, self.n
        force, velocity = observed_force(states, dt)
        v = velocity[begin:end]
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        self.force_truth = force[begin:end]
        self.force_base = np.column_stack(
            (np.ones(n), np.zeros((n, 2 * m)), v, v * speed)
        )
        pf = np.r_[0.0, np.full(2 * m, 0.001), np.full(6, 0.01)]
        self.force_penalty = np.diag(np.sqrt(pf))
        self.force_response = np.vstack((self.force_truth, np.zeros((len(pf), 3))))

        w = 0.5 * (states[:-1, 3:6] + states[1:, 3:6])
        dw = np.diff(states[:, 3:6], axis=0) / dt
        self.rate_truth = (dw @ J.T + np.cross(w, w @ J.T))[begin:end].reshape(-1)
        self.cross_e = np.stack(
            [np.cross(w[begin:end], axis) for axis in np.eye(3)], axis=-1
        )
        self.width = 2 + 2 * m
        self.torque_base = np.zeros((n, 3, 3 * m + 3 * self.width))
        for axis in range(3):
            off = 3 * m + axis * self.width
            self.torque_base[:, axis, off] = 1.0
            self.torque_base[:, axis, off + 1 + 2 * m] = -w[begin:end, axis]
        pt = np.r_[
            np.full(3 * m, 0.1), np.tile(np.r_[0.01, np.full(2 * m, 1e-6), 0.001], 3)
        ]
        self.torque_ridge = pt
        self.damping = np.array(
            [3 * m + axis * self.width + self.width - 1 for axis in range(3)]
        )
        self.damping_basis = np.eye(len(pt))[:, self.damping]

    def solve(self, r):
        m, n = self.m, self.n
        mid = 0.5 * (r[self.begin : self.end] + r[self.begin + 1 : self.end + 1])
        dr = (r[self.begin + 1 : self.end + 1] - r[self.begin : self.end]) / self.dt
        Xf = self.force_base.copy()
        Xf[:, 1 : 1 + m] = mid
        Xf[:, 1 + m : 1 + 2 * m] = mid * mid
        Cf = np.linalg.lstsq(
            np.vstack((Xf, self.force_penalty)), self.force_response, rcond=None
        )[0]
        rf = Xf @ Cf - self.force_truth

        X = self.torque_base.copy()
        eye = np.eye(3)
        for j in range(m):
            X[:, :, 3 * j : 3 * j + 3] = (
                dr[:, j, None, None] * eye[None, :, :]
                + mid[:, j, None, None] * self.cross_e
            )
        for axis in range(3):
            off = 3 * m + axis * self.width
            X[:, axis, off + 1 : off + 1 + m] = mid
            X[:, axis, off + 1 + m : off + 1 + 2 * m] = mid * mid
        X = X.reshape(3 * n, -1)
        gram = X.T @ X
        gram.flat[:: len(gram) + 1] += self.torque_ridge
        target = X.T @ self.rate_truth
        factor = cho_factor(gram, check_finite=False)
        unconstrained = cho_solve(factor, target, check_finite=False)
        inverse_damping = cho_solve(factor, self.damping_basis, check_finite=False)
        Ct = None
        for mask in range(8):
            active = [axis for axis in range(3) if mask & (1 << axis)]
            if active:
                covariance = inverse_damping[np.ix_(self.damping[active], active)]
                multiplier = np.linalg.solve(
                    covariance, unconstrained[self.damping[active]]
                )
                candidate = unconstrained - inverse_damping[:, active] @ multiplier
            else:
                multiplier = np.empty(0)
                candidate = unconstrained
            if np.all(candidate[self.damping] >= -1e-8) and np.all(multiplier <= 1e-8):
                Ct = candidate
                break
        if Ct is None:
            raise ArithmeticError("nonnegative damping solve had no KKT point")
        rt = X @ Ct - self.rate_truth
        return rf, rt, Cf, Ct


def fit(states, commands, begin=100, end=150, dt=0.01, J=None):
    if J is None:
        J = fit_inertia(states, commands, dt)
    problem = ReadoutProblem(states, commands, J, begin, end, dt)
    initial = latent_trace(commands, 0.01, np.array([15.0, 2.0, 8.0, 5.0]), dt)
    initial_rf, _, _, _ = problem.solve(initial)
    force_scale = np.maximum(
        np.sqrt(np.mean(initial_rf**2, axis=0)), np.array([0.02, 0.02, 0.05])
    )

    def loss(theta, detail=False):
        q = np.exp(theta[0])
        coeff = np.exp(theta[1:])
        r = latent_trace(commands, q, coeff, dt)
        rf, rt, Cf, Ct = problem.solve(r)
        # Common physical-scale residuals, with simple complexity shrinkage.
        value = np.mean((rf / force_scale) ** 2) + np.mean((rt / 1.0) ** 2)
        if detail:
            return value, q, coeff, r, rf, rt, Cf, Ct, J
        return value

    started = time.perf_counter()
    solves = []
    for init in (
        (0.001, 10.0, 1.0, 5.0, 1.0),
        (0.01, 15.0, 2.0, 8.0, 5.0),
        (0.1, 20.0, 0.1, 20.0, 0.1),
    ):
        x = np.log(init)
        bounds = [(np.log(1e-7), np.log(10.0))] + [(np.log(0.01), np.log(100.0))] * 4
        result = minimize(
            loss,
            x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 120, "ftol": 1e-8},
        )
        solves.append(result)
    best = min(solves, key=lambda z: z.fun)
    return loss(best.x, True), best, time.perf_counter() - started


def screen():
    spec = read(SPEC)
    cases = []
    for name in spec["smoke_cases"]:
        case = source_case(spec, name, "smoke")
        x, u, dt = case["states"], case["commands"], case["dt"]
        for origin in case["origins"]:
            cases.append(
                (
                    name,
                    int(origin),
                    x[: origin + 1],
                    u[:origin],
                    u[: origin + case["horizon"]],
                    x,
                    dt,
                    case["begin"],
                )
            )
    for arm in ("085", "140"):
        for branch in ("control", "orthogonal"):
            saved = np.load(ROOT / f"quad-arm-{arm}-heldout-{branch}.npz")
            control = np.load(ROOT / f"quad-arm-{arm}-heldout-control.npz")
            x = observed(saved["states"])
            u = saved["commands"]
            xt = observed(control["states"])
            ut = control["commands"]
            cases.append(
                (
                    f"highspin-{arm}-{branch}",
                    150,
                    x[:151],
                    u[:150],
                    ut[:175],
                    xt,
                    0.01,
                    100,
                )
            )
    for name, origin, prefix, training, issued, truth_states, dt, begin in cases:
        J = fit_inertia(prefix, training, dt)
        (value, q, coeff, _, rf, rt, _, ct, _), _, seconds = fit(
            prefix, training, begin, origin, dt, J
        )
        latent = latent_trace(issued, q, coeff, dt)
        forecast = rate_forecast(truth_states[origin, 3:6], latent, origin, J, ct, dt)
        horizon = len(issued) - origin
        error = np.linalg.norm(forecast[-1] - truth_states[origin + horizon, 3:6])
        print(
            name,
            origin,
            "rate_error_rad_s",
            round(error, 6),
            "fit_seconds",
            round(seconds, 3),
            "objective",
            round(value, 6),
            "force_rmse",
            np.round(np.sqrt(np.mean(rf * rf, axis=0)), 4),
            "torque_rmse",
            round(np.sqrt(np.mean(rt * rt)), 4),
            flush=True,
        )


def response_screen():
    source = ARTIFACTS / "heldout-quad-v1/source"
    truth_root = ARTIFACTS / "high-spin-response-v1/truth"
    epsilon = 0.05
    for arm in ("085", "140"):
        name = f"quad-arm-{arm}-heldout"
        with np.load(source / name / "stream.npz") as tape:
            states = observed(tape["states"])
            commands = tape["commands"]
        with np.load(truth_root / f"{name}.npz") as saved:
            rows = saved["rows"]
            truth = saved["jacobian"]
            assert np.array_equal(saved["commands"], commands[rows])
        for index, origin in enumerate(rows):
            origin = int(origin)
            prefix, training = states[: origin + 1], commands[:origin]
            J = fit_inertia(prefix, training, 0.01)
            begin = max(50, origin - 50)
            (value, q, coeff, _, _, _, _, ct, _), _, seconds = fit(
                prefix, training, begin, origin, 0.01, J
            )
            issued = commands[: origin + 25]
            factual_latent = latent_trace(issued, q, coeff, 0.01)
            factual = rate_forecast(states[origin, 3:6], factual_latent, origin, J, ct)
            rate_error = np.linalg.norm(factual[-1] - states[origin + 25, 3:6])
            response = []
            for channel in range(commands.shape[1]):
                low, high = issued.copy(), issued.copy()
                low[origin, channel] = np.clip(low[origin, channel] - epsilon, 0, 1)
                high[origin, channel] = np.clip(high[origin, channel] + epsilon, 0, 1)
                low_latent = latent_trace(low, q, coeff, 0.01)
                high_latent = latent_trace(high, q, coeff, 0.01)
                minus = rate_forecast(states[origin, 3:6], low_latent, origin, J, ct)
                plus = rate_forecast(states[origin, 3:6], high_latent, origin, J, ct)
                response.append(
                    (plus - minus) / (high[origin, channel] - low[origin, channel])
                )
            response = np.stack(response, axis=-1)
            errors = {}
            for ms in (10, 50, 100, 250):
                step = ms // 10 - 1
                errors[ms] = round(
                    float(
                        np.linalg.norm(response[step] - truth[index, step])
                        / np.linalg.norm(truth[index, step])
                    ),
                    4,
                )
            print(
                "response",
                name,
                origin,
                errors,
                "rate_error_rad_s",
                round(rate_error, 6),
                "fit_seconds",
                round(seconds, 3),
                "objective",
                round(value, 6),
                flush=True,
            )


if __name__ == "__main__":
    screen()
    response_screen()
