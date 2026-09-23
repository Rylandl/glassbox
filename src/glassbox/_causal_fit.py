"""Fit generic causal actuator dynamics from observed episode motion."""

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import RLock

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from ._causal_actuator import CausalActuatorModel
from ._causal_actuator import latent_trace as _latent_trace

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


def _inertia_rows(states, commands, dt):
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
    return design * root[:, None], -derivative.ravel() * root


def fit_inertia_many(segments, dt):
    """Use low-command free motion for trace-normalized inertia; otherwise I."""
    rows = [_inertia_rows(states, commands, dt) for states, commands in segments]
    design = np.vstack([item[0] for item in rows])
    target = np.concatenate([item[1] for item in rows])
    beta = np.linalg.lstsq(
        np.vstack((design, 0.01 * np.eye(5))),
        np.r_[target, np.zeros(5)],
        rcond=None,
    )[0]
    return np.eye(3) + sum(beta[i] * J_BASIS[i] for i in range(5))


def fit_inertia(states, commands, dt):
    return fit_inertia_many(((states, commands),), dt)


@partial(jax.jit, static_argnames=("dt",))
def _latent_trace_compiled(commands, q, coeff, dt):
    return _latent_trace(commands, q, coeff, dt)


def latent_trace(commands, q, coeff, dt):
    with jax.enable_x64(True):
        return np.asarray(
            _latent_trace_compiled(
                jnp.asarray(commands), jnp.asarray(q), jnp.asarray(coeff), dt
            )
        )


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
        # A zero-command, zero-speed force is the data-independent prior. An
        # unpenalized intercept can cancel large slopes on short prefixes.
        pf = np.r_[0.01, np.full(2 * m, 0.001), np.full(6, 0.01)]
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
        mid = 0.5 * (r[self.begin : self.end] + r[self.begin + 1 : self.end + 1])
        dr = (r[self.begin + 1 : self.end + 1] - r[self.begin : self.end]) / self.dt
        return self._solve_from_mid(mid, dr)

    def _solve_from_mid(self, mid, dr):
        m, n = self.m, self.n
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


class JointReadoutProblem(ReadoutProblem):
    """One force/torque head shared across independent episode segments."""

    def __init__(self, segments, J, dt):
        parts = [
            ReadoutProblem(states, commands, J, begin, len(commands), dt)
            for states, commands, begin in segments
        ]
        first = parts[0]
        self.m = first.m
        self.n = sum(part.n for part in parts)
        self.dt = dt
        self.width = first.width
        self.parts = parts
        self.force_base = np.concatenate([part.force_base for part in parts])
        self.force_truth = np.concatenate([part.force_truth for part in parts])
        self.force_penalty = first.force_penalty
        self.force_response = np.vstack(
            (self.force_truth, np.zeros((len(first.force_penalty), 3)))
        )
        self.rate_truth = np.concatenate([part.rate_truth for part in parts])
        self.cross_e = np.concatenate([part.cross_e for part in parts])
        self.torque_base = np.concatenate([part.torque_base for part in parts])
        self.torque_ridge = first.torque_ridge
        self.damping = first.damping
        self.damping_basis = first.damping_basis

    def solve(self, traces):
        mid = np.concatenate(
            [
                0.5
                * (trace[part.begin : part.end] + trace[part.begin + 1 : part.end + 1])
                for trace, part in zip(traces, self.parts, strict=True)
            ]
        )
        dr = np.concatenate(
            [
                (trace[part.begin + 1 : part.end + 1] - trace[part.begin : part.end])
                / self.dt
                for trace, part in zip(traces, self.parts, strict=True)
            ]
        )
        return self._solve_from_mid(mid, dr)


def _optimize(problem, command_tapes, dt, J, joint):
    def traces(q, coeff):
        result = [latent_trace(u, q, coeff, dt) for u in command_tapes]
        return result if joint else result[0]

    initial = traces(0.01, np.array([15.0, 2.0, 8.0, 5.0]))
    initial_rf, _, _, _ = problem.solve(initial)
    force_scale = np.maximum(
        np.sqrt(np.mean(initial_rf**2, axis=0)), np.array([0.02, 0.02, 0.05])
    )

    def loss(theta, detail=False):
        q = np.exp(theta[0])
        coeff = np.exp(theta[1:])
        r = traces(q, coeff)
        rf, rt, Cf, Ct = problem.solve(r)
        # Common physical-scale residuals, with simple complexity shrinkage.
        value = np.mean((rf / force_scale) ** 2) + np.mean((rt / 1.0) ** 2)
        # A weak, data-decaying zero/full-command prior on actuator rise/fall
        # time constants. Either direction remains free when data supports it.
        zero_asymmetry = np.log(coeff[2] / coeff[0])
        full_asymmetry = np.log((coeff[2] + 2 * coeff[3]) / (coeff[0] + 2 * coeff[1]))
        value += 0.03 / problem.n * (zero_asymmetry**2 + full_asymmetry**2)
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


def fit(states, commands, begin=100, end=150, dt=0.01, J=None):
    if J is None:
        J = fit_inertia(states, commands, dt)
    problem = ReadoutProblem(states, commands, J, begin, end, dt)
    return _optimize(problem, (commands,), dt, J, False)


def fit_segments(segments, dt_s):
    """Fit one shared model across complete, separately reset segments."""
    segments = tuple(segments)
    if not segments:
        raise ValueError("at least one training segment is required")
    channels = np.asarray(segments[0][1]).shape[1]
    checked = []
    for states, commands, begin in segments:
        states, commands = map(np.asarray, (states, commands))
        if (
            states.ndim != 2
            or states.shape[1] != 15
            or commands.ndim != 2
            or commands.shape[1] != channels
            or len(states) != len(commands) + 1
            or len(commands) < 5
            or not np.isfinite(states).all()
            or not np.isfinite(commands).all()
            or type(begin) is not int
            or not 0 <= begin <= len(commands) - 5
        ):
            raise ValueError("invalid training segment")
        checked.append((states, commands, begin))
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("invalid sample interval")
    J = fit_inertia_many(((s, u) for s, u, _ in checked), dt_s)
    problem = JointReadoutProblem(checked, J, dt_s)
    detail, optimization, seconds = _optimize(
        problem, tuple(u for _, u, _ in checked), dt_s, J, True
    )
    objective, q, coeff, _, force_error, torque_error, Cf, Ct, _ = detail
    model = CausalActuatorModel(dt_s, q, coeff, J, Cf, Ct)
    report = {
        "objective": float(objective),
        "force_rmse_m_s2": np.sqrt(np.mean(force_error**2, axis=0)).tolist(),
        "torque_equation_rmse": float(np.sqrt(np.mean(torque_error**2))),
        "fit_seconds": float(seconds),
        "optimizer_success": bool(optimization.success),
        "completed_transitions": sum(len(u) for _, u, _ in checked),
        "fit_segments": len(checked),
    }
    return model, report


def fit_episode(states, commands, dt_s, *, begin=0):
    """Fit one contiguous causal prefix and return an immutable model revision."""
    states, commands = map(np.asarray, (states, commands))
    if (
        states.ndim != 2
        or states.shape[1] != 15
        or commands.ndim != 2
        or commands.shape[1] < 1
        or len(states) != len(commands) + 1
        or len(commands) < 5
        or not np.isfinite(states).all()
        or not np.isfinite(commands).all()
        or not np.isfinite(dt_s)
        or dt_s <= 0
        or type(begin) is not int
        or not 0 <= begin <= len(commands) - 5
    ):
        raise ValueError(
            "fit needs a finite contiguous episode with completed transitions"
        )
    detail, optimization, seconds = fit(
        states, commands, begin, len(commands), float(dt_s)
    )
    objective, q, coeff, _, force_error, torque_error, Cf, Ct, J = detail
    model = CausalActuatorModel(dt_s, q, coeff, J, Cf, Ct)
    report = {
        "objective": float(objective),
        "force_rmse_m_s2": np.sqrt(np.mean(force_error**2, axis=0)).tolist(),
        "torque_equation_rmse": float(np.sqrt(np.mean(torque_error**2))),
        "fit_seconds": float(seconds),
        "optimizer_success": bool(optimization.success),
        "completed_transitions": len(commands),
        "fit_begin_transition": begin,
    }
    return model, report


class BackgroundFit:
    """Collect every observation; publish immutable full-prefix fits when ready."""

    def __init__(self, states, commands, dt_s, start_row=0):
        states, commands = map(np.asarray, (states, commands))
        model, report = fit_episode(states, commands, dt_s)
        self._states = [row.copy() for row in states]
        self._commands = [row.copy() for row in commands]
        self._dt_s = float(dt_s)
        self._cursor = int(start_row) + len(commands)
        self._published_cursor = self._cursor
        self._model = model
        self._report = report
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._fitting_cursor = None
        self._lock = RLock()

    @property
    def cursor(self):
        with self._lock:
            return self._cursor

    @property
    def published_cursor(self):
        with self._lock:
            self._poll()
            return self._published_cursor

    @property
    def model(self):
        with self._lock:
            self._poll()
            return self._model

    @property
    def report(self):
        with self._lock:
            self._poll()
            return dict(self._report, published_cursor=self._published_cursor)

    def _start(self):
        self._fitting_cursor = self._cursor
        states = np.stack(self._states)
        commands = np.stack(self._commands)
        self._future = self._executor.submit(fit_episode, states, commands, self._dt_s)

    def _poll(self):
        if self._future is not None and self._future.done():
            model, report = self._future.result()
            self._model = model
            self._report = report
            self._published_cursor = self._fitting_cursor
            self._future = None
            self._fitting_cursor = None
            if self._cursor > self._published_cursor:
                self._start()

    def observe(self, row, command, next_state):
        with self._lock:
            command, next_state = map(np.asarray, (command, next_state))
            if (
                row != self._cursor
                or command.shape != (self._model.input_count,)
                or next_state.shape != (15,)
                or not np.isfinite(command).all()
                or not np.isfinite(next_state).all()
            ):
                raise ValueError("observation row, command or state differs")
            self._commands.append(np.array(command, dtype=float, copy=True))
            self._states.append(np.array(next_state, dtype=float, copy=True))
            self._cursor += 1
            self._poll()
            if self._future is None:
                self._start()

    def wait_for_publication(self, timeout=None):
        with self._lock:
            if self._future is None and self._cursor > self._published_cursor:
                self._start()
            future = self._future
        if future is not None:
            future.result(timeout=timeout)
        with self._lock:
            self._poll()
            return self._published_cursor

    def snapshot(self):
        with self._lock:
            self._poll()
            return (
                np.stack(self._states),
                np.stack(self._commands),
                self._cursor,
                self._published_cursor,
                self._model,
                dict(self._report),
            )

    @classmethod
    def from_snapshot(
        cls, states, commands, dt_s, start_row, published_cursor, model, report
    ):
        if (
            not isinstance(model, CausalActuatorModel)
            or model.dt_s != dt_s
            or model.input_count != commands.shape[1]
            or states.shape != (len(commands) + 1, 15)
            or not start_row <= published_cursor <= start_row + len(commands)
        ):
            raise ValueError("invalid saved background fit")
        obj = cls.__new__(cls)
        obj._states = [row.copy() for row in states]
        obj._commands = [row.copy() for row in commands]
        obj._dt_s = float(dt_s)
        obj._cursor = start_row + len(commands)
        obj._published_cursor = published_cursor
        obj._model = model
        obj._report = dict(report)
        obj._executor = ThreadPoolExecutor(max_workers=1)
        obj._future = None
        obj._fitting_cursor = None
        obj._lock = RLock()
        return obj

    def close(self):
        self._executor.shutdown(wait=True)
