"""Independent NumPy verification of recorded online proposal arithmetic.

This module never imports a learner, calls a model, or differentiates a residual.
Recorded derivative actions are checked on their visited Krylov span; separate
source-bound derivative recomputation owns their provenance outside that span.
"""

import numpy as np

CHECKPOINTS = (4, 8, 16, 32, 64, 128)
PROBES = ("four_raw", "four_trusted", "reference_raw", "reference_trusted")
STATUSES = (
    "running",
    "converged",
    "capped",
    "stalled",
    "breakdown",
    "nonfinite",
    "metric_underflow",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _Checks:
    def __init__(self):
        self.count = 0
        self.maximum_absolute = 0.0
        self.maximum_krylov_relative = 0.0

    def close(self, actual, expected, name, *, production=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        _require(actual.shape == expected.shape, name + " shape differs")
        tolerance = 1e-9 if production else 1e-10
        _require(
            np.allclose(actual, expected, rtol=1e-8, atol=tolerance, equal_nan=True),
            name + " arithmetic differs",
        )
        finite = np.isfinite(actual) & np.isfinite(expected)
        if finite.any():
            self.maximum_absolute = max(
                self.maximum_absolute,
                float(np.max(np.abs(actual[finite] - expected[finite]))),
            )
        self.count += 1

    def flag(self, actual, expected, name):
        _require(bool(actual) == bool(expected), name + " flag differs")
        self.count += 1

    def krylov(self, left, right, name):
        _require(
            np.isfinite(left).all() and np.isfinite(right).all(), name + " is nonfinite"
        )
        scale = max(1.0, float(np.linalg.norm(left)), float(np.linalg.norm(right)))
        relative = float(np.linalg.norm(left - right)) / scale
        _require(relative <= 1e-8, name + " normwise identity differs")
        self.maximum_krylov_relative = max(self.maximum_krylov_relative, relative)
        self.count += 1


def _groups(raw):
    return np.stack(
        [
            np.sum(raw[..., left:right] ** 2, axis=-1)
            for left, right in ((0, 3), (3, 6), (6, 15))
        ],
        axis=-1,
    )


def _data_loss(raw, weight):
    groups = _groups(raw)
    huber = np.where(groups <= 1, 0.5 * groups, np.sqrt(np.maximum(1.0, groups)) - 0.5)
    return float(np.sum(weight * huber))


def _metric(residual, preconditioner):
    return float(np.sum(residual * (residual / preconditioner)))


def _probe(
    theta, prior, gradient, raw, trial_raw, weight, irls, delta, projected, damping
):
    data = _data_loss(raw, weight)
    penalty = float(0.5 * np.sum(prior * theta**2))
    trial_data = _data_loss(trial_raw, weight)
    trial_prior = float(0.5 * np.sum(prior * (theta + delta) ** 2))
    predicted = float(
        -gradient @ delta
        - 0.5 * (np.sum(weight * irls * projected**2) + np.sum(prior * delta**2))
    )
    current, trial = data + penalty, trial_data + trial_prior
    finite = bool(
        all(
            np.isfinite(value).all()
            for value in (
                theta + delta,
                delta,
                gradient,
                trial_raw,
                projected,
                current,
                trial,
                predicted,
            )
        )
    )
    gain = (current - trial) / predicted if finite and predicted > 0 else -np.inf
    return dict(
        current_data=data,
        current_prior=penalty,
        current=current,
        trial_data=trial_data,
        trial_prior=trial_prior,
        trial=trial,
        predicted=predicted,
        actual=current - trial,
        gain=float(gain),
        finite=finite,
        would_accept=bool(finite and trial < current and gain >= 0.1),
        data_change=trial_data - data,
        prior_change=trial_prior - penalty,
        damped_q=float(-predicted + 0.5 * damping * (delta @ delta)),
    )


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _gradient_summary(data, prior, metric):
    left, right = np.sqrt(metric) * data, np.sqrt(metric) * prior
    nl, nr = np.linalg.norm(left), np.linalg.norm(right)
    total = np.linalg.norm(left + right)
    return dict(
        data_norm=_number(nl),
        prior_norm=_number(nr),
        total_norm=_number(total),
        cosine=_number((left @ right) / (nl * nr)) if nl > 0 and nr > 0 else None,
        cancellation_ratio=_number(total / (nl + nr)) if nl + nr > 0 else None,
    )


def _summary(checks, actual, expected, path="summary"):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), path + " is not an object")
        for key, value in expected.items():
            _require(key in actual, path + " missing " + key)
            _summary(checks, actual[key], value, path + "." + key)
    elif isinstance(expected, list):
        _require(
            isinstance(actual, list) and len(actual) == len(expected),
            path + " length differs",
        )
        for index, value in enumerate(expected):
            _summary(checks, actual[index], value, path + "." + str(index))
    elif expected is None or isinstance(expected, (str, bool)):
        _require(actual == expected, path + " differs")
    else:
        checks.close(
            np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), path
        )


def _verify_arrays(summary, arrays):
    """Verify saved PCG/IRLS/prior/trust arithmetic; raise ValueError on mismatch."""
    _require(
        summary.get("format") == "online-proposal-diagnostics-v1",
        "diagnostic format differs",
    )
    a = {key: np.asarray(value) for key, value in arrays.items()}
    checks = _Checks()
    theta, prior, preconditioner = (
        a[key] for key in ("theta", "prior", "preconditioner")
    )
    gradient, data_gradient, prior_gradient = (
        a[key] for key in ("gradient", "gradient_data", "gradient_prior")
    )
    raw, weight, irls = (a[key] for key in ("raw", "weight", "irls"))
    damping = float(a["damping"])
    parameters = len(theta)
    stop = 0
    for leaf in summary["parameter_schema"]:
        _require(
            isinstance(leaf["path"], str) and leaf["start"] == stop,
            "parameter schema ordering differs",
        )
        _require(
            all(type(value) is int and value >= 0 for value in leaf["shape"]),
            "parameter schema shape differs",
        )
        stop += int(np.prod(leaf["shape"]))
        _require(leaf["stop"] == stop, "parameter schema size differs")
    _require(stop == parameters, "parameter schema does not cover theta")
    _require(
        theta.shape
        == prior.shape
        == preconditioner.shape
        == gradient.shape
        == data_gradient.shape
        == prior_gradient.shape
        == (parameters,),
        "parameter vector shapes differ",
    )
    _require(raw.ndim == 3 and raw.shape[-1] == 15, "residual shape differs")
    _require(
        weight.shape == (len(raw), 1, 1) and irls.shape == raw.shape,
        "weight shapes differ",
    )
    _require(np.isfinite(weight).all() and np.all(weight >= 0), "invalid data weights")
    checks.close(
        np.asarray(weight.sum() * raw.shape[1] * 3),
        np.asarray(1.0),
        "role weight normalization",
    )
    with np.errstate(all="ignore"):
        expected_irls = np.repeat(
            1 / np.sqrt(np.maximum(1.0, _groups(raw))), (3, 3, 9), axis=-1
        )
        checks.close(irls, expected_irls, "frozen radial IRLS")
        checks.close(prior_gradient, prior * theta, "prior gradient")
        checks.close(gradient, data_gradient + prior_gradient, "total gradient")
        checks.close(preconditioner, prior + damping, "preconditioner")
        expected_radius = float(
            np.maximum(1.0, 0.5 * np.linalg.norm(np.sqrt(weight) * raw))
        )
        checks.close(a["radius"], np.asarray(expected_radius), "forecast trust radius")

        count = len(a["trace_direction"])
        _require(0 <= count <= 128, "reference budget differs")
        _require(int(a["iteration"]) == count, "iteration counter differs")
        for key in ("trace_direction", "trace_curvature"):
            _require(a[key].shape == (count, parameters), key + " shape differs")
        _require(
            a["trace_projected"].shape == (count, *raw.shape),
            "trace projection shape differs",
        )
        for key in (
            "alpha",
            "beta",
            "rho",
            "next_rho",
            "denominator",
            "valid",
            "active",
            "finite",
        ):
            _require(
                a["trace_" + key].shape == (count,),
                "trace scalar shape differs: " + key,
            )
        delta = np.zeros_like(theta)
        residual = -gradient.copy()
        direction = residual / preconditioner
        rho = float(residual @ direction)
        finite = bool(
            np.isfinite(prior).all()
            and np.all(prior >= 0)
            and np.isfinite(gradient).all()
            and np.isfinite(preconditioner).all()
            and np.all(preconditioner > 0)
        )
        initial_finite = finite
        states = [(delta.copy(), residual.copy(), finite)]
        combined_projection = np.zeros_like(raw)
        projections = [combined_projection.copy()]
        terminal = None
        for index in range(count):
            p, projected, action = (
                a[key][index]
                for key in ("trace_direction", "trace_projected", "trace_curvature")
            )
            checks.close(p, direction, "PCG search direction")
            checks.close(a["trace_rho"][index], np.asarray(rho), "PCG incoming rho")
            # Audit each recorded transition locally. Re-solving with NumPy
            # reductions would amplify roundoff in tiny late-CG ratios.
            rho = float(a["trace_rho"][index])
            denominator = float(p @ action)
            checks.close(
                a["trace_denominator"][index],
                np.asarray(denominator),
                "PCG denominator",
            )
            denominator = float(a["trace_denominator"][index])
            valid = bool(
                np.isfinite(action).all()
                and np.isfinite(projected).all()
                and np.isfinite(rho)
                and np.isfinite(denominator)
                and (rho == 0 or (rho > 0 and denominator > 0))
            )
            active = valid and rho > 0
            alpha = rho / denominator if active else 0.0
            checks.close(a["trace_alpha"][index], np.asarray(alpha), "PCG alpha")
            alpha = float(a["trace_alpha"][index])
            following_delta = delta + alpha * p if active else delta.copy()
            following = residual - alpha * action if active else residual.copy()
            z = following / preconditioner
            next_rho = float(following @ z)
            valid &= bool(
                np.isfinite(z).all() and np.isfinite(next_rho) and next_rho >= 0
            )
            checks.close(
                a["trace_next_rho"][index], np.asarray(next_rho), "PCG outgoing rho"
            )
            next_rho = float(a["trace_next_rho"][index])
            beta = next_rho / rho if active else 0.0
            checks.close(a["trace_beta"][index], np.asarray(beta), "PCG beta")
            beta = float(a["trace_beta"][index])
            next_direction = z + beta * p if active else np.zeros_like(p)
            finite &= valid
            for key, value in (
                ("active", active),
                ("valid", valid),
                ("finite", finite),
            ):
                checks.flag(a["trace_" + key][index], value, "PCG " + key)
            delta, residual, direction, rho = (
                following_delta,
                following,
                next_direction,
                next_rho,
            )
            if active:
                combined_projection = combined_projection + alpha * projected
            states.append((delta.copy(), residual.copy(), finite))
            projections.append(combined_projection.copy())
            if not finite or rho == 0:
                terminal = index + 1
                _require(
                    terminal == count, "reference continued after terminal recurrence"
                )

        iterations = a["checkpoint_iterations"]
        _require(
            iterations.ndim == 1 and len(iterations) and iterations.dtype.kind in "iu",
            "checkpoint iteration shape differs",
        )
        _require(
            int(a["checkpoint_count"]) == len(iterations), "checkpoint counter differs"
        )
        _require(
            iterations[0] == 0
            and iterations[-1] == count
            and np.all(np.diff(iterations) > 0),
            "checkpoint ordering differs",
        )
        expected_iterations = sorted(
            {0, count, *(value for value in CHECKPOINTS if value <= count)}
        )
        _require(
            iterations.tolist() == expected_iterations, "checkpoint inventory differs"
        )
        for key in ("delta", "recursive_residual", "true_residual", "curvature"):
            _require(
                a["checkpoint_" + key].shape == (len(iterations), parameters),
                "checkpoint vector shape differs: " + key,
            )
        _require(
            a["checkpoint_projected"].shape == (len(iterations), *raw.shape),
            "checkpoint projection shape differs",
        )
        for key in ("relative", "q", "gap_bound"):
            _require(
                a["checkpoint_" + key].shape == iterations.shape,
                "checkpoint scalar shape differs: " + key,
            )
        exact_zero = bool(np.all(gradient == 0))
        initial_metric = _metric(gradient, preconditioner)
        first_converged = None
        for index, iteration in enumerate(iterations):
            d, recurrence, _ = states[int(iteration)]
            hd, jd = a["checkpoint_curvature"][index], a["checkpoint_projected"][index]
            true = -gradient - hd
            value = _metric(true, preconditioner)
            relative = (
                (0.0 if exact_zero else 1.0 if initial_metric > 0 else np.inf)
                if iteration == 0
                else float(np.sqrt(value / initial_metric))
            )
            checks.close(a["checkpoint_delta"][index], d, "checkpoint delta")
            checks.close(
                a["checkpoint_recursive_residual"][index],
                recurrence,
                "checkpoint recurrence residual",
            )
            checks.close(
                a["checkpoint_true_residual"][index], true, "checkpoint true residual"
            )
            checks.close(
                a["checkpoint_relative"][index],
                np.asarray(relative),
                "checkpoint relative residual",
            )
            checks.close(
                a["checkpoint_q"][index],
                np.asarray(gradient @ d + 0.5 * (d @ hd)),
                "damped quadratic",
            )
            checks.close(
                a["checkpoint_gap_bound"][index],
                np.asarray(0.5 * value),
                "quadratic gap bound",
            )
            checks.close(
                jd, projections[int(iteration)], "checkpoint projection linearity"
            )
            if iteration == 0:
                checks.close(hd, np.zeros_like(theta), "zero checkpoint action")
            else:
                used = np.flatnonzero(a["trace_active"][:iteration])
                summed = np.sum(
                    a["trace_alpha"][used, None] * a["trace_curvature"][used], axis=0
                )
                checks.close(hd, summed, "checkpoint curvature linearity")
            if (
                relative <= 1e-6
                and np.isfinite(relative)
                and (iteration > 0 or exact_zero)
            ):
                if first_converged is None:
                    first_converged = int(iteration)
        if first_converged is not None:
            _require(
                first_converged == count,
                "reference did not stop at first convergence checkpoint",
            )

        if not initial_finite or not np.isfinite(initial_metric) or initial_metric < 0:
            expected_status = 5
            _require(count == 0, "nonfinite system entered PCG")
        elif exact_zero:
            expected_status = 1
            _require(count == 0, "zero gradient entered PCG")
        elif initial_metric == 0:
            expected_status = 6
            _require(count == 0, "underflowed metric entered PCG")
        else:
            _require(count > 0, "nonzero system has no iterations")
            true = a["checkpoint_true_residual"][-1]
            true_metric = _metric(true, preconditioner)
            relative = float(a["checkpoint_relative"][-1])
            action_finite = bool(
                np.isfinite(a["checkpoint_projected"][-1]).all()
                and np.isfinite(a["checkpoint_curvature"][-1]).all()
                and np.isfinite(relative)
                and true_metric >= 0
            )
            numerical = bool(
                not np.isfinite(a["trace_curvature"][-1]).all()
                or not np.isfinite(a["trace_projected"][-1]).all()
                or not np.isfinite(rho)
                or not np.isfinite(residual / preconditioner).all()
            )
            expected_status = (
                5
                if not action_finite or numerical
                else 1
                if relative <= 1e-6
                else 4
                if not bool(a["trace_valid"][-1])
                else 3
                if rho == 0
                else 2
                if count == 128
                else 0
            )
            _require(
                expected_status != 0, "reference stopped without a terminal condition"
            )
        _require(
            int(a["status"]) == expected_status, "reference termination status differs"
        )
        for name, expected in (
            ("delta", delta),
            ("residual", residual),
            ("direction", direction),
            ("rho", np.asarray(rho)),
        ):
            checks.close(a[name], expected, "terminal " + name)
        checks.flag(a["finite"], finite, "terminal finite")

        four_index = min(4, count)
        four_delta, _, four_finite = states[four_index]
        checks.close(a["four_delta"], four_delta, "four-step delta")
        checks.flag(a["four_finite"], four_finite, "four-step finite")
        _require(
            a["probe_delta"].shape == (4, parameters)
            and a["probe_projected"].shape
            == a["probe_trial_raw"].shape
            == (4, *raw.shape)
            and a["probe_shrink"].shape == (4,),
            "probe shapes differ",
        )
        for index, d in ((0, four_delta), (2, delta)):
            checks.close(a["probe_delta"][index], d, "unshrunk probe direction")
            checks.close(
                a["probe_projected"][index],
                projections[four_index if index == 0 else count],
                "unshrunk probe projection",
            )
            checks.close(a["probe_shrink"][index], np.asarray(1.0), "raw shrink")
            norm = float(np.linalg.norm(np.sqrt(weight) * a["probe_projected"][index]))
            shrink = float(np.minimum(1.0, expected_radius / np.maximum(norm, 1e-30)))
            checks.close(
                a["probe_shrink"][index + 1], np.asarray(shrink), "trust shrink"
            )
            checks.close(a["probe_delta"][index + 1], shrink * d, "trusted direction")
            checks.close(
                a["probe_projected"][index + 1],
                shrink * a["probe_projected"][index],
                "trusted projection",
            )

        probes = []
        for index, label in enumerate(PROBES):
            d, jd = a["probe_delta"][index], a["probe_projected"][index]
            row = _probe(
                theta,
                prior,
                gradient,
                raw,
                a["probe_trial_raw"][index],
                weight,
                irls,
                d,
                jd,
                damping,
            )
            row["finite"] &= four_finite if index < 2 else finite
            row["gain"] = (
                (row["current"] - row["trial"]) / row["predicted"]
                if row["finite"] and row["predicted"] > 0
                else -np.inf
            )
            row["would_accept"] = bool(
                row["finite"] and row["trial"] < row["current"] and row["gain"] >= 0.1
            )
            row.update(
                label=label,
                data_directional=float(data_gradient @ d),
                prior_directional=float(prior_gradient @ d),
            )
            probes.append(row)
        for key, value in (
            ("theta", theta + a["probe_delta"][1]),
            ("current", probes[1]["current"]),
            ("trial", probes[1]["trial"]),
            ("predicted", probes[1]["predicted"]),
        ):
            checks.close(
                a["production_" + key],
                np.asarray(value),
                "production versus instrumented " + key,
                production=True,
            )
        checks.flag(
            a["production_finite"],
            probes[1]["finite"],
            "production versus instrumented finite",
        )

        valid_rows = np.flatnonzero(a["trace_active"] & a["trace_valid"])
        if len(valid_rows):
            p = a["trace_direction"][valid_rows]
            jp = a["trace_projected"][valid_rows].reshape(len(valid_rows), -1)
            hp = a["trace_curvature"][valid_rows]
            norms = np.sqrt(np.sum(p * preconditioner * p, axis=1))
            _require(
                np.isfinite(norms).all() and np.all(norms > 0),
                "invalid Krylov normalization",
            )
            p, jp, hp = p / norms[:, None], jp / norms[:, None], hp / norms[:, None]
            w = np.broadcast_to(weight * irls, raw.shape).ravel()
            checks.krylov(p @ data_gradient, jp @ (w * raw.ravel()), "Krylov gradient")
            left = p @ (hp - p * preconditioner).T
            right = (jp * w) @ jp.T
            checks.krylov(left, right, "Krylov Hessian")
            checks.krylov(left, left.T, "Krylov symmetry")

        probe_summary = []
        for index, row in enumerate(probes):
            probe_summary.append(
                dict(
                    label=row["label"],
                    finite=row["finite"],
                    shrink=_number(a["probe_shrink"][index]),
                    projection_norm=_number(
                        np.linalg.norm(np.sqrt(weight) * a["probe_projected"][index])
                    ),
                    data_loss=_number(row["trial_data"]),
                    prior_loss=_number(row["trial_prior"]),
                    total_loss=_number(row["trial"]),
                    data_improvement=_number(-row["data_change"]),
                    prior_improvement=_number(-row["prior_change"]),
                    improvement=_number(row["actual"]),
                    predicted_reduction=_number(row["predicted"]),
                    damped_quadratic=_number(row["damped_q"]),
                    damped_quadratic_decrease=_number(-row["damped_q"]),
                    gain=_number(row["gain"]),
                    would_accept=row["would_accept"],
                    gradient_data_dot=_number(row["data_directional"]),
                    gradient_prior_dot=_number(row["prior_directional"]),
                )
            )
        _summary(
            checks,
            summary,
            dict(
                reference=dict(
                    status=STATUSES[expected_status],
                    iterations=count,
                    relative_residual=_number(a["checkpoint_relative"][-1]),
                ),
                current=dict(
                    data_loss=_number(probes[0]["current_data"]),
                    prior_loss=_number(probes[0]["current_prior"]),
                    total_loss=_number(probes[0]["current"]),
                ),
                radius=_number(expected_radius),
                probes=probe_summary,
                gradients=dict(
                    coordinate=_gradient_summary(
                        data_gradient, prior_gradient, np.ones_like(theta)
                    ),
                    preconditioned=_gradient_summary(
                        data_gradient, prior_gradient, 1 / preconditioner
                    ),
                ),
                work=dict(
                    residual_linearizations=1,
                    gradient_vjp_calls=1,
                    cg_iterations=count,
                    cg_jvp_vjp_pairs=count,
                    checkpoint_jvp_vjp_pairs=len(iterations) - 1,
                    raw_probe_jvp_calls=2,
                    trial_residual_calls=4,
                    production_calls=0,
                    applied_updates=0,
                    observe_calls=0,
                    initialization_calls=0,
                ),
                production_comparison_passed=True,
            ),
        )

    return dict(
        verified=True,
        arithmetic_checks=checks.count,
        maximum_absolute_difference=checks.maximum_absolute,
        maximum_krylov_relative_difference=checks.maximum_krylov_relative,
        iterations=count,
        krylov_directions=len(valid_rows),
        model_calls=0,
        optimizer_calls=0,
        reference_status=STATUSES[expected_status],
    )


def verify_arrays(summary, arrays):
    """Check saved math and summaries without derivative, model or optimizer calls."""
    try:
        return _verify_arrays(summary, arrays)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("invalid proposal evidence schema") from error
