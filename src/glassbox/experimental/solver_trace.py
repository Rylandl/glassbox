"""Observe four saved optimizer failures without changing their numerical work."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from . import quasi_newton as optimizer
from .first_order_qualification import FirstOrderSolver

EXPONENTS = (-4, -8, -12, -16, -20, -24)


def _scalar(value):
    number = float(np.asarray(value))
    return number if np.isfinite(number) else None


def _array(value):
    array = np.asarray(value)
    if np.isfinite(array).all():
        return array.tolist()
    # Failed requests remain JSON-readable without inventing finite values.
    result = np.asarray(array, dtype=object)
    result[~np.isfinite(array)] = None
    return result.tolist()


@contextmanager
def trace_minimize(seed_blocks, seed_value, seed_gradient):
    """Wrap the existing backend during one sequential solve, then restore it."""
    seed = np.asarray(seed_blocks).copy()
    best_value = float(np.asarray(seed_value))
    trace = dict(
        seed=dict(
            canonical_blocks=_array(seed),
            value=_scalar(seed_value),
            gradient=_array(seed_gradient),
        ),
        requests=[],
        callbacks=[],
        raw_backend=None,
        retained_work=None,
    )
    original = optimizer.minimize

    def traced_minimize(function, initial, **options):
        callback = options["callback"]

        def requested(host):
            nonlocal best_value
            host_copy = np.asarray(host, dtype=np.float64).reshape(seed.shape).copy()
            canonical = np.asarray(host_copy, dtype=seed.dtype)
            row = dict(
                index=len(trace["requests"]),
                host_blocks=_array(host_copy),
                canonical_blocks=_array(canonical),
                value=None,
                gradient=None,
                accepted_before=len(trace["callbacks"]),
                cached_seed=bool(np.array_equal(canonical, seed)),
                best_updated=False,
                error=None,
            )
            trace["requests"].append(row)
            try:
                value, gradient = function(host)
            except Exception as error:
                row["error"] = type(error).__name__
                raise
            row["value"] = _scalar(value)
            row["gradient"] = _array(np.asarray(gradient).reshape(seed.shape))
            if (
                np.isfinite(value)
                and np.isfinite(gradient).all()
                and value < best_value
            ):
                best_value = float(value)
                row["best_updated"] = True
            return value, gradient

        def accepted(host):
            host_copy = np.asarray(host, dtype=np.float64).reshape(seed.shape).copy()
            canonical = np.asarray(host_copy, dtype=seed.dtype)
            result = callback(host)
            matched = next(
                (
                    row["index"]
                    for row in reversed(trace["requests"])
                    if row["error"] is None
                    and np.array_equal(np.asarray(row["host_blocks"]), host_copy)
                ),
                None,
            )
            trace["callbacks"].append(
                dict(
                    index=len(trace["callbacks"]),
                    host_blocks=_array(host_copy),
                    canonical_blocks=_array(canonical),
                    request_count=len(trace["requests"]),
                    matched_request=matched,
                )
            )
            return result

        result = original(requested, initial, **(options | {"callback": accepted}))
        trace["raw_backend"] = dict(
            x=_array(result.x),
            fun=_scalar(result.fun),
            jac=_array(result.jac),
            status=int(result.status),
            message=str(result.message),
            success=bool(result.success),
            nit=int(result.nit),
            nfev=int(result.nfev),
            njev=int(result.njev),
        )
        return result

    optimizer.minimize = traced_minimize
    try:
        yield trace
    finally:
        optimizer.minimize = original


def direction_audit(
    function, canonical_blocks, expected_value, expected_gradient, exponents
):
    """Measure all fixed feasible directions after the unchanged solve finishes."""
    point = np.asarray(canonical_blocks)
    gradient = np.asarray(expected_gradient)
    if (
        point.dtype != np.float32
        or not point.size
        or gradient.shape != point.shape
        or not np.isfinite(point).all()
        or np.any(np.abs(point) > 1)
    ):
        raise ValueError("direction audit requires finite bounded float32 blocks")
    value, fresh_gradient = function(point.copy())
    np.testing.assert_array_equal(value, expected_value)
    np.testing.assert_array_equal(fresh_gradient, expected_gradient)
    if not np.isfinite(value) or not np.isfinite(fresh_gradient).all():
        raise ValueError("direction audit base objective/gradient is nonfinite")
    value = float(value)
    gradient = np.asarray(fresh_gradient, dtype=np.float64)
    point64 = point.astype(np.float64)
    spacing = float(np.abs(np.spacing(np.float32(value))))
    directions = []
    for coordinate in range(point.size):
        for sign, label in ((-1.0, "negative"), (1.0, "positive")):
            direction = np.zeros(point.size, dtype=np.float64)
            direction[coordinate] = sign
            directions.append(
                dict(
                    index=len(directions),
                    name=f"coordinate_{coordinate}_{label}",
                    vector=direction.reshape(point.shape).tolist(),
                    zero_direction=False,
                )
            )
    projected = np.clip(point64 - gradient, -1.0, 1.0) - point64
    norm = float(np.max(np.abs(projected)))
    directions.append(
        dict(
            index=len(directions),
            name="projected_descent",
            vector=(projected / norm if norm else projected).tolist(),
            zero_direction=norm == 0.0,
        )
    )
    audit = dict(
        base=dict(
            canonical_blocks=point.tolist(),
            value=value,
            gradient=gradient.tolist(),
            objective_spacing=spacing,
        ),
        step_exponents=list(exponents),
        directions=directions,
        probes=[],
        diagnostic_function_calls=1,
    )
    for direction in directions:
        vector = np.asarray(direction["vector"], dtype=np.float64)
        for exponent in exponents:
            step = float(2.0**exponent)
            host = point64 + step * vector
            row = dict(
                direction_index=direction["index"],
                exponent=exponent,
                requested_step=step,
                host_blocks=host.tolist(),
                canonical_blocks=None,
                actual_displacement=None,
                status="zero_direction"
                if direction["zero_direction"]
                else "infeasible",
                value=None,
                gradient=None,
                objective_spacing=None,
                observed_change=None,
                predicted_change=None,
                predicted_change_in_spacings=None,
                absolute_linearization_discrepancy=None,
                observed_predicted_ratio=None,
            )
            audit["probes"].append(row)
            if direction["zero_direction"] or np.any(np.abs(host) > 1.0):
                continue
            canonical = host.astype(np.float32)
            displacement = canonical.astype(np.float64) - point64
            probe_value, probe_gradient = function(canonical)
            audit["diagnostic_function_calls"] += 1
            if (
                np.asarray(probe_gradient).shape != point.shape
                or not np.isfinite(probe_value)
                or not np.isfinite(probe_gradient).all()
            ):
                raise ValueError("direction audit probe objective/gradient is invalid")
            observed = float(probe_value) - value
            predicted = float(np.sum(gradient * displacement, dtype=np.float64))
            row.update(
                canonical_blocks=canonical.tolist(),
                actual_displacement=displacement.tolist(),
                status="collapsed" if not np.any(displacement) else "evaluated",
                value=float(probe_value),
                gradient=np.asarray(probe_gradient).tolist(),
                objective_spacing=float(np.abs(np.spacing(np.float32(probe_value)))),
                observed_change=observed,
                predicted_change=predicted,
                predicted_change_in_spacings=abs(predicted) / spacing,
                absolute_linearization_discrepancy=abs(observed - predicted),
                observed_predicted_ratio=observed / predicted if predicted else None,
            )
    return audit


class TracedFirstOrderSolver(FirstOrderSolver):
    """Trace existing calls and audit directions without changing solver work."""

    def __init__(self, model, policy):
        super().__init__(model, policy)
        self.last_trace = None
        self.last_directions = None

    def solve(self, *args, **kwargs):
        self.last_trace = None
        self.last_directions = None
        return super().solve(*args, **kwargs)

    def _optimize_plan(self, blocks, value, gradient, *args, **kwargs):
        kernels = self._kernels
        objective_arguments = None

        def capture(*arguments):
            nonlocal objective_arguments
            objective_arguments = arguments[1:]
            return kernels.objective_and_gradient(*arguments)

        self.last_directions = None
        with trace_minimize(blocks, value, gradient) as trace:
            self.last_trace = trace
            self._kernels = replace(kernels, objective_and_gradient=capture)
            try:
                outcome = super()._optimize_plan(
                    blocks, value, gradient, *args, **kwargs
                )
            finally:
                self._kernels = kernels
        self.last_trace["retained_work"] = copy.deepcopy(self.last_work)
        if objective_arguments is None:
            raise ValueError("traced solver did not independently audit its output")

        def original_objective(candidate):
            return kernels.objective_and_gradient(
                jnp.asarray(candidate), *objective_arguments
            )

        self.last_directions = direction_audit(
            original_objective,
            outcome.blocks,
            outcome.value,
            outcome.gradient,
            EXPONENTS,
        )
        return outcome
