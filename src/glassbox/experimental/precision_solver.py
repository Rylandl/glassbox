"""Isolated float64 planning experiment with an unchanged float32 seed choice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.plan import ReferenceTrajectory
from glassbox.control.solver import SolveStatus, _SolveAbort

from .first_order_qualification import FirstOrderSolver


def _snapshot(tree):
    """Copy the numbers the default-precision JAX call actually consumes."""
    return jax.tree.map(lambda leaf: np.asarray(jnp.asarray(leaf)).copy(), tree)


def _lift(tree):
    def convert(leaf):
        array = np.asarray(leaf)
        return jnp.asarray(
            array, dtype=jnp.float64 if array.dtype.kind == "f" else None
        )

    return jax.tree.map(convert, tree)


def _fingerprint(tree):
    return {
        "tree": str(jax.tree.structure(tree)),
        "leaves": [
            {
                "shape": list(np.asarray(leaf).shape),
                "dtype": np.asarray(leaf).dtype.str,
                "sha256": hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest(),
            }
            for leaf in jax.tree.leaves(tree)
        ],
    }


def _require_float64(tree, name):
    count = 0
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        if array.dtype.kind == "f":
            if array.dtype != np.dtype("float64"):
                raise ValueError(f"{name} contains {array.dtype}, expected float64")
            count += 1
        elif array.dtype.kind == "c":
            raise ValueError(f"{name} contains unsupported complex values")
    return count


def _equal_trees(actual, expected, name):
    if jax.tree.structure(actual) != jax.tree.structure(expected):
        raise ValueError(f"{name} tree changed after capture")
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise ValueError(f"{name} values changed after capture")


@dataclass(frozen=True)
class CapturedSeed:
    blocks: object
    warm_start_used: bool
    state: object
    latent: object
    reference_states: object
    previous_command: object
    exogenous: object
    values: object
    command_minimum: object
    command_maximum: object
    local_state_scale: object
    native_seed_value: float
    native_seed_gradient: object

    def runtime(self):
        return (
            self.state,
            self.latent,
            self.reference_states,
            self.previous_command,
            self.exogenous,
            self.values,
        )

    def record(self):
        return {
            "blocks": np.asarray(self.blocks).tolist(),
            "warm_start_used": self.warm_start_used,
            "native_seed_value": self.native_seed_value,
            "native_seed_gradient": np.asarray(self.native_seed_gradient).tolist(),
            "inputs": {
                name: _fingerprint(getattr(self, name))
                for name in (
                    "state",
                    "latent",
                    "reference_states",
                    "previous_command",
                    "exogenous",
                    "values",
                    "command_minimum",
                    "command_maximum",
                    "local_state_scale",
                )
            },
        }


class SeedCaptureSolver(FirstOrderSolver):
    """Observe the original seed path without changing calls or arithmetic."""

    def solve(self, *args, **kwargs):
        if jax.config.x64_enabled:
            raise ValueError("baseline seed capture requires default float32")
        self.last_capture = None
        return super().solve(*args, **kwargs)

    def _seed_plan(
        self,
        cold_blocks,
        warm_start,
        state,
        latent,
        reference,
        previous_command,
        exogenous,
        *,
        budget=None,
    ):
        result = super()._seed_plan(
            cold_blocks,
            warm_start,
            state,
            latent,
            reference,
            previous_command,
            exogenous,
            budget=budget,
        )
        blocks, value, gradient, _, used_warm = result
        self.last_capture = CapturedSeed(
            blocks=_snapshot(blocks),
            warm_start_used=bool(used_warm),
            state=_snapshot(state),
            latent=_snapshot(latent),
            reference_states=_snapshot(reference.states),
            previous_command=_snapshot(previous_command),
            exogenous=_snapshot(exogenous),
            values=_snapshot(self.model.values),
            command_minimum=_snapshot(self.model.command_minimum),
            command_maximum=_snapshot(self.model.command_maximum),
            local_state_scale=_snapshot(self.model.tolerances.local_state_scale),
            native_seed_value=float(np.asarray(value)),
            native_seed_gradient=_snapshot(gradient),
        )
        return result


class PrecisionSolver(FirstOrderSolver):
    """One fixed, already selected seed; all inherited optimization is unchanged."""

    def __init__(self, model, policy, capture):
        super().__init__(model, policy)
        self.capture = capture

    def _seed_plan(
        self,
        cold_blocks,
        warm_start,
        state,
        latent,
        reference,
        previous_command,
        exogenous,
        *,
        budget=None,
    ):
        del cold_blocks, budget
        if warm_start is not None:
            raise ValueError("precision solve must use only its captured fixed seed")
        actual = (
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self.model.values,
        )
        expected = _lift(self.capture.runtime())
        _require_float64(actual, "candidate runtime")
        _equal_trees(actual, expected, "candidate runtime")
        blocks = _lift(self.capture.blocks)
        value, gradient = self._kernels.objective_and_gradient(blocks, *actual)
        self._seed_evaluations = 1
        self.last_work["seed_objective_evaluations"] = 1
        _require_float64((blocks, value, gradient), "candidate seed evaluation")
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE, "fixed float64 seed is non-finite"
            )
        return blocks, value, gradient, float(value), self.capture.warm_start_used


class _Cascade64Equations:
    """Same public forecast equations, with lifted model/environment snapshots."""

    def __init__(self, model, environment):
        from cascade.canonical import rigid_body_to_canonical
        from cascade.initialization import control_from_array
        from cascade.integration import repeat_control, rk4_step, rollout

        from .learned_plan import observed_from_state

        self.model = model
        self.environment = environment

        def advance_core(state, control, environment):
            return rollout(
                model,
                state,
                repeat_control(control, 20),
                environment,
                1 / 400,
                step=rk4_step,
            )[0]

        advance_kernel = jax.jit(advance_core)

        def predict(state, commands):
            def step(current, command):
                control = control_from_array(model, jnp.asarray(command))
                future = advance_kernel(current, control, environment)
                observed = observed_from_state(
                    rigid_body_to_canonical(future.rigid_body)
                )
                return future, observed

            return jax.lax.scan(step, state, commands)[1]

        self.predict = jax.jit(predict)


def audit_jaxpr_dtypes(closed_jaxpr):
    """Reject lower precision arithmetic, recursively including scans and JITs.

    A static float32 integer/sign constant may travel through call boundaries
    solely to an immediate float64 conversion. Runtime inputs must already be64.
    No primitive may compute or return a float32 arithmetic intermediate.
    """
    counts = {
        "jaxprs": 0,
        "equations": 0,
        "static_float32_to_float64_conversions": 0,
        "float32_arithmetic": 0,
    }

    def dtype(variable):
        value = getattr(getattr(variable, "aval", None), "dtype", None)
        return None if value is None else np.dtype(value)

    def lower(variable):
        value = dtype(variable)
        return value is not None and value.kind == "f" and value != np.dtype("float64")

    def static_constant(value):
        array = np.asarray(value)
        if array.dtype.kind == "f" and array.dtype != np.dtype("float64"):
            if (
                array.dtype != np.dtype("float32")
                or not np.all(np.isfinite(array))
                or not np.all(array == np.round(array))
            ):
                raise ValueError(
                    "JAXPR has a non-exact lower precision static constant"
                )

    def nested(value):
        if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
            return [value]
        if hasattr(value, "eqns") and hasattr(value, "invars"):
            return [value]
        if isinstance(value, dict):
            return [item for child in value.values() for item in nested(child)]
        if isinstance(value, (tuple, list)):
            return [item for child in value for item in nested(child)]
        return []

    def visit(closed, *, top=False):
        graph = getattr(closed, "jaxpr", closed)
        counts["jaxprs"] += 1
        if top and any(lower(var) for var in graph.invars):
            raise ValueError("JAXPR has a lower precision floating runtime input")
        for value in getattr(closed, "consts", ()):
            static_constant(value)
        for equation in graph.eqns:
            counts["equations"] += 1
            children = nested(equation.params)
            for var in equation.invars:
                if hasattr(var, "val"):
                    static_constant(var.val)
            low_inputs = any(lower(var) for var in equation.invars)
            low_outputs = any(lower(var) for var in equation.outvars)
            if low_inputs or low_outputs:
                if (
                    equation.primitive.name == "convert_element_type"
                    and low_inputs
                    and not low_outputs
                    and all(
                        dtype(var) == np.dtype("float64") for var in equation.outvars
                    )
                ):
                    counts["static_float32_to_float64_conversions"] += 1
                elif children and not low_outputs:
                    pass  # A nested call forwards a static constant to its cast.
                else:
                    raise ValueError(
                        f"JAXPR contains lower precision arithmetic: {equation.primitive.name}"
                    )
            for child in children:
                visit(child)
        if any(lower(var) for var in graph.outvars):
            raise ValueError("JAXPR returns a lower precision floating value")

    visit(closed_jaxpr, top=True)
    return counts


def _common_audit(solver, capture, blocks):
    blocks = jnp.asarray(blocks, dtype=jnp.float64)
    if (
        blocks.shape != np.asarray(capture.blocks).shape
        or not np.all(np.isfinite(blocks))
        or np.any(np.abs(blocks) > 1)
    ):
        raise ValueError("common audit requires finite bounded canonical blocks")
    state, latent, reference, previous, exogenous, values = _lift(capture.runtime())
    value, gradient = solver._kernels.objective_and_gradient(
        blocks,
        state,
        latent,
        reference,
        previous,
        exogenous,
        values,
    )
    prediction = solver._kernels.rollout(blocks, state, latent, exogenous, values)
    _require_float64((blocks, value, gradient, prediction), "common audit output")
    arrays = (value, gradient, prediction.mean_states, prediction.commands)
    if not all(np.all(np.isfinite(item)) for item in arrays):
        raise ValueError("common audit has a non-finite output")
    command = np.asarray(prediction.commands)
    minimum, maximum = (
        np.asarray(solver.model.command_minimum),
        np.asarray(solver.model.command_maximum),
    )
    residual = float(
        np.max(
            np.abs(
                np.asarray(blocks)
                - np.clip(np.asarray(blocks) - np.asarray(gradient), -1, 1)
            )
        )
    )
    return {
        "blocks": np.asarray(blocks).tolist(),
        "value": float(value),
        "gradient": np.asarray(gradient).tolist(),
        "residual": residual,
        "commands": command.tolist(),
        "states": np.asarray(prediction.mean_states).tolist(),
        "bound_violation": float(
            max(0, np.max(minimum - command), np.max(command - maximum))
        ),
    }


class PrecisionFactory:
    """Build a separate64 planning cache without touching causal replay or defaults."""

    def __init__(
        self, cascade_model_float32, *, _equations_factory=None, _environment=None
    ):
        if jax.config.x64_enabled:
            raise ValueError(
                "precision factory must snapshot the original float32 context"
            )
        if _environment is None:
            import cascade

            _environment = cascade.standard_environment()
        self.model_snapshot = _snapshot(cascade_model_float32)
        self.environment_snapshot = _snapshot(_environment)
        self._equations_factory = (
            _Cascade64Equations if _equations_factory is None else _equations_factory
        )
        self._dtype_cache = {}
        with jax.enable_x64(True):
            self.model64 = _lift(self.model_snapshot)
            self.environment64 = _lift(self.environment_snapshot)
            self.equations64 = self._equations_factory(self.model64, self.environment64)
        self._identity = hashlib.sha256(
            repr(
                (
                    _fingerprint(self.model_snapshot),
                    _fingerprint(self.environment_snapshot),
                )
            ).encode()
        ).hexdigest()

    def _model(self, model32, capture):
        values = _lift(capture.values)
        scale = np.asarray(capture.local_state_scale, dtype=np.float64)
        tolerances = replace(
            model32.tolerances,
            position_m=tuple(scale[:3]),
            velocity_m_s=tuple(scale[3:6]),
            attitude_rad=tuple(scale[6:9]),
            angular_velocity_rad_s=tuple(scale[9:12]),
        )
        signature = hashlib.sha256(
            (
                model32.compile_signature
                + ":solver-precision-v1-float64:"
                + self._identity
            ).encode()
        ).hexdigest()
        base64 = replace(
            model32.base,
            values=values,
            tolerances=tolerances,
            command_minimum_array=np.asarray(capture.command_minimum, dtype=np.float64),
            command_maximum_array=np.asarray(capture.command_maximum, dtype=np.float64),
            compile_signature=signature,
        )
        return replace(
            model32,
            base=base64,
            equations=self.equations64,
            values=values,
            compile_signature=signature,
        )

    def solve(self, model32, policy, capture, baseline_blocks):
        if jax.config.x64_enabled:
            raise ValueError("precision solve must start in original float32 context")
        if not isinstance(capture, CapturedSeed):
            raise ValueError(
                "precision solve requires a successful baseline seed capture"
            )
        with jax.enable_x64(True):
            model64 = self._model(model32, capture)
            solver = PrecisionSolver(model64, policy, capture)
            runtime = _lift(capture.runtime())
            seed = _lift(capture.blocks)
            evidence = {
                "floating_dtype": "float64",
                "compile_signature": model64.compile_signature,
                "baseline_compile_signature": model32.compile_signature,
                "runtime_float_leaf_count": _require_float64(
                    (seed, runtime), "lifted runtime"
                ),
                "model_float_leaf_count": _require_float64(
                    self.model64, "Cascade model"
                ),
                "environment_float_leaf_count": _require_float64(
                    self.environment64, "Cascade environment"
                ),
                "plan_float_leaf_count": _require_float64(
                    (
                        model64.values,
                        model64.command_minimum,
                        model64.command_maximum,
                        model64.tolerances.local_state_scale,
                    ),
                    "planning values",
                ),
            }
            key = (
                model64.compile_signature,
                policy,
                str(jax.tree.structure(runtime)),
                tuple(
                    (np.asarray(item).shape, np.asarray(item).dtype.str)
                    for item in jax.tree.leaves(runtime)
                ),
            )
            if key not in self._dtype_cache:
                graph = jax.make_jaxpr(solver._kernels.objective_and_gradient)(
                    seed, *runtime
                )
                self._dtype_cache[key] = audit_jaxpr_dtypes(graph)
            evidence["jaxpr"] = dict(self._dtype_cache[key])
            state, latent, reference, previous, exogenous, _ = runtime
            result = solver.solve(
                state,
                ReferenceTrajectory(reference, exogenous),
                previous,
                latent_state=latent,
            )
            evidence["output_float_leaf_count"] = _require_float64(
                (result.command, result.predicted_commands, result.predicted_states),
                "candidate result",
            )
            baseline = _common_audit(solver, capture, baseline_blocks)
            candidate_blocks = solver.last_work["returned_canonical_blocks"]
            candidate = (
                None
                if result.used_fallback or candidate_blocks is None
                else _common_audit(solver, capture, candidate_blocks)
            )
            return (
                result,
                dict(solver.last_work),
                {"baseline": baseline, "candidate": candidate},
                evidence,
            )
