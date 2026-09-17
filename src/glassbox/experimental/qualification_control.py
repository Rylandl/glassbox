"""No-fit controls for the frozen evaluation qualification.

The oracle substitutes public simulator equations for the learned mean behind
the unchanged generic plan map, objective and bounded shooting solver. Its
borrowed covariance is a numerical objective offset, not oracle uncertainty.
Only the predictor owns its causal simulator state; no running plant is read.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.plan import PlanValues, Prediction
from glassbox.control.solver import BoundedShootingSolver

from .harness import _GenericArm, _StructuredArm, control_reference
from .learned_plan import states_from_observed

HORIZON_STEPS = 5
BLOCK_COUNT = 5
MAXIMUM_ITERATIONS = 4
WARMUP_INTERVALS = 2


class CascadeEquations:
    """Public equations with equilibrium reset and issued-command replay."""

    def __init__(self, model):
        import cascade
        from cascade.canonical import rigid_body_from_canonical, rigid_body_to_canonical
        from cascade.initialization import (
            control_from_array,
            equilibrate_internal_state,
            zero_state,
        )
        from cascade.integration import repeat_control, rollout

        from .learned_plan import observed_from_state

        environment = cascade.standard_environment()
        digest = hashlib.sha256(str(jax.tree.structure(model)).encode())
        for leaf in jax.tree.leaves(model):
            array = np.asarray(leaf)
            digest.update(repr((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        self.compile_signature = digest.hexdigest()

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

        def canonical(state):
            return rigid_body_to_canonical(state.rigid_body)

        def predict(state, commands):
            def step(current, command):
                future = advance(current, command)
                return future, observed_from_state(canonical(future))

            return jax.lax.scan(step, state, commands)[1]

        self.reset = jax.jit(reset)
        self.advance = jax.jit(advance)
        self.canonical = jax.jit(canonical)
        self.predict = jax.jit(predict)


@dataclass(frozen=True)
class OraclePlanModel:
    """Delegate the whole generic seam except its forecast mean."""

    base: object
    equations: object
    values: PlanValues
    compile_signature: str

    def __getattr__(self, name):
        return getattr(self.base, name)

    def with_causal_state(self, state):
        return replace(self, values=self.values._replace(observed_history=state))

    def rollout(self, blocks, initial_state, initial_latent, exogenous, values):
        commands = self.base._commands_from_normalized(
            self.base._expand_normalized_blocks(blocks)
        )
        return self.rollout_commands(
            commands, initial_state, initial_latent, exogenous, values
        )

    def rollout_commands(
        self, commands, initial_state, initial_latent, exogenous, values
    ):
        del initial_latent
        if commands.shape != (self.horizon_steps, self.command_size):
            raise ValueError("commands must have one row per prediction interval")
        if values.observed_history is None:
            raise ValueError("oracle prediction requires its causal replay state")
        observed = self.equations.predict(values.observed_history, commands)
        return Prediction(
            mean_states=states_from_observed(
                initial_state, observed, self.sample_period_s
            ),
            tangent_covariance=values.forecast_error_covariance,
            commands=commands,
            latent_states=jnp.zeros((self.horizon_steps + 1, self.latent_size)),
            exogenous=exogenous,
        )


class OracleArm:
    """Control-trial arm retaining every returned command-plan forecast."""

    name = "oracle_generic_seam"

    def __init__(self, manifest, learned, cascade_model, *, _equations=None):
        base = _GenericArm(manifest, learned).controller.plan
        if (
            base.policy.horizon_steps != HORIZON_STEPS
            or base.policy.block_count != BLOCK_COUNT
            or base.policy.maximum_iterations != MAXIMUM_ITERATIONS
            or not np.isclose(base.sample_period_s, 0.05, rtol=0, atol=1e-12)
        ):
            raise ValueError("the saved generic plan differs from frozen qualification")
        self.equations = (
            CascadeEquations(cascade_model) if _equations is None else _equations
        )
        signature = hashlib.sha256(
            (
                base.compile_signature
                + ":qualification-oracle-public-equations-v1:"
                + self.equations.compile_signature
            ).encode()
        ).hexdigest()
        self.plan = OraclePlanModel(base, self.equations, base.values, signature)
        self.policy = base.policy
        self._state = None

    @property
    def prediction_steps(self):
        return self.plan.horizon_steps

    @property
    def ready(self):
        return self._observed >= WARMUP_INTERVALS + 1

    def reset(self, initial_state, initial_command):
        self._state = self.equations.reset(
            jnp.asarray(initial_state), jnp.asarray(initial_command)
        )
        self._observed = self._applied = 0
        self._causal = [np.asarray(self.equations.canonical(self._state))]
        self._errors = []
        self._indices = []
        self._commands = []
        self._forecasts = []
        self._objectives = []

    def observe(self, state):
        if self._observed != self._applied:
            raise ValueError("observe and command_applied must alternate")
        canonical = np.asarray(self.equations.canonical(self._state))
        np.testing.assert_allclose(canonical, state, rtol=1e-5, atol=1e-5)
        self._errors.append(float(np.max(np.abs(canonical - state))))
        self._observed += 1

    def command_applied(self, command):
        if self._applied != self._observed - 1:
            raise ValueError("a command follows its observed state")
        self._state = self.equations.advance(self._state, jnp.asarray(command))
        self._causal.append(np.asarray(self.equations.canonical(self._state)))
        self._applied += 1

    def solve(self, state, reference, previous_command, **keywords):
        if not self.ready or self._applied != self._observed - 1:
            raise ValueError("an oracle solve requires a ready, observed origin")
        plan = self.plan.with_causal_state(self._state)
        result = BoundedShootingSolver(plan, self.policy).solve(
            state, reference, previous_command, **keywords
        )
        if result.used_fallback:
            raise ValueError("oracle qualification cannot save an absent forecast")
        self._indices.append(self._applied)
        self._commands.append(np.asarray(result.predicted_commands))
        self._forecasts.append(np.asarray(result.predicted_states))
        self._objectives.append(result.diagnostics.final_objective)
        return result

    def summary(self):
        return dict(
            arm=self.name,
            horizon_steps=self.prediction_steps,
            horizon_s=self.prediction_steps * self.plan.sample_period_s,
            block_count=self.policy.block_count,
            maximum_iterations=self.policy.maximum_iterations,
            warmup_intervals=WARMUP_INTERVALS,
            compile_signature=self.plan.compile_signature,
            covariance_meaning="saved generic numerical objective offset; not oracle uncertainty",
            meaning="public equations and causal command replay behind the unchanged generic seam",
        )

    def diagnostic_arrays(self):
        """NPZ-ready replay material; call after a trial, before the next reset."""
        return dict(
            causal_states=np.asarray(self._causal),
            observed_max_abs_error=np.asarray(self._errors),
            solve_indices=np.asarray(self._indices, dtype=np.int64),
            candidate_commands=np.asarray(self._commands).reshape(
                -1, HORIZON_STEPS, self.plan.command_size
            ),
            forecast_states=np.asarray(self._forecasts).reshape(
                -1, HORIZON_STEPS + 1, 13
            ),
            final_objectives=np.asarray(self._objectives),
        )


class MatchedStructuredArm(_StructuredArm):
    """Saved structured mean with matched horizon and initial command holds."""

    name = "structured_matched_horizon"

    def __init__(self, manifest, belief):
        from glassbox.control.fitted import NMPCController, default_solver_policy

        self.belief = belief
        self.controller = NMPCController(
            belief.model,
            policy=replace(
                default_solver_policy(belief),
                horizon_steps=HORIZON_STEPS,
                block_count=BLOCK_COUNT,
                maximum_iterations=MAXIMUM_ITERATIONS,
                allow_unresolved_parameters=manifest["controller"][
                    "allow_unresolved_parameters"
                ],
            ),
        )
        self.policy = self.controller.plan.policy
        self._history = None

    def reset(self, initial_state, initial_command):
        super().reset(initial_state, initial_command)
        self._observed = 0

    def observe(self, state):
        self._observed += 1

    @property
    def ready(self):
        return self._observed >= WARMUP_INTERVALS + 1

    def summary(self):
        return dict(
            super().summary(),
            warmup_intervals=WARMUP_INTERVALS,
            meaning="saved structured mean at 250 ms; its own propagation and support penalty remain",
        )


def build_arms(manifest, learned, belief, cascade_model):
    """Return both arms for the pinned control-v5 manifest, without fitting."""
    return {
        OracleArm.name: OracleArm(manifest, learned, cascade_model),
        MatchedStructuredArm.name: MatchedStructuredArm(manifest, belief),
    }


def verify_oracle_diagnostics(
    manifest,
    learned,
    cascade_model,
    tracking,
    diagnostics,
    replay_tolerance,
    *,
    _equations=None,
):
    """Rebuild causal states, returned-plan forecasts and objective scores.

    ``tracking`` and ``diagnostics`` are mappings loaded from the saved NPZs.
    ``replay_tolerance`` is the frozen plan's control_qualification.replay block.
    Input/source integrity and independent physical-plant replay belong to the
    qualification runner. This check does not rerun command optimization.
    """
    arm = OracleArm(manifest, learned, cascade_model, _equations=_equations)
    equations = arm.equations
    states = np.asarray(tracking["states"])
    commands = np.asarray(tracking["commands"])
    if len(states) != len(commands) + 1:
        raise ValueError("saved command and state counts differ")
    expected_indices = np.arange(WARMUP_INTERVALS, len(commands))
    if not np.array_equal(diagnostics["solve_indices"], expected_indices):
        raise ValueError("oracle solve origins differ from the frozen warmup")
    expected_shapes = {
        "causal_states": states.shape,
        "observed_max_abs_error": (len(commands),),
        "candidate_commands": (
            len(expected_indices),
            HORIZON_STEPS,
            arm.plan.command_size,
        ),
        "forecast_states": (len(expected_indices), HORIZON_STEPS + 1, 13),
        "final_objectives": (len(expected_indices),),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(diagnostics[name])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"oracle diagnostic shape or finiteness differs: {name}")
    state_tolerance = dict(
        rtol=replay_tolerance["state_rtol"], atol=replay_tolerance["state_atol"]
    )
    score_tolerance = dict(
        rtol=replay_tolerance["score_rtol"], atol=replay_tolerance["score_atol"]
    )
    initial_command = np.asarray(tracking["initial_command"])
    np.testing.assert_allclose(
        commands[:WARMUP_INTERVALS],
        np.broadcast_to(initial_command, commands[:WARMUP_INTERVALS].shape),
        rtol=0,
        atol=0,
    )
    current = equations.reset(jnp.asarray(states[0]), jnp.asarray(initial_command))
    causal, errors, maximum_forecast_difference = [], [], 0.0
    anchor = np.asarray(tracking["reference_anchor_state"])
    for index in range(len(states)):
        canonical = np.asarray(equations.canonical(current))
        causal.append(canonical)
        np.testing.assert_allclose(canonical, states[index], **state_tolerance)
        if index == len(commands):
            break
        errors.append(float(np.max(np.abs(canonical - states[index]))))
        if index >= WARMUP_INTERVALS:
            row = index - WARMUP_INTERVALS
            candidate = jnp.asarray(diagnostics["candidate_commands"][row])
            np.testing.assert_allclose(candidate[0], commands[index], rtol=0, atol=0)
            if np.any(candidate < arm.plan.command_minimum) or np.any(
                candidate > arm.plan.command_maximum
            ):
                raise ValueError(
                    "saved oracle candidate commands exceed declared bounds"
                )
            plan = arm.plan.with_causal_state(current)
            prediction = plan.rollout_commands(
                candidate,
                jnp.asarray(states[index]),
                jnp.zeros(0),
                jnp.zeros((HORIZON_STEPS, 0)),
                plan.values,
            )
            forecast = np.asarray(prediction.mean_states)
            saved_forecast = diagnostics["forecast_states"][row]
            np.testing.assert_allclose(forecast, saved_forecast, **state_tolerance)
            maximum_forecast_difference = max(
                maximum_forecast_difference,
                float(np.max(np.abs(forecast - saved_forecast))),
            )
            future = (index + np.arange(HORIZON_STEPS + 1)) * manifest["trial"][
                "sample_interval_s"
            ]
            reference = control_reference(
                anchor, future, manifest["tracking_reference"]
            )
            objective = float(
                plan.stage_cost(
                    prediction,
                    jnp.asarray(reference),
                    jnp.asarray(commands[index - 1]),
                    plan.policy,
                )
            )
            np.testing.assert_allclose(
                objective, diagnostics["final_objectives"][row], **score_tolerance
            )
        current = equations.advance(current, jnp.asarray(commands[index]))
    np.testing.assert_allclose(causal, diagnostics["causal_states"], **state_tolerance)
    np.testing.assert_allclose(
        errors, diagnostics["observed_max_abs_error"], **score_tolerance
    )
    return dict(
        checked_forecasts=len(expected_indices),
        checked_causal_states=len(states),
        maximum_causal_state_difference=float(
            np.max(np.abs(np.asarray(causal) - states))
        ),
        maximum_forecast_difference=maximum_forecast_difference,
    )
