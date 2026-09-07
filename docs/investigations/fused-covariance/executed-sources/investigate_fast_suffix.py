"""One bounded GN-SQP suffix experiment, through the normal solve boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from investigate_sqp_recovery import GaussNewtonReference
from investigate_terminal_suffix import SUFFIX_STEPS, fingerprints

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import BeliefPlanModel, NMPCController
from glassbox.core.synthetic import resting_state


class SuffixValues(NamedTuple):
    parameters: object
    covariance_factor: object
    forecast_error_covariance: object
    frozen_commands: object


class SuffixPlan(BeliefPlanModel):
    """Six independent commands; the physical prefix is a dynamic kernel input."""

    def rollout(self, blocks, initial_state, initial_latent, exogenous, values):
        commands = jnp.concatenate(
            (values.frozen_commands, self._commands_from_normalized(blocks))
        )
        return self.rollout_commands(
            commands, initial_state, initial_latent, exogenous, values
        )


class FastSuffixSolver(GaussNewtonReference):
    """Exactly one supplied waveform seed, with no cold/warm competition.

    The supplied seed already represents this request (it is shifted by the
    caller). It is not silently shifted again, averaged, or replaced by a hold.
    Cold requests are outside this restricted formulation's scope.
    """

    def __init__(self, plan, commands, *, updates=2, work_estimates=None):
        if plan.horizon_steps != 30 or updates not in (1, 2):
            raise ValueError("predeclared 30-stage, one/two-update experiment")
        self.seed_commands = self.validate_commands(plan, commands)
        policy = replace(plan.policy, block_count=SUFFIX_STEPS)
        suffix_plan = SuffixPlan(
            belief=plan.belief,
            tolerances=plan.tolerances,
            safety_envelope=plan.safety_envelope,
            policy=policy,
            values=SuffixValues(*plan.values, self.seed_commands[:-SUFFIX_STEPS]),
            compile_signature=plan.compile_signature + ":six-command-suffix-v1",
        )
        super().__init__(
            suffix_plan, policy, warm_iterations=updates, work_estimates=work_estimates
        )
        self.counts = {}
        for name in ("evaluate", "linearize", "finalize"):
            original = getattr(self, name)

            def counted(*args, _name=name, _original=original):
                self.counts[_name] = self.counts.get(_name, 0) + 1
                return _original(*args)

            setattr(self, name, counted)

    @staticmethod
    def validate_commands(plan, commands):
        commands = np.asarray(commands)
        if (
            commands.shape != (30, plan.command_size)
            or not np.all(np.isfinite(commands))
            or np.any(commands < np.asarray(plan.command_minimum))
            or np.any(commands > np.asarray(plan.command_maximum))
        ):
            raise ValueError(
                "suffix seed must be an exact finite bounded physical waveform"
            )
        return jnp.asarray(commands)

    def set_seed(self, commands):
        self.seed_commands = self.validate_commands(self.model, commands)
        self.model = replace(
            self.model,
            values=self.model.values._replace(
                frozen_commands=self.seed_commands[:-SUFFIX_STEPS]
            ),
        )
        self.counts.clear()

    def _warm_blocks(self, warm_start):
        return None

    def _cold_blocks(self, previous_command):
        return self._normalized_from_commands(self.seed_commands[-SUFFIX_STEPS:])

    def _seed_plan(self, *args, **kwargs):
        result = super()._seed_plan(*args, **kwargs)
        self._iteration_budget = self.warm_iterations
        return result


def summarize(solver, result, plan, reference, state, latent, previous):
    row = {
        "updates_budget": solver.warm_iterations,
        "seed_waveform_sha256": hashlib.sha256(
            np.asarray(solver.seed_commands).tobytes()
        ).hexdigest(),
        "selected_seed": "only supplied shifted waveform",
        "api_warm_start_used": result.diagnostics.warm_start_used,
        "counts": dict(solver.counts),
        "status": str(result.status),
        "command_usable": result.command_usable,
        "deadline_met": result.deadline_met,
        "nonlinear_feasibility": {
            name: getattr(result.nonlinear_feasibility, name)
            for name in result.nonlinear_feasibility.__dataclass_fields__
        },
        "optimizer": {
            key: value
            for key, value in solver.reports[-1].items()
            if not key.endswith("_s")
        }
        if solver.reports
        else None,
    }
    if result.command_usable:
        commands = result.predicted_commands
        np.testing.assert_array_equal(
            commands[:-SUFFIX_STEPS], solver.seed_commands[:-SUFFIX_STEPS]
        )
        prediction = plan.rollout_commands(
            commands, state, latent, solver._exogenous_forecast(reference), plan.values
        )
        terms = plan.optimization_terms(
            prediction, reference.states, previous, plan.policy
        )
        margins = np.asarray(terms.inequality_margins)
        for array in (
            prediction.mean_states,
            prediction.latent_states,
            prediction.tangent_covariance,
        ):
            assert np.all(np.isfinite(array))
        assert margins.min() >= -1e-6
        np.testing.assert_array_equal(prediction.commands, commands)
        np.testing.assert_allclose(
            prediction.mean_states, result.predicted_states, rtol=2e-6, atol=2e-6
        )
        row.update(
            independent_maximum_utilization=float(1 - margins.min()),
            independent_cost=float(jnp.sum(terms.residuals**2)),
            exact_frozen_prefix=True,
            commands=np.asarray(commands).tolist(),
        )
    return row


def run(fixtures, output):
    belief = DynamicsBelief.load(fixtures / "rich-belief.json")
    checkpoint = np.load(fixtures / "horizon-shift-states.npz")
    controller = NMPCController(belief)
    plan = controller.plan
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    np.testing.assert_array_equal(reference.states, checkpoint["reference_states"])
    state, latent, previous, commands = [
        jnp.asarray(checkpoint["tick4_" + name])
        for name in ("state", "latent", "previous_command", "shifted_commands")
    ]
    report = {
        "diagnostic_only": True,
        "glassbox_import": glassbox.__file__,
        "parameterization": "24 frozen commands; six independent commands, 24 variables",
        "algorithm": "existing GN SQP with feasible retention; unchanged residuals and constraints",
        "seed_semantics": "warm waveform supplied already shifted, no optimizer state reused, one seed; cold startup not studied",
        "timing_qualification": False,
        "deadlines_disabled": True,
        "known_request": [],
        "continuation": [],
    }
    for updates in (1, 2):
        solver = FastSuffixSolver(plan, commands, updates=updates)
        result = solver.solve(state, reference, previous, latent_state=latent)
        report["known_request"].append(
            summarize(solver, result, plan, reference, state, latent, previous)
        )
    if any(r["command_usable"] for r in report["known_request"]):
        from investigate_recovery import initial_state

        from glassbox.core.dynamics import hover_control, step_with_latent
        from glassbox.core.synthetic import true_parameters
        from glassbox.workflows.benchmarks import recovery

        # Prescribed two-update arm is continued once, without a solver sweep.
        target = recovery._arm_configuration_parameters(
            true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
        )
        first_suffix = (
            np.asarray(result.predicted_commands[24]).copy()
            if result.command_usable
            else None
        )
        for tick in range(36):
            if not result.command_usable:
                report["continuation_stop"] = "no_feasible_candidate"
                break
            previous = result.command
            if tick == 24:
                np.testing.assert_array_equal(previous, first_suffix)
            state, latent = step_with_latent(
                target,
                state,
                latent,
                previous,
                plan.sample_period_s,
                belief.input_spec.control_roles,
            )
            actual = float(
                np.max(
                    plan._validity_utilization(
                        state, solver._exogenous_forecast(reference)[0]
                    )
                )
            )
            row = {"tick": tick, "actual_support_utilization": actual}
            report["continuation"].append(row)
            if (
                not np.all(np.isfinite(state))
                or not np.all(np.isfinite(latent))
                or not np.isfinite(actual)
                or actual > 1 + 1e-6
            ):
                report["continuation_stop"] = "actual_state_failed"
                break
            if tick < 35:
                solver.set_seed(
                    jnp.concatenate(
                        (result.predicted_commands[1:], result.predicted_commands[-1:])
                    )
                )
                result = solver.solve(state, reference, previous, latent_state=latent)
                row["next_solve"] = summarize(
                    solver, result, plan, reference, state, latent, previous
                )
        report.setdefault("continuation_stop", "bounded_36_intervals_complete")
        report["controls"] = []
        previous = jnp.asarray(hover_control(plan.values.parameters))
        for kind in ("small", "outside"):
            state = jnp.asarray(initial_state(belief, "small"))
            if kind == "outside":
                envelope = belief.model.runtime_spec.validity_envelope
                state = state.at[10].set(
                    envelope.angular_velocity_center_rad_s[0]
                    + 1.1 * envelope.angular_velocity_half_width_rad_s[0]
                )
            latent = previous
            solver.set_seed(jnp.repeat(previous[None, :], 30, axis=0))
            result = solver.solve(state, reference, previous, latent_state=latent)
            row = summarize(solver, result, plan, reference, state, latent, previous)
            row["control"] = kind
            report["controls"].append(row)
    old = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/investigations/terminal-suffix.json"
        ).read_text()
    )
    report["slsqp_reference"] = old["repair"]
    report["slsqp_reference"]["provenance"] = (
        "prior recorded artifact; no repeated SLSQP solve"
    )
    report["stop_reason"] = (
        "known_request_not_repaired_at_either_budget"
        if not any(r["command_usable"] for r in report["known_request"])
        else report.get("continuation_stop", "two_update_arm_not_usable")
    )
    report.update(fingerprints(fixtures))
    report["source_sha256"]["scripts/investigate_fast_suffix.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["known_request"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.output)
