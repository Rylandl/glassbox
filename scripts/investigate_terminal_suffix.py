"""One bounded offline suffix-feasibility experiment; no production changes.

The first 24 commands are fixed; six independent commands optimize the existing
30-stage objective and robust constraints. This is a restricted NMPC subproblem,
not a terminal invariant set or a recursive-feasibility certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from investigate_sqp_recovery import FEASIBILITY_TOLERANCE
from scipy.optimize import minimize

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.core.dynamics import latent_response_time_constants, step_with_latent
from glassbox.core.synthetic import resting_state

SUFFIX_STEPS = 6


def replace_suffix(commands, normalized, lower, upper):
    """Keep every retained command exact, without projecting onto block starts."""
    suffix = lower + (normalized + 1) * (upper - lower) / 2
    return commands.at[-len(normalized) :].set(suffix)


def finite_feasible(commands, margins, lower, upper):
    return bool(
        np.all(np.isfinite(commands))
        and np.all(np.isfinite(margins))
        and np.all(commands >= np.asarray(lower))
        and np.all(commands <= np.asarray(upper))
        and np.min(margins) >= -FEASIBILITY_TOLERANCE
    )


class SuffixProblem:
    def __init__(self, plan, reference):
        self.plan = plan
        self.reference = reference
        if plan.horizon_steps != 30:
            raise ValueError(
                "this predeclared experiment requires the 30-stage horizon"
            )
        if not plan.uncertainty_complete:
            raise ValueError("complete uncertainty is required")
        self.lower, self.upper = plan.command_minimum, plan.command_maximum
        self.exogenous = jnp.zeros((plan.horizon_steps, plan.exogenous_size))

        def terms(vector, commands, state, latent, previous):
            waveform = replace_suffix(
                commands, vector.reshape(SUFFIX_STEPS, -1), self.lower, self.upper
            )
            prediction = plan.rollout_commands(
                waveform, state, latent, self.exogenous, plan.values
            )
            terms = plan.optimization_terms(
                prediction, reference.states, previous, plan.policy
            )
            return jnp.sum(terms.residuals**2), terms.inequality_margins

        self.terms = jax.jit(terms)
        self.derivative = jax.jit(jax.jacfwd(terms))

    def solve(self, commands, state, latent, previous):
        commands = jnp.asarray(commands)
        context = commands, state, latent, previous
        start = np.asarray(
            2 * (commands[-SUFFIX_STEPS:] - self.lower) / (self.upper - self.lower) - 1
        ).ravel()
        cache = {}
        best = None
        counts = {"evaluations": 0, "derivatives": 0}

        def values(vector):
            nonlocal best
            key = np.asarray(vector).tobytes()
            if key not in cache:
                cost, margins = self.terms(jnp.asarray(vector), *context)
                cost, margins = float(cost), np.asarray(margins)
                cache[key] = cost, margins
                counts["evaluations"] += 1
                waveform = np.asarray(
                    replace_suffix(
                        commands,
                        jnp.asarray(vector.reshape(SUFFIX_STEPS, -1)),
                        self.lower,
                        self.upper,
                    )
                )
                if np.isfinite(cost) and finite_feasible(
                    waveform, margins, self.lower, self.upper
                ):
                    if best is None or cost < best[0]:
                        best = cost, waveform
            return cache[key]

        def derivatives(vector):
            counts["derivatives"] += 1
            return tuple(
                np.asarray(part, dtype=float)
                for part in self.derivative(jnp.asarray(vector), *context)
            )

        initial_cost, initial_margins = values(start)
        result = minimize(
            lambda vector: values(vector)[0],
            start,
            jac=lambda vector: derivatives(vector)[0],
            bounds=[(-1.0, 1.0)] * len(start),
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda vector: values(vector)[1],
                "jac": lambda vector: derivatives(vector)[1],
            },
            options={"maxiter": 100, "ftol": 1e-9},
        )
        final_cost, final_margins = values(result.x)
        report = {
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "iterations": int(result.nit),
            **counts,
            "initial_cost": initial_cost,
            "final_iterate_cost": final_cost,
            "initial_maximum_utilization": float(1 - initial_margins.min()),
            "initial_prefix_utilization": float(
                1 - initial_margins.reshape(31, -1)[1:-1].min()
            ),
            "initial_tail_utilization": float(
                1 - initial_margins.reshape(31, -1)[-1].min()
            ),
            "final_iterate_maximum_utilization": float(1 - final_margins.min()),
            "feasible_found": best is not None,
        }
        if best is None:
            return None, report
        accepted = jnp.asarray(best[1])
        # Independently evaluate the returned physical waveform, not the SLSQP cache.
        prediction = self.plan.rollout_commands(
            accepted, state, latent, self.exogenous, self.plan.values
        )
        margins = np.asarray(
            self.plan.optimization_terms(
                prediction, self.reference.states, previous, self.plan.policy
            ).inequality_margins
        )
        if not all(
            np.all(np.isfinite(np.asarray(part)))
            for part in (
                prediction.mean_states,
                prediction.latent_states,
                prediction.tangent_covariance,
            )
        ):
            raise AssertionError("nonfinite independent prediction")
        if not finite_feasible(accepted, margins, self.lower, self.upper):
            raise AssertionError("independent nonlinear acceptance failed")
        np.testing.assert_array_equal(
            accepted[:-SUFFIX_STEPS], commands[:-SUFFIX_STEPS]
        )
        report.update(
            accepted_cost=best[0],
            maximum_utilization=float(1 - margins.min()),
            suffix_maximum_command_change=float(
                np.max(np.abs(np.asarray(accepted - commands)))
            ),
            terminal_latent=np.asarray(prediction.latent_states[-1]).tolist(),
            terminal_command=np.asarray(accepted[-1]).tolist(),
            suffix_commands=np.asarray(accepted[-SUFFIX_STEPS:]).tolist(),
            tail_utilization=float(1 - margins.reshape(31, -1)[-1].min()),
        )
        return accepted, report


def fingerprints(fixtures):
    from glassbox.workflows.benchmarks import recovery

    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("investigate_sqp_recovery.py")]
    paths.extend(
        Path(glassbox.__file__).parent / name
        for name in recovery.BENCHMARK_SOURCE_FILES
    )
    return {
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "fixture_sha256": {
            name: hashlib.sha256((fixtures / name).read_bytes()).hexdigest()
            for name in (
                "rich-belief.json",
                "horizon-shift-states.npz",
                "manifest.json",
            )
        },
        "fixture_manifest": json.loads((fixtures / "manifest.json").read_text()),
    }


def run(belief, target, checkpoint, output, fixtures):
    controller = NMPCController(belief)
    plan = controller.plan
    if (
        plan.horizon_steps * plan.sample_period_s
        > belief.maximum_error_horizon_s + 1e-12
    ):
        raise ValueError("forecast exceeds evidence")
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    np.testing.assert_array_equal(reference.states, checkpoint["reference_states"])
    suffix = SuffixProblem(plan, reference)
    state, latent = (
        jnp.asarray(checkpoint["tick4_state"]),
        jnp.asarray(checkpoint["tick4_latent"]),
    )
    previous = jnp.asarray(checkpoint["tick4_previous_command"])
    report = {
        "diagnostic_only": True,
        "recursive_feasibility_claim": False,
        "glassbox_import": glassbox.__file__,
        "horizon_steps": plan.horizon_steps,
        "maximum_error_horizon_s": belief.maximum_error_horizon_s,
        "sample_period_s": plan.sample_period_s,
        "suffix_steps": SUFFIX_STEPS,
        "suffix_variables": SUFFIX_STEPS * plan.command_size,
        "baseline_variables": plan.block_count * plan.command_size,
        "parameterization": "24 frozen per-step commands, six independent per-step commands; no block averaging",
        "fitted_actuator_time_constants_s": np.asarray(
            latent_response_time_constants(plan.values.parameters)
        ).tolist(),
        "fixture_failure": "tick4 original disturbance warm1",
        "continuation": [],
    }
    commands = jnp.asarray(checkpoint["tick4_shifted_commands"])
    accepted, probe = suffix.solve(commands, state, latent, previous)
    report["repair"] = probe
    report["repair_tick"] = 4
    first_repaired_command = (
        None if accepted is None else np.asarray(accepted[24]).copy()
    )
    # The accepted repair and up to 35 further exact shifts form one short
    # restricted receding-horizon experiment. Every iteration solves the same
    # suffix OCP; no projection onto the original blocks is performed.
    for continuation_tick in range(36):
        if accepted is None:
            report["stop_reason"] = "no_feasible_suffix_candidate"
            break
        previous = accepted[0]
        if continuation_tick == 24:
            np.testing.assert_array_equal(previous, first_repaired_command)
        state, latent = step_with_latent(
            target,
            state,
            latent,
            previous,
            plan.sample_period_s,
            belief.input_spec.control_roles,
        )
        actual = float(
            np.max(np.asarray(plan._validity_utilization(state, suffix.exogenous[0])))
        )
        report["continuation"].append(
            {"tick": continuation_tick, "actual_support_utilization": actual}
        )
        if (
            not np.isfinite(actual)
            or not np.all(np.isfinite(state))
            or not np.all(np.isfinite(latent))
        ):
            report["stop_reason"] = "nonfinite_actual_state"
            report["continuation"][-1]["actual_support_utilization"] = None
            break
        if actual > 1 + FEASIBILITY_TOLERANCE:
            report["stop_reason"] = "actual_state_left_support"
            break
        if continuation_tick == 35:
            break
        commands = jnp.concatenate((accepted[1:], accepted[-1:]))
        accepted, row = suffix.solve(commands, state, latent, previous)
        report["continuation"][-1]["next_solve"] = row
    report["continuation_applied_intervals"] = len(report["continuation"])
    report["initial_observation_intervals"] = 10
    report["extended_run_limit_intervals"] = 36
    report["first_repaired_suffix_application_tick"] = (
        24 if len(report["continuation"]) > 24 else None
    )
    report["continuation_applied_repaired_suffix_commands"] = (
        len(report["continuation"]) > 24
    )
    report["applied_intervals_originating_in_optimized_suffix"] = max(
        len(report["continuation"]) - 24, 0
    )
    report.setdefault("stop_reason", "bounded_continuation_complete")
    report.update(fingerprints(fixtures))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from glassbox.core.synthetic import true_parameters
    from glassbox.workflows.benchmarks import recovery

    data = np.load(args.fixtures / "horizon-shift-states.npz")
    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    run(
        DynamicsBelief.load(args.fixtures / "rich-belief.json"),
        target,
        data,
        args.output,
        args.fixtures,
    )
