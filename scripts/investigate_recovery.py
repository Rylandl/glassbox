"""Reproduce the recovery uncertainty ablations and an offline solver reference.

Run with the project environment:
    python scripts/investigate_recovery.py --output docs/investigations/recovery.json

The ablations deliberately omit evidence contributions while holding the mean
fixed. They are diagnostic comparisons, not deployable belief artifacts. The
SciPy reference is offline only and uses the same bounded command variables,
objective, derivatives, horizon, and warm-start selection as the maintained
solver. No controller weights or covariance eigenvalue cutoffs are tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import minimize

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController, default_solver_policy
from glassbox.control.solver import BoundedShootingSolver, _OptimizerOutcome
from glassbox.core.dynamics import (
    hover_control,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import quaternion_from_euler, rigid_body_local_error
from glassbox.core.metrics import predict_windows
from glassbox.core.synthetic import resting_state
from glassbox.workflows.benchmarks import recovery

ADDITIONAL_SEEDS = tuple(range(31, 37))
ADDITIONAL_DURATION_S = 2.0


class OfflineReferenceSolver(BoundedShootingSolver):
    """L-BFGS-B over exactly the maintained objective; no real-time claim."""

    def _optimize_plan(
        self,
        blocks,
        value,
        gradient,
        state,
        latent,
        reference,
        previous_command,
        exogenous,
    ):
        shape = blocks.shape

        def objective(vector):
            result, derivative = self._kernels.objective_and_gradient(
                jnp.asarray(vector.reshape(shape)),
                state,
                latent,
                reference.states,
                previous_command,
                exogenous,
                self.model.values,
            )
            return float(result), np.asarray(derivative, dtype=np.float64).ravel()

        result = minimize(
            objective,
            np.asarray(blocks, dtype=np.float64).ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-1.0, 1.0)] * blocks.size,
            options={
                "maxiter": 500,
                "maxls": 40,
                "maxcor": 20,
                "ftol": 1e-12,
                "gtol": self.policy.gradient_tolerance,
            },
        )
        final_value, final_gradient = objective(result.x)
        candidate = result.x.reshape(shape)
        derivative = final_gradient.reshape(shape)
        finite = bool(np.isfinite(final_value) and np.all(np.isfinite(derivative)))
        progressed = finite and final_value < float(value)
        # Never retain a worse or nonfinite result from the offline optimizer.
        if not finite or final_value > float(value):
            candidate = np.asarray(blocks)
            final_value = float(value)
            derivative = np.asarray(gradient)
        residual = np.max(np.abs(candidate - np.clip(candidate - derivative, -1, 1)))
        converged = bool(residual <= self.policy.gradient_tolerance)
        return _OptimizerOutcome(
            blocks=jnp.asarray(candidate),
            value=jnp.asarray(final_value),
            gradient=jnp.asarray(derivative),
            iterations=result.nit,
            converged=converged,
            stalled=not converged,
            line_search_failed=False,
            progressed=progressed,
            finite=True,
            stall_message=str(result.message),
        )


class ObservedController(NMPCController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = []

    def solve(self, *args, **kwargs):
        result = super().solve(*args, **kwargs)
        self.diagnostics.append(asdict(result.diagnostics))
        return result


def initial_state(belief: DynamicsBelief, disturbance: str) -> np.ndarray:
    if disturbance == "original":
        return recovery._recovery_initial_state(belief)
    state = resting_state()
    state[:3] = (0.025, -0.020, -0.015)
    state[6:10] = quaternion_from_euler(0.024, -0.017, 0.012)
    state[10:] = (0.03, -0.03, 0.03)
    return state


def controller_for(belief, contributions, optimizer):
    policy = replace(default_solver_policy(belief), allow_unresolved_parameters=True)
    if optimizer == "projected_128":
        policy = replace(
            policy, maximum_iterations=128, relative_improvement_tolerance=1e-8
        )
    controller = ObservedController(belief, policy=policy)
    values = controller.plan.values
    if contributions not in ("parameter", "both"):
        values = values._replace(
            covariance_factor=jnp.zeros_like(values.covariance_factor)
        )
    if contributions not in ("forecast", "both"):
        values = values._replace(
            forecast_error_covariance=jnp.zeros_like(values.forecast_error_covariance)
        )
    controller.plan = replace(controller.plan, values=values)
    solver = (
        OfflineReferenceSolver if optimizer == "lbfgsb_500" else BoundedShootingSolver
    )
    controller.solver = solver(controller.plan, policy)
    return controller


def covariance_probe(belief, state):
    """Compare local derivatives with nonlinear +/- one-factor displacements.

    These are points on the one-Mahalanobis-radius ellipsoid, not a Monte Carlo
    coverage estimate. A finite result does not establish calibration. Probing
    both signs makes the check independent of eigensolver sign conventions.
    """
    controller = controller_for(belief, "both", "projected_8")
    plan = controller.plan
    previous = hover_control(belief.params)
    latent = plan.initial_latent(previous[None, :], plan.values)
    exogenous = jnp.zeros((plan.horizon_steps, plan.exogenous_size))
    normalized = (
        2
        * (previous - plan.command_minimum)
        / (plan.command_maximum - plan.command_minimum)
        - 1
    )
    blocks = jnp.tile(normalized, (plan.block_count, 1))
    states, _, _ = plan._mean_rollout(blocks, state, latent, exogenous, belief.params)
    center = structured_parameter_vector(belief.params)

    def error(vector):
        parameters = with_structured_parameter_vector(belief.params, vector)
        varied, _, _ = plan._mean_rollout(blocks, state, latent, exogenous, parameters)
        return jax.vmap(rigid_body_local_error)(states[1:], varied[1:])

    factor = plan.values.covariance_factor
    linear = jax.jit(jax.vmap(lambda col: jax.jvp(error, (center,), (col,))[1]))(
        factor.T
    )
    nonlinear = jax.jit(jax.vmap(error))(
        jnp.concatenate((center + factor.T, center - factor.T))
    )
    linear = np.asarray(linear, dtype=np.float64)
    nonlinear = np.asarray(nonlinear, dtype=np.float64)
    rank = factor.shape[1]
    scale = np.asarray(controller.tolerances.local_state_scale)
    spread = linear / scale
    cost = np.mean(
        np.sum(spread**2, axis=2), axis=1
    ) + plan.policy.terminal_weight * np.sum(spread[:, -1] ** 2, axis=1)
    modes = []
    for index in np.argsort(cost)[::-1]:
        column = np.asarray(factor[:, index])
        largest = np.argsort(np.abs(column))[::-1][:5]
        relative_errors = []
        for sign, row in ((1, index), (-1, rank + index)):
            actual = nonlinear[row]
            if not np.all(np.isfinite(actual)):
                relative_errors.append(None)
            else:
                relative_errors.append(
                    float(
                        np.linalg.norm(actual - sign * linear[index])
                        / max(np.linalg.norm(linear[index]), 1e-12)
                    )
                )
        modes.append(
            {
                "factor_index": int(index),
                "linearized_tracking_spread_cost": float(cost[index]),
                "cost_fraction": float(cost[index] / np.sum(cost)),
                "largest_parameter_displacements": {
                    belief.information.names[i]: float(column[i]) for i in largest
                },
                "relative_nonlinear_error_plus_minus": relative_errors,
            }
        )
    information = belief.information
    normalized_covariance = information.covariance() / np.outer(
        information.scale, information.scale
    )
    return {
        "resolved_rank": information.resolved_rank(),
        "complete": information.complete,
        "maximum_normalized_parameter_standard_deviation": float(
            np.sqrt(np.max(np.linalg.eigvalsh(normalized_covariance)))
        ),
        "nonfinite_sigma_point_count": int(
            np.count_nonzero(~np.all(np.isfinite(nonlinear), axis=(1, 2)))
        ),
        "sigma_point_count": 2 * rank,
        "modes": modes,
    }


def prediction_rms(belief, trajectory):
    errors = predict_windows(
        belief.params,
        trajectory,
        horizon_steps=recovery.CONTROL_HORIZON_STEPS,
        stride=recovery.CONTROL_HORIZON_STEPS,
    ).endpoint_tangent_errors()
    scale = controller_for(belief, "both", "projected_8").tolerances.local_state_scale
    return float(np.sqrt(np.mean(np.square(errors / np.asarray(scale)))))


def run(output: Path):
    stale, adapted, target, evidence = recovery._build_beliefs()
    small = initial_state(stale, "small")
    report: dict[str, Any] = {
        "format_version": 1,
        "diagnostic_only": True,
        "implementation": {
            "recovery_source_sha256": recovery.adaptive_recovery_source_fingerprint(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration": {
            "additional_telemetry_seeds": list(ADDITIONAL_SEEDS),
            "additional_duration_per_seed_s": ADDITIONAL_DURATION_S,
            "disturbance_states": {
                name: initial_state(stale, name).tolist()
                for name in ("small", "original")
            },
            "contributions": ["point", "forecast", "parameter", "both"],
            "reference": "offline L-BFGS-B, identical objective and command bounds",
        },
        "initial_evidence": evidence,
        "covariance_probes": {
            "short_adaptation": covariance_probe(adapted, jnp.asarray(small))
        },
        "additional_updates": [],
        "runs": [],
    }
    informed = adapted
    for seed in ADDITIONAL_SEEDS:
        telemetry = recovery._configuration_trajectory(
            target,
            recovery.TARGET_LOG_ARM_LENGTH_RATIO,
            seed=seed,
            duration_s=ADDITIONAL_DURATION_S,
            source_group=f"additional-adaptation-{seed}",
        )
        informed, update = informed.absorb(telemetry)
        report["additional_updates"].append({"seed": seed, **update.to_dict()})
    report["covariance_probes"]["additional_evidence"] = covariance_probe(
        informed, jnp.asarray(small)
    )
    evaluation = recovery._configuration_trajectory(
        target,
        recovery.TARGET_LOG_ARM_LENGTH_RATIO,
        seed=22,
        duration_s=recovery.EVALUATION_DURATION_S,
        source_group="target-configuration-independent-evaluation",
    )
    report["independent_prediction_rms"] = {
        name: prediction_rms(belief, evaluation)
        for name, belief in (
            ("stale", stale),
            ("short_adaptation", adapted),
            ("additional_evidence", informed),
        )
    }
    cases = (
        [
            ("short_adaptation", "small", contribution, optimizer)
            for optimizer in ("projected_8", "projected_128")
            for contribution in ("point", "forecast", "parameter", "both")
        ]
        + [
            ("short_adaptation", disturbance, contribution, "lbfgsb_500")
            for disturbance in ("small", "original")
            for contribution in ("point", "both")
        ]
        + [
            ("additional_evidence", disturbance, contribution, optimizer)
            for disturbance in ("small", "original")
            for optimizer in ("projected_8", "lbfgsb_500")
            for contribution in ("point", "both")
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for belief_name, disturbance, contribution, optimizer in cases:
        print(belief_name, disturbance, contribution, optimizer, flush=True)
        belief = adapted if belief_name == "short_adaptation" else informed
        controller = controller_for(belief, contribution, optimizer)
        metrics = recovery._simulate_recovery(
            contribution,
            "controlled diagnostic ablation",
            controller,
            target,
            initial_state(stale, disturbance),
        )
        diagnostics = controller.diagnostics[2:]  # Exclude the two prewarm solves.
        report["runs"].append(
            {
                "belief": belief_name,
                "disturbance": disturbance,
                "contributions": contribution,
                "optimizer": optimizer,
                "uncertainty_contributions_omitted": contribution != "both",
                "metrics": asdict(metrics),
                "median_projected_gradient_inf_norm": float(
                    np.median(
                        [
                            row["final_projected_gradient_inf_norm"]
                            for row in diagnostics
                        ]
                    )
                ),
                "maximum_projected_gradient_inf_norm": max(
                    row["final_projected_gradient_inf_norm"] for row in diagnostics
                ),
                "first_solve": diagnostics[0],
            }
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            "tail tracking",
            metrics.tail_normalized_tracking_rms,
            "statuses",
            metrics.solve_status_counts,
            flush=True,
        )
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
