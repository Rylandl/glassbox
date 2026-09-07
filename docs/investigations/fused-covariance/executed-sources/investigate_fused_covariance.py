"""Study sharing the nominal rollout with parameter-sensitivity propagation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from investigate_feedback_recovery import RecoverySolver
from investigate_feedback_suffix import FeedbackSuffixPlan, checker
from investigate_single_seed import load_request, request, same_tree
from investigate_terminal_suffix import fingerprints

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import BeliefPlanModel, NMPCController
from glassbox.control.plan import Prediction
from glassbox.core.dynamics import (
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.synthetic import resting_state


class FusedCovariancePlan(BeliefPlanModel):
    """Experimental exact-command rollout with one shared nominal scan."""

    def rollout_commands(self, commands, state, latent, exogenous, values):
        if commands.shape != (self.horizon_steps, self.command_size):
            raise ValueError("commands must have one row per prediction interval")
        if values.covariance_factor is None:
            return super().rollout_commands(commands, state, latent, exogenous, values)
        center = structured_parameter_vector(values.parameters)

        def varied_states(vector):
            parameters = with_structured_parameter_vector(values.parameters, vector)
            states, latents, _ = self._mean_rollout_commands(
                commands, state, latent, exogenous, parameters
            )
            return states, latents

        def direction(column):
            states, tangent, latents = jax.jvp(
                varied_states, (center,), (column,), has_aux=True
            )

            # Hold the nominal trajectory fixed only in the inner parameter
            # derivative. Outer command derivatives still see its local frame.
            def errors(varied):
                return jax.vmap(rigid_body_local_error)(states[1:], varied[1:])

            spread = jax.jvp(errors, (states,), (tangent,))[1]
            return spread, states, latents

        directions, states, latents = jax.vmap(direction, out_axes=(0, None, None))(
            values.covariance_factor.T
        )
        covariance = values.forecast_error_covariance + jnp.einsum(
            "kti,ktj->tij", directions, directions
        )
        return Prediction(states, covariance, commands, latents, exogenous)


class FusedFeedbackSuffixPlan(FusedCovariancePlan, FeedbackSuffixPlan):
    """Retain the same free head/tail and dynamic frozen middle waveform."""


class FusedRecoverySolver(RecoverySolver):
    """Isolate the experiment from the baseline's compiled kernel cache."""

    plan_type = FusedFeedbackSuffixPlan
    plan_signature = RecoverySolver.plan_signature + ":fused-covariance-v1"


def compare(left, right, *, rtol=2e-5, atol=2e-5):
    """Record numerical differences; do not relax tolerances after a result."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    pairs = list(zip(jax.tree.leaves(left), jax.tree.leaves(right)))
    return {
        "bitwise_equal": same_tree(left, right),
        "within_tolerance": all(
            np.allclose(a, b, rtol=rtol, atol=atol) for a, b in pairs
        ),
        "maximum_absolute_difference": max(
            (
                float(np.max(np.abs(np.asarray(a) - np.asarray(b)), initial=0))
                for a, b in pairs
            ),
            default=0.0,
        ),
    }


def scan_lengths(tree):
    """Count traced scan sites, including nested JIT bodies, once per call site."""
    if hasattr(tree, "eqns"):
        lengths = []
        for equation in tree.eqns:
            if equation.primitive.name == "scan":
                lengths.append(equation.params["length"])
            lengths.extend(scan_lengths(equation.params))
        return lengths
    if hasattr(tree, "jaxpr"):
        return scan_lengths(tree.jaxpr)
    if isinstance(tree, dict):
        return scan_lengths(tuple(tree.values()))
    if isinstance(tree, (tuple, list)):
        return [length for value in tree for length in scan_lengths(value)]
    return []


def kernel_work(function, arguments):
    traced = jax.make_jaxpr(function)(*arguments)
    compiled = jax.jit(function).lower(*arguments).compile()
    analysis = compiled.cost_analysis()
    return {
        "traced_scan_lengths": scan_lengths(traced),
        "compiled_while_sites": len(re.findall(r"\bwhile\(", compiled.as_text())),
        "compiler_cost_estimates": {
            name: analysis[name]
            for name in ("flops", "transcendentals", "bytes accessed")
            if name in analysis
        },
    }


def run(fixtures, records, output):
    output.mkdir(parents=True, exist_ok=False)
    prior_path = records / "single-seed-reuse/report.json"
    prior = json.loads(prior_path.read_text())
    provenance = fingerprints(fixtures)
    assert provenance["fixture_sha256"] == prior["fixture_sha256"]
    controller = NMPCController(DynamicsBelief.load(fixtures / "rich-belief.json"))
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    with np.load(fixtures / "horizon-shift-states.npz") as fixture:
        previous = fixture["tick0_previous_command"].copy()
        np.testing.assert_array_equal(reference.states, fixture["reference_states"])
    solvers = [
        cls(
            controller.plan,
            jnp.repeat(jnp.asarray(previous)[None, :], 30, axis=0),
            work_estimates=None,
        )
        for cls in (RecoverySolver, FusedRecoverySolver)
    ]
    for solver in solvers:
        solver.reuse_single_seed = True
    assert solvers[0]._kernels is not solvers[1]._kernels
    check = checker(controller.plan, reference)
    report = {
        "diagnostic_only": True,
        "deadline_s": None,
        "host_timing_acceptance_gate": False,
        "runtime_performance_claim": False,
        "design": "Same 26 saved requests as the single-seed comparison, baseline then fused, without deadlines or refitting. Compare seed derivatives and prepared predictions, full solves and independent physical-waveform feasibility.",
        "tolerances": {"rtol": 2e-5, "atol": 2e-5},
        "work_count_scope": "Scan/while sites are structural; compiler cost estimates depend on compiler/backend and are not measured whole-request operation counts or speedup estimates.",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "platform": platform.platform(),
        "input_report_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "checks": [],
        **provenance,
    }
    root = Path(__file__).resolve().parents[1]
    report["baseline_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    archive = output / "executed-sources"
    archive.mkdir()
    for name in (
        Path(__file__).name,
        "investigate_sqp_recovery.py",
        "investigate_single_seed.py",
        "investigate_feedback_recovery.py",
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
    ):
        content = Path(__file__).with_name(name).read_bytes()
        (archive / name).write_bytes(content)
        report["source_sha256"]["scripts/" + name] = hashlib.sha256(content).hexdigest()

    def save():
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    for name, source in prior["inputs"].items():
        directory = "feedback-recovery"
        if name == "recorded_tick10":
            directory = "feedback-recovery-runtime"
        elif name == "recorded_small68":
            directory = "seed-timing-history"
        item = load_request(
            records / directory, source["case"], source["tick"], previous
        )
        assert item["source"] == source
        report["pending_input"] = name
        save()
        seeds, outcomes, descriptions = [], [], []
        work = {}
        for arm, solver in zip(("baseline", "fused"), solvers):
            solver.set_seed(item["request"])
            arguments = (
                solver._cold_blocks(item["previous"]),
                item["state"],
                item["latent"],
                reference.states,
                item["previous"],
                solver._exogenous_forecast(reference),
                solver.model.values,
            )
            seeds.append(solver.linearize(*arguments))
            if not report["checks"]:
                rollout_arguments = (
                    item["seed"],
                    item["state"],
                    item["latent"],
                    arguments[-2],
                    arguments[-1],
                )
                work[arm] = {
                    "rollout": kernel_work(
                        solver.model.rollout_commands, rollout_arguments
                    ),
                    "linearization": kernel_work(solver.linearize, arguments),
                    "finalizer": kernel_work(solver.finalize, arguments),
                }
            result, description = request(solver, item, reference, check)
            outcomes.append(result)
            descriptions.append(description)
        before, after = outcomes
        row = {
            "input": name,
            "source": source,
            "seed_jacobians": compare(seeds[0][0], seeds[1][0]),
            "seed_values_and_prediction": compare(seeds[0][1], seeds[1][1]),
            "returned_arrays": compare(
                tuple(
                    getattr(before, field)
                    for field in (
                        "command",
                        "predicted_commands",
                        "predicted_states",
                        "predicted_latent_states",
                    )
                ),
                tuple(
                    getattr(after, field)
                    for field in (
                        "command",
                        "predicted_commands",
                        "predicted_states",
                        "predicted_latent_states",
                    )
                ),
            ),
            "objective": compare(
                before.diagnostics.final_objective, after.diagnostics.final_objective
            ),
            "both_usable": before.command_usable and after.command_usable,
            "status_equal": before.status == after.status,
            "feasibility_equal": before.nonlinear_feasibility
            == after.nonlinear_feasibility,
            "baseline_counts": descriptions[0]["counts"],
            "fused_counts": descriptions[1]["counts"],
            "work_counts_equal": descriptions[0]["counts"] == descriptions[1]["counts"],
            "no_deadline_assessment": before.deadline_met is None
            and after.deadline_met is None,
        }
        row["passed"] = all(
            row[field]["within_tolerance"]
            for field in (
                "seed_jacobians",
                "seed_values_and_prediction",
                "returned_arrays",
                "objective",
            )
        ) and all(
            row[field]
            for field in (
                "both_usable",
                "status_equal",
                "feasibility_equal",
                "work_counts_equal",
                "no_deadline_assessment",
            )
        )
        report["checks"].append(row)
        if work:
            report["kernel_work"] = work
        print(f"{name}: {'pass' if row['passed'] else 'FAIL'}", flush=True)
        save()
    report.pop("pending_input", None)
    report["all_passed"] = len(report["checks"]) == 26 and all(
        row["passed"] for row in report["checks"]
    )
    save()
    if not report["all_passed"]:
        raise SystemExit(
            "Fused rollout verification failed; retain the recorded differences."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("docs/investigations"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.records, args.output)
