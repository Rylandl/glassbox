"""Measure constrained NMPC seed overhead, startup, and enforced deadlines.

This synthetic experiment prewarms compilation without advancing the plant.
A larger startup deadline is an explicit separate case, not an exemption from
its subsequent 20 ms deadlines. An unusable solve ends the run without applying
its hold. No supervisor or secondary controller participates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import jax
import numpy as np
import scipy
from investigate_constrained_recovery import simulate
from investigate_recovery import ADDITIONAL_DURATION_S, ADDITIONAL_SEEDS, initial_state
from investigate_sqp_recovery import (
    CURVATURE_REGULARIZATION,
    FEASIBILITY_TOLERANCE,
    NUMERICAL_INTERIOR_MARGIN,
    GaussNewtonReference,
)

from glassbox.control.fitted import NMPCController
from glassbox.workflows.benchmarks import recovery


def run(output):
    stale, belief, target, _ = recovery._build_beliefs()
    for seed in ADDITIONAL_SEEDS:
        trajectory = recovery._configuration_trajectory(
            target,
            recovery.TARGET_LOG_ARM_LENGTH_RATIO,
            seed=seed,
            duration_s=ADDITIONAL_DURATION_S,
            source_group=f"additional-adaptation-{seed}",
        )
        belief, _ = belief.absorb(trajectory)
    report = {
        "format_version": 1,
        "diagnostic_only": True,
        "complete": False,
        "semantics": {
            "synthetic": True,
            "supervisor_or_secondary_controller": False,
            "kernels_prewarmed_without_advancing_plant": True,
            "initial_mean_support_is_constrained": True,
            "future_support_includes_marginal_uncertainty": True,
            "objective_and_uncertainty_unchanged": True,
            "model_support_unchanged": True,
            "failed_solve_stops_without_applying_fallback": True,
            "feasibility_is_not_a_convergence_claim": True,
            "timing_is_not_a_hard_realtime_guarantee": True,
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration": {
            "sample_period_s": recovery.SAMPLE_DT_S,
            "duration_s": 2.4,
            "cold_sqp_iterations": 8,
            "warm_sqp_iterations": 2,
            "constraint_tolerance": FEASIBILITY_TOLERANCE,
            "numerical_interior_margin": NUMERICAL_INTERIOR_MARGIN,
            "curvature_regularization": CURVATURE_REGULARIZATION,
            "support": belief.support.to_dict(),
            "additional_identification_seeds": list(ADDITIONAL_SEEDS),
        },
        "scenarios": [],
    }
    cases = [
        ("legacy_seed_original", "original", True, None, None),
        ("reused_seed_original", "original", False, None, None),
        ("reused_seed_small", "small", False, None, None),
        ("strict_deadline_original", "original", False, 0.02, 0.02),
        ("startup_deadline_original", "original", False, 0.02, 0.1),
        ("startup_deadline_small", "small", False, 0.02, 0.1),
        ("outside_initial_support", "outside", False, None, None),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, disturbance, legacy, deadline, startup in cases:
        print(name, flush=True)
        controller = NMPCController(belief)
        controller.solver = GaussNewtonReference(
            controller.plan, controller.plan.policy, legacy_seeding=legacy
        )
        initial = np.asarray(
            initial_state(
                stale, "original" if disturbance == "outside" else disturbance
            )
        ).copy()
        if disturbance == "outside":
            envelope = belief.model.runtime_spec.validity_envelope
            initial[10] = (
                envelope.angular_velocity_center_rad_s[0]
                + 1.1 * envelope.angular_velocity_half_width_rad_s[0]
            )
        row = simulate(
            belief,
            target,
            initial,
            optimizer="gauss_newton_sqp",
            controller=controller,
            prewarm=True,
            deadline_s=deadline,
            startup_deadline_s=startup,
        )
        times = np.asarray(row["solve_times_s"])
        row.update(
            name=name,
            disturbance=disturbance,
            legacy_seeding=legacy,
            cold_solve_time_s=float(times[0]),
            steady_solve_time_median_s=float(np.median(times[1:]))
            if len(times) > 1
            else None,
            steady_solve_time_maximum_s=float(np.max(times[1:]))
            if len(times) > 1
            else None,
            steady_deadline_miss_count=int(np.sum(times[1:] > recovery.SAMPLE_DT_S)),
        )
        report["scenarios"].append(row)
        print(
            {
                key: row[key]
                for key in (
                    "complete",
                    "executed_intervals",
                    "stop_status",
                    "tail_normalized_tracking_rms",
                    "maximum_actual_validity_utilization",
                    "cold_solve_time_s",
                    "steady_solve_time_median_s",
                    "steady_deadline_miss_count",
                )
            },
            flush=True,
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    sources = [
        Path(__file__),
        *(
            Path(__file__).with_name(name)
            for name in (
                "investigate_sqp_recovery.py",
                "investigate_constrained_recovery.py",
                "investigate_recovery.py",
            )
        ),
        *(
            Path(recovery.glassbox.__file__).parent / name
            for name in recovery.BENCHMARK_SOURCE_FILES
        ),
    ]
    report["source_sha256"] = {
        str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
