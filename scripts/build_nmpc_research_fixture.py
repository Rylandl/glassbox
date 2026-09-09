"""Rebuild shared NMPC checkpoints for terminal and uncertainty research.

One fresh belief fit and the known fifth-solve failure supply both investigations.
This is an offline diagnostic fixture, not a control or timing qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from investigate_recovery import ADDITIONAL_DURATION_S, ADDITIONAL_SEEDS, initial_state
from investigate_sqp_recovery import GaussNewtonReference

from glassbox.control.fitted import NMPCController
from glassbox.core.dynamics import hover_control, step_with_latent
from glassbox.core.synthetic import resting_state
from glassbox.workflows.benchmarks import recovery


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "READY").unlink(missing_ok=True)
    print("Building shared beliefs", flush=True)
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
    stale.save(output / "stale-belief.json")
    belief.save(output / "rich-belief.json")
    controller = NMPCController(belief)
    controller.solver = GaussNewtonReference(
        controller.plan, controller.plan.policy, warm_iterations=1, work_estimates=None
    )
    state, latent = jnp.asarray(initial_state(stale, "original")), hover_control(target)
    previous, warm = latent, None
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    arrays = {"reference_states": np.asarray(reference.states)}
    rows = []
    for tick in range(5):
        prefix = f"tick{tick}_"
        arrays.update(
            {
                prefix + "state": np.asarray(state),
                prefix + "latent": np.asarray(latent),
                prefix + "previous_command": np.asarray(previous),
            }
        )
        if warm is not None:
            arrays[prefix + "warm_commands"] = np.asarray(warm.commands)
            arrays[prefix + "shifted_commands"] = np.asarray(
                jnp.concatenate((warm.commands[1:], warm.commands[-1:]))
            )
        result = controller.solve(
            state,
            reference,
            previous,
            applied_command=latent,
            warm_start=warm,
            deadline_s=None,
        )
        rows.append(
            {
                "tick": tick,
                "status": result.status.value,
                "usable": result.command_usable,
                "message": result.message,
            }
        )
        print(rows[-1], flush=True)
        if not result.command_usable:
            break
        arrays.update(
            {
                prefix + "predicted_states": np.asarray(result.predicted_states),
                prefix + "predicted_latent_states": np.asarray(
                    result.predicted_latent_states
                ),
                prefix + "predicted_commands": np.asarray(result.predicted_commands),
            }
        )
        state, latent = step_with_latent(
            target,
            state,
            latent,
            result.command,
            controller.sample_period_s,
            belief.input_spec.control_roles,
        )
        previous, warm = result.command, result.warm_start
    np.savez_compressed(output / "horizon-shift-states.npz", **arrays)
    sources = [
        Path(__file__),
        Path(__file__).with_name("investigate_sqp_recovery.py"),
        Path(__file__).with_name("investigate_recovery.py"),
        *(
            Path(recovery.glassbox.__file__).parent / name
            for name in recovery.BENCHMARK_SOURCE_FILES
        ),
    ]
    root = Path(__file__).resolve().parents[1]
    reproduced = [row["usable"] for row in rows] == [True] * 4 + [False]
    manifest = {
        "scenario_baseline_revision": "4f5cddb1b0095010235ea919fbd2aa6553b2589c",
        "known_fifth_solve_failure_reproduced": reproduced,
        "tick_results": rows,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "source_sha256": {
            str(path.resolve().relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not reproduced:
        raise RuntimeError("known checkpoint was not reproduced; inspect manifest.json")
    np.savez_compressed(
        output / "first-plan-audit.npz",
        commands=arrays["tick0_predicted_commands"],
        state=arrays["tick0_state"],
        latent=arrays["tick0_latent"],
        actual_state=arrays["tick1_state"],
        actual_latent=arrays["tick1_latent"],
    )
    (output / "READY").write_text(
        "Complete. Treat these shared fixtures as read-only.\n"
    )
    print(f"Fixtures ready in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
