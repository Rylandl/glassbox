"""Predeclared six-replicate synthetic identification pilot; no covariance changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.forecast_error import EmpiricalErrorSample, ForecastErrorEnvelope
from glassbox.belief.information import ParameterInformation, innovation_noise_floor
from glassbox.core.dynamics import (
    hover_control,
    step_with_latent,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.model import ExecutableModel, runtime_spec_from_trajectory
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.workflows.benchmarks import recovery

DESIGN = dict(
    replicates=6,
    train_seeds=list(range(91000, 91006)),
    calibration_seeds=list(range(92000, 92012)),
    test_seeds=[93000, 93001],
    train_duration_s=3.0,
    command_source_duration_s=1.2,
    dt_s=0.02,
    horizons_steps=[5, 30],
    command_selection="last 30 commands; score prefixes",
    initial_state="resting_state",
    initial_latent="true hover",
    estimator="one production DynamicsBelief.absorb per independent training flight; same deterministic prior",
    stochasticity="independent excitation phases only; no process or observation noise",
    retuning=False,
)


def second(x):
    x = np.asarray(x)
    return np.einsum("...i,...j->...ij", x, x)


def decompose(errors, linear):
    errors, linear = np.asarray(errors), np.asarray(linear)
    remainder = errors - linear
    cross = (
        linear[..., :, None] * remainder[..., None, :]
        + remainder[..., :, None] * linear[..., None, :]
    )
    return dict(
        actual=second(errors),
        linear=second(linear),
        remainder=second(remainder),
        cross=cross,
    )


def bias_variance(errors):
    errors = np.asarray(errors)
    mean = errors.mean(axis=0)
    return dict(
        bias_squared=second(mean),
        centered_variance=second(errors - mean).mean(axis=0),
        second_moment=second(errors).mean(axis=0),
    )


def scalar_control():
    # Exhaustive symmetric finite population, exactly linear, no residual noise.
    delta = np.array([-2.0, -1.0, 0.0, 0.0, 1.0, 2.0])[:, None]
    parts = decompose(3 * delta, 3 * delta)
    mse = float(parts["actual"].mean())
    return dict(
        actual=mse,
        empirical=mse,
        parameter=mse,
        sum=2 * mse,
        remainder=float(parts["remainder"].max()),
        cross=float(parts["cross"].max()),
        scope="exact finite-population algebra control only",
    )


def rollout(params, commands, latent):
    def step(carry, u):
        x, a = step_with_latent(params, *carry, u, 0.02)
        return (x, a), x

    return jax.lax.scan(
        step, (jnp.asarray(resting_state()), latent), jnp.asarray(commands)
    )[1]


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "design.json").write_text(json.dumps(DESIGN, indent=2) + "\n")
    base = true_parameters()
    target = recovery._arm_configuration_parameters(
        base, recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    truth = jnp.asarray(structured_parameter_vector(target))
    latent = hover_control(target)
    members = tuple(
        recovery._arm_configuration_parameters(base, v)
        for v in recovery.FLEET_LOG_ARM_LENGTH_RATIOS
    )
    prior = ParameterInformation.seeded_from_members(
        base,
        members,
        innovation_noise=innovation_noise_floor(),
        source="deterministic_recovery_grid_prior",
    )
    sources = []

    def flight(seed, duration, role):
        t = recovery._configuration_trajectory(
            target,
            recovery.TARGET_LOG_ARM_LENGTH_RATIO,
            seed=seed,
            duration_s=duration,
            source_group=f"{role}-{seed}",
        )
        sources.append(
            dict(
                seed=seed,
                role=role,
                duration_s=duration,
                sha256=digest_arrays(t.states, t.controls),
                source_group=t.labels["source_group"],
            )
        )
        return t

    tests = [flight(s, 1.2, "test").controls[-30:] for s in DESIGN["test_seeds"]]
    actual = [rollout(target, u, latent) for u in tests]
    truth_jac = [
        jax.jacfwd(
            lambda theta, x=x, u=u: jax.vmap(rigid_body_local_error)(
                x, rollout(with_structured_parameter_vector(target, theta), u, latent)
            )
        )(truth)
        for x, u in zip(actual, tests)
    ]
    rows = []
    all_errors = []
    all_linear = []
    all_e = []
    all_p = []
    theta_errors = []
    for r in range(6):
        print(f"FIT {r + 1}/6 active", flush=True)
        train = flight(91000 + r, 3.0, "train")
        model = ExecutableModel(base, train.spec, runtime_spec_from_trajectory(train))
        fitted, result = DynamicsBelief(model, prior).absorb(train)
        samples = {h * 0.02: [] for h in [5, 30]}
        for seed in [92000 + 2 * r, 92001 + 2 * r]:
            calibration = flight(seed, 1.2, "calibration").controls[-30:]
            cal_truth = rollout(target, calibration, latent)
            cal_fit = rollout(fitted.params, calibration, latent)
            cal_error = np.asarray(jax.vmap(rigid_body_local_error)(cal_truth, cal_fit))
            for h in [5, 30]:
                samples[h * 0.02].append(
                    EmpiricalErrorSample(
                        cal_error[h - 1 : h],
                        f"calibration-{seed}",
                        f"calibration-{seed}",
                    )
                )
        envelope = ForecastErrorEnvelope.from_samples(samples)
        fitted = replace(fitted, forecast_error=envelope)
        delta = np.asarray(structured_parameter_vector(fitted.params) - truth)
        theta_errors.append(delta)
        errors = []
        linear = []
        p = []
        e = []
        for u, x, jac in zip(tests, actual, truth_jac):
            prediction = fitted.rollout(
                jnp.asarray(resting_state()),
                jnp.asarray(u),
                initial_latent_state=latent,
            )
            errors.append(
                np.asarray(jax.vmap(rigid_body_local_error)(x, prediction.states[1:]))[
                    [4, 29]
                ]
            )
            linear.append(np.einsum("hip,p->hi", np.asarray(jac), delta)[[4, 29]])
            p.append(np.asarray(prediction.parameter_covariance)[[5, 30]])
            e.append(np.asarray(prediction.forecast_error_covariance)[[5, 30]])
        all_errors.append(errors)
        all_linear.append(linear)
        all_p.append(p)
        all_e.append(e)
        fitted.save(output / f"belief-{r}.json")
        rows.append(
            dict(
                replicate=r,
                update=result.to_dict(),
                rank=fitted.information.resolved_rank(),
                estimable_count=fitted.information.estimable_count,
                complete=fitted.information.complete,
                calibration_independent_group_count=list(
                    envelope.independent_group_count
                ),
            )
        )
        print(f"FIT {r + 1}/6 complete rank={rows[-1]['rank']}", flush=True)
    errors = np.asarray(all_errors)
    linear = np.asarray(all_linear)
    e = np.asarray(all_e)
    p = np.asarray(all_p)
    parts = decompose(errors, linear)
    bv = bias_variance(errors)
    closure = float(
        np.max(
            np.abs(
                parts["actual"] - parts["linear"] - parts["remainder"] - parts["cross"]
            )
        )
    )
    arrays = dict(
        errors=errors,
        linear_errors=linear,
        empirical=e,
        parameter=p,
        combined=e + p,
        theta_errors=np.asarray(theta_errors),
        **parts,
        **bv,
    )
    np.savez_compressed(output / "arrays.npz", **arrays)
    # Report each tangent coordinate separately: heterogeneous units are never summed.
    summary = {
        name: np.diagonal(value.mean(axis=(0, 1)), axis1=-2, axis2=-1).tolist()
        for name, value in {
            **parts,
            "empirical": e,
            "parameter": p,
            "combined": e + p,
        }.items()
    }
    summary.update(
        {
            name: np.diagonal(value.mean(axis=0), axis1=-2, axis2=-1).tolist()
            for name, value in bv.items()
        }
    )
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        *list((root / "src/glassbox/belief").glob("*.py")),
        root / "src/glassbox/core/synthetic.py",
        root / "src/glassbox/workflows/benchmarks/recovery.py",
    ]
    report = dict(
        design=DESIGN,
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        import_path=glassbox.__file__,
        jax_version=jax.__version__,
        x64=bool(jax.config.jax_enable_x64),
        sources=sources,
        source_sha256={
            str(v.relative_to(root)): hashlib.sha256(v.read_bytes()).hexdigest()
            for v in paths
        },
        replicates=rows,
        known_theta=np.asarray(truth).tolist(),
        prior=prior.to_dict(),
        summary_diagonal=summary,
        decomposition_closure_max=closure,
        scalar_control=scalar_control(),
        unresolved_flags=[
            "Six fits are a conditional-design pilot, not population calibration.",
            "Only two independent calibration endpoints per horizon per fitted model; empirical covariance rank is at most two.",
            "Two shared test designs; errors across designs and horizons are not independent replicates.",
            "No process/measurement noise; known-state deterministic plant errors are entirely parameter-induced.",
            "Production training uses actuator history reconstruction; its estimation effects are not separately identified.",
            "One absorb update is a bounded identification pass, not a converged batch optimizer.",
            "Fixed noise floor and prior are declarations, not empirically calibrated sampling covariance.",
            "Training support is per-fit; no support expansion; rejected transitions remain excluded.",
            "Truth-chart linear decomposition and nominal-chart predicted covariance have different tangent origins; full matrix equality is not asserted.",
            "No hardware timing or coverage claims.",
        ],
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(dict(closure=closure, summary=summary)), flush=True)


def annotate_validity(output):
    """Evaluate saved fits on existing designs; never fit or change support."""
    report = json.loads((output / "report.json").read_text())
    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    latent = hover_control(target)
    tests = [
        recovery._configuration_trajectory(
            target,
            recovery.TARGET_LOG_ARM_LENGTH_RATIO,
            seed=seed,
            duration_s=1.2,
            source_group=f"test-{seed}",
        ).controls[-30:]
        for seed in DESIGN["test_seeds"]
    ]
    for r, row in enumerate(report["replicates"]):
        belief = DynamicsBelief.load(output / f"belief-{r}.json")
        row["training_transition_count"] = 150
        row["training_retained_count"] = row["update"]["window_count"]
        row["training_omitted_count"] = 150 - row["training_retained_count"]
        row["test_validity"] = []
        for seed, u in zip(DESIGN["test_seeds"], tests):
            actual = rollout(target, u, latent)
            nominal = rollout(belief.params, u, latent)
            true_v = np.asarray(
                jax.vmap(belief.model.validity_utilization)(
                    jnp.concatenate([jnp.asarray(resting_state())[None], actual])
                )
            )
            nominal_v = np.asarray(
                jax.vmap(belief.model.validity_utilization)(
                    jnp.concatenate([jnp.asarray(resting_state())[None], nominal])
                )
            )
            row["test_validity"].append(
                dict(
                    seed=seed,
                    true_utilization=true_v.tolist(),
                    nominal_utilization=nominal_v.tolist(),
                    horizons=[
                        dict(
                            steps=h,
                            true_max=float(true_v[: h + 1].max()),
                            nominal_max=float(nominal_v[: h + 1].max()),
                            both_inside=bool(
                                max(true_v[: h + 1].max(), nominal_v[: h + 1].max())
                                <= 1
                            ),
                        )
                        for h in [5, 30]
                    ],
                )
            )
    report["validity_annotation"] = (
        "Saved-fit postprocessing after pilot; no additional fitting; same test sources regenerated exactly."
    )
    report["annotation_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotate-only", action="store_true")
    args = parser.parse_args()
    if not args.annotate_only:
        run(args.output)
    annotate_validity(args.output)
