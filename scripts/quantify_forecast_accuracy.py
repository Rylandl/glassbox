"""Replay frozen forecasts into empirical tolerances; do not fit or select models."""

import argparse
import hashlib
import json
import math
import platform
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glassbox.experimental.forecast_accuracy import assess_forecasts


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feedback_example():
    """A scalar output-error recurrence, not a vehicle model or controller."""
    n, amplitude, tolerance = 500, 0.01, 0.02
    rows = []
    for contraction in (0.2, 0.8):
        for kind in ("bias", "alternating"):
            residual = amplitude * (
                np.ones(n) if kind == "bias" else (-1.0) ** np.arange(n)
            )
            error = np.zeros(n + 1)
            for k in range(n):
                error[k + 1] = contraction * error[k] + residual[k]
            rows.append(
                dict(
                    contraction=contraction,
                    residual_kind=kind,
                    residual_rms=float(np.sqrt(np.mean(residual**2))),
                    asymptotic_error_amplitude=amplitude
                    / (1 - contraction if kind == "bias" else 1 + contraction),
                    measured_tail_rms=float(np.sqrt(np.mean(error[-100:] ** 2))),
                    residual_bound_for_tracking_tolerance=(1 - contraction) * tolerance,
                    residual=residual.tolist(),
                    tracking_error=error.tolist(),
                )
            )
    return dict(
        definition="e[k+1] = contraction * e[k] + residual[k], e[0]=0",
        units="arbitrary common output units, not vehicle coordinates",
        tracking_tolerance=tolerance,
        rows=rows,
        limitation="The contraction and additive residual are stipulated, not identified from the flight forecasts. Forecast RMSE is not a per-step disturbance bound.",
    )


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    plan = read(a.plan)
    write(a.output / "plan.json", plan)
    repo = Path(__file__).resolve().parents[1]
    sources = [
        repo / "src/glassbox/experimental/forecast_accuracy.py",
        Path(__file__).resolve(),
        repo / "tests/test_forecast_accuracy.py",
    ]
    with zipfile.ZipFile(
        a.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sources:
            z.write(p, p.relative_to(repo))
    write(
        a.output / "sources.json", {str(p.relative_to(repo)): sha(p) for p in sources}
    )
    write(
        a.output / "environment.json",
        dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
    )
    reports, summary, sources = {}, [], {}
    prior = read(repo / "docs/investigations/default-recipe/run-artifacts.json")
    for folder in sorted(p for p in a.run.iterdir() if p.is_dir()):
        for name in ("report.json", "evaluation.npz", "evaluation-usage.json"):
            path = folder / name
            digest = sha(path)
            assert digest == prior[str(path.relative_to(a.run))]["sha256"]
            sources[str(path.resolve())] = digest
        meta = read(folder / "report.json")
        usage = read(folder / "evaluation-usage.json")
        ids = [k["recording_id"] for k in usage["keys"]]
        sensor = meta["case"] == "crazyflie-sensor"
        metrics = (
            ("acceleration_g", "gyro_rad_s")
            if sensor
            else ("velocity_m_s", "body_rate_rad_s")
        )
        limits = {m: plan["illustrative_limits"][m] for m in metrics}
        arms = {}
        with np.load(folder / "evaluation.npz") as z:
            truth = z["future_states"]
            times = np.arange(1, truth.shape[1] + 1) * meta["dt_s"]
            for arm in meta["rmse"]:
                difference = z[f"prediction_{arm}"] - truth
                errors = {
                    metric: np.linalg.norm(difference[:, :, list(group)], axis=-1)
                    for metric, group in zip(metrics, meta["groups"][:2])
                }
                arms[arm] = assess_forecasts(errors, ids, times_s=times, limits=limits)
        latest = "updated" if "updated" in arms else "initial"
        reports[meta["case"]] = dict(
            case=meta["case"],
            latest=latest,
            revision_fingerprints={
                n: r["fingerprint"] for n, r in meta["models"].items()
            },
            forecast_contract=meta["contract"],
            evaluation_origins=usage,
            arms=arms,
            rotation_entry_rmse={
                n: [row[2] for row in v] for n, v in meta["rmse"].items()
            }
            if not sensor
            else None,
        )
        summary.append(
            dict(
                case=meta["case"],
                latest=latest,
                horizon_s=float(times[-1]),
                recordings=len(set(ids)),
                windows=len(ids),
                metrics=metrics,
                arms={
                    n: dict(
                        final_required_marginal_tolerance={
                            m: curve[-1]
                            for m, curve in r["required_marginal_tolerance"].items()
                        },
                        final_minimum_recording_joint_fraction=r[
                            "minimum_recording_joint_fraction"
                        ][-1],
                        observed_95pct_horizon_s=r["observed_95pct_horizon_s"],
                    )
                    for n, r in arms.items()
                },
            )
        )
    write(a.output / "source-artifacts.json", sources)
    write(a.output / "reports.json", reports)
    write(a.output / "summary.json", summary)
    write(a.output / "feedback-example.json", feedback_example())
    write(
        a.output / "independent-trial-requirements.json",
        dict(
            assumption="Independent identically distributed Bernoulli task trials, zero failures, fixed task and model, one-sided exact 95% confidence lower bound",
            rows=[
                dict(
                    desired_success_probability=p,
                    zero_failure_trials=math.ceil(math.log(0.05) / math.log(p)),
                )
                for p in (0.95, 0.99, 0.999)
            ],
            limitation="Not applied to these dependent forecast windows; repeated selection requires its own evaluation protocol.",
        ),
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    order = ["nano", "x8", *[f"arp-fold{i}" for i in range(4)]]
    for i, case in enumerate(order):
        r = reports[case]
        latest = r["arms"][r["latest"]]
        label = case.upper() if case in ("nano", "x8") else f"ARP log{63 + i - 2}"
        for ax, metric in zip(axes, ("velocity_m_s", "body_rate_rad_s")):
            ax.plot(
                np.array(latest["times_s"]) * 1000,
                latest["required_marginal_tolerance"][metric],
                label=label,
            )
    for ax, unit in zip(
        axes, ("Velocity allowance [m/s]", "Body-rate allowance [rad/s]")
    ):
        ax.set(xlabel="Forecast horizon [ms]", ylabel=unit, ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Allowance needed for 95% of observed windows in every recording\n"
        "Maximum error anywhere through each horizon · latest saved revision · empirical, not a guarantee",
        fontsize=11,
    )
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"forecast-allowance.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", type=Path, default=Path("../artifacts/default-recipe/comparison-01")
    )
    p.add_argument(
        "--plan",
        type=Path,
        default=Path("../artifacts/accuracy-requirements/plan.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
