"""Independently replay physical errors, empirical ranks and whole-prefix counts."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def audit(a):
    comparisons, maximum, scored = 0, 0.0, 0

    def check(x, y):
        nonlocal comparisons, maximum
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert x.shape == y.shape
        maximum = max(maximum, float(np.max(np.abs(x - y))))
        np.testing.assert_allclose(x, y, atol=5e-12, rtol=5e-12)
        comparisons += 1

    with zipfile.ZipFile(a.run / "executed-sources.zip") as z:
        for name, digest in read(a.run / "sources.json").items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    for source, digest in read(a.run / "source-artifacts.json").items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    reports = read(a.run / "reports.json")
    for case, report in reports.items():
        original = a.forecasts / case
        old = read(original / "report.json")
        usage = read(original / "evaluation-usage.json")
        assert report["forecast_contract"] == old["contract"]
        assert report["evaluation_origins"] == usage
        assert report["latest"] == (
            "updated" if "updated" in old["models"] else "initial"
        )
        assert report["revision_fingerprints"] == {
            n: r["fingerprint"] for n, r in old["models"].items()
        }
        ids = [k["recording_id"] for k in usage["keys"]]
        with np.load(original / "evaluation.npz") as z:
            truth = z["future_states"]
            for arm, result in report["arms"].items():
                times = (np.arange(truth.shape[1]) + 1) * old["dt_s"]
                check(times, result["times_s"])
                errors = [
                    np.sqrt(
                        sum(
                            (z[f"prediction_{arm}"][:, :, c] - truth[:, :, c]) ** 2
                            for c in group
                        )
                    )
                    for group in old["groups"][:2]
                ]
                curves = {name: [] for name in result["limits"]}
                fractions = []
                for name, record in result["recordings"].items():
                    index = [i for i, r in enumerate(ids) if r == name]
                    assert len(index) == record["windows"]
                    n, h = len(index), len(times)
                    rank = math.ceil(0.95 * n) - 1
                    within = np.ones((n, h), dtype=bool)
                    for metric, err in zip(result["limits"], errors):
                        end = err[index]
                        # Independent explicit prefix maxima, rather than accumulate.
                        prefix = np.stack(
                            [end[:, : t + 1].max(axis=1) for t in range(h)], axis=1
                        )
                        q_end = [sorted(end[:, t])[rank] for t in range(h)]
                        q_prefix = [sorted(prefix[:, t])[rank] for t in range(h)]
                        values = record["metrics"][metric]
                        check(q_end, values["endpoint_empirical_p95"])
                        check(q_prefix, values["prefix_empirical_p95"])
                        check(np.sqrt(np.mean(end**2, axis=0)), values["endpoint_rmse"])
                        check(prefix.max(axis=0), values["prefix_observed_max"])
                        qualified = prefix <= result["limits"][metric]
                        check(
                            qualified.sum(axis=0), values["prefix_within_limit_count"]
                        )
                        check(np.zeros(h), values["prefix_unavailable_count"])
                        within &= qualified
                        curves[metric].append(q_prefix)
                    fraction = within.sum(axis=0) / n
                    check(
                        within.sum(axis=0), record["joint_prefix_within_limits_count"]
                    )
                    check(fraction, record["joint_prefix_within_limits_fraction"])
                    fractions.append(fraction)
                    scored += n
                for name, required in curves.items():
                    check(
                        np.max(required, axis=0),
                        result["required_marginal_tolerance"][name],
                    )
                worst = np.min(fractions, axis=0)
                check(worst, result["minimum_recording_joint_fraction"])
                eligible = [float(t) for t, f in zip(times, worst) if f >= 0.95]
                assert result["observed_95pct_horizon_s"] == (
                    eligible[-1] if eligible else None
                )
    feedback = read(a.run / "feedback-example.json")
    for row in feedback["rows"]:
        rho, residual = row["contraction"], np.array(row["residual"])
        response = np.convolve(residual, rho ** np.arange(len(residual)))[
            : len(residual)
        ]
        check(response, row["tracking_error"][1:])
        check(np.sqrt(np.mean(response[-100:] ** 2)), row["asymptotic_error_amplitude"])
        check(np.sqrt(np.mean(residual**2)), row["residual_rms"])
        check(
            (1 - rho) * feedback["tracking_tolerance"],
            row["residual_bound_for_tracking_tolerance"],
        )
    for r in read(a.run / "independent-trial-requirements.json")["rows"]:
        p, n = r["desired_success_probability"], r["zero_failure_trials"]
        assert p**n <= 0.05 and p ** (n - 1) > 0.05
    summary = read(a.run / "summary.json")
    for s in summary:
        report = reports[s["case"]]
        assert s["latest"] == report["latest"]
        for arm, v in s["arms"].items():
            r = report["arms"][arm]
            assert v["final_required_marginal_tolerance"] == {
                k: a[-1] for k, a in r["required_marginal_tolerance"].items()
            }
            assert (
                v["final_minimum_recording_joint_fraction"]
                == r["minimum_recording_joint_fraction"][-1]
            )
            assert v["observed_95pct_horizon_s"] == r["observed_95pct_horizon_s"]
    result = dict(
        cases=len(reports),
        forecast_arms=sum(len(r["arms"]) for r in reports.values()),
        scored_window_arm_pairs=scored,
        numerical_comparisons=comparisons,
        maximum_absolute_difference=maximum,
        original_forecast_hashes_verified=True,
        executed_sources_verified=True,
        scalar_recurrence_replayed_by_convolution=True,
        independent_trial_counts_verified=True,
        limitation="Checks calculations from frozen arrays, not forecast accuracy against independent physical truth or control performance.",
    )
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        default=Path("../artifacts/accuracy-requirements/comparison-02"),
    )
    p.add_argument(
        "--forecasts",
        type=Path,
        default=Path("../artifacts/default-recipe/comparison-01"),
    )
    p.add_argument("--output", type=Path, required=True)
    audit(p.parse_args())
