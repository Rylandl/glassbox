"""Independent reconstruction and summaries for the output/history diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from report_error_calibration import rq
from report_real_transition import gp_replay, model_fingerprint, rotations


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(values):
    return {
        name: float(reducer(values))
        for name, reducer in (("mean", np.mean), ("min", np.min), ("max", np.max))
    }


def verify_windows(saved, role, raw_by_name, case, lags, usage):
    h = case["horizon_steps"]
    ids = saved[f"{role}_ids"]
    assert len(ids) == len(set(ids))
    used = {item["recording"]: item for item in usage}
    assert set(saved[f"{role}_groups"]) == set(used)
    for name, item in used.items():
        mask = saved[f"{role}_groups"] == name
        raw = raw_by_name[name]
        anchors = saved[f"{role}_anchors"][mask]
        pool = np.arange(20, len(raw) - 25)
        if role == "train":
            identity = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(
                np.random.SeedSequence([case["seed"], identity])
            )
            expected = rng.permutation(pool)[: case["budget"] // 6]
        else:
            expected = pool[::10]
        np.testing.assert_array_equal(anchors, expected)
        states, controls = raw[:, [8, 9, 10, 11, 12, 13]], raw[:, [17, 14, 15, 16]]
        edges = np.linspace(0, h, min(10, h) + 1, dtype=int)
        future = np.stack(
            [
                controls[anchors[:, None] + np.arange(edges[i], edges[i + 1])].mean(
                    axis=1
                )
                for i in range(len(edges) - 1)
            ],
            axis=1,
        ).reshape(len(anchors), -1)
        context = [rotations(raw)[anchors], future]
        for lag in lags:
            context.extend(
                (
                    states[anchors - lag] - states[anchors],
                    controls[anchors - lag] - controls[anchors],
                )
            )
        context = np.concatenate(context, axis=1)
        expected_fields = {
            "states": states[anchors],
            "commands": controls[anchors],
            "next_states": states[anchors + h],
            "context": context,
            "features": np.concatenate(
                (states[anchors], controls[anchors], context), axis=1
            ),
            "trend": states[anchors] + h / 5 * (states[anchors] - states[anchors - 5]),
        }
        for key, value in expected_fields.items():
            np.testing.assert_allclose(saved[f"{role}_{key}"][mask], value, atol=1e-9)
        np.testing.assert_array_equal(
            ids[mask], [f"{name}/anchor{a}/h{h}" for a in anchors]
        )
        state_rows = set(anchors) | set(anchors + h)
        control_rows = set((anchors[:, None] + np.arange(h)).ravel())
        for lag in lags:
            state_rows.update(anchors - lag)
            control_rows.update(anchors - lag)
        assert item["windows"] == len(anchors)
        assert item["unique_state_rows"] == len(state_rows)
        assert item["unique_control_intervals"] == len(control_rows)
        np.testing.assert_allclose(
            item["control_interval_union_s"], len(control_rows) * 0.01
        )


def verify_gp(path, saved, output, horizon):
    with np.load(path, allow_pickle=False) as model:
        meta = json.loads(str(model["metadata"]))
        train = saved["train_features"]
        delta = saved["train_next_states"] - saved["train_states"]
        order = (
            list(range(train.shape[1]))
            if output is None
            else [
                output,
                *range(6, 10),
                *[i for i in range(6) if i != output],
                *range(10, train.shape[1]),
            ]
        )
        assert sorted(order) == list(range(train.shape[1]))
        x = train[:, order]
        target = delta if output is None else delta[:, output : output + 1]
        assert meta["kernel"] == "rq" and meta["mean_mode"] == "increment"
        assert meta["state_size"] == target.shape[1]
        assert meta["command_size"] == 4
        assert meta["context_size"] == x.shape[1] - 4 - target.shape[1]
        np.testing.assert_allclose(meta["dt_s"], horizon * 0.01)
        for label, array in (("feature", x), ("target", target)):
            center, scale = array.mean(axis=0), array.std(axis=0)
            scale = np.where(scale > 1e-8, scale, 1)
            np.testing.assert_allclose(model[f"{label}_mean"], center, atol=1e-10)
            np.testing.assert_allclose(model[f"{label}_scale"], scale, atol=1e-10)
        z = (x - model["feature_mean"]) / model["feature_scale"]
        y = (target - model["target_mean"]) / model["target_scale"]
        np.testing.assert_allclose(z, model["features"], atol=1e-10)
        assert meta["fit_report"]["numerical_jitter_standardized"] == 1e-6
        covariance = rq(z, z, model["theta"]) + (
            np.exp(2 * model["theta"][-1]) + 1e-6
        ) * np.eye(len(z))
        np.testing.assert_allclose(
            model["chol"] @ model["chol"].T, covariance, atol=1e-9
        )
        np.testing.assert_allclose(covariance @ model["alpha"], y, atol=1e-7)
        whitened = np.linalg.solve(model["chol"], y)
        nll = (
            0.5 * np.sum(whitened**2)
            + target.shape[1] * np.log(np.diag(model["chol"])).sum()
            + 0.5 * y.size * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(nll, meta["fit_report"]["nll"], atol=1e-6)
        replay = {
            role: gp_replay(model, meta, saved[f"{role}_features"][:, order])
            for role in ("development", "replication")
        }
        return replay, float(nll), len(model["theta"]), model_fingerprint(model, meta)


def verify_metrics(mean, saved, role, metrics):
    for group, values in metrics.items():
        mask = (
            np.ones(len(mean), bool)
            if group == "all"
            else saved[f"{role}_groups"] == group
        )
        error = mean[mask] - saved[f"{role}_next_states"][mask]
        assert values["count"] == int(mask.sum())
        np.testing.assert_allclose(
            values["per_channel_rmse"], np.sqrt(np.mean(error**2, axis=0)), atol=1e-10
        )
        for sl, key in (
            (slice(0, 3), "velocity_rmse_m_s"),
            (slice(3, 6), "rate_rmse_rad_s"),
        ):
            np.testing.assert_allclose(
                values[key],
                np.sqrt(np.mean(np.sum(error[:, sl] ** 2, axis=1))),
                atol=1e-10,
            )


def audit(run, corpus, output):
    plan = json.loads((run / "plan.json").read_text())
    complete = json.loads((run / "summary.json").read_text())
    assert len(complete["reports"]) == complete["cases"] == len(plan["cases"])
    sources = json.loads((run / "sources.json").read_text())
    with zipfile.ZipFile(run / "executed-sources.zip") as archive:
        assert archive.testzip() is None
        for name, digest in sources.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    raw_by_name = {}
    for recording in json.loads((run / "data-audit.json").read_text()):
        path = corpus / "raw" / recording["benchmark"]["relative_path"]
        assert path.parent.name == "train" and not path.stem.startswith("melon")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == recording["sha256"]
        raw_by_name[path.stem] = np.loadtxt(path, delimiter=",", skiprows=1)
    assert len(raw_by_name) == 12
    roles = [
        set(plan["split"][role]) for role in ("train", "development", "replication")
    ]
    assert sum(map(len, roles)) == len(set.union(*roles)) == 12
    reports, seen, identity = [], set(), {}
    max_difference, models = 0.0, 0
    for file in sorted(run.glob("h*-seed*/report.json")):
        folder = file.parent
        report = json.loads(file.read_text())
        reports.append(report)
        case = {
            key: report[key] for key in ("seed", "budget", "history", "horizon_steps")
        }
        assert json.dumps(case, sort_keys=True) not in seen
        seen.add(json.dumps(case, sort_keys=True))
        with (
            np.load(folder / "samples.npz", allow_pickle=False) as saved,
            np.load(folder / "predictions.npz", allow_pickle=False) as predictions,
        ):
            for role in ("train", "development", "replication"):
                assert set(saved[f"{role}_groups"]) == set(plan["split"][role])
                verify_windows(
                    saved,
                    role,
                    raw_by_name,
                    case,
                    plan["histories"][case["history"]],
                    report["usage"][role],
                )
                if role != "train":
                    key = (role, case["horizon_steps"])
                    if key in identity:
                        np.testing.assert_array_equal(
                            saved[f"{role}_ids"], identity[key]
                        )
                    else:
                        identity[key] = saved[f"{role}_ids"].copy()
            for method in ("shared", "shared-long", "outputwise"):
                indices = range(6) if method == "outputwise" else [None]
                results, nll, params, fingerprints = [], 0.0, 0, []
                for i in indices:
                    path = folder / (
                        f"outputwise/output-{i}.npz"
                        if i is not None
                        else f"{method}.npz"
                    )
                    replay, member_nll, member_params, fingerprint = verify_gp(
                        path, saved, i, case["horizon_steps"]
                    )
                    results.append(replay)
                    nll += member_nll
                    params += member_params
                    fingerprints.append(fingerprint)
                    models += 1
                if method == "outputwise":
                    fingerprint = hashlib.sha256(
                        json.dumps(
                            [
                                "glassbox.experimental.outputwise_transition.v1",
                                *fingerprints,
                            ]
                        ).encode()
                    ).hexdigest()
                    manifest = json.loads(
                        (folder / "outputwise/manifest.json").read_text()
                    )
                    assert (
                        manifest["outputs"] == 6
                        and manifest["fingerprint"] == fingerprint
                    )
                else:
                    fingerprint = fingerprints[0]
                arm = report["methods"][method]
                assert (
                    arm["fingerprint"] == fingerprint
                    and arm["parameter_count"] == params
                )
                np.testing.assert_allclose(arm["training_nll_sum"], nll, atol=1e-6)
                steps = plan[
                    {
                        "shared": "shared_steps",
                        "shared-long": "shared_long_steps",
                        "outputwise": "outputwise_steps_per_member",
                    }[method]
                ]
                assert arm["optimizer_updates"] == steps * plan["restarts"] * len(
                    results
                )
                for role in ("development", "replication"):
                    for k, label in enumerate(
                        ("mean", "function_variance", "observation_variance")
                    ):
                        actual = np.concatenate([r[role][k] for r in results], axis=1)
                        expected = predictions[f"{method}_{role}_{label}"]
                        max_difference = max(
                            max_difference, float(abs(actual - expected).max())
                        )
                        np.testing.assert_allclose(actual, expected, atol=1e-8)
            with np.load(folder / "ridge.npz", allow_pickle=False) as ridge:
                x, delta = (
                    saved["train_features"],
                    saved["train_next_states"] - saved["train_states"],
                )
                for label, array in (("feature", x), ("target", delta)):
                    np.testing.assert_allclose(
                        ridge[f"{label}_mean"], array.mean(axis=0), atol=1e-10
                    )
                    std = array.std(axis=0)
                    np.testing.assert_allclose(
                        ridge[f"{label}_scale"],
                        np.where(std > 1e-8, std, 1),
                        atol=1e-10,
                    )
                z = (x - ridge["feature_mean"]) / ridge["feature_scale"]
                y = (delta - ridge["target_mean"]) / ridge["target_scale"]
                np.testing.assert_allclose(
                    (z.T @ z + np.eye(z.shape[1])) @ ridge["weights"],
                    z.T @ y,
                    atol=1e-7,
                )
                for role in ("development", "replication"):
                    mean = (
                        saved[f"{role}_states"]
                        + (
                            (saved[f"{role}_features"] - ridge["feature_mean"])
                            / ridge["feature_scale"]
                            @ ridge["weights"]
                        )
                        * ridge["target_scale"]
                        + ridge["target_mean"]
                    )
                    np.testing.assert_allclose(
                        mean, predictions[f"ridge_{role}_mean"], atol=1e-9
                    )
                    np.testing.assert_array_equal(
                        predictions[f"hold_{role}_mean"], saved[f"{role}_states"]
                    )
                    np.testing.assert_array_equal(
                        predictions[f"trend_{role}_mean"], saved[f"{role}_trend"]
                    )
            for method, arm in report["methods"].items():
                for role, metrics in arm["scores"].items():
                    verify_metrics(
                        predictions[f"{method}_{role}_mean"], saved, role, metrics
                    )
    assert seen == {json.dumps(case, sort_keys=True) for case in plan["cases"]}
    assert {json.dumps(r, sort_keys=True) for r in reports} == {
        json.dumps(r, sort_keys=True) for r in complete["reports"]
    }
    summary = {}
    for report in reports:
        key = f"h{report['horizon_steps']}-{report['history']}-n{report['budget']}"
        if key in summary:
            continue
        selected = [
            r
            for r in reports
            if (r["horizon_steps"], r["history"], r["budget"])
            == (report["horizon_steps"], report["history"], report["budget"])
        ]
        summary[key] = {
            "horizon_steps": report["horizon_steps"],
            "history": report["history"],
            "budget": report["budget"],
            "methods": {},
        }
        for method in report["methods"]:
            arm = {
                "scores": {
                    role: {
                        group: {
                            metric: stats(
                                [
                                    r["methods"][method]["scores"][role][group][metric]
                                    for r in selected
                                ]
                            )
                            for metric in ("velocity_rmse_m_s", "rate_rmse_rad_s")
                        }
                        for group in report["methods"][method]["scores"][role]
                    }
                    for role in ("development", "replication")
                }
            }
            for metric in (
                "training_nll_sum",
                "parameter_count",
                "optimizer_updates",
                "fit_seconds_including_compile",
            ):
                if metric in report["methods"][method]:
                    arm[metric] = stats(
                        [r["methods"][method][metric] for r in selected]
                    )
            arm["per_channel_rmse"] = {
                role: {
                    name: stats(
                        [
                            r["methods"][method]["scores"][role]["all"][
                                "per_channel_rmse"
                            ][i]
                            for r in selected
                        ]
                    )
                    for i, name in enumerate(plan["outputs"])
                }
                for role in ("development", "replication")
            }
            summary[key]["methods"][method] = arm
    write_json(output / "summary.json", summary)
    contrasts = {
        "scope": "Descriptive paired comparisons over correlated planned configurations, not independent statistical trials. Positive error change means worse prediction.",
        "cases": len(reports),
        "outputwise_lower_training_nll": {
            baseline: sum(
                r["methods"]["outputwise"]["training_nll_sum"]
                < r["methods"][baseline]["training_nll_sum"]
                for r in reports
            )
            for baseline in ("shared", "shared-long")
        },
        "error_change_against_shared": {},
    }
    for role in ("development", "replication"):
        contrasts["error_change_against_shared"][role] = {}
        for method in ("shared-long", "outputwise"):
            values = {}
            for metric in ("velocity_rmse_m_s", "rate_rmse_rad_s"):
                changes = np.asarray(
                    [
                        100
                        * (
                            r["methods"][method]["scores"][role]["all"][metric]
                            / r["methods"]["shared"]["scores"][role]["all"][metric]
                            - 1
                        )
                        for r in reports
                    ]
                )
                values[metric] = {
                    "percent_change": stats(changes),
                    "cases_better": int(np.sum(changes < 0)),
                    "cases_worse": int(np.sum(changes > 0)),
                    "max_absolute_percent_change": float(abs(changes).max()),
                }
            contrasts["error_change_against_shared"][role][method] = values
    write_json(output / "comparisons.json", contrasts)
    write_json(
        output / "audit.json",
        {
            "status": "passed",
            "recordings": len(raw_by_name),
            "cases": len(reports),
            "gp_components": models,
            "ridge_models": len(reports),
            "maximum_numpy_replay_difference": max_difference,
            "checks": [
                "source SHA256/ZIP integrity and pinned raw checksums",
                "complete planned cases and disjoint whole-flight roles",
                "every feature/target and supporting-row count reconstructed",
                "nested training draws and identical evaluation origins",
                "bijective input coordinates for every scalar output",
                "training-only normalizers, covariance factors, alpha and marginal likelihood",
                "independent NumPy means, variances, ridge predictions and all reported errors",
                "composite fingerprints and parameter/update counts",
            ],
            "limitations": "No optimizer-gradient replay or physical-memory causal attribution. Reused non-Melon recordings and correlated windows; three seeds are not new flights.",
        },
    )
    return plan, summary


def render(output, plan, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    methods = {
        "shared": ("Shared / 160 steps", "#cf903a", "o"),
        "shared-long": ("Shared / 480 steps", "#956c99", "s"),
        "outputwise": ("Per output / 160 steps each", "#188e8a", "o"),
        "ridge": ("Linear change", "#31445d", "^"),
    }
    histories = list(plan["histories"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.17, top=0.89, wspace=0.28, hspace=0.42
    )
    for col, h in enumerate(plan["horizons"]):
        for row, metric in enumerate(("velocity_rmse_m_s", "rate_rmse_rad_s")):
            ax = axes[row, col]
            for method, (label, color, marker) in methods.items():
                values = [
                    summary[f"h{h}-{history}-n384"]["methods"][method]["scores"][
                        "replication"
                    ]["all"][metric]
                    for history in histories
                ]
                ax.errorbar(
                    range(len(histories)),
                    [v["mean"] for v in values],
                    yerr=[
                        [v["mean"] - v["min"] for v in values],
                        [v["max"] - v["mean"] for v in values],
                    ],
                    color=color,
                    marker=marker,
                    label=label,
                    capsize=3,
                )
            hold = summary[f"h{h}-none-n384"]["methods"]["hold"]["scores"][
                "replication"
            ]["all"][metric]["mean"]
            ax.axhline(hold, color="#9ba5b0", ls=":", label="Hold current value")
            ax.set(
                xticks=range(len(histories)),
                xticklabels=histories,
                xlabel="Available history",
                ylabel="Velocity vector RMSE [m/s]"
                if row == 0
                else "Body-rate vector RMSE [rad/s]",
                title=f"{h * 10} ms forecast",
                ylim=(0, None),
            )
            ax.grid(alpha=0.15)
    fig.suptitle("Which temporal information and model structure help?", fontsize=16)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.018,
        "384 mean-fit windows · run-4 evaluation on three known flights · processed data and supplied future motor speeds · whiskers: three window-draw ranges",
        ha="center",
        fontsize=7.5,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"diagnosis-history.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.27, top=0.83, wspace=0.30)
    for col, metric in enumerate(("velocity_rmse_m_s", "rate_rmse_rad_s")):
        for method, (label, color, _) in methods.items():
            for budget, style, marker in ((192, "--", "s"), (384, "-", "o")):
                values = [
                    summary[f"h{h}-100ms-n{budget}"]["methods"][method]["scores"][
                        "replication"
                    ]["all"][metric]["mean"]
                    for h in plan["horizons"]
                ]
                axes[col].plot(
                    np.asarray(plan["horizons"]) * 0.01,
                    values,
                    color=color,
                    ls=style,
                    marker=marker,
                    label=f"{label} / {budget}",
                )
        axes[col].set(
            xscale="log",
            yscale="log",
            xlabel="Forecast horizon [s]",
            ylabel="Velocity vector RMSE [m/s]"
            if col == 0
            else "Body-rate vector RMSE [rad/s]",
            xticks=np.asarray(plan["horizons"]) * 0.01,
            xticklabels=[str(h * 0.01) for h in plan["horizons"]],
        )
        axes[col].grid(alpha=0.15)
    fig.suptitle("Does doubling the observation budget close the gap?", fontsize=15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        fontsize=7.5,
    )
    fig.text(
        0.5,
        0.015,
        "100 ms history · same six training flights · dashed: 192 windows, solid: 384 · additional windows are nested within each flight",
        ha="center",
        fontsize=8,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"diagnosis-budget.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan, summary = audit(args.run, args.corpus, args.output)
    if not plan["smoke"]:
        render(args.output, plan, summary)
    sources = [
        Path(__file__).resolve(),
        *[
            Path(__file__).resolve().with_name(name)
            for name in ("report_real_transition.py", "report_error_calibration.py")
        ],
    ]
    for path in sources:
        shutil.copyfile(path, args.output / path.name)
    write_json(
        args.output / "renderer-sources.json",
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    print((args.output / "audit.json").read_text())
