"""End-to-end smoke tests for the ``glassbox fit`` subcommand.

These drive ``glassbox.cli.main`` exactly as the README does so that
argparse-layer and serialization failures, which library-level tests cannot
see, are caught.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from glassbox import DynamicsBelief, cli
from glassbox.core.data import load_trajectory_npz, save_trajectory_npz


def _write_flights(tmp_path, quadrotor_flight, count: int = 3) -> list[str]:
    paths = []
    for seed in range(count):
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(quadrotor_flight(seed), path)
        paths.append(str(path))
    return paths


def _write_benchmark_split_flights(tmp_path, quadrotor_flight, splits) -> list[str]:
    paths = []
    for seed, split in enumerate(splits):
        path = tmp_path / f"split_flight_{seed}.npz"
        trajectory = quadrotor_flight(seed)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "benchmark_split": split},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("family", ["multirotor", "fixedwing"])
def test_fit_cli_writes_belief_and_report_together(
    tmp_path, quadrotor_flight, fixedwing_flight, family
) -> None:
    generate = quadrotor_flight if family == "multirotor" else fixedwing_flight
    paths = _write_flights(tmp_path, generate)
    model_path = tmp_path / "belief.json"
    report_path = tmp_path / "report.json"

    cli.main(
        [
            "fit",
            *paths[:2],
            "--horizon",
            "5",
            "--steps",
            "1",
            "--evaluation-horizons",
            "0.1",
            "--model",
            str(model_path),
            "--report",
            str(report_path),
        ]
    )

    belief = DynamicsBelief.load(model_path)
    assert belief.information.resolved_rank() > 0
    assert belief.provenance["fit_report"] == str(report_path)
    # The no-lag ablation is opt-in, so it is not written unless asked for.
    assert not (tmp_path / "belief_no_motor_lag.json").exists()
    assert "no_lag" not in json.loads(report_path.read_text())["models"]
    report = json.loads(report_path.read_text())
    evidence = report["models"]["learned_lag"]["parameter_evidence"]
    assert evidence["kind"] == "structured_parameter_information"
    assert evidence["resolved_rank"] == belief.information.resolved_rank()
    assert (
        evidence["rank_relative_tolerance"]
        == belief.information.rank_relative_tolerance
    )
    # The noise model is measured whether or not the caller asked for the
    # precision, and the floor it can never fall below travels beside it.
    assert evidence["innovation_noise"] == belief.information.innovation_noise.tolist()
    assert evidence["noise_floor"] == belief.information.noise_floor.tolist()
    assert evidence["effective_count"] > 0.0
    # The innovation diagnostics are opt-in, so the report carries no block.
    validation = report["models"]["learned_lag"]["validation"]
    assert report["configuration"]["diagnostics"] is False
    assert "one_step_innovation" not in validation["aggregate"]
    assert "one_step_innovation" not in validation["per_flight"][0]

    evaluation_path = tmp_path / "evaluation.json"
    cli.main(
        [
            "evaluate",
            str(model_path),
            paths[2],
            "--horizons",
            "0.1",
            "--report",
            str(evaluation_path),
        ]
    )
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation["independent_holdout"] is True
    assert np.isfinite(evaluation["score_vs_baseline"])

    updated, _ = belief.absorb(load_trajectory_npz(paths[2]))
    updated_path = tmp_path / "updated.json"
    updated.save(updated_path)
    restored = DynamicsBelief.load(updated_path)
    assert restored.information.effective_count > belief.information.effective_count
    np.testing.assert_array_equal(
        restored.information.precision, updated.information.precision
    )
    assert np.isfinite(restored.information.precision).all()


def test_fit_cli_records_innovation_diagnostics_when_asked(
    tmp_path, quadrotor_flight
) -> None:
    paths = _write_flights(tmp_path, quadrotor_flight)
    report_path = tmp_path / "report.json"

    cli.main(
        [
            "fit",
            *paths,
            "--horizon",
            "5",
            "--steps",
            "1",
            "--evaluation-horizons",
            "0.1",
            "--diagnostics",
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text())
    validation = report["models"]["learned_lag"]["validation"]
    assert report["configuration"]["diagnostics"] is True
    assert validation["aggregate"]["one_step_innovation"]["status"] == "ok"
    assert validation["per_flight"][0]["one_step_innovation"]["policy"] == (
        "measured_state_reset_innovation_v1"
    )


def test_fit_cli_rejects_two_holdout_rules_at_once(
    tmp_path, capsys, quadrotor_flight
) -> None:
    paths = _write_benchmark_split_flights(
        tmp_path, quadrotor_flight, ("training", "validation")
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "fit",
                *paths,
                "--holdout-label",
                "benchmark_split=validation",
                "--holdout-count",
                "1",
                "--steps",
                "1",
            ]
        )

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "select different holdouts" in stderr
    assert stderr.startswith("usage: glassbox fit")


def test_fit_cli_reserves_flights_by_label(tmp_path, quadrotor_flight) -> None:
    paths = _write_benchmark_split_flights(
        tmp_path, quadrotor_flight, ("training", "training", "validation")
    )
    report_path = tmp_path / "report.json"

    cli.main(
        [
            "fit",
            *paths,
            "--holdout-label",
            "benchmark_split=validation",
            "--horizon",
            "5",
            "--steps",
            "1",
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text())
    assert report["split"]["mode"] == "leave_labeled_out"
    assert [item["path"] for item in report["split"]["validation_flights"]] == [
        paths[-1]
    ]


def test_fit_cli_writes_the_no_lag_ablation_when_asked(
    tmp_path, quadrotor_flight
) -> None:
    paths = _write_flights(tmp_path, quadrotor_flight)
    model_path = tmp_path / "belief.json"
    report_path = tmp_path / "report.json"

    cli.main(
        [
            "fit",
            *paths,
            "--ablation",
            "no-lag",
            "--horizon",
            "5",
            "--steps",
            "1",
            "--evaluation-horizons",
            "0.1",
            "--model",
            str(model_path),
            "--report",
            str(report_path),
        ]
    )

    ablation_path = tmp_path / "belief_no_motor_lag.json"
    assert ablation_path.exists()
    ablation = DynamicsBelief.load(ablation_path)
    assert ablation.provenance["ablation"] == (
        "fixed near-zero applied-control response"
    )
    report = json.loads(report_path.read_text())
    assert report["configuration"]["ablations"] == ["no_lag"]
    assert report["comparison"]["aggregate_full_rollout"]["position_rmse_m"] > 0.0
