"""End-to-end smoke tests for the ``glassbox fit`` subcommand.

These drive ``glassbox.cli.main`` exactly as the README does so that
argparse-layer and serialization failures, which library-level tests cannot
see, are caught.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from glassbox import DynamicsBelief, LocalParameterInformation, cli
from glassbox.core.data import save_trajectory_npz
from glassbox.core.synthetic import generate_trajectory


def _write_flights(tmp_path, count: int = 3) -> list[str]:
    paths = []
    for seed in range(count):
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(generate_trajectory(seed=seed, duration_s=0.4), path)
        paths.append(str(path))
    return paths


def _write_benchmark_split_flights(tmp_path, splits) -> list[str]:
    paths = []
    for seed, split in enumerate(splits):
        path = tmp_path / f"split_flight_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.4)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "benchmark_split": split},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(str(path))
    return paths


def test_fit_cli_writes_belief_and_report_together(tmp_path) -> None:
    paths = _write_flights(tmp_path)
    model_path = tmp_path / "belief.json"
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
            "--model",
            str(model_path),
            "--report",
            str(report_path),
        ]
    )

    belief = DynamicsBelief.load(model_path)
    assert isinstance(belief.parameter_evidence, LocalParameterInformation)
    assert (tmp_path / "belief_no_motor_lag.json").exists()
    report = json.loads(report_path.read_text())
    evidence = report["models"]["learned_lag"]["parameter_evidence"]
    assert evidence["kind"] == "local_structured_parameter_information"
    assert (
        evidence["rank_relative_tolerance"]
        == belief.parameter_evidence.rank_relative_tolerance
    )


def test_fit_cli_rejects_holdout_count_when_benchmark_split_labels_are_present(
    tmp_path, capsys
) -> None:
    paths = _write_benchmark_split_flights(tmp_path, ("training", "validation"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["fit", *paths, "--holdout-count", "2", "--steps", "1"])

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "benchmark_split" in stderr
    assert stderr.startswith("usage: glassbox fit")
