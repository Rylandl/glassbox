"""End-to-end smoke tests for the ``glassbox-fit`` console script.

These drive ``fitting.main`` exactly as the README does so that argparse-layer
and serialization failures, which library-level tests cannot see, are caught.
"""

from __future__ import annotations

import json
import sys

from glassbox import DynamicsBelief, LocalParameterInformation
from glassbox.core.data import save_trajectory_npz
from glassbox.core.synthetic import generate_trajectory
from glassbox.workflows import fitting


def _write_flights(tmp_path, count: int = 3) -> list[str]:
    paths = []
    for seed in range(count):
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(generate_trajectory(seed=seed, duration_s=0.4), path)
        paths.append(str(path))
    return paths


def test_fit_cli_writes_belief_and_report_together(tmp_path, monkeypatch) -> None:
    paths = _write_flights(tmp_path)
    model_path = tmp_path / "belief.json"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glassbox-fit",
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
        ],
    )

    fitting.main()

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
