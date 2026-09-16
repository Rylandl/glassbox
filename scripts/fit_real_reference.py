"""Fit the existing structured reference on the declared real-data mean flights.

This model has platform priors and a larger full-state supervision budget than
the generic forecast experiments. It supplies performance context only.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from glassbox import FitSpec, Holdout, fit
from glassbox.core.data import load_trajectory_npz
from glassbox.workflows.evaluate import save_report


def run(corpus, output):
    output.mkdir(parents=True, exist_ok=False)
    flights = []
    for path in sorted((corpus / "canonical/train").glob("*.npz")):
        flight = load_trajectory_npz(path)
        if flight.labels["replicate"] == 4:
            continue
        role = "scale" if flight.labels["replicate"] == 3 else "train"
        flights.append(
            replace(
                flight,
                labels={**flight.labels, "source_group": path.stem, "data_role": role},
                provenance={**flight.provenance, "path": str(path.resolve())},
            )
        )
    result = fit(
        flights,
        FitSpec(
            holdout=Holdout.by_label("data_role", ["scale"]),
            horizons_s=(0.1, 0.25, 0.5),
            evaluation_horizons_s=(0.1, 0.25, 0.5),
            steps=80,
            stride=10,
            parameter_evidence=False,
        ),
    )
    result.belief.save(output / "belief.json")
    save_report(result.report, output / "fit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.corpus, args.output)
