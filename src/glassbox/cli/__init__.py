"""Fit and evaluate the single rigid-body dynamics learner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from glassbox import LearnedDynamics, fit
from glassbox.io.recordings import concatenate_recordings, load_recordings
from glassbox.workflows.forecast import evaluate


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="glassbox", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit", help="fit a model from recording archives")
    fitting.add_argument("recordings", type=Path, nargs="+")
    fitting.add_argument("--model", type=Path, required=True, help="output model NPZ")
    fitting.add_argument("--report", type=Path, help="output fit report JSON")
    scoring = commands.add_parser("evaluate", help="score unseen recording archives")
    scoring.add_argument("model", type=Path)
    scoring.add_argument("recordings", type=Path, nargs="+")
    scoring.add_argument(
        "--report", type=Path, help="output report JSON; default stdout"
    )
    args = parser.parse_args(argv)
    try:
        recordings = concatenate_recordings(
            [load_recordings(path) for path in args.recordings]
        )
        if args.command == "fit":
            model = fit(recordings)
            args.model.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.model)
            report = model.report
            print(f"wrote model {args.model}")
        else:
            report = evaluate(LearnedDynamics.load(args.model), recordings)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text)
            print(f"wrote report {args.report}")
        elif args.command == "evaluate":
            print(text, end="")
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
