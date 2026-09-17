"""Evaluate saved generic forecasts on previously unseen recording archives."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from glassbox import LearnedDynamics
from glassbox.io.recordings import concatenate_recordings, load_recordings
from glassbox.workflows.forecast import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="saved generic model NPZ")
    parser.add_argument(
        "recordings",
        type=Path,
        nargs="+",
        help="held-out generic recording NPZ archives",
    )
    parser.add_argument("--report", type=Path, help="output forecast report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        model = LearnedDynamics.load(args.model)
        recordings = concatenate_recordings(
            [load_recordings(path) for path in args.recordings]
        )
        report = evaluate(model, recordings)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report is None:
        print(text, end="")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
        print(f"wrote forecast report {args.report}")


if __name__ == "__main__":
    main()
