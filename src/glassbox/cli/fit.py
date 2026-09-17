"""Fit the single generic dynamics recipe from recording archives."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from glassbox import fit
from glassbox.io.recordings import concatenate_recordings, load_recordings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recordings", type=Path, nargs="+", help="generic recording NPZ archives"
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="output generic model NPZ"
    )
    parser.add_argument("--report", type=Path, help="output fit report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        recordings = concatenate_recordings(
            [load_recordings(path) for path in args.recordings]
        )
        model = fit(recordings)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with args.model.open("wb") as handle:
        model.save(handle)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(model.report, indent=2, allow_nan=False) + "\n"
        )
    print(f"wrote generic model {args.model}")
    if args.report is not None:
        print(f"wrote report {args.report}")


if __name__ == "__main__":
    main()
