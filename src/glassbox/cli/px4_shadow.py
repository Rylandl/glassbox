"""Run a fitted model against live PX4 telemetry without ever transmitting.

The shadow reads state and the applied command from a passive MAVLink
connection, solves the NMPC problem the vehicle's own controller is solving,
and records what it would have commanded. The link is read-only: no command
message is ever sent, so the shadow can run beside a real flight.

Each interval is written as one JSON record, to ``--output`` or to standard
output, and the run ends with the summary of solve statuses, bounded holds and
interval timings.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController, default_solver_policy
from glassbox.integrations.px4 import PX4MavlinkStateSource
from glassbox.integrations.px4_nmpc_shadow import px4_shadow_link, run_px4_nmpc_shadow


def _command(value: str, *, expected_size: int) -> np.ndarray:
    parts = [item.strip() for item in value.split(",")]
    try:
        command = np.asarray([float(item) for item in parts], dtype=np.float64)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "the applied command must be comma-separated numbers"
        ) from error
    if command.size != expected_size:
        raise argparse.ArgumentTypeError(
            f"the model expects {expected_size} command channels, got {command.size}"
        )
    return command


@contextlib.contextmanager
def _line_writer(output: Path | None) -> Iterator[Callable[[str], None]]:
    """Write interval lines to ``output``, or to standard output without one."""

    if output is None:
        yield print
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:

        def write_line(line: str) -> None:
            handle.write(line + "\n")

        yield write_line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--previous-command",
        required=True,
        help="comma-separated command currently applied to the vehicle",
    )
    parser.add_argument(
        "--connection",
        default="udpin:0.0.0.0:14550",
        help="passive pymavlink connection string",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--telemetry-timeout-s",
        type=float,
        default=1.0,
        help="maximum wait for fresh telemetry; does not extend the solve deadline",
    )
    parser.add_argument(
        "--allow-unresolved-parameters",
        action="store_true",
        help="explicitly permit shadow plans with incomplete parameter uncertainty",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="file to write one JSON interval record per line to",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    belief = DynamicsBelief.load(args.model)
    model = belief.model
    previous_command = _command(args.previous_command, expected_size=model.command_size)
    controller = NMPCController(
        belief,
        policy=replace(
            default_solver_policy(belief),
            allow_unresolved_parameters=args.allow_unresolved_parameters,
        ),
    )
    with PX4MavlinkStateSource.connect(args.connection) as state_source:
        link = px4_shadow_link(state_source, model, previous_command=previous_command)
        with _line_writer(args.output) as write_line:
            summary = run_px4_nmpc_shadow(
                link,
                controller,
                steps=args.samples,
                write_line=write_line,
                telemetry_timeout_s=args.telemetry_timeout_s,
            )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    if args.output is not None:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
