"""Run a fitted artifact against live PX4 telemetry without transmitting.

This leaf is one :func:`~glassbox.integrations.loop.run_control_loop` over a
read-only :class:`~glassbox.integrations.px4.PX4MavlinkLink`. It writes one
compact JSON object per interval, carrying the state that was read, the command
that was solved for, the solver's status and timing, and the plan's
diagnostics, followed by one summary object recording status counts, solve-time
median and p90, and the telemetry skew the run saw. Nothing is transmitted to
the vehicle: the link is not writable, so the loop never calls its writer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np

from glassbox.control.fitted import NMPCController
from glassbox.control.plan import ReferenceTrajectory
from glassbox.core.model import ExecutableModel
from glassbox.integrations.loop import (
    LoopSample,
    LoopSummary,
    Observation,
    run_control_loop,
)
from glassbox.integrations.px4 import PX4MavlinkLink, PX4MavlinkStateSource

_MAXIMUM_APPLIED_COMMAND_STATE_SKEW_S = 0.10


def _command(value: str, *, expected_size: int) -> np.ndarray:
    try:
        command = np.asarray([float(item) for item in value.split(",")])
    except ValueError as error:
        raise ValueError("previous command must be comma-separated numbers") from error
    if command.shape != (expected_size,) or not np.all(np.isfinite(command)):
        raise ValueError(f"previous command must contain {expected_size} finite values")
    return command


def px4_shadow_link(
    state_source: PX4MavlinkStateSource,
    model: ExecutableModel,
    *,
    previous_command: np.ndarray | None = None,
    applied_command_source: object | None = None,
) -> PX4MavlinkLink:
    """Build the read-only link one artifact is shadowed against.

    The alignment limit is the tighter of the module's fixed limit and the
    artifact's own sample period: a command further from its state than one
    control interval is not the command that produced it.
    """

    return PX4MavlinkLink(
        state_source,
        command_size=model.command_size,
        command_bounds=(model.command_minimum, model.command_maximum),
        applied_command_source=applied_command_source,
        fixed_command=previous_command,
        maximum_applied_command_state_skew_s=min(
            _MAXIMUM_APPLIED_COMMAND_STATE_SKEW_S,
            model.runtime_spec.sample_period_s,
        ),
    )


def run_px4_nmpc_shadow(
    link: PX4MavlinkLink,
    controller: NMPCController,
    *,
    steps: int = 10,
    write_line: Callable[[str], None] | None = None,
) -> LoopSummary:
    """Solve against live PX4 telemetry, holding the measured state."""

    exogenous = np.zeros(controller.model.exogenous_size, dtype=np.float64)

    def reference(observation: Observation) -> ReferenceTrajectory:
        return controller.hold_reference(observation.state, exogenous=exogenous)

    def on_sample(sample: LoopSample) -> None:
        assert write_line is not None
        write_line(json.dumps(sample.to_dict(), sort_keys=True, allow_nan=False))

    return run_control_loop(
        link,
        controller,
        steps=steps,
        reference=reference,
        on_sample=None if write_line is None else on_sample,
    )


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fitted Glassbox model against passive PX4 MAVLink telemetry. "
            "No command messages are transmitted."
        )
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
        "--output",
        type=Path,
        help="file to write one JSON interval record per line to",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    model = ExecutableModel.load(args.model)
    previous_command = _command(args.previous_command, expected_size=model.command_size)
    controller = NMPCController(model)
    with PX4MavlinkStateSource.connect(args.connection) as state_source:
        link = px4_shadow_link(state_source, model, previous_command=previous_command)
        with _line_writer(args.output) as write_line:
            summary = run_px4_nmpc_shadow(
                link,
                controller,
                steps=args.samples,
                write_line=write_line,
            )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    if args.output is not None:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
