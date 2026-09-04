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

import json
from collections.abc import Callable

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
