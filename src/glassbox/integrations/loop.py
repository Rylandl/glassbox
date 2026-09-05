"""One control interval, and the link contract every vehicle satisfies.

A vehicle is a :class:`VehicleLink`: it reads out an :class:`Observation` and,
if it is writable, accepts a bounded command. PX4 telemetry is a read-only
link, a simulated plant is a writable one, and neither knows anything about the
controller driving it. :func:`run_control_loop` is the interval both share:
read the vehicle, solve from the previous interval's warm start with the
interval as the deadline, supervise the candidate when a supervisor is given,
write it when the link accepts writes, and hand the whole record to a callback.

The loop does not assemble a report. It returns a :class:`LoopSummary` of what
happened, and every interval's detail reaches the caller through ``on_sample``
as it happens, so a long run costs the caller whatever the caller chooses to
keep rather than a growing document.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from glassbox.control.plan import (
    NMPCDiagnostics,
    NMPCWarmStart,
    ReferenceTrajectory,
    SolveResult,
)

Array = Any


def _finite_or_none(value: float) -> float | None:
    """Return a float, or ``None`` when it is not finite and cannot be written."""

    result = float(value)
    return result if np.isfinite(result) else None


def _nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class Observation:
    """One aligned state and applied-command sample read from a vehicle.

    ``state`` is the canonical rigid-body 13-vector and ``applied_command`` is
    what the vehicle was actually driving when that state was measured.
    ``received_at_s`` is the host monotonic clock at reception, which is the
    clock the supervisor's freshness rules are written against, while
    ``source_time_s`` is the vehicle's own clock when the link can report one.

    The remaining fields are the alignment diagnostics a link measures while
    assembling the sample: how far apart the messages it paired were, how old
    the oldest of them was, how far the source clock has fallen behind, and how
    far the applied command sits from the state it is paired with. A loop
    summarizes them; it does not interpret them.
    """

    state: np.ndarray
    applied_command: np.ndarray
    received_at_s: float
    source_time_s: float | None = None
    message_skew_s: float = 0.0
    receive_age_s: float = 0.0
    source_clock_lag_s: float = 0.0
    applied_command_skew_s: float | None = None
    armed: bool | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float64).copy()
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            raise ValueError("observed state must contain 13 finite values")
        command = np.asarray(self.applied_command, dtype=np.float64).copy()
        if command.ndim != 1 or command.size < 1 or not np.all(np.isfinite(command)):
            raise ValueError("applied command must be a nonempty finite vector")
        if not np.isfinite(self.received_at_s):
            raise ValueError("received_at_s must be finite")
        if self.source_time_s is not None and not np.isfinite(self.source_time_s):
            raise ValueError("source_time_s must be finite when it is reported")
        _nonnegative("message_skew_s", self.message_skew_s)
        _nonnegative("receive_age_s", self.receive_age_s)
        _nonnegative("source_clock_lag_s", self.source_clock_lag_s)
        if self.applied_command_skew_s is not None:
            _nonnegative("applied_command_skew_s", self.applied_command_skew_s)
        state.flags.writeable = False
        command.flags.writeable = False
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "applied_command", command)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.tolist(),
            "applied_command": self.applied_command.tolist(),
            "received_at_s": float(self.received_at_s),
            "source_time_s": self.source_time_s,
            "message_skew_s": float(self.message_skew_s),
            "receive_age_s": float(self.receive_age_s),
            "source_clock_lag_s": float(self.source_clock_lag_s),
            "applied_command_skew_s": self.applied_command_skew_s,
            "armed": self.armed,
        }


@runtime_checkable
class VehicleLink(Protocol):
    """A vehicle a control loop can read from, and sometimes write to.

    ``writable`` is the whole difference between shadow mode and closed-loop
    control, and it is a property of the link rather than a flag on the loop.
    A read-only link raises from :meth:`write` rather than silently accepting a
    command it will never transmit.
    """

    command_size: int
    command_bounds: tuple[Array, Array]
    writable: bool

    def read(self, *, timeout_s: float) -> Observation:
        """Return the next aligned state and applied-command sample."""

    def write(self, command: Array) -> None:
        """Apply one bounded command, or raise when the link is read-only."""


class Controller(Protocol):
    """What a control loop needs from whatever produces its commands."""

    sample_period_s: float

    def solve(
        self,
        state: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        *,
        applied_command: Array | None = ...,
        latent_state: Array | None = ...,
        warm_start: NMPCWarmStart | None = ...,
        deadline_s: float | None = ...,
    ) -> SolveResult:
        """Optimize one bounded command, returning a hold on every failure."""


class CommandSupervisor(Protocol):
    """An independent boundary that may replace the controller's command."""

    def supervise(
        self,
        *,
        state: Any,
        state_received_at_s: float,
        candidate_command: Any,
        command_generated_at_s: float,
        now_s: float,
        controller_command_usable: bool,
        previous_applied_command: Any,
    ) -> Any:
        """Select the command that is actually allowed to reach the vehicle."""


Reference = ReferenceTrajectory | Callable[[Observation], ReferenceTrajectory]


def _diagnostics_dict(diagnostics: NMPCDiagnostics) -> dict[str, Any]:
    """Report one solve's measurements, with unbounded margins as ``None``."""

    return {
        "final_projected_gradient_inf_norm": _finite_or_none(
            diagnostics.final_projected_gradient_inf_norm
        ),
        "initial_objective": _finite_or_none(diagnostics.initial_objective),
        "final_objective": _finite_or_none(diagnostics.final_objective),
        "maximum_command_bound_violation": float(
            diagnostics.maximum_command_bound_violation
        ),
        "maximum_validity_utilization": _finite_or_none(
            diagnostics.maximum_validity_utilization
        ),
        "maximum_normalized_safety_violation": _finite_or_none(
            diagnostics.maximum_normalized_safety_violation
        ),
        "maximum_normalized_model_uncertainty_standard_deviation": _finite_or_none(
            diagnostics.maximum_normalized_model_uncertainty_standard_deviation
        ),
        "warm_start_used": bool(diagnostics.warm_start_used),
        "prediction_horizon_s": float(diagnostics.prediction_horizon_s),
        "parameter_uncertainty_complete": diagnostics.parameter_uncertainty_complete,
        "unresolved_parameters_allowed": diagnostics.unresolved_parameters_allowed,
    }


@dataclass(frozen=True)
class LoopSample:
    """One control interval: what was read, solved, supervised, and written."""

    step: int
    observation: Observation
    result: SolveResult
    command: np.ndarray
    written: bool
    decision: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the compact record one interval contributes to a log."""

        return {
            "step": self.step,
            **self.observation.to_dict(),
            "status": self.result.status.value,
            "command_usable": self.result.command_usable,
            "used_fallback": self.result.used_fallback,
            "command": np.asarray(self.command).tolist(),
            "written": self.written,
            "solve_time_s": float(self.result.diagnostics.solve_time_s),
            "iterations": int(self.result.diagnostics.iterations),
            "diagnostics": _diagnostics_dict(self.result.diagnostics),
            "supervisor": (None if self.decision is None else self.decision.to_dict()),
        }


@dataclass(frozen=True)
class LoopSummary:
    """What one control loop did, over every interval it ran."""

    steps: int
    interval_s: float
    status_counts: dict[str, int]
    usable_command_count: int
    fallback_count: int
    written_command_count: int
    supervisor_intervention_count: int
    deadline_miss_count: int
    solve_time_median_s: float
    solve_time_p90_s: float
    solve_time_maximum_s: float
    maximum_message_skew_s: float
    maximum_receive_age_s: float
    maximum_source_clock_lag_s: float
    maximum_applied_command_skew_s: float | None
    host_elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "interval_s": self.interval_s,
            "status_counts": dict(self.status_counts),
            "usable_command_count": self.usable_command_count,
            "fallback_count": self.fallback_count,
            "written_command_count": self.written_command_count,
            "supervisor_intervention_count": self.supervisor_intervention_count,
            "deadline_miss_count": self.deadline_miss_count,
            "solve_time_median_s": self.solve_time_median_s,
            "solve_time_p90_s": self.solve_time_p90_s,
            "solve_time_maximum_s": self.solve_time_maximum_s,
            "maximum_message_skew_s": self.maximum_message_skew_s,
            "maximum_receive_age_s": self.maximum_receive_age_s,
            "maximum_source_clock_lag_s": self.maximum_source_clock_lag_s,
            "maximum_applied_command_skew_s": self.maximum_applied_command_skew_s,
            "host_elapsed_s": self.host_elapsed_s,
        }


def run_control_loop(
    link: VehicleLink,
    controller: Controller,
    supervisor: CommandSupervisor | None = None,
    *,
    steps: int,
    reference: Reference,
    on_sample: Callable[[LoopSample], None] | None = None,
) -> LoopSummary:
    """Run ``steps`` control intervals over one link and summarize them.

    The interval is the controller's sample period: it is the read timeout, the
    solver deadline, and what a solve time is compared against. ``reference`` is
    either one fixed trajectory or a callable handed each observation, which is
    what a regulator holding the measured state needs.

    A failed solve never stops the loop. The solver returns a bounded hold with
    an explicit failure status on every failure path, so the loop records the
    status, supervises and writes the hold like any other command, and goes on
    to the next interval. Only the link itself can end a run early, by raising
    when it can no longer deliver telemetry.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    interval_s = float(controller.sample_period_s)
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("the controller's sample period must be finite and positive")
    resolve = (
        reference if callable(reference) else (lambda _observation: reference)  # type: ignore[misc, return-value]
    )

    status_counts: dict[str, int] = {}
    solve_times_s: list[float] = []
    usable_command_count = 0
    fallback_count = 0
    written_command_count = 0
    supervisor_intervention_count = 0
    message_skews_s: list[float] = []
    receive_ages_s: list[float] = []
    source_clock_lags_s: list[float] = []
    applied_command_skews_s: list[float] = []
    warm_start: NMPCWarmStart | None = None
    started_at_s = time.monotonic()

    for step in range(steps):
        observation = link.read(timeout_s=interval_s)
        result = controller.solve(
            observation.state,
            resolve(observation),
            observation.applied_command,
            applied_command=observation.applied_command,
            warm_start=warm_start,
            deadline_s=interval_s,
        )
        solved_at_s = time.monotonic()
        warm_start = result.warm_start
        command = np.asarray(result.command, dtype=np.float64)

        decision = None
        if supervisor is not None:
            decision = supervisor.supervise(
                state=observation.state,
                state_received_at_s=observation.received_at_s,
                candidate_command=command,
                command_generated_at_s=solved_at_s,
                now_s=time.monotonic(),
                controller_command_usable=result.command_usable,
                previous_applied_command=observation.applied_command,
            )
            command = np.asarray(decision.command, dtype=np.float64)
            supervisor_intervention_count += int(decision.intervened)

        written = bool(link.writable)
        if written:
            link.write(command)
            written_command_count += 1

        status = result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        usable_command_count += int(result.command_usable)
        fallback_count += int(result.used_fallback)
        solve_times_s.append(float(result.diagnostics.solve_time_s))
        message_skews_s.append(observation.message_skew_s)
        receive_ages_s.append(observation.receive_age_s)
        source_clock_lags_s.append(observation.source_clock_lag_s)
        if observation.applied_command_skew_s is not None:
            applied_command_skews_s.append(observation.applied_command_skew_s)

        if on_sample is not None:
            on_sample(
                LoopSample(
                    step=step,
                    observation=observation,
                    result=result,
                    command=command,
                    written=written,
                    decision=decision,
                )
            )

    solve_times = np.asarray(solve_times_s)
    return LoopSummary(
        steps=steps,
        interval_s=interval_s,
        status_counts=dict(sorted(status_counts.items())),
        usable_command_count=usable_command_count,
        fallback_count=fallback_count,
        written_command_count=written_command_count,
        supervisor_intervention_count=supervisor_intervention_count,
        deadline_miss_count=int(np.count_nonzero(solve_times > interval_s)),
        solve_time_median_s=float(np.median(solve_times)),
        solve_time_p90_s=float(np.quantile(solve_times, 0.9)),
        solve_time_maximum_s=float(np.max(solve_times)),
        maximum_message_skew_s=max(message_skews_s),
        maximum_receive_age_s=max(receive_ages_s),
        maximum_source_clock_lag_s=max(source_clock_lags_s),
        maximum_applied_command_skew_s=(
            max(applied_command_skews_s) if applied_command_skews_s else None
        ),
        host_elapsed_s=time.monotonic() - started_at_s,
    )
