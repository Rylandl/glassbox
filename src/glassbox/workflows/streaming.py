"""Bounded telemetry transport and a single-owner background refinement worker.

The producer supplies aligned transitions with the command actually applied
over each interval. This module does not align raw sensor messages or run a
vehicle. Controller preparation and the adoption policy belong to the caller.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from glassbox.belief.belief import DynamicsBelief
from glassbox.core.data import Trajectory, TrajectorySpec
from glassbox.workflows.refinement import ModelRefiner, ModelRevision, RefinementResult


@dataclass(frozen=True)
class TelemetryBlock:
    recording_id: str
    start_interval: int
    trajectory: Trajectory
    received_at_s: float

    def __post_init__(self):
        if (
            not self.recording_id
            or not isinstance(self.start_interval, int)
            or self.start_interval < 0
        ):
            raise ValueError(
                "a telemetry block requires a recording ID and nonnegative interval index"
            )
        if not np.isfinite(self.received_at_s):
            raise ValueError("received_at_s must be finite")

    @property
    def stop_interval(self) -> int:
        return self.start_interval + len(self.trajectory.controls)


class TransitionBuffer:
    """Assemble fixed-size blocks with a bounded history of known commands.

    Source times denote interval starts; reception uses a separate monotonic
    host clock. Missing state pairs discard the partial block, never join states
    across a gap. Commands remain known during a state-telemetry interruption.
    Initial intervals fill command history before any block is emitted. The
    finite history is an actuator initialization approximation, not a measured
    latent state. One producer owns this buffer.
    """

    def __init__(
        self,
        spec: TrajectorySpec,
        sample_period_s: float,
        *,
        recording_id: str,
        block_steps: int,
        history_steps: int,
    ):
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in (block_steps, history_steps)
        ):
            raise ValueError("block_steps and history_steps must be positive integers")
        if not recording_id or not np.isfinite(sample_period_s) or sample_period_s <= 0:
            raise ValueError("recording ID and positive sample period required")
        self.spec, self.sample_period_s = spec, float(sample_period_s)
        self.recording_id, self.block_steps = recording_id, block_steps
        self.history: deque[np.ndarray] = deque(maxlen=history_steps)
        self.next_interval = 0
        self.discarded_intervals = 0
        self._origin: float | None = None
        self._commands: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._exogenous: list[np.ndarray] = []
        self._prefix: np.ndarray | None = None
        self._start = 0

    def push(
        self,
        *,
        interval: int,
        source_time_s: float,
        command: np.ndarray,
        state: np.ndarray | None,
        next_state: np.ndarray | None,
        received_at_s: float,
        exogenous: np.ndarray | None = None,
        next_exogenous: np.ndarray | None = None,
    ) -> TelemetryBlock | None:
        if (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval != self.next_interval
        ):
            raise ValueError("transition index must match the next interval")
        if not np.isfinite(source_time_s) or not np.isfinite(received_at_s):
            raise ValueError("source and reception times must be finite")
        origin = source_time_s if self._origin is None else self._origin
        expected_time = origin + interval * self.sample_period_s
        if not np.isclose(
            source_time_s,
            expected_time,
            rtol=0,
            atol=max(1e-8, 4 * abs(np.spacing(expected_time))),
        ):
            raise ValueError("source timestamp does not match the fixed-rate interval")
        command = np.array(command, dtype=float, copy=True)
        if (
            command.shape != (len(self.spec.controls),)
            or not np.isfinite(command).all()
        ):
            raise ValueError("command does not match the finite telemetry contract")
        for value, channel in zip(command, self.spec.controls):
            if (channel.minimum is not None and value < channel.minimum) or (
                channel.maximum is not None and value > channel.maximum
            ):
                raise ValueError("command exceeds declared bounds")
        if (state is None) != (next_state is None):
            raise ValueError("both endpoint states must be present or absent")
        context = (
            np.zeros(len(self.spec.exogenous))
            if exogenous is None
            else np.array(exogenous, copy=True)
        )
        next_context = (
            np.zeros(len(self.spec.exogenous))
            if next_exogenous is None
            else np.array(next_exogenous, copy=True)
        )
        if state is not None:
            state, next_state = (
                np.array(state, copy=True),
                np.array(next_state, copy=True),
            )
            if (
                state.shape != (13,)
                or next_state.shape != (13,)
                or not np.isfinite([state, next_state]).all()
            ):
                raise ValueError("endpoint states must each contain 13 finite values")
            if (
                context.shape != (len(self.spec.exogenous),)
                or next_context.shape != context.shape
                or not np.isfinite([context, next_context]).all()
            ):
                raise ValueError(
                    "exogenous inputs must match the finite channel contract"
                )
            if self._states and (
                not np.array_equal(state, self._states[-1])
                or not np.array_equal(context, self._exogenous[-1])
            ):
                raise ValueError("shared endpoint changed within a telemetry block")
        prefix = np.asarray(self.history)
        have_history = len(self.history) == self.history.maxlen
        self._origin = origin
        self.next_interval += 1
        self.history.append(command)
        if state is None or not have_history:
            self.discarded_intervals += len(self._commands) + 1
            self._commands.clear()
            self._states.clear()
            self._exogenous.clear()
            self._prefix = None
            return None
        if not self._commands:
            self._start = interval
            self._prefix = prefix.copy()
            self._states.append(state)
            self._exogenous.append(context)
        self._commands.append(command)
        self._states.append(next_state)
        self._exogenous.append(next_context)
        if len(self._commands) < self.block_steps:
            return None
        trajectory = Trajectory(
            time_s=np.arange(self.block_steps + 1) * self.sample_period_s,
            states=np.asarray(self._states),
            controls=np.asarray(self._commands),
            spec=self.spec,
            exogenous=np.asarray(self._exogenous),
            control_prefix=self._prefix,
            provenance={
                "source_time_start_s": origin + self._start * self.sample_period_s,
                "source_time_end_s": origin
                + (self._start + self.block_steps) * self.sample_period_s,
                "timestamp_policy": "validated_fixed_rate_then_rebased",
            },
        )
        result = TelemetryBlock(
            self.recording_id, self._start, trajectory, received_at_s
        )
        self._commands.clear()
        self._states.clear()
        self._exogenous.clear()
        self._prefix = None
        return result

    @property
    def partial_intervals(self) -> int:
        return len(self._commands)


@dataclass(frozen=True)
class ControllerOffer:
    revision: ModelRevision
    expected_active_revision: str
    controller: Any
    scored_recording_id: str
    scored_stop_interval: int
    prepared_at_s: float


class Refiner(Protocol):
    """Everything :class:`RefinementWorker` asks of the thing that learns.

    :class:`~glassbox.workflows.refinement.ModelRefiner` is the implementation
    for the structured belief, and the worker builds one when it is handed a
    belief. A caller whose learner is something else supplies its own instead:
    the worker calls exactly what is listed here, owns none of it, and never
    reaches past it into a model. The contract is the one ``ModelRefiner``
    already keeps -- blocks arrive in order on a recording whose gaps are
    declared by :meth:`skip`, ``observe`` scores the incoming block before
    absorbing it, and a revision becomes active only through :meth:`adopt`.
    """

    history_steps: int | None
    skipped_interval_count: int
    retained_revision_count: int

    @property
    def active(self) -> Any:
        """The revision a consumer is currently using."""

    @property
    def candidate(self) -> Any:
        """The revision the most recent block was absorbed into."""

    @property
    def results(self) -> tuple[Any, ...]:
        """The retained per-block results, oldest first."""

    def observe(
        self, telemetry: Trajectory, *, recording_id: str, start_interval: int
    ) -> Any:
        """Score this block, absorb it, and return a result carrying both.

        The result must expose ``candidate_score.revision``, the evaluated
        snapshot a consumer may be offered, and ``to_dict()`` for the journal.
        """

    def skip(
        self, *, recording_id: str, start_interval: int, stop_interval: int, reason: str
    ) -> dict:
        """Account for intervals that never arrived, without inventing them."""

    def adopt(
        self, revision_id: str, *, expected_active_revision: str, reason: str
    ) -> Any:
        """Make a previously scored revision active.

        The record it returns is journalled through ``to_dict()``.
        """

    def hold_for_adoption(self, revision_id: str) -> None:
        """Retain one scored revision while a consumer decides about it."""

    def release_adoption_hold(self) -> None:
        """Release that retention once the decision is in."""


class RefinementWorker:
    """Learn and prepare controllers off the producer/control thread.

    Queue overflow discards the oldest pending block. Every resulting interval
    gap is acknowledged explicitly by the refiner. One controller offer may be
    outstanding; the control owner acknowledges it after applying or rejecting
    it at a solve boundary. Callbacks run only on the worker thread. Events should
    be streamed to a journal rather than retained indefinitely by the caller.

    Supply either a belief, which the worker wraps in a
    :class:`~glassbox.workflows.refinement.ModelRefiner`, or a ``refiner`` of
    the caller's own that keeps the :class:`Refiner` contract. The transport,
    the queue bound, the gap accounting and the acknowledged handoff are the
    same either way; what learns behind them is the caller's.

    ``synchronous`` runs the same learning on the producer's own thread inside
    :meth:`submit`, and carries out an acknowledged decision inside
    :meth:`acknowledge`. Nothing is queued, dropped or waited for, and no result
    arrives at an interval the producer did not choose. It is for a caller whose
    measurement must not depend on when a thread happened to be scheduled. It is
    not a lower-latency mode: the producer pays the learning inline.
    """

    def __init__(
        self,
        belief: DynamicsBelief | None = None,
        *,
        history_steps: int,
        block_steps: int,
        queue_capacity: int = 2,
        retained_blocks: int = 2,
        refiner: Refiner | None = None,
        synchronous: bool = False,
        prepare: Callable[[ModelRevision, Trajectory], Any] | None = None,
        should_offer: Callable[[RefinementResult], bool] | None = None,
        on_event: Callable[[dict], None] | None = None,
        delay_s: Callable[[int], float] | None = None,
    ):
        if (
            not isinstance(queue_capacity, int)
            or queue_capacity < 1
            or not isinstance(block_steps, int)
            or block_steps < 1
        ):
            raise ValueError("queue_capacity and block_steps must be positive integers")
        if (prepare is None) != (should_offer is None):
            raise ValueError(
                "controller preparation and adoption policy must be supplied together"
            )
        if (belief is None) == (refiner is None):
            raise ValueError("supply either a belief or a refiner, and not both")
        if refiner is None:
            refiner = ModelRefiner(
                belief,
                history_steps=history_steps,
                retained_blocks=retained_blocks,
                max_recordings=1,
                propagate_parameter_covariance=False,
            )
        elif refiner.history_steps != history_steps:
            raise ValueError(
                "the supplied refiner keeps a different command history than the "
                "blocks this worker accepts"
            )
        self.refiner = refiner
        self.block_steps = block_steps
        self._input: queue.Queue[TelemetryBlock] = queue.Queue(queue_capacity)
        self._offers: queue.Queue[ControllerOffer] = queue.Queue(1)
        self._acks: queue.Queue[tuple[ControllerOffer, bool]] = queue.Queue(1)
        self._pending: ControllerOffer | None = None
        self._prepare, self._should_offer = prepare, should_offer
        self._on_event = on_event or (lambda event: None)
        self._delay = delay_s or (lambda index: 0.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.synchronous = bool(synchronous)
        self._started = False
        self.error: str | None = None
        self.submitted_blocks = self.dropped_blocks = self.dropped_intervals = 0
        self.processed_blocks = self.peak_queue_blocks = 0
        self.next_interval = 0
        self.maximum_queue_age_s = self.maximum_update_time_s = 0.0

    @property
    def initial_revision(self) -> ModelRevision:
        if self._started:
            raise RuntimeError(
                "capture the initial revision before starting the worker"
            )
        return self.refiner.active

    def start(self) -> None:
        if self._started or self._stop.is_set():
            raise RuntimeError("worker has already started")
        self._started = True
        if self.synchronous:
            return
        self._thread = threading.Thread(
            target=self._run, name="glassbox-refinement", daemon=True
        )
        self._thread.start()

    def submit(self, block: TelemetryBlock) -> bool:
        """Hand over one block: enqueued for the worker thread, or learned inline.

        Asynchronously this enqueues without waiting for learning or for free
        queue capacity, and the oldest pending block is discarded when the queue
        is full. Synchronously the producer's own thread runs the learning
        before this returns, so nothing is ever pending and nothing is ever
        dropped: the queue bound does not apply, and no offer can arrive at an
        interval other than the one that handed the block over.
        """
        if not self._started or self._stop.is_set() or self.error is not None:
            return False
        if (
            len(block.trajectory.controls) != self.block_steps
            or block.trajectory.control_prefix is None
            or len(block.trajectory.control_prefix) != self.refiner.history_steps
        ):
            raise ValueError("worker requires fixed block and command-history sizes")
        self.submitted_blocks += 1
        if self.synchronous:
            try:
                self._acknowledgements()
                self._process(block)
            except Exception as error:
                self._fail(error)
                return False
            return True
        try:
            self._input.put_nowait(block)
        except queue.Full:
            try:
                dropped = self._input.get_nowait()
                self.dropped_blocks += 1
                self.dropped_intervals += len(dropped.trajectory.controls)
            except queue.Empty:
                pass  # The worker removed it after the full check.
            self._input.put_nowait(block)
        self.peak_queue_blocks = max(self.peak_queue_blocks, self._input.qsize())
        return True

    def poll_offer(self) -> ControllerOffer | None:
        try:
            return self._offers.get_nowait()
        except queue.Empty:
            return None

    def acknowledge(self, offer: ControllerOffer, *, applied: bool) -> None:
        """The control owner reports its completed boundary decision.

        Synchronously the decision is carried out before this returns, so an
        adopted revision is active for the caller's very next interval rather
        than whenever a worker thread next looks at its queue.
        """
        self._acks.put_nowait((offer, applied))
        if self.synchronous:
            try:
                self._acknowledgements()
            except Exception as error:
                self._fail(error)

    def close(self, *, timeout_s: float = 30.0) -> bool:
        """Request shutdown, drain accepted telemetry, and wait up to the timeout.

        Synchronously there is nothing to drain and nothing to wait for: every
        block was learned inside the ``submit`` that handed it over.
        """
        self._stop.set()
        if self.synchronous:
            return True
        if self._thread is not None:
            self._thread.join(timeout_s)
        return self._thread is None or not self._thread.is_alive()

    def _fail(self, error: Exception) -> None:
        self.error = f"{type(error).__name__}: {error}"
        self._stop.set()
        # A failed journal callback must not hide the original worker error.
        try:
            self._on_event({"kind": "worker_error", "error": self.error})
        except Exception:
            pass

    def _acknowledgements(self) -> None:
        try:
            offer, applied = self._acks.get_nowait()
        except queue.Empty:
            return
        if offer is not self._pending:
            raise ValueError("acknowledgement does not identify the outstanding offer")
        if applied:
            adoption = self.refiner.adopt(
                offer.revision.revision_id,
                expected_active_revision=offer.expected_active_revision,
                reason="application gate and prepared controller accepted at a solve boundary",
            )
            self._on_event({"kind": "adoption", **adoption.to_dict()})
        else:
            self._on_event(
                {"kind": "offer_rejected", "revision_id": offer.revision.revision_id}
            )
        self._pending = None
        self.refiner.release_adoption_hold()

    def _process(self, block: TelemetryBlock) -> None:
        """Learn from one block and, if the policy says so, prepare an offer.

        The same body either way: on the worker thread when one is running, and
        on the producer's own thread when the worker is synchronous.
        """
        delay = float(self._delay(self.processed_blocks))
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("worker delay must be finite and nonnegative")
        self._stop.wait(delay)
        if block.start_interval > self.next_interval:
            gap = self.refiner.skip(
                recording_id=block.recording_id,
                start_interval=self.next_interval,
                stop_interval=block.start_interval,
                reason="unavailable telemetry or queue overflow",
            )
            self._on_event({"kind": "gap", **gap})
        started = time.monotonic()
        age = started - block.received_at_s
        if age < 0:
            raise ValueError(
                "reception timestamp is ahead of the worker monotonic clock"
            )
        result = self.refiner.observe(
            block.trajectory,
            recording_id=block.recording_id,
            start_interval=block.start_interval,
        )
        elapsed = time.monotonic() - started
        self.next_interval = block.stop_interval
        self.processed_blocks += 1
        self.maximum_queue_age_s = max(self.maximum_queue_age_s, age)
        self.maximum_update_time_s = max(self.maximum_update_time_s, elapsed)
        self._on_event(
            {
                "kind": "block",
                "queue_age_s": age,
                "update_time_s": elapsed,
                **result.to_dict(),
            }
        )
        if (
            not self._stop.is_set()
            and self._pending is None
            and self._prepare is not None
            and self._should_offer(result)
        ):
            revision = result.candidate_score.revision
            if revision.revision_id != self.refiner.active.revision_id:
                started = time.monotonic()
                controller = self._prepare(revision, block.trajectory)
                offer = ControllerOffer(
                    revision,
                    self.refiner.active.revision_id,
                    controller,
                    block.recording_id,
                    block.stop_interval,
                    time.monotonic(),
                )
                self._pending = offer
                self.refiner.hold_for_adoption(revision.revision_id)
                self._offers.put_nowait(offer)
                self._on_event(
                    {
                        "kind": "offer",
                        "revision": revision.to_dict(),
                        "scored_stop_interval": block.stop_interval,
                        "preparation_time_s": time.monotonic() - started,
                    }
                )

    def _run(self) -> None:
        try:
            while True:
                self._acknowledgements()
                if self._stop.is_set() and self._input.empty():
                    break
                try:
                    block = self._input.get(timeout=0.02)
                except queue.Empty:
                    continue
                self._process(block)
            self._acknowledgements()
        except Exception as error:
            self._fail(error)

    def summary(self) -> dict:
        """Call after close, or treat concurrent values as approximate counters."""
        return {
            "submitted_blocks": self.submitted_blocks,
            "processed_blocks": self.processed_blocks,
            "dropped_blocks": self.dropped_blocks,
            "dropped_intervals": self.dropped_intervals,
            "skipped_intervals": self.refiner.skipped_interval_count,
            "peak_queue_blocks": self.peak_queue_blocks,
            "maximum_queue_age_s": self.maximum_queue_age_s,
            "maximum_update_time_s": self.maximum_update_time_s,
            "retained_blocks": len(self.refiner.results),
            "retained_revisions": self.refiner.retained_revision_count,
            "pending_offer": self._pending is not None,
            "error": self.error,
            "final_candidate": self.refiner.candidate.to_dict(),
            "acknowledged_active": self.refiner.active.to_dict(),
        }
