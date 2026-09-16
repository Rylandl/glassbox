"""Bounded transport and thread ownership, with numerical kernels stubbed.

Real multirotor and fixed-wing updates are exercised in test_refinement.
Events make scheduling assertions independent of the machine's solver speed.
"""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox import DynamicsBelief, ExecutableModel, UpdateResult
from glassbox.core.model import runtime_spec_from_trajectory
from glassbox.core.synthetic import true_parameters
from glassbox.workflows import refinement
from glassbox.workflows.refinement import ForecastScore
from glassbox.workflows.streaming import RefinementWorker, TransitionBuffer


@pytest.fixture
def setup(quadrotor_flight, monkeypatch):
    flight = quadrotor_flight(27, 0.4)
    belief = DynamicsBelief(
        ExecutableModel(
            true_parameters(), flight.spec, runtime_spec_from_trajectory(flight)
        )
    )

    def score(revision, trajectory, **kwargs):
        prediction = SimpleNamespace(
            states=trajectory.states,
            validity_utilization=np.zeros(len(trajectory.states)),
            forecast_error_horizon_supported=False,
            parameter_covariance=None,
        )
        return ForecastScore(revision, prediction, {"position_rmse_m": 0.1})

    def absorb(self, trajectory):
        return replace(
            self, provenance={"update_count": self.update_count + 1}
        ), UpdateResult(True, None, len(trajectory.controls), 1.0, 0.9, 0.0, None, 0.5)

    monkeypatch.setattr(refinement, "_score", score)
    monkeypatch.setattr(DynamicsBelief, "absorb", absorb)
    return belief, flight


def blocks(flight, *, missing=()):
    buffer = TransitionBuffer(
        flight.spec,
        flight.nominal_dt_s,
        recording_id="f",
        block_steps=2,
        history_steps=2,
    )
    result = []
    for i, command in enumerate(flight.controls):
        block = buffer.push(
            interval=i,
            source_time_s=flight.time_s[i],
            command=command,
            state=None if i in missing else flight.states[i],
            next_state=None if i in missing else flight.states[i + 1],
            received_at_s=time.monotonic(),
        )
        if block is not None:
            result.append(block)
        assert len(buffer.history) <= 2
        assert buffer.partial_intervals < 2
    return buffer, result


def test_gap_does_not_bridge_missing_states_or_lose_known_command_history(setup):
    _, flight = setup
    buffer, emitted = blocks(flight, missing=(5,))
    assert buffer.discarded_intervals == 4  # Two startup, one partial, one missing.
    assert [b.start_interval for b in emitted[:2]] == [2, 6]
    for block in emitted:
        i = block.start_interval
        np.testing.assert_array_equal(
            block.trajectory.control_prefix, flight.controls[i - 2 : i]
        )
        np.testing.assert_array_equal(block.trajectory.states, flight.states[i : i + 3])


def test_timestamp_and_endpoint_rejection_do_not_advance_buffer(setup):
    _, flight = setup
    buffer = TransitionBuffer(
        flight.spec,
        flight.nominal_dt_s,
        recording_id="f",
        block_steps=2,
        history_steps=2,
    )

    def push(i, **overrides):
        values = dict(
            interval=i,
            source_time_s=flight.time_s[i],
            command=flight.controls[i],
            state=flight.states[i],
            next_state=flight.states[i + 1],
            received_at_s=time.monotonic(),
        )
        values.update(overrides)
        return buffer.push(**values)

    push(0)
    with pytest.raises(ValueError, match="timestamp"):
        push(1, source_time_s=0.5)
    assert buffer.next_interval == 1
    push(1)
    push(2)
    with pytest.raises(ValueError, match="endpoint changed"):
        push(3, state=flight.states[3] + 0.01)
    assert buffer.next_interval == 3
    assert push(3) is not None


def test_large_source_epoch_is_preserved_separately_from_model_intervals(setup):
    _, flight = setup
    buffer = TransitionBuffer(
        flight.spec,
        flight.nominal_dt_s,
        recording_id="f",
        block_steps=2,
        history_steps=2,
    )
    epoch = 1_700_000_000.0
    for i in range(4):
        block = buffer.push(
            interval=i,
            source_time_s=epoch + i * flight.nominal_dt_s,
            command=flight.controls[i],
            state=flight.states[i],
            next_state=flight.states[i + 1],
            received_at_s=time.monotonic(),
        )
    np.testing.assert_allclose(block.trajectory.time_s, [0.0, 0.02, 0.04])
    assert block.trajectory.provenance["source_time_start_s"] == epoch + 0.04
    assert block.trajectory.provenance["source_time_end_s"] == epoch + 0.08


def test_control_producer_continues_while_learner_is_blocked_and_queue_is_bounded(
    setup,
):
    belief, flight = setup
    _, emitted = blocks(flight)
    entered, release = threading.Event(), threading.Event()
    events = []

    def delay(index):
        if index == 0:
            entered.set()
            assert release.wait(5)
        return 0.0

    worker = RefinementWorker(
        belief,
        history_steps=2,
        block_steps=2,
        queue_capacity=2,
        retained_blocks=1,
        delay_s=delay,
        on_event=events.append,
    )
    worker.start()
    try:
        worker.submit(emitted[0])
        assert entered.wait(5)
        for block in emitted[1:]:
            assert worker.submit(block)
        assert worker.processed_blocks == 0
        assert worker.dropped_blocks == len(emitted) - 3
        assert worker.peak_queue_blocks <= 2
    finally:
        release.set()
        assert worker.close()
    assert worker.error is None
    accepted = [e for e in events if e["kind"] == "block"]
    assert [e["start_interval"] for e in accepted] == [2, 16, 18]
    assert worker.refiner.skipped_interval_count == 14
    assert worker.refiner.block_count == 3
    assert len(worker.refiner.results) == 1
    assert not worker.submit(emitted[-1])


def test_preparation_and_acknowledgement_are_separate_from_control(setup):
    belief, flight = setup
    _, emitted = blocks(flight)
    offered, adopted = threading.Event(), threading.Event()
    owner = threading.get_ident()
    prepared_on = []

    def prepare(revision, trajectory):
        prepared_on.append(threading.get_ident())
        return object()

    def event(record):
        if record["kind"] == "offer":
            offered.set()
        if record["kind"] == "adoption":
            adopted.set()

    worker = RefinementWorker(
        belief,
        history_steps=2,
        block_steps=2,
        prepare=prepare,
        should_offer=lambda result: True,
        on_event=event,
    )
    original = worker.initial_revision
    worker.start()
    try:
        worker.submit(emitted[0])
        worker.submit(emitted[1])
        assert offered.wait(5)
        offer = worker.poll_offer()
        assert offer is not None and offer.revision is not original
        assert offer.expected_active_revision == original.revision_id
        assert worker.refiner.active is original
        assert prepared_on == [worker._thread.ident]
        assert prepared_on[0] != owner
        worker.acknowledge(offer, applied=True)
        assert adopted.wait(5)
        assert worker.refiner.active is offer.revision
    finally:
        assert worker.close()
    assert worker.error is None


def test_worker_failure_is_reported_and_does_not_replace_active_model(setup):
    belief, flight = setup
    _, emitted = blocks(flight)
    failed = threading.Event()

    def prepare(revision, trajectory):
        raise RuntimeError("injected controller preparation failure")

    worker = RefinementWorker(
        belief,
        history_steps=2,
        block_steps=2,
        prepare=prepare,
        should_offer=lambda result: True,
        on_event=lambda e: failed.set() if e["kind"] == "worker_error" else None,
    )
    original = worker.initial_revision
    worker.start()
    worker.submit(emitted[0])
    worker.submit(emitted[1])
    assert failed.wait(5)
    assert worker.close()
    assert "injected" in worker.error
    assert worker.refiner.active is original
    assert worker.poll_offer() is None
    assert not worker.submit(emitted[-1])
