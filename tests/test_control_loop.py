"""The loop contract: one interval, one link, one recorded outcome."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from glassbox.control.plan import (
    NMPCDiagnostics,
    NMPCWarmStart,
    ReferenceTrajectory,
    SolveResult,
    SolveStatus,
)
from glassbox.integrations.loop import (
    LoopSample,
    Observation,
    VehicleLink,
    run_control_loop,
)

INTERVAL_S = 0.02
RESTING_STATE = np.asarray(
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)


def diagnostics(*, solve_time_s: float = 0.001) -> NMPCDiagnostics:
    return NMPCDiagnostics(
        iterations=3,
        solve_time_s=solve_time_s,
        initial_objective=2.0,
        final_objective=1.0,
        final_projected_gradient_inf_norm=1e-4,
        maximum_command_bound_violation=0.0,
        maximum_validity_utilization=0.25,
        maximum_normalized_safety_violation=0.0,
        maximum_normalized_model_uncertainty_standard_deviation=math.inf,
        warm_start_used=False,
        prediction_horizon_s=0.12,
        prediction_horizon_certified=False,
    )


def solve_result(
    command: np.ndarray,
    *,
    status: SolveStatus = SolveStatus.CONVERGED,
    used_fallback: bool = False,
    solve_time_s: float = 0.001,
) -> SolveResult:
    return SolveResult(
        status=status,
        command=np.asarray(command, dtype=np.float64),
        predicted_states=np.zeros((2, 13)),
        predicted_latent_states=np.zeros((2, 4)),
        predicted_commands=np.zeros((1, 4)),
        warm_start=NMPCWarmStart(np.tile(command, (2, 1))),
        diagnostics=diagnostics(solve_time_s=solve_time_s),
        used_fallback=used_fallback,
        message="",
    )


class FakeLink:
    """A link that hands out scripted observations and records every write."""

    command_size = 4
    command_bounds = (np.zeros(4), np.ones(4))

    def __init__(
        self, *, writable: bool = True, skews_s: tuple[float, ...] = ()
    ) -> None:
        self.writable = writable
        self.skews_s = skews_s
        self.read_timeouts_s: list[float] = []
        self.written: list[np.ndarray] = []
        self.reads = 0

    def read(self, *, timeout_s: float) -> Observation:
        self.read_timeouts_s.append(timeout_s)
        index = self.reads
        self.reads += 1
        skew_s = self.skews_s[index] if index < len(self.skews_s) else 0.0
        return Observation(
            state=RESTING_STATE,
            applied_command=np.full(4, 0.5),
            received_at_s=float(index) * INTERVAL_S,
            source_time_s=float(index) * INTERVAL_S,
            message_skew_s=skew_s,
            receive_age_s=0.5 * skew_s,
            source_clock_lag_s=0.25 * skew_s,
            applied_command_skew_s=skew_s,
            armed=True,
        )

    def write(self, command: np.ndarray) -> None:
        if not self.writable:
            raise RuntimeError("this link is read-only")
        self.written.append(np.asarray(command, dtype=np.float64))


class FakeController:
    """A controller that records its arguments and returns scripted results."""

    sample_period_s = INTERVAL_S

    def __init__(self, results: tuple[SolveResult, ...] | None = None) -> None:
        self.results = results
        self.deadlines_s: list[float | None] = []
        self.warm_starts: list[NMPCWarmStart | None] = []
        self.references: list[ReferenceTrajectory] = []
        self.calls = 0

    def hold_reference(self, state, *, exogenous=None) -> ReferenceTrajectory:
        return ReferenceTrajectory.hold(state, 2, exogenous=exogenous)

    def solve(
        self,
        state,
        reference,
        previous_command,
        *,
        applied_command=None,
        latent_state=None,
        warm_start=None,
        deadline_s=None,
    ) -> SolveResult:
        index = self.calls
        self.calls += 1
        self.deadlines_s.append(deadline_s)
        self.warm_starts.append(warm_start)
        self.references.append(reference)
        if self.results is not None:
            return self.results[index]
        return solve_result(np.full(4, 0.25 + 0.05 * index))


class RefusingSupervisor:
    """A supervisor that always replaces the candidate with a fixed hold."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def supervise(self, **keywords):
        self.calls.append(keywords)

        class Decision:
            command = np.full(4, 0.5)
            intervened = True

            @staticmethod
            def to_dict() -> dict[str, object]:
                return {"mode": "collective_hold"}

        return Decision()


def test_observation_refuses_an_incomplete_or_backwards_sample() -> None:
    with pytest.raises(ValueError, match="13 finite values"):
        Observation(state=np.zeros(12), applied_command=np.ones(4), received_at_s=0.0)
    with pytest.raises(ValueError, match="nonempty finite vector"):
        Observation(state=RESTING_STATE, applied_command=np.empty(0), received_at_s=0.0)
    with pytest.raises(ValueError, match="message_skew_s"):
        Observation(
            state=RESTING_STATE,
            applied_command=np.ones(4),
            received_at_s=0.0,
            message_skew_s=-1e-9,
        )
    observation = Observation(
        state=RESTING_STATE, applied_command=np.ones(4), received_at_s=1.0
    )
    assert not observation.state.flags.writeable
    assert not observation.applied_command.flags.writeable


def test_loop_reads_solves_and_writes_one_command_per_interval() -> None:
    link = FakeLink()
    controller = FakeController()

    summary = run_control_loop(
        link,
        controller,
        steps=3,
        reference=lambda observation: controller.hold_reference(observation.state),
    )

    assert link.reads == 3
    assert len(link.written) == 3
    np.testing.assert_allclose(link.written[-1], np.full(4, 0.35))
    assert link.read_timeouts_s == [INTERVAL_S] * 3
    assert controller.deadlines_s == [INTERVAL_S] * 3
    assert summary.steps == 3
    assert summary.status_counts == {"converged": 3}
    assert summary.usable_command_count == 3
    assert summary.written_command_count == 3
    assert summary.fallback_count == 0
    assert summary.supervisor_intervention_count == 0


def test_loop_hands_each_solve_the_previous_interval_warm_start() -> None:
    link = FakeLink()
    controller = FakeController()

    run_control_loop(
        link, controller, steps=3, reference=ReferenceTrajectory.hold(RESTING_STATE, 2)
    )

    assert controller.warm_starts[0] is None
    assert controller.warm_starts[1] is not None
    np.testing.assert_allclose(
        np.asarray(controller.warm_starts[1].commands)[0], np.full(4, 0.25)
    )
    np.testing.assert_allclose(
        np.asarray(controller.warm_starts[2].commands)[0], np.full(4, 0.30)
    )


def test_a_failed_solve_is_recorded_and_the_loop_keeps_running() -> None:
    link = FakeLink()
    controller = FakeController(
        results=(
            solve_result(np.full(4, 0.4)),
            solve_result(
                np.full(4, 0.5),
                status=SolveStatus.DEADLINE_EXCEEDED,
                used_fallback=True,
                solve_time_s=10.0 * INTERVAL_S,
            ),
            solve_result(np.full(4, 0.6)),
        )
    )

    summary = run_control_loop(
        link, controller, steps=3, reference=ReferenceTrajectory.hold(RESTING_STATE, 2)
    )

    assert summary.status_counts == {"converged": 2, "deadline_exceeded": 1}
    assert summary.fallback_count == 1
    assert summary.usable_command_count == 2
    assert summary.deadline_miss_count == 1
    assert summary.solve_time_maximum_s == pytest.approx(10.0 * INTERVAL_S)
    assert len(link.written) == 3


def test_a_read_only_link_is_never_written_to() -> None:
    link = FakeLink(writable=False)
    controller = FakeController()

    summary = run_control_loop(
        link, controller, steps=2, reference=ReferenceTrajectory.hold(RESTING_STATE, 2)
    )

    assert link.written == []
    assert summary.written_command_count == 0


def test_the_supervised_command_is_the_one_that_reaches_the_link() -> None:
    link = FakeLink()
    controller = FakeController()
    supervisor = RefusingSupervisor()

    summary = run_control_loop(
        link,
        controller,
        supervisor,
        steps=2,
        reference=ReferenceTrajectory.hold(RESTING_STATE, 2),
    )

    assert summary.supervisor_intervention_count == 2
    for written in link.written:
        np.testing.assert_allclose(written, np.full(4, 0.5))
    first = supervisor.calls[0]
    assert first["controller_command_usable"] is True
    assert first["state_received_at_s"] == 0.0
    assert first["now_s"] >= first["command_generated_at_s"]


def test_summary_reports_solve_time_and_skew_statistics() -> None:
    link = FakeLink(skews_s=(0.001, 0.004, 0.002))
    controller = FakeController(
        results=tuple(
            solve_result(np.full(4, 0.3), solve_time_s=time_s)
            for time_s in (0.001, 0.003, 0.002)
        )
    )

    summary = run_control_loop(
        link, controller, steps=3, reference=ReferenceTrajectory.hold(RESTING_STATE, 2)
    )

    assert summary.solve_time_median_s == pytest.approx(0.002)
    assert summary.solve_time_p90_s == pytest.approx(0.0028)
    assert summary.maximum_message_skew_s == pytest.approx(0.004)
    assert summary.maximum_receive_age_s == pytest.approx(0.002)
    assert summary.maximum_source_clock_lag_s == pytest.approx(0.001)
    assert summary.maximum_applied_command_skew_s == pytest.approx(0.004)
    assert summary.host_elapsed_s >= 0.0
    json.dumps(summary.to_dict(), allow_nan=False)


def test_every_interval_record_is_json_and_carries_the_solver_outcome() -> None:
    link = FakeLink()
    controller = FakeController()
    records: list[LoopSample] = []

    run_control_loop(
        link,
        controller,
        RefusingSupervisor(),
        steps=2,
        reference=ReferenceTrajectory.hold(RESTING_STATE, 2),
        on_sample=records.append,
    )

    assert [record.step for record in records] == [0, 1]
    first = records[0].to_dict()
    json.dumps(first, allow_nan=False)
    assert first["status"] == "converged"
    assert first["written"] is True
    assert first["armed"] is True
    assert first["command"] == pytest.approx([0.5] * 4)
    assert first["supervisor"] == {"mode": "collective_hold"}
    assert first["solve_time_s"] == pytest.approx(0.001)
    # A margin that no evidence bounds is written as null rather than infinity.
    assert (
        first["diagnostics"]["maximum_normalized_model_uncertainty_standard_deviation"]
        is None
    )


def test_the_loop_refuses_a_controller_without_a_usable_interval() -> None:
    controller = FakeController()
    controller.sample_period_s = 0.0

    with pytest.raises(ValueError, match="sample period"):
        run_control_loop(
            FakeLink(),
            controller,
            steps=1,
            reference=ReferenceTrajectory.hold(RESTING_STATE, 2),
        )

    with pytest.raises(ValueError, match="steps must be positive"):
        run_control_loop(
            FakeLink(),
            FakeController(),
            steps=0,
            reference=ReferenceTrajectory.hold(RESTING_STATE, 2),
        )


def test_a_link_is_recognized_by_the_protocol_it_satisfies() -> None:
    assert isinstance(FakeLink(), VehicleLink)
    assert not isinstance(FakeController(), VehicleLink)
