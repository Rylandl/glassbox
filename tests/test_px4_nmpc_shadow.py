"""The PX4 shadow leaf: one loop over a link that never transmits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.control.plan import (
    NMPCDiagnostics,
    NMPCWarmStart,
    ReferenceTrajectory,
    SolveResult,
    SolveStatus,
)
from glassbox.core.dynamics import hover_control
from glassbox.core.model import (
    DirectActuationMap,
    ExecutableModel,
    ModelValidityEnvelope,
    RuntimeModelSpec,
)
from glassbox.core.synthetic import generate_trajectory, resting_state, true_parameters
from glassbox.integrations.px4 import (
    PX4AppliedCommandSample,
    PX4StateSample,
    PX4TelemetryError,
)
from glassbox.integrations.px4_nmpc_shadow import (
    px4_shadow_link,
    run_px4_nmpc_shadow,
)

MODEL_PERIOD_S = 0.2


def runtime_model() -> ExecutableModel:
    params = true_parameters()
    spec = generate_trajectory(seed=0, duration_s=0.02).spec
    return ExecutableModel(
        params,
        spec,
        RuntimeModelSpec(
            sample_period_s=MODEL_PERIOD_S,
            validity_envelope=ModelValidityEnvelope(
                body_velocity_center_m_s=(0.0, 0.0, 0.0),
                body_velocity_half_width_m_s=(10.0, 10.0, 10.0),
                angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
                angular_velocity_half_width_rad_s=(10.0, 10.0, 10.0),
            ),
        ),
        DirectActuationMap(spec.controls),
    )


class StateSource:
    def __init__(self) -> None:
        self.sample_index = 0
        self.timeouts_s: list[float] = []

    def next_sample(self, *, timeout_s: float) -> PX4StateSample:
        self.timeouts_s.append(timeout_s)
        self.sample_index += 1
        return PX4StateSample(
            state=resting_state(),
            position_time_boot_ms=1_000 + 20 * self.sample_index,
            attitude_time_boot_ms=1_002 + 20 * self.sample_index,
            message_skew_s=0.002,
            maximum_receive_age_s=0.001,
        )


class CommandSource:
    def __init__(self) -> None:
        self.sample_index = 0

    def sample_nearest(
        self, time_boot_ms: int, *, timeout_s: float
    ) -> PX4AppliedCommandSample:
        self.sample_index += 1
        assert time_boot_ms == 1_000 + 20 * self.sample_index
        return PX4AppliedCommandSample(
            command=np.full(4, 0.2 + 0.05 * self.sample_index),
            source_time_us=(1_000 + 20 * self.sample_index) * 1_000,
            mav_mode=145,
            armed=True,
            receive_age_s=0.002,
        )


class Controller:
    """A stand-in for the real NMPC that records what the loop hands it."""

    def __init__(self, model: ExecutableModel, *, solve_time_s: float = 0.01) -> None:
        self.model = model
        self.sample_period_s = model.runtime_spec.sample_period_s
        self.solve_time_s = solve_time_s
        self.deadlines_s: list[float | None] = []
        self.applied_commands: list[np.ndarray] = []

    def hold_reference(self, state, *, exogenous=None) -> ReferenceTrajectory:
        assert np.asarray(state).shape == (13,)
        assert np.asarray(exogenous).shape == (self.model.exogenous_size,)
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
        self.deadlines_s.append(deadline_s)
        command = np.asarray(applied_command, dtype=np.float64)
        self.applied_commands.append(command)
        return SolveResult(
            status=SolveStatus.CONVERGED,
            command=command,
            predicted_states=np.zeros((2, 13)),
            predicted_latent_states=np.zeros((2, 4)),
            predicted_commands=np.zeros((1, 4)),
            warm_start=NMPCWarmStart(np.tile(command, (2, 1))),
            diagnostics=NMPCDiagnostics(
                iterations=1,
                solve_time_s=self.solve_time_s,
                initial_objective=2.0,
                final_objective=1.0,
                final_projected_gradient_inf_norm=1e-5,
                maximum_command_bound_violation=0.0,
                maximum_validity_utilization=0.1,
                maximum_normalized_safety_violation=0.0,
                maximum_normalized_model_uncertainty_standard_deviation=0.0,
                warm_start_used=warm_start is not None,
                prediction_horizon_s=0.4,
            ),
            used_fallback=False,
            message="",
        )


def test_shadow_run_records_every_interval_and_transmits_nothing() -> None:
    model = runtime_model()
    state_source = StateSource()
    link = px4_shadow_link(
        state_source,
        model,
        previous_command=np.asarray(hover_control(model.params)),
    )
    controller = Controller(model)
    lines: list[str] = []

    summary = run_px4_nmpc_shadow(link, controller, steps=2, write_line=lines.append)

    assert link.writable is False
    assert link.applied_command_source_kind == "fixed"
    assert state_source.sample_index == 2
    assert state_source.timeouts_s == [1.0] * 2
    assert controller.deadlines_s == [MODEL_PERIOD_S] * 2
    assert summary.steps == 2
    assert summary.status_counts == {"converged": 2}
    assert summary.written_command_count == 0
    assert summary.fallback_count == 0
    assert summary.deadline_miss_count == 0
    assert summary.solve_time_median_s == pytest.approx(0.01)
    assert summary.solve_time_p90_s == pytest.approx(0.01)
    assert summary.maximum_message_skew_s == pytest.approx(0.002)
    assert summary.maximum_receive_age_s >= 0.001
    assert summary.maximum_source_clock_lag_s == 0.0
    assert summary.maximum_applied_command_skew_s is None

    records = [json.loads(line) for line in lines]
    assert [record["step"] for record in records] == [0, 1]
    assert all(record["written"] is False for record in records)
    assert all(record["status"] == "converged" for record in records)
    assert records[0]["source_time_s"] == pytest.approx(1.020)
    json.dumps(summary.to_dict(), allow_nan=False)


def test_solve_times_over_the_model_period_are_counted_as_deadline_misses() -> None:
    model = runtime_model()
    link = px4_shadow_link(StateSource(), model, previous_command=np.full(4, 0.5))

    summary = run_px4_nmpc_shadow(
        link,
        Controller(model, solve_time_s=2.0 * MODEL_PERIOD_S),
        steps=2,
    )

    assert summary.deadline_miss_count == 2
    assert summary.interval_s == pytest.approx(MODEL_PERIOD_S)


def test_applied_command_telemetry_is_paired_with_the_state_it_matches() -> None:
    model = runtime_model()
    command_source = CommandSource()
    link = px4_shadow_link(StateSource(), model, applied_command_source=command_source)
    controller = Controller(model)
    lines: list[str] = []

    summary = run_px4_nmpc_shadow(link, controller, steps=2, write_line=lines.append)

    assert link.applied_command_source_kind == "telemetry"
    assert command_source.sample_index == 2
    np.testing.assert_allclose(controller.applied_commands[0], [0.25] * 4)
    np.testing.assert_allclose(controller.applied_commands[1], [0.30] * 4)
    assert summary.maximum_applied_command_skew_s == 0.0
    records = [json.loads(line) for line in lines]
    assert all(record["armed"] is True for record in records)


def test_misaligned_applied_command_telemetry_is_refused() -> None:
    class MisalignedCommandSource:
        def sample_nearest(
            self, time_boot_ms: int, *, timeout_s: float
        ) -> PX4AppliedCommandSample:
            return PX4AppliedCommandSample(
                command=np.full(4, 0.3),
                source_time_us=1_200_000,
                mav_mode=145,
                armed=True,
                receive_age_s=0.001,
            )

    model = runtime_model()
    link = px4_shadow_link(
        StateSource(), model, applied_command_source=MisalignedCommandSource()
    )

    with pytest.raises(PX4TelemetryError, match="alignment limit"):
        run_px4_nmpc_shadow(link, Controller(model), steps=1)


def test_the_link_refuses_a_command_outside_the_artifact_bounds() -> None:
    model = runtime_model()

    with pytest.raises(ValueError, match="inside the artifact bounds"):
        px4_shadow_link(StateSource(), model, previous_command=np.full(4, 1.1))


def test_the_link_requires_exactly_one_applied_command_source() -> None:
    model = runtime_model()

    with pytest.raises(ValueError, match="provide either"):
        px4_shadow_link(StateSource(), model)
    with pytest.raises(ValueError, match="mutually exclusive"):
        px4_shadow_link(
            StateSource(),
            model,
            previous_command=np.full(4, 0.5),
            applied_command_source=CommandSource(),
        )


def test_the_link_never_transmits_a_command() -> None:
    model = runtime_model()
    link = px4_shadow_link(StateSource(), model, previous_command=np.full(4, 0.5))

    assert not hasattr(link, "send")
    with pytest.raises(PX4TelemetryError, match="read-only"):
        link.write(np.full(4, 0.5))


def test_interval_records_are_written_one_json_object_per_line(
    tmp_path: Path,
) -> None:
    from glassbox.cli import px4_shadow

    model = runtime_model()
    link = px4_shadow_link(StateSource(), model, previous_command=np.full(4, 0.5))
    output = tmp_path / "nested" / "shadow.jsonl"

    with px4_shadow._line_writer(output) as write_line:
        run_px4_nmpc_shadow(link, Controller(model), steps=3, write_line=write_line)

    lines = output.read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 1, 2]
