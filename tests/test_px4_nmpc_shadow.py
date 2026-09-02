from __future__ import annotations

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import glassbox.integrations.px4_nmpc_shadow as px4_nmpc_shadow
from glassbox.control.nmpc import NMPCWarmStart
from glassbox.core.dynamics import hover_control
from glassbox.core.runtime import (
    DirectActuationMap,
    ModelValidityEnvelope,
    RuntimeDynamicsModel,
    RuntimeModelSpec,
)
from glassbox.core.synthetic import generate_trajectory, resting_state, true_parameters
from glassbox.integrations.px4 import (
    PX4AppliedCommandSample,
    PX4StateSample,
    PX4TelemetryError,
)
from glassbox.integrations.px4_nmpc_shadow import run_px4_nmpc_shadow


def runtime_model() -> RuntimeDynamicsModel:
    params = true_parameters()
    spec = generate_trajectory(seed=0, duration_s=0.02).spec
    return RuntimeDynamicsModel(
        params,
        spec,
        RuntimeModelSpec(
            sample_period_s=0.2,
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

    def next_sample(self, *, timeout_s: float) -> PX4StateSample:
        assert timeout_s == 1.0
        self.sample_index += 1
        return PX4StateSample(
            state=resting_state(),
            position_time_boot_ms=1_000 + 20 * self.sample_index,
            attitude_time_boot_ms=1_002 + 20 * self.sample_index,
            message_skew_s=0.002,
            maximum_receive_age_s=0.001,
        )


def test_shadow_runner_exercises_both_warmup_paths_without_transmission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = runtime_model()
    source = StateSource()
    deadlines: list[float | None] = []

    class Controller:
        prediction_horizon_s = 0.4

        def __init__(self, received_model: RuntimeDynamicsModel) -> None:
            assert received_model is model
            self.model = received_model

        def hold_reference(self, state: np.ndarray, *, exogenous: np.ndarray) -> object:
            assert state.shape == (13,)
            assert exogenous.shape == (0,)
            return object()

        def solve(
            self,
            state: jnp.ndarray,
            reference: object,
            previous_command: jnp.ndarray,
            *,
            applied_command: jnp.ndarray,
            warm_start: NMPCWarmStart | None,
            deadline_s: float | None,
        ) -> SimpleNamespace:
            assert state.shape == (13,)
            assert reference is not None
            np.testing.assert_allclose(applied_command, previous_command)
            deadlines.append(deadline_s)
            return SimpleNamespace(
                status=SimpleNamespace(value="converged"),
                command_usable=True,
                used_fallback=False,
                command=previous_command,
                warm_start=NMPCWarmStart(np.tile(previous_command, (2, 1))),
                diagnostics=SimpleNamespace(
                    solve_time_s=0.01,
                    iterations=1,
                    maximum_validity_utilization=0.1,
                    maximum_command_bound_violation=0.0,
                    support_filter_mode=SimpleNamespace(value="nominal_safe"),
                    support_filter_applied=False,
                    support_command_fraction=1.0,
                    next_step_mean_validity_utilization=0.1,
                    next_step_robust_validity_utilization=0.1,
                    current_angular_rate_energy=0.0,
                    next_step_angular_rate_energy=0.0,
                    support_horizon_s=0.2,
                    support_horizon_maximum_robust_validity_utilization=0.1,
                    support_horizon_terminal_robust_validity_utilization=0.1,
                    support_horizon_terminal_angular_rate_energy=0.0,
                ),
            )

    monkeypatch.setattr(px4_nmpc_shadow, "NMPCController", Controller)

    report = run_px4_nmpc_shadow(
        source,
        model,
        np.asarray(hover_control(model.params)),
        sample_count=2,
        telemetry_timeout_s=1.0,
    )

    assert report["mode"] == "read_only_shadow"
    assert report["commands_transmitted"] is False
    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["fallback_count"] == 0
    assert report["summary"]["usable_command_count"] == 2
    assert report["warmup"]["cold"]["command_usable"]
    assert report["warmup"]["warm"]["command_usable"]
    assert len(report["samples"]) == 2
    assert source.sample_index == 3
    assert deadlines == [None, None, 0.2, 0.2]
    assert report["schema_version"] == 5
    assert report["applied_command_source"] == "fixed"
    assert report["summary"]["maximum_applied_command_state_skew_s"] is None
    assert report["summary"]["maximum_estimated_source_clock_lag_s"] == 0.0
    assert report["summary"]["support_filter_mode_counts"] == {"nominal_safe": 2}
    assert report["summary"]["support_filter_applied_count"] == 0
    one_step = report["summary"]["one_step_model_audit"]
    assert one_step["transition_count"] == 2
    assert one_step["evaluated_transition_count"] == 0
    assert one_step["timing_ineligible_transition_count"] == 2
    assert all(
        sample["one_step_model_audit"]["status"] == "timing_ineligible"
        for sample in report["samples"]
    )
    json.dumps(report, allow_nan=False)


def test_shadow_runner_rejects_command_outside_artifact_bounds() -> None:
    model = runtime_model()

    with pytest.raises(ValueError, match="inside the artifact bounds"):
        run_px4_nmpc_shadow(StateSource(), model, np.full(4, 1.1))


def test_shadow_runner_uses_aligned_applied_command_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = runtime_model()
    state_source = StateSource()
    solved_commands: list[np.ndarray] = []

    class CommandSource:
        def __init__(self) -> None:
            self.sample_index = 0

        def sample_nearest(
            self, time_boot_ms: int, *, timeout_s: float
        ) -> PX4AppliedCommandSample:
            assert timeout_s == 1.0
            self.sample_index += 1
            assert time_boot_ms == 1_000 + 20 * self.sample_index
            return PX4AppliedCommandSample(
                command=np.full(4, 0.2 + 0.05 * self.sample_index),
                source_time_us=(1_000 + 20 * self.sample_index) * 1_000,
                mav_mode=145,
                armed=True,
                receive_age_s=0.002,
            )

    command_source = CommandSource()

    class Controller:
        prediction_horizon_s = 0.4

        def __init__(self, received_model: RuntimeDynamicsModel) -> None:
            assert received_model is model
            self.model = received_model

        def hold_reference(self, state: np.ndarray, *, exogenous: np.ndarray) -> object:
            return object()

        def solve(
            self,
            state: jnp.ndarray,
            reference: object,
            previous_command: jnp.ndarray,
            *,
            applied_command: jnp.ndarray,
            warm_start: NMPCWarmStart | None,
            deadline_s: float | None,
        ) -> SimpleNamespace:
            np.testing.assert_allclose(applied_command, previous_command)
            command = np.asarray(applied_command)
            solved_commands.append(command)
            return SimpleNamespace(
                status=SimpleNamespace(value="converged"),
                command_usable=True,
                used_fallback=False,
                command=command,
                warm_start=NMPCWarmStart(np.tile(command, (2, 1))),
                diagnostics=SimpleNamespace(
                    solve_time_s=0.01,
                    iterations=1,
                    maximum_validity_utilization=0.1,
                    maximum_command_bound_violation=0.0,
                    support_filter_mode=SimpleNamespace(value="nominal_safe"),
                    support_filter_applied=False,
                    support_command_fraction=1.0,
                    next_step_mean_validity_utilization=0.1,
                    next_step_robust_validity_utilization=0.1,
                    current_angular_rate_energy=0.0,
                    next_step_angular_rate_energy=0.0,
                    support_horizon_s=0.2,
                    support_horizon_maximum_robust_validity_utilization=0.1,
                    support_horizon_terminal_robust_validity_utilization=0.1,
                    support_horizon_terminal_angular_rate_energy=0.0,
                ),
            )

    monkeypatch.setattr(px4_nmpc_shadow, "NMPCController", Controller)

    report = run_px4_nmpc_shadow(
        state_source,
        model,
        applied_command_source=command_source,
        sample_count=2,
        telemetry_timeout_s=1.0,
    )

    assert report["applied_command_source"] == "telemetry"
    assert report["initial_applied_command"] == pytest.approx([0.25] * 4)
    assert report["samples"][0]["applied_command"] == pytest.approx([0.3] * 4)
    assert report["samples"][1]["applied_command"] == pytest.approx([0.35] * 4)
    assert report["summary"]["maximum_applied_command_state_skew_s"] == 0.0
    assert report["summary"]["maximum_applied_command_receive_age_s"] == 0.002
    assert report["summary"]["all_applied_command_samples_armed"] is True
    assert report["summary"]["applied_command_peak_to_peak"] == pytest.approx(
        [0.05] * 4
    )
    assert command_source.sample_index == 3
    np.testing.assert_allclose(solved_commands[0], [0.25] * 4)
    np.testing.assert_allclose(solved_commands[1], [0.25] * 4)
    np.testing.assert_allclose(solved_commands[2], [0.3] * 4)
    np.testing.assert_allclose(solved_commands[3], [0.35] * 4)


def test_shadow_runner_requires_exactly_one_applied_command_source() -> None:
    model = runtime_model()

    with pytest.raises(ValueError, match="provide either"):
        run_px4_nmpc_shadow(StateSource(), model)


def test_shadow_runner_rejects_misaligned_applied_command_telemetry() -> None:
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

    with pytest.raises(PX4TelemetryError, match="alignment limit"):
        run_px4_nmpc_shadow(
            StateSource(),
            runtime_model(),
            applied_command_source=MisalignedCommandSource(),
            sample_count=1,
            telemetry_timeout_s=1.0,
        )
