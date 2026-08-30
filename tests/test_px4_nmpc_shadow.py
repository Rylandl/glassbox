from __future__ import annotations

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import glassbox.integrations.px4_nmpc_shadow as px4_nmpc_shadow
from glassbox.dynamics import hover_control
from glassbox.integrations.px4 import PX4StateSample
from glassbox.integrations.px4_nmpc_shadow import run_px4_nmpc_shadow
from glassbox.nmpc import NMPCWarmStart
from glassbox.runtime import (
    DirectActuationMap,
    ModelValidityEnvelope,
    RuntimeDynamicsModel,
    RuntimeModelSpec,
)
from glassbox.synthetic import generate_trajectory, resting_state, true_parameters


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

        def hold_reference(
            self, state: np.ndarray, *, exogenous: np.ndarray
        ) -> object:
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
    assert report["schema_version"] == 2
    assert report["summary"]["maximum_estimated_source_clock_lag_s"] == 0.0
    json.dumps(report, allow_nan=False)


def test_shadow_runner_rejects_command_outside_artifact_bounds() -> None:
    model = runtime_model()

    with pytest.raises(ValueError, match="inside the artifact bounds"):
        run_px4_nmpc_shadow(StateSource(), model, np.full(4, 1.1))
