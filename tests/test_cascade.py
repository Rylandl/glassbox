from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glassbox.core.data import (
    RIGID_BODY_STATE_SCHEMA,
    load_trajectory_npz,
    trajectory_windows,
)
from glassbox.core.metrics import (
    kinematic_persistence_windowed_metrics,
    state_error_metrics,
)
from glassbox.io.x8_reference import x8_trajectory_spec

LEVEL_18_M_S = np.array(
    [0.0, 0.0, 100.0, 18.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
X8_VALIDATION_CANDIDATES = (
    Path("artifacts/x8_cascade/canonical/validation/longitudinal_doublet_4.npz"),
    Path("artifacts/x8_reference/canonical/validation/longitudinal_doublet_4.npz"),
)


def _load_x8_validation():
    """Load one real validation maneuver or skip when none is prepared at the current format."""

    for candidate in X8_VALIDATION_CANDIDATES:
        if candidate.exists():
            try:
                return load_trajectory_npz(candidate)
            except ValueError:
                continue
    pytest.skip(
        "prepare the X8 campaign at the current trajectory format, e.g. "
        "`glassbox corpus prepare x8 artifacts/x8_cascade --raw artifacts/x8_reference/raw`"
    )


@pytest.mark.cascade
def test_cascade_canonical_schema_matches_glassbox() -> None:
    pytest.importorskip("cascade")
    from cascade.canonical import CANONICAL_STATE_SCHEMA

    assert CANONICAL_STATE_SCHEMA == RIGID_BODY_STATE_SCHEMA


@pytest.mark.cascade
def test_cascade_plant_exposes_the_x8_control_layout() -> None:
    pytest.importorskip("cascade")
    from glassbox.integrations.cascade import CascadePlant

    plant = CascadePlant()
    first = plant.reset(LEVEL_18_M_S, applied_control=np.array([0.45, 0.0, 0.0]))
    second = plant.step(
        np.array([0.45, 0.05, -0.02]), wind_nwu=np.array([1.0, 0.0, 0.0])
    )

    assert plant.control_names == ("throttle", "aileron", "elevator")
    assert first.state.shape == (13,)
    assert second.time_s == pytest.approx(plant.sample_period_s)
    assert np.all(np.isfinite(second.state))
    assert np.allclose(second.wind_nwu_m_s, [1.0, 0.0, 0.0])
    assert second.applied_control.shape == (3,)


@pytest.mark.cascade
def test_cascade_window_predictions_reproduce_a_cascade_generated_trajectory() -> None:
    pytest.importorskip("cascade")
    from glassbox.integrations.cascade import (
        CascadePlant,
        trajectory_from_plant_samples,
    )
    from glassbox.workflows.benchmarks.cascade_x8 import cascade_window_predictions

    plant = CascadePlant()
    command = np.array([0.45, 0.02, 0.01])
    samples = [plant.reset(LEVEL_18_M_S, applied_control=command)]
    for _ in range(140):
        samples.append(plant.step(command))
    trajectory = trajectory_from_plant_samples(
        samples, x8_trajectory_spec(trusted_wind=True)
    )
    windows = trajectory_windows([trajectory], horizon=8, stride=4)

    predicted = cascade_window_predictions(
        [plant.model], windows, vertical_wind_fractions=[1.0]
    )

    assert predicted.shape == (1, windows.initial_states.shape[0], 9, 13)
    assert np.allclose(predicted[0], windows.target_states, atol=1e-3)


@pytest.mark.cascade
def test_published_x8_variants_are_finite_and_the_documented_one_beats_persistence() -> (
    None
):
    """Regression of the recorded validation result, see docs/validation.md.

    The published model as-is is untrimmed at the flight condition and loses to persistence;
    the documented variant (50 mm forward CG within the paper's stated uncertainty, half the
    campaign's inferred vertical wind) beats it on attitude at half a second.
    """

    pytest.importorskip("cascade")
    from glassbox.workflows.benchmarks.cascade_x8 import (
        cascade_window_predictions,
        x8_variant_models,
    )

    trajectory = _load_x8_validation()
    windows = trajectory_windows([trajectory], horizon=20, stride=1)
    variants, models = x8_variant_models(
        cg_shifts_forward_m=(0.0, 0.05),
        masses_kg=(3.364,),
        yaw_damping=(-0.012,),
        inertia_scales=(1.0,),
        vertical_wind_fractions=(0.5, 1.0),
    )
    fractions = [variant.vertical_wind_fraction for variant in variants]
    documented = next(
        index
        for index, variant in enumerate(variants)
        if variant.cg_shift_forward_m == 0.05 and variant.vertical_wind_fraction == 0.5
    )
    assert any(variant.primary for variant in variants)

    predicted = cascade_window_predictions(
        models, windows, vertical_wind_fractions=fractions
    )
    persistence = kinematic_persistence_windowed_metrics(
        trajectory, horizon_steps=20, stride=1
    )
    metrics = state_error_metrics(
        predicted[documented], windows.target_states, duration_s=0.5
    )

    assert np.all(np.isfinite(predicted))
    assert metrics["attitude_rmse_deg"] < persistence["attitude_rmse_deg"]


@pytest.mark.cascade
def test_residual_regressions_vanish_on_a_cascade_generated_trajectory() -> None:
    pytest.importorskip("cascade")
    from glassbox.integrations.cascade import (
        CascadePlant,
        trajectory_from_plant_samples,
    )
    from glassbox.workflows.benchmarks.cascade_x8 import residual_regressions

    plant = CascadePlant()
    samples = [plant.reset(LEVEL_18_M_S, applied_control=np.array([0.45, 0.0, 0.05]))]
    for step in range(160):
        aileron = 0.15 * np.sin(step / 8.0)
        elevator = 0.05 + 0.08 * np.sin(step / 13.0)
        samples.append(plant.step(np.array([0.45, aileron, elevator])))
    trajectory = trajectory_from_plant_samples(
        samples, x8_trajectory_spec(trusted_wind=True)
    )

    regressions = residual_regressions(
        plant.model, [trajectory], vertical_wind_fraction=1.0
    )

    for channel, item in regressions.items():
        # Central differences of a 40 Hz trajectory leave small discretization residuals; the
        # frames, signs, mass, and inertia plumbing must not add systematic ones. Regression
        # coefficients are not bounded here: with near-zero residuals they are ill-conditioned.
        mean_bound, rms_bound = (0.1, 0.5) if item.unit == "N" else (0.02, 0.05)
        assert abs(item.mean) < mean_bound, (channel, item.mean)
        assert item.rms < rms_bound, (channel, item.rms)


def _flying_wing_model(command_bounds, sample_period_s):
    """A synthetic fixed-wing model whose command box is the plant's own.

    This is deliberately not a model of the Cascade X8: nothing here is fitted
    to it. It exists so the loop has a controller that produces bounded,
    writable commands, which is what the smoke test below is about.
    """

    from glassbox.core.data import (
        Channel,
        TrajectorySpec,
        VehicleConfigurationSpec,
    )
    from glassbox.core.fixedwing_synthetic import true_fixed_wing_parameters
    from glassbox.core.model import (
        DirectActuationMap,
        ExecutableModel,
        ModelValidityEnvelope,
        RuntimeModelSpec,
    )

    minimum, maximum = command_bounds
    controls = (
        Channel(
            name="propulsion_command",
            role="throttle",
            semantic="normalized_command",
            unit="1",
            kind="control",
            minimum=float(minimum[0]),
            maximum=float(maximum[0]),
        ),
        Channel(
            name="elevon_roll_command",
            role="roll",
            semantic="normalized_generalized_command",
            unit="1",
            kind="control",
            frame="FLU",
            minimum=float(minimum[1]),
            maximum=float(maximum[1]),
        ),
        Channel(
            name="elevon_pitch_command",
            role="pitch",
            semantic="normalized_generalized_command",
            unit="1",
            kind="control",
            frame="FLU",
            minimum=float(minimum[2]),
            maximum=float(maximum[2]),
        ),
    )
    spec = TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="simulator_truth",
        channels=controls,
        vehicle=VehicleConfigurationSpec(
            family="fixedwing",
            configuration_id="synthetic_flying_wing",
            controlled_axes=("roll", "pitch"),
        ),
    )
    return ExecutableModel(
        true_fixed_wing_parameters(),
        spec,
        RuntimeModelSpec(
            sample_period_s=sample_period_s,
            validity_envelope=ModelValidityEnvelope(
                body_velocity_center_m_s=(0.0, 0.0, 0.0),
                body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
                angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
                angular_velocity_half_width_rad_s=(100.0, 100.0, 100.0),
            ),
        ),
        DirectActuationMap(spec.controls),
    )


@pytest.mark.cascade
def test_a_cascade_plant_is_a_link_one_control_loop_flies() -> None:
    """The plant, the solver and the loop, with nothing between them."""

    pytest.importorskip("cascade")
    from glassbox.control.fitted import NMPCController
    from glassbox.control.plan import SolverPolicy
    from glassbox.integrations.cascade import CascadePlant
    from glassbox.integrations.loop import LoopSample, VehicleLink, run_control_loop

    plant = CascadePlant()
    plant.reset(LEVEL_18_M_S, applied_control=np.array([0.45, 0.0, 0.0]))

    assert isinstance(plant, VehicleLink)
    assert plant.writable
    assert plant.command_size == 3
    minimum, maximum = plant.command_bounds
    np.testing.assert_allclose(minimum, [0.0, -0.7, -0.7])
    np.testing.assert_allclose(maximum, [1.0, 0.7, 0.7])

    model = _flying_wing_model(plant.command_bounds, plant.sample_period_s)
    controller = NMPCController(
        model,
        policy=SolverPolicy(
            allow_unresolved_parameters=True,
            horizon_steps=4,
            block_count=2,
            maximum_iterations=3,
            line_search_steps=4,
        ),
    )
    exogenous = np.zeros(model.exogenous_size)
    # One solve outside the loop pays the compile, which no control interval can
    # absorb; inside the loop it would be recorded as a deadline miss.
    warmup = plant.read(timeout_s=0.0)
    controller.solve(
        warmup.state,
        controller.hold_reference(warmup.state, exogenous=exogenous),
        warmup.applied_command,
    )
    records: list[LoopSample] = []

    summary = run_control_loop(
        plant,
        controller,
        steps=5,
        reference=lambda observation: controller.hold_reference(
            observation.state, exogenous=exogenous
        ),
        on_sample=records.append,
    )

    assert summary.steps == 5
    assert summary.written_command_count == 5
    assert sum(summary.status_counts.values()) == 5
    assert summary.usable_command_count + summary.fallback_count == 5
    assert summary.usable_command_count >= 1
    assert summary.interval_s == pytest.approx(plant.sample_period_s)
    # Five intervals were written, so the plant advanced five control periods.
    assert plant.snapshot().time_s == pytest.approx(5.0 * plant.sample_period_s)
    for record in records:
        assert record.written
        assert np.all(record.command >= minimum - 1e-9)
        assert np.all(record.command <= maximum + 1e-9)
        assert np.all(np.isfinite(record.observation.state))
        if record.result.used_fallback:
            # A host too slow for this interval is not a reason to stop flying:
            # the interval holds the command the plant was already applying.
            np.testing.assert_allclose(
                record.command,
                record.observation.applied_command,
                atol=1e-9,
            )
    # Each observation is the plant's own clock, one interval apart.
    times = [record.observation.source_time_s for record in records]
    np.testing.assert_allclose(
        np.diff(times), plant.sample_period_s * np.ones(4), atol=1e-9
    )
