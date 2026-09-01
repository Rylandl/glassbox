from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glassbox.data import (
    RIGID_BODY_STATE_SCHEMA,
    load_trajectory_npz,
    trajectory_windows,
)
from glassbox.evaluation import (
    _state_error_metrics,
    kinematic_persistence_windowed_metrics,
)
from glassbox.x8_reference import x8_trajectory_spec

LEVEL_18_M_S = np.array([0.0, 0.0, 100.0, 18.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
        "`glassbox-x8 extract-dataset artifacts/x8_reference/raw artifacts/x8_cascade/canonical`"
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
    second = plant.step(np.array([0.45, 0.05, -0.02]), wind_nwu=np.array([1.0, 0.0, 0.0]))

    assert plant.control_names == ("throttle", "aileron", "elevator")
    assert first.state.shape == (13,)
    assert second.time_s == pytest.approx(plant.sample_period_s)
    assert np.all(np.isfinite(second.state))
    assert np.allclose(second.wind_nwu_m_s, [1.0, 0.0, 0.0])
    assert second.applied_control.shape == (3,)


@pytest.mark.cascade
def test_predict_windows_reproduces_a_cascade_generated_trajectory() -> None:
    pytest.importorskip("cascade")
    from glassbox.integrations.cascade import (
        CascadePlant,
        predict_windows,
        trajectory_from_plant_samples,
    )

    plant = CascadePlant()
    command = np.array([0.45, 0.02, 0.01])
    samples = [plant.reset(LEVEL_18_M_S, applied_control=command)]
    for _ in range(140):
        samples.append(plant.step(command))
    trajectory = trajectory_from_plant_samples(samples, x8_trajectory_spec(trusted_wind=True))
    windows = trajectory_windows([trajectory], horizon=8, stride=4)

    predicted = predict_windows([plant.model], windows, vertical_wind_fractions=[1.0])

    assert predicted.shape == (1, windows.initial_states.shape[0], 9, 13)
    assert np.allclose(predicted[0], windows.target_states, atol=1e-3)


@pytest.mark.cascade
def test_published_x8_variants_are_finite_and_the_documented_one_beats_persistence() -> None:
    """Regression of the recorded validation result, see docs/cascade-x8-validation.md.

    The published model as-is is untrimmed at the flight condition and loses to persistence;
    the documented variant (50 mm forward CG within the paper's stated uncertainty, half the
    campaign's inferred vertical wind) beats it on attitude at half a second.
    """

    from glassbox.integrations.cascade import predict_windows, x8_variant_models

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

    predicted = predict_windows(models, windows, vertical_wind_fractions=fractions)
    persistence = kinematic_persistence_windowed_metrics(
        trajectory, horizon_steps=20, stride_steps=1
    )
    metrics = _state_error_metrics(predicted[documented], windows.target_states, duration_s=0.5)

    assert np.all(np.isfinite(predicted))
    assert metrics["attitude_rmse_deg"] < persistence["attitude_rmse_deg"]


@pytest.mark.cascade
def test_residual_regressions_vanish_on_a_cascade_generated_trajectory() -> None:
    from glassbox.integrations.cascade import (
        CascadePlant,
        residual_regressions,
        trajectory_from_plant_samples,
    )

    plant = CascadePlant()
    samples = [plant.reset(LEVEL_18_M_S, applied_control=np.array([0.45, 0.0, 0.05]))]
    for step in range(160):
        aileron = 0.15 * np.sin(step / 8.0)
        elevator = 0.05 + 0.08 * np.sin(step / 13.0)
        samples.append(plant.step(np.array([0.45, aileron, elevator])))
    trajectory = trajectory_from_plant_samples(samples, x8_trajectory_spec(trusted_wind=True))

    regressions = residual_regressions(plant.model, [trajectory], vertical_wind_fraction=1.0)

    for channel, item in regressions.items():
        # Central differences of a 40 Hz trajectory leave small discretization residuals; the
        # frames, signs, mass, and inertia plumbing must not add systematic ones. Regression
        # coefficients are not bounded here: with near-zero residuals they are ill-conditioned.
        mean_bound, rms_bound = (0.1, 0.5) if item.unit == "N" else (0.02, 0.05)
        assert abs(item.mean) < mean_bound, (channel, item.mean)
        assert item.rms < rms_bound, (channel, item.rms)
