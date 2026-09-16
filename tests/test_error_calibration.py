"""Data separation, finite-sample ranks, vector coverage and stale evidence."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from test_transition_gp import repeated_observation_model

from glassbox.experimental.error_calibration import (
    ErrorCalibration,
    PredictionContract,
    ResidualSamples,
    conformal_quantile,
    fit_error_calibration,
)
from glassbox.experimental.transition_gp import GaussianTransition


def batch(role, features, residuals, contract=None):
    return ResidualSamples(
        contract or PredictionContract("frozen", ("x [m]",), ("next_x [m]",), 0.1),
        features,
        residuals,
        tuple(f"{role}-{i}" for i in range(len(features))),
    )


def example(mode="local", outputs=1):
    contract = PredictionContract(
        "frozen", ("x",), tuple(f"y{i}" for i in range(outputs)), 0.1
    )
    x = np.linspace(-1, 1, 30)[:, None]
    scale = batch("scale", x, np.ones((30, outputs)), contract)
    residual = np.arange(1, 20)[:, None] * np.ones((1, outputs))
    calibration = batch("cal", x[:19], residual, contract)
    return fit_error_calibration(
        scale, calibration, training_sample_ids=("train",), mode=mode
    )


def test_finite_sample_quantile_and_insufficient_evidence():
    assert conformal_quantile(np.arange(1, 20), 0.05) == 19
    assert conformal_quantile(np.arange(1, 21), 0.05) == 20
    assert conformal_quantile([3, 1, 2], 0.5) == 2
    assert np.isinf(conformal_quantile(np.arange(18), 0.05))
    # Ties do not require arbitrary randomization or interpolation.
    assert conformal_quantile(np.ones(20), 0.05) == 1


@pytest.mark.parametrize("alpha", [0, 1, np.nan, -0.1])
def test_invalid_miscoverage(alpha):
    with pytest.raises(ValueError, match="miscoverage"):
        conformal_quantile([1, 2, 3], alpha)


def test_vector_score_counts_rows_and_covers_outputs_jointly():
    fitted = example(outputs=2)
    errors = np.column_stack((np.arange(1, 20), np.arange(19, 0, -1)))
    scale = batch("scale", np.zeros((30, 1)), np.ones((30, 2)), fitted.contract)
    calibration = batch("cal", np.zeros((19, 1)), errors, fitted.contract)
    result = fit_error_calibration(scale, calibration, training_sample_ids=("train",))
    np.testing.assert_array_equal(result.calibration_scores, np.max(errors, axis=1))
    assert result.quantile == 19
    bounds = result.interval([0], [2, -3], contract=result.contract)
    np.testing.assert_array_equal(bounds.lower, [-17, -22])
    np.testing.assert_array_equal(bounds.upper, [21, 16])


def test_local_scale_uses_reserved_errors_and_does_not_add_gp_variance():
    x = np.r_[np.full(30, -1), np.full(30, 1)][:, None]
    errors = np.where(x < 0, 0.1, 1.0)
    scale = batch("scale", x, errors)
    calibration = batch("cal", x, errors * 2)
    result = fit_error_calibration(
        scale, calibration, training_sample_ids=("train",), neighbors=12
    )
    np.testing.assert_allclose(
        result.error_scale([[-1], [1]], contract=result.contract), [[0.1], [1]]
    )
    assert result.quantile == 2
    changed = fit_error_calibration(
        scale,
        replace(calibration, residuals=errors * 3),
        training_sample_ids=("train",),
        neighbors=12,
    )
    # Calibration labels change the order statistic, never the scale fit.
    np.testing.assert_array_equal(
        result.error_scale(x, contract=result.contract),
        changed.error_scale(x, contract=result.contract),
    )
    assert changed.quantile == pytest.approx(3)


def test_overlap_rejected_across_every_data_role():
    x = np.zeros((20, 1))
    scale, calibration = batch("scale", x, x), batch("cal", x, x)
    with pytest.raises(ValueError, match="disjoint"):
        fit_error_calibration(scale, scale, training_sample_ids=("train",))
    for overlapping in (scale.sample_ids[0], calibration.sample_ids[0]):
        with pytest.raises(ValueError, match="disjoint"):
            fit_error_calibration(
                scale, calibration, training_sample_ids=(overlapping,)
            )


def test_units_timing_revision_and_history_contract_invalidate_evidence():
    result = example()
    for contract in (
        replace(result.contract, model_id="refitted"),
        replace(result.contract, dt_s=0.2),
        replace(result.contract, input_names=("x [cm]",)),
        replace(result.contract, output_names=("next_x [cm]",)),
    ):
        with pytest.raises(ValueError, match="stale"):
            result.interval([0], [0], contract=contract)


@pytest.mark.parametrize("mode", ["global", "local"])
def test_roundtrip_and_arbitrary_batch_shape(mode, tmp_path):
    result = example(mode)
    dest = tmp_path / "evidence.npz"
    result.save(dest)
    loaded = ErrorCalibration.load(dest)
    assert loaded.contract == result.contract
    assert loaded.training_sample_ids == result.training_sample_ids
    features = np.zeros((2, 3, 1))
    left = result.interval(features, np.ones_like(features), contract=result.contract)
    right = loaded.interval(features, np.ones_like(features), contract=loaded.contract)
    np.testing.assert_array_equal(left, right)
    assert left.lower.shape == (2, 3, 1)
    assert not loaded.reference_features.flags.writeable


def test_all_zero_errors_and_too_few_calibration_rows_are_well_defined(tmp_path):
    x = np.zeros((3, 1))
    result = fit_error_calibration(
        batch("scale", x, x), batch("cal", x, x), training_sample_ids=("train",)
    )
    bounds = result.interval([0], [0], contract=result.contract)
    assert np.isneginf(bounds.lower[0]) and np.isposinf(bounds.upper[0])
    dest = tmp_path / "small.npz"
    result.save(dest)
    assert np.isinf(ErrorCalibration.load(dest).quantile)
    enough = replace(
        result,
        calibration_scores=np.zeros(19),
        calibration_sample_ids=tuple(f"c{i}" for i in range(19)),
    )
    np.testing.assert_array_equal(
        enough.interval([0], [0], contract=enough.contract), [[0], [0]]
    )


def test_fingerprint_survives_serialization_and_changes_with_prediction(tmp_path):
    model = repeated_observation_model("rbf")
    dest = tmp_path / "model.npz"
    model.save(dest)
    assert model.fingerprint() == GaussianTransition.load(dest).fingerprint()
    for changed in (
        replace(model, dt_s=0.2),
        replace(model, alpha=model.alpha + 0.01),
        replace(model, theta=model.theta + jnp.array([0, 0, 0, 0.1])),
    ):
        assert model.fingerprint() != changed.fingerprint()


def test_bad_shapes_nonfinite_values_and_mutable_inputs():
    x = np.zeros((20, 1))
    data = batch("scale", x, x)
    x[0] = 20
    assert data.features[0, 0] == 0
    with pytest.raises(ValueError, match="finite"):
        replace(data, residuals=np.full((20, 1), np.nan))
    with pytest.raises(ValueError, match="unique"):
        replace(data, sample_ids=("duplicate",) * 20)
    with pytest.raises(ValueError, match="contract"):
        replace(data, features=np.zeros((20, 2)))
    with pytest.raises(ValueError, match="positive"):
        example_result = example()
        replace(example_result, neighbors=0)


def test_split_rank_coverage_by_enumerating_exchangeable_test_assignment():
    # An exact finite exchangeability check: hold out each of 20 distinct scores
    # once. The 19-row 95% order statistic misses only the largest held-out score.
    scores = np.arange(20, dtype=float)
    covered = [
        scores[i] <= conformal_quantile(np.delete(scores, i), 0.05) for i in range(20)
    ]
    assert sum(covered) == 19
