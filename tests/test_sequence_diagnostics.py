"""Evidence contracts: disjoint folds, causal indexing, scaling, and no updates."""

import inspect
from dataclasses import replace

import numpy as np
import pytest

from glassbox.experimental.default_model import fit
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)
from glassbox.experimental.sequence_diagnostics import _fit, _run


def recording(name, seed, *, constant=False):
    rng = np.random.default_rng(seed)
    inputs = np.zeros((79, 1)) if constant else rng.uniform(-1, 1, (79, 1))
    states = np.zeros((80, 1))
    states[0] = seed / 100
    for i in range(len(inputs)):
        states[i + 1] = 0.8 * states[i] + 0.2 * inputs[i]
    return SequenceSegment(name, "whole", states, inputs, 0.05)


def collection(rows):
    return SequenceCollection(
        tuple(rows),
        configuration_id="diagnostic-test",
        state_channels=("x [unitless]",),
        input_channels=("u [requested,unitless]",),
    )


@pytest.fixture(scope="module")
def fitted():
    return fit(collection([recording("train-a", 1), recording("train-b", 2)]))


@pytest.fixture(scope="module")
def evidence(fitted):
    rows = [recording(f"reserved-{i}", i + 10) for i in range(3)]
    report, arrays = _run(fitted, collection(rows))
    return rows, report, arrays


def test_diagnose_is_read_only_and_has_no_tuning_parameters(fitted, evidence):
    rows, report, _ = evidence
    before = fitted.fingerprint()
    assert list(inspect.signature(fitted.diagnose).parameters) == ["recordings"]
    assert fitted.diagnose(collection(rows)) == report
    assert fitted.fingerprint() == before == report["model_fingerprint"]
    report["recipe"]["ridge_fraction"] = 100
    assert fitted.diagnose(collection(rows))["recipe"]["ridge_fraction"] == 0.001
    report["recipe"]["ridge_fraction"] = 0.001


def test_window_indexing_and_foreign_donors_do_not_leak(evidence):
    """The recipe consumes ten transitions; diagnostics look twice as far back."""
    rows, report, arrays = evidence
    assert report["model_history_steps"] == 10
    assert report["extended_history_steps"] == 20
    assert report["windows"] == 3 * 59
    np.testing.assert_array_equal(arrays["past_states"][0], rows[0].states[10:21])
    np.testing.assert_array_equal(arrays["past_inputs"][0], rows[0].inputs[10:20])
    np.testing.assert_array_equal(
        arrays["older_features"][0],
        np.r_[rows[0].states[:10, 0], rows[0].inputs[:10, 0]],
    )
    np.testing.assert_array_equal(arrays["current_inputs"][0], rows[0].inputs[20])
    np.testing.assert_array_equal(arrays["targets"][0], rows[0].states[21])
    for held in range(3):
        prefix = f"fold_{held}_"
        train, test = arrays[prefix + "train"], arrays[prefix + "test"]
        assert not set(train) & set(test)
        assert np.all(arrays["record_index"][train] != held)
        assert np.all(arrays["record_index"][test] == held)
        donor_train, donor_test = (
            arrays[prefix + "donor_train"],
            arrays[prefix + "donor_test"],
        )
        assert set(donor_train) <= set(train) and set(donor_test) <= set(train)
        assert np.all(
            arrays["record_index"][donor_train] != arrays["record_index"][train]
        )


@pytest.mark.parametrize(
    "problem", ["known_id", "known_content", "too_few", "contract", "short"]
)
def test_diagnostic_evidence_must_be_disjoint_and_aligned(fitted, problem):
    rows = [recording(f"fresh-{i}", i + 20) for i in range(3)]
    if problem == "known_id":
        rows[0] = replace(rows[0], recording_id="train-a")
    if problem == "known_content":
        rows[0] = recording("renamed-train", 1)
    if problem == "too_few":
        rows = rows[:2]
    if problem == "short":
        rows[0] = replace(rows[0], states=rows[0].states[:4], inputs=rows[0].inputs[:3])
    supplied = collection(rows)
    if problem == "contract":
        supplied = replace(supplied, configuration_id="different")
    with pytest.raises(ValueError):
        fitted.diagnose(supplied)


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.parametrize("sample_count", [7, 70])
def test_regression_matches_augmented_least_squares_and_unit_conversion(
    quadratic, sample_count
):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(sample_count, 3))
    y = np.column_stack((x[:, 0] ** 2 + 0.2 * x[:, 1], x[:, 2] - x[:, 1]))
    model = _fit(x, y, quadratic=quadratic)
    normalized = (x - model.input_mean) / model.input_scale
    if quadratic:
        i, j = np.triu_indices(3)
        normalized = np.column_stack((normalized, normalized[:, i] * normalized[:, j]))
    z = (normalized - model.feature_mean) / model.feature_scale
    target = (y - model.target_mean) / model.target_scale
    augmented = np.vstack((z, np.sqrt(0.001 * len(x)) * np.eye(z.shape[1])))
    rhs = np.vstack((target, np.zeros((z.shape[1], y.shape[1]))))
    expected = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    np.testing.assert_allclose(model.coefficients, expected, atol=1e-11)
    factors, offsets = np.array([1000.0, 0.001, 2.0]), np.array([10.0, -4.0, 0.0])
    yfactor, yoffset = np.array([0.01, 30.0]), np.array([5.0, 0.0])
    encoded = _fit(x * factors + offsets, y * yfactor + yoffset, quadratic=quadratic)
    np.testing.assert_allclose(
        (encoded.predict(x * factors + offsets) - yoffset) / yfactor,
        model.predict(x),
        atol=1e-9,
    )


def test_zero_input_reference_is_unavailable_not_zero_uncertainty(fitted):
    report = fitted.diagnose(
        collection(
            [recording(f"constant-{i}", 40 + i, constant=True) for i in range(3)]
        )
    )
    for predictor in ("affine", "quadratic"):
        metric = report["aggregate"]["input_predictability"][predictor]
        assert metric["reference_rms"] == [0.0]
        assert metric["remaining_rms_fraction"] == [None]


def test_diagnostic_histories_do_not_cross_segment_gaps(fitted):
    segments = []
    for i in range(3):
        for start in (0, 40):
            x = np.arange(26, dtype=float)[:, None] / 10 + i + start
            segments.append(
                SequenceSegment(
                    f"gapped-{i}", f"part-{start}", x, np.zeros((25, 1)), 0.05, start
                )
            )
    _, arrays = _run(fitted, collection(segments))
    assert set(arrays["source_origins"]) == {20, 21, 22, 23, 24, 60, 61, 62, 63, 64}
    np.testing.assert_allclose(np.diff(arrays["past_states"], axis=1), 0.1)
