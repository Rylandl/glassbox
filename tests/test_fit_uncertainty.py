import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "fit_uncertainty",
    Path(__file__).parents[1] / "scripts/evaluate_fit_uncertainty.py",
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_calibration_separates_bias_from_variation_and_omits_unknown_directions():
    # Arbitrarily large errors in the unresolved second coordinate must not
    # enter a standardized error computed from the resolved first coordinate.
    vectors = np.array([[2.0, 1000.0], [4.0, -1000.0]])
    result = study.parameter_summary(
        vectors,
        np.array([np.diag([1.0, 0.0])] * 2),
        [np.array([[1.0], [0.0]])] * 2,
        [np.array([1.0])] * 2,
        np.zeros(2),
    )

    np.testing.assert_allclose(result["resolved_error_over_variance"], [4.0, 16.0])
    np.testing.assert_allclose(
        result["resolved_centered_variation_over_variance"], [2.0, 2.0]
    )
    np.testing.assert_allclose(result["mean_error"], [3.0, 0.0])
    np.testing.assert_allclose(result["centered_sample_variance"], [2.0, 2_000_000.0])


def test_unresolved_information_provides_no_variance_calibration_measurement():
    result = study.parameter_summary(
        np.array([[1.0], [-1.0]]),
        np.zeros((2, 1, 1)),
        [np.empty((1, 0))] * 2,
        [np.empty(0)] * 2,
        np.zeros(1),
    )

    assert result["resolved_error_over_variance"] == []
    assert result["resolved_centered_variation_over_variance"] == []
    assert result["centered_sample_variance"] == [2.0]


def test_study_refuses_to_mix_existing_results_with_a_new_design(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("previous study")

    with pytest.raises(ValueError, match="output directory must be empty"):
        study.run(tmp_path, 1, ["clean"])

    assert report.read_text() == "previous study"
