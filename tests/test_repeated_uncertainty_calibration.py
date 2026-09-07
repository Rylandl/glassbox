import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "calibration",
    Path(__file__).parents[1] / "scripts/calibrate_repeated_uncertainty.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_signed_cross_and_bias_variance_identity():
    errors = np.array([[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0]])
    linear = 2 * errors
    parts = module.decompose(errors, linear)
    np.testing.assert_allclose(
        parts["actual"], parts["linear"] + parts["remainder"] + parts["cross"]
    )
    assert np.trace(parts["cross"].mean(axis=0)) < 0
    bv = module.bias_variance(errors)
    np.testing.assert_allclose(
        bv["second_moment"], bv["bias_squared"] + bv["centered_variance"]
    )


def test_sources_disjoint_and_control_identifies_exact_overlap():
    d = module.DESIGN
    groups = [set(d[k]) for k in ("train_seeds", "calibration_seeds", "test_seeds")]
    assert not any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    c = module.scalar_control()
    assert c["sum"] == 2 * c["actual"] and c["remainder"] == c["cross"] == 0


def test_saved_pilot_decomposition_and_provenance():
    import json

    root = (
        Path(__file__).parents[1]
        / "docs/investigations/repeated-uncertainty-calibration"
    )
    report = json.loads((root / "report.json").read_text())
    arrays = np.load(root / "arrays.npz")
    assert arrays["errors"].shape == (6, 2, 2, 12)
    np.testing.assert_allclose(
        arrays["actual"],
        arrays["linear"] + arrays["remainder"] + arrays["cross"],
        atol=2e-12,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        arrays["second_moment"],
        arrays["bias_squared"] + arrays["centered_variance"],
        atol=2e-12,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        arrays["combined"], arrays["empirical"] + arrays["parameter"]
    )
    assert len(report["sources"]) == 20
    assert len({s["sha256"] for s in report["sources"]}) == 20
    assert all(
        r["calibration_independent_group_count"] == [2, 2] for r in report["replicates"]
    )
    assert all(
        r["training_retained_count"] + r["training_omitted_count"] == 150
        for r in report["replicates"]
    )
    assert not report["replicates"][4]["test_validity"][0]["horizons"][1]["both_inside"]
