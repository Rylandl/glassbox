"""Saved-evidence integrity and independent geometry without real model calls."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verify_baseline", Path(__file__).parents[1] / "scripts/verify_baseline.py"
)
baseline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(baseline)


def test_exact_checks_dtype_shape_values_and_nan():
    value = np.array([1.0, np.nan], dtype=np.float32)
    baseline.exact(value, value.copy(), "positive")
    for changed in (
        value.astype(np.float64),
        value[None],
        np.array([2.0, np.nan], dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="differ"):
            baseline.exact(changed, value, "altered")


def test_external_manifest_authority_rejects_repaired_local_seal(tmp_path):
    original = np.array([1.0, 2.0], dtype=np.float32)
    np.savez(tmp_path / "payload.npz", values=original)
    manifest = dict(
        format="glassbox-portable-baseline-v1",
        files={"payload.npz": baseline.digest(tmp_path / "payload.npz")},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    authority = baseline.digest(tmp_path / "manifest.json")
    np.savez(tmp_path / "payload.npz", values=original + 0.125)
    with pytest.raises(ValueError, match="archive hash"):
        baseline.verify_integrity(tmp_path, authority)
    manifest["files"]["payload.npz"] = baseline.digest(tmp_path / "payload.npz")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash"):
        baseline.verify_integrity(tmp_path, authority)


def test_manifest_rejects_missing_evidence_after_hash_validation(tmp_path):
    (tmp_path / "data").write_text("fixture")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                format="glassbox-portable-baseline-v1",
                files={"data": baseline.digest(tmp_path / "data")},
            )
        )
    )
    with pytest.raises(ValueError, match="incomplete baseline"):
        baseline.verify_integrity(tmp_path, baseline.digest(tmp_path / "manifest.json"))


def target():
    return dict(
        center_m=[0, 0, 0],
        normal=[-1, 0, 0],
        body_contact_offset_m=[0, 0, 0],
        radius_m_max=0.02,
        axis_error_deg_max=5,
        normal_speed_m_s_range=[0.2, 2.5],
        tangential_speed_m_s_max=0.2,
    )


def states():
    q = [np.sqrt(0.5), 0, np.sqrt(0.5), 0]  # body top axis points +world x
    return np.array(
        [[-0.1, 0, 0, 1, 0, 0, *q, 0, 0, 0], [0.1, 0, 0, 1, 0, 0, *q, 0, 0, 0]]
    )


def test_independent_contact_uses_first_crossing_and_all_thresholds():
    result = baseline.contact_score(states(), [0.0, 0.2], target())
    assert result["contact"] and result["hit"] and all(result["checks"].values())
    assert result["interval"] == 0
    assert result["time_s"] == pytest.approx(0.1, abs=1e-8)
    displaced = states()
    displaced[:, 1] = 0.021
    miss = baseline.contact_score(displaced, [0.0, 0.2], target())
    assert miss["contact"] and not miss["hit"] and not miss["checks"]["bullseye"]


def test_contact_velocity_includes_rotating_offset():
    state = states()[0]
    state[10:13] = [0, 2, 0]
    t = target()
    t["body_contact_offset_m"] = [0, 0, 0.04]
    point, _, velocity = baseline.kinematics(state, t)
    assert point[0] == pytest.approx(state[0] + 0.04)
    assert velocity[2] == pytest.approx(-0.08)


def test_no_crossing_does_not_report_task_success():
    trace = states()
    trace[:, 0] -= 1
    score = baseline.contact_score(trace, [0, 0.2], target())
    assert not score["contact"] and not score["hit"]


def test_contact_comparison_never_relaxes_boolean_thresholds():
    score = baseline.contact_score(states(), [0.0, 0.2], target())
    protocol = dict(
        contact_comparison=dict(
            exact=["contact", "hit", "checks", "interval"],
            rtol=1e-5,
            angle_atol_deg=0.05,
            fraction_atol=5e-4,
            time_atol_s=5e-6,
            speed_atol_m_s=1e-4,
            state_atol=1e-4,
            length_atol_m=1e-5,
        )
    )
    baseline.compare_contact(score, score, protocol)
    altered = dict(score, hit=False)
    with pytest.raises(ValueError, match="contact hit"):
        baseline.compare_contact(altered, score, protocol)
