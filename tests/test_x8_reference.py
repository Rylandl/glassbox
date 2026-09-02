from __future__ import annotations

import hashlib
import io
import urllib.request

import numpy as np

import glassbox.io.x8_reference as x8_module
from glassbox.core.data import Trajectory, save_trajectory_npz
from glassbox.core.fixedwing_synthetic import true_fixed_wing_parameters
from glassbox.core.model_io import save_dynamics_model
from glassbox.core.runtime import runtime_spec_from_trajectory
from glassbox.io.x8_reference import (
    X8Recording,
    X8ReferenceAdapter,
    fetch_x8_reference,
    x8_trajectory_spec,
)
from glassbox.workflows.x8_evaluation import evaluate_x8_reference_models


def _write_fixture(path) -> np.ndarray:
    data = np.zeros((4, 41), dtype=np.float64)
    data[:, 0] = np.arange(4) * 0.025
    data[:, 1] = (0.10, 0.11, 0.12, 0.13)
    data[:, 2] = (-0.20, -0.19, -0.18, -0.17)
    data[:, 3] = (0.40, 0.41, 0.42, 0.43)
    data[:, 13:16] = (0.1, 0.2, 0.3)
    data[:, 16:19] = (10.0, 2.0, -1.0)
    data[:, 19:22] = (10.0, 2.0, -1.0)
    data[:, 22:25] = (4.0, -3.0, -0.5)
    data[:, 32] = np.arange(4) * 0.25
    data[:, 33] = np.arange(4) * 0.05
    data[:, 34] = np.arange(4) * -0.025
    np.savetxt(path, data, delimiter=",")
    return data


def test_x8_adapter_emits_typed_flying_wing_trajectory(tmp_path) -> None:
    source = tmp_path / "lateral_121_1.csv"
    data = _write_fixture(source)

    trajectory = X8ReferenceAdapter(verify_checksum=False).load(source)

    assert trajectory.spec == x8_trajectory_spec()
    assert trajectory.spec.vehicle.fixed_states["airframe_layout"] == "flying_wing"
    assert trajectory.spec.control_roles == ("throttle", "roll", "pitch")
    assert trajectory.spec.control_names == ("throttle", "aileron", "elevator")
    np.testing.assert_allclose(trajectory.controls[0], data[0, (3, 2, 1)])
    np.testing.assert_allclose(trajectory.states[0, 3:6], (10.0, -2.0, 1.0))
    np.testing.assert_allclose(trajectory.states[0, 6:10], (1.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(trajectory.states[0, 10:13], (0.1, -0.2, -0.3))
    np.testing.assert_allclose(trajectory.states[1, 0:3], (0.25, -0.05, 0.025))
    assert trajectory.labels["benchmark_split"] == "training"
    assert trajectory.labels["profile"] == "lateral_121"
    assert trajectory.labels["replicate"] == 1
    assert trajectory.exogenous_size == 3
    assert trajectory.spec.exogenous_roles == (
        "wind_north",
        "wind_west",
        "wind_up",
    )
    np.testing.assert_allclose(trajectory.exogenous[0], (4.0, 3.0, 0.5))
    assert "exogenous" in trajectory.provenance


def test_x8_inspection_audits_source_consistency(tmp_path) -> None:
    source = tmp_path / "lateral_121_1.csv"
    _write_fixture(source)

    inventory = X8ReferenceAdapter(verify_checksum=False).inspect(source)

    assert inventory["intervals"] == 3
    assert np.isclose(inventory["duration_s"], 0.075)
    assert inventory["quality"]["sample_rate_hz"] == 40.0
    assert inventory["quality"]["maximum_body_velocity_consistency_error_m_s"] < 1e-12
    assert inventory["checksum_matches_pinned_snapshot"] is False


def test_x8_adapter_can_exclude_the_wind_estimate_for_ablation(tmp_path) -> None:
    source = tmp_path / "lateral_121_1.csv"
    _write_fixture(source)

    trajectory = X8ReferenceAdapter(
        verify_checksum=False,
        use_trusted_wind_estimate=False,
    ).load(source)

    assert trajectory.spec.exogenous_roles == ()
    assert trajectory.exogenous_size == 0
    assert trajectory.provenance["excluded_exogenous"]["prediction_policy"] == (
        "not_supplied_to_model"
    )


def test_x8_fetch_verifies_pinned_files(tmp_path, monkeypatch) -> None:
    readme_payload = b"pinned x8 readme"
    csv_payload = b"pinned x8 csv"
    recording = X8Recording(
        filename="lateral_121_1.csv",
        split="training",
        file_id=20,
        size_bytes=len(csv_payload),
        md5=hashlib.md5(csv_payload).hexdigest(),
    )
    monkeypatch.setattr(x8_module, "X8_RECORDINGS", (recording,))
    monkeypatch.setattr(x8_module, "X8_README_FILE_ID", 10)
    monkeypatch.setattr(x8_module, "X8_README_SIZE_BYTES", len(readme_payload))
    monkeypatch.setattr(
        x8_module, "X8_README_MD5", hashlib.md5(readme_payload).hexdigest()
    )

    def fake_urlopen(request, timeout):
        assert timeout == 5.0
        payload = readme_payload if request.full_url.endswith("/10") else csv_payload
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    paths = fetch_x8_reference(tmp_path, timeout_s=5.0)

    assert paths == (tmp_path / "training" / recording.filename,)
    assert (tmp_path / x8_module.X8_README_FILENAME).read_bytes() == readme_payload
    assert paths[0].read_bytes() == csv_payload


def test_x8_evaluation_requires_and_scores_upstream_validation(tmp_path) -> None:
    source = tmp_path / "lateral_121_1.csv"
    _write_fixture(source)
    trajectory = X8ReferenceAdapter(verify_checksum=False).load(source)
    trajectory = Trajectory(
        time_s=trajectory.time_s,
        states=trajectory.states,
        controls=trajectory.controls,
        exogenous=trajectory.exogenous,
        spec=trajectory.spec,
        labels={**trajectory.labels, "benchmark_split": "validation"},
        provenance=trajectory.provenance,
    )
    trajectory_path = tmp_path / "validation.npz"
    model_path = tmp_path / "model.json"
    save_trajectory_npz(trajectory, trajectory_path)
    save_dynamics_model(
        true_fixed_wing_parameters(),
        model_path,
        input_spec=x8_trajectory_spec(),
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )

    report = evaluate_x8_reference_models(
        {"structured": model_path},
        [trajectory_path],
        horizons_s=(0.025,),
    )

    assert report["protocol"]["split"] == "upstream_validation"
    assert report["dataset"]["validation_trajectory_count"] == 1
    assert "0.025s" in report["models"]["structured"]["aggregate"]["horizon_rollouts"]
    assert np.isfinite(report["models"]["structured"]["score_vs_kinematic_persistence"])
