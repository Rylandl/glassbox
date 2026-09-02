import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

import glassbox.io.nanodrone_reference as benchmark_module
from glassbox.core.adapter import TrajectoryAdapter
from glassbox.core.data import save_trajectory_npz
from glassbox.core.dynamics import QUADROTOR_CONTROL_NAMES
from glassbox.core.model_io import save_dynamics_model
from glassbox.core.runtime import runtime_spec_from_trajectory
from glassbox.core.synthetic import initial_parameter_guess
from glassbox.io.nanodrone_reference import (
    BENCHMARK_COMMIT,
    SOURCE_COLUMNS,
    BenchmarkRecording,
    NanoDroneBenchmarkAdapter,
    fetch_nanodrone_benchmark,
)
from glassbox.workflows.nanodrone_evaluation import (
    PUBLISHED_PHYS_PLUS_RES,
    _per_horizon_errors,
    _published_reference_comparison,
    evaluate_nanodrone_benchmark,
    evaluate_nanodrone_model_artifact,
    save_nanodrone_benchmark_report,
)


def _fixture_data() -> np.ndarray:
    data = np.zeros((4, len(SOURCE_COLUMNS)), dtype=np.float64)
    data[:, 0] = (0.0, 0.01, 0.02, 0.03)
    data[:, 1:4] = (
        (1.0, 2.0, 3.0),
        (1.1, 2.2, 3.3),
        (1.2, 2.4, 3.6),
        (1.3, 2.6, 3.9),
    )
    data[:, 4:8] = (0.0, 0.0, 0.0, 1.0)
    data[:, 8:11] = (0.1, 0.2, 0.3)
    data[:, 11:14] = (0.4, 0.5, 0.6)
    data[:, 14:18] = (
        (1000.0, 1100.0, 1200.0, 1300.0),
        (1010.0, 1110.0, 1210.0, 1310.0),
        (1020.0, 1120.0, 1220.0, 1320.0),
        (1030.0, 1130.0, 1230.0, 1330.0),
    )
    data[:, 18:21] = (0.0, 0.0, 9.81)
    return data


def _write_fixture(path: Path, data: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        _fixture_data() if data is None else data,
        delimiter=",",
        header=",".join(SOURCE_COLUMNS),
        comments="",
    )
    return path


def test_adapter_emits_strict_canonical_trajectory(tmp_path) -> None:
    source = _write_fixture(tmp_path / "chirp_20251017_run1.csv")
    adapter = NanoDroneBenchmarkAdapter(verify_checksum=False)

    assert isinstance(adapter, TrajectoryAdapter)
    trajectory = adapter.load(source)

    assert trajectory.time_s.tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(
        trajectory.states[0],
        [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6],
    )
    expected_first_control = (
        np.asarray([1300.0, 1000.0, 1100.0, 1200.0]) / 2500.0
    ) ** 2
    np.testing.assert_allclose(trajectory.controls[0], expected_first_control)
    assert trajectory.controls.shape == (3, 4)
    assert trajectory.control_names == QUADROTOR_CONTROL_NAMES
    assert trajectory.spec.control_roles == QUADROTOR_CONTROL_NAMES
    assert {channel.semantic for channel in trajectory.spec.controls} == {
        "squared_rotor_speed_ratio"
    }
    assert trajectory.spec.observation_source == ("processed_mocap_and_onboard_sensors")
    assert trajectory.spec.observation_roles == (
        "specific_force_x",
        "specific_force_y",
        "specific_force_z",
    )
    np.testing.assert_allclose(
        trajectory.observations, np.tile([0.0, 0.0, 9.81], (4, 1))
    )
    assert trajectory.spec.vehicle.fixed_states["rotor_speed_reference_rad_s"] == 2500.0
    assert trajectory.labels == {
        "profile": "chirp",
        "recording_date": "2025-10-17",
        "replicate": 1,
        "benchmark_split": "train",
        "benchmark": "idsia_nanodrone_sysid",
        "condition": "aggressive_real_flight",
    }
    assert trajectory.provenance["benchmark"]["commit"] == BENCHMARK_COMMIT
    assert trajectory.provenance["source_schema"]["motor_order"] == [
        "front_right",
        "rear_right",
        "rear_left",
        "front_left",
    ]


def test_inspection_reports_source_quality(tmp_path) -> None:
    source = _write_fixture(tmp_path / "melon_20251017_run2.csv")

    inspection = NanoDroneBenchmarkAdapter(verify_checksum=False).inspect(source)

    assert inspection["rows"] == 4
    assert inspection["intervals"] == 3
    assert inspection["duration_s"] == pytest.approx(0.03)
    assert inspection["adapter"]["schema_version"] == 2
    assert inspection["quality"]["sample_rate_hz"] == pytest.approx(100.0)
    assert inspection["labels"]["benchmark_split"] == "test"
    assert inspection["checksum_matches_pinned_snapshot"] is False


def test_adapter_rejects_modified_pinned_recording_by_default(tmp_path) -> None:
    source = _write_fixture(tmp_path / "square_20251017_run1.csv")

    with pytest.raises(ValueError, match="checksum mismatch"):
        NanoDroneBenchmarkAdapter().load(source)


def test_adapter_rejects_nonuniform_timing(tmp_path) -> None:
    data = _fixture_data()
    data[:, 0] = (0.0, 0.01, 0.021, 0.03)
    source = _write_fixture(tmp_path / "random_20251017_run1.csv", data)

    with pytest.raises(ValueError, match="not uniformly sampled"):
        NanoDroneBenchmarkAdapter(verify_checksum=False).load(source)


def test_fetch_verifies_download_and_reuses_valid_file(tmp_path, monkeypatch) -> None:
    payload = b"pinned benchmark bytes"
    recording = BenchmarkRecording(
        "data/train/test.csv",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(benchmark_module, "BENCHMARK_RECORDINGS", (recording,))
    monkeypatch.setattr(benchmark_module.urllib.request, "urlopen", fake_urlopen)

    first = fetch_nanodrone_benchmark(tmp_path)
    second = fetch_nanodrone_benchmark(tmp_path)

    assert first == second == (tmp_path / recording.relative_path,)
    assert first[0].read_bytes() == payload
    assert len(calls) == 1


def test_benchmark_protocol_uses_every_start_and_euclidean_mae(tmp_path) -> None:
    source = _write_fixture(tmp_path / "melon_20251017_run1.csv")
    trajectory = NanoDroneBenchmarkAdapter(verify_checksum=False).load(source)

    report = evaluate_nanodrone_benchmark(
        initial_parameter_guess(), [trajectory], max_horizon_steps=2
    )

    one_step_displacement = np.linalg.norm([0.1, 0.2, 0.3])
    assert report["naive"]["window_count"] == 2
    assert report["naive"]["per_horizon"]["position_mae_m"] == pytest.approx(
        [one_step_displacement, 2.0 * one_step_displacement]
    )
    assert report["naive"]["cumulative_simulation_error"][
        "position_mae_m"
    ] == pytest.approx(3.0 * one_step_displacement)
    assert report["protocol"]["start_stride_steps"] == 1
    assert report["protocol"]["flight_boundaries_crossed"] is False
    assert report["protocol"]["maximum_horizon_steps"] == 2


def test_benchmark_attitude_metric_resolves_quaternion_double_cover() -> None:
    predicted = np.zeros((1, 2, 13), dtype=np.float64)
    target = np.zeros((1, 2, 13), dtype=np.float64)
    predicted[..., 6] = 1.0
    target[..., 6] = -1.0

    errors = _per_horizon_errors(predicted, target)

    assert errors["attitude_mae_rad"] == pytest.approx([0.0])


def test_published_reference_comparison_reports_direct_ratios() -> None:
    selected = {
        step: {"time_s": int(step) * 0.01, **values}
        for step, values in PUBLISHED_PHYS_PLUS_RES["selected_horizons"].items()
    }
    summary = {
        "horizon_steps": list(range(1, 51)),
        "selected_horizons": selected,
        "cumulative_simulation_error": dict(
            PUBLISHED_PHYS_PLUS_RES["cumulative_simulation_error"]
        ),
    }

    comparison = _published_reference_comparison(summary)

    assert comparison is not None
    assert comparison["cumulative_equal_metric_geometric_ratio"] == pytest.approx(1.0)
    assert comparison["beats_every_published_50_step_metric"] is True


def test_saved_model_benchmark_report_round_trip(tmp_path) -> None:
    source = _write_fixture(tmp_path / "melon_20251017_run3.csv")
    trajectory = NanoDroneBenchmarkAdapter(verify_checksum=False).load(source)
    trajectory_path = tmp_path / "melon.npz"
    model_path = tmp_path / "model.json"
    report_path = tmp_path / "report.json"
    save_trajectory_npz(trajectory, trajectory_path)
    save_dynamics_model(
        initial_parameter_guess(),
        model_path,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )

    report = evaluate_nanodrone_model_artifact(
        model_path, [trajectory_path], max_horizon_steps=2
    )
    save_nanodrone_benchmark_report(report, report_path)

    assert report["model_artifact"]["model_type"] == (
        "effective_quadrotor_command_offset_rotational_response_v3"
    )
    assert report["constant_angular_rate_diagnostic"]["window_count"] > 0
    assert report["test_artifacts"] == [str(trajectory_path)]
    assert report_path.read_text().endswith("\n")


def test_benchmark_evaluation_rejects_training_profile(tmp_path) -> None:
    source = _write_fixture(tmp_path / "chirp_20251017_run2.csv")
    trajectory = NanoDroneBenchmarkAdapter(verify_checksum=False).load(source)

    with pytest.raises(ValueError, match="test-split"):
        evaluate_nanodrone_benchmark(
            initial_parameter_guess(), [trajectory], max_horizon_steps=2
        )
