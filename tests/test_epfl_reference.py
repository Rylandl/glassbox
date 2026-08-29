from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

import glassbox.epfl_reference as epfl_module
from glassbox.adapter import TrajectoryAdapter
from glassbox.data import save_trajectory_npz
from glassbox.epfl_evaluation import evaluate_epfl_characterization
from glassbox.epfl_reference import (
    EPFLTopoplaneAdapter,
    _TopoplaneStreams,
    _angular_velocity_from_quaternion,
    _build_trajectories,
    _canonical_quaternion,
    fetch_epfl_topoplane_reference,
    topoplane_trajectory_spec,
)


def _streams() -> _TopoplaneStreams:
    time_s = np.arange(0.0, 80.0 + 0.1, 0.2)
    earth_radius_m = 6_378_137.0
    latitude_deg = 46.0 + np.rad2deg(12.0 * time_s / earth_radius_m)
    longitude_deg = np.full_like(time_s, 6.0)
    altitude_m = np.full_like(time_s, 500.0)
    outage = time_s > 55.0
    altitude_m[outage] += 2.0 * (time_s[outage] - 55.0)
    geodetic = np.column_stack((latitude_deg, longitude_deg, altitude_m))
    velocity_ned = np.tile((12.0, 0.0, 0.0), (len(time_s), 1))
    quaternion = np.tile((1.0, 0.0, 0.0, 0.0), (len(time_s), 1))
    control_pwm = np.tile((1500.0, 1500.0, 1400.0, 1500.0), (len(time_s), 1))
    return _TopoplaneStreams(
        nav_time_s=time_s,
        geodetic_deg_m=geodetic,
        velocity_ned_m_s=velocity_ned,
        quaternion_q0q1q2q3=quaternion,
        published_angular_velocity=np.zeros((len(time_s), 3)),
        control_time_s=time_s,
        control_pwm=control_pwm,
        air_time_s=time_s,
        airspeed_m_s=np.full_like(time_s, 12.0),
        barometric_altitude_m=np.full_like(time_s, 330.0),
    )


def test_topoplane_spec_describes_conventional_four_axis_airframe() -> None:
    spec = topoplane_trajectory_spec()

    assert spec.control_roles == ("throttle", "roll", "pitch", "yaw")
    assert spec.control_names == ("throttle", "aileron", "elevator", "rudder")
    assert spec.exogenous_roles == ("airspeed",)
    assert spec.vehicle.fixed_states == {
        "airframe_layout": "conventional_tail",
        "surface_layout": "aileron_elevator_rudder",
        "surface_mixing": "independent",
    }
    assert isinstance(EPFLTopoplaneAdapter(), TrajectoryAdapter)


def test_scalar_first_quaternion_reorder_and_rate_sign() -> None:
    time_s = np.asarray((0.0, 0.2, 0.4))
    source = np.column_stack(
        (
            np.cos(time_s / 2.0),
            np.zeros_like(time_s),
            np.zeros_like(time_s),
            np.sin(time_s / 2.0),
        )
    )

    canonical = _canonical_quaternion(source)
    angular_velocity = _angular_velocity_from_quaternion(time_s, canonical)

    np.testing.assert_allclose(canonical[:, 0], np.cos(time_s / 2.0))
    np.testing.assert_allclose(canonical[:, 3], -np.sin(time_s / 2.0))
    np.testing.assert_allclose(angular_velocity[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(angular_velocity[:, 2], -1.0, atol=0.01)


def test_build_trajectories_excludes_navigation_drift_and_types_inputs(tmp_path) -> None:
    source = tmp_path / epfl_module.TOPOPLANE_FILENAME
    trajectories = _build_trajectories(
        _streams(), source_path=source, checksum="fixture-md5"
    )

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert 50.0 < trajectory.time_s[-1] < 65.0
    assert trajectory.spec == topoplane_trajectory_spec()
    assert np.allclose(trajectory.controls, (0.4, 0.0, 0.0, 0.0))
    assert np.allclose(trajectory.exogenous, 12.0)
    assert np.allclose(trajectory.states[:, 3:6], (12.0, 0.0, 0.0))
    np.testing.assert_allclose(trajectory.states[:, 6], 1.0)
    np.testing.assert_allclose(trajectory.states[:, 10:13], 0.0)
    assert trajectory.states[-1, 0] > 600.0
    assert trajectory.provenance["evaluation_limitations"]["single_flight"] is True
    assert trajectory.provenance["quality"]["forward_velocity_alignment"][
        "median"
    ] > 0.999


def test_adapter_returns_longest_segment_for_minimal_protocol(tmp_path, monkeypatch) -> None:
    source = tmp_path / epfl_module.TOPOPLANE_FILENAME
    source.write_bytes(b"fixture")
    monkeypatch.setattr(epfl_module, "_read_topoplane_streams", lambda path: _streams())

    adapter = EPFLTopoplaneAdapter(verify_checksum=False)
    trajectory = adapter.load(source)
    inventory = adapter.inspect(source)

    assert trajectory.time_s[-1] > 50.0
    assert inventory["dynamics_ready"] is True
    assert inventory["checksum_matches_pinned_snapshot"] is False
    assert inventory["quality"]["published_angular_velocity_max_abs"] == 0.0


def test_fetch_verifies_and_reuses_pinned_bag(tmp_path, monkeypatch) -> None:
    payload = b"pinned EPFL bag bytes"
    digest = hashlib.md5(payload).hexdigest()
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(epfl_module, "TOPOPLANE_SIZE_BYTES", len(payload))
    monkeypatch.setattr(epfl_module, "TOPOPLANE_MD5", digest)
    monkeypatch.setattr(epfl_module.urllib.request, "urlopen", fake_urlopen)

    first = fetch_epfl_topoplane_reference(tmp_path, timeout_s=5.0)
    second = fetch_epfl_topoplane_reference(tmp_path, timeout_s=5.0)

    assert first == second == tmp_path / epfl_module.TOPOPLANE_FILENAME
    assert first.read_bytes() == payload
    assert len(calls) == 1


def test_characterization_evaluator_preserves_same_flight_limit(tmp_path) -> None:
    trajectory = _build_trajectories(
        _streams(),
        source_path=tmp_path / epfl_module.TOPOPLANE_FILENAME,
        checksum="fixture-md5",
    )[0]
    trajectory_path = tmp_path / "segment.npz"
    save_trajectory_npz(trajectory, trajectory_path)
    metrics = {
        "position_rmse_m": 1.0,
        "velocity_rmse_m_s": 1.0,
        "attitude_rmse_deg": 1.0,
        "angular_velocity_rmse_rad_s": 1.0,
    }

    def write_report(name: str, model_class: str, scale: float) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": {"platform": "fixedwing"},
                    "configuration": {"model_class": model_class},
                    "split": {
                        "mode": (
                            "chronological_segments_within_source_group_characterization"
                        ),
                        "independent_source_group_holdout": False,
                        "training_flights": [{"path": str(trajectory_path)}],
                        "validation_flights": [{"path": str(trajectory_path)}],
                    },
                    "models": {
                        "learned_lag": {
                            "fit": {
                                "initial_loss": 1.0,
                                "final_loss": 0.1,
                                "loss_reduction": 10.0,
                                "wall_time_s": 1.0,
                            },
                            "validation": {
                                "aggregate": {
                                    "horizon_rollouts": {
                                        label: {
                                            key: value * scale
                                            for key, value in metrics.items()
                                        }
                                        for label in ("0.2s", "0.5s", "1s", "2s")
                                    },
                                    "full_rollout": {},
                                }
                            },
                        }
                    },
                }
            )
        )
        return path

    structured = write_report("structured", "structured", 2.0)
    residual = write_report("residual", "structured_residual", 1.0)

    report = evaluate_epfl_characterization(structured, residual)

    assert report["selected_model"] == "structured_residual"
    assert report["can_promote_model"] is False
    assert report["protocol"]["independent_source_group_holdout"] is False
    assert report["protocol"]["requested_and_effective_horizons"]["0.5s"][
        "effective_s"
    ] == pytest.approx(0.4)
