import hashlib
import io
from pathlib import Path

import numpy as np

import glassbox.arp_reference as arp_module
from glassbox.adapter import TrajectoryAdapter
from glassbox.arp_reference import (
    ARP_CONFIGURATION_ID,
    ARP_REFERENCE_COMMIT,
    ARP_REFERENCE_NAME,
    ARPRecording,
    ARPReferenceAdapter,
    _longest_powered_interval,
    fetch_arp_reference,
)
from glassbox.data import Trajectory, make_trajectory_spec


def _base_trajectory() -> Trajectory:
    states = np.zeros((31, 13), dtype=np.float64)
    states[:, 6] = 1.0
    return Trajectory(
        time_s=np.arange(31, dtype=np.float64) * 0.02,
        states=states,
        controls=np.full((30, 4), 0.25),
        spec=make_trajectory_spec(
            (
                "motor_front_left",
                "motor_front_right",
                "motor_rear_right",
                "motor_rear_left",
            ),
            family="multirotor",
            observation_source="estimated",
            configuration_id=ARP_CONFIGURATION_ID,
        ),
        labels={"profile": "published_sysid", "replicate": 1},
        provenance={"source": "fixture.ulg", "px4": {"filters": {}}},
    )


def test_adapter_applies_opinionated_reference_contract(tmp_path, monkeypatch) -> None:
    source = tmp_path / "log_63_2024-1-8-16-37-54.ulg"
    source.write_bytes(b"fixture")
    captured = {}

    def fake_load(path, *, config):
        captured["path"] = path
        captured["config"] = config
        return _base_trajectory()

    monkeypatch.setattr(arp_module, "load_px4_trajectory", fake_load)
    adapter = ARPReferenceAdapter(verify_checksum=False)

    assert isinstance(adapter, TrajectoryAdapter)
    trajectory = adapter.load(source)

    config = captured["config"]
    assert config.sample_rate_hz == 50.0
    assert config.min_height_m is None
    assert config.only_armed is False
    assert config.only_in_air is False
    assert config.motor_indices is None
    assert config.vehicle_id == ARP_CONFIGURATION_ID
    assert trajectory.labels["benchmark"] == ARP_REFERENCE_NAME
    assert trajectory.labels["recording_date"] == "2024-01-08"
    assert trajectory.labels["source_group"] == (
        f"{ARP_REFERENCE_NAME}:logs_large/{source.name}"
    )
    assert trajectory.provenance["adapter"] == {
            "name": "arp_px4_ulog_reference",
            "schema_version": 2,
    }
    reference = trajectory.provenance["reference_dataset"]
    assert reference["commit"] == ARP_REFERENCE_COMMIT
    assert reference["relative_path"].endswith(source.name)
    assert trajectory.spec.vehicle.configuration_id == ARP_CONFIGURATION_ID


def test_inspection_adds_pinned_identity(tmp_path, monkeypatch) -> None:
    source = tmp_path / "log_64_2024-1-8-16-39-44.ulg"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        arp_module,
        "inspect_ulog",
        lambda path: {"path": str(path), "topics": [], "dropout_count": 0},
    )

    inspection = ARPReferenceAdapter(verify_checksum=False).inspect(source)

    assert inspection["reference_dataset"]["relative_path"].endswith(source.name)
    assert inspection["checksum_matches_pinned_snapshot"] is False
    assert inspection["sha256"] == hashlib.sha256(b"fixture").hexdigest()


def test_reference_adapter_selects_longest_powered_interval() -> None:
    base = _base_trajectory()
    states = np.resize(base.states, (9, 13))
    states[:, 6] = 1.0
    controls = np.asarray(
        [
            [0.0] * 4,
            [0.3] * 4,
            [0.3] * 4,
            [0.0] * 4,
            [0.4] * 4,
            [0.4] * 4,
            [0.4] * 4,
            [0.0] * 4,
        ]
    )
    trajectory = Trajectory(
        time_s=np.arange(9) * 0.1,
        states=states,
        controls=controls,
        spec=base.spec,
    )

    selected = _longest_powered_interval(
        trajectory, minimum_duration_s=0.1
    )

    assert np.isclose(selected.time_s[-1], 0.3)
    np.testing.assert_allclose(selected.controls, 0.4)
    assert selected.provenance["reference_powered_interval"][
        "candidate_interval_count"
    ] == 2


def test_fetch_verifies_download_and_reuses_valid_file(tmp_path, monkeypatch) -> None:
    payload = b"pinned ARP ULog bytes"
    recording = ARPRecording(
        "logs_large/reference.ulg",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        1,
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(arp_module, "ARP_RECORDINGS", (recording,))
    monkeypatch.setattr(arp_module.urllib.request, "urlopen", fake_urlopen)

    first = fetch_arp_reference(tmp_path)
    second = fetch_arp_reference(tmp_path)

    assert first == second == (tmp_path / recording.relative_path,)
    assert first[0].read_bytes() == payload
    assert len(calls) == 1
