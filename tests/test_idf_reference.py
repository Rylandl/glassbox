import hashlib
import io
import urllib.request
import zipfile
import zlib
from types import SimpleNamespace

import numpy as np

import glassbox.io.idf_reference as idf_module
from glassbox.core.adapter import TrajectoryAdapter
from glassbox.core.data import (
    Trajectory,
    make_trajectory_spec,
    save_trajectory_npz,
    specific_force_observation_channels,
)
from glassbox.io.idf_reference import (
    IDF_CONFIGURATION_ID,
    IDF_REFERENCE_NAME,
    IDFFixedWingAdapter,
    IDFRecording,
    extract_idf_ulogs,
    fetch_idf_archive,
    idf_corpus_report,
    save_idf_corpus_report,
)


def _trajectory(intervals: int) -> Trajectory:
    states = np.zeros((intervals + 1, 13), dtype=np.float64)
    states[:, 6] = 1.0
    return Trajectory(
        time_s=np.arange(intervals + 1, dtype=np.float64) * 0.02,
        states=states,
        controls=np.full((intervals, 4), 0.25),
        spec=make_trajectory_spec(
            ("throttle", "aileron", "elevator", "rudder"),
            family="fixedwing",
            observation_source="estimated",
            configuration_id=IDF_CONFIGURATION_ID,
            observations=specific_force_observation_channels("fixture_imu"),
        ),
        observations=np.tile([0.1, 0.2, 9.7], (intervals + 1, 1)),
        labels={"profile": "waypoint_circuit"},
        provenance={"source": "fixture.ulg", "px4": {"filters": {}}},
    )


def test_adapter_preserves_every_dropout_safe_segment(tmp_path, monkeypatch) -> None:
    source = tmp_path / "log_67_2025-8-21-14-24-48.ulg"
    source.write_bytes(b"fixture")
    captured = {}

    def fake_load(path, *, config):
        captured["path"] = path
        captured["config"] = config
        return (_trajectory(20), _trajectory(30))

    monkeypatch.setattr(idf_module, "load_px4_trajectories", fake_load)
    adapter = IDFFixedWingAdapter(verify_checksum=False)

    assert isinstance(adapter, TrajectoryAdapter)
    trajectories = adapter.load_all(source)

    config = captured["config"]
    assert config.platform == "fixedwing"
    assert config.sample_rate_hz == 50.0
    assert config.max_gap_s == 0.2
    assert config.min_duration_s == 10.0
    assert config.vehicle_id == IDF_CONFIGURATION_ID
    assert len(trajectories) == 2
    assert [item.labels["segment"] for item in trajectories] == [1, 2]
    assert [item.labels["source_group"] for item in trajectories] == [
        "idf_session_01",
        "idf_session_01",
    ]
    assert trajectories[0].labels["benchmark"] == IDF_REFERENCE_NAME
    assert trajectories[0].labels["session"] == 1
    assert trajectories[0].provenance["adapter"]["schema_version"] == 2
    np.testing.assert_allclose(
        trajectories[0].observations,
        np.tile([0.1, 0.2, 9.7], (21, 1)),
    )
    reference = trajectories[0].provenance["reference_dataset"]
    assert reference["published_flights"] == [1, 2, 3, 4, 5, 7, 8, 34]
    assert reference["missing_raw_flights"] == [117, 118, 119, 120]
    assert len(adapter.load(source).controls) == 30


def test_fetch_archive_verifies_and_reuses_snapshot(tmp_path, monkeypatch) -> None:
    payload = b"pinned IDF archive bytes"
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(idf_module, "IDF_ARCHIVE_SIZE_BYTES", len(payload))
    monkeypatch.setattr(idf_module, "IDF_ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    first = fetch_idf_archive(tmp_path)
    second = fetch_idf_archive(tmp_path)

    assert first == second
    assert first.read_bytes() == payload
    assert len(calls) == 1


def test_extract_ulogs_selects_only_pinned_raw_members(tmp_path, monkeypatch) -> None:
    payload = b"raw ULog fixture"
    recording = IDFRecording(
        "reference.ulg",
        len(payload),
        zlib.crc32(payload),
        1,
        "2025-08-21",
        (1,),
    )
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(recording.archive_path, payload)
        output.writestr("Holybro Pixhawk/processed/ignored.csv", b"ignored")
    monkeypatch.setattr(idf_module, "IDF_RECORDINGS", (recording,))

    first = extract_idf_ulogs(archive, tmp_path / "raw")
    second = extract_idf_ulogs(archive, tmp_path / "raw")

    assert first == second == (tmp_path / "raw" / recording.filename,)
    assert first[0].read_bytes() == payload
    assert not (tmp_path / "raw" / "ignored.csv").exists()


def test_corpus_report_records_coverage_and_excitation(tmp_path, monkeypatch) -> None:
    recording = IDFRecording(
        "reference.ulg",
        1,
        0,
        1,
        "2025-08-21",
        (1,),
    )
    monkeypatch.setattr(idf_module, "IDF_RECORDINGS", (recording,))
    monkeypatch.setattr(
        idf_module,
        "ULog",
        lambda *_args, **_kwargs: SimpleNamespace(
            start_timestamp=0, last_timestamp=1_000_000
        ),
    )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / recording.filename).write_bytes(b"x")
    source = _trajectory(25)
    trajectory = Trajectory(
        time_s=source.time_s,
        states=source.states,
        controls=source.controls,
        spec=source.spec,
        observations=source.observations,
        labels={**source.labels, "session": 1},
        provenance=source.provenance,
    )
    trajectory_path = tmp_path / "segment.npz"
    save_trajectory_npz(trajectory, trajectory_path)

    report = idf_corpus_report((trajectory_path,), raw_root)
    report_path = tmp_path / "report.json"
    save_idf_corpus_report(report, report_path)

    assert report["canonical"]["trajectory_count"] == 1
    assert report["canonical"]["duration_s"] == 0.5
    assert report["canonical"]["retained_duration_ratio"] == 0.5
    assert report["canonical"]["control_statistics"]["names"] == [
        "throttle",
        "aileron",
        "elevator",
        "rudder",
    ]
    assert report["sessions"][0]["segment_count"] == 1
    assert report_path.read_text().endswith("\n")
