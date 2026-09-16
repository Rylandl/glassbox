"""Pinned data preparation for cross-platform sequence experiments.

X8 retains upstream offline processing. ARP uses publication-time holds of its
logged estimator outputs; this does not turn them into independent ground truth.
No airframe dynamics, force terms or learned-model results enter preparation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pyulog import ULog

from glassbox.core.geometry import quaternion_to_rotation_matrices
from glassbox.experimental.causal_sampling import causal_hold
from glassbox.io.arp_reference import ARP_RECORDINGS
from glassbox.io.x8_reference import X8_RECORDINGS, X8ReferenceAdapter


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_x8(raw):
    records = []
    for recording in X8_RECORDINGS:
        path = raw / recording.relative_path
        adapter = X8ReferenceAdapter()
        trajectory = adapter.load(path)
        rotation = quaternion_to_rotation_matrices(trajectory.states[:, 6:10])
        states = np.column_stack(
            (
                trajectory.states[:, 3:6],
                trajectory.states[:, 10:13],
                rotation.reshape(-1, 9),
            )
        )
        role = (
            "evaluation"
            if recording.split == "validation"
            else "development"
            if recording.replicate == 3
            else "train"
        )
        records.append(
            dict(
                name=path.stem,
                role=role,
                states=states,
                inputs=trajectory.controls,
                time_s=trajectory.time_s,
                dt_s=0.025,
                metadata=dict(
                    source=str(path.resolve()),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected_md5=recording.md5,
                    profile=recording.profile,
                    replicate=recording.replicate,
                    upstream_split=recording.split,
                    processing="upstream manual alignment and 40 Hz resampling",
                    inputs=[
                        "normalized throttle",
                        "aileron angle rad",
                        "elevator angle rad",
                    ],
                    derived_wind_used=False,
                ),
            )
        )
    return records


def _column(data, names):
    return np.column_stack([data[name] for name in names]).astype(float)


def load_arp_record(path, recording):
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if raw_hash != recording.sha256 or path.stat().st_size != recording.size_bytes:
        raise ValueError("ARP checksum or size mismatch")
    ulog = ULog(str(path))
    if ulog.file_corruption:
        raise ValueError("corrupt ULog")
    datasets = {(d.name, d.multi_id): d.data for d in ulog.data_list}
    topics = (
        "vehicle_local_position",
        "vehicle_angular_velocity",
        "vehicle_attitude",
        "actuator_motors",
    )
    fields = (
        ("vx", "vy", "vz"),
        tuple(f"xyz[{i}]" for i in range(3)),
        tuple(f"q[{i}]" for i in range(4)),
        tuple(f"control[{i}]" for i in range(4)),
    )
    streams, reset_times = [], []
    for topic, names in zip(topics, fields):
        data = datasets[(topic, 0)]
        values = _column(data, names)
        publication = np.asarray(data["timestamp"], dtype=float) * 1e-6
        sample = (
            np.asarray(data.get("timestamp_sample", data["timestamp"]), dtype=float)
            * 1e-6
        )
        valid = (
            np.isfinite(values).all(1)
            & (publication > 0)
            & (sample > 0)
            & (sample <= publication)
        )
        if topic == "vehicle_local_position":
            for field in ("v_xy_valid", "v_z_valid"):
                valid &= np.asarray(data[field], dtype=bool)
        if topic == "vehicle_attitude":
            valid &= np.linalg.norm(values, axis=1) > 0.5
        for counter in ("vxy_reset_counter", "vz_reset_counter", "quat_reset_counter"):
            if counter in data:
                reset_times.extend(
                    publication[1:][np.diff(data[counter].astype(int)) != 0].tolist()
                )
        if np.any(np.diff(publication) < 0):
            raise ValueError(f"nonmonotonic publication timestamps in {topic}")
        streams.append(
            dict(
                topic=topic,
                fields=names,
                time=publication[valid],
                sample=sample[valid],
                values=values[valid],
                raw_indices=np.flatnonzero(valid),
                total_rows=len(values),
            )
        )
    dt = 0.02
    start = np.ceil(max(s["time"][0] for s in streams) / dt)
    stop = np.floor(min(s["time"][-1] for s in streams) / dt)
    grid = np.arange(start, stop + 1) * dt
    held = [
        causal_hold(s["time"], s["values"], grid, maximum_age_s=0.05) for s in streams
    ]
    valid = np.logical_and.reduce([h.valid for h in held])
    powered = held[-1].values.mean(1) > 0.1
    valid &= powered
    for reset in reset_times:
        index = np.searchsorted(grid, reset)
        if index < len(valid):
            valid[index] = False
    boundaries = np.flatnonzero(
        np.diff(np.r_[False, valid, False].astype(int))
    ).reshape(-1, 2)
    if not len(boundaries):
        raise ValueError("no complete powered interval")
    first, last = max(boundaries, key=lambda row: row[1] - row[0])
    if (last - first) * dt < 2:
        raise ValueError("no sufficiently long uninterrupted interval")
    indices = np.column_stack([h.source_indices[first:last] for h in held])
    publication = np.column_stack(
        [s["time"][indices[:, i]] for i, s in enumerate(streams)]
    )
    sample = np.column_stack(
        [s["sample"][indices[:, i]] for i, s in enumerate(streams)]
    )
    source_indices = np.column_stack(
        [s["raw_indices"][indices[:, i]] for i, s in enumerate(streams)]
    )
    selected = [h.values[first:last] for h in held]
    sign = np.array([1.0, -1.0, -1.0])
    quaternions = selected[2] / np.linalg.norm(selected[2], axis=1, keepdims=True)
    rotation = quaternion_to_rotation_matrices(quaternions)
    rotation = rotation * sign[None, :, None] * sign[None, None, :]
    states = np.column_stack(
        (selected[0] * sign, selected[1] * sign, rotation.reshape(-1, 9))
    )
    selected_grid = grid[first:last]
    age = selected_grid[:, None] - publication
    sample_age = selected_grid[:, None] - sample
    return dict(
        name=path.stem,
        role={1: "train", 2: "train", 3: "development", 4: "evaluation"}[
            recording.replicate
        ],
        states=states,
        inputs=selected[3][:-1],
        time_s=selected_grid - selected_grid[0],
        dt_s=dt,
        publication_s=publication,
        sample_s=sample,
        source_indices=source_indices,
        absolute_grid_s=selected_grid,
        metadata=dict(
            source=str(path.resolve()),
            sha256=raw_hash,
            replicate=recording.replicate,
            topics=list(topics),
            fields=[list(f) for f in fields],
            dropped_invalid_source_rows=[
                s["total_rows"] - len(s["time"]) for s in streams
            ],
            maximum_publication_age_s=age.max(0).tolist(),
            mean_publication_age_s=age.mean(0).tolist(),
            maximum_sample_age_s=sample_age.max(0).tolist(),
            maximum_lookahead_s=float(np.max(publication - selected_grid[:, None])),
            publication_clock="ULog topic timestamp; not timestamp_sample",
            reset_events=len(reset_times),
            logger_dropouts=len(ulog.dropouts),
            selection="longest complete powered interval, mean control >0.1; split at estimator resets",
            all_grid_rows=len(grid),
            retained_rows=int(last - first),
            powered_interval_start_s=float(selected_grid[0]),
            processing="publication-time zero-order hold at 50 Hz, maximum age 50 ms; no interpolation or added filtering",
            inputs=[f"actuator_motors.control[{i}]" for i in range(4)],
        ),
    )


def prepare(corpora, destination):
    destination.mkdir(parents=True, exist_ok=False)
    all_records = {
        "x8": load_x8(corpora / "x8/raw"),
        "arp": [
            load_arp_record(corpora / "arp/raw" / r.relative_path, r)
            for r in ARP_RECORDINGS
        ],
    }
    inventory = {}
    for dataset, records in all_records.items():
        folder = destination / dataset
        folder.mkdir()
        inventory[dataset] = []
        for r in records:
            arrays = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
            metadata = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            np.savez_compressed(folder / f"{r['name']}.npz", **arrays)
            write_json(folder / f"{r['name']}.json", metadata)
            inventory[dataset].append(
                dict(
                    name=r["name"],
                    role=r["role"],
                    rows=len(r["states"]),
                    duration_s=float(r["time_s"][-1]),
                    dt_s=r["dt_s"],
                    sha256=r["metadata"]["sha256"],
                )
            )
    write_json(destination / "inventory.json", inventory)
    return inventory


def load_prepared(root, dataset):
    records = []
    for path in sorted((root / dataset).glob("*.npz")):
        metadata = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as arrays:
            records.append({**metadata, **{k: arrays[k] for k in arrays.files}})
    if not records:
        raise ValueError(f"missing prepared dataset {dataset}")
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpora, args.output)), flush=True)
