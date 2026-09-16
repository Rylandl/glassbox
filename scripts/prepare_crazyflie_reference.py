"""Decode pinned uSD sensor logs with as-of event-time sampling.

Protocol reference: Bitcraze tools/usdlog/cfusdlog.py. Event timestamps are not
separate sensor sample and publication clocks. No velocity or attitude exists
in these files, and none is reconstructed by this experiment.
"""

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from sequence_transfer_data import write_json

from glassbox.experimental.causal_sampling import causal_hold

HASHES = {
    "log10": "da77c31ecf18a65159b7f973cf3246ec80ea2fc3e60183812416fad47f32b402",
    "log15": "364dea627205f4eee3e8d54f72b0ba8fd93442dcea9c8cc4a42b59004ecf3a31",
    "log16": "19070dd8462cdffd547a1c2f727b19308341ac79cc852a0ef0f1e796cd71b3a9",
}


def decode(path):
    data = Path(path).read_bytes()
    if (
        len(data) < 9
        or data[0] != 0xBC
        or zlib.crc32(data[:-4]) != struct.unpack("<I", data[-4:])[0]
    ):
        raise ValueError("invalid uSD header or CRC")
    version, events = struct.unpack_from("<HH", data, 1)
    if version not in (1, 2):
        raise ValueError("unsupported uSD version")
    offset = 5

    def name():
        nonlocal offset
        end = data.index(b"\0", offset)
        value = data[offset:end].decode("utf8")
        offset = end + 1
        return value

    schema = {}
    rows = {}
    for _ in range(events):
        identity = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        label = name()
        width = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        fields = [name() for _ in range(width)]
        fmt = struct.Struct("<" + "".join(f[-2] for f in fields))
        schema[identity] = (label, [f[:-3] for f in fields], fmt)
        rows[label] = []
    clock = struct.Struct("<HI" if version == 1 else "<HQ")
    while offset < len(data) - 4:
        identity, t = clock.unpack_from(data, offset)
        offset += clock.size
        label, fields, fmt = schema[identity]
        values = fmt.unpack_from(data, offset)
        offset += fmt.size
        rows[label].append((t / (1000 if version == 1 else 1000000), *values))
    if offset != len(data) - 4:
        raise ValueError("incomplete uSD record")
    result = {}
    for label, fields, _ in schema.values():
        if rows[label]:
            result[label] = {
                key: column
                for key, column in zip(
                    ["timestamp_s", *fields], np.array(rows[label], dtype=float).T
                )
            }
    return result


def prepare(raw, output, *, include_evaluation=False):
    output.mkdir(parents=True, exist_ok=False)
    inventory = []
    for name, role in [
        ("log10", "train"),
        ("log15", "development"),
        ("log16", "evaluation"),
    ]:
        if role == "evaluation" and not include_evaluation:
            continue
        path = raw / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
            raise ValueError("source hash mismatch")
        record = decode(path)["fixedFrequency"]
        fields = [
            *[f"acc.{a}" for a in "xyz"],
            *[f"gyro.{a}" for a in "xyz"],
            *[f"motor.m{i}" for i in range(1, 5)],
        ]
        values = np.column_stack([record[f] for f in fields])
        t = record["timestamp_s"]
        if not np.isfinite(values).all() or np.any(np.diff(t) < 0):
            raise ValueError("invalid sensor stream")
        dt = 0.02
        grid = np.arange(np.ceil(t[0] / dt), np.floor(t[-1] / dt) + 1) * dt
        held = causal_hold(t, values, grid, maximum_age_s=0.01)
        powered = held.valid & (held.values[:, 6:].mean(1) / 65536 > 0.1)
        runs = np.flatnonzero(
            np.diff(np.r_[False, powered, False].astype(int))
        ).reshape(-1, 2)
        if len(runs) == 0:
            raise ValueError("no powered interval")
        first, last = max(runs, key=lambda r: r[1] - r[0])
        s = held.values[first:last]
        if last - first < 25:
            raise ValueError("insufficient powered observations")
        arrays = dict(
            states=np.column_stack((s[:, :3], np.deg2rad(s[:, 3:6]))),
            inputs=s[:-1, 6:] / 65536,
            time_s=grid[first:last] - grid[first],
            absolute_grid_s=grid[first:last],
            event_s=t[held.source_indices[first:last]],
            source_indices=held.source_indices[first:last],
        )
        meta = dict(
            name=name,
            role=role,
            dt_s=dt,
            metadata=dict(
                source=str(path.resolve()),
                sha256=HASHES[name],
                fields=fields,
                output_units=["g"] * 3 + ["rad/s"] * 3,
                inputs="four unsigned motor commands divided by 65536, original order",
                interpretation="as-of logger event timestamps; separate sensor latency/publication time is unavailable",
                processing="50 Hz zero-order hold, maximum event age 10 ms; longest mean-command >0.1 interval; no added filtering/interpolation",
                rows=int(last - first),
                duration_s=float((last - first - 1) * dt),
                maximum_event_age_s=float(np.max(grid[first:last] - arrays["event_s"])),
            ),
        )
        np.savez_compressed(output / f"{name}.npz", **arrays)
        write_json(output / f"{name}.json", meta)
        inventory.append(meta)
    write_json(output / "inventory.json", inventory)
    return inventory


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-evaluation", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(prepare(a.raw, a.output, include_evaluation=a.include_evaluation)),
        flush=True,
    )
