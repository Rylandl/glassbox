"""Read the frozen online readout cases without fitting or historical code."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/online-readout-benchmark.json"
_AUTHENTICATED = set()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _authenticate_sources(spec):
    for source in spec["sources"].values():
        root = Path(source["path"])
        identity = (str(root), source["manifest_sha256"])
        if identity in _AUTHENTICATED:
            continue
        manifest_path = root / "manifest.json"
        require(_digest(manifest_path) == identity[1], "frozen manifest differs")
        manifest = read(manifest_path)
        expected = manifest["files"]
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        require(set(expected) == actual, "frozen artifact inventory differs")
        for name, wanted in expected.items():
            relative = Path(name)
            require(
                not relative.is_absolute() and ".." not in relative.parts,
                "unsafe frozen artifact path",
            )
            require(
                _digest(root / relative) == wanted, f"frozen artifact differs: {name}"
            )
        _AUTHENTICATED.add(identity)


def rotation(quaternion):
    q = np.asarray(quaternion)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def observed(states):
    states = np.asarray(states)
    return np.concatenate(
        (states[:, 3:6], states[:, 10:13], rotation(states[:, 6:10]).reshape(-1, 9)),
        axis=1,
    )


def source_case(spec, name, suite):
    _authenticate_sources(spec)
    roots = {key: Path(value["path"]) for key, value in spec["sources"].items()}
    direct_root = roots["direct"] / name
    previous = arrays(direct_root / "predictions.npz")
    result = read(direct_root / "result.json")
    count = 2 if suite == "smoke" else len(previous["conditional_rows"])
    origins = previous["conditional_rows"][:count].astype(int)
    begin, first, dt = (
        result["prefix_begin_row"],
        result["first_prediction_row"],
        result["dt_s"],
    )
    require(first == origins[0] and origins[1] > first, "control origin mismatch")
    if name.startswith("paired-quad-"):
        grid = name.removeprefix("paired-quad-")
        tape = arrays(roots["paired_recording"] / f"{grid}.npz")
        full = arrays(roots["paired_full"] / grid / "predictions.npz")
        full_rows = full["origin_rows"]
        full_forecasts = full["conditional_baseline"]
        full_one_rows, full_one = full["rows"], full["one_step_baseline"]
        identity = "paired-quad-v2"
        names = tuple(f"motor_{i} [1]" for i in range(4))
    else:
        base = roots["recordings"]
        tape = arrays(base / "inputs" / f"{name}.npz")
        info = read(base / name / "case.json")
        full = arrays(roots["full"] / name / "predictions.npz")
        full_rows = full["conditional_rows"]
        full_forecasts = full["baseline_conditional"]
        full_one_rows, full_one = full["row"], full["baseline"]
        identity, names = info["opaque_id"], tuple(info["ordered_commands"])
    require(np.array_equal(full_rows[:count], origins), "full control origin mismatch")
    states, commands = observed(tape["states"]), tape["commands"].astype(float)
    horizon = round(0.25 / dt)
    end_row = int(origins[1]) if suite == "smoke" else first + len(previous["one"])
    require(
        np.array_equal(full_one_rows[: end_row - first], np.arange(first, end_row)),
        "full one-step control rows differ",
    )
    require(origins[-1] + horizon < len(states), "missing 250 ms truth")
    require(end_row + 1 <= len(states), "missing one-step truth")
    truth = np.stack([states[row + 1 : row + horizon + 1] for row in origins])
    response = None
    if name in spec.get("response_probes", {}):
        probe = spec["response_probes"][name]
        saved = arrays(roots["response"] / probe["file"])
        rows = saved["rows"].astype(int)
        require(
            np.array_equal(saved["commands"], commands[rows])
            and rows[0] >= first
            and rows[-1] < len(commands),
            "response probe does not match the recorded command tape",
        )
        selected = (rows >= first) & (rows <= end_row)
        response = dict(
            rows=rows[selected],
            truth=saved["jacobian"][selected],
            epsilon=float(probe["epsilon"]),
        )
    return dict(
        name=name,
        identity=identity,
        names=names,
        states=states,
        commands=commands,
        begin=begin,
        origins=origins,
        end_row=end_row,
        dt=dt,
        horizon=horizon,
        truth=truth,
        direct=previous["conditional"][:count],
        direct_one=previous["one"][: end_row - first],
        full=full_forecasts[:count],
        full_one=full_one[: end_row - first],
        response=response,
    )
