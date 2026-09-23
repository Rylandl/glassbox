"""Fit the single causal learner on two sealed, held-out Crazyflow flights.

The behavior controller supplied these recordings independently. Every fit sees
only completed observations after the frozen row-50 prefix boundary. Verification
replays saved fitted equations and scores without fitting or simulation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from benchmark_online_readout import observed
from scipy.spatial.transform import Rotation

from glassbox._causal_actuator import CausalActuatorModel
from glassbox._causal_fit import fit_episode

SOURCE = Path("/Users/ryland/autonomy/glassbox/artifacts/heldout-quad-v1")
AUTHORITY = {
    "source": "5a60009ea431da8609186cb4cf3bd45419991724215ceeabf385b2cd9084ea80",
    "evaluation": "e516c24e78544cd3d758cb7b448aae432f5ebfe920b0c4eb3c9fd64d279cb713",
}
CASES = ("quad-arm-085-heldout", "quad-arm-140-heldout")
PARAMETERS = ("q", "coeff", "inertia", "force", "torque")
BEGIN = 50
HORIZON = 25


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticate(root, authority):
    manifest_path = root / "manifest.json"
    if digest(manifest_path) != authority:
        raise ValueError("frozen manifest authority differs")
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(files) != actual:
        raise ValueError("frozen artifact inventory differs")
    for name, wanted in files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or digest(root / name) != wanted
        ):
            raise ValueError(f"frozen artifact differs: {name}")


def load_case(name):
    with np.load(SOURCE / "source" / name / "stream.npz", allow_pickle=False) as data:
        states = observed(data["states"])
        commands = data["commands"].astype(np.float64)
        time_s = data["time_s"]
    with np.load(SOURCE / "evaluation" / f"{name}.npz", allow_pickle=False) as data:
        origins = data["origins"].astype(int)
        public = data["forecast"].copy()
    dt = float(np.median(np.diff(time_s)))
    if (
        len(states) != len(commands) + 1
        or not np.allclose(np.diff(time_s), dt, rtol=0, atol=1e-8)
        or not np.isclose(dt, 0.01, rtol=0, atol=1e-8)
        or origins[0] != 125
        or not np.array_equal(np.diff(origins), np.full(len(origins) - 1, 25))
        or origins[-1] + HORIZON >= len(states)
        or public.shape != (len(origins), HORIZON, 15)
    ):
        raise ValueError("held-out source timing or frozen origins differ")
    truth = np.stack([states[row + 1 : row + HORIZON + 1] for row in origins])
    hold = np.repeat(states[origins, None, :], HORIZON, axis=1)
    return dict(
        states=states,
        commands=commands,
        dt=dt,
        origins=origins,
        public=public,
        truth=truth,
        hold=hold,
    )


def errors(prediction, truth):
    delta = prediction - truth
    relative = np.swapaxes(
        truth[..., 6:].reshape(*truth.shape[:2], 3, 3), -1, -2
    ) @ prediction[..., 6:].reshape(*prediction.shape[:2], 3, 3)
    orientation = (
        Rotation.from_matrix(relative.reshape(-1, 3, 3))
        .magnitude()
        .reshape(truth.shape[:2])
    )
    result = {}
    for ms in (50, 100, 150, 200, 250):
        index = ms // 10 - 1
        result[str(ms)] = {
            "velocity_m_s": float(
                np.sqrt(np.mean(np.sum(delta[:, index, :3] ** 2, axis=1)))
            ),
            "body_rate_rad_s": float(
                np.sqrt(np.mean(np.sum(delta[:, index, 3:6] ** 2, axis=1)))
            ),
            "orientation_rad": float(np.sqrt(np.mean(orientation[:, index] ** 2))),
        }
    return result


def score(case, forecast, archive):
    truth = case["truth"]
    return {
        "origins": case["origins"].tolist(),
        "candidate": errors(forecast, truth),
        "previous_public": errors(case["public"], truth),
        "hold": errors(case["hold"], truth),
        "candidate_250_rate_by_origin": np.linalg.norm(
            forecast[:, -1, 3:6] - truth[:, -1, 3:6], axis=1
        ).tolist(),
        "previous_public_250_rate_by_origin": np.linalg.norm(
            case["public"][:, -1, 3:6] - truth[:, -1, 3:6], axis=1
        ).tolist(),
        "archive_sha256": digest(archive),
    }


def evaluate(name, output):
    case = load_case(name)
    models, forecasts, seconds = [], [], []
    for origin in case["origins"]:
        origin = int(origin)
        model, report = fit_episode(
            case["states"][BEGIN : origin + 1],
            case["commands"][BEGIN:origin],
            case["dt"],
        )
        with jax.enable_x64(True):
            forecast = np.asarray(
                model.forecast(
                    case["states"][origin],
                    case["commands"][BEGIN:origin],
                    case["commands"][origin : origin + HORIZON],
                )
            )
        if forecast.shape != (HORIZON, 15) or not np.isfinite(forecast).all():
            raise ValueError("nonfinite held-out forecast")
        models.append(model)
        forecasts.append(forecast)
        seconds.append(report["fit_seconds"])
    archive = output / f"{name}.npz"
    np.savez_compressed(
        archive,
        origins=case["origins"],
        forecast=np.stack(forecasts),
        fit_seconds=np.asarray(seconds),
        **{
            key: np.stack([getattr(model, key) for model in models])
            for key in PARAMETERS
        },
    )
    return score(case, np.stack(forecasts), archive)


def verify(name, output):
    case = load_case(name)
    archive = output / f"{name}.npz"
    with np.load(archive, allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    if (
        set(saved) != {"origins", "forecast", "fit_seconds", *PARAMETERS}
        or not np.array_equal(saved["origins"], case["origins"])
        or saved["forecast"].shape != case["truth"].shape
        or len(saved["fit_seconds"]) != len(case["origins"])
    ):
        raise ValueError("saved held-out roster or origins differ")
    for i, origin in enumerate(case["origins"]):
        model = CausalActuatorModel(
            case["dt"], **{key: saved[key][i] for key in PARAMETERS}
        )
        with jax.enable_x64(True):
            prediction = np.asarray(
                model.forecast(
                    case["states"][origin],
                    case["commands"][BEGIN:origin],
                    case["commands"][origin : origin + HORIZON],
                )
            )
        np.testing.assert_allclose(saved["forecast"][i], prediction, rtol=0, atol=2e-11)
    return score(case, saved["forecast"], archive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    authenticate(SOURCE / "source", AUTHORITY["source"])
    authenticate(SOURCE / "evaluation", AUTHORITY["evaluation"])
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.verify:
        for name in CASES:
            print(name, evaluate(name, args.output)["candidate"]["250"], flush=True)
    result = {name: verify(name, args.output) for name in CASES}
    path = args.output / "summary.json"
    if args.verify:
        if result != json.loads(path.read_text()):
            raise ValueError("saved held-out scores differ")
        print("verified both held-out recordings without fitting")
    else:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
