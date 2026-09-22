"""Collect one longer 10/50 ms quad pair with a frozen behavior-only pilot."""

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from collect_paired_quad import GIT, ROOT, digest, paired, read, save, seal, write


PROTOCOL = ROOT / "docs/harness/paired-quad-sampling-v2.json"


def source_binding(spec):
    source = spec["source"]
    repository = Path(source["repository"])
    commit = subprocess.check_output([GIT, "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    assert commit == source["commit"]
    changed = subprocess.check_output(
        [GIT, "-C", str(repository), "diff", "HEAD", "--name-only", "--", "src", "pyproject.toml", "uv.lock"],
        text=True,
    ).splitlines()
    assert not changed, changed
    assert importlib.metadata.version("crazyflow") == source["crazyflow_version"]
    return dict(
        repository=str(repository), commit=commit,
        crazyflow_version=importlib.metadata.version("crazyflow"),
        script_sha256=digest(Path(__file__)), protocol_sha256=digest(PROTOCOL),
        python=platform.python_version(),
    )


class Pilot:
    """Known-plant metadata is used only to generate issued behavior commands."""

    def __init__(self, plant):
        from glassbox_throw.plant import crazyflow_to_glassbox_motors

        self.to_glassbox = crazyflow_to_glassbox_motors
        params = plant._simulator.data.params
        self.mass = plant._mass_kg
        self.gravity = plant._gravity_m_s2
        self.inertia = np.asarray(params.J, dtype=float).reshape(3, 3)
        arm = float(np.asarray(params.L))
        mixing = np.asarray(params.mixing_matrix, dtype=float).reshape(3, 4)
        thrust_curve = np.asarray(params.rpm2thrust, dtype=float).reshape(3)
        torque_curve = np.asarray(params.rpm2torque, dtype=float).reshape(3)
        hover = np.full(4, plant.hover_motor_thrust_fraction)
        hover_rpm = float(np.mean(plant._normalized_to_rpm(hover)))
        thrust_slope = thrust_curve[1] + 2 * thrust_curve[2] * hover_rpm
        torque_slope = torque_curve[1] + 2 * torque_curve[2] * hover_rpm
        assert thrust_slope > 0
        yaw_ratio = torque_slope / thrust_slope
        self.allocation = np.vstack((
            np.ones(4), arm * mixing[0], arm * mixing[1], yaw_ratio * mixing[2]
        ))
        assert np.linalg.matrix_rank(self.allocation) == 4
        self.maximum_motor_thrust = plant.config.maximum_motor_thrust_n
        self.clipped_motors = 0
        self.minimum_unclipped = float("inf")
        self.maximum_unclipped = float("-inf")

    def command(self, state, time_s):
        position, velocity, quaternion, omega = (
            np.asarray(state[:3]), np.asarray(state[3:6]),
            np.asarray(state[6:10]), np.asarray(state[10:13])
        )
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        desired = Rotation.from_euler("xyz", (
            0.12 * np.sin(0.9 * time_s),
            0.10 * np.sin(1.3 * time_s + 0.3),
            0.08 * np.sin(0.5 * time_s),
        )).as_matrix()
        skew = 0.5 * (desired.T @ rotation - rotation.T @ desired)
        angle_error = np.array((skew[2, 1], skew[0, 2], skew[1, 0]))
        acceleration = -25.0 * angle_error - 8.0 * omega
        moment = self.inertia @ acceleration + np.cross(omega, self.inertia @ omega)
        reference_height = 10.0 + 0.5 * np.sin(0.7 * time_s)
        vertical_acceleration = 1.8 * (reference_height - position[2]) - 1.7 * velocity[2]
        total_force = self.mass * (self.gravity + vertical_acceleration) / max(rotation[2, 2], 0.5)
        target = np.concatenate(([total_force], moment))
        motor_forces = np.linalg.solve(self.allocation, target)
        unbounded = motor_forces / self.maximum_motor_thrust
        self.clipped_motors += int(np.sum((unbounded < 0) | (unbounded > 1)))
        self.minimum_unclipped = min(self.minimum_unclipped, float(unbounded.min()))
        self.maximum_unclipped = max(self.maximum_unclipped, float(unbounded.max()))
        return self.to_glassbox(np.clip(unbounded, 0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = read(PROTOCOL)
    bound = source_binding(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", bound)
    from glassbox_throw.plant import CrazyflowPlant, CrazyflowPlantConfig

    plant = None
    states, times, commands, applied, requested = [], [], [], [], []
    started = time.monotonic()
    try:
        plant = CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
        pilot = Pilot(plant)
        initial = np.array([0, 0, 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        sample = plant.reset(
            initial, applied_motor_thrust_fraction=np.full(4, plant.hover_motor_thrust_fraction)
        )
        states.append(sample.state.copy())
        times.append(sample.time_s)
        applied.append(sample.applied_motor_thrust_fraction.copy())
        stopped = "duration"
        for block in range(round(spec["collection"]["duration_s"] / spec["collection"]["policy_period_s"])):
            if time.monotonic() - started > spec["budget"]["maximum_wall_s_collection"]:
                raise TimeoutError("frozen collection wall limit exceeded")
            command = pilot.command(sample.state, sample.time_s)
            requested.append(command.copy())
            for _ in range(5):
                sample = plant.step(command)
                states.append(sample.state.copy())
                times.append(sample.time_s)
                applied.append(sample.applied_motor_thrust_fraction.copy())
                commands.append(command.copy())
            if (block + 1) % 20 == 0:
                print("elapsed_s", round(sample.time_s, 2), "height_m", round(sample.state[2], 3), flush=True)
            if sample.state[2] < 0.2:
                stopped = "floor"
                break
        fine = dict(
            states=np.asarray(states), time_s=np.asarray(times), commands=np.asarray(commands),
            applied=np.asarray(applied), requested_macro=np.asarray(requested).reshape(-1, 4),
        )
        coarse = paired(fine)
        save(args.output / "fine.npz", **fine)
        save(args.output / "coarse.npz", **coarse)
        write(args.output / "result.json", {
            "status": "complete", "stop": stopped,
            "fine_intervals": len(fine["commands"]),
            "coarse_intervals": len(coarse["commands"]),
            "duration_s": float(fine["time_s"][-1]),
            "minimum_height_m": float(fine["states"][:, 2].min()),
            "maximum_speed_m_s": float(np.linalg.norm(fine["states"][:, 3:6], axis=-1).max()),
            "maximum_rate_rad_s": float(np.linalg.norm(fine["states"][:, 10:13], axis=-1).max()),
            "clipped_motor_commands": pilot.clipped_motors,
            "minimum_unclipped_command": pilot.minimum_unclipped,
            "maximum_unclipped_command": pilot.maximum_unclipped,
            "wall_s": time.monotonic() - started,
        })
        print("complete", read(args.output / "result.json"), flush=True)
    except BaseException as error:
        if states:
            save(args.output / "partial.npz", states=np.asarray(states),
                 commands=np.asarray(commands).reshape(-1, 4), time_s=np.asarray(times))
        write(args.output / "failure.json", {"error": repr(error), "wall_s": time.monotonic() - started})
        raise
    finally:
        if plant is not None:
            plant.close()
        print("manifest_sha256", seal(args.output), flush=True)


if __name__ == "__main__":
    main()
