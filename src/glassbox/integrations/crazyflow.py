"""Optional Crazyflow hidden-plant adapter for simulation-only experiments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

import jax.numpy as jnp
import numpy as np

# Glassbox: front-left, front-right, rear-right, rear-left.
# Crazyflow: front-right, rear-right, rear-left, front-left.
GLASSBOX_TO_CRAZYFLOW_MOTOR_ORDER = (1, 2, 3, 0)
CRAZYFLOW_TO_GLASSBOX_MOTOR_ORDER = (3, 0, 1, 2)


class CrazyflowUnavailableError(RuntimeError):
    """Raised when the optional pinned Crazyflow dependency is unavailable."""


def _finite_vector(name: str, value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def glassbox_to_crazyflow_motors(values: Any) -> np.ndarray:
    """Permute canonical Glassbox motor values into Crazyflow order."""

    motors = _finite_vector("Glassbox motor values", values, 4)
    return motors[np.asarray(GLASSBOX_TO_CRAZYFLOW_MOTOR_ORDER)]


def crazyflow_to_glassbox_motors(values: Any) -> np.ndarray:
    """Permute Crazyflow motor values into canonical Glassbox order."""

    motors = _finite_vector("Crazyflow motor values", values, 4)
    return motors[np.asarray(CRAZYFLOW_TO_GLASSBOX_MOTOR_ORDER)]


def canonical_state_to_crazyflow(state: Any) -> dict[str, np.ndarray]:
    """Convert a canonical NWU/FLU WXYZ rigid-body state to Crazyflow arrays.

    Crazyflow's first-principles simulator is also z-up with body angular rates,
    so the only storage conversion is WXYZ to XYZW quaternion order. The adapter
    names Crazyflow's world x/y axes north/west for this isolated experiment.
    """

    canonical = _finite_vector("canonical state", state, 13)
    quaternion = canonical[6:10]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("canonical state quaternion must have nonzero norm")
    quaternion = quaternion / norm
    return {
        "pos": canonical[0:3].copy(),
        "vel": canonical[3:6].copy(),
        "quat": quaternion[[1, 2, 3, 0]],
        "ang_vel": canonical[10:13].copy(),
    }


def crazyflow_state_to_canonical(
    *,
    pos: Any,
    vel: Any,
    quat_xyzw: Any,
    ang_vel: Any,
) -> np.ndarray:
    """Convert one Crazyflow state into canonical NWU/FLU WXYZ storage."""

    position = _finite_vector("Crazyflow position", pos, 3)
    velocity = _finite_vector("Crazyflow velocity", vel, 3)
    quaternion = _finite_vector("Crazyflow quaternion", quat_xyzw, 4)
    angular_velocity = _finite_vector("Crazyflow angular velocity", ang_vel, 3)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("Crazyflow state quaternion must have nonzero norm")
    quaternion = quaternion / norm
    return np.concatenate(
        (position, velocity, quaternion[[3, 0, 1, 2]], angular_velocity)
    )


def motor_thrust_from_rpm(rpm: Any, coefficients: Any) -> np.ndarray:
    """Evaluate Crazyflow's per-motor quadratic RPM-to-thrust curve."""

    speeds = np.asarray(rpm, dtype=np.float64)
    curve = _finite_vector("RPM-to-thrust coefficients", coefficients, 3)
    if not np.all(np.isfinite(speeds)) or np.any(speeds < 0.0):
        raise ValueError("motor RPM must be finite and nonnegative")
    return curve[0] + curve[1] * speeds + curve[2] * np.square(speeds)


def motor_rpm_from_thrust(thrust_n: Any, coefficients: Any) -> np.ndarray:
    """Invert the supported positive branch of the quadratic thrust curve."""

    thrust = np.asarray(thrust_n, dtype=np.float64)
    curve = _finite_vector("RPM-to-thrust coefficients", coefficients, 3)
    if not np.all(np.isfinite(thrust)) or np.any(thrust < 0.0):
        raise ValueError("motor thrust must be finite and nonnegative")
    constant, linear, quadratic = curve
    if quadratic <= 0.0:
        raise ValueError("RPM-to-thrust quadratic coefficient must be positive")
    discriminant = np.square(linear) - 4.0 * quadratic * (constant - thrust)
    if np.any(discriminant < 0.0):
        raise ValueError("requested motor thrust is outside the calibrated curve")
    return (-linear + np.sqrt(discriminant)) / (2.0 * quadratic)


@dataclass(frozen=True)
class CrazyflowPlantConfig:
    """Fixed execution and normalized-actuation contract for one plant."""

    drone: str = "cf21B_500"
    simulation_frequency_hz: int = 500
    control_frequency_hz: int = 50
    maximum_motor_thrust_n: float = 0.2
    device: str = "cpu"
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.simulation_frequency_hz < 1 or self.control_frequency_hz < 1:
            raise ValueError("Crazyflow frequencies must be positive")
        if self.simulation_frequency_hz % self.control_frequency_hz:
            raise ValueError(
                "Crazyflow simulation frequency must be divisible by control frequency"
            )
        if (
            not np.isfinite(self.maximum_motor_thrust_n)
            or self.maximum_motor_thrust_n <= 0.0
        ):
            raise ValueError("maximum_motor_thrust_n must be finite and positive")

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.control_frequency_hz

    @property
    def simulation_steps_per_control(self) -> int:
        return self.simulation_frequency_hz // self.control_frequency_hz


@dataclass(frozen=True)
class CrazyflowPlantSample:
    """One canonical plant sample with both requested and applied actuation."""

    time_s: float
    state: np.ndarray
    commanded_motor_thrust_fraction: np.ndarray
    applied_motor_thrust_fraction: np.ndarray
    commanded_motor_rpm: np.ndarray
    applied_motor_rpm: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("Crazyflow sample time must be finite and nonnegative")
        object.__setattr__(self, "state", _finite_vector("state", self.state, 13))
        for name in (
            "commanded_motor_thrust_fraction",
            "applied_motor_thrust_fraction",
            "commanded_motor_rpm",
            "applied_motor_rpm",
        ):
            object.__setattr__(self, name, _finite_vector(name, getattr(self, name), 4))


class CrazyflowPlant:
    """Single-world, single-drone first-principles plant hidden behind telemetry."""

    def __init__(
        self,
        config: CrazyflowPlantConfig | None = None,
        *,
        simulator: Any | None = None,
    ) -> None:
        config = CrazyflowPlantConfig() if config is None else config
        self.config = config
        if simulator is None:
            try:
                from crazyflow import Control, Dynamics, Sim
            except (ImportError, RuntimeError) as error:
                # Crazyflow raises RuntimeError instead of ImportError when
                # SciPy was imported before it without SCIPY_ARRAY_API=1.
                raise CrazyflowUnavailableError(
                    "install the pinned simulator with `uv sync --extra crazyflow`; "
                    "if crazyflow is installed, this may be "
                    "`SCIPY_ARRAY_API=1` not being set before SciPy's first "
                    "import in this process"
                ) from error
            simulator = Sim(
                n_worlds=1,
                n_drones=1,
                drone=config.drone,
                dynamics=Dynamics.first_principles,
                control=Control.rotor_vel,
                freq=config.simulation_frequency_hz,
                device=config.device,
                rng_key=config.rng_seed,
            )
        if simulator.n_worlds != 1 or simulator.n_drones != 1:
            raise ValueError("CrazyflowPlant requires exactly one world and one drone")
        if simulator.freq != config.simulation_frequency_hz:
            raise ValueError(
                "Crazyflow simulator frequency disagrees with plant config"
            )
        self._simulator = simulator
        self._baseline_arm_length_m = float(np.asarray(simulator.data.params.L))
        self._baseline_inertia_kg_m2 = np.asarray(
            simulator.data.params.J, dtype=np.float64
        ).copy()
        self._rpm_to_thrust = np.asarray(
            simulator.data.params.rpm2thrust, dtype=np.float64
        ).copy()
        self._mass_kg = float(np.ravel(np.asarray(simulator.data.params.mass))[0])
        self._gravity_m_s2 = abs(
            float(np.ravel(np.asarray(simulator.data.params.gravity_vec))[-1])
        )
        self._arm_length_ratio = 1.0

    @property
    def crazyflow_version(self) -> str | None:
        try:
            return metadata.version("crazyflow")
        except metadata.PackageNotFoundError:
            return None

    @property
    def sample_period_s(self) -> float:
        return self.config.sample_period_s

    @property
    def arm_length_ratio(self) -> float:
        return self._arm_length_ratio

    @property
    def baseline_arm_length_m(self) -> float:
        return self._baseline_arm_length_m

    @property
    def hover_motor_thrust_fraction(self) -> float:
        return (
            self._mass_kg
            * self._gravity_m_s2
            / (4.0 * self.config.maximum_motor_thrust_n)
        )

    def _normalized_to_rpm(self, command: Any) -> np.ndarray:
        normalized = _finite_vector("normalized motor command", command, 4)
        if np.any(normalized < 0.0) or np.any(normalized > 1.0):
            raise ValueError("normalized motor command must lie inside [0, 1]")
        thrust = normalized * self.config.maximum_motor_thrust_n
        canonical_rpm = motor_rpm_from_thrust(thrust, self._rpm_to_thrust)
        return glassbox_to_crazyflow_motors(canonical_rpm)

    def _rpm_to_normalized(self, rpm: Any) -> np.ndarray:
        crazyflow_rpm = _finite_vector("Crazyflow motor RPM", rpm, 4)
        thrust = motor_thrust_from_rpm(crazyflow_rpm, self._rpm_to_thrust)
        normalized = thrust / self.config.maximum_motor_thrust_n
        normalized = np.where(np.abs(normalized) < 1e-9, 0.0, normalized)
        normalized = np.where(np.abs(normalized - 1.0) < 1e-9, 1.0, normalized)
        return crazyflow_to_glassbox_motors(normalized)

    def set_arm_length_ratio(self, ratio: float) -> None:
        """Apply a hidden arm/inertia change without exposing it in telemetry."""

        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("arm length ratio must be finite and positive")
        self._arm_length_ratio = float(ratio)
        self._apply_configuration()

    def _apply_configuration(self) -> None:
        ratio = self._arm_length_ratio
        inertia = self._baseline_inertia_kg_m2 * ratio**2
        inverse = np.linalg.inv(inertia)
        params = self._simulator.data.params.replace(
            L=jnp.asarray(self._baseline_arm_length_m * ratio),
            J=jnp.asarray(inertia),
            J_inv=jnp.asarray(inverse),
        )
        self._simulator.data = self._simulator.data.replace(params=params)

    def reset(
        self,
        state: Any,
        *,
        applied_motor_thrust_fraction: Any | None = None,
    ) -> CrazyflowPlantSample:
        """Reset to one canonical state and a known applied rotor condition."""

        self._simulator.reset()
        self._apply_configuration()
        converted = canonical_state_to_crazyflow(state)
        applied = (
            np.full(4, self.hover_motor_thrust_fraction, dtype=np.float64)
            if applied_motor_thrust_fraction is None
            else _finite_vector(
                "applied motor thrust fraction",
                applied_motor_thrust_fraction,
                4,
            )
        )
        rpm = self._normalized_to_rpm(applied)
        data = self._simulator.data
        states = data.states.replace(
            pos=jnp.asarray(converted["pos"])[None, None, :],
            quat=jnp.asarray(converted["quat"])[None, None, :],
            vel=jnp.asarray(converted["vel"])[None, None, :],
            ang_vel=jnp.asarray(converted["ang_vel"])[None, None, :],
            rotor_vel=jnp.asarray(rpm)[None, None, :],
        )
        controls = data.controls.replace(rotor_vel=jnp.asarray(rpm)[None, None, :])
        self._simulator.data = data.replace(states=states, controls=controls)
        return self.snapshot()

    def snapshot(self) -> CrazyflowPlantSample:
        """Return current canonical telemetry without advancing the plant."""

        data = self._simulator.data
        states = data.states
        commanded_rpm_cf = np.asarray(data.controls.rotor_vel)[0, 0]
        applied_rpm_cf = np.asarray(states.rotor_vel)[0, 0]
        state = crazyflow_state_to_canonical(
            pos=np.asarray(states.pos)[0, 0],
            vel=np.asarray(states.vel)[0, 0],
            quat_xyzw=np.asarray(states.quat)[0, 0],
            ang_vel=np.asarray(states.ang_vel)[0, 0],
        )
        steps = int(np.ravel(np.asarray(data.core.steps))[0])
        return CrazyflowPlantSample(
            time_s=steps / self.config.simulation_frequency_hz,
            state=state,
            commanded_motor_thrust_fraction=self._rpm_to_normalized(commanded_rpm_cf),
            applied_motor_thrust_fraction=self._rpm_to_normalized(applied_rpm_cf),
            commanded_motor_rpm=crazyflow_to_glassbox_motors(commanded_rpm_cf),
            applied_motor_rpm=crazyflow_to_glassbox_motors(applied_rpm_cf),
        )

    def step(self, command: Any) -> CrazyflowPlantSample:
        """Hold one normalized thrust command for one Glassbox sample interval."""

        rpm = self._normalized_to_rpm(command)
        self._simulator.rotor_vel_control(jnp.asarray(rpm)[None, None, :])
        self._simulator.step(self.config.simulation_steps_per_control)
        return self.snapshot()

    def close(self) -> None:
        close = getattr(self._simulator, "close", None)
        if close is not None:
            close()
