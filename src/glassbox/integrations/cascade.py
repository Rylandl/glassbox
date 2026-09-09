"""The optional Cascade fixed-wing plant, presented as one vehicle link.

Cascade (``cascade-flight``) is a differentiable fixed-wing flight-dynamics core. Its canonical
state boundary is the same NWU/FLU scalar-first 13-vector Glassbox uses, so no frame conversion
happens in this module; the schema strings are compared at import time. A plant is a writable
:class:`~glassbox.integrations.loop.VehicleLink`, so the control loop that reads live PX4
telemetry drives a simulated aircraft without changing a line: reading reports where the plant
is, and writing advances it one control interval.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from glassbox.core.data import RIGID_BODY_STATE_SCHEMA, Trajectory
from glassbox.integrations.loop import Observation


class CascadeUnavailableError(RuntimeError):
    """Raised when the optional Cascade dependency is unavailable or incompatible."""


def require_cascade() -> Any:
    try:
        import cascade
        from cascade.canonical import CANONICAL_STATE_SCHEMA
    except ImportError as error:
        raise CascadeUnavailableError(
            "install the optional simulator with `uv sync --group cascade`"
        ) from error
    if CANONICAL_STATE_SCHEMA != RIGID_BODY_STATE_SCHEMA:
        raise CascadeUnavailableError(
            f"Cascade canonical schema {CANONICAL_STATE_SCHEMA!r} does not match "
            f"{RIGID_BODY_STATE_SCHEMA!r}"
        )
    return cascade


def _command_bounds(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-channel command box this aircraft accepts.

    Commands follow ``control_names``: propellers first, then the specification's
    control channels. A propeller takes a normalized throttle in ``[0, 1]``. A
    control channel drives one or more surfaces through its entry in each
    surface's ``control_map_rad``, so the channel's own limit is the tightest
    surface deflection limit divided by the gain that surface applies to it: the
    largest command that cannot saturate anything it drives.
    """

    minimum = [0.0] * len(spec.propellers)
    maximum = [1.0] * len(spec.propellers)
    for index in range(len(spec.control_channels)):
        limits = [
            surface.actuator_limit_rad / abs(surface.control_map_rad[index])
            for surface in spec.surfaces
            if abs(surface.control_map_rad[index]) > 0.0
        ]
        limit = min(limits) if limits else 0.0
        minimum.append(-limit)
        maximum.append(limit)
    return np.asarray(minimum), np.asarray(maximum)


@dataclass(frozen=True)
class CascadePlantConfig:
    """Fixed execution contract for one Cascade plant."""

    aircraft: str = "skywalker_x8"
    simulation_frequency_hz: int = 400
    control_frequency_hz: int = 40
    density_kg_m3: float = 1.225

    def __post_init__(self) -> None:
        if self.aircraft not in ("skywalker_x8", "aerobatic_reference"):
            raise ValueError("aircraft must be 'skywalker_x8' or 'aerobatic_reference'")


class CascadePlant:
    """Single-world Cascade plant behind canonical telemetry, and one vehicle link.

    ``reset``, ``step`` and ``snapshot`` return Cascade's ``PlantSample`` whose ``state`` is the
    canonical 13-vector, ``commanded_control`` and ``applied_control`` follow ``control_names``
    (propellers first, then the specification's channels in their own units), and
    ``wind_nwu_m_s`` is the held wind.

    The same plant is a writable ``VehicleLink``: :meth:`read` reports the sample the plant is
    currently at and :meth:`write` advances it one control interval under the command it is
    handed. The applied command an observation carries is the plant's ``applied_control``, which
    is what the airframe is actually driving after actuator lag rather than what was last asked
    for. A plant that has never been reset has no state to report and says so.
    """

    writable = True

    def __init__(
        self,
        config: CascadePlantConfig | None = None,
        *,
        spec: Any | None = None,
        model: Any | None = None,
    ) -> None:
        cascade = require_cascade()
        self.config = CascadePlantConfig() if config is None else config
        if spec is None:
            loaders = {
                "skywalker_x8": cascade.skywalker_x8_spec,
                "aerobatic_reference": cascade.aerobatic_reference_spec,
            }
            spec = loaders[self.config.aircraft]()
        self.spec = spec
        self._plant = cascade.Plant(
            spec,
            cascade.PlantConfig(
                simulation_frequency_hz=self.config.simulation_frequency_hz,
                control_frequency_hz=self.config.control_frequency_hz,
                density_kg_m3=self.config.density_kg_m3,
            ),
            model=model,
        )
        self.control_names: tuple[str, ...] = self._plant.control_names
        self.command_size = len(self.control_names)
        self.command_bounds = _command_bounds(spec)

    @property
    def model(self) -> Any:
        return self._plant.model

    @property
    def sample_period_s(self) -> float:
        return self._plant.sample_period_s

    def reset(
        self,
        state: Any,
        *,
        applied_control: Any | None = None,
        wind_nwu: Any | None = None,
    ):
        return self._plant.reset(
            state, applied_control=applied_control, wind_nwu=wind_nwu
        )

    def step(self, command: Any, *, wind_nwu: Any | None = None):
        return self._plant.step(command, wind_nwu=wind_nwu)

    def snapshot(self):
        return self._plant.snapshot()

    def read(self, *, timeout_s: float = 0.0) -> Observation:
        """Return where the plant is now; a simulated link is never late."""

        sample = self.snapshot()
        return Observation(
            state=np.asarray(sample.state, dtype=np.float64),
            applied_command=np.asarray(sample.applied_control, dtype=np.float64),
            received_at_s=time.monotonic(),
            source_time_s=float(sample.time_s),
        )

    def write(self, command: Any) -> None:
        """Advance the plant by one control interval under ``command``."""

        self.step(command)


def trajectory_from_plant_samples(samples: Sequence[Any], spec: Any) -> Trajectory:
    """Assemble canonical plant telemetry into a Glassbox trajectory with zero typed wind."""

    time_s = np.asarray([sample.time_s for sample in samples], dtype=np.float64)
    states = np.stack([sample.state for sample in samples])
    controls = np.stack([sample.commanded_control for sample in samples[1:]])
    exogenous = np.stack([sample.wind_nwu_m_s for sample in samples])
    return Trajectory(
        time_s=time_s,
        states=states,
        controls=controls,
        exogenous=exogenous if spec.exogenous else None,
        spec=spec,
        labels={"source": "cascade_plant"},
        provenance={"source": "cascade_plant"},
    )
