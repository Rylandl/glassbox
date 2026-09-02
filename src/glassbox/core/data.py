"""Canonical in-memory trajectory data structures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

STATE_SIZE = 13
RIGID_BODY_STATE_SCHEMA = "rigid_body_13_nwu_flu_wxyz_v1"
NORMALIZED_MOTOR_COMMAND_SEMANTICS = frozenset(
    {"normalized_command", "normalized_actuator_output"}
)
PHYSICAL_MOTOR_THRUST_SEMANTICS = frozenset({"squared_rotor_speed_ratio"})


def duration_to_steps(duration_s: float, dt_s: float) -> int:
    """Resolve a requested duration to reproducible whole telemetry steps.

    The selected horizon does not exceed the request, except that every
    positive request receives at least one step. A small tolerance snaps ratios
    that are only numerically below an integer because of timestamp precision.
    """

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    ratio = duration_s / dt_s
    tolerance = 1e-9 * max(1.0, abs(ratio))
    return max(1, int(np.floor(ratio + tolerance)))


@dataclass(frozen=True)
class ControlChannel:
    """Meaning of one column in a canonical trajectory control array."""

    name: str
    role: str
    semantic: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    frame: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "role", "semantic", "unit"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"control channel {field_name} cannot be empty")
        if self.minimum is not None and not np.isfinite(self.minimum):
            raise ValueError("control channel minimum must be finite")
        if self.maximum is not None and not np.isfinite(self.maximum):
            raise ValueError("control channel maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum >= self.maximum
        ):
            raise ValueError("control channel minimum must be less than maximum")
        if self.frame is not None and not self.frame.strip():
            raise ValueError("control channel frame cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "semantic": self.semantic,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlChannel:
        return cls(
            name=str(payload["name"]),
            role=str(payload["role"]),
            semantic=str(payload["semantic"]),
            unit=str(payload["unit"]),
            minimum=(
                None if payload.get("minimum") is None else float(payload["minimum"])
            ),
            maximum=(
                None if payload.get("maximum") is None else float(payload["maximum"])
            ),
            frame=(None if payload.get("frame") is None else str(payload["frame"])),
        )


@dataclass(frozen=True)
class ExogenousChannel:
    """Meaning of one measured, non-control input available at prediction time."""

    name: str
    role: str
    semantic: str
    unit: str
    frame: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "role", "semantic", "unit"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"exogenous channel {field_name} cannot be empty")
        if self.frame is not None and not self.frame.strip():
            raise ValueError("exogenous channel frame cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "semantic": self.semantic,
            "unit": self.unit,
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExogenousChannel:
        return cls(
            name=str(payload["name"]),
            role=str(payload["role"]),
            semantic=str(payload["semantic"]),
            unit=str(payload["unit"]),
            frame=(None if payload.get("frame") is None else str(payload["frame"])),
        )


@dataclass(frozen=True)
class ObservationChannel:
    """Meaning of one state-aligned measurement used during identification.

    Observations are measured outputs such as accelerometer specific force.
    Unlike exogenous channels, they are not assumed to be available when a
    fitted model is rolled forward.
    """

    name: str
    role: str
    semantic: str
    unit: str
    frame: str
    source: str

    def __post_init__(self) -> None:
        for field_name in ("name", "role", "semantic", "unit", "frame", "source"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"observation channel {field_name} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "semantic": self.semantic,
            "unit": self.unit,
            "frame": self.frame,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObservationChannel:
        return cls(
            name=str(payload["name"]),
            role=str(payload["role"]),
            semantic=str(payload["semantic"]),
            unit=str(payload["unit"]),
            frame=str(payload["frame"]),
            source=str(payload["source"]),
        )


def specific_force_observation_channels(
    source: str,
) -> tuple[ObservationChannel, ObservationChannel, ObservationChannel]:
    """Return the canonical FLU accelerometer output contract."""

    return tuple(
        ObservationChannel(
            name=f"specific_force_{axis}_m_s2",
            role=f"specific_force_{axis}",
            semantic="accelerometer_specific_force_including_gravity",
            unit="m/s^2",
            frame="FLU",
            source=source,
        )
        for axis in ("x", "y", "z")
    )  # type: ignore[return-value]


def angular_acceleration_observation_channels(
    source: str,
) -> tuple[ObservationChannel, ObservationChannel, ObservationChannel]:
    """Return the canonical FLU body-angular-acceleration contract."""

    return tuple(
        ObservationChannel(
            name=f"angular_acceleration_{axis}_rad_s2",
            role=f"angular_acceleration_{axis}",
            semantic="bias_corrected_body_angular_acceleration",
            unit="rad/s^2",
            frame="FLU",
            source=source,
        )
        for axis in ("x", "y", "z")
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class VehicleConfigurationSpec:
    """Vehicle configuration facts that determine safe model-data pooling."""

    family: str
    configuration_id: str | None = None
    controlled_axes: tuple[str, ...] = ()
    propulsion: str = "unknown"
    fixed_states: Mapping[str, Any] = field(default_factory=dict)
    auxiliary_controls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise ValueError("vehicle family cannot be empty")
        if self.configuration_id is not None and not self.configuration_id.strip():
            raise ValueError("vehicle configuration_id cannot be empty")
        if not self.propulsion.strip():
            raise ValueError("vehicle propulsion cannot be empty")
        controlled_axes = tuple(str(axis) for axis in self.controlled_axes)
        auxiliary_controls = tuple(str(role) for role in self.auxiliary_controls)
        if any(not axis.strip() for axis in controlled_axes):
            raise ValueError("controlled_axes cannot contain empty values")
        if any(not role.strip() for role in auxiliary_controls):
            raise ValueError("auxiliary_controls cannot contain empty values")
        if len(set(controlled_axes)) != len(controlled_axes):
            raise ValueError("controlled_axes must be unique")
        if len(set(auxiliary_controls)) != len(auxiliary_controls):
            raise ValueError("auxiliary_controls must be unique")
        if any(not isinstance(key, str) or not key for key in self.fixed_states):
            raise ValueError("fixed_states keys must be non-empty strings")
        object.__setattr__(self, "controlled_axes", controlled_axes)
        object.__setattr__(self, "auxiliary_controls", auxiliary_controls)
        object.__setattr__(self, "fixed_states", dict(self.fixed_states))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "configuration_id": self.configuration_id,
            "controlled_axes": list(self.controlled_axes),
            "propulsion": self.propulsion,
            "fixed_states": dict(self.fixed_states),
            "auxiliary_controls": list(self.auxiliary_controls),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VehicleConfigurationSpec:
        return cls(
            family=str(payload["family"]),
            configuration_id=(
                None
                if payload.get("configuration_id") is None
                else str(payload["configuration_id"])
            ),
            controlled_axes=tuple(
                str(value) for value in payload.get("controlled_axes", ())
            ),
            propulsion=str(payload.get("propulsion", "unknown")),
            fixed_states=dict(payload.get("fixed_states", {})),
            auxiliary_controls=tuple(
                str(value) for value in payload.get("auxiliary_controls", ())
            ),
        )


@dataclass(frozen=True)
class TrajectorySpec:
    """Versioned semantic contract for canonical state and control arrays."""

    state_schema: str
    observation_source: str
    controls: tuple[ControlChannel, ...]
    vehicle: VehicleConfigurationSpec
    exogenous: tuple[ExogenousChannel, ...] = ()
    observations: tuple[ObservationChannel, ...] = ()

    def __post_init__(self) -> None:
        if self.state_schema != RIGID_BODY_STATE_SCHEMA:
            raise ValueError(
                f"unsupported state schema {self.state_schema!r}; "
                f"expected {RIGID_BODY_STATE_SCHEMA!r}"
            )
        if not self.observation_source.strip():
            raise ValueError("observation_source cannot be empty")
        controls = tuple(self.controls)
        if not controls:
            raise ValueError("trajectory spec needs at least one control channel")
        names = tuple(channel.name for channel in controls)
        roles = tuple(channel.role for channel in controls)
        if len(set(names)) != len(names):
            raise ValueError("trajectory spec control names must be unique")
        if len(set(roles)) != len(roles):
            raise ValueError("trajectory spec control roles must be unique")
        exogenous = tuple(self.exogenous)
        exogenous_names = tuple(channel.name for channel in exogenous)
        exogenous_roles = tuple(channel.role for channel in exogenous)
        if len(set(exogenous_names)) != len(exogenous_names):
            raise ValueError("trajectory spec exogenous names must be unique")
        if len(set(exogenous_roles)) != len(exogenous_roles):
            raise ValueError("trajectory spec exogenous roles must be unique")
        observations = tuple(self.observations)
        observation_names = tuple(channel.name for channel in observations)
        observation_roles = tuple(channel.role for channel in observations)
        if len(set(observation_names)) != len(observation_names):
            raise ValueError("trajectory spec observation names must be unique")
        if len(set(observation_roles)) != len(observation_roles):
            raise ValueError("trajectory spec observation roles must be unique")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "exogenous", exogenous)
        object.__setattr__(self, "observations", observations)

    @property
    def control_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.controls)

    @property
    def control_roles(self) -> tuple[str, ...]:
        return tuple(channel.role for channel in self.controls)

    @property
    def control_semantics(self) -> tuple[str, ...]:
        return tuple(channel.semantic for channel in self.controls)

    @property
    def exogenous_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.exogenous)

    @property
    def exogenous_roles(self) -> tuple[str, ...]:
        return tuple(channel.role for channel in self.exogenous)

    @property
    def observation_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.observations)

    @property
    def observation_roles(self) -> tuple[str, ...]:
        return tuple(channel.role for channel in self.observations)

    def prediction_spec(self) -> TrajectorySpec:
        """Return the runtime contract, excluding training-only observations."""

        return TrajectorySpec(
            state_schema=self.state_schema,
            observation_source=self.observation_source,
            controls=self.controls,
            vehicle=self.vehicle,
            exogenous=self.exogenous,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_schema": self.state_schema,
            "observation_source": self.observation_source,
            "controls": [channel.to_dict() for channel in self.controls],
            "vehicle": self.vehicle.to_dict(),
            "exogenous": [channel.to_dict() for channel in self.exogenous],
            "observations": [channel.to_dict() for channel in self.observations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrajectorySpec:
        return cls(
            state_schema=str(payload["state_schema"]),
            observation_source=str(payload["observation_source"]),
            controls=tuple(
                ControlChannel.from_dict(channel)
                for channel in payload.get("controls", ())
            ),
            vehicle=VehicleConfigurationSpec.from_dict(payload["vehicle"]),
            exogenous=tuple(
                ExogenousChannel.from_dict(channel)
                for channel in payload.get("exogenous", ())
            ),
            observations=tuple(
                ObservationChannel.from_dict(channel)
                for channel in payload.get("observations", ())
            ),
        )


def _control_channel_for_name(name: str, platform: str) -> ControlChannel:
    fixed_wing_roles = {
        "throttle": "throttle",
        "aileron": "roll",
        "elevator": "pitch",
        "rudder": "yaw",
        "flap": "flap",
        "spoiler": "spoiler",
        "airbrake": "airbrake",
    }
    role = fixed_wing_roles.get(name, name)
    if name.startswith("motor_") or role == "throttle":
        minimum, maximum = 0.0, 1.0
    elif role in {"roll", "pitch", "yaw", "flap", "spoiler", "airbrake"}:
        minimum, maximum = -1.0, 1.0
    else:
        minimum, maximum = None, None
    semantic = (
        "normalized_generalized_command"
        if platform == "fixedwing" and role in {"roll", "pitch", "yaw"}
        else "normalized_command"
    )
    frame = "FLU" if role in {"roll", "pitch", "yaw"} else None
    return ControlChannel(
        name=name,
        role=role,
        semantic=semantic,
        unit="1",
        minimum=minimum,
        maximum=maximum,
        frame=frame,
    )


def make_trajectory_spec(
    control_names: Sequence[str],
    *,
    family: str,
    observation_source: str,
    configuration_id: str | None = None,
    fixed_states: Mapping[str, Any] | None = None,
    exogenous: Sequence[ExogenousChannel] = (),
    observations: Sequence[ObservationChannel] = (),
) -> TrajectorySpec:
    """Build the standard canonical contract for one vehicle control layout."""

    names = tuple(str(name) for name in control_names)
    channels = tuple(_control_channel_for_name(name, family) for name in names)
    roles = {channel.role for channel in channels}
    controlled_axes = tuple(axis for axis in ("roll", "pitch", "yaw") if axis in roles)
    if family == "multirotor" and any(name.startswith("motor_") for name in names):
        controlled_axes = ("roll", "pitch", "yaw")
        propulsion = "quadrotor" if len(names) == 4 else "distributed_electric"
    elif family == "fixedwing":
        propulsion = "single_propeller"
    else:
        propulsion = "unknown"
    auxiliary_controls = tuple(
        role for role in ("flap", "spoiler", "airbrake") if role in roles
    )
    return TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source=observation_source,
        controls=channels,
        vehicle=VehicleConfigurationSpec(
            family=family,
            configuration_id=configuration_id,
            controlled_axes=controlled_axes,
            propulsion=propulsion,
            fixed_states={} if fixed_states is None else fixed_states,
            auxiliary_controls=auxiliary_controls,
        ),
        exogenous=tuple(exogenous),
        observations=tuple(observations),
    )


@dataclass(frozen=True)
class Trajectory:
    """A complete flight or maneuver sampled on a common time base.

    States include both endpoints, so a trajectory with ``T`` states has
    ``T - 1`` controls. Control ``i`` is applied over the interval from state
    ``i`` to state ``i + 1``.

    Every array field is copied on construction and marked read-only, so a
    ``Trajectory`` never aliases the caller's arrays and mutating it after
    construction raises.

    ``control_prefix``, when given, holds the real controls immediately
    preceding this trajectory (oldest first, same channel count as
    ``controls``), for segments cut mid-flight so window extraction can pad
    early motor histories from true prior commands instead of repeating the
    segment's own first control. It is an in-memory-only convenience: it is
    never written by :func:`save_trajectory_npz` and is always ``None`` after
    :func:`load_trajectory_npz`.
    """

    time_s: npt.NDArray[np.float64]
    states: npt.NDArray[np.float64]
    controls: npt.NDArray[np.float64]
    spec: TrajectorySpec
    exogenous: npt.NDArray[np.float64] | None = None
    observations: npt.NDArray[np.float64] | None = None
    labels: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    control_prefix: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        time_s = np.array(self.time_s, dtype=np.float64, copy=True)
        states = np.array(self.states, dtype=np.float64, copy=True)
        controls = np.array(self.controls, dtype=np.float64, copy=True)
        if not isinstance(self.spec, TrajectorySpec):
            raise TypeError("spec must be a TrajectorySpec")
        exogenous = (
            np.empty((len(time_s), 0), dtype=np.float64)
            if self.exogenous is None
            else np.array(self.exogenous, dtype=np.float64, copy=True)
        )
        observations = (
            np.empty((len(time_s), 0), dtype=np.float64)
            if self.observations is None
            else np.array(self.observations, dtype=np.float64, copy=True)
        )
        control_prefix = (
            None
            if self.control_prefix is None
            else np.array(self.control_prefix, dtype=np.float64, copy=True)
        )

        if time_s.ndim != 1:
            raise ValueError("time_s must be one-dimensional")
        if len(time_s) < 2:
            raise ValueError("a trajectory needs at least two timestamps")
        if states.shape != (len(time_s), STATE_SIZE):
            raise ValueError(
                f"states must have shape ({len(time_s)}, {STATE_SIZE}), "
                f"got {states.shape}"
            )
        if controls.ndim != 2 or controls.shape[0] != len(time_s) - 1:
            raise ValueError(
                "controls must have shape "
                f"({len(time_s) - 1}, positive channel count), got {controls.shape}"
            )
        if controls.shape[1] < 1:
            raise ValueError("controls must contain at least one channel")
        if control_prefix is not None:
            if control_prefix.ndim != 2 or control_prefix.shape[1] != controls.shape[1]:
                raise ValueError(
                    "control_prefix must have shape (history, "
                    f"{controls.shape[1]}), got {control_prefix.shape}"
                )
            if not np.all(np.isfinite(control_prefix)):
                raise ValueError("control_prefix values must be finite")
        if exogenous.shape != (len(time_s), len(self.spec.exogenous)):
            raise ValueError(
                "exogenous must have shape "
                f"({len(time_s)}, {len(self.spec.exogenous)}), got {exogenous.shape}"
            )
        if observations.shape != (len(time_s), len(self.spec.observations)):
            raise ValueError(
                "observations must have shape "
                f"({len(time_s)}, {len(self.spec.observations)}), "
                f"got {observations.shape}"
            )
        if not np.all(np.diff(time_s) > 0.0):
            raise ValueError("timestamps must be strictly increasing")
        if not (
            np.all(np.isfinite(time_s))
            and np.all(np.isfinite(states))
            and np.all(np.isfinite(controls))
            and np.all(np.isfinite(exogenous))
            and np.all(np.isfinite(observations))
        ):
            raise ValueError("trajectory values must be finite")

        if len(self.spec.controls) != controls.shape[1]:
            raise ValueError(
                "trajectory spec must contain one channel per control column"
            )
        labels = dict(self.labels)
        provenance = dict(self.provenance)
        if any(not isinstance(key, str) or not key for key in labels):
            raise ValueError("trajectory label keys must be non-empty strings")
        if any(not isinstance(key, str) or not key for key in provenance):
            raise ValueError("trajectory provenance keys must be non-empty strings")

        time_s.setflags(write=False)
        states.setflags(write=False)
        controls.setflags(write=False)
        exogenous.setflags(write=False)
        observations.setflags(write=False)
        if control_prefix is not None:
            control_prefix.setflags(write=False)

        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "exogenous", exogenous)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "control_prefix", control_prefix)

    @property
    def control_names(self) -> tuple[str, ...]:
        return self.spec.control_names

    @property
    def nominal_dt_s(self) -> float:
        """Return the median sample interval."""

        return float(np.median(np.diff(self.time_s)))

    @property
    def control_size(self) -> int:
        """Return the number of actuator or command channels."""

        return int(self.controls.shape[1])

    @property
    def exogenous_size(self) -> int:
        """Return the number of measured non-control input channels."""

        return int(self.exogenous.shape[1])

    @property
    def observation_size(self) -> int:
        """Return the number of state-aligned identification measurements."""

        return int(self.observations.shape[1])


@dataclass(frozen=True)
class TrajectoryWindows:
    """Fixed-horizon rollout windows from one or more trajectories."""

    initial_states: npt.NDArray[np.float64]
    control_histories: npt.NDArray[np.float64]
    controls: npt.NDArray[np.float64]
    target_states: npt.NDArray[np.float64]
    dt_s: float
    initial_exogenous: npt.NDArray[np.float64] | None = None
    window_weights: npt.NDArray[np.float64] | None = None
    trajectory_indices: npt.NDArray[np.int64] | None = None
    start_indices: npt.NDArray[np.int64] | None = None
    candidate_window_counts: npt.NDArray[np.int64] | None = None
    selection_policy: str = "all_candidates"
    control_names: tuple[str, ...] | None = None
    control_roles: tuple[str, ...] | None = None
    control_semantics: tuple[str, ...] | None = None
    exogenous_names: tuple[str, ...] | None = None
    exogenous_roles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        count = self.initial_states.shape[0]
        if self.initial_states.shape != (count, STATE_SIZE):
            raise ValueError("initial_states has an invalid shape")
        if self.controls.ndim != 3 or self.controls.shape[0] != count:
            raise ValueError("controls must have shape (windows, horizon, controls)")
        if self.controls.shape[2] < 1:
            raise ValueError("controls must contain at least one channel")
        if (
            self.control_histories.ndim != 3
            or self.control_histories.shape[0] != count
            or self.control_histories.shape[2] != self.controls.shape[2]
            or self.control_histories.shape[1] < 1
        ):
            raise ValueError(
                "control_histories must have shape "
                "(windows, positive history, controls)"
            )
        expected_targets = (count, self.controls.shape[1] + 1, STATE_SIZE)
        if self.target_states.shape != expected_targets:
            raise ValueError(
                f"target_states must have shape {expected_targets}, "
                f"got {self.target_states.shape}"
            )
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        initial_exogenous = (
            np.empty((count, 0), dtype=np.float64)
            if self.initial_exogenous is None
            else np.asarray(self.initial_exogenous, dtype=np.float64)
        )
        if initial_exogenous.ndim != 2 or initial_exogenous.shape[0] != count:
            raise ValueError(
                "initial_exogenous must have shape (windows, exogenous channels)"
            )
        if not np.all(np.isfinite(initial_exogenous)):
            raise ValueError("initial_exogenous values must be finite")
        object.__setattr__(self, "initial_exogenous", initial_exogenous)
        control_names = self.control_names
        if control_names is None:
            control_names = tuple(
                f"control_{index}" for index in range(self.controls.shape[2])
            )
        else:
            control_names = tuple(control_names)
        if len(control_names) != self.controls.shape[2]:
            raise ValueError("control_names must contain one name per control channel")
        if any(not name.strip() for name in control_names):
            raise ValueError("control_names cannot contain empty names")
        if len(set(control_names)) != len(control_names):
            raise ValueError("control_names must be unique")
        object.__setattr__(self, "control_names", control_names)
        control_roles = self.control_roles
        if control_roles is None:
            control_roles = control_names
        else:
            control_roles = tuple(control_roles)
        if len(control_roles) != self.controls.shape[2]:
            raise ValueError("control_roles must contain one role per control channel")
        if any(not role.strip() for role in control_roles):
            raise ValueError("control_roles cannot contain empty values")
        if len(set(control_roles)) != len(control_roles):
            raise ValueError("control_roles must be unique")
        object.__setattr__(self, "control_roles", control_roles)
        control_semantics = self.control_semantics
        if control_semantics is None:
            control_semantics = tuple("unspecified" for _ in control_names)
        else:
            control_semantics = tuple(control_semantics)
        if len(control_semantics) != self.controls.shape[2]:
            raise ValueError(
                "control_semantics must contain one semantic per control channel"
            )
        if any(not semantic.strip() for semantic in control_semantics):
            raise ValueError("control_semantics cannot contain empty values")
        object.__setattr__(self, "control_semantics", control_semantics)
        exogenous_names = (
            tuple(f"exogenous_{index}" for index in range(initial_exogenous.shape[1]))
            if self.exogenous_names is None
            else tuple(self.exogenous_names)
        )
        exogenous_roles = (
            exogenous_names
            if self.exogenous_roles is None
            else tuple(self.exogenous_roles)
        )
        if len(exogenous_names) != initial_exogenous.shape[1]:
            raise ValueError(
                "exogenous_names must contain one name per exogenous channel"
            )
        if len(exogenous_roles) != initial_exogenous.shape[1]:
            raise ValueError(
                "exogenous_roles must contain one role per exogenous channel"
            )
        if len(set(exogenous_names)) != len(exogenous_names):
            raise ValueError("exogenous_names must be unique")
        if len(set(exogenous_roles)) != len(exogenous_roles):
            raise ValueError("exogenous_roles must be unique")
        object.__setattr__(self, "exogenous_names", exogenous_names)
        object.__setattr__(self, "exogenous_roles", exogenous_roles)
        if self.window_weights is not None:
            window_weights = np.asarray(self.window_weights, dtype=np.float64)
            if window_weights.shape != (count,):
                raise ValueError("window_weights must have shape (windows,)")
            if not np.all(np.isfinite(window_weights)) or np.any(window_weights <= 0.0):
                raise ValueError("window_weights must be finite and positive")
            object.__setattr__(self, "window_weights", window_weights)
        if self.trajectory_indices is not None:
            trajectory_indices = np.asarray(self.trajectory_indices, dtype=np.int64)
            if trajectory_indices.shape != (count,):
                raise ValueError("trajectory_indices must have shape (windows,)")
            if np.any(trajectory_indices < 0):
                raise ValueError("trajectory_indices cannot be negative")
            object.__setattr__(self, "trajectory_indices", trajectory_indices)
        if self.start_indices is not None:
            start_indices = np.asarray(self.start_indices, dtype=np.int64)
            if start_indices.shape != (count,):
                raise ValueError("start_indices must have shape (windows,)")
            if np.any(start_indices < 0):
                raise ValueError("start_indices cannot be negative")
            object.__setattr__(self, "start_indices", start_indices)
        if self.candidate_window_counts is not None:
            candidate_window_counts = np.asarray(
                self.candidate_window_counts, dtype=np.int64
            )
            if candidate_window_counts.ndim != 1:
                raise ValueError("candidate_window_counts must be one-dimensional")
            if np.any(candidate_window_counts < 0):
                raise ValueError("candidate_window_counts cannot be negative")
            if int(np.sum(candidate_window_counts)) < count:
                raise ValueError(
                    "candidate_window_counts cannot total fewer than selected windows"
                )
            if (
                self.trajectory_indices is not None
                and count > 0
                and int(np.max(self.trajectory_indices)) >= len(candidate_window_counts)
            ):
                raise ValueError(
                    "candidate_window_counts must cover every trajectory index"
                )
            object.__setattr__(self, "candidate_window_counts", candidate_window_counts)
        if not self.selection_policy.strip():
            raise ValueError("selection_policy cannot be empty")

    @property
    def control_size(self) -> int:
        """Return the number of control channels in each rollout window."""

        return int(self.controls.shape[2])

    @property
    def candidate_window_count(self) -> int:
        """Return the number of windows available before deterministic selection."""

        if self.candidate_window_counts is None:
            return len(self.initial_states)
        return int(np.sum(self.candidate_window_counts))


def _bounded_weighted_allocation(
    capacities: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64],
    budget: int,
) -> npt.NDArray[np.int64]:
    """Allocate an integer budget proportionally without exceeding capacities."""

    allocation = np.zeros_like(capacities)
    remaining = min(int(budget), int(np.sum(capacities)))
    while remaining > 0:
        available = allocation < capacities
        active_weights = np.where(available, weights, 0.0)
        weight_total = float(np.sum(active_weights))
        if weight_total <= 0.0:
            break
        ideal = remaining * active_weights / weight_total
        additions = np.minimum(
            np.floor(ideal).astype(np.int64), capacities - allocation
        )
        added = int(np.sum(additions))
        if added > 0:
            allocation += additions
            remaining -= added
            continue

        fractional_order = np.argsort(-ideal, kind="stable")
        for index in fractional_order:
            if allocation[index] >= capacities[index]:
                continue
            allocation[index] += 1
            remaining -= 1
            if remaining == 0:
                break
    return allocation


def _selected_window_locations(
    candidate_counts: npt.NDArray[np.int64],
    *,
    stride: int,
    maximum_windows: int | None,
    balance_trajectories: bool,
    trajectory_weights: Sequence[float] | None,
    trajectory_groups: Sequence[str | int] | None,
    trajectory_group_weights: Mapping[str | int, float] | None,
) -> tuple[list[tuple[int, int]], str]:
    """Select midpoint-stratified windows without materializing all candidates."""

    candidate_total = int(np.sum(candidate_counts))
    if maximum_windows is None or candidate_total <= maximum_windows:
        return (
            [
                (trajectory_index, candidate_index * stride)
                for trajectory_index, count in enumerate(candidate_counts)
                for candidate_index in range(int(count))
            ],
            "all_candidates",
        )

    if trajectory_groups is not None:
        group_order = tuple(dict.fromkeys(trajectory_groups))
        strata = [
            [
                index
                for index, group in enumerate(trajectory_groups)
                if group == selected_group
            ]
            for selected_group in group_order
        ]
        stratum_weights = np.asarray(
            [
                1.0
                if trajectory_group_weights is None
                else trajectory_group_weights[group]
                for group in group_order
            ],
            dtype=np.float64,
        )
    elif balance_trajectories:
        strata = [[index] for index in range(len(candidate_counts))]
        stratum_weights = np.ones(len(strata), dtype=np.float64)
    elif trajectory_weights is not None:
        strata = [[index] for index in range(len(candidate_counts))]
        stratum_weights = np.asarray(trajectory_weights, dtype=np.float64)
    else:
        strata = [list(range(len(candidate_counts)))]
        stratum_weights = np.ones(1, dtype=np.float64)

    capacities = np.asarray(
        [sum(int(candidate_counts[index]) for index in stratum) for stratum in strata],
        dtype=np.int64,
    )
    allocations = _bounded_weighted_allocation(
        capacities,
        stratum_weights,
        maximum_windows,
    )
    locations: list[tuple[int, int]] = []
    for stratum, capacity, allocation in zip(strata, capacities, allocations):
        if allocation == 0:
            continue
        candidate_ordinals = (
            (2 * np.arange(int(allocation), dtype=np.int64) + 1) * int(capacity)
        ) // (2 * int(allocation))
        member_counts = candidate_counts[stratum]
        cumulative_counts = np.cumsum(member_counts)
        member_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), cumulative_counts[:-1])
        )
        member_positions = np.searchsorted(
            cumulative_counts, candidate_ordinals, side="right"
        )
        for ordinal, member_position in zip(candidate_ordinals, member_positions):
            trajectory_index = stratum[int(member_position)]
            candidate_index = int(ordinal - member_offsets[member_position])
            locations.append((trajectory_index, candidate_index * stride))

    locations.sort()
    return locations, "deterministic_stratified_midpoint"


def _control_window_history(
    trajectory: Trajectory, start: int, motor_history_steps: int
) -> npt.NDArray[np.float64]:
    """Return the ``motor_history_steps`` controls immediately before ``start``.

    Prefers real history: the trajectory's own prior controls first, then any
    ``control_prefix`` carried from a parent trajectory (populated by
    :func:`trajectory_segment`/:func:`split_trajectory` for a segment cut
    mid-flight). Only pads with a repeated earliest known control when
    neither source has enough samples, e.g. at the true start of a flight.
    """

    history_start = max(0, start - motor_history_steps)
    local_history = trajectory.controls[history_start:start]
    deficit = motor_history_steps - len(local_history)
    if deficit <= 0:
        return local_history
    if trajectory.control_prefix is not None and len(trajectory.control_prefix):
        prefix = trajectory.control_prefix[-deficit:]
    else:
        prefix = np.empty((0, trajectory.control_size), dtype=np.float64)
    deficit -= len(prefix)
    if deficit <= 0:
        return np.concatenate((prefix, local_history), axis=0)
    fill_source = (
        prefix[0]
        if len(prefix)
        else local_history[0]
        if len(local_history)
        else trajectory.controls[0]
    )
    padding = np.repeat(fill_source[np.newaxis, :], deficit, axis=0)
    return np.concatenate((padding, prefix, local_history), axis=0)


def trajectory_windows(
    trajectories: list[Trajectory] | tuple[Trajectory, ...],
    *,
    horizon: int,
    stride: int | None = None,
    motor_history_s: float = 1.0,
    dt_tolerance_s: float = 1e-7,
    balance_trajectories: bool = False,
    trajectory_weights: Sequence[float] | None = None,
    trajectory_groups: Sequence[str | int] | None = None,
    trajectory_group_weights: Mapping[str | int, float] | None = None,
    maximum_windows: int | None = None,
) -> TrajectoryWindows:
    """Extract rollout windows without crossing flight boundaries.

    When ``balance_trajectories`` is enabled, each trajectory contributes equal
    total loss weight regardless of its duration or number of extracted windows.
    ``maximum_windows`` applies deterministic midpoint sampling across the same
    weighting strata, preserving broad temporal and source coverage without first
    materializing every candidate window. ``trajectory_group_weights`` changes
    the total contribution of each group while retaining uniform weight among
    windows in that group; a zero group weight excludes that group's windows. It
    is intended for complete-group resampling.
    """

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if stride is None:
        stride = horizon
    if stride < 1:
        raise ValueError("stride must be positive")
    if motor_history_s <= 0.0:
        raise ValueError("motor_history_s must be positive")
    if maximum_windows is not None and maximum_windows < 1:
        raise ValueError("maximum_windows must be positive")
    weighting_modes = sum(
        (
            bool(balance_trajectories),
            trajectory_weights is not None,
            trajectory_groups is not None,
        )
    )
    if weighting_modes > 1:
        raise ValueError(
            "balance_trajectories, trajectory_weights, and trajectory_groups "
            "are mutually exclusive"
        )
    if trajectory_weights is not None:
        if len(trajectory_weights) != len(trajectories):
            raise ValueError("trajectory_weights must match trajectories")
        if any(weight <= 0.0 for weight in trajectory_weights):
            raise ValueError("trajectory_weights must be positive")
    if trajectory_groups is not None:
        if len(trajectory_groups) != len(trajectories):
            raise ValueError("trajectory_groups must match trajectories")
        trajectory_groups = tuple(trajectory_groups)
        if any(
            not isinstance(group, (str, int))
            or (isinstance(group, str) and not group.strip())
            for group in trajectory_groups
        ):
            raise ValueError(
                "trajectory_groups must contain non-empty strings or integers"
            )
    if trajectory_group_weights is not None:
        if trajectory_groups is None:
            raise ValueError("trajectory_group_weights requires trajectory_groups")
        group_order = tuple(dict.fromkeys(trajectory_groups))
        if set(trajectory_group_weights) != set(group_order):
            raise ValueError(
                "trajectory_group_weights must contain exactly the trajectory groups"
            )
        weights = np.asarray(list(trajectory_group_weights.values()), dtype=np.float64)
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise ValueError(
                "trajectory_group_weights values must be finite and nonnegative "
                "with at least one positive group"
            )

    dt_s = trajectories[0].nominal_dt_s
    control_size = trajectories[0].control_size
    control_names = trajectories[0].control_names
    control_roles = trajectories[0].spec.control_roles
    control_semantics = trajectories[0].spec.control_semantics
    exogenous_names = trajectories[0].spec.exogenous_names
    exogenous_roles = trajectories[0].spec.exogenous_roles
    history_ratio = motor_history_s / dt_s
    nearest_history_steps = round(history_ratio)
    if np.isclose(history_ratio, nearest_history_steps, atol=1e-9, rtol=0.0):
        motor_history_steps = int(nearest_history_steps)
    else:
        motor_history_steps = int(np.ceil(history_ratio))
    motor_history_steps = max(1, motor_history_steps)
    candidate_counts: list[int] = []

    for trajectory_index, trajectory in enumerate(trajectories):
        if trajectory.control_size != control_size:
            raise ValueError(
                "all trajectories must have the same control channel count"
            )
        if trajectory.control_names != control_names:
            raise ValueError(
                "all trajectories must have the same ordered control_names"
            )
        if trajectory.spec.control_roles != control_roles:
            raise ValueError(
                "all trajectories must have the same ordered control_roles"
            )
        if trajectory.spec.control_semantics != control_semantics:
            raise ValueError(
                "all trajectories must have the same ordered control_semantics"
            )
        if trajectory.spec.exogenous_names != exogenous_names:
            raise ValueError(
                "all trajectories must have the same ordered exogenous_names"
            )
        if trajectory.spec.exogenous_roles != exogenous_roles:
            raise ValueError(
                "all trajectories must have the same ordered exogenous_roles"
            )
        intervals = np.diff(trajectory.time_s)
        if not np.allclose(intervals, dt_s, atol=dt_tolerance_s, rtol=0.0):
            raise ValueError(
                "all trajectories must have the same fixed sample interval"
            )

        candidate_count = len(range(0, len(trajectory.controls) - horizon + 1, stride))
        candidate_counts.append(candidate_count)
        if weighting_modes and candidate_count == 0:
            raise ValueError(
                f"trajectory {trajectory_index} is too short for horizon {horizon}"
            )

    candidate_count_array = np.asarray(candidate_counts, dtype=np.int64)
    if not np.any(candidate_count_array):
        raise ValueError("no windows fit within the provided trajectories")

    selection_candidate_counts = candidate_count_array.copy()
    if trajectory_groups is not None and trajectory_group_weights is not None:
        selection_candidate_counts = np.asarray(
            [
                count if trajectory_group_weights[trajectory_groups[index]] > 0.0 else 0
                for index, count in enumerate(candidate_count_array)
            ],
            dtype=np.int64,
        )
    selected_locations, selection_policy = _selected_window_locations(
        selection_candidate_counts,
        stride=stride,
        maximum_windows=maximum_windows,
        balance_trajectories=balance_trajectories,
        trajectory_weights=trajectory_weights,
        trajectory_groups=trajectory_groups,
        trajectory_group_weights=trajectory_group_weights,
    )
    if (
        not np.array_equal(selection_candidate_counts, candidate_count_array)
        and selection_policy == "all_candidates"
    ):
        selection_policy = "all_positive_weight_candidates"
    initial_states: list[npt.NDArray[np.float64]] = []
    control_histories: list[npt.NDArray[np.float64]] = []
    controls: list[npt.NDArray[np.float64]] = []
    targets: list[npt.NDArray[np.float64]] = []
    initial_exogenous: list[npt.NDArray[np.float64]] = []
    trajectory_indices: list[int] = []
    start_indices: list[int] = []
    for trajectory_index, start in selected_locations:
        trajectory = trajectories[trajectory_index]
        stop = start + horizon
        initial_states.append(trajectory.states[start])
        control_histories.append(
            _control_window_history(trajectory, start, motor_history_steps)
        )
        controls.append(trajectory.controls[start:stop])
        targets.append(trajectory.states[start : stop + 1])
        initial_exogenous.append(trajectory.exogenous[start])
        trajectory_indices.append(trajectory_index)
        start_indices.append(start)

    trajectory_index_array = np.asarray(trajectory_indices, dtype=np.int64)
    window_weights = None
    if balance_trajectories:
        counts = np.bincount(trajectory_index_array, minlength=len(trajectories))
        window_weights = 1.0 / counts[trajectory_index_array]
    elif trajectory_weights is not None:
        counts = np.bincount(trajectory_index_array, minlength=len(trajectories))
        weights = np.asarray(trajectory_weights, dtype=np.float64)
        window_weights = (
            weights[trajectory_index_array] / counts[trajectory_index_array]
        )
    elif trajectory_groups is not None:
        group_order = tuple(dict.fromkeys(trajectory_groups))
        group_indices = np.asarray(
            [
                group_order.index(trajectory_groups[index])
                for index in trajectory_index_array
            ],
            dtype=np.int64,
        )
        group_counts = np.bincount(group_indices, minlength=len(group_order))
        group_weights = np.asarray(
            [
                1.0
                if trajectory_group_weights is None
                else trajectory_group_weights[group]
                for group in group_order
            ],
            dtype=np.float64,
        )
        window_weights = group_weights[group_indices] / group_counts[group_indices]

    return TrajectoryWindows(
        initial_states=np.stack(initial_states),
        control_histories=np.stack(control_histories),
        controls=np.stack(controls),
        target_states=np.stack(targets),
        dt_s=dt_s,
        initial_exogenous=np.stack(initial_exogenous),
        window_weights=window_weights,
        trajectory_indices=trajectory_index_array,
        start_indices=np.asarray(start_indices, dtype=np.int64),
        candidate_window_counts=selection_candidate_counts,
        selection_policy=selection_policy,
        control_names=control_names,
        control_roles=control_roles,
        control_semantics=control_semantics,
        exogenous_names=exogenous_names,
        exogenous_roles=exogenous_roles,
    )


def save_trajectory_npz(trajectory: Trajectory, path: str | Path) -> None:
    """Write a canonical trajectory to a compressed, self-describing NPZ file.

    ``trajectory.control_prefix`` is in-memory only and is not written here;
    :func:`load_trajectory_npz` always returns ``control_prefix=None``.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_version=np.asarray(3, dtype=np.int64),
        time_s=trajectory.time_s,
        states=trajectory.states,
        controls=trajectory.controls,
        exogenous=trajectory.exogenous,
        observations=trajectory.observations,
        spec_json=np.asarray(json.dumps(trajectory.spec.to_dict(), sort_keys=True)),
        labels_json=np.asarray(json.dumps(dict(trajectory.labels), sort_keys=True)),
        provenance_json=np.asarray(
            json.dumps(dict(trajectory.provenance), sort_keys=True)
        ),
    )


def load_trajectory_npz(path: str | Path) -> Trajectory:
    """Load a canonical trajectory written by :func:`save_trajectory_npz`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        version = int(archive["format_version"])
        if version != 3:
            raise ValueError(f"unsupported trajectory format version: {version}")
        return Trajectory(
            time_s=archive["time_s"],
            states=archive["states"],
            controls=archive["controls"],
            exogenous=archive["exogenous"],
            observations=archive["observations"],
            spec=TrajectorySpec.from_dict(json.loads(str(archive["spec_json"]))),
            labels=json.loads(str(archive["labels_json"])),
            provenance=json.loads(str(archive["provenance_json"])),
        )


def trajectory_segment(
    trajectory: Trajectory, start_interval: int, stop_interval: int
) -> Trajectory:
    """Return a contiguous interval slice, including both endpoint states."""

    interval_count = len(trajectory.controls)
    if not 0 <= start_interval < stop_interval <= interval_count:
        raise ValueError(
            f"segment bounds must satisfy 0 <= start < stop <= {interval_count}"
        )
    provenance = dict(trajectory.provenance)
    transformations = list(provenance.get("transformations", ()))
    transformations.append(
        {
            "type": "interval_segment",
            "start_interval": start_interval,
            "stop_interval": stop_interval,
            "source_interval_count": interval_count,
        }
    )
    provenance["transformations"] = transformations
    start_time_s = trajectory.time_s[start_interval]
    if start_interval > 0:
        # The controls dropped from the front of this segment are real prior
        # commands, not fabricated ones; carry them (and any prefix the
        # parent already carried) so window extraction can pad early motor
        # histories from what was actually commanded before this segment.
        local_prefix = trajectory.controls[:start_interval]
        if trajectory.control_prefix is not None and len(trajectory.control_prefix):
            control_prefix = np.concatenate(
                (trajectory.control_prefix, local_prefix), axis=0
            )
        else:
            control_prefix = local_prefix
    else:
        control_prefix = trajectory.control_prefix
    return Trajectory(
        time_s=trajectory.time_s[start_interval : stop_interval + 1] - start_time_s,
        states=trajectory.states[start_interval : stop_interval + 1],
        controls=trajectory.controls[start_interval:stop_interval],
        spec=trajectory.spec,
        exogenous=trajectory.exogenous[start_interval : stop_interval + 1],
        observations=trajectory.observations[start_interval : stop_interval + 1],
        labels=trajectory.labels,
        provenance=provenance,
        control_prefix=control_prefix,
    )


def split_trajectory(
    trajectory: Trajectory, *, train_fraction: float = 0.70
) -> tuple[Trajectory, Trajectory]:
    """Split one flight into contiguous training and validation segments."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    interval_count = len(trajectory.controls)
    split = round(interval_count * train_fraction)
    split = min(max(split, 1), interval_count - 1)
    return (
        trajectory_segment(trajectory, 0, split),
        trajectory_segment(trajectory, split, interval_count),
    )
