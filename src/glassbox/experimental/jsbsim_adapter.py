"""Fresh-instance JSBSim recording adapter for the frozen onboarding audit.

This module does not trim, start engines, fit a learner, or repair configurations.
The caller owns inventory/runtime verification, timeouts, replay, and sealing.
"""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

SURFACE_COMMANDS = (
    "fcs/aileron-cmd-norm",
    "fcs/elevator-cmd-norm",
    "fcs/rudder-cmd-norm",
)
OBSERVATION_PROPERTIES = (
    "velocities/v-north-fps",
    "velocities/v-east-fps",
    "velocities/v-down-fps",
    "velocities/p-rad_sec",
    "velocities/q-rad_sec",
    "velocities/r-rad_sec",
    "position/h-sl-ft",
)
FEET_TO_METERS = 0.3048


class SimulationFailure(RuntimeError):
    """An expected simulator/configuration failure, not a harness exception."""


def _call(function, *args):
    # JSBSim BaseError and its specialized exceptions derive from RuntimeError.
    # Keep this boundary narrow: Python programming errors must abort the harness.
    try:
        return function(*args)
    except RuntimeError as error:
        raise SimulationFailure(f"{type(error).__name__}: {error}") from error


def _read(fdm: Any, name: str) -> float:
    return _call(fdm.__getitem__, name)


def _write(fdm: Any, name: str, value: float) -> None:
    _call(fdm.__setitem__, name, value)


class _ObservationFailure(SimulationFailure):
    def __init__(self, error: Exception, raw: np.ndarray, mapped: np.ndarray, time: float):
        super().__init__(str(error))
        self.raw = raw
        self.mapped = mapped
        self.time = time


def _asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Asset is not a file within the pinned root: {relative}")
    return path


def _catalog(fdm: Any) -> set[str]:
    return {line.rsplit(" (", 1)[0].strip() for line in _call(fdm.get_property_catalog)}


def _require(catalog: set[str], names: tuple[str, ...] | list[str]) -> None:
    missing = sorted(set(names) - catalog)
    if missing:
        raise SimulationFailure(f"Required properties absent from catalog: {missing}")


def _observation(fdm: Any) -> tuple[np.ndarray, np.ndarray, float]:
    raw = np.full(16, np.nan, dtype=np.float64)
    mapped = raw.copy()
    time = float("nan")
    try:
        for index, name in enumerate(OBSERVATION_PROPERTIES[:6]):
            raw[index] = _read(fdm, name)
        propagate = _call(fdm.get_propagate)
        matrix = np.asarray(_call(propagate.get_Tl2b), dtype=np.float64)
        if matrix.shape != (3, 3):
            raise SimulationFailure(f"Local-to-body matrix has shape {matrix.shape}")
        raw[6:15] = matrix.reshape(9)
        raw[15] = _read(fdm, OBSERVATION_PROPERTIES[-1])
        time = float(_call(fdm.get_sim_time))
        mapped[:] = raw
        mapped[:3] *= FEET_TO_METERS
        mapped[6:15] = matrix.T.reshape(9)
        mapped[15] *= FEET_TO_METERS
        if not np.isfinite(raw).all() or not np.isfinite(mapped).all():
            raise SimulationFailure("Nonfinite observation")
        if not np.isfinite(time):
            raise SimulationFailure("Nonfinite simulation time")
        rotation = mapped[6:15].reshape(3, 3)
        orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if orthogonality > 1e-8 or abs(determinant - 1.0) > 1e-8:
            raise SimulationFailure(
                f"Invalid rotation: orthogonality={orthogonality!r}, determinant={determinant!r}"
            )
    except SimulationFailure as error:
        raise _ObservationFailure(error, raw, mapped, time) from error
    return raw, mapped, time


def _diagnostics(fdm: Any, catalog: set[str], engine_count: int) -> dict:
    names = [
        "position/h-agl-ft", "velocities/vtrue-fps", "velocities/vc-fps",
        "velocities/u-fps", "velocities/v-fps", "velocities/w-fps",
        "aero/alpha-rad", "aero/beta-rad", "gear/gear-pos-norm",
        "fcs/elevator-pos-rad", "fcs/left-aileron-pos-rad",
        "fcs/right-aileron-pos-rad", "fcs/rudder-pos-rad",
        "simulation/randomseed", "simulation/trim-completed",
        "simulation/integrator/rate/rotational",
        "simulation/integrator/rate/translational",
        "simulation/integrator/position/rotational",
        "simulation/integrator/position/translational",
    ]
    for index in range(engine_count):
        suffix = "" if index == 0 else f"[{index}]"
        names.append(f"fcs/throttle-pos-norm{suffix}")
        for field in ("running", "rpm", "n1", "n2", "thrust-lbs"):
            names.append(f"propulsion/engine{suffix}/{field}")
    values: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name in names:
        if name not in catalog:
            continue
        try:
            value = float(_read(fdm, name))
            values[name] = value if np.isfinite(value) else None
        except SimulationFailure as error:
            errors[name] = f"{type(error).__name__}: {error}"
    running = {name: value for name, value in values.items() if name.endswith("/running")}
    agl = values.get("position/h-agl-ft")
    return {
        "initial_properties": values,
        "initial_property_errors": errors,
        "initial_running_properties": running,
        "initial_ground_clearance_m": None if agl is None else agl * FEET_TO_METERS,
    }


def _tape(protocol: dict, entry: dict, initial: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    digest = hashlib.sha256(f"{protocol['id']}:{entry['id']}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    random = np.random.Generator(np.random.PCG64(seed))
    count = protocol["recording"]["transitions"]
    tape = np.empty((count, len(initial)), dtype=np.float64)
    for start in range(0, count, 5):
        offset = random.uniform(-1.0, 1.0, size=len(initial))
        command = np.clip(initial + 0.05 * np.diff(bounds, axis=1)[:, 0] * offset,
                          bounds[:, 0], bounds[:, 1])
        tape[start:start + 5] = command
    return tape


def simulate(
    root: Path, entry: dict, commands: np.ndarray | None, protocol: dict, *, progress=None
) -> dict:
    """Run one fresh initialized recording, preserving any failed prefix.

    ``commands`` in the result always contains the complete intended tape.
    Readback rows cover attempted intervals; NaNs and explicit native-step masks
    distinguish unattempted slots from returned data. Valid observation arrays
    contain initial/interval-boundary snapshots only. A rejected snapshot is kept
    separately, including partially read coordinates if a getter raised.
    """
    result: dict[str, Any] = {
        "status": "load_failure", "stage": "construct", "error": None,
        "failed_interval": None, "failed_substep": None,
        "command_names": [], "bounds": [], "diagnostics": {},
    }
    tape = np.empty((0, 0), dtype=np.float64)
    native_rows: list[np.ndarray] = []
    observation_rows: list[np.ndarray] = []
    times: list[float] = []
    immediate = np.empty((0, 6, 0), dtype=np.float64)
    end = immediate.copy()
    attempted = np.empty((0, 6), dtype=np.float64)
    completed = attempted.copy()
    attempted_intervals = 0
    failed_raw = np.empty((0, 16), dtype=np.float64)
    failed_observation = failed_raw.copy()
    failed_time = np.empty(0, dtype=np.float64)

    def finish() -> dict:
        result["completed_intervals"] = max(0, len(observation_rows) - 1)
        result["arrays"] = {
            "commands": tape,
            "observations": np.asarray(observation_rows, dtype=np.float64).reshape(-1, 16),
            "native_observations": np.asarray(native_rows, dtype=np.float64).reshape(-1, 16),
            "time_s": np.asarray(times, dtype=np.float64),
            "immediate_readbacks": immediate[:attempted_intervals],
            "end_readbacks": end[:attempted_intervals],
            "native_attempted": attempted[:attempted_intervals],
            "native_completed": completed[:attempted_intervals],
            "failed_native_observation": failed_raw,
            "failed_observation": failed_observation,
            "failed_time_s": failed_time,
        }
        return result

    def publish(*, terminal: bool = False) -> dict:
        current = finish()
        if progress is not None:
            snapshot = deepcopy(current)
            if not terminal:
                snapshot["status"] = "running"
            progress(snapshot)
        return current

    def breadcrumb() -> None:
        if progress is not None:
            progress({
                "progress_only": True, "stage": result["stage"],
                "failed_interval": result["failed_interval"],
                "failed_substep": result["failed_substep"],
            })

    try:
        if commands is not None:
            tape = np.array(commands, dtype=np.float64, copy=True)
        import jsbsim

        root = Path(root).resolve(strict=True)
        model_path = _asset(root, entry["path"])
        os.environ["JSBSIM_DISPERSE"] = "0"
        breadcrumb()
        fdm = _call(jsbsim.FGFDMExec, str(root))
        _call(fdm.set_debug_level, 0)
        _call(fdm.disable_input)
        _call(fdm.disable_output)
        # JSBSim builds its catalog only after loading a model. A noncreating
        # node lookup checks this constructor-bound property before that load.
        result["stage"] = "random_seed"
        manager = _call(fdm.get_property_manager)
        if _call(manager.get_node, "simulation/randomseed", False) is None:
            raise SimulationFailure("Required pre-load simulation/randomseed property is absent")
        _write(fdm, "simulation/randomseed", 0)
        result["stage"] = "model_load"
        breadcrumb()
        _call(fdm.set_aircraft_path, str(model_path.parent))
        _call(fdm.set_engine_path, str(root / "engine"))
        _call(fdm.set_systems_path, str(root / "systems"))
        if not _call(fdm.load_model, model_path.stem, False):
            raise SimulationFailure("load_model returned false")
        _call(fdm.disable_input)
        _call(fdm.disable_output)
        result["status"] = "initialization_failure"
        result["stage"] = "command_contract"
        propulsion = _call(fdm.get_propulsion)
        engine_count = int(_call(propulsion.get_num_engines))
        if engine_count < 0:
            raise SimulationFailure("Negative engine count")
        names = list(SURFACE_COMMANDS) + [
            "fcs/throttle-cmd-norm" + ("" if index == 0 else f"[{index}]")
            for index in range(engine_count)
        ]
        bounds = np.asarray(
            [protocol["recording"]["bounds"]["surfaces"]] * 3
            + [protocol["recording"]["bounds"]["throttle"]] * engine_count,
            dtype=np.float64,
        )
        result["command_names"] = names
        result["bounds"] = bounds.tolist()
        result["diagnostics"]["engine_count"] = engine_count
        if commands is None:
            tape = np.empty((0, len(names)), dtype=np.float64)
        immediate = np.empty((0, 6, len(names)), dtype=np.float64)
        end = immediate.copy()
        result["stage"] = "initialization_selection"
        selected = entry["selected_initialization"]
        if selected is None:
            result["status"] = "missing_initialization"
            result["error"] = "No selected trim-free shipped initialization"
            return publish(terminal=True)
        ic_path = _asset(root, selected)
        document = ElementTree.parse(ic_path).getroot()
        if any(element.tag.rsplit("}", 1)[-1].lower() == "trim" for element in document.iter()):
            result["status"] = "unsupported_initialization"
            raise SimulationFailure("Selected initialization contains a forbidden trim element")
        result["stage"] = "property_validation"
        catalog = _catalog(fdm)
        _require(catalog, names)
        _require(catalog, OBSERVATION_PROPERTIES)
        result["stage"] = "initialization_load"
        publish()
        if not _call(fdm.load_ic, str(ic_path), False):
            raise SimulationFailure("load_ic returned false")
        _call(fdm.set_dt, protocol["recording"]["integration_dt_s"])
        result["stage"] = "run_ic"
        breadcrumb()
        if not _call(fdm.run_ic):
            raise SimulationFailure("run_ic returned false")
        result["stage"] = "command_baseline"
        initial = np.asarray([_read(fdm, name) for name in names], dtype=np.float64)
        result["diagnostics"]["initial_commands"] = [
            float(value) if np.isfinite(value) else None for value in initial
        ]
        if not np.isfinite(initial).all() or np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
            raise SimulationFailure("Initial command baseline is nonfinite or out of bounds")
        result["diagnostics"].update(_diagnostics(fdm, catalog, engine_count))
        result["stage"] = "initial_observation"
        raw, observed, time = _observation(fdm)
        native_rows.append(raw)
        observation_rows.append(observed)
        times.append(time)
        result["diagnostics"]["initial_velocity_m_s"] = observed[:3].tolist()
        result["diagnostics"]["initial_body_rates_rad_s"] = observed[3:6].tolist()
        result["diagnostics"]["initial_speed_m_s"] = float(np.linalg.norm(observed[:3]))
        result["status"] = "recording_failure"
        result["stage"] = "command_tape"
        tape = _tape(protocol, entry, initial, bounds) if commands is None else np.array(commands, dtype=np.float64, copy=True)
        if tape.ndim != 2 or tape.shape[1] != len(names):
            raise ValueError(f"Command tape has invalid shape {tape.shape}")
        if not np.isfinite(tape).all() or np.any(tape < bounds[:, 0]) or np.any(tape > bounds[:, 1]):
            raise ValueError("Command tape is nonfinite or out of bounds")
        substeps = protocol["recording"]["substeps"]
        immediate = np.full((len(tape), substeps, len(names)), np.nan, dtype=np.float64)
        end = immediate.copy()
        attempted = np.zeros((len(tape), substeps), dtype=np.float64)
        completed = attempted.copy()
        publish()
        for interval, command in enumerate(tape):
            attempted_intervals = interval + 1
            result["failed_interval"] = interval
            for substep in range(substeps):
                result["failed_substep"] = substep
                result["stage"] = "command_write"
                breadcrumb()
                for name, value in zip(names, command):
                    _write(fdm, name, float(value))
                result["stage"] = "immediate_readback"
                for index, name in enumerate(names):
                    immediate[interval, substep, index] = _read(fdm, name)
                result["stage"] = "run"
                breadcrumb()
                attempted[interval, substep] = 1.0
                ran = _call(fdm.run)
                completed[interval, substep] = float(bool(ran))
                result["stage"] = "end_readback"
                for index, name in enumerate(names):
                    end[interval, substep, index] = _read(fdm, name)
                if not ran:
                    result["stage"] = "run"
                    raise SimulationFailure("run returned false")
            result["stage"] = "observation"
            raw, observed, time = _observation(fdm)
            native_rows.append(raw)
            observation_rows.append(observed)
            times.append(time)
            publish()
        result.update(status="completed", stage="completed", failed_interval=None, failed_substep=None)
    except SimulationFailure as error:
        if isinstance(error, _ObservationFailure):
            failed_raw = error.raw.reshape(1, 16)
            failed_observation = error.mapped.reshape(1, 16)
            failed_time = np.asarray([error.time], dtype=np.float64)
        result["error"] = f"{type(error).__name__}: {error}"
    return publish(terminal=True)
