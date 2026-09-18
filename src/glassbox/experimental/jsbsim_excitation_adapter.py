"""Read-only instrumentation and the frozen engine-startup intervention.

The inherited adapter remains unchanged. This process-local factory wrapper is
for sequential simulator calls inside an isolated worker, not threaded use.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from . import jsbsim_adapter as inherited


def telemetry_names(protocol: dict, engine_count: int) -> list[str]:
    """Return the requested ordered roster, before catalog-based selection."""
    specification = protocol["telemetry"]
    names = list(specification["global_properties"])
    for index in range(engine_count):
        suffix = "" if index == 0 else f"[{index}]"
        names.extend(f"propulsion/engine{suffix}/{name}"
                     for name in specification["engine_suffixes"])
        names.extend(f"fcs/{name}{suffix}" for name in specification["fcs_per_engine"])
    return names


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _scalar(function, errors: dict, name: str) -> float:
    """Preserve returned scalars; only native RuntimeError becomes missing."""
    try:
        value = float(function())
    except RuntimeError as error:
        errors[name] = _error(error)
        return float("nan")
    if not np.isfinite(value):
        errors[name] = f"Nonfinite value: {value!r}"
    return value


def _telemetry(fdm, names: list[str]) -> tuple[np.ndarray, dict]:
    errors: dict[str, str] = {}
    values = np.asarray([
        _scalar(lambda name=name: fdm[name], errors, name) for name in names
    ], dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    return values, errors


def _snapshot(fdm, names: list[str], command_names: list[str]) -> tuple[dict, dict]:
    """Best-effort diagnostic snapshot, with no observation/geometry gate."""
    values, errors = _telemetry(fdm, names)
    raw = np.full(16, np.nan, dtype=np.float64)
    for index, name in enumerate(inherited.OBSERVATION_PROPERTIES[:6]):
        raw[index] = _scalar(lambda name=name: fdm[name], errors, name)
    try:
        matrix = np.asarray(fdm.get_propagate().get_Tl2b(), dtype=np.float64)
        if matrix.shape == (3, 3):
            raw[6:15] = matrix.reshape(9)
            if not np.isfinite(matrix).all():
                errors["get_Tl2b"] = "Nonfinite local-to-body matrix"
        else:
            errors["get_Tl2b"] = f"Local-to-body matrix has shape {matrix.shape}"
    except RuntimeError as error:
        errors["get_Tl2b"] = _error(error)
    altitude = inherited.OBSERVATION_PROPERTIES[-1]
    raw[15] = _scalar(lambda: fdm[altitude], errors, altitude)
    mapped = raw.copy()
    mapped[:3] *= inherited.FEET_TO_METERS
    mapped[6:15] = raw[6:15].reshape(3, 3).T.reshape(9)
    mapped[15] *= inherited.FEET_TO_METERS
    commands = np.asarray([
        _scalar(lambda name=name: fdm[name], errors, name) for name in command_names
    ], dtype=np.float64)
    time = np.asarray([_scalar(fdm.get_sim_time, errors, "get_sim_time")], dtype=np.float64)
    return {
        "telemetry": values[None, :],
        "telemetry_valid": np.isfinite(values)[None, :].astype(np.float64),
        "native_observation": raw[None, :],
        "observation": mapped[None, :],
        "observation_valid": np.isfinite(mapped)[None, :].astype(np.float64),
        "commands": commands[None, :],
        "commands_valid": np.isfinite(commands)[None, :].astype(np.float64),
        "time_s": time,
        "time_valid": np.isfinite(time).astype(np.float64),
    }, errors


class _ProgressFailure(Exception):
    """Keep callback RuntimeError out of the inherited native-call boundary."""

    def __init__(self, original: Exception):
        self.original = original


class _Instrumentation:
    def __init__(self, arm, protocol, commands, progress):
        self.arm = arm
        self.protocol = protocol
        self.commands = commands
        self.progress = progress
        self.latest = None
        self.catalog: set[str] = set()
        self.names: list[str] = []
        self.missing: list[str] = []
        self.rows: list[np.ndarray] = []
        self.errors: list[dict] = []
        self.snapshots: dict[str, dict] = {}
        self.startup = {
            "arm": arm, "native_run_ic_attempted_count": 0,
            "native_run_ic_returned_count": 0, "native_run_ic_result": None,
            "native_run_ic_error": None, "attempted_count": 0,
            "returned_count": 0, "error": None, "phase": "uninitialized",
            "snapshot_errors": {"pre": {}, "post": {}},
        }

    def enrich(self, result):
        result = deepcopy(result)
        result["startup"] = deepcopy(self.startup)
        if result.get("progress_only"):
            return result
        arrays = result["arrays"]
        count = len(arrays["observations"])
        dimension = len(self.names)
        if not count <= len(self.rows) <= count + 1:
            raise ValueError("Telemetry capture is not aligned with observations")
        values = np.asarray(self.rows, dtype=np.float64).reshape(len(self.rows), dimension)
        arrays["telemetry"] = values[:count].copy()
        arrays["telemetry_valid"] = np.isfinite(values[:count]).astype(np.float64)
        arrays["failed_telemetry"] = values[count:].copy()
        arrays["failed_telemetry_valid"] = np.isfinite(values[count:]).astype(np.float64)
        result["telemetry"] = {
            "names": self.names.copy(), "missing": self.missing.copy(),
            "errors": deepcopy(self.errors[:count]),
            "failed_errors": deepcopy(self.errors[count:]),
        }
        widths = {"telemetry": dimension, "telemetry_valid": dimension,
                  "native_observation": 16, "observation": 16,
                  "observation_valid": 16, "commands": len(result["command_names"]),
                  "commands_valid": len(result["command_names"])}
        for side in ("pre", "post"):
            snapshot = self.snapshots.get(side, {})
            for name, width in widths.items():
                arrays[f"startup_{side}_{name}"] = snapshot.get(
                    name, np.empty((0, width), dtype=np.float64)).copy()
            for name in ("time_s", "time_valid"):
                arrays[f"startup_{side}_{name}"] = snapshot.get(
                    name, np.empty(0, dtype=np.float64)).copy()
        return result

    def emit(self, value):
        if self.progress is not None:
            try:
                self.progress(value)
            except Exception as error:
                raise _ProgressFailure(error) from error

    def receive(self, result):
        if not result.get("progress_only"):
            self.latest = deepcopy(result)
        self.emit(self.enrich(result))

    def checkpoint(self):
        # The inherited adapter publishes a full empty prefix before load_ic.
        # Keep that prefix and the supplied tape if native startup never returns.
        if self.latest is None:
            raise ValueError("Startup lacks the inherited initialization checkpoint")
        result = deepcopy(self.latest)
        result.update(status="running", stage="run_ic")
        self.emit(self.enrich(result))

    def run_ic(self, fdm):
        startup = self.startup
        startup["phase"] = "native_run_ic"
        startup["native_run_ic_attempted_count"] += 1
        self.checkpoint()
        try:
            result = fdm.run_ic()
        except RuntimeError as error:
            startup["native_run_ic_error"] = _error(error)
            raise
        startup["native_run_ic_returned_count"] += 1
        startup["native_run_ic_result"] = bool(result)
        if not result:
            return result
        if self.commands is None:
            raise ValueError("Successful initialization requires the frozen common command tape")
        requested = telemetry_names(self.protocol, self.latest["diagnostics"]["engine_count"])
        self.names = [name for name in requested if name in self.catalog]
        self.missing = [name for name in requested if name not in self.catalog]
        startup["phase"] = "pre_startup"
        names = self.latest["command_names"]
        self.snapshots["pre"], startup["snapshot_errors"]["pre"] = _snapshot(fdm, self.names, names)
        self.checkpoint()
        if self.arm == "engine_bootstrap":
            startup["phase"] = "bootstrap"
            startup["attempted_count"] += 1
            self.checkpoint()
            try:
                fdm.get_propulsion().init_running(-1)
            except RuntimeError as error:
                startup["error"] = _error(error)
                raise
            startup["returned_count"] += 1
        startup["phase"] = "post_startup"
        self.snapshots["post"], startup["snapshot_errors"]["post"] = _snapshot(fdm, self.names, names)
        startup["phase"] = "ready"
        self.checkpoint()
        return result


class _Proxy:
    def __init__(self, fdm, instrumentation):
        self._fdm = fdm
        self._instrumentation = instrumentation

    def __getattr__(self, name):
        return getattr(self._fdm, name)

    def __getitem__(self, name):
        return self._fdm[name]

    def __setitem__(self, name, value):
        self._fdm[name] = value

    def get_property_catalog(self):
        catalog = self._fdm.get_property_catalog()
        self._instrumentation.catalog = {line.rsplit(" (", 1)[0].strip() for line in catalog}
        return catalog

    def run_ic(self):
        return self._instrumentation.run_ic(self._fdm)

    def get_sim_time(self):
        time = self._fdm.get_sim_time()
        values, errors = _telemetry(self._fdm, self._instrumentation.names)
        self._instrumentation.rows.append(values)
        self._instrumentation.errors.append(errors)
        return time


def simulate(
    root: Path, entry: dict, commands: np.ndarray | None, inherited_protocol: dict,
    arm: str, excitation_protocol: dict, *, progress=None,
) -> dict:
    """Fresh simulation, with startup snapshots and masked diagnostic telemetry.

    A supplied common tape is mandatory if initialization succeeds. ``None`` is
    reserved for reproducing the inherited unknown-baseline setup outcomes.
    Telemetry rows align with validated core observations; a rejected boundary
    captured by the time hook is retained separately. No extra integration runs.
    """
    if arm not in excitation_protocol["arms"] or arm not in ("as_shipped", "engine_bootstrap"):
        raise ValueError(f"Unknown excitation arm: {arm}")
    import jsbsim

    factory = jsbsim.FGFDMExec
    state = _Instrumentation(arm, excitation_protocol, commands, progress)

    def instrumented_factory(*args, **kwargs):
        return _Proxy(factory(*args, **kwargs), state)

    jsbsim.FGFDMExec = instrumented_factory
    try:
        result = inherited.simulate(root, entry, commands, inherited_protocol, progress=state.receive)
        return state.enrich(result)
    except _ProgressFailure as error:
        raise error.original.with_traceback(error.original.__traceback__) from error
    finally:
        jsbsim.FGFDMExec = factory
