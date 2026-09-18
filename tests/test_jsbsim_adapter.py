"""Mock-only contracts; no JSBSim trajectories or model fits are executed."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import jsbsim_adapter as adapter


class FakeFDM:
    def __init__(self, root, *, fail_run=None, overwrite=False, engines=2):
        self.root = root
        self.fail_run = fail_run
        self.overwrite = overwrite
        self.engines = engines
        self.calls = []
        self.loaded = False
        self.steps = 0
        self.dt = 0
        self.time = 0.0
        self.observed_commands = []
        self.values = {name: 0.0 for name in adapter.SURFACE_COMMANDS}
        self.values.update({name: float(index + 1) for index, name in enumerate(adapter.OBSERVATION_PROPERTIES)})
        self.values.update({"simulation/randomseed": 9.0, "position/h-agl-ft": 10.0,
                            "propulsion/engine/running": 0.0})
        for index in range(engines):
            self.values["fcs/throttle-cmd-norm" + ("" if index == 0 else f"[{index}]")] = 0.4
        self.matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def set_debug_level(self, level):
        self.calls.append(("debug", level))

    def disable_input(self):
        self.calls.append(("disable_input",))

    def disable_output(self):
        self.calls.append(("disable_output",))

    def get_property_manager(self):
        def get_node(name, create=False):
            assert not create
            return object() if name in self.values else None
        return SimpleNamespace(get_node=get_node)

    def get_property_catalog(self):
        return [f"{name} (RW)" for name in self.values] if self.loaded else []

    def __getitem__(self, name):
        return self.values[name]

    def __setitem__(self, name, value):
        assert name in self.values, "Dynamic creation would hide a missing property"
        self.calls.append(("write", name, value))
        self.values[name] = value

    def set_aircraft_path(self, path):
        self.calls.append(("aircraft_path", path))

    def set_engine_path(self, path):
        self.calls.append(("engine_path", path))

    def set_systems_path(self, path):
        self.calls.append(("systems_path", path))

    def load_model(self, stem, add_model_to_path):
        self.calls.append(("load", stem, add_model_to_path))
        self.loaded = True
        return True

    def get_propulsion(self):
        return SimpleNamespace(get_num_engines=lambda: self.engines)

    def load_ic(self, path, use_aircraft_path):
        self.calls.append(("load_ic", path, use_aircraft_path))
        return True

    def set_dt(self, dt):
        self.calls.append(("dt", dt))
        self.dt = dt

    def run_ic(self):
        self.calls.append(("run_ic",))
        return True

    def get_propagate(self):
        return SimpleNamespace(get_Tl2b=lambda: self.matrix)

    def get_sim_time(self):
        return self.time

    def run(self):
        self.steps += 1
        names = list(adapter.SURFACE_COMMANDS) + [
            "fcs/throttle-cmd-norm" + ("" if index == 0 else f"[{index}]")
            for index in range(self.engines)
        ]
        self.observed_commands.append([self.values[name] for name in names])
        self.time += self.dt
        self.values[adapter.OBSERVATION_PROPERTIES[0]] += sum(self.observed_commands[-1])
        if self.overwrite:
            self.values[adapter.SURFACE_COMMANDS[0]] = 0.75
        return self.steps != self.fail_run


@pytest.fixture
def setup(tmp_path, monkeypatch):
    aircraft = tmp_path / "aircraft" / "fixture"
    aircraft.mkdir(parents=True)
    (tmp_path / "engine").mkdir()
    (tmp_path / "systems").mkdir()
    (aircraft / "variant.xml").write_text("<fdm_config/>")
    (aircraft / "reset00.xml").write_text("<initialize/>")
    entry = {"id": "fixture/variant.xml", "path": "aircraft/fixture/variant.xml",
             "selected_initialization": "aircraft/fixture/reset00.xml"}
    protocol = json.loads((Path(__file__).parents[1] / "docs/harness/jsbsim-onboarding-v1.json").read_text())
    instances = []

    def factory(root):
        fdm = FakeFDM(root)
        instances.append(fdm)
        return fdm

    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=factory))
    return tmp_path, entry, protocol, instances


def test_uniform_initialization_order_and_native_coordinate_conversion(setup, monkeypatch):
    root, entry, protocol, instances = setup
    monkeypatch.setenv("JSBSIM_DISPERSE", "1")
    result = adapter.simulate(root, entry, None, protocol)
    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["failed_interval"] is None
    assert result["failed_substep"] is None
    assert result["completed_intervals"] == 40
    fdm = instances[0]
    assert fdm.calls[:6] == [
        ("debug", 0), ("disable_input",), ("disable_output",),
        ("write", "simulation/randomseed", 0),
        ("aircraft_path", str(root / "aircraft/fixture")),
        ("engine_path", str(root / "engine")),
    ]
    assert ("load", "variant", False) in fdm.calls
    assert ("load_ic", str(root / "aircraft/fixture/reset00.xml"), False) in fdm.calls
    assert fdm.calls.index(("dt", 1 / 120)) < fdm.calls.index(("run_ic",))
    assert fdm.calls.count(("disable_input",)) == 2
    assert fdm.calls.count(("disable_output",)) == 2
    arrays = result["arrays"]
    assert arrays["observations"].shape == (41, 16)
    assert arrays["commands"].shape == (40, 5)
    assert arrays["immediate_readbacks"].shape == (40, 6, 5)
    assert arrays["native_attempted"].shape == (40, 6)
    np.testing.assert_array_equal(arrays["native_completed"], np.ones((40, 6)))
    np.testing.assert_array_equal(arrays["observations"][0, :3], np.arange(1, 4) * 0.3048)
    np.testing.assert_array_equal(arrays["observations"][0, 6:15], fdm.matrix.T.reshape(9))
    np.testing.assert_array_equal(arrays["native_observations"][0, 6:15], fdm.matrix.reshape(9))
    np.testing.assert_allclose(np.diff(arrays["time_s"]), 0.05, rtol=0, atol=1e-15)
    assert result["diagnostics"]["initial_ground_clearance_m"] == 3.048


def test_frozen_random_tape_and_fresh_replay(setup):
    root, entry, protocol, instances = setup
    first = adapter.simulate(root, entry, None, protocol)
    second = adapter.simulate(root, entry, first["arrays"]["commands"], protocol)
    assert len(instances) == 2
    for name, values in first["arrays"].items():
        np.testing.assert_array_equal(values, second["arrays"][name])
    seed = int.from_bytes(hashlib.sha256(f"{protocol['id']}:{entry['id']}".encode()).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    expected = np.clip(np.array([0, 0, 0, .4, .4]) + rng.uniform(-1, 1, 5) * np.array([.1, .1, .1, .05, .05]),
                       [-1, -1, -1, 0, 0], 1)
    np.testing.assert_array_equal(first["arrays"]["commands"][:5], np.tile(expected, (5, 1)))


def test_reasserts_written_commands_before_every_native_substep(setup, monkeypatch):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root), overwrite=True)
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    tape = np.array([[.1, -.2, .3, .4, .5], [.2, -.1, .4, .5, .6]])
    result = adapter.simulate(root, entry, tape, protocol)
    assert result["status"] == "completed"
    np.testing.assert_array_equal(fdm.observed_commands, np.repeat(tape, 6, axis=0))
    np.testing.assert_array_equal(result["arrays"]["immediate_readbacks"], np.repeat(tape[:, None, :], 6, axis=1))
    np.testing.assert_array_equal(result["arrays"]["end_readbacks"][:, :, 0], np.full((2, 6), .75))


def test_load_is_attempted_before_missing_initialization(setup):
    root, entry, protocol, instances = setup
    entry["selected_initialization"] = None
    result = adapter.simulate(root, entry, None, protocol)
    assert result["status"] == "missing_initialization"
    assert instances[0].loaded
    assert not any(call[0] == "load_ic" for call in instances[0].calls)
    assert result["arrays"]["observations"].shape == (0, 16)
    assert len(result["command_names"]) == 5


def test_trim_is_rejected_before_initialization(setup):
    root, entry, protocol, instances = setup
    (root / entry["selected_initialization"]).write_text("<initialize><trim>full</trim></initialize>")
    result = adapter.simulate(root, entry, None, protocol)
    assert result["status"] == "unsupported_initialization"
    assert instances[0].loaded
    assert not any(call[0] == "load_ic" for call in instances[0].calls)


@pytest.mark.parametrize("property_name", [adapter.SURFACE_COMMANDS[1], adapter.OBSERVATION_PROPERTIES[0], "fcs/throttle-cmd-norm[1]"])
def test_catalog_rejects_missing_property_without_dynamic_creation(setup, monkeypatch, property_name):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root))
    del fdm.values[property_name]
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    result = adapter.simulate(root, entry, None, protocol)
    assert result["status"] == "initialization_failure"
    assert result["stage"] == "property_validation"
    assert property_name in result["error"]
    assert not any(call[0] == "load_ic" for call in fdm.calls)


def test_false_native_run_preserves_all_intended_commands_and_failed_prefix(setup, monkeypatch):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root), fail_run=9)
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    tape = np.zeros((4, 5))
    result = adapter.simulate(root, entry, tape, protocol)
    assert (result["status"], result["stage"]) == ("recording_failure", "run")
    assert (result["failed_interval"], result["failed_substep"]) == (1, 2)
    assert result["completed_intervals"] == 1
    arrays = result["arrays"]
    assert arrays["commands"].shape == (4, 5)
    assert arrays["observations"].shape == (2, 16)
    assert arrays["immediate_readbacks"].shape == (2, 6, 5)
    np.testing.assert_array_equal(arrays["native_attempted"][1], [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(arrays["native_completed"][1], [1, 1, 0, 0, 0, 0])
    assert np.isfinite(arrays["end_readbacks"][1, :3]).all()
    assert np.isnan(arrays["end_readbacks"][1, 3:]).all()


@pytest.mark.parametrize("value", [float("nan"), 1.1])
def test_invalid_initial_commands_are_setup_failures(setup, monkeypatch, value):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root))
    fdm.values[adapter.SURFACE_COMMANDS[0]] = value
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    result = adapter.simulate(root, entry, None, protocol)
    assert result["status"] == "initialization_failure"
    assert result["stage"] == "command_baseline"
    assert fdm.steps == 0


def test_invalid_observation_retains_native_and_mapped_failed_snapshot(setup, monkeypatch):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root))
    run = fdm.run

    def invalid_at_boundary():
        result = run()
        if fdm.steps == 12:
            fdm.matrix[0, 0] = 1
        return result

    fdm.run = invalid_at_boundary
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    result = adapter.simulate(root, entry, np.zeros((4, 5)), protocol)
    assert result["stage"] == "observation"
    assert result["completed_intervals"] == 1
    assert result["arrays"]["observations"].shape == (2, 16)
    assert result["arrays"]["failed_observation"].shape == (1, 16)
    np.testing.assert_array_equal(result["arrays"]["failed_native_observation"][0, 6:15], fdm.matrix.reshape(9))
    np.testing.assert_array_equal(result["arrays"]["native_completed"], np.ones((2, 6)))


def test_load_failure_retains_supplied_tape(setup, monkeypatch):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root))
    fdm.load_model = lambda *_: False
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    tape = np.zeros((25, 5))
    result = adapter.simulate(root, entry, tape, protocol)
    assert result["status"] == "load_failure"
    np.testing.assert_array_equal(result["arrays"]["commands"], tape)
    assert result["arrays"]["observations"].shape == (0, 16)


def test_zero_engine_configuration_keeps_declared_surface_channels(setup, monkeypatch):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root), engines=0)
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    result = adapter.simulate(root, entry, None, protocol)
    assert result["status"] == "completed"
    assert result["command_names"] == list(adapter.SURFACE_COMMANDS)
    assert result["arrays"]["commands"].shape == (40, 3)


def test_progress_snapshots_preserve_independent_prefixes_and_breadcrumbs(setup):
    root, entry, protocol, _ = setup
    events = []
    result = adapter.simulate(root, entry, np.zeros((2, 5)), protocol, progress=events.append)
    snapshots = [event for event in events if not event.get("progress_only")]
    assert snapshots[-1]["status"] == "completed"
    assert all(event["status"] == "running" for event in snapshots[:-1])
    ready = next(event for event in snapshots if event["stage"] == "command_tape")
    assert ready["arrays"]["commands"].shape == (2, 5)
    assert ready["arrays"]["observations"].shape == (1, 16)
    assert ready["arrays"]["immediate_readbacks"].shape == (0, 6, 5)
    assert [event["completed_intervals"] for event in snapshots if event["stage"] == "observation"] == [1, 2]
    breadcrumbs = [event for event in events if event.get("progress_only") and event["stage"] == "run"]
    assert [(event["failed_interval"], event["failed_substep"]) for event in breadcrumbs] == [
        (interval, native) for interval in range(2) for native in range(6)
    ]
    assert result["status"] == "completed"


@pytest.mark.parametrize("exception", [KeyError("bug"), TypeError("bug"), AttributeError("bug"), IndexError("bug")])
def test_programming_exceptions_escape_instead_of_counting_as_model_failure(setup, monkeypatch, exception):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root))

    def broken():
        raise exception

    fdm.run = broken
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    with pytest.raises(type(exception), match="bug"):
        adapter.simulate(root, entry, None, protocol)


def test_sdk_runtime_failure_preserves_failed_native_step(setup, monkeypatch):
    root, entry, protocol, _ = setup
    fdm = FakeFDM(str(root))

    def broken():
        raise RuntimeError("simulator failure")

    fdm.run = broken
    monkeypatch.setitem(sys.modules, "jsbsim", SimpleNamespace(FGFDMExec=lambda _: fdm))
    events = []
    result = adapter.simulate(root, entry, None, protocol, progress=events.append)
    assert result["status"] == "recording_failure"
    assert "simulator failure" in result["error"]
    assert (result["failed_interval"], result["failed_substep"]) == (0, 0)
    assert result["arrays"]["native_attempted"][0, 0] == 1
    assert result["arrays"]["native_completed"][0, 0] == 0
    assert events[-1]["status"] == "recording_failure"


def test_progress_storage_failure_escapes(setup):
    root, entry, protocol, _ = setup

    def broken(_):
        raise RuntimeError("storage failure")

    with pytest.raises(RuntimeError, match="storage failure"):
        adapter.simulate(root, entry, None, protocol, progress=broken)
