"""Mock-only checks for the startup intervention; no native trials or fits."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_jsbsim_adapter import FakeFDM

from glassbox.experimental import jsbsim_adapter as inherited
from glassbox.experimental import jsbsim_excitation_adapter as adapter


class ExcitationFDM(FakeFDM):
    def __init__(self, root, *, engines=2):
        super().__init__(root, engines=engines)
        self.read_errors = {}
        self.ic_error = None
        self.ic_result = True
        self.bootstrap_error = None
        self.invalid_boundary = False
        self.early_boundary_error = False
        self.time_reads = 0
        self.values["velocities/vtrue-fps"] = 4.0
        for index in range(engines):
            suffix = "" if index == 0 else f"[{index}]"
            self.values[f"propulsion/engine{suffix}/set-running"] = 0.0
            self.values[f"propulsion/engine{suffix}/power-hp"] = 0.0
            self.values[f"fcs/mixture-cmd-norm{suffix}"] = 0.0

    def __getitem__(self, name):
        if name in self.read_errors:
            raise self.read_errors[name]
        if self.early_boundary_error and self.steps >= 6 and name == inherited.OBSERVATION_PROPERTIES[0]:
            raise RuntimeError("core getter failed before time capture")
        return super().__getitem__(name)

    def get_propulsion(self):
        return SimpleNamespace(get_num_engines=lambda: self.engines, init_running=self.init_running)

    def init_running(self, index):
        self.calls.append(("init_running", index))
        if self.bootstrap_error is not None:
            raise self.bootstrap_error
        for index in range(self.engines):
            suffix = "" if index == 0 else f"[{index}]"
            self.values[f"propulsion/engine{suffix}/set-running"] = 1.0
            self.values[f"propulsion/engine{suffix}/power-hp"] = 10.0
            self.values[f"fcs/mixture-cmd-norm{suffix}"] = 1.0

    def run_ic(self):
        super().run_ic()
        if self.ic_error is not None:
            raise self.ic_error
        return self.ic_result

    def get_sim_time(self):
        self.time_reads += 1
        return super().get_sim_time()

    def run(self):
        result = super().run()
        if self.invalid_boundary and self.steps == 6:
            self.values[inherited.OBSERVATION_PROPERTIES[0]] = np.nan
        return result


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    directory = tmp_path / "aircraft" / "fixture"
    directory.mkdir(parents=True)
    (tmp_path / "engine").mkdir()
    (tmp_path / "systems").mkdir()
    (directory / "variant.xml").write_text("<fdm_config/>")
    (directory / "reset00.xml").write_text("<initialize/>")
    entry = {"id": "fixture/variant.xml", "path": "aircraft/fixture/variant.xml",
             "selected_initialization": "aircraft/fixture/reset00.xml"}
    harness = Path(__file__).parents[1] / "docs/harness"
    old = json.loads((harness / "jsbsim-onboarding-v1.json").read_text())
    protocol = json.loads((harness / "jsbsim-excitation-v1.json").read_text())
    fdm = ExcitationFDM(str(tmp_path))
    def factory(_):
        return fdm

    module = SimpleNamespace(FGFDMExec=factory)
    monkeypatch.setitem(sys.modules, "jsbsim", module)
    tape = np.asarray([[.1, -.2, .3, .4, .5], [.2, -.1, .4, .5, .6]])
    return tmp_path, entry, old, protocol, fdm, module, factory, tape


def simulate(fixture, arm="as_shipped", **kwargs):
    root, entry, old, protocol, _, _, _, tape = fixture
    return adapter.simulate(root, entry, tape, old, arm, protocol, **kwargs)


def test_as_shipped_core_arrays_match_unchanged_adapter(fixture):
    root, entry, old, _, fdm, module, factory, tape = fixture
    native = inherited.simulate(root, entry, tape, old)
    replacement = ExcitationFDM(str(root))
    module.FGFDMExec = lambda _: replacement
    result = simulate(fixture)
    assert result["status"] == "completed"
    for name, values in native["arrays"].items():
        np.testing.assert_array_equal(result["arrays"][name], values)
    assert result["diagnostics"] == native["diagnostics"]
    assert replacement.steps == fdm.steps == 12
    assert replacement.calls == fdm.calls
    assert replacement.time_reads == fdm.time_reads + 2
    assert result["startup"]["attempted_count"] == 0
    assert result["startup"]["returned_count"] == 0
    assert result["startup"]["phase"] == "ready"
    assert module.FGFDMExec is not factory


@pytest.mark.parametrize("engines", [0, 2])
def test_bootstrap_once_after_ic_before_commands_without_advancing_time(fixture, engines):
    root, entry, old, protocol, _, module, _, _ = fixture
    fdm = ExcitationFDM(str(root), engines=engines)
    module.FGFDMExec = lambda _: fdm
    tape = np.zeros((2, 3 + engines))
    snapshots = []
    result = adapter.simulate(root, entry, tape, old, "engine_bootstrap", protocol,
                              progress=snapshots.append)
    assert result["status"] == "completed"
    assert fdm.calls.count(("run_ic",)) == 1
    assert fdm.calls.count(("init_running", -1)) == 1
    assert fdm.calls.index(("run_ic",)) < fdm.calls.index(("init_running", -1))
    first_command_write = next(index for index, value in enumerate(fdm.calls)
                               if value[:2] == ("write", inherited.SURFACE_COMMANDS[0]))
    assert fdm.calls.index(("init_running", -1)) < first_command_write
    assert fdm.steps == 12
    arrays = result["arrays"]
    np.testing.assert_array_equal(arrays["startup_pre_time_s"], [0.0])
    np.testing.assert_array_equal(arrays["startup_post_time_s"], [0.0])
    np.testing.assert_array_equal(arrays["startup_pre_observation"], arrays["startup_post_observation"])
    np.testing.assert_array_equal(arrays["commands"], tape)
    for prefix in ("startup_pre", "startup_post"):
        np.testing.assert_array_equal(arrays[prefix + "_observation_valid"], 1)
    if engines:
        index = result["telemetry"]["names"].index("propulsion/engine/set-running")
        assert arrays["startup_pre_telemetry"][0, index] == 0
        assert arrays["startup_post_telemetry"][0, index] == 1
        np.testing.assert_array_equal(arrays["telemetry"][:, index], 1)
    startup = result["startup"]
    assert startup["attempted_count"] == startup["returned_count"] == 1
    assert startup["native_run_ic_attempted_count"] == startup["native_run_ic_returned_count"] == 1
    assert startup["native_run_ic_result"] is True
    before = [value for value in snapshots if not value.get("progress_only")
              and value["startup"]["phase"] == "bootstrap"]
    assert len(before) == 1
    assert before[0]["startup"]["attempted_count"] == 1
    assert before[0]["startup"]["returned_count"] == 0
    assert before[0]["arrays"]["startup_pre_observation"].shape == (1, 16)
    assert before[0]["arrays"]["startup_post_observation"].shape == (0, 16)


def test_ordered_catalog_selection_and_native_units(fixture):
    _, _, _, protocol, fdm, _, _, _ = fixture
    result = simulate(fixture)
    requested = adapter.telemetry_names(protocol, fdm.engines)
    names = [name for name in requested if name in fdm.values]
    missing = [name for name in requested if name not in fdm.values]
    assert result["telemetry"]["names"] == names
    assert result["telemetry"]["missing"] == missing
    assert "propulsion/engine/set-running" in names
    assert "propulsion/engine[1]/set-running" in names
    assert "propulsion/engine/running" not in names
    arrays = result["arrays"]
    assert arrays["telemetry"].shape == arrays["telemetry_valid"].shape == (3, len(names))
    assert arrays["failed_telemetry"].shape == (0, len(names))
    np.testing.assert_array_equal(arrays["telemetry"][:, names.index("position/h-agl-ft")], 10)
    assert result["telemetry"]["errors"] == [{}, {}, {}]
    for values in arrays.values():
        assert values.dtype == np.float64


@pytest.mark.parametrize("failure", [RuntimeError("optional telemetry unavailable"), np.inf, np.nan])
def test_telemetry_failure_is_masked_and_does_not_gate_core(fixture, failure):
    _, _, _, _, fdm, _, _, _ = fixture
    name = "propulsion/engine/power-hp"
    if isinstance(failure, Exception):
        fdm.read_errors[name] = failure
    else:
        fdm.values[name] = failure
    result = simulate(fixture)
    assert result["status"] == "completed"
    index = result["telemetry"]["names"].index(name)
    assert np.isnan(result["arrays"]["telemetry"][:, index]).all()
    np.testing.assert_array_equal(result["arrays"]["telemetry_valid"][:, index], 0)
    assert all(name in row for row in result["telemetry"]["errors"])
    assert name in result["startup"]["snapshot_errors"]["pre"]
    assert name in result["startup"]["snapshot_errors"]["post"]


@pytest.mark.parametrize("error", [KeyError("typo"), TypeError("bad call"), AttributeError("missing method")])
def test_programming_errors_escape_and_restore_factory(fixture, error):
    _, _, _, _, fdm, module, factory, _ = fixture
    fdm.read_errors["propulsion/engine/power-hp"] = error
    with pytest.raises(type(error), match=str(error).strip("'")):
        simulate(fixture)
    assert module.FGFDMExec is factory


@pytest.mark.parametrize("early", [False, True])
def test_failed_core_boundary_keeps_only_captured_telemetry(fixture, early):
    _, _, _, _, fdm, _, _, _ = fixture
    fdm.invalid_boundary = not early
    fdm.early_boundary_error = early
    result = simulate(fixture)
    assert result["status"] == "recording_failure"
    arrays = result["arrays"]
    assert arrays["observations"].shape == (1, 16)
    assert arrays["failed_observation"].shape == (1, 16)
    assert len(arrays["telemetry"]) == 1
    assert len(arrays["failed_telemetry"]) == int(not early)
    assert len(result["telemetry"]["failed_errors"]) == int(not early)
    assert len(result["telemetry"]["errors"]) == 1


@pytest.mark.parametrize("failure", [False, RuntimeError("native IC failure")])
def test_native_ic_failure_never_calls_bootstrap(fixture, failure):
    _, _, _, _, fdm, module, factory, _ = fixture
    if isinstance(failure, Exception):
        fdm.ic_error = failure
    else:
        fdm.ic_result = failure
    result = simulate(fixture, "engine_bootstrap")
    assert result["status"] == "initialization_failure"
    startup = result["startup"]
    assert startup["phase"] == "native_run_ic"
    assert startup["native_run_ic_attempted_count"] == 1
    assert startup["native_run_ic_returned_count"] == int(failure is False)
    assert startup["native_run_ic_result"] is (False if failure is False else None)
    assert startup["attempted_count"] == 0
    assert startup["returned_count"] == 0
    assert result["telemetry"]["names"] == result["telemetry"]["missing"] == []
    assert result["arrays"]["startup_pre_observation"].shape == (0, 16)
    assert module.FGFDMExec is factory


def test_bootstrap_failure_preserves_pre_snapshot_and_distinct_error(fixture):
    _, _, _, _, fdm, module, factory, tape = fixture
    fdm.bootstrap_error = RuntimeError("startup failed")
    result = simulate(fixture, "engine_bootstrap")
    assert result["status"] == "initialization_failure"
    startup = result["startup"]
    assert startup["native_run_ic_result"] is True
    assert startup["native_run_ic_error"] is None
    assert startup["phase"] == "bootstrap"
    assert startup["attempted_count"] == 1
    assert startup["returned_count"] == 0
    assert startup["error"] == "RuntimeError: startup failed"
    assert result["arrays"]["startup_pre_observation"].shape == (1, 16)
    assert result["arrays"]["startup_post_observation"].shape == (0, 16)
    assert result["arrays"]["observations"].shape == (0, 16)
    np.testing.assert_array_equal(result["arrays"]["commands"], tape)
    assert module.FGFDMExec is factory


def test_no_common_tape_can_only_reproduce_setup_failures(fixture):
    root, entry, old, protocol, fdm, module, factory, _ = fixture
    missing = dict(entry, selected_initialization=None)
    result = adapter.simulate(root, missing, None, old, "engine_bootstrap", protocol)
    assert result["status"] == "missing_initialization"
    assert result["startup"]["native_run_ic_attempted_count"] == 0
    assert result["arrays"]["telemetry"].shape == (0, 0)
    with pytest.raises(ValueError, match="frozen common command tape"):
        adapter.simulate(root, entry, None, old, "engine_bootstrap", protocol)
    assert ("init_running", -1) not in fdm.calls
    assert module.FGFDMExec is factory


@pytest.mark.parametrize("phase", ["native_run_ic", "bootstrap", "ready"])
def test_progress_runtime_error_is_harness_failure_not_simulator_failure(fixture, phase):
    _, _, _, _, _, module, factory, _ = fixture

    def progress(value):
        if value["startup"]["phase"] == phase:
            raise RuntimeError("checkpoint storage failure")

    with pytest.raises(RuntimeError, match="checkpoint storage failure"):
        simulate(fixture, "engine_bootstrap", progress=progress)
    assert module.FGFDMExec is factory


def test_snapshot_nonfinite_core_values_are_preserved_with_validity(fixture):
    _, _, _, _, fdm, _, _, _ = fixture
    fdm.values[inherited.OBSERVATION_PROPERTIES[0]] = np.inf
    result = simulate(fixture)
    arrays = result["arrays"]
    assert result["status"] == "initialization_failure"
    assert np.isinf(arrays["startup_pre_native_observation"][0, 0])
    assert np.isinf(arrays["startup_pre_observation"][0, 0])
    assert arrays["startup_pre_observation_valid"][0, 0] == 0
    assert len(arrays["failed_telemetry"]) == 1
