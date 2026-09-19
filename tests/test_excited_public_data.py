"""Toy-data checks for historical imports; no simulator or learner execution."""

import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from test_command_excitation_data import reseal, tiny_data

from glassbox.experimental import command_excitation_data as historical_data
from glassbox.experimental import command_excitation_experiment as historical
from glassbox.experimental import excited_public_data as data
from glassbox.experimental import two_simulator_flight as flight


def toy_import(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "current"
    old_protocol, old_runtime = tiny_data(source, monkeypatch)
    historical_data.generate_excitation(
        "crazyflow", source, flight, old_protocol, old_runtime
    )
    root = tmp_path / "checkout"
    monkeypatch.setattr(data, "ROOT", root)
    protocol_relative = "docs/harness/prior.json"
    old_path = root / protocol_relative
    flight.write_json(old_path, old_protocol)
    flight.freeze_files(
        source,
        "run.json",
        {
            "protocol_sha256": flight.digest(old_path),
            "runtime": old_runtime,
        },
    )
    common = output / "crazyflow/data"
    shutil.copytree(source / "crazyflow/data", common)
    runtime = {"toy": "current runtime"}
    seal = flight.read_json(common / "seal.json")
    seal["runtime"] = runtime
    flight.write_json(common / "seal.json", seal)
    p = deepcopy(old_protocol)
    p["counts"]["calibration_pool_per_simulator"] = 6
    p["prior_excitation_protocol"] = {
        "path": protocol_relative,
        "sha256": flight.digest(old_path),
    }
    p["prior_excitation_bundle"] = {
        "path": str(source),
        "manifest_sha256": flight.digest(source / "run.json"),
    }
    p["imported_calibration"] = {
        "excitation_seal_sha256": {
            "crazyflow": flight.digest(source / "crazyflow/excitation/seal.json")
        }
    }
    return output, source, p, runtime, old_protocol, old_runtime


def test_import_preserves_old_seal_and_separates_current_runtime(tmp_path, monkeypatch):
    output, source, p, runtime, _, old_runtime = toy_import(tmp_path, monkeypatch)
    result = data.import_excitation("crazyflow", output, flight, p, runtime)
    assert result["current_runtime"] == runtime
    assert result["historical_runtime"] == old_runtime
    assert runtime != old_runtime
    assert (
        result["historical_common_data_seal_sha256"]
        != result["current_common_data_seal_sha256"]
    )
    assert result["original_calibration_parents_exact"] == 6
    assert result["development_parent_files_exact"] == 2
    assert result["copied_files_and_seal_byte_exact"]
    source_files = {
        str(f.relative_to(source / "crazyflow/excitation")): f.read_bytes()
        for f in (source / "crazyflow/excitation").rglob("*")
        if f.is_file()
    }
    copied_files = {
        str(f.relative_to(output / "crazyflow/excitation")): f.read_bytes()
        for f in (output / "crazyflow/excitation").rglob("*")
        if f.is_file()
    }
    assert source_files == copied_files
    assert result["imported_file_count"] == len(source_files)
    assert data.check_import("crazyflow", output, flight, p, runtime) == result
    with pytest.raises(FileExistsError):
        data.import_excitation("crazyflow", output, flight, p, runtime)


@pytest.mark.parametrize(
    "mutation", ["source_anchor", "source_payload", "seal_pin", "protocol"]
)
def test_rejects_invalid_historical_anchors_before_copy(
    tmp_path, monkeypatch, mutation
):
    output, source, p, runtime, _, _ = toy_import(tmp_path, monkeypatch)
    if mutation == "source_anchor":
        p["prior_excitation_bundle"]["manifest_sha256"] = "0" * 64
    elif mutation == "source_payload":
        (source / "crazyflow/excitation/roles.json").write_text("{}")
    elif mutation == "seal_pin":
        p["imported_calibration"]["excitation_seal_sha256"]["crazyflow"] = "0" * 64
    else:
        (data.ROOT / p["prior_excitation_protocol"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        data.import_excitation("crazyflow", output, flight, p, runtime)
    assert not (output / "crazyflow/excitation").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_payload",
        "coherent_payload",
        "current_runtime_in_old_seal",
        "extra",
        "missing",
        "symlink",
    ],
)
def test_rejects_altered_import_even_after_local_resealing(
    tmp_path, monkeypatch, mutation
):
    output, _, p, runtime, _, _ = toy_import(tmp_path, monkeypatch)
    data.import_excitation("crazyflow", output, flight, p, runtime)
    directory = output / "crazyflow/excitation"
    if mutation in ("raw_payload", "coherent_payload"):
        roles = flight.read_json(directory / "roles.json")
        roles["training"].reverse()
        flight.write_json(directory / "roles.json", roles)
        if mutation == "coherent_payload":
            reseal(directory)
    elif mutation == "current_runtime_in_old_seal":
        seal = flight.read_json(directory / "seal.json")
        seal["runtime"] = runtime
        flight.write_json(directory / "seal.json", seal)
    elif mutation == "extra":
        (directory / "extra.json").write_text("{}")
    elif mutation == "missing":
        (directory / "roles.json").unlink()
    else:
        (directory / "extra-link").symlink_to(directory / "roles.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        data.check_import("crazyflow", output, flight, p, runtime)


@pytest.mark.parametrize("mutation", ["runtime", "roles", "metadata", "arrays"])
def test_current_common_must_match_historical_original_calibration(
    tmp_path, monkeypatch, mutation
):
    output, _, p, runtime, _, _ = toy_import(tmp_path, monkeypatch)
    data.import_excitation("crazyflow", output, flight, p, runtime)
    directory = output / "crazyflow/data"
    if mutation == "runtime":
        runtime = {"toy": "incorrect current runtime"}
    elif mutation == "roles":
        roles = flight.read_json(directory / "roles.json")
        roles["training"].reverse()
        flight.write_json(directory / "roles.json", roles)
        reseal(directory)
    else:
        records = flight.read_json(directory / "records.json")
        record = records[0]
        if mutation == "metadata":
            record["metadata"]["seed"] += 1
            flight.write_json(directory / "records.json", records)
            flight.write_json(directory / (record["prefix"] + ".json"), record)
        else:
            path = directory / (record["prefix"] + ".npz")
            arrays = flight.load_arrays(path)
            arrays["states"][-1, 3] += 0.1
            flight.save_arrays(path, arrays)
        reseal(directory)
    with pytest.raises(ValueError):
        data.check_import("crazyflow", output, flight, p, runtime)


def historical_stubs(monkeypatch, p, old_protocol, old_runtime):
    monkeypatch.setattr(
        historical, "PROTOCOL", data.ROOT / p["prior_excitation_protocol"]["path"]
    )
    monkeypatch.setattr(historical, "protocol", lambda: old_protocol)
    monkeypatch.setattr(historical, "check_sources", lambda: old_runtime)
    sentinel = object()
    monkeypatch.setattr(historical, "flight", lambda: sentinel)
    return sentinel


def replay_evidence():
    return {
        "replay_arrays": 99,
        "arrays": 99,
        "exact": True,
        "regenerated_training_parents": 4,
        "copied_development_parents": 2,
        "calibration_parents": 6,
    }


def test_replay_uses_trusted_prior_bundle_and_own_runtime(tmp_path, monkeypatch):
    output, source, p, runtime, old_protocol, old_runtime = toy_import(
        tmp_path, monkeypatch
    )
    data.import_excitation("crazyflow", output, flight, p, runtime)
    sentinel = historical_stubs(monkeypatch, p, old_protocol, old_runtime)
    calls = []

    def replay(simulator, root, private_flight, protocol, current_runtime):
        calls.append((simulator, root, private_flight, protocol, current_runtime))
        return replay_evidence()

    monkeypatch.setattr(historical_data, "replay_excitation", replay)
    result = data.replay_import("crazyflow", output, flight, p, runtime)
    assert calls == [("crazyflow", source, sentinel, old_protocol, old_runtime)]
    assert result["historical_excitation_replay"]["exact"]
    assert result["copied_files_reverified"]
    assert result["reuse"]["current_runtime"] == runtime


def test_replay_rejects_changed_historical_runtime_before_physics(
    tmp_path, monkeypatch
):
    output, _, p, runtime, old_protocol, old_runtime = toy_import(tmp_path, monkeypatch)
    data.import_excitation("crazyflow", output, flight, p, runtime)
    historical_stubs(monkeypatch, p, old_protocol, old_runtime)
    monkeypatch.setattr(historical, "check_sources", lambda: runtime)
    monkeypatch.setattr(
        historical_data,
        "replay_excitation",
        lambda *_: pytest.fail("physics must not run with current runtime"),
    )
    with pytest.raises(ValueError, match="historical physics replay runtime"):
        data.replay_import("crazyflow", output, flight, p, runtime)


def test_replay_rechecks_copied_files_after_physics(tmp_path, monkeypatch):
    output, _, p, runtime, old_protocol, old_runtime = toy_import(tmp_path, monkeypatch)
    data.import_excitation("crazyflow", output, flight, p, runtime)
    historical_stubs(monkeypatch, p, old_protocol, old_runtime)

    def replay(*_):
        (output / "crazyflow/excitation/roles.json").write_text("{}")
        return replay_evidence()

    monkeypatch.setattr(historical_data, "replay_excitation", replay)
    with pytest.raises(ValueError, match="altered artifact"):
        data.replay_import("crazyflow", output, flight, p, runtime)


def test_partial_historical_replay_is_not_accepted(tmp_path, monkeypatch):
    output, _, p, runtime, old_protocol, old_runtime = toy_import(tmp_path, monkeypatch)
    data.import_excitation("crazyflow", output, flight, p, runtime)
    historical_stubs(monkeypatch, p, old_protocol, old_runtime)
    incomplete = replay_evidence()
    incomplete["regenerated_training_parents"] -= 1
    monkeypatch.setattr(historical_data, "replay_excitation", lambda *_: incomplete)
    with pytest.raises(ValueError, match="historical excitation replay is incomplete"):
        data.replay_import("crazyflow", output, flight, p, runtime)


def test_static_protocol_preserves_prior_excitation_seals():
    root = Path(__file__).parents[1]
    p = flight.read_json(root / "docs/harness/excited-public-architecture-v1.json")
    previous = flight.read_json(root / p["prior_excitation_protocol"]["path"])
    assert p["planned_automatic_roles"] == previous["planned_automatic_roles"]
    assert p["counts"]["additional_simulated_parent_trajectories"] == 0
    for simulator in ("crazyflow", "cascade"):
        path = (
            Path(p["prior_excitation_bundle"]["path"])
            / simulator
            / "excitation/seal.json"
        )
        if not path.is_file():
            pytest.skip("trusted preceding bundle is not installed")
        assert (
            flight.digest(path)
            == p["imported_calibration"]["excitation_seal_sha256"][simulator]
        )
