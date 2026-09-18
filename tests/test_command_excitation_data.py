"""Toy-kernel checks for the frozen collection intervention; no simulator runs."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import command_excitation_data as data
from glassbox.experimental import two_simulator_flight as flight


def protocol():
    return json.loads(
        (
            Path(__file__).parents[1]
            / "docs/harness/independent-command-excitation-v1.json"
        ).read_text()
    )


class ToyFixture:
    def __init__(self, dt=0.05, failure=None):
        self.dt, self.steps, self.failure = dt, round(3 / dt), failure
        self.lower, self.upper = np.array([0.0, -0.35]), np.array([1.0, 0.35])
        self.spec = SimpleNamespace(
            to_dict=lambda: {
                "channels": [
                    {"kind": "control", "minimum": lo, "maximum": hi}
                    for lo, hi in zip(
                        self.lower.tolist(), self.upper.tolist(), strict=True
                    )
                ]
            }
        )
        self._advance = self.kernel
        self.visited = []

    def config(self):
        return {
            "dt_s": self.dt,
            "bounds": np.stack((self.lower, self.upper), axis=-1).tolist(),
        }

    def kernel(self, state, command, index, wind):
        assert isinstance(index, np.int64)
        if self.failure == int(index):
            raise RuntimeError("deliberate toy failure")
        self.visited.append(np.asarray(command).copy())
        following = np.asarray(state).copy()
        following[3:5] += self.dt * np.asarray(command)
        return following, np.zeros((1, 3)), np.zeros((1, 3))

    def generate(self, entry, cell):
        state = np.zeros(13)
        state[2], state[6] = 4.0, 1.0
        states, commands, winds, forces, targets, raw = [state], [], [], [], [], []
        for index in range(self.steps):
            command = np.clip(
                np.array([0.99 + 0.1 * state[3], -0.34]), self.lower, self.upper
            )
            # A negative zero in the unchanged prefix exercises byte preservation.
            if index == 0:
                command[0] = -0.0
            state, wind, force = self._advance(state, command, np.int64(index), 0)
            states.append(state)
            commands.append(command)
            winds.append(wind)
            forces.append(force)
            targets.append(np.array([index, 1.0]))
            raw.append(command)
        states = np.asarray(states)
        arrays = {
            "time_s": np.arange(self.steps + 1) * self.dt,
            "states": states,
            "full_states": states.copy(),
            "commands": np.asarray(commands),
            "actuator_outputs": states[:, 3:5].copy(),
            "wind": np.zeros((self.steps + 1, 3)),
            "integration_wind": np.asarray(winds),
            "integration_force": np.asarray(forces),
            "pilot_targets": np.asarray(targets),
            "pilot_raw_commands": np.asarray(raw),
            "requested_dither": np.zeros((self.steps, 2)),
            "control_prefix": np.zeros((10, 2)),
        }
        return arrays, {
            "seed": entry["seed"],
            "initial_full_state": states[0].tolist(),
            "condition": cell,
        }


def training_entry(p, sim="crazyflow"):
    ids = p["planned_automatic_roles"][sim]["training"]
    return next(e for e in p["recordings"] if e["id"] == ids[0])


@pytest.mark.parametrize("dt,start,hold", [(0.01, 75, 10), (0.05, 15, 2)])
def test_schedule_rng_boundary_and_last_block(dt, start, hold):
    p, fixture = protocol(), ToyFixture(dt)
    entry = training_entry(p)
    detail, indices, assigned = data._schedule(
        dt, fixture.steps, fixture.lower, fixture.upper, entry, p
    )
    material = json.dumps(
        {
            "namespace": "independent-command-excitation-v1",
            "parent": entry["id"],
            "seed": entry["seed"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest(), "big")
    bits = np.random.Generator(np.random.PCG64(seed)).integers(
        0, 2, size=(23, 2), dtype=np.int64
    )
    assert detail["seed_integer"] == seed
    assert detail["seed_material"] == material
    np.testing.assert_array_equal(detail["signs"], 2 * bits - 1)
    assert detail["start_index"] == start and detail["block_steps"] == hold
    np.testing.assert_array_equal(indices[:start], -1)
    np.testing.assert_array_equal(indices[start : start + hold], 0)
    assert np.sum(indices == 22) == round(0.05 / dt)
    np.testing.assert_array_equal(assigned[:start], 0)
    np.testing.assert_array_equal(
        np.abs(assigned[start:]),
        np.broadcast_to(0.05 * (fixture.upper - fixture.lower), assigned[start:].shape),
    )


def test_simulator_id_separates_rng_even_with_identical_seed():
    p, f = protocol(), ToyFixture()
    entry = training_entry(p)
    other = dict(
        entry, id=entry["id"].replace("crazyflow", "cascade"), simulator="cascade"
    )
    a = data._schedule(f.dt, f.steps, f.lower, f.upper, entry, p)[0]
    b = data._schedule(f.dt, f.steps, f.lower, f.upper, other, p)[0]
    assert a["seed_integer"] != b["seed_integer"]
    assert a["signs"] != b["signs"]


@pytest.mark.parametrize("simulator", ["crazyflow", "cascade"])
def test_archived_configurations_share_control_spec_bounds(simulator):
    p = protocol()
    path = Path(p["previous_bundle"]["path"]) / simulator / "data/configuration.json"
    if not path.is_file():
        pytest.skip("archived physical configuration is not present")
    config = json.loads(path.read_text())
    # The physical schemas differ, but their public control specifications agree.
    assert ("bounds" in config) == (simulator == "crazyflow")
    channels = [c for c in config["spec"]["channels"] if c["kind"] == "control"]
    lower = np.array([c["minimum"] for c in channels])
    upper = np.array([c["maximum"] for c in channels])
    entry = training_entry(p, simulator)
    dt, steps = config["dt_s"], round(3 / config["dt_s"])
    detail, indices, assigned = data._schedule(dt, steps, lower, upper, entry, p)
    base = np.broadcast_to((lower + upper) / 2, (steps, len(lower))).copy()
    commands, clipped = data._issued(
        base, assigned, detail["start_index"], lower, upper
    )
    old = {
        "commands": base,
        "states": np.zeros((steps + 1, 13)),
        "control_prefix": base[:10].copy(),
    }
    new = dict(
        old,
        commands=commands,
        excitation_pilot_commands=base,
        excitation_block_index=indices,
        excitation_assigned_delta=assigned,
        excitation_realized_delta=commands - base,
        excitation_clipped=clipped,
        excitation_clipping_residual=commands - (base + assigned),
    )
    original = {"metadata": {"initial": "unchanged"}}
    record = {"metadata": dict(original["metadata"], command_excitation=detail)}
    data._check_parent(new, old, record, original, config, entry, flight, p)


@pytest.mark.parametrize("dt", [0.01, 0.05])
def test_actual_commands_drive_physics_and_prefix_is_exact(dt):
    p, f = protocol(), ToyFixture(dt)
    entry = training_entry(p)
    original, metadata = ToyFixture(dt).generate(entry, {})
    kernel = f._advance
    arrays, detail = data.excite(f, entry, {}, p)
    assert f._advance is kernel
    start = round(0.75 / dt)
    for key in original:
        count = start + int(key in data.BOUNDARY_ARRAYS)
        actual, expected = (
            (arrays[key], original[key])
            if key == "control_prefix"
            else (arrays[key][:count], original[key][:count])
        )
        assert actual.tobytes() == expected.tobytes(), key
    assert np.signbit(arrays["commands"][0, 0])
    np.testing.assert_array_equal(arrays["commands"], f.visited)
    np.testing.assert_allclose(
        np.diff(arrays["states"][:, 3:5], axis=0), dt * arrays["commands"], atol=3e-16
    )
    assert not np.array_equal(
        arrays["commands"][start:], arrays["excitation_pilot_commands"][start:]
    )
    assert np.any(arrays["excitation_clipped"])
    assert np.any(arrays["excitation_clipping_residual"])
    assert {k: v for k, v in detail.items() if k != "command_excitation"} == metadata


def test_kernel_restored_on_failure_and_next_original_call_is_unchanged():
    p, f = protocol(), ToyFixture(failure=19)
    entry, kernel = training_entry(p), f._advance
    with pytest.raises(RuntimeError, match="deliberate"):
        data.excite(f, entry, {}, p)
    assert f._advance is kernel
    f.failure = None
    actual, _ = f.generate(entry, {})
    expected, _ = ToyFixture().generate(entry, {})
    flight.array_equal(actual, expected, "restored fixture")


@pytest.mark.parametrize("role", ["development", "test"])
def test_no_excitation_outside_literal_training_roster(role):
    p = protocol()
    if role == "development":
        name = p["planned_automatic_roles"]["crazyflow"]["development"][0]
        entry = next(e for e in p["recordings"] if e["id"] == name)
    else:
        entry = next(e for e in p["recordings"] if e["role"] == "test")
    with pytest.raises(ValueError, match="training roster"):
        data.excite(ToyFixture(), entry, {}, p)


@pytest.mark.parametrize(
    "mutation",
    ["fractional_start", "fractional_block", "missing_training", "duplicate_roles"],
)
def test_invalid_schedule_or_roles_are_rejected(mutation):
    p = protocol()
    entry = training_entry(p)
    if mutation == "fractional_start":
        p["collection_intervention"]["start_s"] = 0.751
    elif mutation == "fractional_block":
        p["collection_intervention"]["block_s"] = 0.11
    elif mutation == "missing_training":
        p["planned_automatic_roles"]["crazyflow"]["training"].pop()
    else:
        roles = p["planned_automatic_roles"]["crazyflow"]
        roles["development"][0] = roles["training"][0]
    with pytest.raises(ValueError):
        data.excite(ToyFixture(), entry, {}, p)


def tiny_data(tmp_path, monkeypatch):
    p, runtime = protocol(), {"toy": True}
    entries = [
        {
            "id": f"crazyflow/toy/train-{i}",
            "seed": i,
            "simulator": "crazyflow",
            "cell": "toy",
            "role": "calibration_pool",
        }
        for i in range(6)
    ]
    p["recordings"] = entries
    roles = flight.roles_for([e["id"] for e in entries])
    p["planned_automatic_roles"]["crazyflow"] = roles
    p["counts"]["excited_training_parents_per_simulator"] = 4
    p["counts"]["unchanged_development_copies_per_simulator"] = 2
    f = ToyFixture()
    monkeypatch.setattr(flight, "fixture_for", lambda *_: ToyFixture())
    directory = tmp_path / "crazyflow/data"
    directory.mkdir(parents=True)
    flight.write_json(
        directory / "configuration.json", dict(f.config(), spec=f.spec.to_dict())
    )
    records = []
    for entry in entries:
        arrays, metadata = f.generate(entry, {"id": "toy"})
        record = dict(
            entry,
            prefix="parents/" + entry["id"],
            metadata=metadata,
            cell_facts={"id": "toy"},
            validity=flight.validity(arrays),
            achieved=flight.achieved(arrays),
        )
        flight.save_arrays(directory / (record["prefix"] + ".npz"), arrays)
        flight.write_json(directory / (record["prefix"] + ".json"), record)
        records.append(record)
    flight.write_json(directory / "records.json", records)
    flight.write_json(directory / "roles.json", roles)
    flight.freeze_files(
        directory,
        "seal.json",
        {"runtime": runtime, "roles": roles, "admitted": [e["id"] for e in entries]},
    )
    return p, runtime


def reseal(directory):
    seal = flight.read_json(directory / "seal.json")
    del seal["files"]
    (directory / "seal.json").unlink()
    flight.freeze_files(directory, "seal.json", seal)


def test_generate_copy_roles_and_exact_physical_replay(tmp_path, monkeypatch):
    p, runtime = tiny_data(tmp_path, monkeypatch)
    result = data.generate_excitation("crazyflow", tmp_path, flight, p, runtime)
    assert result["excited_training_parents"] == 4
    assert result["copied_development_parents"] == 2
    assert result["development_files_exact"]
    replay = data.replay_excitation("crazyflow", tmp_path, flight, p, runtime)
    assert replay["regenerated_training_parents"] == 4
    assert replay["exact"]
    assert replay["replay_arrays"] == result["arrays"]
    for name in p["planned_automatic_roles"]["crazyflow"]["development"]:
        for suffix in (".json", ".npz"):
            path = "parents/" + name + suffix
            assert (tmp_path / "crazyflow/excitation" / path).read_bytes() == (
                tmp_path / "crazyflow/data" / path
            ).read_bytes()


def test_first_excited_boundary_failure_remains_visible_and_admitted(
    tmp_path, monkeypatch
):
    p, runtime = tiny_data(tmp_path, monkeypatch)

    class FallingFixture(ToyFixture):
        def kernel(self, state, command, index, wind):
            following, winds, forces = super().kernel(state, command, index, wind)
            if index >= 15:
                following[2] = 0.4
            return following, winds, forces

    monkeypatch.setattr(flight, "fixture_for", lambda *_: FallingFixture())
    data.generate_excitation("crazyflow", tmp_path, flight, p, runtime)
    records = flight.read_json(tmp_path / "crazyflow/excitation/records.json")
    training = set(p["planned_automatic_roles"]["crazyflow"]["training"])
    failed = [r for r in records if r["id"] in training]
    assert len(failed) == 4
    assert all(r["validity"]["reason"] == "altitude_below_0.5m" for r in failed)
    assert all(r["validity"]["valid_transitions"] == 15 for r in failed)
    result = data.replay_excitation("crazyflow", tmp_path, flight, p, runtime)
    assert result["roles_and_admission_exact"] and result["exact"]


@pytest.mark.parametrize(
    "mutation", ["role", "development", "assigned", "metadata", "prefix", "physics"]
)
def test_coherent_alterations_are_rejected(tmp_path, monkeypatch, mutation):
    p, runtime = tiny_data(tmp_path, monkeypatch)
    data.generate_excitation("crazyflow", tmp_path, flight, p, runtime)
    directory = tmp_path / "crazyflow/excitation"
    records = flight.read_json(directory / "records.json")
    roles = p["planned_automatic_roles"]["crazyflow"]
    name = roles["development" if mutation == "development" else "training"][0]
    record = next(r for r in records if r["id"] == name)
    path = directory / (record["prefix"] + ".npz")
    arrays = flight.load_arrays(path)
    if mutation == "role":
        changed = deepcopy(roles)
        changed["training"][0], changed["development"][0] = (
            changed["development"][0],
            changed["training"][0],
        )
        flight.write_json(directory / "roles.json", changed)
    elif mutation == "metadata":
        record["metadata"]["command_excitation"]["signs"][0][0] *= -1
    else:
        if mutation == "assigned":
            arrays["excitation_assigned_delta"][15, 0] += 0.01
        elif mutation == "prefix":
            arrays["states"][10, 3] += 0.01
        else:
            arrays["states"][-1, 3] += 0.01
            arrays["full_states"][-1, 3] += 0.01
        flight.save_arrays(path, arrays)
        record["validity"], record["achieved"] = (
            flight.validity(arrays),
            flight.achieved(arrays),
        )
    flight.write_json(directory / (record["prefix"] + ".json"), record)
    flight.write_json(directory / "records.json", records)
    reseal(directory)
    if mutation == "physics":
        # A self-consistent post-prefix physical rewrite needs fresh kernel execution.
        data.compare_excitation("crazyflow", tmp_path, flight, p)
        with pytest.raises(ValueError, match="fresh array bytes"):
            data.replay_excitation("crazyflow", tmp_path, flight, p, runtime)
    else:
        with pytest.raises(ValueError):
            data.compare_excitation("crazyflow", tmp_path, flight, p)
