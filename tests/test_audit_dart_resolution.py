"""No native calls: qualification boundaries, tape semantics and authority."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

s = importlib.util.spec_from_file_location(
    "audit", Path(__file__).parents[1] / "scripts/audit_dart_resolution.py"
)
audit = importlib.util.module_from_spec(s)
s.loader.exec_module(audit)


def score():
    return dict(
        contact=True,
        hit=True,
        miss_distance_m=0.0005,
        axis_error_deg=1.0,
        normal_speed_m_s=1.5,
        tangent_speed_m_s=0.1,
        time_s=1.1,
        contact_position_m=[0.65, 0.0005, 1.0],
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("miss_distance_m", 0.001),
        ("axis_error_deg", 5.01),
        ("normal_speed_m_s", 0.19),
        ("normal_speed_m_s", 2.51),
        ("tangent_speed_m_s", 0.201),
        ("time_s", 1.201),
        ("contact", False),
        ("hit", False),
    ],
)
def test_primary_limits_are_not_relaxed(key, value):
    assert audit.precision_pass(score())
    assert not audit.precision_pass(dict(score(), **{key: value}))


def test_finer_tape_preserves_every_issued_command_without_interpolation():
    tape = np.arange(12, dtype=np.float64).reshape(3, 4)
    for dt, repeat in [(0.001, 10), (0.0005, 20)]:
        actual = audit.expand_commands(tape, dt)
        assert actual.shape == (3 * repeat, 4) and actual.dtype == tape.dtype
        for i, command in enumerate(tape):
            np.testing.assert_array_equal(
                actual[i * repeat : (i + 1) * repeat], np.tile(command, (repeat, 1))
            )
    with pytest.raises(ValueError, match="frozen audit grid"):
        audit.expand_commands(tape, 0.002)


def test_contact_point_and_time_convergence_checks_both_scorers():
    a = dict(canonical=score(), independent=score())
    b = dict(canonical=score(), independent=score())
    assert audit.convergence([a, b])["passed"]
    b["canonical"] = dict(score(), contact_position_m=[0.65, 0.00053, 1.0])
    assert not audit.convergence([a, b])["passed"]
    b["canonical"] = score()
    b["independent"] = dict(score(), time_s=1.10003)
    assert not audit.convergence([a, b])["passed"]
    b["independent"] = dict(contact=False, hit=False)
    assert not audit.convergence([a, b])["passed"]


def fixture_trial(path):
    protocol = path / "protocol.json"
    protocol.write_text('{"frozen":true}')
    binding = dict(protocol_sha256=audit.digest(protocol))
    report = dict(
        protocol_sha256=audit.digest(protocol),
        status="complete",
        all_finite=True,
        coarse_submillimeter=True,
        contact=score(),
        independent_contact=score(),
    )
    (path / "binding.json").write_text(json.dumps(binding))
    (path / "report.json").write_text(json.dumps(report))
    np.savez(path / "trajectory.npz", states=np.zeros((2, 17), dtype=np.float32))
    np.savez(path / "prelude.npz", states=np.zeros((51, 17), dtype=np.float32))
    (path / "arrays.bin").write_bytes(b"")
    (path / "events.jsonl").write_text("")
    manifest = {
        name: audit.digest(path / name)
        for name in [
            "binding.json",
            "report.json",
            "trajectory.npz",
            "prelude.npz",
            "arrays.bin",
            "events.jsonl",
        ]
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return protocol, audit.digest(path / "manifest.json")


def test_original_external_authority_rejects_numeric_mutation(tmp_path):
    protocol, seal = fixture_trial(tmp_path)
    audit.authenticate_trial(tmp_path, seal, protocol)
    np.savez(tmp_path / "trajectory.npz", states=np.ones((2, 17), dtype=np.float32))
    with pytest.raises(ValueError, match="trial file hash"):
        audit.authenticate_trial(tmp_path, seal, protocol)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["trajectory.npz"] = audit.digest(tmp_path / "trajectory.npz")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trial manifest hash"):
        audit.authenticate_trial(tmp_path, seal, protocol)


def test_protocol_identity_is_required_even_for_a_complete_trial(tmp_path):
    protocol, seal = fixture_trial(tmp_path)
    protocol.write_text('{"frozen":false}')
    with pytest.raises(ValueError, match="trial protocol"):
        audit.authenticate_trial(tmp_path, seal, protocol)


def test_existing_output_is_never_modified(tmp_path):
    with pytest.raises(FileExistsError):
        audit.run(tmp_path, tmp_path, tmp_path, "bad", tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("raise_inside", [False, True])
def test_scoped_precision_promotes_exact_coefficients_and_restores_on_every_exit(
    raise_inside,
):
    with jax.enable_x64(False):
        # Deliberately not a consistent inverse: promotion must not recompute J_inv.
        original = dict(mass=jnp.asarray(0.0319), J_inv=jnp.asarray([[1.234567]]))
        plant = SimpleNamespace(params=original, thrust_max=0.12)
        try:
            with audit.native_precision(plant) as values:
                assert jax.config.x64_enabled
                for key, value in original.items():
                    assert plant.params[key].dtype == np.float64
                    np.testing.assert_array_equal(
                        plant.params[key], np.asarray(value).astype(np.float64)
                    )
                    assert values["source_" + key].dtype == np.float32
                    assert values["promoted_" + key].dtype == np.float64
                assert float(plant.thrust_max) == float(np.float32(0.12)) != 0.12
                if raise_inside:
                    raise RuntimeError("fixture failure")
        except RuntimeError as error:
            assert raise_inside and str(error) == "fixture failure"
        assert not jax.config.x64_enabled
        assert plant.params is original and plant.thrust_max == 0.12


def native_journal(path):
    commands = np.asarray([[0.1234567890123, 0.2, 0.3, 0.4], [0.8, 0.7, 0.6, 0.5]])
    states = np.arange(51, dtype=np.float32).reshape(3, 17)
    journal = audit.Journal(path)
    for i in range(2):
        journal.call(
            "native",
            i,
            dict(state=states[i], command=commands[i].astype(np.float32)),
            lambda x: x,
            (states[i + 1],),
            lambda result: dict(state=result),
        )
    journal.close()
    return dict(states=states, commands=commands)


def test_actual_native_journal_discards_extra_optimizer_precision(tmp_path):
    trajectory = native_journal(tmp_path)
    issued = audit.issued_commands(tmp_path, trajectory)
    assert issued.dtype == np.float32
    np.testing.assert_array_equal(issued, trajectory["commands"].astype(np.float32))
    assert issued.astype(np.float64)[0, 0] != trajectory["commands"][0, 0]


def test_actual_native_journal_rejects_command_mismatch(tmp_path):
    trajectory = native_journal(tmp_path)
    trajectory["commands"][0, 0] += 0.01
    with pytest.raises(ValueError, match="actually issued command"):
        audit.issued_commands(tmp_path, trajectory)


def test_actual_native_journal_rejects_unreturned_call(tmp_path):
    trajectory = native_journal(tmp_path)
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="native command count"):
        audit.issued_commands(tmp_path, trajectory)


def test_supplement_pins_trial_protocol_and_preserved_float32_failure(
    tmp_path, monkeypatch
):
    trial, previous = tmp_path / "trial", tmp_path / "previous"
    trial.mkdir()
    previous.mkdir()
    protocol, authority = fixture_trial(trial)
    audit.write(
        previous / "report.json",
        dict(
            trial_manifest_sha256=authority,
            protocol_sha256=audit.digest(protocol),
            resolution_qualified=False,
        ),
    )
    audit.write(
        previous / "manifest.json",
        {"report.json": audit.digest(previous / "report.json")},
    )
    supplement = tmp_path / "supplement.json"
    audit.write(
        supplement,
        dict(
            trial_manifest_sha256=authority,
            trial_protocol_sha256=audit.digest(protocol),
            original_float32_audit_manifest_sha256=audit.digest(
                previous / "manifest.json"
            ),
        ),
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "SUPPLEMENT", supplement)
    monkeypatch.setattr(audit.subprocess, "check_call", lambda *a, **k: None)
    result = audit.precision_contract(trial, authority, protocol)
    assert result["arithmetic"] == "float64"
    assert result["supplemental_contract_sha256"] == audit.digest(supplement)
    assert result["original_float32_resolution_qualified"] is False
    with pytest.raises(ValueError, match="precision supplement"):
        audit.precision_contract(trial, "another trial", protocol)
    (previous / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="original float32 audit file"):
        audit.precision_contract(trial, authority, protocol)
