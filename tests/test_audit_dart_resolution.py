"""No native calls: qualification boundaries, tape semantics and authority."""

import importlib.util
import json
import sys
from pathlib import Path

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
    manifest = {
        name: audit.digest(path / name)
        for name in ["binding.json", "report.json", "trajectory.npz", "prelude.npz"]
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
