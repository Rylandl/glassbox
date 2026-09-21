"""Reconstruct a real analytic-stream update without applying it to a session."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from glassbox import (
    STATE_CHANNELS,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
    online,
)
from glassbox._learner_arrays import load_arrays

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import audit_online as audit


def analytic_prefix():
    """Independent linear velocity/rate plant with trapezoidal SO3 transport."""
    dt, count = 0.05, 17
    rng = np.random.default_rng(428)
    commands = rng.uniform(-0.5, 0.5, (count, 2))
    states = np.zeros((count + 1, 15))
    states[0, :6] = [0.2, -0.1, 0.3, 0.1, -0.05, 0.08]
    states[0, 6:] = np.eye(3).ravel()
    mixing = np.array([[0.8, -0.3], [0.2, 0.6], [-0.4, 0.3]])
    for row, command in enumerate(commands):
        states[row + 1, :3] = states[row, :3] + dt * (
            mixing @ command - 0.2 * states[row, :3]
        )
        states[row + 1, 3:6] = states[row, 3:6] + dt * (
            mixing @ command - 0.8 * states[row, 3:6]
        )
        states[row + 1, 6:] = (
            states[row, 6:].reshape(3, 3)
            @ Rotation.from_rotvec(
                0.5 * dt * (states[row, 3:6] + states[row + 1, 3:6])
            ).as_matrix()
        ).ravel()
    prefix = SequenceCollection(
        (SequenceSegment("analytic", "prefix", states[:16], commands[:15], dt, 0),),
        "opaque",
        STATE_CHANNELS,
        ("command-a [1]", "command-b [1]"),
    )
    return prefix, commands[15], states[16]


def test_reconstructed_production_update_matches_real_observe_without_mutation(
    tmp_path, monkeypatch
):
    prefix, command, truth = analytic_prefix()
    session = OnlineFit(prefix)
    before = session.fingerprint()
    session.save(tmp_path / "before.npz")
    oracle = OnlineFit.load(tmp_path / "before.npz")
    oracle.observe(15, command, truth)
    oracle.save(tmp_path / "after.npz")
    expected_meta, expected_arrays = load_arrays(tmp_path / "after.npz")

    def forbidden(*args, **kwargs):
        raise AssertionError("audit initialized or applied an online update")

    monkeypatch.setattr(OnlineFit, "__init__", forbidden)
    monkeypatch.setattr(OnlineFit, "observe", forbidden)
    result = audit.reconstruct_update(
        online, session, dict(origin=np.asarray(15), command=command, truth=truth)
    )
    assert session.fingerprint() == before
    assert result["after_metadata"] == expected_meta
    assert set(result["after_arrays"]) == set(expected_arrays)
    for name, expected in expected_arrays.items():
        actual = result["after_arrays"][name]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), name
    assert result["summary"]["core_after"] == oracle.model.fingerprint
    assert result["summary"]["session_after"] == oracle.fingerprint()
    # The derivative-only path must not compute a PCG proposal.
    monkeypatch.setattr(online, "_proposal", forbidden)
    prepared = audit.prepare_update(online, session, dict(command=command, truth=truth))
    assert session.fingerprint() == before
    for key in ("params", "norms"):
        for name, expected in result["prepared"][key].items():
            np.testing.assert_array_equal(prepared[key][name], expected)
