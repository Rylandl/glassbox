"""Integrity contracts for the no-fit qualification runner."""

import json
from pathlib import Path

import pytest

from glassbox.experimental import qualification as q


def test_frozen_plan_checks_inherited_sources():
    plan, raw = q.frozen_plan()
    assert plan["no_fit"] is True
    assert q.digest(raw) == q.PLAN_SHA256
    assert plan["control_qualification"]["seeds"] == [101, 102]


def test_input_snapshot_preflights_all_bytes_and_remains_stable(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b'{"value": 1}')
    second.write_bytes(b'{"value": 2}')
    plan = {
        "input_sha256": {
            "control/first.json": q.digest(first.read_bytes()),
            "control/second.json": q.digest(second.read_bytes()),
        }
    }
    snapshot = q.input_snapshot(plan, {"control": tmp_path})
    first.write_bytes(b'{"value": 3}')
    assert json.loads(snapshot["control/first.json"]) == {"value": 1}
    with pytest.raises(ValueError, match="input changed"):
        q.input_snapshot(plan, {"control": tmp_path})


@pytest.mark.parametrize(
    "actual, expected",
    [
        ({"a": 1}, {"a": 1, "b": 2}),
        (True, 1),
        (1, True),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        ([1], [1, 2]),
        (0.9, 1.0),
    ],
)
def test_report_comparison_rejects_structural_and_numeric_forgery(actual, expected):
    with pytest.raises(ValueError):
        q.same(actual, expected)


def test_report_comparison_accepts_only_declared_numeric_tolerance():
    q.same({"a": [1.0 + 1e-10, False, 2]}, {"a": [1.0, False, 2]})


def test_prospective_inventory_requires_all_four_trials():
    plan, _ = q.frozen_plan()
    names = q._artifact_names(plan, True)
    assert len([name for name in names if name.endswith("tracking.npz")]) == 4
    assert len([name for name in names if name.endswith("oracle.npz")]) == 2
    assert "report.json" in names
    assert q._artifact_names(plan, False) == {"report.json", "run.json"}


def test_runner_never_calls_fit_or_update():
    # The runner loads archived models; a fitting call would violate this
    # experiment even if its output happened to be unchanged.
    import ast

    tree = ast.parse(Path(q.__file__).read_text())
    calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(
        (isinstance(f, ast.Name) and f.id in {"fit", "update"})
        or (isinstance(f, ast.Attribute) and f.attr in {"fit", "update_recordings"})
        for f in calls
    )


def test_context_builds_equations_at_plant_precision(monkeypatch):
    """A cast x64 inertia inverse differs from a directly built float32 one."""
    from types import SimpleNamespace

    import jax
    import numpy as np

    from glassbox import learner as default_model
    from glassbox.belief import belief_io
    from glassbox.experimental import harness

    construction = []
    spec = SimpleNamespace(to_model=lambda: construction.append(jax.config.x64_enabled))

    def fixture(_manifest):
        assert jax.config.x64_enabled
        return spec, "trim_model_only", None, np.zeros(13), np.zeros(3)

    monkeypatch.setattr(harness, "control_fixture", fixture)
    monkeypatch.setattr(
        default_model.LearnedDynamics,
        "load",
        lambda _data: SimpleNamespace(fingerprint=lambda: "pinned"),
    )
    monkeypatch.setattr(
        belief_io, "dynamics_belief_from_payload", lambda _payload: None
    )
    inputs = {
        "control/manifest.json": b"{}",
        "control/calibration.json": json.dumps(
            {
                "initial_state": [0.0] * 13,
                "initial_command": [0.0] * 3,
                "generic_fingerprint": "pinned",
            }
        ).encode(),
        "control/generic.npz": b"unused",
        "control/structured.json": b"{}",
    }
    with jax.enable_x64(False):
        q._context(inputs)
    assert construction == [False]
    with jax.enable_x64(True), pytest.raises(ValueError, match="float32"):
        q._context(inputs)
