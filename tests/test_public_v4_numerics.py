"""Qualification packet, independent reduction and process isolation contracts.

Numerical tests run only after the implementation commit is bound by the root
qualification supervisor. These tests do not fit models or select a recipe.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import public_v4_numerics as n


def test_artificial_fixture_roster_and_explicit_nonzero_quadratic_paths():
    for spec in n.FIXTURES:
        metadata, arrays = n.artificial_fixture(spec)
        name, d, m, _, delay, context, horizon = spec
        f = (delay + 1) * (d + m) + 8
        assert metadata["fixture"] == name
        assert arrays["past_states"].shape == (3, context + 1, d)
        assert arrays["past_inputs"].shape == (3, context, m)
        assert arrays["future_inputs"].shape == (3, horizon, m)
        assert arrays["param_linear"].shape == (f, d)
        assert arrays["param_interaction"].shape == (d * m, d)
        assert arrays["param_autonomous"].shape == (d * (d + 1) // 2, d)
        assert all(np.any(arrays["param_" + key]) for key in n.PARAMETERS)
        n._check_arrays({**metadata, "horizon": horizon}, arrays)


def test_stresses_change_only_declared_observation_units_and_offsets():
    spec = n.FIXTURES[1]
    _, nominal = n.artificial_fixture(spec)
    for stress in ("offset-2p30", "small-scale-offset"):
        _, altered = n.artificial_fixture(spec, stress=stress)
        for key in nominal:
            if key not in {"norm_state_mean", "norm_state_scale", "past_states"}:
                np.testing.assert_array_equal(nominal[key], altered[key])
    with pytest.raises(ValueError, match="unknown"):
        n.artificial_fixture(spec, stress="posthoc")


def test_packet_has_five_nominal_and_four_stress_horizon_cases(tmp_path):
    result = n.prepare(
        tmp_path / "inputs", provenance={"protocol_sha256": n.PROTOCOL_SHA256}
    )
    manifest = n.read_json(result["manifest"])
    assert result["cases"] == 9
    assert sum(c["stress"] is None for c in manifest["cases"]) == 5
    assert len({c["id"] for c in manifest["cases"]}) == 9
    for case in manifest["cases"]:
        path = Path(result["manifest"]).parent / case["path"]
        assert n.digest(path) == case["sha256"]
    with pytest.raises(ValueError, match="24"):
        n.prepare(
            tmp_path / "other",
            provenance={"protocol_sha256": n.PROTOCOL_SHA256},
            archive_cases=[{}],
        )


@pytest.mark.parametrize(
    "change", ["missing", "float32", "nonfinite", "history", "horizon"]
)
def test_packet_rejects_wrong_array_contract(change):
    meta, arrays = n.artificial_fixture(n.FIXTURES[1])
    meta["horizon"] = meta["maximum_horizon"]
    if change == "missing":
        arrays.pop("param_autonomous")
    elif change == "float32":
        arrays["past_states"] = arrays["past_states"].astype(np.float32)
    elif change == "nonfinite":
        arrays["future_inputs"][0, 0, 0] = np.nan
    elif change == "history":
        arrays["past_inputs"] = arrays["past_inputs"][:, 1:]
    else:
        meta["horizon"] += 1
    with pytest.raises(ValueError):
        n._check_arrays(meta, arrays)


def test_elementwise_bound_does_not_average_away_one_failure():
    reference = np.ones(100)
    altered = reference.copy()
    altered[-1] += 2e-4
    result = n.element_check(altered, reference, (1e-5, 1e-4))
    assert not result["passed"]
    assert result["worst_index"] == [99]
    assert n.element_check(np.array([0.125]), np.array([0.0]), (0.125, 0.0))["passed"]
    assert not n.element_check(np.array([np.nan]), np.ones(1), (1.0, 1.0))["passed"]
    assert not n.element_check(np.ones((1, 1)), np.ones(1), (1.0, 1.0))["passed"]


def _saved_transforms(arrays, dtype):
    b, h, d = (
        len(arrays["past_states"]),
        len(arrays["future_inputs"][0]),
        len(arrays["norm_state_mean"]),
    )
    mean = np.broadcast_to(arrays["norm_state_mean"], (b, h, d)).astype(dtype).copy()
    row = {k: mean.copy() for k in ("eager", "jit", "vmap", "singles", "jit_singles")}
    for i, key in enumerate(n.INPUTS):
        shape = arrays[key][0].shape
        for method in ("grad", "jit_grad", "grad_jit"):
            row[f"{method}_{i}"] = np.zeros(shape, dtype=dtype)
        for method in ("direction", "achieved_plus", "achieved_minus"):
            row[f"{method}_{i}"] = np.ones(shape, dtype=dtype)
        for method in ("jvp", "jit_jvp"):
            row[f"{method}_{i}"] = np.zeros((h, d), dtype=dtype)
        row[f"fd_{i}"] = np.array(0, dtype=dtype)
    row["causal_perturbed"] = mean[0].copy()
    row["causal_jvp"] = np.zeros((h, d), dtype=dtype)
    row["first_command_jvp"] = np.ones((h, d), dtype=dtype)
    return row


def _evidence(stress=None):
    case, arrays = n.artificial_fixture(n.FIXTURES[1], stress=stress)
    case.update(id="reducer-fixture", horizon=case["maximum_horizon"])
    p64, p32 = (
        _saved_transforms(arrays, np.float64),
        _saved_transforms(arrays, np.float32),
    )
    workers = {
        "public64": {"actual__" + k: v for k, v in p64.items()},
        "public32": {"actual__" + k: v for k, v in p32.items()},
        "oracle64": {
            **{"original__" + k: v for k, v in p64.items()},
            **{"quantized_inputs__" + k: v for k, v in p64.items()},
            "rounded_parameters__eager": p64["eager"].copy(),
        },
    }
    return case, arrays, workers


def test_reduction_requires_each_input_gradient_and_actual_dtype():
    case, arrays, workers = _evidence()
    assert n._case_reduction(case, arrays, workers)["passed"]
    workers["public32"]["actual__grad_1"][0, 0] = 1
    result = n._case_reduction(case, arrays, workers)
    assert not result["passed"]
    assert not result["checks"]["32/grad/1"]["passed"]


@pytest.mark.parametrize(
    "defect", ["omission", "extra", "nan", "causality", "zero_command", "dtype"]
)
def test_reduction_rejects_missing_or_corrupt_transforms(defect):
    case, arrays, workers = _evidence()
    saved = workers["public32"]
    if defect == "omission":
        del saved["actual__grad_jit_2"]
    elif defect == "extra":
        saved["undeclared"] = np.zeros(1)
    elif defect == "nan":
        saved["actual__causal_jvp"][-1] = np.nan
    elif defect == "causality":
        saved["actual__causal_jvp"][0] = 1
    elif defect == "zero_command":
        saved["actual__first_command_jvp"][:] = 0
    else:
        saved["actual__jit"] = saved["actual__jit"].astype(np.float64)
    if defect in ("omission", "extra"):
        with pytest.raises(ValueError, match="roster"):
            n._case_reduction(case, arrays, workers)
    else:
        assert not n._case_reduction(case, arrays, workers)["passed"]


def test_named_stress_is_diagnostic_but_cannot_excuse_causality_failure():
    case, arrays, workers = _evidence("offset-2p30")
    workers["public32"]["actual__grad_0"][0, 0] = 100
    result = n._case_reduction(case, arrays, workers)
    assert result["passed"] and not result["checks"]["32/grad/0"]["passed"]
    workers["public32"]["actual__causal_jvp"][0] = 1
    assert not n._case_reduction(case, arrays, workers)["passed"]


def test_actual_small_command_derivative_is_retained_without_a_new_cutoff():
    case, arrays, workers = _evidence()
    case["source"] = "public_archive"
    workers["public32"]["actual__first_command_jvp"][:] = 0
    assert n._case_reduction(case, arrays, workers)["passed"]


def test_worker_can_start_without_importing_either_glassbox_version():
    tree = ast.parse(Path(n.__file__).read_text())
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("glassbox")
        for node in imports
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name.startswith(("glassbox", "jax")) for alias in node.names)
        for node in imports
    )


def test_runner_uses_three_isolated_processes_and_preserves_failed_stages(
    tmp_path, monkeypatch
):
    public = Path(n.__file__).resolve().parents[3]
    oracle = tmp_path / "old"
    implementation = "a" * 40
    provenance = dict(
        protocol_sha256=n.PROTOCOL_SHA256,
        implementation_commit=implementation,
        public_source_sha256={
            str(Path(n.__file__).resolve().relative_to(public)): n.digest(n.__file__)
        },
        oracle_source_sha256={"old.py": "b" * 64},
        runtime={},
    )
    packet = n.prepare(tmp_path / "packet", provenance=provenance)
    monkeypatch.setattr(
        n,
        "_git",
        lambda root, *args: (
            ""
            if args[0] == "status"
            else n.ORACLE_COMMIT
            if root == oracle
            else implementation
        ),
    )
    monkeypatch.setattr(n, "_sources", lambda *args: None)
    calls = []

    def child(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(n.subprocess, "run", child)
    result = n.run(
        packet["manifest"],
        tmp_path / "result",
        expected_manifest_sha256=packet["manifest_sha256"],
        public_root=public,
        oracle_root=oracle,
    )
    assert not result["passed"] and result["fits"] == 0
    assert len(calls) == 3
    assert "JAX_ENABLE_X64" not in calls[0][1]["env"]
    assert calls[1][1]["env"]["JAX_ENABLE_X64"] == "1"
    assert calls[2][1]["env"]["PYTHONPATH"] == str(oracle / "src")
    assert calls[0][1]["env"]["PYTHONPATH"] == str(public / "src")
    assert all(call[0][1] == str(Path(n.__file__).resolve()) for call in calls)
    assert len(n.read_json(tmp_path / "result/stages.json")) == 3


def test_wrong_external_anchor_rejects_before_process_launch(tmp_path, monkeypatch):
    packet = n.prepare(
        tmp_path / "packet", provenance={"protocol_sha256": n.PROTOCOL_SHA256}
    )
    monkeypatch.setattr(
        n.subprocess, "run", lambda *a, **k: pytest.fail("worker must not start")
    )
    with pytest.raises(ValueError, match="external"):
        n.run(
            packet["manifest"],
            tmp_path / "output",
            expected_manifest_sha256="0" * 64,
            public_root=tmp_path,
            oracle_root=tmp_path,
        )


def test_imported_source_must_come_from_requested_checkout(tmp_path, monkeypatch):
    monkeypatch.setitem(
        n.sys.modules,
        "glassbox.wrong_version_fixture",
        SimpleNamespace(__file__=str(tmp_path / "unexpected.py")),
    )
    with pytest.raises(ValueError, match="escaped"):
        n._imported_sources(tmp_path / "declared", {})
