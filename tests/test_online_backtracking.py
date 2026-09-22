"""Explicit nonlinear and residual-only tests for the backtracking diagnostic."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import audit_backtracking as audit

ALPHAS = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])


def _raw(x):
    value = np.zeros((1, 1, 15), dtype=np.float64)
    value[0, 0, 0] = x**2 - 1
    return value


def _data(raw):
    result = 0.0
    for start, end in ((0, 3), (3, 6), (6, 15)):
        norm = np.linalg.norm(raw[0, 0, start:end])
        result += (norm - 0.5 if norm > 1 else 0.5 * norm**2) / 3
    return result


def _fixture():
    theta = np.array([0.25])
    prior = np.array([0.001])
    raw = _raw(theta[0])
    gradient = np.array([0.5 * raw[0, 0, 0] / 3]) + prior * theta
    deltas = np.array([[0.75], [0.75], [2.25], [2.25]])
    projected = np.zeros((4, *raw.shape))
    projected[:, 0, 0, 0] = 0.5 * deltas[:, 0]
    trial = np.array([_raw(theta[0] + d[0]) for d in deltas])
    arrays = dict(
        theta=theta,
        prior=prior,
        raw=raw,
        weight=np.full((1, 1, 1), 1 / 3),
        irls=np.ones_like(raw),
        gradient=gradient,
        gradient_data=gradient - prior * theta,
        gradient_prior=prior * theta,
        damping=np.array(1000.0),
        finite=np.array(True),
        four_finite=np.array(True),
        probe_delta=deltas,
        probe_projected=projected,
        probe_trial_raw=trial,
        production_trial=np.array(0.0005),
    )
    data = _data(raw)
    penalty = 0.5 * float(theta @ (prior * theta))
    probes = []
    for i, delta in enumerate(deltas):
        candidate = theta + delta
        trial_data = _data(trial[i])
        trial_prior = 0.5 * float(candidate @ (prior * candidate))
        predicted = -float(gradient @ delta) - 0.5 * (
            np.sum(projected[i] ** 2) / 3 + float(delta @ (prior * delta))
        )
        total = trial_data + trial_prior
        gain = (data + penalty - total) / predicted
        probes.append(
            dict(
                label=(
                    "four_raw",
                    "four_trusted",
                    "reference_raw",
                    "reference_trusted",
                )[i],
                finite=True,
                data_loss=trial_data,
                prior_loss=trial_prior,
                total_loss=total,
                data_improvement=data - trial_data,
                prior_improvement=penalty - trial_prior,
                improvement=data + penalty - total,
                predicted_reduction=predicted,
                gain=gain,
                would_accept=bool(total < data + penalty and gain >= 0.1),
            )
        )
    summary = dict(
        current=dict(data_loss=data, prior_loss=penalty, total_loss=data + penalty),
        reference=dict(status="capped", iterations=128, relative_residual=0.03),
        probes=probes,
    )
    trials = np.array([_raw(theta[0] + alpha * deltas[3, 0]) for alpha in ALPHAS])
    return summary, arrays, trials


def _assert_preserved(before, after):
    assert jax.tree.structure(before) == jax.tree.structure(after)
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_array_equal(actual, expected)


def test_nonlinear_shortening_first_pass_and_comparison_to_actual_four():
    parent, arrays, trials = _fixture()
    before = copy.deepcopy((parent, arrays, trials))
    result = audit.summarize(parent, arrays, trials)
    steps = result["steps"]
    assert not steps[0]["would_accept"]
    assert steps[0]["total_loss"] > result["current"]["total_loss"]
    assert steps[1]["would_accept"]
    assert steps[2]["total_loss"] < steps[1]["total_loss"]
    assert result["selected_index"] == 1
    assert result["selected_alpha"] == 0.5
    assert result["selection_comparison"] == "loss"
    assert result["retained_comparison"] == "loss"
    assert result["retained_total_loss"] == steps[1]["total_loss"]
    assert result["original_four"]["total_loss"] == 0.0005
    assert result["hypothetical_residual_evaluations"] == 2
    assert len(steps) == 5
    assert result["parent_reference"] == parent["reference"]

    # Explicit scalar formula: the very large damping must not enter prediction.
    for alpha, step in zip(ALPHAS, steps):
        displacement = alpha * 2.25
        x = 0.25 + displacement
        expected_prediction = -arrays["gradient"][0] * displacement - 0.5 * (
            (0.5 * displacement) ** 2 / 3 + 0.001 * displacement**2
        )
        np.testing.assert_allclose(step["data_loss"], _data(_raw(x)), atol=1e-14)
        np.testing.assert_allclose(step["prior_loss"], 0.0005 * x**2, atol=1e-14)
        np.testing.assert_allclose(
            step["predicted_reduction"], expected_prediction, atol=1e-14
        )
        assert step["alpha"] == alpha
    assert result["work"]["new_residual_calls"] == 4
    assert result["work"]["reused_residuals"] == 1
    for key in (
        "cg_iterations",
        "conditioning_calls",
        "gradient_calls",
        "initialization_calls",
        "observe_calls",
        "applied_updates",
    ):
        assert result["work"][key] == 0
    _assert_preserved(before, (parent, arrays, trials))


@pytest.mark.parametrize("invalid", ["conditioning", "reference"])
def test_ineligible_direction_cannot_be_rescued_by_a_smaller_step(invalid):
    parent, arrays, trials = _fixture()
    if invalid == "reference":
        arrays["finite"] = np.array(False)
        parent["probes"][3]["finite"] = False
    result = audit.summarize(
        parent, arrays, trials, conditioning_finite=invalid != "conditioning"
    )
    assert all(
        not step["finite"] and not step["would_accept"] for step in result["steps"]
    )
    assert result["selected_index"] is None
    assert result["selected_alpha"] is None
    assert result["selection_comparison"] == "no_selection"
    assert result["retained_comparison"] == "equal"
    assert result["retained_total_loss"] == arrays["production_trial"]
    assert result["hypothetical_residual_evaluations"] == 5


def test_original_production_scalar_controls_strict_comparison():
    parent, arrays, trials = _fixture()
    selected = audit.summarize(parent, arrays, trials)["steps"][1]["total_loss"]
    # Instrumented parent scalar deliberately differs: only production_trial is authoritative.
    parent["probes"][1]["total_loss"] = selected * 10
    for baseline, expected in (
        (np.nextafter(selected, -np.inf), "loss"),
        (selected, "equal"),
        (np.nextafter(selected, np.inf), "win"),
    ):
        arrays["production_trial"] = np.array(baseline)
        result = audit.summarize(parent, arrays, trials)
        assert result["selection_comparison"] == expected
        assert result["original_four"]["total_loss"] == baseline


def _forbidden(*args, **kwargs):
    raise AssertionError("residual-only evaluation called an optimizer or derivative")


def test_residual_only_evaluation_is_float64_and_preserves_inputs(monkeypatch):
    def residual(params, norms, data, scale, delay, dt_s):
        assert params["x"].dtype == jnp.float64
        assert jax.config.x64_enabled
        x = params["x"][0]
        # Every supplied conditioned argument participates in an explicit expression.
        value = (x**2 - norms["target"] + data[0][0] * dt_s + delay) / scale[0]
        return jnp.zeros((1, 1, 15), dtype=jnp.float64).at[0, 0, 0].set(value)

    module = SimpleNamespace(
        _residual=residual,
        _proposal=_forbidden,
        _recondition=_forbidden,
        _curvature_diagonal=_forbidden,
        OnlineFit=_forbidden,
        observe=_forbidden,
    )
    prepared = dict(
        params={"x": np.array([0.25], dtype=np.float64)},
        norms={"target": np.array(1.0)},
        data=(np.array([2.0]),),
        scale=np.array([2.0]),
        delay=2,
        dt_s=0.05,
    )
    direction = np.array([2.25])
    before = copy.deepcopy((prepared, direction, ALPHAS))
    for name in (
        "linearize",
        "linear_transpose",
        "jvp",
        "vjp",
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "hessian",
    ):
        monkeypatch.setattr(jax, name, _forbidden)
    prior_x64 = jax.config.x64_enabled
    actual = audit.evaluate_residuals(module, prepared, direction, ALPHAS)
    expected = np.zeros((5, 1, 1, 15))
    expected[:, 0, 0, 0] = ((0.25 + ALPHAS * 2.25) ** 2 - 1 + 0.1 + 2) / 2
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert jax.config.x64_enabled == prior_x64
    _assert_preserved(before, (prepared, direction, ALPHAS))


def test_residual_evaluation_obeys_saved_pytree_order_and_partial_ladder():
    def residual(params, norms, data, scale, delay, dt_s):
        del norms, data, scale, delay, dt_s
        value = params["a"][0] ** 2 + 2 * params["z"]["b"][0] - params["z"]["b"][1]
        return jnp.full((2, 1, 15), value, dtype=jnp.float64)

    prepared = dict(
        params={"z": {"b": np.array([0.5, 0.25])}, "a": np.array([0.1])},
        norms={},
        data=(),
        scale=np.ones((1, 15)),
        delay=0,
        dt_s=0.01,
    )
    direction = np.array([0.4, 0.6, 0.8])
    alphas = ALPHAS[1:]
    actual = audit.evaluate_residuals(
        SimpleNamespace(_residual=residual), prepared, direction, alphas
    )
    expected = (
        (0.1 + alphas * 0.4) ** 2 + 2 * (0.5 + alphas * 0.6) - (0.25 + alphas * 0.8)
    )
    assert actual.shape == (4, 2, 1, 15)
    np.testing.assert_allclose(
        actual, np.broadcast_to(expected[:, None, None, None], actual.shape), atol=1e-14
    )


def test_nonfinite_full_trial_does_not_invalidate_a_finite_shorter_direction():
    parent, arrays, trials = _fixture()
    trials[0, 0, 0, 0] = np.inf
    parent["probes"][3]["finite"] = False
    result = audit.summarize(parent, arrays, trials)
    assert not result["steps"][0]["finite"]
    assert not result["steps"][0]["would_accept"]
    assert result["steps"][1]["finite"]
    assert result["steps"][1]["would_accept"]
    assert result["selected_index"] == 1


@pytest.mark.parametrize(
    "field", ["id", "ladder", "parent_authority", "parent_protocol"]
)
def test_protocol_rejects_changed_identity_ladder_or_parent(field):
    protocol = audit.read(audit.PROTOCOL)
    if field == "id":
        protocol["id"] = "unknown"
    elif field == "ladder":
        protocol["ladder"][-1] *= 0.5
    elif field == "parent_authority":
        protocol["parent"]["manifest_sha256"] = "0" * 64
    else:
        protocol["parent"]["protocol_id"] = "unknown"
    with pytest.raises(ValueError):
        audit.check_protocol(protocol)


def test_verify_parent_keeps_original_authority_and_requires_package_and_coverage(
    tmp_path, monkeypatch
):
    protocol = audit.read(audit.PROTOCOL)
    protocol["parent"]["path"] = str(tmp_path)
    audit.write(tmp_path / "protocol.json", {"id": "online-proposal-audit-v1"})
    verified = dict(snapshots=46, links=list(range(10)))
    calls = []

    def verify(parent, authority):
        calls.append(("verify", parent, authority))
        return verified

    def check_package(parent_protocol, parent):
        calls.append(("package", parent_protocol, parent))

    monkeypatch.setattr(audit.parent_audit, "verify", verify)
    monkeypatch.setattr(audit.parent_audit, "check_package_sources", check_package)
    parent, parent_protocol, result = audit.verify_parent(protocol)
    assert parent == tmp_path.resolve()
    assert parent_protocol == {"id": "online-proposal-audit-v1"}
    assert result is verified
    assert calls[0] == ("verify", parent, audit.PARENT_AUTHORITY)
    assert calls[1] == ("package", parent_protocol, parent)
    verified["snapshots"] = 45
    with pytest.raises(ValueError, match="parent coverage"):
        audit.verify_parent(protocol)
    verified["snapshots"] = 46
    verified["links"].pop()
    with pytest.raises(ValueError, match="parent coverage"):
        audit.verify_parent(protocol)


def _minimal_pack(path, *, wrong_source=False):
    protocol = audit.read(audit.PROTOCOL)
    audit.write(path / "protocol.json", protocol)
    source = "scripts/audit_backtracking.py"
    audit.write(
        path / "binding.json",
        dict(
            protocol_sha256=audit.digest(path / "protocol.json"),
            source={
                "files": {
                    source: "0" * 64
                    if wrong_source
                    else audit.digest(audit.ROOT / source)
                }
            },
        ),
    )
    (path / "parent-manifest.json").write_text("{}\n")
    return audit.seal(path, "glassbox-online-backtracking-v1")


def test_verifier_rejects_altered_payload_before_model_or_parent_calls(
    tmp_path, monkeypatch
):
    authority = _minimal_pack(tmp_path)
    (tmp_path / "parent-manifest.json").write_text("changed\n")
    monkeypatch.setattr(audit, "verify_parent", _forbidden)
    monkeypatch.setattr(audit, "evaluate_residuals", _forbidden)
    with pytest.raises(ValueError, match="artifact hash differs"):
        audit.verify(tmp_path, authority)


@pytest.mark.parametrize("wrong_source", [False, True])
def test_verifier_rejects_changed_source_or_parent_authority_before_evaluation(
    tmp_path, monkeypatch, wrong_source
):
    authority = _minimal_pack(tmp_path, wrong_source=wrong_source)
    monkeypatch.setattr(audit, "verify_parent", _forbidden)
    monkeypatch.setattr(audit, "evaluate_residuals", _forbidden)
    message = "source changed" if wrong_source else "copied parent authority differs"
    with pytest.raises(ValueError, match=message):
        audit.verify(tmp_path, authority, recompute=True)


def test_synthetic_pack_run_numpy_verify_and_residual_recompute(tmp_path, monkeypatch):
    """Exercise the full pack boundary using only an explicit analytic residual."""
    parent = tmp_path / "parent"
    parent.mkdir()
    runtime = {"fixture": "explicit nonlinear residual"}
    audit.write(parent / "binding.json", {"runtime": runtime})
    authority = audit.seal(parent, "synthetic-parent")
    monkeypatch.setattr(audit, "PARENT_AUTHORITY", authority)
    contract = audit.read(audit.PROTOCOL)
    contract["parent"].update(path=str(parent), manifest_sha256=authority)
    protocol_path = tmp_path / "protocol.json"
    audit.write(protocol_path, contract)
    parent_protocol = {
        "id": "online-proposal-audit-v1",
        "references": {"v6": {}},
        "origins": {"analytic": list(range(46))},
    }
    parent_verified = {"snapshots": 46, "links": list(range(10)), "verified": True}

    def verify_parent(protocol):
        audit.check_protocol(protocol)
        audit.authenticate(parent, authority)
        return parent, parent_protocol, parent_verified

    def bound(protocol):
        return {
            "protocol_sha256": audit.digest(protocol),
            "runtime": runtime,
            "source": {"files": {}},
        }

    def residual(params, norms, data, scale, delay, dt_s):
        del norms, data, scale, delay, dt_s
        return (
            jnp.zeros((1, 1, 15), dtype=jnp.float64)
            .at[0, 0, 0]
            .set(params["x"][0] ** 2 - 1)
        )

    module = SimpleNamespace(_residual=residual)
    prepared = {
        "params": {"x": np.array([0.25])},
        "norms": {},
        "data": (),
        "scale": np.ones((1, 15)),
        "delay": 0,
        "dt_s": 0.05,
        "conditioning_finite": True,
    }

    def point_inputs(parent_path, protocol, version, case, origin):
        assert parent_path == parent
        assert protocol == parent_protocol
        summary, values, _ = _fixture()
        return (
            prepared,
            summary,
            values,
            dict(version=version, case=case, origin=origin),
        )

    counts = []
    evaluate = audit.evaluate_residuals

    def counted(*args):
        counts.append(len(args[-1]))
        return evaluate(*args)

    monkeypatch.setattr(audit, "binding", bound)
    monkeypatch.setattr(audit, "verify_parent", verify_parent)
    monkeypatch.setattr(audit, "point_inputs", point_inputs)
    monkeypatch.setattr(audit, "evaluate_residuals", counted)
    monkeypatch.setattr(audit.parent_audit, "historical_module", lambda *args: module)
    output = tmp_path / "audit"
    audit.run(output, protocol_path)
    sealed = audit.digest(output / "manifest.json")
    assert counts == [4] * 46
    assert audit.read(output / "summary.json")["new_residual_calls"] == 184
    with monkeypatch.context() as readonly:
        readonly.setattr(audit, "evaluate_residuals", _forbidden)
        readonly.setattr(audit.parent_audit, "historical_module", _forbidden)
        verified = audit.verify(output, sealed)
    assert verified["verified"] and verified["snapshots"] == 46
    assert verified["residual_calls"] == 0
    recomputed = audit.verify(output, sealed, recompute=True)
    assert recomputed["verified"] and recomputed["residual_calls"] == 230
    assert counts == [4] * 46 + [5] * 46
    assert recomputed["arithmetic_checks"] == verified["arithmetic_checks"]
    with pytest.raises(FileExistsError):
        audit.run(output, protocol_path)
    summary = audit.read(output / "summary.json")
    summary["aggregate"]["all"]["count"] = 45
    (output / "summary.json").write_text(json.dumps(summary))
    manifest = audit.read(output / "manifest.json")
    manifest["files"]["summary.json"] = audit.digest(output / "summary.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    altered = audit.digest(output / "manifest.json")
    with pytest.raises(ValueError, match=r"aggregate|summary"):
        audit.verify(output, altered)
