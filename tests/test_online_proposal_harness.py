"""Synthetic proposal orchestration checks, without fitting or applying updates."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import audit_online as audit


def fixture(*, origin=8, horizon=1):
    count, first, history = 80, 6, 2
    states = np.zeros((count + 1, 15))
    states[:, :6] = np.arange(count + 1)[:, None] * np.arange(1, 7)[None, :] / 100
    states[:, 6:] = np.eye(3).ravel()
    inputs = np.arange(count * 2, dtype=float).reshape(count, 2) / 10
    params = dict(
        bias=np.arange(6.0) / 10,
        linear=np.ones((12, 6)),
        quadratic=np.ones((3, 6)),
        w1=np.ones((12, 2)),
        b1=np.ones(2),
        w2=np.ones((2, 6)),
        memory=np.ones((12, 8)),
        memory_bias=np.zeros(8),
        raw_tau=np.zeros(2),
    )
    norms = dict(
        feature_scale=np.ones(12),
        quadratic_scale=np.ones(3),
        output_scale=np.ones(6),
        body_scale=np.ones(9),
        input_scale=np.ones(2),
    )
    bootstrap = audit.windows(
        states,
        inputs,
        range(first - horizon - 1, first - horizon + 1),
        history,
        horizon,
    )
    n = min(32, origin - first)
    recent = (
        audit.windows(
            states,
            inputs,
            range(origin - horizon + 1 - n, origin - horizon + 1),
            history,
            horizon,
        )
        if n
        else {name: value[:0] for name, value in bootstrap.items()}
    )
    meta = dict(
        format="glassbox-online-fit-v6",
        model=dict(format="synthetic", history_steps=history, delay_steps=1, dt_s=0.05),
        recipe=dict(proposals=1),
        contract=dict(input_channels=["a", "b"]),
        initial_cursor=first,
        initial_count=2,
        cursor=origin,
        horizon=horizon,
        damping=1.0,
        counts=dict(
            conditioning_calls=origin - first,
            optimizer_steps=origin - first,
            gradient_calls=origin - first,
            objective_calls=2 * (origin - first),
            cg_iterations=4 * (origin - first),
            curvature_calls=4 * (origin - first),
            accepted_proposals=0,
        ),
    )
    values = dict(
        **{"param_" + name: value for name, value in params.items()},
        **{"norm_" + name: value for name, value in norms.items()},
        scale=np.ones((horizon, 15)),
        tail_states=states[origin - history - horizon : origin + 1],
        tail_inputs=inputs[origin - history - horizon : origin],
        **{"bootstrap_" + name: value for name, value in bootstrap.items()},
        **{"recent_" + name: value for name, value in recent.items()},
    )
    context = dict(
        origin=np.asarray(origin, dtype=np.int64),
        past_states=states[origin - history : origin + 1],
        past_inputs=inputs[origin - history : origin],
        command=inputs[origin],
        truth=states[origin + 1],
        recorded_prediction=states[origin].astype(np.float32),
    )
    return meta, values, context, states, inputs


def prepared(meta, values, context):
    return dict(
        audit.post_cache(meta, values, context),
        params={
            name[6:]: value
            for name, value in values.items()
            if name.startswith("param_")
        },
        norms={
            name[5:]: value
            for name, value in values.items()
            if name.startswith("norm_")
        },
        scale=values["scale"],
        damping=np.asarray(meta["damping"]),
        conditioning_finite=True,
    )


@pytest.mark.parametrize("origin,horizon", [(6, 1), (8, 3), (45, 1), (45, 3)])
def test_post_reveal_cache_growth_eviction_and_multistep_origin(origin, horizon):
    meta, values, context, states, inputs = fixture(origin=origin, horizon=horizon)
    before = {key: value.tobytes() for key, value in values.items()}
    result = audit.post_cache(meta, values, context)
    count = min(32, origin - meta["initial_cursor"] + 1)
    starts = range(origin - horizon + 2 - count, origin - horizon + 2)
    expected = audit.windows(
        states, inputs, starts, meta["model"]["history_steps"], horizon
    )
    assert len(result["recent"]["past_states"]) == count
    for name in audit.FIELDS:
        np.testing.assert_array_equal(result["recent"][name], expected[name])
        np.testing.assert_array_equal(
            result["data"][audit.FIELDS.index(name)][32 : 32 + count], expected[name]
        )
    np.testing.assert_array_equal(
        result["recent"]["future_states"][-1, -1], context["truth"]
    )
    np.testing.assert_array_equal(
        result["recent"]["past_states"][-1, -1], states[origin - horizon + 1]
    )
    assert result["weights"].shape == (64,)
    assert np.sum(result["weights"][:32]) == pytest.approx(0.5)
    assert np.sum(result["weights"][32:]) == pytest.approx(0.5)
    assert np.count_nonzero(result["weights"]) == 2 + count
    assert before == {key: value.tobytes() for key, value in values.items()}


@pytest.mark.parametrize(
    "gain,accepted,factor",
    [
        (0.099, False, 4),
        (0.1, True, 4),
        (0.249, True, 4),
        (0.25, True, 1),
        (0.75, True, 1),
        (0.751, True, 0.5),
        (-1, False, 4),
    ],
)
def test_exact_acceptance_thresholds_and_damping(gain, accepted, factor):
    meta, values, context, *_ = fixture()
    p = prepared(meta, values, context)
    # Exact binary-valued reduction avoids rounding a mathematical threshold.
    production = (dict(p["params"], bias=np.ones(6)), gain, 0.0, 1.0, True)
    after, arrays, result = audit.after_update(meta, values, p, production)
    assert result["accepted"] is accepted
    assert after["damping"] == factor
    assert after["cursor"] == meta["cursor"] + 1
    assert after["counts"]["accepted_proposals"] == int(accepted)
    assert after["counts"]["cg_iterations"] == meta["counts"]["cg_iterations"] + 4
    np.testing.assert_array_equal(
        arrays["param_bias"],
        production[0]["bias"] if accepted else values["param_bias"],
    )
    np.testing.assert_array_equal(arrays["tail_states"][-1], context["truth"])


@pytest.mark.parametrize(
    "problem",
    ["conditioning", "proposal", "objective", "no_decrease", "clip_low", "clip_high"],
)
def test_rejection_restores_original_coordinates_but_consumes_valid_data(problem):
    meta, values, context, *_ = fixture()
    p = prepared(meta, values, context)
    p["norms"] = copy.deepcopy(p["norms"])
    p["norms"]["output_scale"] *= 5
    production = (dict(p["params"], bias=np.ones(6) * 9), 2.0, 1.0, 1.0, True)
    if problem == "conditioning":
        p["conditioning_finite"] = False
    if problem == "proposal":
        production = (*production[:-1], False)
    if problem == "objective":
        production = (production[0], 2.0, np.nan, 1.0, True)
    if problem == "no_decrease":
        production = (production[0], 1.0, 1.0, 1.0, True)
    if problem == "clip_low":
        meta["damping"] = 1e-8
    if problem == "clip_high":
        meta["damping"] = 1e8
        production = (production[0], 1.0, 2.0, 1.0, True)
    after, arrays, result = audit.after_update(meta, values, p, production)
    if problem == "clip_low":
        assert result["accepted"] and after["damping"] == 1e-8
    else:
        assert not result["accepted"]
        for name in values:
            if name.startswith(("param_", "norm_")):
                audit.paired_exact(arrays[name], values[name], name)
    if problem == "clip_high":
        assert after["damping"] == 1e8
    assert after["cursor"] == meta["cursor"] + 1
    assert len(arrays["recent_past_states"]) == len(values["recent_past_states"]) + 1


def write_original(root, origin=8):
    meta, values, context, states, inputs = fixture(origin=origin)
    initial_meta, initial, *_ = fixture(origin=6)
    version, case = "v6", "fixedwing-80"
    evaluation = root / f"evidence/{version}/evaluation"
    point = root / f"evidence/{version}/captures/{case}/{origin}"
    point.mkdir(parents=True)
    (evaluation / case).mkdir(parents=True)
    (evaluation / "inputs").mkdir()
    save_arrays(point / "session.npz", meta, values)
    save_arrays(evaluation / case / "initial-online.npz", initial_meta, initial)
    audit.checkpoint(point / "context.npz", **context)
    p = prepared(meta, values, context)
    production = (p["params"], 1.0, 0.5, 1.0, True)
    after_meta, after_values, outcome = audit.after_update(meta, values, p, production)
    report = dict(
        meta["counts"],
        cursor=origin,
        observations=origin - 6,
        recent_windows=origin - 6,
        tail_rows=len(values["tail_states"]),
        damping=meta["damping"],
    )
    following = dict(
        report,
        **after_meta["counts"],
        cursor=origin + 1,
        observations=origin - 5,
        recent_windows=origin - 5,
        damping=after_meta["damping"],
    )
    core = array_fingerprint(
        meta["model"],
        {
            name: value
            for name, value in values.items()
            if name.startswith(("param_", "norm_"))
        },
    )
    audit.write(
        point / "capture.json",
        dict(
            origin=origin,
            time_s=float(origin * 0.05),
            session_fingerprint=array_fingerprint(meta, values),
            model_fingerprint=core,
            report=report,
        ),
    )
    data = dict(
        origin=np.array([origin]),
        candidate=context["recorded_prediction"][None],
        truth=context["truth"][None],
        model_before=np.array([core]),
        model_after=np.array([outcome["core_after"]]),
        **{key: np.array([True]) for key in ("predicted", "revealed", "assimilated")},
    )
    for name in audit.COUNTERS:
        data[name + "_before"] = np.array([report[name]])
        data[name + "_after"] = np.array([following[name]])
    audit.checkpoint(evaluation / case / "predictions.npz", **data)
    audit.write(evaluation / case / "case.json", dict(first=6))
    native = np.zeros((len(states), 13))
    native[:, 6] = 1
    native[:, 3:6] = states[:, :3]
    native[:, 10:13] = states[:, 3:6]
    audit.checkpoint(
        evaluation / f"inputs/{case}.npz",
        states=native,
        commands=inputs,
        time_s=np.arange(len(states)) * 0.05,
    )
    (evaluation / case / "events.jsonl").write_text(
        "\n".join(
            json.dumps(
                dict(
                    index=origin,
                    phase=phase,
                    report=value,
                    model=core if phase == "predicted" else outcome["core_after"],
                )
            )
            for phase, value in (("predicted", report), ("assimilated", following))
        )
        + "\n"
    )
    return (
        dict(origins={case: [origin]}),
        dict(
            prepared=p,
            production=production,
            after_metadata=after_meta,
            after_arrays=after_values,
            summary=outcome,
        ),
        point,
    )


def test_original_and_after_identity_use_saved_arrays_without_session_loader(
    tmp_path, monkeypatch
):
    from glassbox import OnlineFit

    protocol, reconstruction, _point = write_original(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("archive check called a model")

    monkeypatch.setattr(OnlineFit, "load", forbidden)
    monkeypatch.setattr(OnlineFit, "__init__", forbidden)
    monkeypatch.setattr(OnlineFit, "observe", forbidden)
    original = audit.original_point(tmp_path, protocol, "v6", "fixedwing-80", 8)
    assert not audit.check_after(
        original, reconstruction, tmp_path, protocol, "v6", "fixedwing-80", 8
    )
    protocol["origins"]["fixedwing-80"].append(9)
    path = tmp_path / "evidence/v6/captures/fixedwing-80/9"
    path.mkdir()
    audit.write(
        path / "capture.json",
        dict(session_fingerprint=reconstruction["summary"]["session_after"]),
    )
    assert audit.check_after(
        original, reconstruction, tmp_path, protocol, "v6", "fixedwing-80", 8
    )


@pytest.mark.parametrize(
    "problem",
    [
        "cursor",
        "format",
        "recent",
        "command",
        "recorded_prediction",
        "after_counter",
        "before_model",
        "after_model",
    ],
)
def test_original_identity_rejects_resigned_causal_and_context_tampering(
    tmp_path, problem
):
    protocol, _reconstruction, path = write_original(tmp_path)
    if problem in ("cursor", "format", "recent"):
        meta, values = load_arrays(path / "session.npz")
        if problem == "cursor":
            meta["cursor"] += 1
        elif problem == "format":
            meta["format"] = "glassbox-online-fit-v7"
        else:
            values["recent_future_states"][0, 0, 0] += 1
        save_arrays(path / "session.npz", meta, values)
    elif problem in ("before_model", "after_model"):
        journal = tmp_path / "evidence/v6/evaluation/fixedwing-80/events.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        rows[int(problem == "after_model")]["model"] = "0" * 64
        journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    elif problem == "after_counter":
        p = tmp_path / "evidence/v6/evaluation/fixedwing-80/predictions.npz"
        values = audit.arrays(p)
        values["gradient_calls_after"][0] += 1
        audit.checkpoint(p, **values)
    else:
        values = audit.arrays(path / "context.npz")
        values[problem].flat[0] += 1
        audit.checkpoint(path / "context.npz", **values)
    with pytest.raises(ValueError):
        audit.original_point(tmp_path, protocol, "v6", "fixedwing-80", 8)


def test_private_source_alias_authenticates_bytes_and_preserves_format(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "check_package_sources", lambda *args: None)
    source = tmp_path / "sources/v7-online.py"
    source.parent.mkdir()
    source.write_text('_FORMAT="glassbox-online-fit-v7"\n')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    evaluation = tmp_path / "evidence/v7/evaluation"
    evaluation.mkdir(parents=True)
    audit.write(
        evaluation / "binding.json",
        dict(source=dict(files={"src/glassbox/online.py": sha})),
    )
    protocol = dict(references=dict(v7=dict(online_source_sha256=sha)))
    module = audit.historical_module(tmp_path, protocol, "v7")
    assert module.__name__ == "glassbox._proposal_audit_v7"
    assert module._FORMAT == "glassbox-online-fit-v7"
    source.write_text('raise AssertionError("unauthenticated code executed")\n')
    with pytest.raises(ValueError, match="source differs"):
        audit.historical_module(tmp_path, protocol, "v7")


def test_prepared_and_production_binding_rejects_disconnected_diagnostic_data():
    meta, values, context, *_ = fixture()
    p = prepared(meta, values, context)
    production = (
        p["params"],
        np.array(1.0),
        np.array(0.5),
        np.array(1.0),
        np.array(True),
    )
    theta = np.concatenate([p["params"][name].ravel() for name in sorted(p["params"])])
    schema = []
    start = 0
    for name in sorted(p["params"]):
        value = p["params"][name]
        stop = start + value.size
        schema.append(
            dict(path=f"[{name!r}]", shape=list(value.shape), start=start, stop=stop)
        )
        start = stop
    arrays = dict(
        theta=theta,
        production_theta=theta.copy(),
        damping=p["damping"],
        raw=np.zeros_like(p["data"][3]),
        weight=p["weights"][:, None, None] / 3,
        **{
            "production_" + key: value
            for key, value in zip(
                ("current", "trial", "predicted", "finite"), production[1:]
            )
        },
    )
    summary = dict(parameter_schema=schema)
    audit.check_diagnostic_binding(p, production, summary, arrays)
    arrays["theta"][0] += 1
    with pytest.raises(ValueError, match="conditioned parameters"):
        audit.check_diagnostic_binding(p, production, summary, arrays)


def test_failed_attempt_is_sealed_without_touching_inputs(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(dict(id="online-proposal-audit-v1", references={})))

    def fail(*args):
        raise ValueError("synthetic source rejection")

    monkeypatch.setattr(audit, "binding", fail)
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="source rejection"):
        audit.run(output, protocol)
    assert (output / "failure.json").exists() and (output / "manifest.json").exists()
    assert json.loads(protocol.read_text())["references"] == {}


@pytest.mark.parametrize("recompute", [False, True])
def test_verification_rejects_changed_scientific_source_before_any_actions(
    tmp_path, monkeypatch, recompute
):
    output = tmp_path / "audit"
    output.mkdir()
    source = tmp_path / "helper.py"
    source.write_text("original")
    audit.write(output / "protocol.json", dict(id="online-proposal-audit-v1"))
    audit.write(
        output / "binding.json",
        dict(
            protocol_sha256=audit.digest(output / "protocol.json"),
            source=dict(files={str(source): audit.digest(source)}),
            runtime={},
        ),
    )
    source.write_text("changed")
    monkeypatch.setattr(audit, "authenticate", lambda *args: None)
    monkeypatch.setattr(audit, "verify_evidence", lambda *args: None)

    def forbidden(*args, **kwargs):
        pytest.fail("changed scientific source reached numerical verification")

    monkeypatch.setattr(audit, "verify_points", forbidden)
    monkeypatch.setattr(audit, "binding", forbidden)
    with pytest.raises(ValueError, match="source changed"):
        audit.verify(output, "fixture", recompute=recompute)
