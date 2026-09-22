"""Analytic checks for the frozen profiling harness; no recorded-model calls."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import profile_online_cost as profile
import pytest


def cache_fixture(recent_count):
    history, horizon = 2, 2
    values = dict(
        tail_states=np.arange(5 * 15, dtype=float).reshape(5, 15),
        tail_inputs=np.arange(4 * 2, dtype=float).reshape(4, 2),
    )
    shapes = ((history + 1, 15), (history, 2), (horizon, 2), (horizon, 15))
    for role, count in (("bootstrap", 3), ("recent", recent_count)):
        for name, shape in zip(profile.FIELDS, shapes):
            values[role + "_" + name] = np.arange(
                count * np.prod(shape), dtype=float
            ).reshape((count, *shape))
    meta = dict(cursor=19, horizon=horizon, model=dict(history_steps=history))
    context = dict(
        origin=np.asarray(19), truth=np.full(15, 999.0), command=np.array([87.0, 88.0])
    )
    return meta, values, context


@pytest.mark.parametrize("recent_count", [0, 2, 32])
def test_next_cache_uses_one_new_transition_and_retains_bounded_roles(recent_count):
    meta, values, context = cache_fixture(recent_count)
    saved = {key: value.copy() for key, value in values.items()}
    data, weights = profile.prepare_cache(meta, values, context)
    actual_count = min(32, recent_count + 1)
    assert weights.shape == (64,)
    assert weights[:32].sum() == 0.5
    assert weights[32:].sum() == 0.5
    assert np.count_nonzero(weights[:32]) == 3
    assert np.count_nonzero(weights[32:]) == actual_count
    latest = 32 + actual_count - 1
    assert np.array_equal(data[0][latest], values["tail_states"][1:4])
    assert np.array_equal(data[1][latest], values["tail_inputs"][1:3])
    assert np.array_equal(
        data[2][latest], np.vstack((values["tail_inputs"][-1], context["command"]))
    )
    assert np.array_equal(
        data[3][latest], np.vstack((values["tail_states"][-1], context["truth"]))
    )
    for key in saved:
        assert np.array_equal(saved[key], values[key])


def test_final_cache_is_only_retained_data_and_rejects_wrong_origin():
    meta, values, context = cache_fixture(32)
    data, weights = profile.prepare_cache(meta, values)
    for name, actual in zip(profile.FIELDS, data):
        assert np.array_equal(actual[32:], values["recent_" + name])
    assert np.array_equal(weights[32:], np.full(32, 0.5 / 32))
    context["origin"] += 1
    with pytest.raises(ValueError, match="cursor"):
        profile.prepare_cache(meta, values, context)


def test_frozen_roster_includes_six_endpoints_and_all_capture_origins(tmp_path):
    protocol = json.loads(profile.PROTOCOL.read_text())
    for case in protocol["points"]["case_order"]:
        root = tmp_path / case
        root.mkdir()
        (root / "case.json").write_text(
            json.dumps(dict(family="quad" if case.startswith("quad") else "fixedwing"))
        )
        for name, cursor in (("initial", 15), ("final", 240)):
            np.savez(
                root / (name + "-online.npz"), metadata=json.dumps(dict(cursor=cursor))
            )
    points = profile.roster(protocol, tmp_path)
    assert len(points) == 35
    assert sum(point["kind"] != "final" for point in points) == 29
    assert points[0]["id"] == "quad-arm-115/initial"
    assert points[1]["id"] == "quad-arm-115/final"
    assert [
        point["origin"]
        for point in points
        if point["case"] == "fixedwing-81" and point["kind"] == "capture"
    ] == protocol["points"]["capture_origins"]["fixedwing-81"]


def test_component_comparison_preserves_masks_and_exact_discrete_decisions():
    reference = {
        "value": np.array([1.0, np.nan, np.inf, -np.inf]),
        "flag": np.array(True),
        "selected_alpha": np.array(0.5),
    }
    candidate = copy.deepcopy(reference)
    candidate["value"][0] += 1e-9
    result = profile.compare(candidate, reference, rtol=2e-8, atol=2e-10)
    assert 0 < result["maximum_scaled_error"] < 1
    for key, changed in (
        ("value", np.array([1.0, np.inf, np.inf, -np.inf])),
        ("flag", np.array(False)),
        ("selected_alpha", np.array(0.5 + 1e-10)),
    ):
        wrong = dict(reference, **{key: changed})
        with pytest.raises(ValueError):
            profile.compare(wrong, reference, rtol=2e-8, atol=2e-10)
    with pytest.raises(ValueError, match="shape/dtype"):
        profile.compare(
            {"value": np.ones(1, dtype=np.float32)},
            {"value": np.ones(1)},
            rtol=2e-8,
            atol=2e-10,
        )
    with pytest.raises(ValueError, match="tolerance"):
        profile.compare(
            {"value": np.array([2.0])},
            {"value": np.array([1.0])},
            rtol=2e-8,
            atol=2e-10,
        )


def test_timed_call_excludes_input_synchronization_but_waits_for_output(monkeypatch):
    events = []
    values = iter((7.0, 11.0))
    monkeypatch.setattr(
        profile, "_synchronize", lambda value: events.append(("sync", value))
    )

    def clock():
        events.append(("clock",))
        return next(values)

    monkeypatch.setattr(profile.time, "perf_counter", clock)

    def function(value, *, increment):
        events.append(("call",))
        return value + increment

    output, elapsed = profile._timed(function, (3,), dict(increment=4))
    assert output == 7 and elapsed == 4
    assert [item[0] for item in events] == ["sync", "clock", "call", "sync", "clock"]


def test_disposable_replay_times_observe_and_snapshot_separately(monkeypatch):
    import jax

    import glassbox.online as online

    events, clocks = [], iter((10.0, 14.0, 15.0))
    model = SimpleNamespace(params={}, fingerprint="same-model")

    class Clone:
        @property
        def report(self):
            return {"observations": 1}

        def fingerprint(self):
            return "session"

        def observe(self, index, command, truth):
            events.append(("observe", index))

        @property
        def model(self):
            events.append(("snapshot",))
            return model

    monkeypatch.setattr(
        online.OnlineFit, "load", lambda path: events.append(("load",)) or Clone()
    )
    monkeypatch.setattr(profile.time, "perf_counter", lambda: next(clocks))
    monkeypatch.setattr(
        jax, "block_until_ready", lambda value: events.append(("sync",))
    )
    record, timings = profile._public_replay(
        Path("unused"),
        dict(origin=9, command=np.zeros(2), truth=np.zeros(15)),
        dict(model="same-model", report={"observations": 1}),
        0,
        "qualification",
    )
    assert timings == (4.0, 1.0, 5.0)
    assert record["matches_parent"]
    assert events == [("load",), ("observe", 9), ("snapshot",), ("sync",)]


def test_statistics_are_descriptive_and_groups_keep_fixedwing_kind():
    raw = np.arange(1, 22, dtype=float)
    values = profile.statistics(raw)
    assert values == dict(count=21, median_s=11.0, p95_s=20.0, max_s=21.0, min_s=1.0)
    with pytest.raises(ValueError):
        profile.statistics([1.0, np.nan])
    scopes = ("setup", "solve", "trust", "proposal", "instrumented", "public_combined")
    point = dict(
        id="fixedwing-80/initial",
        family="fixedwing",
        kind="initial",
        status="complete",
        statistics={key: values for key in scopes},
        native_observe_calls=25,
        component_dispatches=300,
    )
    result = profile.summary([point], dict(id="online-cost-profile-v1"))
    assert "fixedwing/initial" in result["groups"]
    assert "quad/initial" not in result["groups"]
    assert result["native_observe_calls"] == 25
    assert result["diagnostic_ratios"][point["id"]]["native/public"] == 1


def test_failed_attempt_is_exclusive_and_sealed(monkeypatch, tmp_path):
    from collect_throw import authenticate

    destination = tmp_path / "failed"

    def fail(*args):
        raise ValueError("deliberate missing authority")

    monkeypatch.setattr(profile, "authenticate", fail)
    result = profile.run(tmp_path / "missing-parent", destination)
    assert result["status"] == "failed"
    assert "missing authority" in result["error"]
    assert (destination / "attempt.json").exists()
    manifest_hash = profile.digest(destination / "manifest.json")
    assert len(authenticate(destination, manifest_hash)["files"]) >= 3
    with pytest.raises(FileExistsError):
        profile.run(tmp_path / "missing-parent", destination)


def test_failed_point_records_identity_even_when_session_cannot_load(tmp_path):
    protocol = json.loads(profile.PROTOCOL.read_text())
    point = dict(
        id="quad-arm-115/initial",
        case="quad-arm-115",
        family="quad",
        kind="initial",
        origin=125,
        session="missing.npz",
    )
    output = tmp_path / "point"
    with pytest.raises(FileNotFoundError):
        profile.profile_point(point, tmp_path, output, protocol, {}, set())
    details = json.loads((output / "point.json").read_text())
    assert details["status"] == "failed" and details["id"] == point["id"]
    assert details["native_observe_calls"] == 0
    assert details["component_dispatches"] == 0


def test_analytic_point_exercises_all_scopes_repetitions_and_saved_schema(
    monkeypatch, tmp_path
):
    from functools import partial

    import jax
    import jax.numpy as jnp
    from _online_cost_kernels import make_kernels

    from glassbox import online

    meta, saved, _ = cache_fixture(0)
    meta.update(fingerprint="analytic-session", damping=1.0)
    meta["model"].update(delay_steps=0, dt_s=0.1)
    saved.update(
        param_coefficient=np.asarray([0.5]),
        norm_unused=np.asarray([1.0]),
        scale=np.ones((2, 15)),
    )
    case = tmp_path / "analytic"
    case.mkdir()
    np.savez(case / "initial-online.npz", metadata=json.dumps(meta), **saved)
    (tmp_path / "inputs").mkdir()
    native = np.zeros((21, 13))
    native[:, 6] = 1.0
    np.savez(
        tmp_path / "inputs/analytic.npz", states=native, commands=np.zeros((20, 2))
    )
    before = dict(observations=0)
    after = dict(observations=1)
    events = [
        dict(phase="predicted", index=19, model="before-model", report=before),
        dict(phase="assimilated", index=19, model="after-model", report=after),
    ]
    (case / "events.jsonl").write_text("\n".join(map(json.dumps, events)))

    class Session:
        report = None

        @property
        def model(self):
            return SimpleNamespace(fingerprint="before-model")

        def fingerprint(self):
            return "analytic-session"

    session = Session()
    session.report = before
    monkeypatch.setattr(online.OnlineFit, "load", lambda path: session)
    public_calls = []

    def replay(source, context, expected, index, phase):
        assert not jax.config.x64_enabled
        public_calls.append(index)
        return dict(
            index=index,
            phase=phase,
            before_session_fingerprint="analytic-session",
            after_session_fingerprint="after-session",
            model="after-model",
            report=after,
            matches_parent=True,
        ), (0.002, 0.001, 0.003)

    monkeypatch.setattr(profile, "_public_replay", replay)

    def residual(params, norms, data, scale, delay, dt_s):
        return jnp.broadcast_to(params["coefficient"] ** 2 - 1, (64, 2, 15))

    def prior(params, norms, data, scale, weights, *, delay, dt_s):
        return jnp.full_like(params["coefficient"], 0.05)

    @partial(jax.jit, static_argnames=("delay", "dt_s"))
    def condition(params, norms, data, weights, *, delay, dt_s):
        return params, norms, jnp.asarray(True)

    monkeypatch.setattr(online, "_residual", residual)
    monkeypatch.setattr(online, "_curvature_diagonal", prior)
    monkeypatch.setattr(online, "_recondition", condition)
    native_proposal = partial(jax.jit, static_argnames=("delay", "dt_s"))(
        online._proposal.__wrapped__
    )
    monkeypatch.setattr(online, "_proposal", native_proposal)
    protocol = json.loads(profile.PROTOCOL.read_text())
    point = dict(
        id="analytic/initial",
        case="analytic",
        family="quad",
        kind="initial",
        origin=19,
        session="analytic/initial-online.npz",
    )
    destination = tmp_path / "result"
    details = profile.profile_point(
        point, tmp_path, destination, protocol, make_kernels(), set()
    )
    assert details["status"] == "complete"
    assert details["native_observe_calls"] == 25
    assert details["component_dispatches"] == 300
    assert public_calls == list(range(25))
    with np.load(destination / "timings.npz") as values:
        assert len(values.files) == 15
        assert all(values[name].shape == (25,) for name in values.files)
    with np.load(destination / "outputs.npz") as values:
        assert {
            "native/params/coefficient",
            "instrumented/native/params/coefficient",
            "setup/gradient",
            "instrumented/setup/gradient",
            "linearization/tape_hashes",
        } <= set(values.files)
    assert all(
        len(values) == 25 and len(set(values)) == 1
        for values in details["output_hashes"].values()
    )
    assert all(values["count"] == 21 for values in details["statistics"].values())

    # Exercise the separate no-model auditor against the driver's actual saved
    # schema, including Partial signatures and all per-call output identities.
    import verify_online_cost as audit

    saved_outputs = profile.arrays(destination / "outputs.npz")
    prepared = profile.arrays(destination / "prepared.npz")
    errors, _ = audit.verify_components(saved_outputs, prepared)
    audit.verify_qualification(details, errors, saved_outputs)
    audit.verify_signatures(prepared, saved_outputs, details, set())
    audit.verify_output_hashes(saved_outputs, details)
    assert (
        audit.verify_timings(
            profile.arrays(destination / "timings.npz"), details["statistics"], True
        )
        == 375
    )
