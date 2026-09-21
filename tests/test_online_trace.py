"""Independent causal-trace bookkeeping checks; no recording fit or simulator."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import trace_online as trace


def selected_fixture(*, angular=False):
    origins = np.arange(15, 240, dtype=np.int64)
    velocity = (np.arange(225) % 19).astype(float) / 20
    rate = (np.arange(225) % 17).astype(float) / 18
    velocity[np.isin(origins, [70, 80])] = 5  # Anchor ties prefer source row 70.
    rate[np.isin(origins, [180, 185])] = 4
    truth = np.zeros((225, 15))
    truth[:, 6:] = np.eye(3).ravel()
    prediction = truth.copy()
    prediction[:, 0], prediction[:, 3] = velocity, rate
    data = dict(
        origin=origins,
        candidate=prediction,
        truth=truth,
        candidate_residual=prediction - truth,
        time_s=origins * 0.05,
    )
    measures = dict(velocity=velocity, rate=rate)
    if angular:
        angles = (np.arange(225) % 13).astype(float) / 40
        angles[np.isin(origins, [180, 185])] = 0.7
        cosine, sine = np.cos(angles), np.sin(angles)
        rotation = np.zeros((225, 3, 3))
        rotation[:, 0, 0] = rotation[:, 1, 1] = cosine
        rotation[:, 1, 0], rotation[:, 0, 1] = sine, -sine
        rotation[:, 2, 2] = 1
        # Nonorthogonal matrices require proper-rotation projection, not acos(trace).
        prediction[:, 6:] = (1.1 * rotation).reshape(225, 9)
        measures["orientation"] = angles
    # Deliberately literal average ranks, independent of a library rank function.
    ranks = [
        np.array([sum(values < x) + (sum(values == x) + 1) / 2 for x in values])
        for values in measures.values()
    ]
    scores = sum(ranks)
    center = np.median(scores)
    anchors = {
        name: int(min(origins[values == max(values)]))
        for name, values in measures.items()
    }
    controls = {}
    for name, anchor in anchors.items():
        eligible = [
            int(row)
            for row in origins
            if abs(row - anchor) <= 10
            and all(abs(row - other) > 1 for other in anchors.values())
        ]
        controls[name] = min(
            eligible,
            key=lambda row: (abs(scores[row - 15] - center), abs(row - anchor), row),
        )
    captures = sorted(
        {row + delta for row in anchors.values() for delta in (-1, 0, 1)}
        | {row + delta for row in controls.values() for delta in (-1, 0)}
    )
    return data, dict(
        id="fixedwing-80", anchors=anchors, controls=controls, captures=captures
    )


@pytest.mark.parametrize("angular", [False, True])
def test_selection_uses_average_ranks_ties_and_frozen_rows_without_mutating_data(
    angular,
):
    data, case = selected_fixture(angular=angular)
    initial = copy.deepcopy(data)
    assert case["anchors"] == dict(
        velocity=70, rate=180, **({"orientation": 180} if angular else {})
    )
    assert trace.selected_origins(data, case) == case["captures"]
    assert trace.selected_origins(data, case) == case["captures"]
    for name in data:
        np.testing.assert_array_equal(data[name], initial[name])


@pytest.mark.parametrize("field", ["anchors", "controls", "captures"])
def test_selection_rejects_altered_frozen_declaration(field):
    data, case = selected_fixture()
    if field == "captures":
        case[field] = case[field][1:]
    else:
        case[field]["velocity"] += 1
    with pytest.raises(ValueError):
        trace.selected_origins(data, case)


@pytest.mark.parametrize("duplicate", [False, True])
def test_three_metric_controls_exclude_every_anchor_then_break_distance_and_row_ties(
    duplicate,
):
    from scipy.spatial.transform import Rotation

    origins = np.arange(100, 131)
    truth = np.zeros((31, 15))
    truth[:, 6:] = np.eye(3).ravel()
    prediction = truth.copy()
    rate_anchor = 115 if duplicate else 118
    prediction[15, 0] = 5
    prediction[rate_anchor - 100, 3] = 4
    prediction[rate_anchor - 100, 6:] = (
        Rotation.from_rotvec([0, 0, 0.4]).as_matrix().ravel()
    )
    data = dict(origin=origins, candidate=prediction, truth=truth)
    controls = dict(
        velocity=113,
        rate=113 if duplicate else 120,
        orientation=113 if duplicate else 120,
    )
    captures = list(range(112, 117 if duplicate else 121))
    case = dict(
        anchors=dict(velocity=115, rate=rate_anchor, orientation=rate_anchor),
        controls=controls,
        captures=captures,
    )
    assert trace.selected_origins(data, case) == captures
    assert len(captures) == len(set(captures))


@pytest.mark.parametrize("unknown", ["position", "angular_acceleration"])
def test_selection_rejects_unknown_anchor_keys(unknown):
    data, case = selected_fixture(angular=True)
    case["anchors"][unknown] = case["anchors"].pop("orientation")
    with pytest.raises(ValueError, match="anchor metrics"):
        trace.selected_origins(data, case)


def test_protocol_requires_all_three_v2_metrics_and_exact_reference_inventory(tmp_path):
    _data, case = selected_fixture(angular=True)
    protocol = dict(
        id="online-causal-trace-v2",
        cases=[case],
        reference=dict(
            protocol_id="online-fit-v6",
            source_commit="scientific",
            manifest_sha256="sealed",
        ),
    )
    bound = dict(
        source=dict(
            commit="diagnostic",
            files={"src/glassbox/online.py": "hash", "scripts/trace_online.py": "new"},
        ),
        runtime={"x64_enabled": False},
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    original = copy.deepcopy(bound)
    original["source"]["commit"] = "scientific"
    original["source"]["files"]["scripts/trace_online.py"] = "old"
    trace.write(reference / "protocol.json", dict(id="online-fit-v6"))
    trace.write(reference / "binding.json", original)
    trace.validate_protocol(protocol)
    trace.validate_reference(protocol, bound, reference)
    for changed in ("removed", "changed", "extra"):
        altered = copy.deepcopy(bound)
        files = altered["source"]["files"]
        if changed == "removed":
            del files["src/glassbox/online.py"]
        elif changed == "changed":
            files["src/glassbox/online.py"] = "different"
        else:
            files["src/glassbox/extra.py"] = "extra"
        with pytest.raises(ValueError, match="source inventory"):
            trace.validate_reference(protocol, altered, reference)
    protocol["cases"][0]["anchors"].pop("orientation")
    with pytest.raises(ValueError, match="anchor metrics"):
        trace.validate_protocol(protocol)
    assert trace.PROTOCOL.name == "online-causal-trace-v2.json"


def analytic_model(dt, *, varying_heads):
    """Construct a bounded model directly; no initializer or fitting is called."""
    from glassbox._dynamics import VehicleSequenceModel

    rng = np.random.default_rng(22)
    commands, current, memory, delay, width = 3, 15, 8, 2, 32
    feature, quadratic = (delay + 1) * current + memory, current * (current + 1) // 2
    shapes = dict(
        linear=(feature, 6),
        quadratic=(quadratic, 6),
        bias=(6,),
        w1=(feature, width),
        b1=(width,),
        w2=(width, 6),
        memory=(feature, memory),
        memory_bias=(memory,),
        raw_tau=(commands,),
    )
    params = {
        name: rng.normal(0, 0.003, shape) if varying_heads else np.zeros(shape)
        for name, shape in shapes.items()
    }
    params["bias"][2] += 9.80665
    params["bias"][5] += 0.6
    params["raw_tau"][:] = [-3, -2.7, -3.3]
    norms = dict(
        body_mean=np.zeros(9),
        body_scale=np.ones(9),
        motion_bound_scale=np.full(6, 8.0),
        input_mean=np.zeros(commands),
        input_scale=np.ones(commands),
        feature_scale=np.ones(feature),
        quadratic_scale=np.ones(quadratic),
        output_scale=np.ones(6),
        state_mean=np.zeros(15),
        state_scale=np.ones(15),
    )
    return VehicleSequenceModel(dt, 10, delay, params, norms)


@pytest.mark.parametrize("dt", [0.01, 0.05])
def test_instrumented_native_stages_match_unchanged_core_with_active_memory(dt):
    import _online_trace_diagnostics as diagnostic
    import jax
    from scipy.spatial.transform import Rotation

    model = analytic_model(dt, varying_heads=True)
    past = np.zeros((11, 15))
    past[:, :3] = np.arange(11)[:, None] * [0.01, -0.02, 0.003]
    past[:, 3:6] = [0.1, -0.15, 0.3]
    past[:, 6:] = (
        Rotation.from_rotvec(np.arange(11)[:, None] * [0.005, -0.003, 0.004])
        .as_matrix()
        .reshape(11, 9)
    )
    inputs = np.sin(np.arange(30).reshape(10, 3)) * 0.3
    command = np.array([0.3, -0.2, 0.1])
    before = jax.config.x64_enabled
    stages = diagnostic.integrate_stages(model, past, inputs, command, factor=1)
    assert jax.config.x64_enabled == before
    with jax.enable_x64(True):
        expected = np.asarray(model.rollout(past, inputs, command[None]))[0]
    np.testing.assert_allclose(stages["prediction"], expected, rtol=1e-12, atol=1e-12)
    assert stages["states"].shape == (int(np.ceil(dt / 0.025)), 3, 15)
    np.testing.assert_array_equal(stages["states"][0, 0], past[-1])
    np.testing.assert_array_equal(stages["states"][-1, 2], stages["prediction"])
    if len(stages["states"]) > 1:
        np.testing.assert_array_equal(stages["states"][1:, 0], stages["states"][:-1, 2])
        np.testing.assert_array_equal(
            stages["filtered"][1:, 0], stages["filtered"][:-1, 2]
        )


@pytest.mark.parametrize("factor", [1, 2, 4, 16, 64])
def test_stage_refinement_preserves_exact_constant_angular_acceleration(factor):
    import _online_trace_diagnostics as diagnostic
    from scipy.spatial.transform import Rotation

    model = analytic_model(0.05, varying_heads=False)
    past = np.zeros((11, 15))
    past[:, :3] = [0.2, -0.1, 0.3]
    past[:, 5] = 0.4
    past[:, 6:] = np.eye(3).ravel()
    inputs = np.zeros((10, 3))
    command = np.array([0.3, -0.2, 0.1])
    stages = diagnostic.integrate_stages(model, past, inputs, command, factor=factor)
    expected = past[-1].copy()
    expected[5] += 0.6 * model.dt_s
    expected[6:] = (
        Rotation.from_rotvec([0, 0, 0.4 * model.dt_s + 0.5 * 0.6 * model.dt_s**2])
        .as_matrix()
        .ravel()
    )
    np.testing.assert_allclose(stages["prediction"], expected, rtol=2e-13, atol=2e-13)
    assert stages["states"].shape == (2 * factor, 3, 15)
    tau = 0.001 + np.logaddexp(0, model.params["raw_tau"])
    expected_filter = command * (1 - np.exp(-model.dt_s / tau))
    np.testing.assert_allclose(
        stages["filtered"][-1, 2], expected_filter, rtol=2e-13, atol=2e-13
    )


def replay_fixture(tmp_path, monkeypatch, *, angular=False):
    data, case = selected_fixture(angular=angular)
    data["candidate"] = data["candidate"].astype(np.float32)
    source_states = np.zeros((241, 15))
    source_states[:, 6:] = np.eye(3).ravel()
    commands = np.column_stack(
        (np.arange(240), np.arange(240) * 0.01, np.zeros(240))
    ).astype(float)
    events = []

    class Session:
        def __init__(self, cursor):
            self.cursor = cursor
            self._states = source_states[cursor - 11 : cursor + 1].copy()
            self._inputs = commands[cursor - 11 : cursor].copy()
            self.corrupt = 0
            self._contract = {"identity": "synthetic"}
            self._initial_cursor, self._initial_count, self._horizon = 15, 5, 1
            self._scale = np.ones((1, 15))

            def windows(origins):
                return dict(
                    past_states=np.stack(
                        [source_states[k - 10 : k + 1] for k in origins]
                    ),
                    past_inputs=np.stack([commands[k - 10 : k] for k in origins]),
                    future_inputs=np.stack([commands[k : k + 1] for k in origins]),
                    future_states=np.stack(
                        [source_states[k + 1 : k + 2] for k in origins]
                    ),
                )

            self._bootstrap = windows(range(10, 15))
            self._recent = (
                windows(range(max(15, cursor - 32), cursor))
                if cursor > 15
                else {key: value[:0] for key, value in self._bootstrap.items()}
            )

        @property
        def model(self):
            # Half of observations leave the core unchanged, like rejected proposals.
            core = (self.cursor - 15) // 2
            return SimpleNamespace(
                history_steps=10,
                fingerprint=hashlib.sha256(str(core).encode()).hexdigest(),
            )

        @property
        def report(self):
            n = self.cursor - 15
            return dict(
                observations=n,
                optimizer_steps=n,
                gradient_calls=n,
                objective_calls=2 * n,
                accepted_proposals=n // 2,
                cg_iterations=4 * n,
                curvature_calls=4 * n,
                conditioning_calls=n,
                damping=1.0 if n % 2 == 0 else 4.0,
            )

        def fingerprint(self):
            return hashlib.sha256(
                str((self.cursor, self.corrupt)).encode()
                + self._states.tobytes()
                + self._inputs.tobytes()
            ).hexdigest()

        def predict(self, past, inputs, future):
            row = int(future[0, 0])
            events.append(("predict", self.cursor, row))
            np.testing.assert_array_equal(past, source_states[row - 10 : row + 1])
            np.testing.assert_array_equal(inputs, commands[row - 10 : row])
            prediction = data["candidate"][row - 15 : row - 14].copy()
            prediction[:, 0] += np.float32(
                ((self.cursor - 15) // 2 - (row - 15) // 2) * 0.0001
            )
            return prediction

        def observe(self, index, command, truth):
            assert index == self.cursor
            assert events[-1] == ("predict", self.cursor, self.cursor)
            events.append(("observe", index))
            self._states = np.concatenate((self._states[1:], truth[None]))
            self._inputs = np.concatenate((self._inputs[1:], command[None]))
            self.cursor += 1
            self._recent = Session(self.cursor)._recent

        def save(self, path):
            events.append(("save", self.cursor))
            np.savez(path, cursor=np.asarray(self.cursor))

        @classmethod
        def load(cls, path):
            with np.load(path) as saved:
                return cls(int(saved["cursor"]))

    before, after = (
        [Session(int(row)) for row in data["origin"]],
        [Session(int(row) + 1) for row in data["origin"]],
    )
    data["model_before"] = np.array([s.model.fingerprint for s in before])
    data["model_after"] = np.array([s.model.fingerprint for s in after])
    for key in trace.COUNTERS:
        data[key + "_before"] = np.array(
            [s.report[key] for s in before], dtype=np.int64
        )
        data[key + "_after"] = np.array([s.report[key] for s in after], dtype=np.int64)
    reference = tmp_path / "reference"
    original = reference / case["id"]
    original.mkdir(parents=True)
    (reference / "inputs").mkdir()
    np.savez(original / "predictions.npz", **data)
    journal = []
    for row, previous, following in zip(data["origin"], before, after):
        journal.extend(
            (
                dict(index=int(row), phase="predicted", report=previous.report),
                dict(index=int(row), phase="assimilated", report=following.report),
            )
        )
    (original / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in journal) + "\n"
    )
    np.savez(
        reference / "inputs" / (case["id"] + ".npz"),
        states=source_states,
        commands=commands,
        time_s=np.arange(241) * 0.05,
    )
    info = dict(first=15)
    (original / "case.json").write_text(json.dumps(info))
    for name, cursor in (("initial", 15), ("final", 240)):
        np.savez(original / (name + "-online.npz"), cursor=np.asarray(cursor))
    monkeypatch.setattr(trace, "OnlineFit", Session)
    monkeypatch.setattr(trace, "observed", lambda x: np.asarray(x).copy())
    return reference, case, data, info, Session, events


@pytest.mark.parametrize("angular", [False, True])
def test_replay_captures_before_prediction_and_completes_all_225_updates(
    tmp_path, monkeypatch, angular
):
    reference, case, data, info, Session, events = replay_fixture(
        tmp_path, monkeypatch, angular=angular
    )
    output = tmp_path / "case"
    result = trace.replay_case(reference, output, case)
    assert result["transitions"] == 225 and result["captures"] == case["captures"]
    assert [e[1] for e in events if e[0] == "observe"] == list(range(15, 240))
    assert result["final_session_fingerprint"] == Session(240).fingerprint()
    for row in case["captures"]:
        at = events.index(("save", row))
        assert events[at : at + 3] == [
            ("save", row),
            ("predict", row, row),
            ("observe", row),
        ]
        session = Session.load(output / str(row) / "session.npz")
        context = trace.arrays(output / str(row) / "context.npz")
        trace.validate_capture(session, context, data, info)
        assert session.cursor == row


def test_capture_rejects_post_update_even_when_core_fingerprint_is_unchanged(
    tmp_path, monkeypatch
):
    reference, case, data, info, Session, _events = replay_fixture(
        tmp_path, monkeypatch
    )
    output = tmp_path / "case"
    trace.replay_case(reference, output, case)
    row = next(row for row in case["captures"] if (row - 15) % 2 == 0)
    context = trace.arrays(output / str(row) / "context.npz")
    assert Session(row).model.fingerprint == Session(row + 1).model.fingerprint
    with pytest.raises(ValueError, match="before assimilation"):
        trace.validate_capture(Session(row + 1), context, data, info)
    before = Session(row)
    before._inputs[-1, 0] += 1
    with pytest.raises(ValueError, match="causal command tail"):
        trace.validate_capture(before, context, data, info)


@pytest.mark.parametrize("problem", ["bytes", "mutation"])
def test_replay_stops_before_assimilation_when_prediction_proof_fails(
    tmp_path, monkeypatch, problem
):
    reference, case, _data, _info, Session, events = replay_fixture(
        tmp_path, monkeypatch
    )
    actual = Session.predict

    def corrupt(self, *args):
        prediction = actual(self, *args)
        if problem == "bytes":
            prediction[0, 1] = -0.0  # Numerically equal, but not byte-identical.
        else:
            self.corrupt += 1
        return prediction

    monkeypatch.setattr(Session, "predict", corrupt)
    with pytest.raises(
        ValueError, match="bytes differ" if problem == "bytes" else "mutated session"
    ):
        trace.replay_case(reference, tmp_path / "case", case)
    assert not any(e[0] == "observe" for e in events)
    assert not (tmp_path / "case/replay.json").exists()


@pytest.mark.parametrize("angular", [False, True])
@pytest.mark.parametrize("tampering", ["diagnostic", "stage"])
def test_verify_recomputes_saved_diagnostics_without_assimilation(
    tmp_path, monkeypatch, tampering, angular
):
    import evaluate_online

    output = tmp_path / "trace"
    output.mkdir()
    reference, case, data, _info, Session, _events = replay_fixture(
        output, monkeypatch, angular=angular
    )
    result = trace.replay_case(reference, output / case["id"], case)
    protocol = dict(
        id="online-causal-trace-v2" if angular else "online-causal-trace-v1",
        reference=dict(
            manifest_sha256="fixture",
            protocol_id="online-fit-v6" if angular else "online-fit-v4",
            source_commit="scientific",
        ),
        cases=[case],
    )
    bound = dict(
        protocol_sha256="unused",
        source=dict(commit="scientific", files={"src/glassbox/online.py": "same"}),
        runtime={"x64_enabled": False},
    )
    trace.write(reference / "binding.json", bound)
    trace.write(
        reference / "protocol.json", dict(id=protocol["reference"]["protocol_id"])
    )
    trace.write(output / "manifest.json", dict(format="glassbox-" + protocol["id"]))
    trace.write(output / "protocol.json", protocol)
    trace.write(
        output / "binding.json",
        dict(bound, protocol_sha256=trace.digest(output / "protocol.json")),
    )

    def diagnostics(path, **kwargs):
        assert kwargs == ({"angular": True} if angular else {})
        context = trace.arrays(path / "context.npz")
        return dict(origin=int(context["origin"])), dict(
            prediction=context["recorded_prediction"].astype(np.float64)
        )

    monkeypatch.setattr(trace, "snapshot_diagnostics", diagnostics)
    for row in case["captures"]:
        path = output / case["id"] / str(row)
        summary, values = diagnostics(path, **({"angular": True} if angular else {}))
        trace.write(path / "diagnostics.json", summary)
        trace.checkpoint(path / "diagnostics.npz", **values)
    summary, values = trace.update_comparison(output / case["id"], case, data)
    trace.write(output / case["id"] / "updates.json", summary)
    trace.checkpoint(output / case["id"] / "updates.npz", **values)
    trace.write(
        output / "summary.json",
        dict(
            complete=True,
            cases=[result],
            transitions=225,
            captures=len(case["captures"]),
            learner_changed=False,
            accuracy_adoption=False,
        ),
    )
    monkeypatch.setattr(trace, "authenticate", lambda *args: None)
    monkeypatch.setattr(evaluate_online, "verify", lambda *args: dict(verified=True))

    def forbidden(*args):
        pytest.fail("verification attempted assimilation")

    monkeypatch.setattr(Session, "observe", forbidden)
    verified = trace.verify(output, "fixture")
    assert verified["verified"] and verified["optimizer_steps"] == 0
    if tampering == "diagnostic":
        changed = output / case["id"] / str(case["captures"][0]) / "diagnostics.npz"
        saved = trace.arrays(changed)
        saved["prediction"][0] += 1
        trace.checkpoint(changed, **saved)
        message = "diagnostic prediction"
    else:
        row = next(row for row in case["captures"] if (row - 15) % 2 == 0)
        assert Session(row).model.fingerprint == Session(row + 1).model.fingerprint
        np.savez(
            output / case["id"] / str(row) / "session.npz", cursor=np.asarray(row + 1)
        )
        message = "before assimilation"
    with pytest.raises(ValueError, match=message):
        trace.verify(output, "fixture")


def test_saved_recent_cache_is_checked_against_actual_causal_tape(
    tmp_path, monkeypatch
):
    reference, case, _data, _info, Session, _events = replay_fixture(
        tmp_path, monkeypatch
    )
    tape = trace.arrays(reference / "inputs" / (case["id"] + ".npz"))
    session, initial = Session(180), Session(15)
    trace.validate_cache(session, tape, initial)
    session._recent["future_states"][0, 0, 0] += 0.2
    with pytest.raises(ValueError, match="causal recent cache future_states"):
        trace.validate_cache(session, tape, initial)


@pytest.mark.parametrize("angular", [False, True])
def test_snapshot_diagnostics_passes_angular_only_for_new_protocol(
    tmp_path, monkeypatch, angular
):
    import _online_trace_diagnostics as diagnostic

    reference, case, _data, _info, _Session, _events = replay_fixture(
        tmp_path, monkeypatch
    )
    output = tmp_path / "case"
    trace.replay_case(reference, output, case)
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return {}, {}

    monkeypatch.setattr(diagnostic, "diagnose_snapshot", record)
    trace.snapshot_diagnostics(output / str(case["captures"][0]), angular=angular)
    assert calls == ([{"angular": True}] if angular else [{}])
