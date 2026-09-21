"""Synthetic bookkeeping fixtures only: no learner fit or simulator construction."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import collect_throw as collect
import evaluate_online as evaluate


def observations(n=3):
    result = np.zeros((n, 15))
    result[:, 6:] = np.eye(3).ravel()
    return result


def test_vector_rmse_is_not_component_mean():
    truth = observations()
    prediction = truth.copy()
    prediction[:, :3] = [3, 4, 0]
    prediction[:, 3:6] = [0, 0, 2]
    actual = evaluate.metrics(prediction, truth)
    assert actual["velocity_rmse_m_s"] == 5
    assert actual["body_rate_rmse_rad_s"] == 2
    assert actual["orientation_rmse_rad"] == 0


def test_geodesic_keeps_small_angles_without_float32_trace_floor():
    truth = observations(1)
    prediction = truth.astype(np.float32)
    prediction[0, 6:] = Rotation.from_rotvec([0, 0, 1e-7]).as_matrix().ravel()
    actual = evaluate.metrics(prediction, truth)["orientation_rmse_rad"]
    assert actual == pytest.approx(1e-7, rel=1e-6)
    prediction[0, 6:] = (np.eye(3) * (1 + 1e-6)).ravel()
    assert evaluate.metrics(prediction, truth)["orientation_rmse_rad"] < 1e-12


def test_kinematics_integrates_body_rate_but_holds_velocity_and_rate():
    x = observations(1)[0]
    x[:6] = [2, 3, 4, 0, 0, 2]
    prediction = evaluate.kinematic(x, 0.1)
    np.testing.assert_array_equal(prediction[:6], x[:6])
    assert Rotation.from_matrix(
        prediction[6:].reshape(3, 3)
    ).magnitude() == pytest.approx(0.2)


def aggregate_case(name, family, value, complete=True):
    return dict(
        id=name,
        family=family,
        complete=complete,
        ratios={"velocity_rmse_m_s": value, "body_rate_rmse_rad_s": value},
        latency={"observe": {"warm_p95_s": 0.001}},
        prefix={"dt_s": 0.01},
    )


def test_aggregation_equal_family_weights_and_no_single_cell_veto():
    cases = [
        aggregate_case(name, "quad", 0.25)
        for name in ("quad-arm-115", "quad-arm-125", "quad-arm-135", "quad-change")
    ]
    cases += [aggregate_case(f"fixedwing-{i}", "fixedwing", 0.81) for i in (80, 81)]
    protocol = evaluate.read(evaluate.ROOT / "docs/harness/online-fit-v2.json")
    result = evaluate.aggregate(cases, protocol)
    assert result["aggregate_ratio"] == pytest.approx(0.45)
    assert result["accuracy_passed"]
    cases[0]["ratios"] = {"velocity_rmse_m_s": 2.0, "body_rate_rmse_rad_s": 2.0}
    assert evaluate.aggregate(cases, protocol)["accuracy_passed"]
    cases[0]["complete"] = False
    assert not evaluate.aggregate(cases, protocol)["accuracy_passed"]
    cases[0]["complete"] = True
    cases[0]["id"] = cases[1]["id"]
    assert not evaluate.aggregate(cases, protocol)["accuracy_passed"]


class FakeOnline:
    """Deterministic API stand-in; holds motion and counts assimilation only."""

    def __init__(self, prefix):
        segment = prefix.segments[0]
        self.cursor = segment.start_row + len(segment.inputs)
        self.initial_cursor = self.cursor
        self.count = 0
        self.history = round(0.5 / segment.dt_s)
        self.norms = dict(
            body_mean=np.zeros(9),
            body_scale=np.ones(9),
            motion_bound_scale=np.full(6, 4.0),
            input_mean=np.zeros(segment.inputs.shape[1]),
            input_scale=np.ones(segment.inputs.shape[1]),
            feature_scale=np.ones(12),
            quadratic_scale=np.ones(6),
            output_scale=np.ones(6),
        )
        self.predicted = False

    @property
    def model(self):
        return SimpleNamespace(
            params={"w": np.asarray([float(self.count)])},
            norms=copy.deepcopy(self.norms),
            history_steps=self.history,
            fingerprint=hashlib.sha256(str(self.count).encode()).hexdigest(),
        )

    @property
    def report(self):
        return dict(
            cursor=self.cursor,
            observations=self.count,
            optimizer_steps=self.count,
            gradient_calls=self.count,
            objective_calls=2 * self.count,
            accepted_proposals=self.count,
            cg_iterations=4 * self.count,
            curvature_calls=4 * self.count,
            conditioning_calls=self.count,
            damping=1.0,
        )

    def predict(self, past, inputs, future):
        assert len(past) == len(inputs) + 1 and len(future) == 1
        self.predicted = True
        return past[-1:].astype(np.float32)

    def observe(self, index, command, observation):
        assert self.predicted and index == self.cursor and observation.shape == (15,)
        self.cursor += 1
        self.count += 1
        self.predicted = False

    def save(self, path):
        np.savez_compressed(
            path,
            param_w=self.model.params["w"],
            **{"norm_" + key: value for key, value in self.norms.items()},
        )


def run_fixture(tmp_path, monkeypatch, *, fail=False, short=False, protocol=None):
    import glassbox

    class Chosen(FakeOnline):
        def observe(self, *args):
            if fail:
                raise RuntimeError("synthetic update failure")
            super().observe(*args)

    monkeypatch.setattr(glassbox, "OnlineFit", Chosen)
    count = 10 if short else 20
    native = np.zeros((count + 1, 13))
    native[:, 6] = 1
    native[:, 3] = np.arange(count + 1) * 0.05
    source = tmp_path / "stream.npz"
    np.savez_compressed(
        source,
        states=native,
        commands=np.zeros((count, 3)),
        time_s=np.arange(count + 1) * 0.05,
    )
    info = dict(
        id="fixture",
        family="fixedwing",
        opaque_id="opaque",
        dt_s=0.05,
        begin=0,
        first=15,
        ordered_commands=["throttle [1]", "aileron [rad]", "elevator [rad]"],
    )
    output = tmp_path / "case"
    if protocol is None:
        protocol = evaluate.read(evaluate.ROOT / "docs/harness/online-fit-v4.json")
    result = evaluate.evaluate_case(source, output, info, protocol)
    return result, output, evaluate.arrays(source)


def test_causal_journal_and_exact_accounting_without_fitting(tmp_path, monkeypatch):
    result, output, source = run_fixture(tmp_path, monkeypatch)
    assert result["complete"] and result["predicted"] == result["assimilated"] == 5
    data, info = (
        evaluate.arrays(output / "predictions.npz"),
        evaluate.read(output / "case.json"),
    )
    evaluate.verify_journal(output, data, source, info)
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in events] == [
        "predicted",
        "revealed",
        "assimilated",
    ] * 5
    events[0]["phase"] = "revealed"
    (output / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    with pytest.raises(ValueError, match="journal order"):
        evaluate.verify_journal(output, data, source, info)


@pytest.mark.parametrize("short", [False, True])
def test_partial_failure_preserves_missing_rows_and_checkpoint(
    tmp_path, monkeypatch, short
):
    result, output, source = run_fixture(tmp_path, monkeypatch, fail=True, short=short)
    assert not result["complete"]
    assert result["assimilated"] == 0
    assert (output / "predictions.npz").is_file()
    data, info = (
        evaluate.arrays(output / "predictions.npz"),
        evaluate.read(output / "case.json"),
    )
    assert info["status"] == "failed"
    evaluate.verify_journal(output, data, source, info)


def test_sealed_numeric_mutation_is_rejected_even_with_new_local_checksum(tmp_path):
    np.savez(tmp_path / "values.npz", x=np.zeros(3))
    authority = collect.seal(tmp_path, "fixture")
    collect.authenticate(tmp_path, authority)
    np.savez(tmp_path / "values.npz", x=np.ones(3))
    with pytest.raises(ValueError, match="artifact hash"):
        collect.authenticate(tmp_path, authority)
    manifest = collect.read(tmp_path / "manifest.json")
    manifest["files"]["values.npz"] = collect.digest(tmp_path / "values.npz")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest authority"):
        collect.authenticate(tmp_path, authority)


def test_output_is_exclusive(tmp_path):
    with pytest.raises(FileExistsError):
        collect.run(tmp_path)
    with pytest.raises(FileExistsError):
        evaluate.run(tmp_path, "unused", tmp_path)
    assert not list(tmp_path.iterdir())


def test_collection_reuse_is_bound_to_original_manifest_and_protocol(tmp_path):
    (tmp_path / "binding.json").write_text(json.dumps(dict(protocol_sha256="v1")))
    protocol = dict(
        collection_protocol_sha256="v1", collection_manifest_sha256="sealed"
    )
    evaluate.collection_contract(tmp_path, "sealed", protocol, "v2")
    with pytest.raises(ValueError, match="manifest differs"):
        evaluate.collection_contract(tmp_path, "wrong", protocol, "v2")
    with pytest.raises(ValueError, match="protocol differs"):
        evaluate.collection_contract(tmp_path, "sealed", {}, "v2")


@pytest.mark.parametrize(
    "version,proposals", [(1, 4), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
)
def test_generic_session_archive_verification_without_optimizer_loader(
    tmp_path, monkeypatch, version, proposals
):
    from glassbox import OnlineFit
    from glassbox._learner_arrays import array_fingerprint, save_arrays

    def forbidden(*args):
        raise AssertionError("must not load a legacy optimizer")

    monkeypatch.setattr(OnlineFit, "load", forbidden)
    protocol = dict(
        id=f"online-fit-v{version}", candidate=dict(proposals_per_observation=proposals)
    )
    core = dict(param_w=np.arange(3.0), norm_scale=np.ones(3))
    model = dict(format="fixture", dt_s=0.01)
    counts = dict(
        optimizer_steps=proposals,
        gradient_calls=proposals,
        objective_calls=2 if version >= 2 else 8,
        accepted_proposals=1,
    )
    if version >= 2:
        counts.update(cg_iterations=4, curvature_calls=4)
    if version >= 3:
        counts["conditioning_calls"] = 1
    meta = dict(
        format=f"glassbox-online-fit-v{version}",
        model=model,
        initial_cursor=75,
        cursor=76,
        recipe=dict(proposals=proposals),
        counts=counts,
        damping=1.0,
    )
    path = tmp_path / "session.npz"
    save_arrays(path, meta, core)
    identity = array_fingerprint(model, core)
    result = evaluate.session_arrays(path, dict(first=75), 1, identity, protocol)
    np.testing.assert_array_equal(result["param_w"], core["param_w"])
    with pytest.raises(ValueError, match="endpoint differs"):
        evaluate.session_arrays(path, dict(first=75), 1, "wrong", protocol)
    if version >= 2:
        meta["counts"]["curvature_calls"] = 3
        save_arrays(path, meta, core)
        with pytest.raises(ValueError, match="curvature accounting"):
            evaluate.session_arrays(path, dict(first=75), 1, identity, protocol)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_other_protocol_cannot_run_maintained_candidate(tmp_path, monkeypatch, version):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(dict(id=f"online-fit-v{version}")))

    def forbidden(*args):
        raise AssertionError("no source or learner work before protocol rejection")

    monkeypatch.setattr(evaluate, "binding", forbidden)
    with pytest.raises(ValueError, match="unsupported candidate protocol"):
        evaluate.run(tmp_path, "unused", tmp_path / "out", protocol)
    assert (tmp_path / "out/failure.json").exists()
    assert (tmp_path / "out/manifest.json").exists()


@pytest.mark.parametrize("version", [3, 4])
def test_paired_gate_uses_saved_online_not_weak_frozen_baseline(version):
    protocol = evaluate.read(evaluate.ROOT / f"docs/harness/online-fit-v{version}.json")
    cases = [
        aggregate_case(name, "quad", 0.1)
        for name in ("quad-arm-115", "quad-arm-125", "quad-arm-135", "quad-change")
    ] + [aggregate_case(f"fixedwing-{i}", "fixedwing", 0.1) for i in (80, 81)]
    for case in cases:
        case["reference_ratios"] = dict.fromkeys(case["ratios"], 1.1)
    actual = evaluate.aggregate(cases, protocol)
    assert actual["aggregate_ratio"] == pytest.approx(1.1)
    assert actual["frozen_comparison"]["aggregate_ratio"] == pytest.approx(0.1)
    assert not actual["accuracy_passed"]
    for case in cases:
        case["reference_ratios"] = dict.fromkeys(case["ratios"], 0.7)
    assert evaluate.aggregate(cases, protocol)["accuracy_passed"]
    cases[0]["reference_ratios"] = dict.fromkeys(cases[0]["ratios"], 1.1)
    assert evaluate.aggregate(cases, protocol)["accuracy_passed"]
    cases[0]["complete"] = False
    assert not evaluate.aggregate(cases, protocol)["accuracy_passed"]


def test_reference_input_comparison_rejects_changed_commands_and_inventory():
    stream = dict(states=np.zeros((3, 13)), commands=np.zeros((2, 3)))
    evaluate.paired_inputs(stream, copy.deepcopy(stream))
    changed = copy.deepcopy(stream)
    changed["commands"][1, 0] = 1.0
    with pytest.raises(ValueError, match="reference input commands"):
        evaluate.paired_inputs(stream, changed)
    with pytest.raises(ValueError, match="input inventory"):
        evaluate.paired_inputs(stream, {"states": stream["states"]})


@pytest.mark.parametrize("key", ["origin", "time_s", "truth", "frozen", "kinematic"])
def test_paired_reference_requires_matching_rows_targets_and_comparators(
    tmp_path, monkeypatch, key
):
    _, output, _ = run_fixture(tmp_path, monkeypatch)
    data = evaluate.arrays(output / "predictions.npz")
    info = evaluate.read(output / "case.json")
    reference = copy.deepcopy(data)
    evaluate.paired_case(data, info, reference, info)
    reference[key].flat[0] += 1
    with pytest.raises(ValueError, match="reference " + key):
        evaluate.paired_case(data, info, reference, info)


def test_paired_metrics_recompute_reference_from_predictions(tmp_path, monkeypatch):
    _, output, _ = run_fixture(tmp_path, monkeypatch)
    data = evaluate.arrays(output / "predictions.npz")
    info = evaluate.read(output / "case.json")
    reference = copy.deepcopy(data)
    reference["candidate"][:, 0] += 2
    result = evaluate.summarize(data, info, reference)
    expected = evaluate.metrics(reference["candidate"], data["truth"])
    assert result["metrics"]["reference"] == expected
    assert result["reference_ratios"]["velocity_rmse_m_s"] == pytest.approx(
        result["metrics"]["candidate"]["velocity_rmse_m_s"]
        / expected["velocity_rmse_m_s"]
    )


def normalization_fixture():
    return dict(
        feature_scale=np.ones(12),
        quadratic_scale=np.ones(3),
        output_scale=np.ones(6),
        input_scale=np.ones(3),
    )


def test_only_declared_coordinates_can_grow_after_accepted_proposal():
    protocol = evaluate.read(evaluate.PROTOCOL)
    initial = normalization_fixture()
    changed = copy.deepcopy(initial)
    changed["feature_scale"][:4] *= 2
    changed["quadratic_scale"] *= 3
    changed["output_scale"] *= 4
    evaluate.check_normalizers(initial, changed, protocol)
    with pytest.raises(ValueError, match="fixed normalization"):
        evaluate.check_normalizers(initial, changed, protocol, accepted=False)
    bad = copy.deepcopy(changed)
    bad["input_scale"] *= 2
    with pytest.raises(ValueError, match="fixed normalization input_scale"):
        evaluate.check_normalizers(initial, bad, protocol)
    bad = copy.deepcopy(changed)
    bad["feature_scale"][-1] *= 2
    with pytest.raises(ValueError, match="fixed hidden"):
        evaluate.check_normalizers(initial, bad, protocol)
    with pytest.raises(ValueError, match="decreasing"):
        evaluate.check_normalizers(initial, initial, protocol, previous=changed)
    bad = copy.deepcopy(changed)
    bad["output_scale"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate.check_normalizers(initial, bad, protocol)
    protocol["candidate"]["dynamic_normalizers"].append("input_scale")
    with pytest.raises(ValueError, match="normalization contract"):
        evaluate.check_normalizers(initial, changed, protocol)


def test_journal_rejects_normalizer_decrease_from_saved_payload(tmp_path, monkeypatch):
    _, output, source = run_fixture(tmp_path, monkeypatch)
    data = evaluate.arrays(output / "predictions.npz")
    info = evaluate.read(output / "case.json")
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    event = next(row for row in events if row["phase"] == "assimilated")
    offset = event["arrays"]["norm_feature_scale"]
    with (output / "arrays.bin").open("r+b") as stream:
        stream.seek(offset)
        changed = np.load(stream, allow_pickle=False)
        changed[0] = 0.5
        stream.seek(offset)
        np.save(stream, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="decreasing dynamic normalization"):
        evaluate.verify_journal(output, data, source, info)


def test_reference_authority_is_checked_before_candidate_or_source_work(
    tmp_path, monkeypatch
):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "protocol.json").write_text(json.dumps(dict(id="online-fit-v2")))
    collect.seal(reference, "fixture")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(evaluate.read(evaluate.PROTOCOL)))

    def forbidden(*args):
        raise AssertionError("no source work before reference authentication")

    monkeypatch.setattr(evaluate, "binding", forbidden)
    with pytest.raises(ValueError, match="manifest authority"):
        evaluate.run(tmp_path, "unused", tmp_path / "out", protocol, reference)
    assert (tmp_path / "out/failure.json").exists()


def test_reference_initial_core_arrays_are_compared_without_session_loader():
    model = SimpleNamespace(params={"w": np.arange(3.0)}, norms={"s": np.ones(3)})
    archive = dict(param_w=np.arange(3.0), norm_s=np.ones(3), irrelevant=np.asarray(1))
    evaluate.paired_initial(model, archive)
    archive["param_w"][0] = 1.0
    with pytest.raises(ValueError, match="reference initialization"):
        evaluate.paired_initial(model, archive)


def test_v5_tail_metrics_keep_all_rows_and_round_upper_decile_up():
    truth = observations(11)
    forecast = truth.copy()
    forecast[:, 0] = np.arange(11)
    result = evaluate.forecast_diagnostics(forecast, truth, truth, 0.05)
    velocity = result["tail"]["velocity_m_s"]
    assert result["count"] == 11
    assert velocity["median"] == 5
    assert velocity["p95"] == 9.5
    assert velocity["p99"] == 9.9
    assert velocity["maximum"] == 10
    assert velocity["upper_decile_count"] == 2
    assert velocity["upper_decile_rmse"] == pytest.approx(np.sqrt((100 + 81) / 2))
    assert velocity["largest_five_squared_error_share"] == pytest.approx(
        np.sum(np.arange(6, 11) ** 2) / np.sum(np.arange(11) ** 2)
    )
    assert evaluate.metrics(forecast, truth)["velocity_rmse_m_s"] == pytest.approx(
        np.sqrt(np.mean(np.arange(11) ** 2))
    )
    assert result["tail"]["orientation_rad"]["largest_five_squared_error_share"] == 0
    forecast[-1, 0] = np.nan
    assert evaluate.forecast_diagnostics(forecast, truth, truth, 0.05) is None
    assert evaluate.forecast_diagnostics(truth[:0], truth[:0], truth[:0], 0.05) is None


def test_v5_rotation_rate_uses_observed_nonlinear_defect_not_zero_target():
    origins = observations(2)
    truth = origins.copy()
    truth[:, 6:] = (
        Rotation.from_rotvec([[0, 0, 0.03], [0, 0, 0.08]]).as_matrix().reshape(2, 9)
    )
    truth[:, 5] = [0.5, 0.9]
    dt = 0.1
    actual = evaluate.forecast_diagnostics(truth, truth, origins, dt)["rotation_rate"]
    expected_defect = np.array([0.3 - 0.25, 0.8 - 0.45])
    assert actual["truth_relative_rmse_rad_s"] == 0
    assert actual["truth_defect_rmse_rad_s"] == pytest.approx(
        np.sqrt(np.mean(expected_defect**2))
    )
    assert actual["predicted_defect_rmse_rad_s"] == actual["truth_defect_rmse_rad_s"]
    prediction = truth.copy()
    prediction[:, 5] += 0.2
    changed = evaluate.forecast_diagnostics(prediction, truth, origins, dt)[
        "rotation_rate"
    ]
    assert changed["truth_relative_rmse_rad_s"] == pytest.approx(0.1)


@pytest.mark.parametrize("version", [5, 6])
def test_prospective_diagnostics_recompute_every_arm(tmp_path, monkeypatch, version):
    _result, output, stream = run_fixture(tmp_path, monkeypatch)
    data = evaluate.arrays(output / "predictions.npz")
    info = evaluate.read(output / "case.json")
    reference = copy.deepcopy(data)
    reference["candidate"][:, 0] += 2
    for old_version in (1, 2, 3, 4):
        historical = dict(id=f"online-fit-v{old_version}")
        result = evaluate.summarize(data, info, reference, historical, stream)
        assert "diagnostics" not in result and "robustness_ratios" not in result
    protocol = evaluate.read(evaluate.ROOT / f"docs/harness/online-fit-v{version}.json")
    result = evaluate.summarize(data, info, reference, protocol, stream)
    assert set(result["diagnostics"]) == {
        "candidate",
        "reference",
        "frozen",
        "kinematic",
    }
    origins = evaluate.observed(stream["states"][data["origin"]])
    expected = evaluate.forecast_diagnostics(
        reference["candidate"], data["truth"], origins, info["dt_s"]
    )
    assert result["diagnostics"]["reference"] == expected
    ratio = result["robustness_ratios"]["velocity_tail"]
    assert ratio == pytest.approx(
        result["diagnostics"]["candidate"]["tail"]["velocity_m_s"]["upper_decile_rmse"]
        / expected["tail"]["velocity_m_s"]["upper_decile_rmse"]
    )


@pytest.mark.parametrize("version", [5, 6])
def test_robustness_is_separate_and_uses_equal_families_without_case_veto(version):
    protocol = evaluate.read(evaluate.ROOT / f"docs/harness/online-fit-v{version}.json")
    cases = [
        aggregate_case(name, "quad", 0.01)
        for name in ("quad-arm-115", "quad-arm-125", "quad-arm-135", "quad-change")
    ] + [aggregate_case(f"fixedwing-{i}", "fixedwing", 0.01) for i in (80, 81)]
    for case in cases:
        case["reference_ratios"] = dict.fromkeys(case["ratios"], 0.7)
        value = 0.25 if case["family"] == "quad" else 0.81
        case["robustness_ratios"] = dict.fromkeys(
            ("velocity_tail", "rate_tail", "orientation", "rotation_rate"), value
        )
    result = evaluate.aggregate(cases, protocol)
    assert result["accuracy_passed"] and result["robustness_passed"]
    assert result["primary_comparison"] == "candidate / saved online-fit-v4"
    assert result["robustness"]["aggregate_ratios"] == pytest.approx(
        dict.fromkeys(("upper_decile", "orientation", "rotation_rate"), 0.45)
    )
    cases[0]["robustness_ratios"] = dict.fromkeys(cases[0]["robustness_ratios"], 2.0)
    assert evaluate.aggregate(cases, protocol)["robustness_passed"]
    for case in cases:
        case["robustness_ratios"]["orientation"] = 1.01
    result = evaluate.aggregate(cases, protocol)
    assert result["accuracy_passed"] and not result["robustness_passed"]
    for case in cases:
        case["robustness_ratios"]["orientation"] = 1.0
        case["reference_ratios"] = dict.fromkeys(case["ratios"], 1.1)
    result = evaluate.aggregate(cases, protocol)
    assert not result["accuracy_passed"] and result["robustness_passed"]
    cases[0]["complete"] = False
    assert not evaluate.aggregate(cases, protocol)["robustness_passed"]


@pytest.mark.parametrize("version", [5, 6])
def test_prospective_run_rejects_weaker_v2_reference(tmp_path, monkeypatch, version):
    protocol = evaluate.read(evaluate.ROOT / f"docs/harness/online-fit-v{version}.json")
    protocol["comparison"]["reference"]["protocol_id"] = "online-fit-v2"

    def forbidden(*args):
        pytest.fail("wrong reference version reached authentication")

    monkeypatch.setattr(evaluate, "authenticate", forbidden)
    with pytest.raises(ValueError, match="reference protocol contract"):
        evaluate.reference_contract(tmp_path, protocol)


def test_current_candidate_runner_uses_v6_against_working_v4():
    protocol = evaluate.read(evaluate.PROTOCOL)
    assert protocol["id"] == "online-fit-v6"
    assert protocol["comparison"]["reference"]["protocol_id"] == "online-fit-v4"
    assert evaluate.run.__defaults__[0] == evaluate.PROTOCOL


def endpoint_fixture(*, initial=False):
    meta = dict(format="synthetic-endpoint", model=dict(dt_s=0.05, delay_steps=1))
    values = dict(
        norm_body_mean=np.zeros(9),
        norm_body_scale=np.r_[np.ones(6), [0.5, 2, 4]],
        norm_motion_bound_scale=np.full(6, 1e6),
        norm_input_mean=np.zeros(1),
        norm_input_scale=np.ones(1),
        norm_output_scale=np.arange(1, 7, dtype=float),
        norm_quadratic_scale=np.full(66, 2.0),
        param_quadratic=np.zeros((66, 6)),
        scale=np.ones((2, 15)),
    )
    left, right = np.triu_indices(11)
    values["param_quadratic"][np.flatnonzero((left == 0) & (right == 0))[0], 0] = 2.0
    values["param_quadratic"][np.flatnonzero((left == 0) & (right == 9))[0], 3] = 0.4
    predictions = {}
    for role, n, velocity, command in (
        ("bootstrap", 2, 2.0, 3.0),
        ("recent", 0 if initial else 3, 4.0, 5.0),
    ):
        past = np.zeros((n, 4, 15))
        past[..., 6:] = np.eye(3).ravel()
        past[..., 0] = velocity
        past[:, 0, 0] = 1000  # Excluded by the declared t>=delay domain.
        future = np.repeat(past[:, -1:], 2, axis=1)
        up = np.full((n, 3, 1), command)
        up[:, 0, 0] = 1000
        uf = np.full((n, 2, 1), command)
        for key, value in zip(
            ("past_states", "past_inputs", "future_inputs", "future_states"),
            (past, up, uf, future),
        ):
            values[role + "_" + key] = value
        predictions[role] = future.copy()
        if role == "bootstrap":
            predictions[role][..., :3] += [3, 4, 0]
        else:
            predictions[role][..., 3:6] += [0, 0.6, 0.8]
    return meta, values, predictions


def test_endpoint_numpy_objective_preserves_role_weights_and_physical_prior():
    meta, values, predictions = endpoint_fixture()
    result = evaluate.endpoint_objective(meta, values, predictions)
    expected_domain = np.r_[
        np.sqrt(10), np.ones(5), [2, 0.5, 0.25], np.sqrt(17), np.sqrt(17)
    ]
    np.testing.assert_allclose(result["domain"], expected_domain, rtol=1e-15)
    # Bootstrap norm 5 gives Huber 4.5; recent norm 1 gives Huber 0.5, over 3 groups.
    assert result["data_loss"] == pytest.approx((4.5 / 3 + 0.5 / 3) / 2)
    # Only two physical Hessian terms are nonzero; the off-diagonal contributes twice.
    expected_prior = (
        0.01 / 4 * ((2 * 10 * 0.05 * 1) ** 2 + 2 * (np.sqrt(10 * 17) * 0.05 * 0.8) ** 2)
    )
    assert result["prior_loss"] == pytest.approx(expected_prior)
    assert result["combined_loss"] == result["data_loss"] + result["prior_loss"]
    assert result["cache_windows"] == dict(bootstrap=2, recent=3)
    transformed = copy.deepcopy(values)
    transformed["norm_quadratic_scale"] *= 7
    transformed["norm_output_scale"] *= 3
    transformed["param_quadratic"] *= 7 / 3
    shifted = evaluate.endpoint_objective(meta, transformed, predictions)
    assert shifted["prior_loss"] == pytest.approx(result["prior_loss"], rel=2e-15)
    initial = evaluate.endpoint_objective(*endpoint_fixture(initial=True))
    assert initial["data_loss"] == 1.5
    assert initial["domain"][-2:] == [3.0, 3.0]
    assert initial["prior_loss"] == pytest.approx(
        0.01 / 4 * (0.4**2 + 2 * (2 * 3 * 0.05 * 0.8) ** 2)
    )


def test_endpoint_verifier_uses_saved_predictions_without_model_or_optimizer(
    tmp_path, monkeypatch
):
    from glassbox import OnlineFit
    from glassbox._dynamics import VehicleSequenceModel
    from glassbox._learner_arrays import save_arrays

    result, saved = {}, {}
    for endpoint in ("initial", "final"):
        meta, values, predictions = endpoint_fixture(initial=endpoint == "initial")
        save_arrays(tmp_path / (endpoint + "-online.npz"), meta, values)
        result[endpoint] = evaluate.endpoint_objective(meta, values, predictions)
        saved.update(
            {
                endpoint + "_" + role: prediction
                for role, prediction in predictions.items()
            }
        )
    evaluate.checkpoint(tmp_path / "endpoint-predictions.npz", **saved)
    evaluate.write(tmp_path / "endpoint-objectives.json", result)

    def forbidden(*args):
        pytest.fail("endpoint verification called a model or optimizer")

    monkeypatch.setattr(OnlineFit, "load", forbidden)
    monkeypatch.setattr(OnlineFit, "observe", forbidden)
    monkeypatch.setattr(VehicleSequenceModel, "rollout", forbidden)
    assert evaluate.verify_endpoint_diagnostics(tmp_path) == result
    saved["final_recent"][0, 0, 0] += 1
    evaluate.checkpoint(tmp_path / "endpoint-predictions.npz", **saved)
    with pytest.raises(ValueError, match="endpoint objective differs"):
        evaluate.verify_endpoint_diagnostics(tmp_path)


def test_v6_endpoint_diagnostics_run_after_all_timed_updates(tmp_path, monkeypatch):
    protocol = evaluate.read(evaluate.ROOT / "docs/harness/online-fit-v6.json")
    calls = []

    def saved(output):
        data = evaluate.arrays(output / "predictions.npz")
        assert data["assimilated"].all() and len(data["origin"]) == 5
        assert np.isfinite(data["observe_s"]).all()
        assert evaluate.arrays(output / "final-online.npz")["param_w"][0] == 5
        calls.append(output)
        return dict(
            initial=dict(data_loss=1.0, prior_loss=2.0, combined_loss=3.0),
            final=dict(data_loss=0.5, prior_loss=1.0, combined_loss=1.5),
        )

    monkeypatch.setattr(evaluate, "save_endpoint_diagnostics", saved)
    result, output, _stream = run_fixture(tmp_path, monkeypatch, protocol=protocol)
    assert calls == [output] and result["complete"]
    assert (
        result["endpoint_objectives"]
        == evaluate.read(output / "case.json")["endpoint_objectives"]
    )
