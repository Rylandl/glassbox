"""Prospective policy tests over artificial physical metric rows; no fitting."""

import json
from copy import deepcopy

import pytest

from glassbox.experimental import shared_vehicle_experiment as subject


def decision_fixture(monkeypatch, factual=0.96, response=0.96):
    p = json.loads((subject.ROOT / subject.PROTOCOL_PATH).read_text())
    policy = p["decision"]["aggregation"]
    policy["horizons_s"] = [0.25]
    policy["scope_weights"] = {"primary": 1.0}
    # Bootstrap is descriptive; a small deterministic toy roster tests sharing.
    p["decision"]["uncertainty"]["draws"] = 20
    p["cells"] = {s: [{"id": "p", "group": "primary"}] for s in subject.SIMULATORS}
    p["recordings"] = [
        dict(id=f"{s}/p/{i}", simulator=s, role="test", cell="p")
        for s in p["cells"]
        for i in range(2)
    ]
    rows = {}
    for sim in p["cells"]:
        values = []
        for i in range(2):
            for kind in ("factual", "response"):
                for group in policy["groups"]:
                    for arm in subject.ARMS:
                        ratio = (
                            (response if kind == "response" else factual)
                            if arm == "candidate"
                            else 1.0
                        )
                        mse = ((i + 1) * ratio) ** 2
                        values.append(
                            dict(
                                simulator=sim,
                                scope="primary",
                                cell="p",
                                parent=f"{sim}/p/{i}",
                                query="q-" + kind,
                                origin=1,
                                kind=kind,
                                arm=arm,
                                horizon_s=0.25,
                                horizon_steps=5,
                                group=group,
                                statistic="endpoint",
                                truth_eligible=True,
                                prediction_finite=True,
                                mse=mse,
                                finite_subset_mse=mse,
                                components=3,
                                pair_nonweak=False,
                            )
                        )
        rows[sim] = values
    monkeypatch.setattr(subject, "resolved", lambda _: deepcopy(p))
    return p, rows


@pytest.mark.parametrize(
    "factual,response,passed",
    [
        (0.96, 0.96, True),  # Both4% gains pass; no stale mandatory5% in either kind.
        (0.88, 1.04, True),
        (1.04, 0.88, True),
        (0.974, 0.974, False),  # Individually safe, insufficient joint improvement.
        (0.99, 0.99, False),
        (0.80, 1.051, False),  # Compensation cannot waive a per-kind limit.
        (1.051, 0.80, False),
    ],
)
def test_symmetric_joint_gain_and_each_kind_cap(monkeypatch, factual, response, passed):
    p, rows = decision_fixture(monkeypatch, factual, response)
    result = subject.decide(rows, p)
    assert result["residual_criteria_pass"] is passed
    ratios = result["comparison"]["weighted_geometric_mean_ratios"]
    assert ratios["factual"] == pytest.approx(factual)
    assert ratios["response"] == pytest.approx(response)
    assert result["public_promotion"] is False


def test_weak_probes_remain_in_the_progress_objective(monkeypatch):
    p, rows = decision_fixture(monkeypatch)
    assert not any(r["pair_nonweak"] for values in rows.values() for r in values)
    result = subject.decide(rows, p)
    assert result["residual_criteria_pass"]
    assert result["comparison"]["weighted_geometric_mean_ratios"][
        "response"
    ] == pytest.approx(0.96)


def test_local_losses_do_not_create_an_all_cell_veto(monkeypatch):
    p, rows = decision_fixture(monkeypatch, factual=0.9, response=0.8)
    for r in rows["cascade"]:
        if (
            r["arm"] == "candidate"
            and r["group"] == "velocity_m_s"
            and r["kind"] == "response"
        ):
            r["mse"] = r["finite_subset_mse"] = (1.08 * (int(r["parent"][-1]) + 1)) ** 2
    result = subject.decide(rows, p)
    assert result["residual_criteria_pass"]
    losses = [
        r
        for r in result["comparison"]["physical_comparisons"]
        if r["absolute_change"] > 0
    ]
    assert len(losses) == 1
    assert (
        result["angular_retention"]["bootstrap"]["parent_draws_sha256"]
        == result["comparison"]["bootstrap"]["parent_draws_sha256"]
    )


def test_targeted_angular_guard_remains_mandatory(monkeypatch):
    p, rows = decision_fixture(monkeypatch, factual=0.6, response=0.6)
    for r in rows["crazyflow"]:
        if (
            r["arm"] == "candidate"
            and r["group"] == "body_rate_rad_s"
            and r["kind"] == "factual"
        ):
            r["mse"] = r["finite_subset_mse"] = (
                1.051 * (int(r["parent"][-1]) + 1)
            ) ** 2
    result = subject.decide(rows, p)
    assert result["comparison"]["residual_criteria_pass"]
    assert not result["angular_retention"]["checks"]["factual_aggregate_rmse"]
    assert not result["residual_criteria_pass"]


@pytest.mark.parametrize("flag", ["all_available", "all_input_finite"])
def test_model_availability_and_input_finiteness_are_independent_gates(
    monkeypatch, flag
):
    p, rows = decision_fixture(monkeypatch)
    result = subject.decide(rows, p, **{flag: False})
    assert result["comparison"]["residual_criteria_pass"]
    assert not result["residual_criteria_pass"]


def test_matching_incomplete_truth_stays_visible_without_dropping_parent(monkeypatch):
    p, rows = decision_fixture(monkeypatch)
    for values in rows.values():
        for r in values:
            if r["parent"].endswith("/1"):
                r.update(
                    truth_eligible=False,
                    prediction_finite=False,
                    mse=None,
                    finite_subset_mse=None,
                    components=0,
                )
    result = subject.decide(rows, p)
    assert result["residual_criteria_pass"]
    for comparison in result["comparison"]["physical_comparisons"]:
        assert comparison["conditional_on_incomplete_truth"]
        assert comparison["truth_eligible"] < comparison["planned"]


@pytest.mark.parametrize(
    "mutation", ["omitted_parent", "different_mask", "missing_hold"]
)
def test_incomplete_or_inconsistent_cohort_is_rejected(monkeypatch, mutation):
    p, rows = decision_fixture(monkeypatch)
    if mutation == "omitted_parent":
        rows["crazyflow"] = [
            r for r in rows["crazyflow"] if r["parent"] != "crazyflow/p/1"
        ]
    elif mutation == "different_mask":
        rows["crazyflow"][1]["truth_eligible"] = False
    else:
        rows["cascade"] = [r for r in rows["cascade"] if r["arm"] != "hold"]
    with pytest.raises(ValueError):
        subject.decide(rows, p)


def test_prediction_failure_does_not_become_a_finite_subset_score(monkeypatch):
    p, rows = decision_fixture(monkeypatch)
    rows["cascade"][1].update(prediction_finite=False, mse=None, finite_subset_mse=None)
    result = subject.decide(rows, p)
    assert not result["comparison"]["checks"]["finite_eligible_predictions"]
    assert not result["residual_criteria_pass"]


def sealed_stage(path, *, stage="fit", status="complete", request_change=None):
    path.mkdir(parents=True)
    request = dict(
        output=str(path.resolve()),
        stage=stage,
        simulator="crazyflow",
        protocol_sha256=subject.PROTOCOL_SHA256,
        binding_sha256="binding",
    )
    request.update(request_change or {})
    subject.write(path / "request.json", request)
    subject.write(
        path / "run.json",
        dict(
            stage=stage,
            simulator="crazyflow",
            status=status,
            protocol_sha256=subject.PROTOCOL_SHA256,
            binding_sha256="binding",
            request_sha256=subject.digest(path / "request.json"),
            files=subject.common.inventory(path),
        ),
    )
    return dict(root=str(path), sha256=subject.digest(path / "run.json"))


@pytest.mark.parametrize(
    "field,value",
    [("output", "/unrelated/evaluation"), ("stage", "score"), ("simulator", "cascade")],
)
def test_resealed_stage_cannot_redirect_its_request(tmp_path, field, value):
    entry = sealed_stage(tmp_path / "stage", request_change={field: value})
    with pytest.raises(
        ValueError,
        match=r"request.*(identity|association|output|stage|simulator)|stage.*request",
    ):
        subject._stage(
            entry, simulator="crazyflow", stage="fit", binding_sha256="binding"
        )


@pytest.mark.parametrize(
    "stage,status,allowed",
    [
        ("fit", "complete", True),
        ("fit", "fit_failed", True),
        ("predict", "fit_failed", False),
        ("fit", "hard_timeout_incomplete", False),
        ("fit", "failed", False),
    ],
)
def test_only_expected_numerical_fit_failure_is_a_usable_stage(
    tmp_path, stage, status, allowed
):
    entry = sealed_stage(tmp_path / "stage", stage=stage, status=status)
    if allowed:
        assert subject._stage(entry, stage=stage)[0]["status"] == status
    else:
        with pytest.raises(ValueError, match="terminal"):
            subject._stage(entry, stage=stage)


def test_supervisor_preserves_timeout_prefix_and_prevents_retry(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(
        subject,
        "_authenticate",
        lambda request: ({}, dict(interpreter="python", implementation_commit="fake")),
    )

    def timed_out(*args, **kwargs):
        (tmp_path / "attempt" / "partial.json").write_text('{"phase":"weights"}')
        raise subprocess.TimeoutExpired(args[0], 14400)

    monkeypatch.setattr(subject.subprocess, "run", timed_out)
    context = dict(binding_path=tmp_path / "binding.json", binding_sha256="external")
    with pytest.raises(subprocess.TimeoutExpired):
        subject.fit_candidate("crazyflow", tmp_path / "attempt", **context)
    stage = subject.read(tmp_path / "attempt/run.json")
    assert stage["status"] == "hard_timeout_incomplete"
    assert "partial.json" in stage["files"]
    assert stage["files"] == subject.common.inventory(tmp_path / "attempt")
    assert (
        subject.read(tmp_path / "attempt/exit.json")["status"]
        == "hard_timeout_incomplete"
    )
    with pytest.raises(FileExistsError):
        subject.fit_candidate("crazyflow", tmp_path / "attempt", **context)


def test_actual_tiny_capture_wrapper_roundtrip(monkeypatch, tmp_path):
    import numpy as np

    from glassbox import learner
    from glassbox.experimental import shared_vehicle as vehicle
    from glassbox.experimental.public_mean_flight_fit import _cache_payload
    from glassbox.recordings import SequenceCollection, SequenceSegment

    segments = []
    for i in range(2):
        t = np.arange(20) * 0.05
        x = np.zeros((20, 15))
        x[:, :3] = t[:, None] * np.array([0.2, 0.1, 0.05])
        x[:, 6:] = np.eye(3).reshape(9)
        u = np.sin(t[:-1, None] * np.array([1.0, 2.0, 3.0]) + i)
        segments.append(SequenceSegment(f"record-{i}", "whole", x, u, 0.05))
    c = SequenceCollection(
        tuple(segments),
        configuration_id="toy",
        state_channels=vehicle.STATE_CHANNELS,
        input_channels=("u0", "u1", "u2"),
    )
    train, dev = (learner._extract(c, [f"record-{i}"], 3) for i in range(2))
    original = vehicle.fit_sequence

    def tiny(*args, **kwargs):
        return original(*args, **kwargs, _steps=2, _check_every=1)

    monkeypatch.setattr(vehicle, "fit_sequence", tiny)
    capture = subject.FitCapture(
        tmp_path / "capture", _expected_steps=2, _check_every=1
    )
    model = vehicle._train(
        train,
        dev,
        learner._contract(c),
        learner._recording_content(c),
        error_scale=np.ones((5, 15)),
        channel_weights=np.ones(15),
        observer=capture,
    )
    work = capture.finish(model)
    assert work["gradient_calls_returned"] == 2
    assert work["checkpoints"] == 3
    assert work["gradient_window_visits"] == 6
    assert subject._npz(tmp_path / "capture/weights.npz")["normalization"].shape == (
        5,
        15,
    )
    assert "initial_training_prediction" not in subject._npz(
        tmp_path / "capture/weights.npz"
    )

    def forbidden(*a, **kw):
        pytest.fail("replay must not fit or initialize")

    monkeypatch.setattr(vehicle, "fit_sequence", forbidden)
    from glassbox.experimental import shared_vehicle_core

    monkeypatch.setattr(shared_vehicle_core, "initialize", forbidden)
    replay = subject.verify_capture(
        tmp_path / "capture", model, replay=True, _expected_steps=2, _check_every=1
    )
    assert replay == work
    from types import SimpleNamespace

    for change in ("weights", "work"):
        report = model.report
        if change == "weights":
            report["optimization"]["channel_weights"][0] *= 2
        else:
            report["optimization"]["gradient"]["known_gradient_window_visits"] += 1
        fake = SimpleNamespace(
            report=report,
            _model=model._model,
            _train=model._train,
            _development=model._development,
        )
        with pytest.raises(
            ValueError, match=r"objective report witness|observed gradient report"
        ):
            subject.verify_capture(
                tmp_path / "capture", fake, _expected_steps=2, _check_every=1
            )
    arrays = subject._npz(
        tmp_path
        / f"capture/checkpoint-{model.report['optimization']['selected_step']:04d}.npz"
    )
    key = next(k for k in arrays if k.startswith("param_"))
    arrays[key] = arrays[key].copy()
    arrays[key].flat[0] += 0.1
    np.savez_compressed(
        tmp_path
        / f"capture/checkpoint-{model.report['optimization']['selected_step']:04d}.npz",
        **arrays,
    )
    with pytest.raises(ValueError, match="checkpoint bytes"):
        subject.verify_capture(
            tmp_path / "capture", model, _expected_steps=2, _check_every=1
        )
    assert _cache_payload(
        model._train, model._development, model.contract, model._seen
    )[0]["seen"] == learner._recording_content(c)


def test_partial_capture_retains_entered_failure_and_rejects_missing_weight_witness(
    tmp_path,
):
    import numpy as np

    cap = subject.FitCapture(tmp_path / "capture")
    cap(dict(phase="initializer_entered", training_windows=2))
    cap(
        dict(
            phase="weights",
            normalization=np.ones((2, 15)),
            channel_weights=np.ones(15),
            initial_training_prediction=None,
        )
    )
    cap.close()
    assert (
        subject.verify_capture(tmp_path / "capture", None)["initializer_returns"] == 0
    )
    (tmp_path / "capture/weights.npz").unlink()
    with pytest.raises(ValueError, match="phase-bound weights"):
        subject.verify_capture(tmp_path / "capture", None)


def test_raw_parent_query_identity_is_two_part():
    # Each parent's query IDs deliberately repeat in the physical fixture.
    rows = [
        dict(parent=p, query="factual-0050", mse=i) for i, p in enumerate(("p0", "p1"))
    ]
    indexed = {(r["parent"], r["query"]): r for r in rows}
    assert len(indexed) == 2 and indexed["p1", "factual-0050"]["mse"] == 1


def test_retained_cache_decoding_has_no_numeric_initialization(monkeypatch):
    from glassbox import learner
    from glassbox.experimental import shared_vehicle_core

    def forbidden(*args, **kwargs):
        pytest.fail("cache decoding entered model initialization or forecast")

    monkeypatch.setattr(shared_vehicle_core, "initialize", forbidden)
    monkeypatch.setattr(learner, "_train", forbidden)
    protocol = subject.common.read_protocol()
    for simulator in subject.SIMULATORS:
        ref = subject.prepare_reference(simulator, protocol)
        assert len(ref.train.keys) == 1536 and len(ref.development.keys) == 256
        assert ref.normalization.shape == ref.train.batch.future_states.shape[1:]
        assert ref.channel_weights.shape == (15,)
