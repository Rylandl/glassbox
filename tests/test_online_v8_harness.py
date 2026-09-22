"""Bounded backtracking evidence, bookkeeping and regression gates; no fitting."""

import copy
import hashlib
import json

import evaluate_online as evaluate
import numpy as np
import pytest
from test_online_harness import CaptureOnline, FakeOnline, aggregate_case, run_fixture


def record(row):
    counts = (1, 2, 3, 5, 5)
    count = counts[row]
    trials = [
        dict(alpha=a, loss=11.0, predicted_reduction=2.0, finite=True)
        for a in evaluate.BACKTRACKING_ALPHAS[:count]
    ]
    if row < 3:
        trials[-1].update(loss=9.0, predicted_reduction=(1.0, 2.0, 8.0)[row])
    if row == 1:
        trials[0].update(loss=None, finite=False)
    if row == 4:
        for trial in trials:
            trial.update(loss=9.0)
    return dict(
        current_loss=10.0,
        conditioning_finite=row != 4,
        direction_finite=True,
        trust_shrink=0.8,
        trials=trials,
        selected_alpha=trials[-1]["alpha"] if row < 3 else 0.0,
        trial_evaluations=count,
    )


class BacktrackingOnline(FakeOnline):
    @property
    def model(self):
        model = super().model
        count = min(self.count, 3)
        model.params = {"w": np.asarray([float(count)])}
        model.fingerprint = hashlib.sha256(str(count).encode()).hexdigest()
        return model

    @property
    def report(self):
        result = super().report
        result.update(
            cg_iterations=16 * self.count,
            curvature_calls=16 * self.count,
            objective_calls=self.count + sum((1, 2, 3, 5, 5)[: self.count]),
            accepted_proposals=min(self.count, 3),
            damping=(1.0, 0.5, 0.5, 2.0, 8.0, 32.0)[self.count],
            last_proposal=record(self.count - 1) if self.count else None,
        )
        return result


class CaptureBacktrackingOnline(CaptureOnline, BacktrackingOnline):
    def metadata(self):
        meta = super().metadata()
        meta.update(
            format="glassbox-online-fit-v8",
            recipe=dict(
                proposals=1,
                cg_iterations=16,
                backtracking=dict(scales=list(evaluate.BACKTRACKING_ALPHAS)),
            ),
            last_proposal=self.report["last_proposal"],
            damping=self.report["damping"],
        )
        return meta

    def archive(self):
        values = super().archive()
        values["param_w"] = self.model.params["w"]
        return values


def v8():
    return evaluate.read(evaluate.ROOT / "docs/harness/online-fit-v8.json")


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "save_endpoint_diagnostics", lambda _: {})
    result, output, stream = run_fixture(
        tmp_path, monkeypatch, protocol=v8(), online_type=BacktrackingOnline
    )
    data = evaluate.arrays(output / "predictions.npz")
    info = evaluate.read(output / "case.json")
    return result, output, stream, data, info


def test_v8_first_acceptance_counts_and_damping_without_learner_calls(
    tmp_path, monkeypatch
):
    import glassbox

    result, output, stream, data, info = fixture(tmp_path, monkeypatch)
    assert result["complete"]
    np.testing.assert_array_equal(data["trial_evaluations"], [1, 2, 3, 5, 5])
    np.testing.assert_array_equal(data["selected_alpha"], [1, 0.5, 0.25, 0, 0])
    assert result["work"] == dict(
        observations=5,
        scheduled_cg_iterations=80,
        gradient_calls=5,
        current_objective_calls=5,
        trial_objective_calls=16,
        total_objective_calls=21,
        trial_count_histogram={"1": 1, "2": 1, "3": 1, "4": 0, "5": 2},
        selected_alpha_histogram={
            "0.0": 2,
            "1.0": 1,
            "0.5": 1,
            "0.25": 1,
            "0.125": 0,
            "0.0625": 0,
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("saved scalar verification called learner")

    monkeypatch.setattr(glassbox.OnlineFit, "load", forbidden, raising=False)
    monkeypatch.setattr(glassbox.OnlineFit, "observe", forbidden)
    evaluate.verify_journal(output, data, stream, info, v8())


def test_v8_prospective_captures_preserve_latest_decision_and_real_caches(
    tmp_path, monkeypatch
):
    from glassbox._learner_arrays import load_arrays

    protocol = v8()
    protocol["causal_captures"]["origins"] = dict(fixture=[16, 18, 19])
    events = []
    monkeypatch.setattr(CaptureBacktrackingOnline, "events", events)
    monkeypatch.setattr(evaluate, "save_endpoint_diagnostics", lambda _: {})
    result, output, stream = run_fixture(
        tmp_path, monkeypatch, protocol=protocol, online_type=CaptureBacktrackingOnline
    )
    assert result["complete"]
    data, info = (
        evaluate.arrays(output / "predictions.npz"),
        evaluate.read(output / "case.json"),
    )
    evaluate.verify_journal(output, data, stream, info, protocol)
    assert evaluate.verify_causal_captures(output, data, stream, info, protocol) == [
        16,
        18,
        19,
    ]
    for origin in (16, 18, 19):
        meta, _ = load_arrays(output / str(origin) / "session.npz")
        assert meta["last_proposal"] == record(origin - 16)
        assert (
            meta["counts"]["objective_calls"]
            == data["objective_calls_before"][origin - 15]
        )
        saved = next(
            i
            for i, event in enumerate(events)
            if event[:2] == ("save", origin) and event[2].endswith("session.npz")
        )
        predicted = next(
            i for i, event in enumerate(events) if event == ("predict", origin)
        )
        assert saved < predicted
    assert data["model_before"][3] == data["model_after"][3]


@pytest.mark.parametrize(
    "change", ["calls", "damping", "alpha", "count", "chain", "early", "rollback"]
)
def test_v8_journal_rejects_resigned_scalar_or_array_inconsistency(
    tmp_path, monkeypatch, change
):
    _, output, stream, data, info = fixture(tmp_path, monkeypatch)
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    after = events[2]["report"]
    if change == "calls":
        after["objective_calls"] += 1
        data["objective_calls_after"][0] += 1
    elif change == "damping":
        after["damping"] *= 2
    elif change == "alpha":
        data["selected_alpha"][0] = 0.5
    elif change == "count":
        data["trial_evaluations"][0] = 2
    elif change == "chain":
        events[3]["report"]["last_proposal"]["trust_shrink"] = 0.7
    elif change == "early":
        item = events[5]["report"]["last_proposal"]["trials"][0]
        item.update(loss=9.0, finite=True)
    else:
        data["model_after"][3] = "f" * 64
        events[11]["model"] = "f" * 64
    (output / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    with pytest.raises(ValueError):
        evaluate.verify_journal(output, data, stream, info, v8())


@pytest.mark.parametrize(
    "change",
    [
        "nan",
        "bool_count",
        "wrong_alpha",
        "late",
        "short_rejection",
        "finite_null",
        "header_null",
        "negative_current",
        "negative_loss",
    ],
)
def test_v8_proposal_rejects_malformed_or_noncausal_selection(change):
    evidence = record(1)
    if change == "nan":
        evidence["current_loss"] = float("nan")
    elif change == "bool_count":
        evidence["trial_evaluations"] = True
    elif change == "wrong_alpha":
        evidence["trials"][1]["alpha"] = 0.25
    elif change == "late":
        evidence["trials"][0].update(loss=9.0, finite=True)
    elif change == "short_rejection":
        evidence["conditioning_finite"] = False
        evidence["selected_alpha"] = 0
    elif change == "finite_null":
        evidence["trials"][0]["finite"] = True
    elif change == "negative_current":
        evidence["current_loss"] = -1.0
    elif change == "negative_loss":
        evidence["trials"][0]["loss"] = -1.0
    else:
        evidence["current_loss"] = None
    with pytest.raises(ValueError):
        evaluate.proposal_decision(evidence)


def test_v8_invalid_direction_retains_all_attempted_trials():
    evidence = record(4)
    evidence.update(conditioning_finite=True, direction_finite=False, current_loss=None)
    decision = evaluate.proposal_decision(evidence)
    assert not decision["accepted"] and decision["trial_evaluations"] == 5
    evidence["selected_alpha"] = 0.0625
    with pytest.raises(ValueError, match="selection"):
        evaluate.proposal_decision(evidence)


def test_v8_gate_retains_primary_and_separate_robustness_and_sums_work():
    protocol = v8()
    cases = [
        aggregate_case(name, "quad", 0.01)
        for name in ("quad-arm-115", "quad-arm-125", "quad-arm-135", "quad-change")
    ] + [aggregate_case(f"fixedwing-{i}", "fixedwing", 0.01) for i in (80, 81)]
    empty = evaluate.make_data(np.arange(2), np.arange(3), np.zeros((3, 1)), protocol)
    empty["assimilated"][:] = True
    empty["trial_evaluations"][:] = [1, 3]
    empty["selected_alpha"][:] = [1, 0.25]
    for case in cases:
        case["reference_ratios"] = dict.fromkeys(case["ratios"], 0.7)
        case["robustness_ratios"] = dict(
            velocity_tail=0.8, rate_tail=0.8, orientation=1.0, rotation_rate=0.9
        )
        case["work"] = evaluate.work_summary(empty)
    result = evaluate.aggregate(cases, protocol)
    assert result["accuracy_passed"] and result["robustness_passed"]
    assert result["primary_comparison"] == "candidate / saved online-fit-v6"
    assert result["work"]["total_objective_calls"] == 36
    assert result["work"]["scheduled_cg_iterations"] == 192
    for case in cases:
        case["reference_ratios"] = dict.fromkeys(case["ratios"], 0.9)
    assert not evaluate.aggregate(cases, protocol)["accuracy_passed"]
    assert evaluate.aggregate(cases, protocol)["robustness_passed"]


def test_v8_session_verification_uses_scalars_not_current_optimizer(
    tmp_path, monkeypatch
):
    from glassbox import OnlineFit
    from glassbox._learner_arrays import array_fingerprint, save_arrays

    monkeypatch.setattr(OnlineFit, "load", lambda *_: pytest.fail("optimizer loaded"))
    core = dict(param_w=np.arange(3.0), norm_scale=np.ones(3))
    model = dict(format="fixture", dt_s=0.01)
    counts = dict(
        optimizer_steps=1,
        gradient_calls=1,
        objective_calls=3,
        accepted_proposals=1,
        cg_iterations=16,
        curvature_calls=16,
        conditioning_calls=1,
    )
    meta = dict(
        format="glassbox-online-fit-v8",
        model=model,
        initial_cursor=75,
        cursor=76,
        recipe=dict(
            proposals=1,
            cg_iterations=16,
            backtracking=dict(scales=list(evaluate.BACKTRACKING_ALPHAS)),
        ),
        counts=counts,
        damping=1.0,
        last_proposal=record(1),
    )
    path = tmp_path / "session.npz"
    save_arrays(path, meta, core)
    endpoint = array_fingerprint(model, core)
    evaluate.session_arrays(path, dict(first=75), 1, endpoint, v8())
    meta["counts"]["objective_calls"] = 2
    save_arrays(path, meta, core)
    with pytest.raises(ValueError, match="last proposal objective"):
        evaluate.session_arrays(path, dict(first=75), 1, endpoint, v8())


def test_v8_reference_contract_rejects_v4_before_authentication(tmp_path, monkeypatch):
    protocol = copy.deepcopy(v8())
    protocol["comparison"]["reference"]["protocol_id"] = "online-fit-v4"
    monkeypatch.setattr(
        evaluate,
        "authenticate",
        lambda *_: pytest.fail("untrusted reference authenticated"),
    )
    with pytest.raises(ValueError, match="reference protocol contract"):
        evaluate.reference_contract(tmp_path, protocol)


@pytest.mark.parametrize("accepted", [0, 2, True])
def test_v8_rejects_infeasible_saved_acceptance_count(accepted):
    report = dict(
        observations=1,
        optimizer_steps=1,
        gradient_calls=1,
        objective_calls=2,
        accepted_proposals=accepted,
        cg_iterations=16,
        curvature_calls=16,
        conditioning_calls=1,
        damping=0.5,
        last_proposal=record(0),
    )
    with pytest.raises(ValueError):
        evaluate.accounting(report, 1, v8())
