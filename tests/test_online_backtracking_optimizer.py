"""Bounded online nonlinear steps on independent scalar analytic objectives."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_online import prefix, snapshot, stream

from glassbox import OnlineFit, online
from glassbox._learner_arrays import load_arrays, save_arrays


def square_proposal(monkeypatch, start, *, cutoff=None, conditioning=True):
    """A Newton direction overshoots x^2=1; runtime callbacks count actual calls."""
    calls = []

    def residual(params, norms, data, scale, delay, dt_s):
        value = params["x"][0]
        jax.debug.callback(lambda x: calls.append(float(x)), value, ordered=True)
        error = value**2 - 1.0
        if cutoff is not None:
            error = jnp.where(value > cutoff, jnp.nan, error)
        return jnp.zeros((1, 1, 15)).at[0, 0, 0].set(error)

    monkeypatch.setattr(online, "_residual", residual)
    monkeypatch.setattr(online, "_curvature_diagonal", lambda *a, **kw: jnp.zeros(1))
    with jax.enable_x64(True):
        result = online._proposal.__wrapped__(
            {"x": jnp.asarray([start])},
            {},
            (),
            jnp.ones((1, 15)),
            jnp.ones(1),
            jnp.asarray(1e-10),
            delay=1,
            dt_s=0.05,
            conditioning_finite=conditioning,
        )
        jax.block_until_ready(result)
        jax.effects_barrier()
    report = online._proposal_report(result[1], conditioning, result[-1])
    online._validate_proposal_report(report)
    assert len(calls) == 1 + report["trial_evaluations"]
    return result, report, calls


def test_first_acceptable_trial_stops_before_a_better_later_loss(monkeypatch):
    result, report, calls = square_proposal(monkeypatch, 0.1)
    proposal, current, trial, predicted, finite, _ = result
    assert finite
    assert report["selected_alpha"] == 0.25
    assert report["trial_evaluations"] == 3
    assert [item["alpha"] for item in report["trials"]] == [1.0, 0.5, 0.25]
    assert report["trials"][0]["loss"] > float(current)
    assert report["trials"][1]["loss"] > float(current)
    assert float(trial) < float(current) and float(predicted) > 0
    delta = (calls[-1] - 0.1) / 0.25
    better = 0.5 * ((0.1 + 0.125 * delta) ** 2 - 1) ** 2 / 3
    assert better < float(trial)
    assert float(proposal["x"][0]) == calls[-1]


def test_nonfinite_large_trials_do_not_poison_a_finite_shorter_trial(monkeypatch):
    result, report, _ = square_proposal(monkeypatch, 0.1, cutoff=2.0)
    assert result[4]
    assert report["direction_finite"]
    assert report["selected_alpha"] == 0.25
    assert report["trials"][0]["loss"] is None
    assert report["trials"][1]["loss"] is None
    assert [item["finite"] for item in report["trials"]] == [False, False, True]


@pytest.mark.parametrize("start", [0.01, 1.0])
def test_full_ladder_rejection_and_zero_gradient_keep_last_trial(monkeypatch, start):
    result, report, calls = square_proposal(monkeypatch, start)
    assert report["trial_evaluations"] == 5
    assert report["selected_alpha"] == 0.0
    assert result[4]  # Finite rejection still uses the ordinary outer loss/gain rule.
    assert float(result[0]["x"][0]) == calls[-1]
    assert float(result[2]) == report["trials"][-1]["loss"]
    assert float(result[3]) == report["trials"][-1]["predicted_reduction"]
    if start == 1.0:
        assert calls == [1.0] * 6
        assert float(result[1]) == float(result[2]) == float(result[3]) == 0.0


def test_invalid_conditioning_disables_early_acceptance_and_consumes_five_trials(
    monkeypatch,
):
    result, report, _ = square_proposal(monkeypatch, 0.1, conditioning=False)
    assert not result[4]
    assert not report["conditioning_finite"]
    assert report["direction_finite"]
    assert report["trial_evaluations"] == 5
    assert report["selected_alpha"] == 0.0
    assert all(item["finite"] for item in report["trials"])


def test_report_is_independent_and_session_resume_preserves_bounded_evidence(tmp_path):
    states, inputs = stream()
    session = OnlineFit(prefix(states, inputs))
    assert session.report["last_proposal"] is None
    initial_path = tmp_path / "initial.npz"
    session.save(initial_path)
    assert snapshot(OnlineFit.load(initial_path)) == snapshot(session)
    session.observe(session.cursor, inputs[15], states[16])
    report = session.report
    last = report["last_proposal"]
    assert online._validate_proposal_report(last) == bool(report["accepted_proposals"])
    assert report["objective_calls"] == 1 + len(last["trials"])
    assert report["cg_iterations"] == report["curvature_calls"] == 16
    last["trials"][0]["loss"] = -1.0
    assert session.report["last_proposal"]["trials"][0]["loss"] >= 0.0
    path = tmp_path / "session.npz"
    session.save(path)
    meta, arrays = load_arrays(path)
    assert meta["format"] == "glassbox-online-accumulator-v1"
    assert meta["last_proposal"] == session.report["last_proposal"]
    assert not any("proposal" in key for key in arrays)
    resumed = OnlineFit.load(path)
    for row in (16, 17):
        session.observe(session.cursor, inputs[row], states[row + 1])
        resumed.observe(resumed.cursor, inputs[row], states[row + 1])
        assert snapshot(resumed) == snapshot(session)


def test_repaired_archive_rejects_inconsistent_last_proposal_and_work_counts(tmp_path):
    states, inputs = stream()
    session = OnlineFit(prefix(states, inputs))
    session.observe(session.cursor, inputs[15], states[16])
    path = tmp_path / "session.npz"
    session.save(path)
    meta, arrays = load_arrays(path)
    mutations = [
        lambda m: m.update(last_proposal=None),
        lambda m: m["last_proposal"].update(trial_evaluations=6),
        lambda m: m["last_proposal"].update(trial_evaluations=True),
        lambda m: m["last_proposal"].update(direction_finite=1),
        lambda m: m["last_proposal"].update(current_loss=float("nan")),
        lambda m: m["last_proposal"].update(trust_shrink=1.1),
        lambda m: m["last_proposal"]["trials"][0].update(alpha=0.5),
        lambda m: m["last_proposal"]["trials"][0].update(loss=None, finite=True),
        lambda m: m["last_proposal"]["trials"][0].update(
            predicted_reduction=None, finite=True
        ),
        lambda m: m["last_proposal"].update(selected_alpha=0.0),
        lambda m: m["last_proposal"]["trials"].append(
            deepcopy(m["last_proposal"]["trials"][0])
        ),
        lambda m: m["counts"].update(objective_calls=6),
        lambda m: m["counts"].update(cg_iterations=4),
        lambda m: m["counts"].update(accepted_proposals=0),
        lambda m: m.update(format="glassbox-online-fit-v6"),
    ]
    assert meta["last_proposal"]["selected_alpha"] != 0.0
    assert meta["last_proposal"]["trial_evaluations"] < 5
    for index, mutate in enumerate(mutations):
        altered = deepcopy(meta)
        mutate(altered)
        changed = tmp_path / f"tamper-{index}.npz"
        save_arrays(changed, altered, arrays)
        with pytest.raises(ValueError):
            OnlineFit.load(changed)


@pytest.mark.parametrize(
    "target_gain", [0.09999, 0.10001, 0.24999, 0.25001, 0.74999, 0.75001]
)
def test_jitted_selection_and_session_decision_agree_near_gain_thresholds(
    monkeypatch, target_gain
):
    # At x=0 the exact local direction is -1/4; choose polynomial curvature so
    # the full-step exact gain straddles the acceptance and damping thresholds.
    curvature = 16 * (np.sqrt(0.25 - 0.1875 * target_gain) - 0.25)

    def residual(params, norms, data, scale, delay, dt_s):
        x = params["x"][0]
        return jnp.zeros((1, 1, 15)).at[0, 0, 0].set(0.5 + x + curvature * x**2)

    monkeypatch.setattr(online, "_residual", residual)
    monkeypatch.setattr(online, "_curvature_diagonal", lambda *a, **kw: jnp.zeros(1))
    with jax.enable_x64(True):
        result = online._proposal.__wrapped__(
            {"x": jnp.zeros(1)},
            {},
            (),
            jnp.ones((1, 15)),
            jnp.ones(1),
            jnp.asarray(1 / 3),
            delay=1,
            dt_s=0.05,
        )
    _, current, trial, predicted, finite, evidence = result
    report = online._proposal_report(current, True, evidence)
    first = report["trials"][0]
    first_gain = (report["current_loss"] - first["loss"]) / first["predicted_reduction"]
    assert first_gain == pytest.approx(target_gain, abs=2e-12)
    assert (report["trial_evaluations"] == 1) == (target_gain >= 0.1)
    assert online._validate_proposal_report(report)

    def selected_proposal(params, norms, data, scale, weights, damping, **kwargs):
        assert bool(kwargs["conditioning_finite"])
        proposal = dict(params, bias=params["bias"] + 1e-9)
        return proposal, current, trial, predicted, finite, evidence

    monkeypatch.setattr(online, "_proposal", selected_proposal)
    states, inputs = stream()
    session = OnlineFit(prefix(states, inputs))
    session.observe(session.cursor, inputs[15], states[16])
    observed = session.report
    assert observed["last_proposal"] == report
    assert observed["accepted_proposals"] == int(bool(report["selected_alpha"])) == 1
    gain = float((current - trial) / predicted)
    expected_damping = 4.0 if gain < 0.25 else 0.5 if gain > 0.75 else 1.0
    assert observed["damping"] == expected_damping
    assert observed["objective_calls"] == 1 + report["trial_evaluations"]


@pytest.mark.parametrize("corruption", ["evidence", "accounting"])
def test_save_rejects_corrupt_live_proposal_state(tmp_path, corruption):
    states, inputs = stream()
    session = OnlineFit(prefix(states, inputs))
    session.observe(session.cursor, inputs[15], states[16])
    if corruption == "evidence":
        session._last_proposal["trials"][0]["alpha"] = 0.5
    else:
        session._counts["objective_calls"] += 1
    target = tmp_path / "invalid.npz"
    with pytest.raises(ValueError):
        session.save(target)
    assert not target.exists()
