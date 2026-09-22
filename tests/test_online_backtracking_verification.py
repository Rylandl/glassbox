"""Independent scalar nonlinear fixtures for the residual-only NumPy verifier."""

from copy import deepcopy

import numpy as np
import pytest
from _online_backtracking_verification import verify_point

LADDER = [1.0, 0.5, 0.25, 0.125, 0.0625]


def _json_number(value):
    return float(value) if np.isfinite(value) else None


def _fixture(
    *,
    residuals=None,
    conditioning_finite=True,
    reference_finite=True,
    gradient=None,
    production=0.06,
    theta=0.2,
    prior=0.1,
    current_raw=0.8,
    delta=1.0,
    projected=-2.0,
):
    """Assemble one-parameter evidence without the producer or verifier helpers.

    f(theta+s)=0.8-2*s+3*s**2 has a bad full step, an acceptable
    half step and a better quarter step. Its first pass is worse than the
    independent original-four production result of 0.06.
    """
    if gradient is None:
        gradient = -2 * current_raw / 3 + prior * theta
    if residuals is None:
        residuals = [current_raw - 2 * alpha + 3 * alpha**2 for alpha in LADDER]
    raw = np.zeros((1, 1, 15))
    raw[0, 0, 0] = current_raw
    trials = np.zeros((5, 1, 1, 15))
    trials[:, 0, 0, 0] = residuals
    directions = np.full((4, 1), delta)
    projections = np.zeros((4, 1, 1, 15))
    projections[:, 0, 0, 0] = projected
    arrays = {
        "theta": np.asarray([theta]),
        "prior": np.asarray([prior]),
        "gradient": np.asarray([gradient]),
        "weight": np.asarray([[[1 / 3]]]),
        "irls": np.ones_like(raw),
        "raw": raw,
        "damping": np.asarray(999.0),
        "finite": np.asarray(reference_finite),
        "probe_delta": directions,
        "probe_projected": projections,
        "probe_trial_raw": np.repeat(trials[:1], 4, axis=0),
        "production_trial": np.asarray(production),
    }

    def huber(value):
        return value**2 / 6 if abs(value) <= 1 else (abs(value) - 0.5) / 3

    initial_data = huber(current_raw)
    initial_prior = prior * theta**2 / 2
    initial = initial_data + initial_prior
    rows = []
    for alpha, residual in zip(LADDER, residuals, strict=True):
        data = huber(residual)
        penalty = prior * (theta + alpha * delta) ** 2 / 2
        total = data + penalty
        predicted = (
            -alpha * gradient * delta
            - alpha**2 * (projected**2 / 3 + prior * delta**2) / 2
        )
        finite = bool(
            reference_finite
            and conditioning_finite
            and np.isfinite(
                [
                    residual,
                    theta + alpha * delta,
                    gradient,
                    projected,
                    initial,
                    total,
                    predicted,
                ]
            ).all()
        )
        gain = (initial - total) / predicted if finite and predicted > 0 else -np.inf
        rows.append(
            dict(
                alpha=alpha,
                finite=finite,
                would_accept=bool(finite and total < initial and gain >= 0.1),
                data_loss=_json_number(data),
                prior_loss=_json_number(penalty),
                total_loss=_json_number(total),
                data_improvement=_json_number(initial_data - data),
                prior_improvement=_json_number(initial_prior - penalty),
                improvement=_json_number(initial - total),
                predicted_reduction=_json_number(predicted),
                gain=_json_number(gain),
            )
        )
    current = dict(data_loss=initial_data, prior_loss=initial_prior, total_loss=initial)
    parent_probe = {key: value for key, value in rows[0].items() if key != "alpha"}
    if not conditioning_finite:
        # The archived direction had finite conditioning; only the new audit's
        # supplied eligibility is false. Its numerical losses remain identical.
        parent_probe["finite"] = bool(reference_finite and np.isfinite(residuals[0]))
        pred = rows[0]["predicted_reduction"]
        gain = (initial - rows[0]["total_loss"]) / pred if pred > 0 else -np.inf
        parent_probe["gain"] = _json_number(gain)
        parent_probe["would_accept"] = bool(
            parent_probe["finite"] and rows[0]["total_loss"] < initial and gain >= 0.1
        )
    parent = {
        "current": current,
        "reference": dict(status="capped", iterations=128, relative_residual=0.02),
        "probes": [
            {},
            dict(data_loss=0.039, prior_loss=0.020999999, total_loss=0.059999999),
            {},
            parent_probe,
        ],
    }
    selected = next((i for i, row in enumerate(rows) if row["would_accept"]), None)
    selected_loss = production if selected is None else rows[selected]["total_loss"]
    comparison = (
        "win"
        if selected_loss < production
        else "loss"
        if selected_loss > production
        else "equal"
    )
    summary = {
        "format": "online-backtracking-point-v1",
        "current": current,
        "original_four": dict(
            total_loss=production,
            instrumented_data_loss=0.039,
            instrumented_prior_loss=0.020999999,
        ),
        "parent_reference": parent["reference"],
        "steps": rows,
        "selected_index": selected,
        "selected_alpha": None if selected is None else LADDER[selected],
        "selection_comparison": "no_selection" if selected is None else comparison,
        "retained_total_loss": selected_loss,
        "retained_comparison": comparison,
        "hypothetical_residual_evaluations": 5 if selected is None else selected + 1,
        "work": dict(
            new_residual_calls=4,
            reused_residuals=1,
            cg_iterations=0,
            conditioning_calls=0,
            gradient_calls=0,
            initialization_calls=0,
            observe_calls=0,
            applied_updates=0,
        ),
    }
    return summary, parent, arrays, trials


def _verify(evidence, *, conditioning_finite=True):
    return verify_point(*evidence, conditioning_finite=conditioning_finite)


def test_independent_nonlinear_fixture_preserves_first_accept_even_when_worse():
    evidence = _fixture()
    summary, _, arrays, trials = evidence
    originals = {key: value.tobytes() for key, value in arrays.items()}
    trial_bytes = trials.tobytes()
    result = _verify(evidence)
    assert result["verified"] and result["arithmetic_checks"] > 70
    assert result["model_calls"] == result["optimizer_calls"] == 0
    assert not summary["steps"][0]["would_accept"]
    assert summary["selected_index"] == 1
    assert summary["steps"][2]["total_loss"] < summary["steps"][1]["total_loss"]
    assert summary["retained_comparison"] == "loss"
    assert arrays["damping"] == 999  # Excluded from the predicted reduction.
    assert summary["steps"][1]["predicted_reduction"] == pytest.approx(0.0775)
    assert summary["hypothetical_residual_evaluations"] == 2
    assert {key: value.tobytes() for key, value in arrays.items()} == originals
    assert trials.tobytes() == trial_bytes


def test_all_rejected_retains_actual_four_not_instrumented_sum():
    evidence = _fixture(residuals=[3.0] * 5)
    _verify(evidence)
    summary = evidence[0]
    assert summary["selected_index"] is None
    assert summary["selected_alpha"] is None
    assert summary["selection_comparison"] == "no_selection"
    assert summary["retained_comparison"] == "equal"
    assert summary["retained_total_loss"] == 0.06
    assert summary["hypothetical_residual_evaluations"] == 5
    damaged = deepcopy(summary)
    damaged["retained_total_loss"] = 0.059999999
    damaged["retained_comparison"] = "win"
    with pytest.raises(ValueError, match="retained"):
        _verify((damaged, *evidence[1:]))


def test_all_accepted_keeps_full_step_and_later_entries():
    evidence = _fixture(residuals=[0.2] * 5, gradient=-1.0, prior=0.0)
    _verify(evidence)
    assert all(row["would_accept"] for row in evidence[0]["steps"])
    assert evidence[0]["selected_index"] == 0
    assert len(evidence[0]["steps"]) == 5
    assert evidence[0]["hypothetical_residual_evaluations"] == 1


def test_nonfinite_intermediate_alpha_is_retained_without_disabling_later_steps():
    evidence = _fixture(residuals=[1.8, np.nan, 0.4875, np.inf, -np.inf])
    _verify(evidence)
    summary = evidence[0]
    assert summary["selected_index"] == 2
    for index in (1, 3, 4):
        row = summary["steps"][index]
        assert not row["finite"] and not row["would_accept"]
        assert row["data_loss"] is row["total_loss"] is row["gain"] is None


@pytest.mark.parametrize(
    "reference_finite,conditioning_finite", [(False, True), (True, False)]
)
def test_parent_direction_and_conditioning_eligibility_gate_every_alpha(
    reference_finite, conditioning_finite
):
    evidence = _fixture(
        reference_finite=reference_finite, conditioning_finite=conditioning_finite
    )
    _verify(evidence, conditioning_finite=conditioning_finite)
    assert all(not row["finite"] for row in evidence[0]["steps"])
    assert evidence[0]["selected_index"] is None


@pytest.mark.parametrize("gradient", [0.0, 1.0])
def test_zero_and_negative_prediction_never_accept(gradient):
    evidence = _fixture(
        residuals=[0.0] * 5, gradient=gradient, projected=0.0, prior=0.0
    )
    _verify(evidence)
    assert all(row["gain"] is None for row in evidence[0]["steps"])
    assert all(row["finite"] for row in evidence[0]["steps"])
    assert evidence[0]["selected_index"] is None


def test_exact_gain_threshold_and_strict_loss_threshold():
    # Start loss is exactly 1/6, end is zero, predicted is ten times the
    # decrease. The same quotient computes precisely the stored 0.1 threshold.
    base = dict(
        residuals=[0.0] * 5,
        gradient=-(10 * (1 / 6)),
        projected=0.0,
        current_raw=1.0,
        theta=0.0,
        prior=0.0,
    )
    evidence = _fixture(**base)
    _verify(evidence)
    assert evidence[0]["steps"][0]["gain"] == 0.1
    assert evidence[0]["steps"][0]["would_accept"]
    base["gradient"] = np.nextafter(base["gradient"], -np.inf)
    evidence = _fixture(**base)
    _verify(evidence)
    assert evidence[0]["steps"][0]["gain"] < 0.1
    assert not evidence[0]["steps"][0]["would_accept"]
    evidence = _fixture(residuals=[0.8] * 5, gradient=-1.0, prior=0.0)
    _verify(evidence)
    assert all(not row["would_accept"] for row in evidence[0]["steps"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("selected_index", 2),
        ("selected_alpha", 0.25),
        ("selection_comparison", "win"),
        ("retained_comparison", "win"),
        ("hypothetical_residual_evaluations", 3),
        ("format", "online-backtracking-point-v2"),
    ],
)
def test_summary_selection_tampering_fails(field, value):
    evidence = _fixture()
    evidence[0][field] = value
    with pytest.raises(ValueError, match=field):
        _verify(evidence)


@pytest.mark.parametrize(
    "field", ["gain", "predicted_reduction", "prior_loss", "data_loss"]
)
def test_individual_alpha_metric_tampering_fails(field):
    evidence = _fixture()
    evidence[0]["steps"][2][field] += 0.01
    with pytest.raises(ValueError, match=field):
        _verify(evidence)


def test_parent_alpha1_byte_exact_including_signed_zero():
    evidence = _fixture()
    evidence[3][0, 0, 0, 1] = -0.0
    with pytest.raises(ValueError, match="byte-exact"):
        _verify(evidence)


def test_changed_residual_and_parent_reference_summary_fail():
    evidence = _fixture()
    evidence[3][2, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="steps"):
        _verify(evidence)
    evidence = _fixture()
    evidence[1]["probes"][3]["prior_loss"] += 0.1
    with pytest.raises(ValueError, match=r"parent\.reference_trusted"):
        _verify(evidence)


def test_missing_alpha_and_fake_work_count_fail():
    evidence = _fixture()
    with pytest.raises(ValueError, match="shape"):
        _verify((*evidence[:3], evidence[3][:-1]))
    evidence[0]["work"]["gradient_calls"] = 1
    with pytest.raises(ValueError, match="gradient_calls"):
        _verify(evidence)


def test_nonfinite_alpha1_does_not_disable_finite_shorter_direction():
    evidence = _fixture(residuals=[np.nan, 0.55, 0.4875, 0.6, 0.7])
    _verify(evidence)
    assert evidence[2]["finite"]
    assert not evidence[1]["probes"][3]["finite"]
    assert evidence[0]["selected_index"] == 1


def test_nonfinite_gain_preserves_original_pre_gain_finite_rule():
    evidence = _fixture(
        residuals=[0.0] * 5,
        theta=0.0,
        prior=0.0,
        projected=0.0,
        gradient=-1e-310,
    )
    _verify(evidence)
    assert evidence[0]["steps"][0]["gain"] is None
    assert evidence[0]["steps"][0]["finite"]
    assert evidence[0]["steps"][0]["would_accept"]


def test_root_summaries_cross_verify_independent_nonlinear_and_threshold_fixtures():
    from audit_backtracking import summarize

    for kwargs in (
        {},
        dict(residuals=[3.0] * 5),
        dict(residuals=[np.nan, 0.55, 0.4875, np.inf, -np.inf]),
        dict(residuals=[0.0] * 5, gradient=0.0, projected=0.0, prior=0.0),
        dict(
            residuals=[0.0] * 5,
            gradient=-(10 * (1 / 6)),
            projected=0.0,
            current_raw=1.0,
            theta=0.0,
            prior=0.0,
        ),
    ):
        _, parent, arrays, trials = _fixture(**kwargs)
        result = summarize(parent, arrays, trials, conditioning_finite=True)
        _verify((result, parent, arrays, trials))


def test_strict_saved_production_comparison_has_no_tie_band():
    from audit_backtracking import summarize

    _, parent, arrays, trials = _fixture()
    total = summarize(parent, arrays, trials)["retained_total_loss"]
    for production, expected in (
        (np.nextafter(total, np.inf), "win"),
        (total, "equal"),
        (np.nextafter(total, -np.inf), "loss"),
    ):
        arrays["production_trial"] = np.asarray(production)
        result = summarize(parent, arrays, trials)
        _verify((result, parent, arrays, trials))
        assert result["retained_comparison"] == expected


def test_independent_aggregate_includes_rejections_and_rejects_group_tampering():
    from _online_backtracking_verification import verify_aggregate
    from audit_backtracking import aggregate

    accepted = _fixture()[0]
    rejected = _fixture(residuals=[3.0] * 5)[0]
    rows = [
        dict(version="v6", case="fw80", result=accepted),
        dict(version="v6", case="fw81", result=rejected),
        dict(version="v7", case="fw80", result=deepcopy(accepted)),
    ]
    rows[2]["result"]["parent_reference"] = dict(
        status="converged", iterations=64, relative_residual=1e-7
    )
    result = aggregate(rows)
    verification = verify_aggregate(result, rows)
    assert verification["verified"] and verification["arithmetic_checks"] > 50
    assert verification["model_calls"] == verification["optimizer_calls"] == 0
    assert result["all"]["selections"] == {"0.5": 2, "None": 1}
    assert result["all"]["reference_status"] == {"capped": 2, "converged": 1}
    assert result["all"]["hypothetical_residual_evaluations"] == 9
    for section, key in (
        ("all", None),
        ("by_version", "v6"),
        ("by_version_case", "v7/fw80"),
    ):
        damaged = deepcopy(result)
        group = damaged[section] if key is None else damaged[section][key]
        group["accepted_per_alpha"][1] += 1
        with pytest.raises(ValueError, match="accepted_per_alpha"):
            verify_aggregate(damaged, rows)
    damaged = deepcopy(result)
    damaged["by_version"]["invented"] = deepcopy(result["by_version"]["v6"])
    with pytest.raises(ValueError, match="keys differ"):
        verify_aggregate(damaged, rows)


@pytest.mark.parametrize("where", ["alpha", "original_four", "retained_total_loss"])
def test_frozen_labels_and_copied_losses_reject_sub_tolerance_changes(where):
    evidence = _fixture()
    if where == "alpha":
        evidence[0]["steps"][1]["alpha"] = np.nextafter(0.5, np.inf)
    elif where == "original_four":
        evidence[0][where]["total_loss"] = np.nextafter(0.06, np.inf)
    else:
        evidence[0][where] = np.nextafter(evidence[0][where], np.inf)
    with pytest.raises(ValueError):
        _verify(evidence)
