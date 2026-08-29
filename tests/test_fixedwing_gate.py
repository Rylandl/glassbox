import pytest

from glassbox.evaluation import ROLLOUT_METRICS
from glassbox.fixedwing_gate import (
    FIXED_WING_GATE_HORIZONS_S,
    FIXED_WING_SCORE_HORIZONS_S,
    _score_against_persistence,
    _summarize_divergence,
    compare_fixedwing_gates,
    screen_fixedwing_airframe_candidate,
)


def _horizons(value: float) -> dict[str, dict[str, float]]:
    return {
        f"{seconds:g}s": {metric: value for metric in ROLLOUT_METRICS}
        for seconds in FIXED_WING_GATE_HORIZONS_S
    }


def test_fixedwing_gate_score_balances_every_metric_and_horizon() -> None:
    assert _score_against_persistence(_horizons(0.5), _horizons(1.0)) == (
        pytest.approx(0.5)
    )
    assert FIXED_WING_SCORE_HORIZONS_S == (0.5, 1.0, 2.0)


def test_fixedwing_gate_summarizes_divergence_distribution() -> None:
    summary = _summarize_divergence(
        [
            {
                "stable_through_s": 2.0,
                "stable_fraction": 0.5,
                "divergence_causes": ["attitude_error_deg"],
                "full_rollout_finite": True,
                "diverged": True,
            },
            {
                "stable_through_s": 4.0,
                "stable_fraction": 1.0,
                "divergence_causes": [],
                "full_rollout_finite": True,
                "diverged": False,
            },
        ]
    )

    assert summary["trajectory_count"] == 2
    assert summary["full_rollout_finite_fraction"] == 1.0
    assert summary["diverged_fraction"] == 0.5
    assert summary["stable_through_s"]["median"] == 3.0
    assert summary["cause_counts"] == {"attitude_error_deg": 1}


def _gate(candidate: str, value: float, stable_s: float) -> dict:
    return {
        "candidate": candidate,
        "acceptance": {"status": "fail"},
        "airframes": {
            name: {
                "aggregate_horizon_rollouts": _horizons(value),
                "full_rollout_finite_fraction": 1.0,
                "divergence": {"stable_through_s": {"median": stable_s}},
            }
            for name in ("conventional", "flying_wing")
        },
    }


def test_fixedwing_gate_comparison_selects_shared_improvement() -> None:
    comparison = compare_fixedwing_gates(
        _gate("structured", 1.0, 2.0),
        _gate("residual", 0.8, 3.0),
    )

    assert comparison["eligible"] is True
    assert comparison["overall_score"] == pytest.approx(0.8)
    assert comparison["selected_candidate"] == "residual"
    assert comparison["interpretation"] == (
        "selected_for_continued_development_but_contract_not_met"
    )


def _benchmark(value: float) -> dict:
    return {
        "models": {
            "residual": {
                "aggregate": {"horizon_rollouts": _horizons(value)}
            }
        }
    }


def test_single_airframe_screen_advances_only_material_improvements() -> None:
    accepted = screen_fixedwing_airframe_candidate(
        _benchmark(1.0),
        _benchmark(0.8),
        model_name="residual",
        airframe_name="flying_wing",
        candidate_name="candidate",
    )
    rejected = screen_fixedwing_airframe_candidate(
        _benchmark(1.0),
        _benchmark(1.01),
        model_name="residual",
        airframe_name="flying_wing",
        candidate_name="candidate",
    )

    assert accepted["eligible_for_cross_airframe_evaluation"] is True
    assert accepted["can_promote_model"] is False
    assert accepted["overall_score"] == pytest.approx(0.8)
    assert rejected["eligible_for_cross_airframe_evaluation"] is False
    assert rejected["interpretation"] == "reject_before_cross_airframe_fit"
    assert rejected["rejection_reasons"]
