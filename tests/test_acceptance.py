from glassbox.acceptance import (
    evaluate_fixedwing_accuracy,
    evaluate_multirotor_accuracy,
)


def _metrics(position: float, attitude: float) -> dict[str, float]:
    return {"position_rmse_m": position, "attitude_rmse_deg": attitude}


def _profile(
    position: float = 0.05,
    attitude: float = 1.0,
    *,
    path_length: float = 10.0,
) -> dict[str, object]:
    return {
        "path_length_m": path_length,
        "full_rollout": _metrics(position, attitude),
        "horizon_rollouts": {
            "0.1s": _metrics(0.0005, 0.1),
            "0.5s": _metrics(0.005, 0.5),
            "1s": _metrics(0.025, 1.0),
            "2s": _metrics(0.1, 2.5),
        },
    }


def test_accuracy_contract_passes_macro_and_worst_profile() -> None:
    per_profile = {"vertical": _profile(), "lateral": _profile()}

    result = evaluate_multirotor_accuracy(
        state_source="ground_truth",
        aggregate_full_rollout=_metrics(0.05, 1.0),
        aggregate_horizon_rollouts={
            label: metrics
            for label, metrics in per_profile["vertical"]["horizon_rollouts"].items()
        },
        per_profile=per_profile,
    )

    assert result["status"] == "pass"
    assert result["passed"] is True
    assert result["full_rollout"]["passed"] is True


def test_accuracy_contract_exposes_worst_profile_failure() -> None:
    per_profile = {
        "vertical": _profile(),
        "lateral": _profile(),
    }
    per_profile["lateral"]["horizon_rollouts"]["2s"] = _metrics(0.21, 4.0)

    result = evaluate_multirotor_accuracy(
        state_source="ground_truth",
        aggregate_full_rollout=_metrics(0.05, 1.0),
        aggregate_horizon_rollouts={
            label: metrics
            for label, metrics in per_profile["vertical"]["horizon_rollouts"].items()
        },
        per_profile=per_profile,
    )

    assert result["status"] == "fail"
    worst = result["horizon_rollouts"]["2s"]["worst_profile"]["position_rmse_m"]
    assert worst["profile"] == "lateral"
    assert worst["passed"] is False


def test_accuracy_contract_marks_missing_required_horizons_incomplete() -> None:
    profile = _profile()
    profile["horizon_rollouts"] = {"0.1s": _metrics(0.0005, 0.1)}

    result = evaluate_multirotor_accuracy(
        state_source="ground_truth",
        aggregate_full_rollout=_metrics(0.05, 1.0),
        aggregate_horizon_rollouts=profile["horizon_rollouts"],
        per_profile={"vertical": profile},
    )

    assert result["status"] == "incomplete"
    assert result["passed"] is None
    assert result["missing_required_horizons_s"] == [0.5, 1.0, 2.0]


def test_accuracy_contract_does_not_score_unknown_state_source() -> None:
    result = evaluate_multirotor_accuracy(
        state_source=None,
        aggregate_full_rollout={},
        aggregate_horizon_rollouts={},
        per_profile={},
    )

    assert result["status"] == "not_scored"
    assert result["passed"] is None


def _fixedwing_airframe(
    *,
    persistence_score: float = 0.8,
    finite_fraction: float = 1.0,
) -> dict[str, object]:
    horizons = {
        "0.5s": _metrics(0.20, 5.0),
        "1s": _metrics(0.50, 8.0),
        "2s": _metrics(1.00, 12.0),
    }
    return {
        "aggregate_horizon_rollouts": horizons,
        "p90_horizon_rollouts": horizons,
        "score_vs_kinematic_persistence": persistence_score,
        "full_rollout_finite_fraction": finite_fraction,
    }


def test_fixedwing_contract_requires_every_airframe_to_pass() -> None:
    result = evaluate_fixedwing_accuracy(
        {
            "conventional": _fixedwing_airframe(),
            "flying_wing": _fixedwing_airframe(),
        }
    )

    assert result["contract_id"] == "fixedwing_prediction_development_v1"
    assert result["status"] == "pass"
    assert result["passed"] is True


def test_fixedwing_contract_rejects_p90_persistence_and_instability() -> None:
    regressed = _fixedwing_airframe(
        persistence_score=1.02,
        finite_fraction=0.75,
    )
    regressed["p90_horizon_rollouts"] = {
        **regressed["p90_horizon_rollouts"],
        "2s": _metrics(1.3, 12.0),
    }

    result = evaluate_fixedwing_accuracy({"flying_wing": regressed})

    assert result["status"] == "fail"
    airframe = result["airframes"]["flying_wing"]
    assert (
        airframe["horizon_rollouts"]["2s"]["p90"]["position_rmse_m"]["passed"] is False
    )
    assert airframe["persistence"]["passed"] is False
    assert airframe["full_rollout_finiteness"]["passed"] is False
