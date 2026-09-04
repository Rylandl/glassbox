"""Pinned metric values for :mod:`glassbox.core.metrics`.

The table below was produced by the five prediction entry points that
``core/evaluation.py`` carried before they were folded into :func:`predict` and
:func:`predict_windows`, on two synthetic flights the repository generates
deterministically. Every value is compared for exact equality, not tolerance:
the fold was a move, so a metric that shifts in its last bit is a regression,
not a rounding difference. Both persistence-score reductions and both floor
tables are exercised, because recorded reports pin the reduction that produced
them.
"""

from __future__ import annotations

import pytest

from glassbox.core.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.metrics import (
    METRIC_FLOORS,
    NEGLIGIBLE_METRIC_FLOORS,
    ROLLOUT_METRICS,
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
    persistence_score,
    predict,
    predict_windows,
    rollout_divergence_metrics,
    rollout_metrics,
)
from glassbox.core.synthetic import (
    generate_trajectory,
    initial_parameter_guess,
    true_parameters,
)

HORIZON_STEPS = 10
PINNED = {
    "fixedwing_true": {
        "full_rollout": {
            "position_rmse_m": 1.3826526692484636e-06,
            "velocity_rmse_m_s": 1.5159950841361098e-08,
            "attitude_rmse_deg": 1.2074182697257333e-06,
            "angular_velocity_rmse_rad_s": 9.784512465847252e-09,
            "final_position_error_m": 3.815179722055421e-06,
            "duration_s": 3.0,
            "sample_count": 150,
            "rollout_count": 1,
        },
        "windowed_stride_horizon": {
            "position_rmse_m": 4.600132364400645e-08,
            "velocity_rmse_m_s": 5.728561236882178e-09,
            "attitude_rmse_deg": 1.1664768691218519e-06,
            "angular_velocity_rmse_rad_s": 6.738354198866855e-09,
            "final_position_error_m": 1.3765776906096174e-07,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "windowed_stride_one": {
            "position_rmse_m": 5.87679454691069e-08,
            "velocity_rmse_m_s": 3.431205065160371e-09,
            "attitude_rmse_deg": 1.2692054062054864e-06,
            "angular_velocity_rmse_rad_s": 6.20990176455663e-09,
            "final_position_error_m": 1.502555917891035e-07,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "persistence_stride_horizon": {
            "position_rmse_m": 0.001614216293519069,
            "velocity_rmse_m_s": 0.02003860549868972,
            "attitude_rmse_deg": 0.06017334842297408,
            "angular_velocity_rmse_rad_s": 0.0075054287278479275,
            "final_position_error_m": 0.005569165386753653,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "persistence_stride_one": {
            "position_rmse_m": 0.0015978287897053768,
            "velocity_rmse_m_s": 0.01987680489721458,
            "attitude_rmse_deg": 0.060191950423183455,
            "angular_velocity_rmse_rad_s": 0.007520853020559245,
            "final_position_error_m": 0.005517498364868825,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "aggregate_sample": {
            "position_rmse_m": 5.766334080312463e-08,
            "velocity_rmse_m_s": 3.714371710472684e-09,
            "attitude_rmse_deg": 1.2596917548674936e-06,
            "angular_velocity_rmse_rad_s": 6.262652490148336e-09,
            "final_position_error_m": 1.4909052628915462e-07,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "aggregate_equal": {
            "position_rmse_m": 5.2772119493443456e-08,
            "velocity_rmse_m_s": 4.7217360178165e-09,
            "attitude_rmse_deg": 1.2189238387482526e-06,
            "angular_velocity_rmse_rad_s": 6.4795176222783024e-09,
            "final_position_error_m": 1.4409442086134837e-07,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "score_metric_floors": 0.1623975288317861,
        "score_negligible_floors": 3.2425028973587664e-06,
        "divergence": {
            "diverged": 0,
            "divergence_time_s": None,
            "stable_through_s": 3.0,
            "stable_fraction": 1.0,
            "duration_s": 3.0,
            "full_rollout_finite": 1,
        },
    },
    "multirotor_guess": {
        "full_rollout": {
            "position_rmse_m": 1.788298109949699,
            "velocity_rmse_m_s": 1.5017418429363598,
            "attitude_rmse_deg": 1.3182781116852216,
            "angular_velocity_rmse_rad_s": 0.033875825281970896,
            "final_position_error_m": 6.783771072826723,
            "duration_s": 3.0,
            "sample_count": 150,
            "rollout_count": 1,
        },
        "windowed_stride_horizon": {
            "position_rmse_m": 0.009483406989124095,
            "velocity_rmse_m_s": 0.11671230715309192,
            "attitude_rmse_deg": 0.11032481444207726,
            "angular_velocity_rmse_rad_s": 0.013732853190638099,
            "final_position_error_m": 0.0326113533100292,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "windowed_stride_one": {
            "position_rmse_m": 0.009492246558428383,
            "velocity_rmse_m_s": 0.11680259500750588,
            "attitude_rmse_deg": 0.11008635290759194,
            "angular_velocity_rmse_rad_s": 0.0137122591409879,
            "final_position_error_m": 0.03263951961511795,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "persistence_stride_horizon": {
            "position_rmse_m": 0.00482116430396733,
            "velocity_rmse_m_s": 0.05951789295465824,
            "attitude_rmse_deg": 0.42161180690291444,
            "angular_velocity_rmse_rad_s": 0.05249084621990437,
            "final_position_error_m": 0.016583205826339473,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "persistence_stride_one": {
            "position_rmse_m": 0.004875844970780752,
            "velocity_rmse_m_s": 0.06027860141747524,
            "attitude_rmse_deg": 0.4230361978093624,
            "angular_velocity_rmse_rad_s": 0.05273315053815951,
            "final_position_error_m": 0.01678094401205716,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "aggregate_sample": {
            "position_rmse_m": 0.009491396957577938,
            "velocity_rmse_m_s": 0.11679391651602357,
            "attitude_rmse_deg": 0.11010930434237863,
            "angular_velocity_rmse_rad_s": 0.013714240681895357,
            "final_position_error_m": 0.03263681237284183,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "aggregate_equal": {
            "position_rmse_m": 0.009487827803226701,
            "velocity_rmse_m_s": 0.11675745980768232,
            "attitude_rmse_deg": 0.11020564817234563,
            "angular_velocity_rmse_rad_s": 0.013722560029113103,
            "final_position_error_m": 0.03262543950215247,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "score_metric_floors": 0.710800006825542,
        "score_negligible_floors": 0.710800006825542,
        "divergence": {
            "diverged": False,
            "divergence_time_s": None,
            "stable_through_s": 3.0,
            "stable_fraction": 1.0,
            "duration_s": 3.0,
            "full_rollout_finite": True,
        },
    },
    "multirotor_true": {
        "full_rollout": {
            "position_rmse_m": 3.548955100806527e-08,
            "velocity_rmse_m_s": 4.7478886720284864e-08,
            "attitude_rmse_deg": 2.164390567386159e-06,
            "angular_velocity_rmse_rad_s": 3.5738067961990456e-08,
            "final_position_error_m": 1.301187962038292e-07,
            "duration_s": 3.0,
            "sample_count": 150,
            "rollout_count": 1,
        },
        "windowed_stride_horizon": {
            "position_rmse_m": 7.301181112057546e-09,
            "velocity_rmse_m_s": 9.898398898930652e-09,
            "attitude_rmse_deg": 1.2313298636799902e-06,
            "angular_velocity_rmse_rad_s": 4.125591436135174e-08,
            "final_position_error_m": 1.7527278323982464e-08,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "windowed_stride_one": {
            "position_rmse_m": 7.664106519915512e-09,
            "velocity_rmse_m_s": 1.295312895272467e-08,
            "attitude_rmse_deg": 1.211692330063465e-06,
            "angular_velocity_rmse_rad_s": 3.796258243561385e-08,
            "final_position_error_m": 1.827713319272417e-08,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "persistence_stride_horizon": {
            "position_rmse_m": 0.00482116430396733,
            "velocity_rmse_m_s": 0.05951789295465824,
            "attitude_rmse_deg": 0.42161180690291444,
            "angular_velocity_rmse_rad_s": 0.05249084621990437,
            "final_position_error_m": 0.016583205826339473,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "persistence_stride_one": {
            "position_rmse_m": 0.004875844970780752,
            "velocity_rmse_m_s": 0.06027860141747524,
            "attitude_rmse_deg": 0.4230361978093624,
            "angular_velocity_rmse_rad_s": 0.05273315053815951,
            "final_position_error_m": 0.01678094401205716,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "aggregate_sample": {
            "position_rmse_m": 7.629960025275951e-09,
            "velocity_rmse_m_s": 1.2691394978571505e-08,
            "attitude_rmse_deg": 1.2135943625515461e-06,
            "angular_velocity_rmse_rad_s": 3.829155927587609e-08,
            "final_position_error_m": 1.8206373842287598e-08,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "aggregate_equal": {
            "position_rmse_m": 7.484843832027403e-09,
            "velocity_rmse_m_s": 1.152739889195012e-08,
            "attitude_rmse_deg": 1.2215505589055272e-06,
            "angular_velocity_rmse_rad_s": 3.964346185042349e-08,
            "final_position_error_m": 1.790613139668819e-08,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "score_metric_floors": 0.035142400490184626,
        "score_negligible_floors": 9.135398691177538e-07,
        "divergence": {
            "diverged": 0,
            "divergence_time_s": None,
            "stable_through_s": 3.0,
            "stable_fraction": 1.0,
            "duration_s": 3.0,
            "full_rollout_finite": 1,
        },
    },
}


def _flight(case: str):
    if case == "fixedwing_true":
        return (
            true_fixed_wing_parameters(),
            generate_fixed_wing_trajectory(seed=3, duration_s=3.0, dt_s=0.02),
        )
    trajectory = generate_trajectory(seed=7, duration_s=3.0, dt_s=0.02)
    if case == "multirotor_true":
        return true_parameters(), trajectory
    return initial_parameter_guess(), trajectory


def _measured(case: str) -> dict[str, dict]:
    params, trajectory = _flight(case)
    windowed_stride_horizon = rollout_metrics(
        predict_windows(params, trajectory, horizon_steps=HORIZON_STEPS)
    )
    windowed_stride_one = rollout_metrics(
        predict_windows(params, trajectory, horizon_steps=HORIZON_STEPS, stride=1)
    )
    persistence_stride_one = kinematic_persistence_windowed_metrics(
        trajectory, horizon_steps=HORIZON_STEPS, stride=1
    )
    label = f"{HORIZON_STEPS}s"
    both = [windowed_stride_horizon, windowed_stride_one]
    return {
        "full_rollout": rollout_metrics(predict(params, trajectory)),
        "windowed_stride_horizon": windowed_stride_horizon,
        "windowed_stride_one": windowed_stride_one,
        "persistence_stride_horizon": kinematic_persistence_windowed_metrics(
            trajectory, horizon_steps=HORIZON_STEPS
        ),
        "persistence_stride_one": persistence_stride_one,
        "aggregate_sample": aggregate_rollout_metrics(both, weighting="sample"),
        "aggregate_equal": aggregate_rollout_metrics(both, weighting="equal"),
        "score_metric_floors": persistence_score(
            {label: windowed_stride_one},
            {label: persistence_stride_one},
            horizons=None,
            floors=METRIC_FLOORS,
            metrics=ROLLOUT_METRICS,
        ),
        "score_negligible_floors": persistence_score(
            {label: windowed_stride_one},
            {label: persistence_stride_one},
            horizons=None,
            floors=NEGLIGIBLE_METRIC_FLOORS,
            metrics=ROLLOUT_METRICS,
        ),
        "divergence": rollout_divergence_metrics(params, trajectory),
    }


@pytest.mark.parametrize("case", sorted(PINNED))
def test_metrics_reproduce_the_pinned_values_exactly(case: str) -> None:
    measured = _measured(case)

    for section, expected in PINNED[case].items():
        if not isinstance(expected, dict):
            assert measured[section] == expected, section
            continue
        for name, value in expected.items():
            assert measured[section][name] == value, f"{section}.{name}"


def test_divergence_report_carries_only_thresholds_and_the_first_crossing() -> None:
    params, trajectory = _flight("multirotor_guess")

    report = rollout_divergence_metrics(params, trajectory)

    assert set(report) == {
        "thresholds",
        "full_rollout_finite",
        "diverged",
        "divergence_time_s",
        "divergence_causes",
        "stable_through_s",
        "stable_fraction",
        "duration_s",
    }


def test_prediction_carries_the_measured_start_and_the_window_shape() -> None:
    params, trajectory = _flight("multirotor_true")

    complete = predict(params, trajectory)
    windows = predict_windows(params, trajectory, horizon_steps=HORIZON_STEPS)

    assert complete.predicted.shape == trajectory.states.shape
    assert complete.predicted[0] == pytest.approx(trajectory.states[0])
    assert complete.duration_s == pytest.approx(
        float(trajectory.time_s[-1] - trajectory.time_s[0])
    )
    assert windows.predicted.shape == windows.target.shape
    assert windows.predicted.shape[1] == HORIZON_STEPS + 1
    assert windows.duration_s == pytest.approx(HORIZON_STEPS * windows.dt_s)
    assert windows.endpoint_tangent_errors().shape == (windows.predicted.shape[0], 12)
