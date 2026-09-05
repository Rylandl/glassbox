"""Pinned metric values for :mod:`glassbox.core.metrics`.

The table records the maintained prediction and persistence entry points on
fixed synthetic flights. It was regenerated after the actuator quadrature
correction, which intentionally changes those flights and their predictions.
Exact comparisons continue to pin metric reductions and both floor tables.
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
            "position_rmse_m": 1.3834691234246512e-06,
            "velocity_rmse_m_s": 4.8961669340316843e-08,
            "attitude_rmse_deg": 1.1496921542081826e-06,
            "angular_velocity_rmse_rad_s": 1.162951402001418e-08,
            "final_position_error_m": 3.8200054762019155e-06,
            "duration_s": 3.0,
            "sample_count": 150,
            "rollout_count": 1,
        },
        "windowed_stride_horizon": {
            "position_rmse_m": 4.601265271780735e-08,
            "velocity_rmse_m_s": 5.6684292610203985e-09,
            "attitude_rmse_deg": 1.0977993164519886e-06,
            "angular_velocity_rmse_rad_s": 4.603459736633104e-09,
            "final_position_error_m": 1.376888843606399e-07,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "windowed_stride_one": {
            "position_rmse_m": 5.8774095111056867e-08,
            "velocity_rmse_m_s": 2.903834430800336e-09,
            "attitude_rmse_deg": 1.1796976562652626e-06,
            "angular_velocity_rmse_rad_s": 4.098885511225258e-09,
            "final_position_error_m": 1.5027192458996387e-07,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "persistence_stride_horizon": {
            "position_rmse_m": 0.0016142154508552045,
            "velocity_rmse_m_s": 0.020038601133167654,
            "attitude_rmse_deg": 0.06017340587995982,
            "angular_velocity_rmse_rad_s": 0.007505426706771022,
            "final_position_error_m": 0.005569163297040713,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "persistence_stride_one": {
            "position_rmse_m": 0.0015978283798525545,
            "velocity_rmse_m_s": 0.01987680092504873,
            "attitude_rmse_deg": 0.06019201882321496,
            "angular_velocity_rmse_rad_s": 0.007520851487005647,
            "final_position_error_m": 0.00551749698323762,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "aggregate_sample": {
            "position_rmse_m": 5.7669874666143925e-08,
            "velocity_rmse_m_s": 3.2727646755120463e-09,
            "attitude_rmse_deg": 1.1720715144998839e-06,
            "angular_velocity_rmse_rad_s": 4.150068910776873e-09,
            "final_position_error_m": 1.4910816649596916e-07,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "aggregate_equal": {
            "position_rmse_m": 5.2780481554515503e-08,
            "velocity_rmse_m_s": 4.503517774401128e-09,
            "attitude_rmse_deg": 1.1394845105134618e-06,
            "angular_velocity_rmse_rad_s": 4.358480468060771e-09,
            "final_position_error_m": 1.4411779938031502e-07,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "score_metric_floors": 0.16239750950192774,
        "score_negligible_floors": 2.7525038891054153e-06,
        "divergence": {
            "diverged": False,
            "divergence_time_s": None,
            "stable_through_s": 3.0,
            "stable_fraction": 1.0,
            "duration_s": 3.0,
            "full_rollout_finite": True,
        },
    },
    "multirotor_guess": {
        "full_rollout": {
            "position_rmse_m": 1.7882975884890815,
            "velocity_rmse_m_s": 1.5017415550153261,
            "attitude_rmse_deg": 1.3183136079637388,
            "angular_velocity_rmse_rad_s": 0.03387693567461553,
            "final_position_error_m": 6.783769598916668,
            "duration_s": 3.0,
            "sample_count": 150,
            "rollout_count": 1,
        },
        "windowed_stride_horizon": {
            "position_rmse_m": 0.00948341433136837,
            "velocity_rmse_m_s": 0.11671234524284545,
            "attitude_rmse_deg": 0.11033408490848692,
            "angular_velocity_rmse_rad_s": 0.013733407922356633,
            "final_position_error_m": 0.03261137599861634,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "windowed_stride_one": {
            "position_rmse_m": 0.009492252688249277,
            "velocity_rmse_m_s": 0.11680262820426629,
            "attitude_rmse_deg": 0.1100957541435938,
            "angular_velocity_rmse_rad_s": 0.013712814556760287,
            "final_position_error_m": 0.03263953855652245,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "persistence_stride_horizon": {
            "position_rmse_m": 0.004821159633568447,
            "velocity_rmse_m_s": 0.05951784415602461,
            "attitude_rmse_deg": 0.42161178532707316,
            "angular_velocity_rmse_rad_s": 0.05249082733578292,
            "final_position_error_m": 0.016583195832192213,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "persistence_stride_one": {
            "position_rmse_m": 0.00487584124610156,
            "velocity_rmse_m_s": 0.06027855306891933,
            "attitude_rmse_deg": 0.42303621226746474,
            "angular_velocity_rmse_rad_s": 0.052733134261823494,
            "final_position_error_m": 0.01678093217296309,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "aggregate_sample": {
            "position_rmse_m": 0.00949140320387964,
            "velocity_rmse_m_s": 0.11679395018293448,
            "attitude_rmse_deg": 0.11011869297786388,
            "angular_velocity_rmse_rad_s": 0.013714796031749651,
            "final_position_error_m": 0.032636831674270686,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "aggregate_equal": {
            "position_rmse_m": 0.009487834538976178,
            "velocity_rmse_m_s": 0.11675749544999073,
            "attitude_rmse_deg": 0.11021498394737408,
            "angular_velocity_rmse_rad_s": 0.013723115102445668,
            "final_position_error_m": 0.032625460316337646,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "score_metric_floors": 0.7108228718131493,
        "score_negligible_floors": 0.7108228718131493,
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
            "position_rmse_m": 7.547019727059356e-08,
            "velocity_rmse_m_s": 1.6324558430821579e-07,
            "attitude_rmse_deg": 3.0386026768501225e-06,
            "angular_velocity_rmse_rad_s": 3.3209178371584374e-08,
            "final_position_error_m": 4.0027373164025876e-07,
            "duration_s": 3.0,
            "sample_count": 150,
            "rollout_count": 1,
        },
        "windowed_stride_horizon": {
            "position_rmse_m": 1.3843769046354645e-09,
            "velocity_rmse_m_s": 1.2017241378729871e-08,
            "attitude_rmse_deg": 1.2547858734911238e-06,
            "angular_velocity_rmse_rad_s": 3.989926552812329e-08,
            "final_position_error_m": 4.883294030142505e-09,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "windowed_stride_one": {
            "position_rmse_m": 1.6189061805150009e-09,
            "velocity_rmse_m_s": 1.0744366735871224e-08,
            "attitude_rmse_deg": 1.2478450844844798e-06,
            "angular_velocity_rmse_rad_s": 3.70394680778439e-08,
            "final_position_error_m": 4.430044128695183e-09,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "persistence_stride_horizon": {
            "position_rmse_m": 0.004821159633568447,
            "velocity_rmse_m_s": 0.05951784415602461,
            "attitude_rmse_deg": 0.42161178532707316,
            "angular_velocity_rmse_rad_s": 0.05249082733578292,
            "final_position_error_m": 0.016583195832192213,
            "duration_s": 0.20000000000000018,
            "sample_count": 150,
            "rollout_count": 15,
        },
        "persistence_stride_one": {
            "position_rmse_m": 0.00487584124610156,
            "velocity_rmse_m_s": 0.06027855306891933,
            "attitude_rmse_deg": 0.42303621226746474,
            "angular_velocity_rmse_rad_s": 0.052733134261823494,
            "final_position_error_m": 0.01678093217296309,
            "duration_s": 0.20000000000000018,
            "sample_count": 1410,
            "rollout_count": 141,
        },
        "aggregate_sample": {
            "position_rmse_m": 1.5978518420294239e-09,
            "velocity_rmse_m_s": 1.0873235515434396e-08,
            "attitude_rmse_deg": 1.248514144744064e-06,
            "angular_velocity_rmse_rad_s": 3.7323971503278004e-08,
            "final_position_error_m": 4.475620883199637e-09,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "aggregate_equal": {
            "position_rmse_m": 1.506213237791671e-09,
            "velocity_rmse_m_s": 1.1398585590097427e-08,
            "attitude_rmse_deg": 1.2513202913692723e-06,
            "angular_velocity_rmse_rad_s": 3.8495932156354554e-08,
            "final_position_error_m": 4.662180367972269e-09,
            "duration_s": 0.40000000000000036,
            "sample_count": 1560,
            "rollout_count": 156,
        },
        "score_metric_floors": 0.03514241665980116,
        "score_negligible_floors": 5.917504586221745e-07,
        "divergence": {
            "diverged": False,
            "divergence_time_s": None,
            "stable_through_s": 3.0,
            "stable_fraction": 1.0,
            "duration_s": 3.0,
            "full_rollout_finite": True,
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
