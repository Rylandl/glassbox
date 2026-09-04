"""Pinned numbers for the three named scoring policies.

The values below were produced by ``workflows/nanodrone_evaluation.py``,
``workflows/x8_evaluation.py`` and ``workflows/epfl_evaluation.py`` before the
three modules were folded into :func:`glassbox.workflows.evaluate.evaluate` as
policies. They are compared for exact equality: the whole point of keeping the
conventions named rather than unifying them is that a published comparison
stays the comparison it was.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.belief_io import save_dynamics_belief
from glassbox.core.data import save_trajectory_npz
from glassbox.core.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.model import ExecutableModel, runtime_spec_from_trajectory
from glassbox.core.synthetic import initial_parameter_guess
from glassbox.io.corpus import REFERENCE_CORPORA
from glassbox.io.nanodrone_reference import (
    SOURCE_COLUMNS,
    NanoDroneBenchmarkAdapter,
)
from glassbox.io.x8_reference import (
    X8ReferenceAdapter,
    x8_trajectory_spec,
)
from glassbox.workflows.evaluate import (
    PROTOCOLS,
    evaluate,
    evaluate_fit_reports,
)

PINNED_NANODRONE = {
    "model_selected_horizons": {
        "1": {
            "angular_velocity_mae_rad_s": 0.019139076960815876,
            "attitude_mae_rad": 0.00882666713560621,
            "position_mae_m": 0.3706644317638345,
            "time_s": 0.01,
            "velocity_mae_m_s": 0.05976445460124762,
        },
        "2": {
            "angular_velocity_mae_rad_s": 0.038321592666469964,
            "attitude_mae_rad": 0.0177583286834975,
            "position_mae_m": 0.7418086562111033,
            "time_s": 0.02,
            "velocity_mae_m_s": 0.11923846841716843,
        },
    },
    "model_cumulative": {
        "angular_velocity_mae_rad_s": 0.05746066962728584,
        "attitude_mae_rad": 0.02658499581910371,
        "position_mae_m": 1.1124730879749378,
        "velocity_mae_m_s": 0.17900292301841605,
    },
    "baseline_cumulative": {
        "angular_velocity_mae_rad_s": 0.0,
        "attitude_mae_rad": 0.0,
        "position_mae_m": 1.1224972160321824,
        "velocity_mae_m_s": 0.0,
    },
    "model_vs_baseline_cumulative": {
        "angular_velocity_mae_rad_s": 0.0,
        "attitude_mae_rad": 0.0,
        "position_mae_m": 1.0090106701596637,
        "velocity_mae_m_s": 0.0,
    },
}

PINNED_X8 = {
    "score_vs_baseline": 218404657.41271302,
    "model_horizon_rollouts": {
        "0.025s": {
            "position_rmse_m": 0.0019531181008086921,
            "velocity_rmse_m_s": 0.15190138364789585,
            "attitude_rmse_deg": 1.5240335138958447,
            "angular_velocity_rmse_rad_s": 0.8599528463388723,
            "final_position_error_m": 0.003382899783783087,
            "sample_count": 3,
            "rollout_count": 3,
        },
        "0.05s": {
            "position_rmse_m": 0.005392502358289542,
            "velocity_rmse_m_s": 0.22342080697946254,
            "attitude_rmse_deg": 3.763235772424415,
            "angular_velocity_rmse_rad_s": 1.33375903017084,
            "final_position_error_m": 0.012768369528573138,
            "sample_count": 4,
            "rollout_count": 2,
        },
    },
    "baseline_horizon_rollouts": {
        "0.025s": {
            "position_rmse_m": 3.700743415417188e-17,
            "velocity_rmse_m_s": 0.0,
            "attitude_rmse_deg": 0.53594903367606,
            "angular_velocity_rmse_rad_s": 0.0,
        },
        "0.05s": {
            "position_rmse_m": 3.204937810639273e-17,
            "velocity_rmse_m_s": 0.0,
            "attitude_rmse_deg": 0.8474098280913862,
            "angular_velocity_rmse_rad_s": 0.0,
        },
    },
}

PINNED_EPFL = {
    "scores": {
        "structured": 6.2967675763778415,
        "structured_residual": 3.148383788188921,
    },
    "selected_model": "structured_residual",
    "baseline_horizon_rollouts": {
        "0.2s": {
            "position_rmse_m": 0.0039391640290234314,
            "velocity_rmse_m_s": 0.039840475393643826,
            "attitude_rmse_deg": 0.1344833938271598,
            "angular_velocity_rmse_rad_s": 0.013652109498898052,
        },
        "0.5s": {
            "position_rmse_m": 0.011189942395923557,
            "velocity_rmse_m_s": 0.06167318101493113,
            "attitude_rmse_deg": 0.3791972505013004,
            "angular_velocity_rmse_rad_s": 0.020979199782071526,
        },
        "1s": {
            "position_rmse_m": 0.046851946053629516,
            "velocity_rmse_m_s": 0.11474736009925932,
            "attitude_rmse_deg": 1.6088144746163684,
            "angular_velocity_rmse_rad_s": 0.038674595522518077,
        },
        "2s": {
            "position_rmse_m": 0.17764807675369315,
            "velocity_rmse_m_s": 0.17688285430957118,
            "attitude_rmse_deg": 5.220851723160359,
            "angular_velocity_rmse_rad_s": 0.05467641467466178,
        },
    },
}


def _nanodrone_trajectory(tmp_path):
    data = np.zeros((4, len(SOURCE_COLUMNS)), dtype=np.float64)
    data[:, 0] = (0.0, 0.01, 0.02, 0.03)
    data[:, 1:4] = (
        (1.0, 2.0, 3.0),
        (1.1, 2.2, 3.3),
        (1.2, 2.4, 3.6),
        (1.3, 2.6, 3.9),
    )
    data[:, 4:8] = (0.0, 0.0, 0.0, 1.0)
    data[:, 8:11] = (0.1, 0.2, 0.3)
    data[:, 11:14] = (0.4, 0.5, 0.6)
    data[:, 14:18] = (
        (1000.0, 1100.0, 1200.0, 1300.0),
        (1010.0, 1110.0, 1210.0, 1310.0),
        (1020.0, 1120.0, 1220.0, 1320.0),
        (1030.0, 1130.0, 1230.0, 1330.0),
    )
    data[:, 18:21] = (0.0, 0.0, 9.81)
    source = tmp_path / "melon_20251017_run1.csv"
    np.savetxt(
        source, data, delimiter=",", header=",".join(SOURCE_COLUMNS), comments=""
    )
    return NanoDroneBenchmarkAdapter(verify_checksum=False).load(source)


def _x8_model_and_trajectory(tmp_path):
    data = np.zeros((4, 41), dtype=np.float64)
    data[:, 0] = np.arange(4) * 0.025
    data[:, 1] = (0.10, 0.11, 0.12, 0.13)
    data[:, 2] = (-0.20, -0.19, -0.18, -0.17)
    data[:, 3] = (0.40, 0.41, 0.42, 0.43)
    data[:, 13:16] = (0.1, 0.2, 0.3)
    data[:, 16:19] = (10.0, 2.0, -1.0)
    data[:, 19:22] = (10.0, 2.0, -1.0)
    data[:, 22:25] = (4.0, -3.0, -0.5)
    data[:, 32] = np.arange(4) * 0.25
    data[:, 33] = np.arange(4) * 0.05
    data[:, 34] = np.arange(4) * -0.025
    source = tmp_path / "lateral_121_1.csv"
    np.savetxt(source, data, delimiter=",")
    trajectory = X8ReferenceAdapter(verify_checksum=False).load(source)
    trajectory = replace(
        trajectory, labels={**trajectory.labels, "benchmark_split": "validation"}
    )
    trajectory_path = tmp_path / "validation.npz"
    model_path = tmp_path / "model.json"
    save_trajectory_npz(trajectory, trajectory_path)
    save_dynamics_belief(
        DynamicsBelief(
            model=ExecutableModel(
                true_fixed_wing_parameters(),
                x8_trajectory_spec(),
                runtime_spec_from_trajectory(trajectory),
            )
        ),
        model_path,
    )
    return model_path, trajectory_path


def _epfl_fit_reports(tmp_path):
    trajectory = generate_fixed_wing_trajectory(seed=5, duration_s=4.0, dt_s=0.2)
    trajectory = replace(
        trajectory, labels={**trajectory.labels, "source_group": "flight-1"}
    )
    trajectory_path = tmp_path / "segment.npz"
    save_trajectory_npz(trajectory, trajectory_path)
    metrics = {
        "position_rmse_m": 0.5,
        "velocity_rmse_m_s": 0.25,
        "attitude_rmse_deg": 2.0,
        "angular_velocity_rmse_rad_s": 0.1,
    }

    def write(name: str, model_class: str, scale: float) -> Path:
        path = tmp_path / f"{name}_report.json"
        path.write_text(
            json.dumps(
                {
                    "configuration": {"model_class": model_class},
                    "dataset": {"trajectory_count": 1},
                    "split": {
                        "mode": "leave_complete_flights_out",
                        "independent_source_group_holdout": False,
                        "training_flights": [{"path": str(trajectory_path)}],
                        "validation_flights": [{"path": str(trajectory_path)}],
                    },
                    "models": {
                        "learned_lag": {
                            "fit": {
                                "initial_loss": 1.0,
                                "final_loss": 0.1,
                                "loss_reduction": 10.0,
                                "wall_time_s": 1.0,
                            },
                            "validation": {
                                "aggregate": {
                                    "horizon_rollouts": {
                                        label: {
                                            key: value * scale
                                            for key, value in metrics.items()
                                        }
                                        for label in ("0.2s", "0.5s", "1s", "2s")
                                    },
                                    "full_rollout": {"position_rmse_m": scale},
                                }
                            },
                        }
                    },
                }
            )
        )
        return path

    return {
        "structured": write("structured", "structured", 2.0),
        "structured_residual": write("residual", "structured_residual", 1.0),
    }


def test_nanodrone_policy_reproduces_the_published_protocol(tmp_path) -> None:
    trajectory = _nanodrone_trajectory(tmp_path)

    report = evaluate(
        initial_parameter_guess(),
        [trajectory],
        protocol="nanodrone",
        maximum_horizon_steps=2,
    )

    assert report["protocol"] == "nanodrone"
    assert report["baseline"] == "hold_state"
    assert report["stride"] == "one_sample"
    assert report["floors"] is None
    assert report["score_vs_baseline"] is None
    for step, expected in PINNED_NANODRONE["model_selected_horizons"].items():
        assert report["model"]["selected_horizons"][step] == expected
    assert (
        report["model"]["cumulative_simulation_error"]
        == PINNED_NANODRONE["model_cumulative"]
    )
    assert (
        report["baseline_metrics"]["cumulative_simulation_error"]
        == PINNED_NANODRONE["baseline_cumulative"]
    )
    assert (
        report["model_vs_baseline"]["cumulative_simulation_error"]
        == PINNED_NANODRONE["model_vs_baseline_cumulative"]
    )


def test_x8_policy_reproduces_the_campaign_protocol(tmp_path) -> None:
    model_path, trajectory_path = _x8_model_and_trajectory(tmp_path)
    _, trajectories = REFERENCE_CORPORA["x8"].load_evaluation_trajectories(
        [trajectory_path]
    )

    report = evaluate(model_path, trajectories, protocol="x8", horizons_s=(0.025, 0.05))

    assert report["protocol"] == "x8"
    assert report["stride"] == "one_sample"
    assert report["floors"] == dict(PROTOCOLS["x8"].floors)
    assert report["score_vs_baseline"] == PINNED_X8["score_vs_baseline"]
    for label, expected in PINNED_X8["model_horizon_rollouts"].items():
        measured = report["model"]["horizon_rollouts"][label]
        for name, value in expected.items():
            assert measured[name] == value, f"{label}.{name}"
    for label, expected in PINNED_X8["baseline_horizon_rollouts"].items():
        measured = report["baseline_metrics"]["horizon_rollouts"][label]
        for name, value in expected.items():
            assert measured[name] == value, f"baseline.{label}.{name}"


def test_windowed_policy_reproduces_the_same_flight_characterization(
    tmp_path,
) -> None:
    report = evaluate_fit_reports(
        _epfl_fit_reports(tmp_path),
        protocol="windowed",
        # TOPOPlane2 samples at 5 Hz, so the campaign's 0.2-second horizon is
        # one sample and the score is taken over the three longer horizons.
        horizons_s=(0.2, 0.5, 1.0, 2.0),
        score_horizons_s=(0.5, 1.0, 2.0),
    )

    assert report["protocol"] == "windowed"
    assert report["stride"] == "one_horizon"
    assert report["floors"] == dict(PROTOCOLS["windowed"].floors)
    assert report["independent_holdout"] is False
    assert report["can_promote_model"] is False
    assert report["selected_model"] == PINNED_EPFL["selected_model"]
    for name, score in PINNED_EPFL["scores"].items():
        assert report["models"][name]["score_vs_baseline"] == score
    for label, expected in PINNED_EPFL["baseline_horizon_rollouts"].items():
        measured = report["baseline_metrics"]["horizon_rollouts"][label]
        for metric, value in expected.items():
            assert measured[metric] == value, f"{label}.{metric}"
    assert report["scoring"]["requested_and_effective_horizons"]["0.5s"][
        "effective_s"
    ] == pytest.approx(0.4)


def test_every_report_says_which_convention_produced_it(tmp_path) -> None:
    trajectory = _nanodrone_trajectory(tmp_path)

    report = evaluate(
        initial_parameter_guess(),
        [trajectory],
        protocol="nanodrone",
        maximum_horizon_steps=2,
        independent_holdout=False,
    )

    assert set(PROTOCOLS) == {"windowed", "x8", "nanodrone"}
    for key in ("protocol", "baseline", "stride", "floors", "independent_holdout"):
        assert key in report
    assert report["can_promote_model"] is False
    assert report["scoring"]["definition"]


def test_unknown_protocol_names_the_ones_that_exist(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown evaluation protocol"):
        evaluate(initial_parameter_guess(), [], protocol="nope")
