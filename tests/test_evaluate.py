"""Pinned numbers for the three named scoring policies.

The values pin the named scoring conventions on deterministic fixtures. They
were regenerated after the actuator quadrature correction changed predictions
and synthetic baseline trajectories; the scoring conventions are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox import cli
from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.belief_io import save_dynamics_belief
from glassbox.core.data import (
    Channel,
    save_trajectory_npz,
    specific_force_observation_channels,
)
from glassbox.core.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.model import ExecutableModel, runtime_spec_from_trajectory
from glassbox.core.model_io import parameter_dict
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
)

PINNED_NANODRONE = {
    "model_selected_horizons": {
        "1": {
            "angular_velocity_mae_rad_s": 0.019139076960815876,
            "attitude_mae_rad": 0.00882666713560621,
            "position_mae_m": 0.3706644317638345,
            "time_s": 0.01,
            "velocity_mae_m_s": 0.05976444715072663,
        },
        "2": {
            "angular_velocity_mae_rad_s": 0.038321607260316866,
            "attitude_mae_rad": 0.0177583286834975,
            "position_mae_m": 0.7418086562111033,
            "time_s": 0.02,
            "velocity_mae_m_s": 0.11923844606578798,
        },
    },
    "model_cumulative": {
        "angular_velocity_mae_rad_s": 0.05746068422113274,
        "attitude_mae_rad": 0.02658499581910371,
        "position_mae_m": 1.1124730879749378,
        "velocity_mae_m_s": 0.1790028932165146,
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
    "score_vs_baseline": 218404657.32186955,
    "model_horizon_rollouts": {
        "0.025s": {
            "position_rmse_m": 0.0019531181008086921,
            "velocity_rmse_m_s": 0.15190138364789585,
            "attitude_rmse_deg": 1.5240335098404667,
            "angular_velocity_rmse_rad_s": 0.8599528462540712,
            "final_position_error_m": 0.003382899783783087,
            "sample_count": 3,
            "rollout_count": 3,
        },
        "0.05s": {
            "position_rmse_m": 0.005392502358289542,
            "velocity_rmse_m_s": 0.22342080697946254,
            "attitude_rmse_deg": 3.7632357681456665,
            "angular_velocity_rmse_rad_s": 1.3337590309297693,
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
        "structured": 6.296769188863287,
        "structured_residual": 3.148384594431643,
    },
    "selected_model": "structured_residual",
    "baseline_horizon_rollouts": {
        "0.2s": {
            "position_rmse_m": 0.003939168878054266,
            "velocity_rmse_m_s": 0.039840444039191784,
            "attitude_rmse_deg": 0.1344836047770969,
            "angular_velocity_rmse_rad_s": 0.013652105124748468,
        },
        "0.5s": {
            "position_rmse_m": 0.011189945157114264,
            "velocity_rmse_m_s": 0.06167312559694999,
            "attitude_rmse_deg": 0.37919738091333444,
            "angular_velocity_rmse_rad_s": 0.020979189730036193,
        },
        "1s": {
            "position_rmse_m": 0.046851934653634145,
            "velocity_rmse_m_s": 0.11474725769420675,
            "attitude_rmse_deg": 1.6088153116345778,
            "angular_velocity_rmse_rad_s": 0.03867459668658846,
        },
        "2s": {
            "position_rmse_m": 0.17764801578160733,
            "velocity_rmse_m_s": 0.17688272800344937,
            "attitude_rmse_deg": 5.220850331025655,
            "angular_velocity_rmse_rad_s": 0.05467639408315728,
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
    capsys,
) -> None:
    reports = _epfl_fit_reports(tmp_path)
    output = tmp_path / "comparison.json"
    cli.main(
        [
            "evaluate",
            "--fit-reports",
            *(f"{name}={path}" for name, path in reports.items()),
            "--horizons",
            "0.2,0.5,1,2",
            "--score-horizons",
            "0.5,1,2",
            "--report",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert f"selected={report['selected_model']}" in capsys.readouterr().out

    assert report["protocol"] == "windowed"
    assert report["stride"] == "one_horizon"
    assert report["floors"] == dict(PROTOCOLS["windowed"].floors)
    assert report["independent_holdout"] is False
    assert "can_promote_model" not in report
    assert report["selected_model"] == PINNED_EPFL["selected_model"]
    for name, score in PINNED_EPFL["scores"].items():
        assert report["models"][name]["score_vs_baseline"] == pytest.approx(
            score, rel=1e-12, abs=1e-12
        )
    for label, expected in PINNED_EPFL["baseline_horizon_rollouts"].items():
        measured = report["baseline_metrics"]["horizon_rollouts"][label]
        for metric, value in expected.items():
            assert measured[metric] == pytest.approx(value, rel=1e-12, abs=1e-12), (
                f"{label}.{metric}"
            )
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
    assert "can_promote_model" not in report
    assert report["scoring"]["definition"]


def test_unknown_protocol_names_the_ones_that_exist(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown evaluation protocol"):
        evaluate(initial_parameter_guess(), [], protocol="nope")


@pytest.fixture
def evaluation_belief(quadrotor_flight):
    flight = quadrotor_flight(0, 0.2)
    wind = Channel(
        name="wind_north_m_s",
        role="wind_north",
        semantic="world_wind_velocity",
        unit="m/s",
        kind="exogenous",
        frame="NWU",
    )
    flight = replace(
        flight,
        spec=replace(flight.spec, channels=(*flight.spec.channels, wind)),
        exogenous=np.zeros((len(flight.states), 1)),
    )
    belief = DynamicsBelief(
        model=ExecutableModel(
            initial_parameter_guess(), flight.spec, runtime_spec_from_trajectory(flight)
        )
    )
    return belief, flight


@pytest.mark.parametrize("saved", [False, True], ids=["in-memory", "saved"])
@pytest.mark.parametrize(
    "kind,attribute,value",
    [
        ("control", "role", "different_motor"),
        ("control", "semantic", "measured_rotor_speed"),
        ("control", "unit", "rad/s"),
        ("control", "frame", "FRD"),
        ("exogenous", "role", "wind_west"),
        ("exogenous", "semantic", "body_wind_velocity"),
        ("exogenous", "unit", "km/h"),
        ("exogenous", "frame", "NED"),
        ("vehicle", "family", "fixedwing"),
    ],
)
def test_evaluation_and_absorption_reject_the_same_mismatched_inputs(
    evaluation_belief, tmp_path, monkeypatch, saved, kind, attribute, value
) -> None:
    belief, flight = evaluation_belief
    model = belief
    if saved:
        model = tmp_path / "belief.json"
        belief.save(model)
    if kind == "vehicle":
        spec = replace(flight.spec, vehicle=replace(flight.spec.vehicle, family=value))
    else:
        channels = list(flight.spec.channels)
        index = next(i for i, channel in enumerate(channels) if channel.kind == kind)
        channels[index] = replace(channels[index], **{attribute: value})
        spec = replace(flight.spec, channels=tuple(channels))
    incompatible = replace(flight, spec=spec)

    def unexpected_scoring(*_args, **_kwargs):
        pytest.fail("validate all trajectories before generating predictions")

    monkeypatch.setattr(
        "glassbox.workflows.evaluate.predict_windows", unexpected_scoring
    )
    monkeypatch.setattr(
        "glassbox.belief.update.one_step_linearization", unexpected_scoring
    )
    with pytest.raises(ValueError, match=rf"trajectory_1:.*{attribute}"):
        evaluate(model, [flight, incompatible], horizons_s=(0.04,))
    update_belief = DynamicsBelief.load(model) if saved else belief
    with pytest.raises(ValueError, match=rf"belief update:.*{attribute}"):
        update_belief.absorb(incompatible)


def test_evaluation_and_absorption_allow_observation_metadata_and_exogenous_names(
    evaluation_belief,
) -> None:
    belief, flight = evaluation_belief
    observed = replace(
        flight,
        spec=replace(
            flight.spec,
            observation_source="another_estimator",
            channels=(
                *(
                    replace(channel, name=f"recorded_{channel.name}")
                    if channel.kind == "exogenous"
                    else channel
                    for channel in flight.spec.channels
                ),
                *specific_force_observation_channels(),
            ),
        ),
        observations=np.zeros((len(flight.states), 3)),
    )
    scored = evaluate(belief, [observed], horizons_s=(0.04,))
    expected = evaluate(belief.params, [flight], horizons_s=(0.04,))
    assert scored["model"] == expected["model"]
    updated, update = belief.absorb(observed)
    expected_belief, expected_update = belief.absorb(flight)
    assert update == expected_update
    assert updated.information.to_dict() == expected_belief.information.to_dict()
    assert parameter_dict(updated.params) == parameter_dict(expected_belief.params)


def test_bare_parameters_still_use_the_callers_input_semantics(
    evaluation_belief,
) -> None:
    belief, flight = evaluation_belief
    spec = replace(
        flight.spec,
        channels=tuple(
            replace(channel, semantic="measured_rotor_speed", unit="rad/s")
            if channel.kind == "control"
            else channel
            for channel in flight.spec.channels
        ),
    )
    scored = evaluate(belief.params, [replace(flight, spec=spec)], horizons_s=(0.04,))
    assert scored["model"]["horizon_rollouts"]
