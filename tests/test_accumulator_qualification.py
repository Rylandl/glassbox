"""Check physical response attribution and availability independently of fitting."""

import json

import numpy as np
from qualify_accumulator import command_lengths, compare, score_flights


def test_command_availability_stops_at_first_missing_interval():
    data = dict(
        past_states=np.zeros((3, 3, 15)),
        past_inputs=np.zeros((3, 2, 2)),
        future_inputs=np.zeros((3, 4, 2)),
    )
    data["future_inputs"][0, 1] = np.nan
    data["past_states"][1, 0] = np.nan
    np.testing.assert_array_equal(command_lengths(data, "future_inputs"), [1, 0, 4])


def test_response_errors_cancel_shared_forecast_bias(tmp_path):
    folder = tmp_path / "flights"
    folder.mkdir()
    (folder / "spec.json").write_text(
        json.dumps(
            dict(
                cohorts=dict(
                    one=dict(
                        simulator="generic", contract=dict(horizons=[1], dt_s=0.05)
                    )
                )
            )
        )
    )
    roster = [
        dict(kind="factual", scope="primary", history_eligible=True),
        dict(kind="response", scope="primary", history_eligible=True),
    ]
    (folder / "one.json").write_text(json.dumps(roster))
    truth = np.zeros((2, 1, 15))
    truth[:, :, 6:] = np.eye(3).reshape(9)
    truth[1, 0, 0] = 3
    factual = truth.copy()
    factual[1, 0, 0] = 1
    predicted = truth.copy()
    predicted[:, :, 0] += 2
    predicted_factual = factual.copy()
    predicted_factual[:, :, 0] += 2
    np.savez(
        folder / "one.npz",
        target=truth,
        factual_target=factual,
        valid=np.ones((2, 1), bool),
    )
    values = dict(
        predicted=predicted,
        predicted_factual=predicted_factual,
        lengths=np.ones(2, int),
        lengths_factual=np.ones(2, int),
    )
    rows = score_flights(tmp_path, dict(one=values))
    assert rows[0]["scores"]["velocity_rmse_m_s"] == 2
    assert rows[1]["scores"]["velocity_rmse_m_s"] == 0
    assert rows[1]["scores"]["orientation_rmse_rad"] == 0


def test_aggregate_weights_families_equally_despite_different_horizon_counts():
    base = []
    candidate = []
    for family, count, ratio in [("one", 1, 2), ("two", 4, 0.5)]:
        for kind in ["factual", "response"]:
            for h in range(count):
                row = dict(
                    family=family,
                    cohort=family,
                    kind=kind,
                    scope="primary",
                    horizon_steps=h + 1,
                    count=10,
                    scores=dict(
                        velocity_rmse_m_s=1.0,
                        body_rate_rmse_rad_s=1.0,
                        orientation_rmse_rad=1.0,
                    ),
                    tail_velocity=1.0,
                    tail_rate=1.0,
                )
                base.append(row)
                candidate.append(
                    dict(
                        row,
                        scores={k: ratio for k in row["scores"]},
                        tail_velocity=ratio,
                        tail_rate=ratio,
                    )
                )
    result = compare(candidate, base)
    assert result["aggregate"]["factual"]["primary"] == 1
    assert result["aggregate"]["response"]["tail"] == 1
