"""Evidence boundary tests with tiny deterministic dynamics, no flight trials."""

import json
from copy import deepcopy

import numpy as np
import pytest

from glassbox.experimental.two_simulator_flight import (
    PROTOCOL,
    achieved,
    array_equal,
    freeze_files,
    make_queries,
    query_roster,
    roles_for,
    validity,
    verify_files,
    write_json,
)


def recording(n=60):
    states = np.zeros((n + 1, 13))
    states[:, 2], states[:, 6] = 4, 1
    return dict(
        states=states,
        full_states=np.c_[states, np.zeros((n + 1, 2))],
        commands=np.zeros((n, 3)),
        time_s=np.arange(n + 1) * 0.05,
    )


def test_validity_preserves_prefix_and_hidden_failures():
    a = recording()
    assert validity(a)["valid_transitions"] == 60
    a["full_states"][12, -1] = np.nan
    assert validity(a) == dict(
        valid_transitions=11,
        valid_initial=True,
        failure_index=12,
        reason="nonfinite_state",
    )
    a = recording()
    a["commands"][12, 1] = np.nan
    assert validity(a)["valid_transitions"] == 12
    a = recording()
    a["states"][0, 2] = 0.49
    assert not validity(a)["valid_initial"]
    a = recording()
    a["states"][7, 6] = 1.01
    assert validity(a)["valid_transitions"] == 6


def test_roles_and_all_planned_queries_are_frozen():
    p = json.loads(PROTOCOL.read_text())
    assert len(p["recordings"]) == len({r["id"] for r in p["recordings"]}) == 360
    for sim, width in (("crazyflow", 4), ("cascade", 3)):
        names = [
            r["id"]
            for r in p["recordings"]
            if r["simulator"] == sim and r["role"] == "calibration_pool"
        ]
        assert roles_for(names) == p["planned_automatic_roles"][sim]
        entry = next(
            r for r in p["recordings"] if r["simulator"] == sim and r["role"] == "test"
        )
        queries = query_roster(entry, p, p["generation"][sim]["dt_s"], width)
        assert len(queries) == 10 + 4 * width
        assert len({q["id"] for q in queries}) == len(queries)
    assert roles_for(["a"]) == {"development": [], "training": ["a"]}


def test_failed_parent_keeps_every_query_and_denominator():
    p = json.loads(PROTOCOL.read_text())

    class Fixture:
        dt = 0.05
        lower = np.array([0, -0.35, -0.35])
        upper = np.array([1, 0.35, 0.35])

    entry = next(
        r
        for r in p["recordings"]
        if r["simulator"] == "cascade" and r["role"] == "test"
    )
    queries, arrays, branches = make_queries(
        Fixture(), None, entry, p["cells"]["cascade"][0], p
    )
    assert len(queries) == 22 and not branches
    assert all(not q["history_eligible"] for q in queries)
    assert all(not a["valid"].any() for a in arrays.values())


def test_array_replay_checks_hidden_state_dtype_and_nan_bytes():
    a = recording()
    array_equal(a, deepcopy(a), "self")
    b = deepcopy(a)
    b["full_states"][10, -1] = 1
    with pytest.raises(ValueError, match="full_states"):
        array_equal(a, b, "altered")
    b = deepcopy(a)
    b["commands"] = b["commands"].astype(np.float32)
    with pytest.raises(ValueError, match="commands"):
        array_equal(a, b, "dtype")


def test_seal_rejects_raw_mutation(tmp_path):
    write_json(tmp_path / "x.json", {"value": 1})
    freeze_files(tmp_path, "seal.json", {"scope": "test"})
    verify_files(tmp_path, "seal.json")
    write_json(tmp_path / "x.json", {"value": 2})
    with pytest.raises(ValueError, match="altered artifact"):
        verify_files(tmp_path, "seal.json")
    with pytest.raises(FileExistsError):
        freeze_files(tmp_path, "seal.json", {})


def test_achieved_preserves_wind_alignment_after_nonfinite_state():
    arrays = recording(3)
    arrays["states"][:, 3] = [3, np.nan, 10, 20]
    arrays["wind"] = np.asarray([[1, 0, 0], [99, 0, 0], [5, 0, 0], [7, 0, 0]])
    result = achieved(arrays)
    prefix, diagnostic = result["valid_prefix"], result["all_finite_diagnostic"]
    assert prefix["finite_boundaries"] == 1
    assert prefix["ground_speed_m_s"] == [3, 3]
    assert prefix["airspeed_m_s"] == [2, 2]
    assert diagnostic["finite_boundaries"] == diagnostic["finite_wind_boundaries"] == 3
    assert diagnostic["airspeed_m_s"] == [2, 13]
    assert diagnostic["body_forward_airspeed_m_s"] == [2, 13]
    json.dumps(result, allow_nan=False)


def test_achieved_excludes_finite_invalid_tail_from_valid_conditions():
    arrays = recording(3)
    arrays["states"][:, 3] = [3, 4, 100, 1000]
    arrays["states"][:, 10:13] = [[1, -2, 3], [2, -1, 4], [50, 60, 70], [80, 90, 100]]
    arrays["full_states"][2, -1] = np.nan
    arrays["wind"] = np.asarray([[1, 0, 0], [np.nan, 0, 0], [5, 0, 0], [7, 0, 0]])
    result = achieved(arrays)
    prefix, diagnostic = result["valid_prefix"], result["all_finite_diagnostic"]
    assert result["validity"]["failure_index"] == 2
    assert prefix["finite_boundaries"] == 2
    assert prefix["ground_speed_m_s"] == [3, 4]
    assert prefix["finite_wind_boundaries"] == prefix["airspeed_boundaries"] == 1
    assert prefix["airspeed_m_s"] == [2, 2]
    assert prefix["body_rate_rad_s"] == {"x": [1, 2], "y": [-2, -1], "z": [3, 4]}
    assert diagnostic["finite_boundaries"] == 4
    assert diagnostic["finite_full_state_boundaries"] == 3
    assert diagnostic["ground_speed_m_s"] == [3, 1000]
    assert diagnostic["airspeed_m_s"] == [2, 993]
    assert diagnostic["body_rate_rad_s"]["z"] == [3, 100]


def test_achieved_physical_angles_forward_speed_and_circular_heading():
    from scipy.spatial.transform import Rotation

    arrays = recording(1)
    rotations = Rotation.from_euler("ZYX", [[179, -10, 20], [-179, -10, 20]], degrees=True)
    xyzw = rotations.as_quat()
    arrays["states"][:, 6:10] = xyzw[:, [3, 0, 1, 2]]
    arrays["wind"] = np.repeat([[2.0, -1.0, 0.25]], 2, axis=0)
    arrays["states"][:, 3:6] = 5 * rotations.as_matrix()[:, :, 0] + arrays["wind"]
    scope = achieved(arrays)["valid_prefix"]
    np.testing.assert_allclose(scope["airspeed_m_s"], [5, 5], atol=1e-12)
    np.testing.assert_allclose(scope["body_forward_airspeed_m_s"], [5, 5], atol=1e-12)
    np.testing.assert_allclose(scope["pitch_deg"], [10, 10], atol=1e-12)
    np.testing.assert_allclose(scope["bank_deg"], [20, 20], atol=1e-12)
    np.testing.assert_allclose(scope["heading_wrapped_deg"], [-179, 179], atol=1e-12)
    assert scope["heading_circular_arc"]["start_deg"] == pytest.approx(179)
    assert scope["heading_circular_arc"]["width_deg"] == pytest.approx(2)


def test_achieved_empty_valid_prefix_and_missing_wind_remain_explicit():
    arrays = recording(1)
    arrays["states"][0, 2] = 0.0
    result = achieved(arrays)
    prefix, diagnostic = result["valid_prefix"], result["all_finite_diagnostic"]
    assert prefix["finite_boundaries"] == prefix["requested_boundaries"] == 0
    assert prefix["ground_speed_m_s"] is None
    assert prefix["max_body_rate_rad_s"] is None
    assert prefix["heading_circular_arc"] is None
    assert diagnostic["finite_boundaries"] == 2
    assert diagnostic["finite_wind_boundaries"] == 0
    assert diagnostic["airspeed_m_s"] is None
    json.dumps(result, allow_nan=False)
    arrays["wind"] = np.zeros((1, 3))
    with pytest.raises(ValueError, match="state-aligned"):
        achieved(arrays)


def test_achieved_does_not_invent_heading_at_vertical_or_zero_attitude():
    arrays = recording(2)
    arrays["wind"] = np.zeros((3, 3))
    arrays["states"][1, 6:10] = [np.sqrt(0.5), 0, -np.sqrt(0.5), 0]
    arrays["states"][2, 6:10] = 0
    result = achieved(arrays)
    scope = result["all_finite_diagnostic"]
    assert scope["finite_boundaries"] == 3
    assert scope["attitude_boundaries"] == 2
    assert scope["heading_bank_boundaries"] == 1
    np.testing.assert_allclose(scope["pitch_deg"], [0, 90], atol=1e-12)
    assert scope["heading_wrapped_deg"] == [0, 0]
    json.dumps(result, allow_nan=False)


def test_achieved_command_failure_still_includes_its_valid_origin_state():
    arrays = recording(3)
    arrays["states"][:, 3] = [1, 2, 50, 100]
    arrays["commands"][1, 0] = np.nan
    result = achieved(arrays)
    assert result["valid_prefix"]["finite_boundaries"] == 2
    assert result["valid_prefix"]["ground_speed_m_s"] == [1, 2]
    assert result["all_finite_diagnostic"]["ground_speed_m_s"] == [1, 100]


def test_response_keeps_early_horizons_before_invalid_command_tail():
    p = json.loads(PROTOCOL.read_text())

    class Fixture:
        dt = 0.05
        lower = np.array([0, -0.35, -0.35])
        upper = np.array([1, 0.35, 0.35])

        def branch(self, full_state, commands, origin_index, cell):
            assert np.isfinite(commands).all()
            a = recording(len(commands))
            a["commands"] = commands.copy()
            a["states"][:, 3] = np.r_[0, np.cumsum(commands[:, 0])]
            a["full_states"][:, :13] = a["states"]
            return a, {}

    parent = recording()
    parent["commands"][23:] = np.nan
    entry = next(
        r
        for r in p["recordings"]
        if r["simulator"] == "cascade" and r["role"] == "test"
    )
    _, arrays, branches = make_queries(
        Fixture(), parent, entry, p["cells"]["cascade"][0], p
    )
    q = arrays["response-0020-0-+1"]
    np.testing.assert_array_equal(q["valid"], [True, True, True, False, False])
    assert len(branches["base-0020"][0]["commands"]) == 3
    assert np.isnan(q["target"][3:]).all()
