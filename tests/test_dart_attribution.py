"""Saved-array attribution and alignment on synthetic trajectories only."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
dart = importlib.import_module("evaluate_dart")
sys.path.pop(0)


def fixture():
    target = dict(
        center_m=[0, 0, 0],
        normal=[-1, 0, 0],
        body_contact_offset_m=[0, 0, 0],
        radius_m_max=0.02,
        axis_error_deg_max=5,
        normal_speed_m_s_range=[0.2, 2.5],
        tangential_speed_m_s_max=0.2,
    )
    spec = dict(
        target=target,
        controller=dict(dt_s=0.01, steps=120),
        contact_comparison=dict(
            exact=["contact", "hit", "reason", "checks", "interval"],
            rtol=1e-5,
            angle_atol_deg=0.05,
            fraction_atol=5e-4,
            time_atol_s=5e-6,
            speed_atol_m_s=1e-4,
            state_atol=1e-4,
            length_atol_m=1e-5,
        ),
    )
    states = np.zeros((241, 17), dtype=np.float32)
    states[:, 0] = np.arange(241) * 0.01 - 1.195
    states[:, 1] = 0.005
    states[:, 3] = 1
    states[:, 6:10] = [np.sqrt(0.5), 0, np.sqrt(0.5), 0]
    command = np.full((120, 4), 0.5, dtype=np.float64)
    origins = np.arange(0, 120, 3, dtype=np.int64)
    data = dict(
        trajectory=dict(
            states=states[:121], commands=command, time_s=np.arange(121) * 0.01
        ),
        selected=dict(
            origin=origins,
            inputs_commands=np.repeat(command[None].astype(np.float32), 40, axis=0),
            result_states=np.stack([states[o : o + 121, :13] for o in origins]),
        ),
    )

    def scorer(x, t):
        return dart.baseline.contact_score(x, t, target)

    return data, spec, scorer


def test_all_saved_prefixes_align_and_late_contacts_use_absolute_deadline():
    data, spec, scorer = fixture()
    # A perturbation beyond the remaining deadline must not change its scored contact.
    data["selected"]["result_states"][-1, 4:, 1] = 1
    result = dart.analyze(data, spec, scorer)
    assert result["counts"] == dict(
        plans=40, prefixes=120, contact_scores=41, no_predicted_crossing=0
    )
    assert result["contact_audit"]["passed"]
    for summary in result["prefix_summary"].values():
        assert summary["count"] == 40
        assert all(m["maximum_norm"] == 0 for m in summary["metrics"].values())
    assert [p["origin"] for p in result["final_three_origins"]] == [111, 114, 117]
    predicted = result["final_prediction"]["predicted"]
    assert predicted["time_s"] == pytest.approx(1.195, abs=1e-6)
    assert predicted["interval"] == 2
    assert result["plans"][-1]["predicted_global_crossing_interval"] == 119
    assert result["final_prediction"]["predicted_minus_actual"]["miss_distance_m"] == 0
    np.testing.assert_allclose(
        result["plans"][-1]["terminal_lateral_error_m"], [0, 0.005, 0]
    )
    assert all(value == 0 for value in result["calls"].values())


def test_no_crossing_is_retained_and_unexecuted_command_comparison_rejected():
    data, spec, scorer = fixture()
    data["selected"]["result_states"][-1, 1:, 0] = -0.05
    result = dart.analyze(data, spec, scorer)
    assert result["counts"]["no_predicted_crossing"] == 1
    assert result["plans"][-1]["predicted_contact"]["reason"] == "no_plane_crossing"
    assert result["final_prediction"]["predicted_minus_actual"] is None
    assert result["prefix_summary"]["10ms"]["metrics"]["position_m"]["maximum_norm"] > 0
    data["selected"]["inputs_commands"][5, 0, 0] += 0.1
    with pytest.raises(ValueError, match="executed command prefix"):
        dart.analyze(data, spec, scorer)
