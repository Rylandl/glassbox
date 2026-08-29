import pytest

from glassbox.sitl_profile import profile_targets


def test_excitation_condition_scales_translation_and_dwell() -> None:
    low = profile_targets("vertical_steps", condition="low")
    high = profile_targets("lateral_steps", condition="high")

    assert low[0].down_m == pytest.approx(-1.2)
    assert low[0].duration_s == 4.0
    assert high[1].north_m == pytest.approx(2.8)
    assert high[1].duration_s == 2.0


def test_initial_yaw_rotates_translation_and_offsets_heading() -> None:
    targets = profile_targets(
        "lateral_steps", condition="medium", initial_yaw_deg=90.0
    )

    assert targets[1].north_m == pytest.approx(0.0, abs=1e-12)
    assert targets[1].east_m == pytest.approx(2.0)
    assert targets[1].yaw_deg == pytest.approx(90.0)


def test_low_condition_reduces_yaw_step_amplitude() -> None:
    targets = profile_targets(
        "yaw_steps", condition="low", initial_yaw_deg=45.0
    )

    assert targets[1].yaw_deg == pytest.approx(90.0)
    assert targets[3].yaw_deg == pytest.approx(135.0)
