from types import SimpleNamespace

import pytest

import glassbox.sitl_profile as sitl_profile
from glassbox.sitl_profile import fly_profile, profile_targets


def test_excitation_condition_scales_translation_and_dwell() -> None:
    low = profile_targets("vertical_steps", condition="low")
    high = profile_targets("lateral_steps", condition="high")

    assert low[0].down_m == pytest.approx(-1.2)
    assert low[0].duration_s == 4.0
    assert high[1].north_m == pytest.approx(2.8)
    assert high[1].duration_s == 2.0


def test_initial_yaw_rotates_translation_and_offsets_heading() -> None:
    targets = profile_targets("lateral_steps", condition="medium", initial_yaw_deg=90.0)

    assert targets[1].north_m == pytest.approx(0.0, abs=1e-12)
    assert targets[1].east_m == pytest.approx(2.0)
    assert targets[1].yaw_deg == pytest.approx(90.0)


def test_low_condition_reduces_yaw_step_amplitude() -> None:
    targets = profile_targets("yaw_steps", condition="low", initial_yaw_deg=45.0)

    assert targets[1].yaw_deg == pytest.approx(90.0)
    assert targets[3].yaw_deg == pytest.approx(135.0)


def test_flight_error_is_not_suppressed_after_successful_landing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Connection:
        target_system = 1
        target_component = 1

        def wait_heartbeat(self, *, timeout: float) -> object:
            return object()

        def mode_mapping(self) -> dict[str, int]:
            return {"OFFBOARD": 1}

        def set_mode(self, mode: int) -> None:
            assert mode == 1

        def motors_armed(self) -> bool:
            return True

        def recv_match(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(base_mode=0)

    connection = Connection()
    stream_count = 0
    landed = False

    def stream_target(*_args: object, **_kwargs: object) -> None:
        nonlocal stream_count
        stream_count += 1
        if stream_count > 1:
            raise ValueError("profile stream failed")

    def command_land(received: object) -> None:
        nonlocal landed
        assert received is connection
        landed = True

    monkeypatch.setattr(
        sitl_profile.mavutil,
        "mavlink_connection",
        lambda connection_string: connection,
    )
    monkeypatch.setattr(sitl_profile, "_stream_target", stream_target)
    monkeypatch.setattr(sitl_profile, "_command_land", command_land)

    with pytest.raises(ValueError, match="profile stream failed"):
        fly_profile("vertical_steps", landing_timeout_s=1.0)

    assert landed
