import numpy as np

from glassbox.fixedwing_sitl_profile import (
    PROFILES,
    TRIM_PITCH_DEG,
    TRIM_THROTTLE,
    _quaternion_from_euler,
    profile_targets,
)


def test_low_and_high_conditions_scale_around_trim() -> None:
    base = PROFILES["combined"][0]
    low = profile_targets("combined", condition="low")[0]
    high = profile_targets("combined", condition="high")[0]

    assert abs(low.roll_deg) < abs(base.roll_deg) < abs(high.roll_deg)
    assert (
        abs(low.pitch_deg - TRIM_PITCH_DEG)
        < abs(base.pitch_deg - TRIM_PITCH_DEG)
        < abs(high.pitch_deg - TRIM_PITCH_DEG)
    )
    assert (
        abs(low.throttle - TRIM_THROTTLE)
        < abs(base.throttle - TRIM_THROTTLE)
        < abs(high.throttle - TRIM_THROTTLE)
    )


def test_attitude_quaternion_is_unit_length() -> None:
    quaternion = _quaternion_from_euler(0.2, -0.1, 1.1)
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
