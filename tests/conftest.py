"""Shared pytest configuration: markers and session-scoped fixture trajectories.

The ``crazyflow`` and ``cascade`` markers (see ``pyproject.toml``) are opt-in
contracts against optional simulator extras. Without this hook, a test
carrying one of these markers fails outright when its extra is not
installed, instead of skipping cleanly. When the extra *is* installed, tests
keep running by default; there is deliberately no environment-variable gate
on top of that.

The ``slow`` marker, declared in ``pyproject.toml``, identifies the three
benchmark-scale tests that take over a minute; ``-m "not slow"`` skips them.

The trajectory fixtures below build the handful of synthetic quadrotor and
fixed-wing rollouts that multiple test modules were each constructing
independently (at collection time, in some cases). Building each one once
per session — instead of once per test, or once per parametrize decorator
at import — is what keeps ``pytest --collect-only`` fast. ``Trajectory`` is
a frozen dataclass with read-only arrays, so sharing one instance across
tests is safe as long as callers derive edits via ``dataclasses.replace``
and ``array.copy()`` rather than mutating in place, which is how every
current caller already works.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from glassbox.core.data import Trajectory
from glassbox.core.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.core.synthetic import generate_trajectory

_OPTIONAL_SIMULATOR_MARKERS = {
    "crazyflow": ("crazyflow", "crazyflow"),
    "cascade": ("cascade", "cascade"),
}


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except (ImportError, RuntimeError):
        # Crazyflow raises RuntimeError instead of ImportError when SciPy
        # was imported before it without SCIPY_ARRAY_API=1; either way the
        # extra is not usable for collection/skip purposes here.
        return False
    return True


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    availability: dict[str, bool] = {}
    for item in items:
        for marker_name, (
            module_name,
            extra_name,
        ) in _OPTIONAL_SIMULATOR_MARKERS.items():
            if item.get_closest_marker(marker_name) is None:
                continue
            if marker_name not in availability:
                availability[marker_name] = _importable(module_name)
            if not availability[marker_name]:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{module_name!r} is not importable; run "
                            f"`uv sync --extra {extra_name}` to enable "
                            f"@pytest.mark.{marker_name} tests"
                        )
                    )
                )


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed0_dur0_1s() -> Trajectory:
    """Quadrotor rollout: seed 0, 0.1s. Shared by short-horizon setup tests."""

    return generate_trajectory(seed=0, duration_s=0.1)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed1_dur0_2s() -> Trajectory:
    """Quadrotor rollout: seed 1, 0.2s."""

    return generate_trajectory(seed=1, duration_s=0.2)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed2_dur0_2s() -> Trajectory:
    """Quadrotor rollout: seed 2, 0.2s."""

    return generate_trajectory(seed=2, duration_s=0.2)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed9_dur4_0s() -> Trajectory:
    """Quadrotor rollout: seed 9, 4.0s. The longer innovation-diagnostics case."""

    return generate_trajectory(seed=9, duration_s=4.0)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed11_dur0_4s() -> Trajectory:
    """Quadrotor rollout: seed 11, 0.4s."""

    return generate_trajectory(seed=11, duration_s=0.4)


@pytest.fixture(scope="session")
def fixedwing_trajectory_seed0_dur0_1s() -> Trajectory:
    """Fixed-wing rollout: seed 0, 0.1s."""

    return generate_fixed_wing_trajectory(seed=0, duration_s=0.1)


@pytest.fixture(scope="session")
def fixedwing_trajectory_seed1_dur0_2s() -> Trajectory:
    """Fixed-wing rollout: seed 1, 0.2s."""

    return generate_fixed_wing_trajectory(seed=1, duration_s=0.2)


@pytest.fixture(scope="session")
def fixedwing_trajectory_seed4_dur4_0s() -> Trajectory:
    """Fixed-wing rollout: seed 4, 4.0s. The longer innovation-diagnostics case."""

    return generate_fixed_wing_trajectory(seed=4, duration_s=4.0)
