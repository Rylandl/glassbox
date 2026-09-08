"""Shared pytest configuration: markers and session-scoped fixture trajectories.

Tests marked ``cascade`` run when the simulator dependency group is installed
and skip otherwise.

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

``quadrotor_flight`` and ``fixedwing_flight`` generalize that: they are
session-scoped build-or-reuse factories keyed by ``(seed, duration, dt)``, so
the short rollouts that module after module builds independently are generated
once for the whole run. The named fixtures below are thin aliases over them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from glassbox.core.data import Trajectory
from glassbox.core.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.core.synthetic import generate_trajectory

_OPTIONAL_SIMULATOR_MARKERS = {
    "cascade": ("cascade", "cascade"),
}


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    availability: dict[str, bool] = {}
    for item in items:
        for marker_name, (
            module_name,
            group_name,
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
                            f"`uv sync --group {group_name}` to enable "
                            f"@pytest.mark.{marker_name} tests"
                        )
                    )
                )


TrajectoryFactory = Callable[..., Trajectory]


@pytest.fixture(scope="session")
def quadrotor_flight() -> TrajectoryFactory:
    """Build-or-reuse a synthetic quadrotor rollout for ``(seed, duration, dt)``.

    Generating a rollout costs roughly half a second of wall time per simulated
    second, and the same handful of short rollouts -- seeds 0 upward at 0.4s
    above all -- is rebuilt by module after module. Routing them through this
    session-scoped cache builds each one once.

    The read-only-array rule of the named fixtures applies here too: derive
    variants with ``dataclasses.replace`` and ``array.copy()``. Callers that
    need non-default ``params`` keep calling ``generate_trajectory`` directly,
    since those rollouts are not shared.
    """

    cache: dict[tuple[int, float, float], Trajectory] = {}

    def build(seed: int, duration_s: float = 0.4, dt_s: float = 0.02) -> Trajectory:
        key = (seed, duration_s, dt_s)
        if key not in cache:
            cache[key] = generate_trajectory(
                seed=seed, duration_s=duration_s, dt_s=dt_s
            )
        return cache[key]

    return build


@pytest.fixture(scope="session")
def fixedwing_flight() -> TrajectoryFactory:
    """The fixed-wing counterpart of :func:`quadrotor_flight`."""

    cache: dict[tuple[int, float, float], Trajectory] = {}

    def build(seed: int, duration_s: float = 0.4, dt_s: float = 0.02) -> Trajectory:
        key = (seed, duration_s, dt_s)
        if key not in cache:
            cache[key] = generate_fixed_wing_trajectory(
                seed=seed, duration_s=duration_s, dt_s=dt_s
            )
        return cache[key]

    return build


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed0_dur0_1s(
    quadrotor_flight: TrajectoryFactory,
) -> Trajectory:
    """Quadrotor rollout: seed 0, 0.1s. Shared by short-horizon setup tests."""

    return quadrotor_flight(0, 0.1)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed1_dur0_2s(
    quadrotor_flight: TrajectoryFactory,
) -> Trajectory:
    """Quadrotor rollout: seed 1, 0.2s."""

    return quadrotor_flight(1, 0.2)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed2_dur0_2s(
    quadrotor_flight: TrajectoryFactory,
) -> Trajectory:
    """Quadrotor rollout: seed 2, 0.2s."""

    return quadrotor_flight(2, 0.2)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed9_dur4_0s(
    quadrotor_flight: TrajectoryFactory,
) -> Trajectory:
    """Quadrotor rollout: seed 9, 4.0s. The longer innovation-diagnostics case."""

    return quadrotor_flight(9, 4.0)


@pytest.fixture(scope="session")
def quadrotor_trajectory_seed11_dur0_4s(
    quadrotor_flight: TrajectoryFactory,
) -> Trajectory:
    """Quadrotor rollout: seed 11, 0.4s."""

    return quadrotor_flight(11, 0.4)


@pytest.fixture(scope="session")
def fixedwing_trajectory_seed0_dur0_1s(
    fixedwing_flight: TrajectoryFactory,
) -> Trajectory:
    """Fixed-wing rollout: seed 0, 0.1s."""

    return fixedwing_flight(0, 0.1)


@pytest.fixture(scope="session")
def fixedwing_trajectory_seed1_dur0_2s(
    fixedwing_flight: TrajectoryFactory,
) -> Trajectory:
    """Fixed-wing rollout: seed 1, 0.2s."""

    return fixedwing_flight(1, 0.2)


@pytest.fixture(scope="session")
def fixedwing_trajectory_seed4_dur4_0s(
    fixedwing_flight: TrajectoryFactory,
) -> Trajectory:
    """Fixed-wing rollout: seed 4, 4.0s. The longer innovation-diagnostics case."""

    return fixedwing_flight(4, 4.0)
