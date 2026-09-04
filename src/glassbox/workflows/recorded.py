"""One comparison of a fresh document against a recorded artifact.

Two things compare a freshly produced result against the file committed under
``docs/results/``: ``glassbox record-results --check``, which regenerates an
artifact and fails when any number moved, and each artifact's pinned test,
which regenerates it and holds every quantity to a tolerance chosen for that
quantity. They ask the same question of the same JSON, so they walk it once,
here, and differ only in the policy they pass.

A policy is three collections of path patterns:

``tolerances``
    Either one ``(relative, absolute)`` pair applied to every float, which is
    what the manifest's ``--check`` uses, or an ordered mapping of pattern to
    tolerance where the first matching pattern wins. Under a mapping, a float
    whose path matches no pattern is reported rather than silently passed, so
    a policy table cannot quietly stop covering part of an artifact.

``exact``
    Patterns whose floats must compare equal. Booleans, integers, strings and
    ``None`` are always compared exactly, so this is only for floats that are
    genuinely deterministic offline.

``ignore``
    Patterns not compared at all: host-dependent values such as wall clock and
    the source fingerprint, and derived quantities that stay in the artifact
    as a recorded value but are not stable enough to pin.

A pattern matches a path when it globs the whole path or a dot-separated
suffix of it, so ``"environment"`` covers that block wherever it appears while
``"scenarios[*].solve_time_p90_s"`` stays specific. ``*`` is the only
metacharacter; brackets are literal.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from typing import Any

__all__ = ["DEFAULT_TOLERANCE", "Tolerance", "matches_path", "recorded_differences"]

Tolerance = float | tuple[float, float]

DEFAULT_TOLERANCE = (1e-5, 1e-7)
"""Relative and absolute tolerance ``--check`` allows on a recorded float."""

_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _pattern(pattern: str) -> re.Pattern[str]:
    compiled = _PATTERN_CACHE.get(pattern)
    if compiled is None:
        body = re.escape(pattern).replace("\\*", ".*")
        compiled = re.compile(rf"(?:.*\.)?{body}\Z")
        _PATTERN_CACHE[pattern] = compiled
    return compiled


def matches_path(patterns: Collection[str], path: str) -> bool:
    """Whether one dotted path matches any pattern."""

    return any(_pattern(pattern).match(path) is not None for pattern in patterns)


def recorded_differences(
    actual: Any,
    recorded: Any,
    *,
    tolerances: Mapping[str, Tolerance] | tuple[float, float] = DEFAULT_TOLERANCE,
    exact: Collection[str] = (),
    ignore: Collection[str] = (),
    path: str = "report",
) -> list[str]:
    """Every way a fresh document differs from the recorded one.

    Ignored paths are skipped entirely. Booleans, integers, strings and
    ``None`` must compare equal; a float is held to the tolerance the policy
    gives its path.
    """

    differences: list[str] = []
    _compare(actual, recorded, path, tolerances, exact, ignore, differences)
    return differences


def _compare(
    actual: Any,
    recorded: Any,
    path: str,
    tolerances: Mapping[str, Tolerance] | tuple[float, float],
    exact: Collection[str],
    ignore: Collection[str],
    differences: list[str],
) -> None:
    if matches_path(ignore, path):
        return
    if isinstance(recorded, dict):
        if not isinstance(actual, dict):
            differences.append(f"{path}: recorded a mapping, produced {type(actual)}")
            return
        for key in sorted(set(recorded) ^ set(actual)):
            if not matches_path(ignore, f"{path}.{key}"):
                side = (
                    "recorded but not produced"
                    if key in recorded
                    else "produced but not recorded"
                )
                differences.append(f"{path}.{key}: {side}")
        for key in recorded:
            if key in actual:
                _compare(
                    actual[key],
                    recorded[key],
                    f"{path}.{key}",
                    tolerances,
                    exact,
                    ignore,
                    differences,
                )
        return
    if isinstance(recorded, list):
        if not isinstance(actual, list):
            differences.append(f"{path}: recorded a list, produced {type(actual)}")
            return
        if len(actual) != len(recorded):
            differences.append(
                f"{path}: recorded {len(recorded)} items, produced {len(actual)}"
            )
            return
        for index, (left, right) in enumerate(zip(actual, recorded)):
            _compare(
                left, right, f"{path}[{index}]", tolerances, exact, ignore, differences
            )
        return
    if _is_number(recorded) and _is_number(actual):
        _compare_number(actual, recorded, path, tolerances, exact, differences)
        return
    if actual != recorded or isinstance(recorded, bool) != isinstance(actual, bool):
        differences.append(f"{path}: recorded {recorded!r}, produced {actual!r}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_number(
    actual: float,
    recorded: float,
    path: str,
    tolerances: Mapping[str, Tolerance] | tuple[float, float],
    exact: Collection[str],
    differences: list[str],
) -> None:
    if not math.isfinite(recorded) or not math.isfinite(actual):
        if repr(actual) != repr(recorded):
            differences.append(f"{path}: recorded {recorded!r}, produced {actual!r}")
        return
    if isinstance(recorded, int) and isinstance(actual, int):
        if actual != recorded:
            differences.append(f"{path}: recorded {recorded!r}, produced {actual!r}")
        return
    if matches_path(exact, path):
        if actual != recorded:
            differences.append(
                f"{path}: recorded {recorded!r}, produced {actual!r} (policy: exact)"
            )
        return
    tolerance = _tolerance_for(tolerances, path)
    if tolerance is None:
        differences.append(
            f"{path}: recorded {recorded!r}, produced {actual!r} "
            "(no tolerance policy covers this path)"
        )
        return
    relative, absolute = tolerance if isinstance(tolerance, tuple) else (tolerance, 0.0)
    difference = abs(actual - recorded)
    if difference > relative * abs(recorded) + absolute:
        differences.append(
            f"{path}: recorded {recorded!r}, produced {actual!r} "
            f"(difference {difference:.3e}, allowed "
            f"{relative:g} relative plus {absolute:g} absolute)"
        )


def _tolerance_for(
    tolerances: Mapping[str, Tolerance] | tuple[float, float],
    path: str,
) -> Tolerance | None:
    if isinstance(tolerances, tuple):
        return tolerances
    for pattern, tolerance in tolerances.items():
        if _pattern(pattern).match(path) is not None:
            return tolerance
    return None
