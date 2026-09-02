"""Comparison policy shared by the recorded-result ("pinned") tests.

Each recorded test has two tiers. The contract tier asserts the claims the
docs actually make and that survive floating-point noise. The recorded tier
compares a fresh report against the checked-in artifact under
``docs/results`` with a tolerance chosen per quantity, never tighter than
that quantity's sensitivity.

``assert_recorded_close`` implements the recorded tier so each artifact's
policy is one short table next to its test instead of a scatter of
``pytest.approx`` calls. Three collections of path patterns steer it:

``tolerances``
    Ordered mapping of pattern to relative tolerance, or to a
    ``(relative, absolute)`` pair. The first matching pattern wins, so list
    specific paths before a ``"*"`` catch-all. A float whose path matches no
    pattern is a failure rather than a silent pass, so a policy table cannot
    quietly stop covering part of an artifact.

``exact``
    Patterns whose values must compare equal. Booleans, integers, strings and
    ``None`` are always compared exactly, so this collection is only needed
    for floats that are genuinely deterministic offline.

``ignore``
    Patterns that are not compared at all: host-dependent values such as wall
    clock, and chaotic derived quantities that stay in the artifact as a
    recorded value but are not stable enough to pin.

A pattern matches a path when it globs the whole path or a dot-separated
suffix of it, so ``"terminal_tilt_rad"`` covers that field in every campaign
case while ``"cases[*].timing.*"`` stays specific. ``*`` is the only
metacharacter; brackets are literal.

Tests read artifacts through :func:`recorded_result` so a check of the
tolerances themselves can point :data:`RESULTS_DIR` at perturbed copies
without touching ``docs/results``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"

Tolerance = float | tuple[float, float]

_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def recorded_result(name: str) -> Any:
    """Load the recorded artifact ``name`` from :data:`RESULTS_DIR`."""

    return json.loads((RESULTS_DIR / name).read_text())


def _pattern(pattern: str) -> re.Pattern[str]:
    compiled = _PATTERN_CACHE.get(pattern)
    if compiled is None:
        body = re.escape(pattern).replace("\\*", ".*")
        compiled = re.compile(rf"(?:.*\.)?{body}\Z")
        _PATTERN_CACHE[pattern] = compiled
    return compiled


def _matches(patterns: Collection[str], path: str) -> bool:
    return any(_pattern(pattern).match(path) is not None for pattern in patterns)


def _tolerance_for(
    tolerances: Mapping[str, Tolerance],
    path: str,
) -> Tolerance | None:
    for pattern, tolerance in tolerances.items():
        if _pattern(pattern).match(path) is not None:
            return tolerance
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def assert_recorded_close(
    actual: Any,
    recorded: Any,
    *,
    tolerances: Mapping[str, Tolerance],
    exact: Collection[str] = (),
    ignore: Collection[str] = (),
    path: str = "report",
) -> None:
    """Compare ``actual`` against a recorded artifact under a path policy.

    Every mismatch is collected, so one failure names every path that moved
    together with both values.
    """

    failures: list[str] = []
    _compare(actual, recorded, path, tolerances, exact, ignore, failures)
    if failures:
        raise AssertionError(
            f"{len(failures)} recorded-value mismatch(es) "
            f"under {path!r}:\n" + "\n".join(failures)
        )


def _compare(
    actual: Any,
    recorded: Any,
    path: str,
    tolerances: Mapping[str, Tolerance],
    exact: Collection[str],
    ignore: Collection[str],
    failures: list[str],
) -> None:
    if _matches(ignore, path):
        return
    if isinstance(recorded, dict):
        if not isinstance(actual, dict):
            failures.append(f"{path}: recorded a mapping, actual {type(actual)}")
            return
        for key in sorted(set(recorded) - set(actual)):
            if not _matches(ignore, f"{path}.{key}"):
                failures.append(f"{path}.{key}: recorded but missing from actual")
        for key in sorted(set(actual) - set(recorded)):
            if not _matches(ignore, f"{path}.{key}"):
                failures.append(f"{path}.{key}: present in actual but not recorded")
        for key in recorded:
            if key in actual:
                _compare(
                    actual[key],
                    recorded[key],
                    f"{path}.{key}",
                    tolerances,
                    exact,
                    ignore,
                    failures,
                )
        return
    if isinstance(recorded, list):
        if not isinstance(actual, list):
            failures.append(f"{path}: recorded a list, actual {type(actual)}")
            return
        if len(actual) != len(recorded):
            failures.append(
                f"{path}: recorded {len(recorded)} items, actual {len(actual)}"
            )
            return
        for index, (actual_item, recorded_item) in enumerate(zip(actual, recorded)):
            _compare(
                actual_item,
                recorded_item,
                f"{path}[{index}]",
                tolerances,
                exact,
                ignore,
                failures,
            )
        return
    if _is_number(recorded) and _is_number(actual):
        _compare_number(actual, recorded, path, tolerances, exact, failures)
        return
    if isinstance(recorded, bool) != isinstance(actual, bool):
        failures.append(
            f"{path}: recorded {recorded!r} ({type(recorded).__name__}), "
            f"actual {actual!r} ({type(actual).__name__})"
        )
        return
    if actual != recorded:
        failures.append(f"{path}: recorded {recorded!r}, actual {actual!r}")


def _compare_number(
    actual: float,
    recorded: float,
    path: str,
    tolerances: Mapping[str, Tolerance],
    exact: Collection[str],
    failures: list[str],
) -> None:
    if not math.isfinite(recorded) or not math.isfinite(actual):
        if repr(actual) != repr(recorded):
            failures.append(f"{path}: recorded {recorded!r}, actual {actual!r}")
        return
    if isinstance(recorded, int) and isinstance(actual, int):
        if actual != recorded:
            failures.append(f"{path}: recorded {recorded!r}, actual {actual!r}")
        return
    if _matches(exact, path):
        if actual != recorded:
            failures.append(
                f"{path}: recorded {recorded!r}, actual {actual!r} (policy: exact)"
            )
        return
    tolerance = _tolerance_for(tolerances, path)
    if tolerance is None:
        failures.append(
            f"{path}: recorded {recorded!r}, actual {actual!r} "
            f"(no tolerance policy covers this path)"
        )
        return
    relative, absolute = tolerance if isinstance(tolerance, tuple) else (tolerance, 0.0)
    difference = abs(actual - recorded)
    if difference > relative * abs(recorded) + absolute:
        failures.append(
            f"{path}: recorded {recorded!r}, actual {actual!r} "
            f"(difference {difference:.3e}, allowed "
            f"{relative:g} relative plus {absolute:g} absolute)"
        )
