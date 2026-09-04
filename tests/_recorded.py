"""Comparison policy shared by the recorded-result ("pinned") tests.

Each recorded test has two tiers. The contract tier asserts the claims the
docs actually make and that survive floating-point noise. The recorded tier
compares a fresh report against the checked-in artifact under
``docs/results`` with a tolerance chosen per quantity, never tighter than
that quantity's sensitivity.

``assert_recorded_close`` implements the recorded tier so each artifact's
policy is one short table next to its test instead of a scatter of
``pytest.approx`` calls. The walk itself is
:func:`glassbox.workflows.recorded.recorded_differences`, the same one
``glassbox record-results --check`` runs, so an artifact's manifest entry and
its pinned test cannot disagree about what a difference is; that module
documents the three pattern collections a policy is made of.

Tests read artifacts through :func:`recorded_result` so a check of the
tolerances themselves can point :data:`RESULTS_DIR` at perturbed copies
without touching ``docs/results``.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from glassbox.workflows.recorded import Tolerance, recorded_differences

RESULTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"


def recorded_result(name: str) -> Any:
    """Load the recorded artifact ``name`` from :data:`RESULTS_DIR`."""

    return json.loads((RESULTS_DIR / name).read_text())


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

    failures = recorded_differences(
        actual,
        recorded,
        tolerances=tolerances,
        exact=exact,
        ignore=ignore,
        path=path,
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} recorded-value mismatch(es) "
            f"under {path!r}:\n" + "\n".join(failures)
        )
