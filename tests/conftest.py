"""Shared pytest configuration: gate the optional-simulator markers.

The ``crazyflow`` and ``cascade`` markers (see ``pyproject.toml``) are opt-in
contracts against optional simulator extras. Without this hook, a test
carrying one of these markers fails outright when its extra is not
installed, instead of skipping cleanly. When the extra *is* installed, tests
keep running by default; there is deliberately no environment-variable gate
on top of that.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

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
        for marker_name, (module_name, extra_name) in (
            _OPTIONAL_SIMULATOR_MARKERS.items()
        ):
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
