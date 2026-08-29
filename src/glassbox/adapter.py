"""Source-adapter contract for canonical Glassbox trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from glassbox.data import Trajectory


@runtime_checkable
class TrajectoryAdapter(Protocol):
    """Minimal boundary implemented by telemetry source adapters.

    Adapters may expose source-specific configuration on their concrete type,
    but downstream dynamics code receives only the canonical ``Trajectory``.
    """

    name: str

    def inspect(self, path: str | Path) -> Mapping[str, Any]:
        """Describe a source recording without producing an artifact."""

    def load(self, path: str | Path) -> Trajectory:
        """Convert one source recording into a canonical trajectory."""
