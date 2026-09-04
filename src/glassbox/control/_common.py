"""Shared, private helpers for the bootstrap identifier and the supervisor.

Every function here previously existed as a byte-for-byte copy in two or more
of :mod:`glassbox.control.identifier` and
:mod:`glassbox.control.supervisor`.  The implementations are kept
exactly as they were, in the same operation order, so consolidating them
changes no number anywhere.

Nothing here is part of the public API. The NumPy and JAX rotation helpers
that once lived here are public in :mod:`glassbox.core.geometry`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def finite_vector(
    name: str,
    values: float | Sequence[float],
    size: int,
) -> np.ndarray:
    """Broadcast a scalar or sequence to a validated float64 vector.

    A scalar is repeated to ``size`` entries.  Anything that is not exactly
    ``size`` finite values is refused by name, so a configuration mistake is
    reported where it is made rather than as a shape error deep in a solve.
    """

    if np.isscalar(values):
        result = np.full(size, float(values), dtype=np.float64)
    else:
        result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def finite_tuple(
    name: str,
    values: float | Sequence[float],
    size: int,
) -> tuple[float, ...]:
    """Return :func:`finite_vector` as a hashable tuple of Python floats.

    Frozen configuration dataclasses store their validated vectors as tuples so
    they stay comparable and hashable; the arithmetic is identical.
    """

    return tuple(float(value) for value in finite_vector(name, values, size))


def immutable_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a validated, read-only float64 copy of one array field.

    Copying then freezing means a frozen result dataclass cannot be mutated
    through the array the caller handed it, and a wrong shape or a non-finite
    entry is refused by name at construction.
    """

    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result.flags.writeable = False
    return result
