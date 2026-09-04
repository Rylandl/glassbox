"""Fit-time parameter evidence: the noise model and the information it implies.

The fit answers two questions about the model it just produced. How large is
its one-step innovation on flights it did not see, which is the noise model
every later observation is weighted by, and how much information about the
structured coefficients its own training transitions carry under that noise
model. Both are the same estimator :func:`glassbox.belief.update.absorb` runs,
so a belief the fit produces and a belief that has absorbed telemetry state
their information in one currency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from glassbox.belief.information import (
    ParameterInformation,
    default_rank_relative_tolerance,
    estimable_structured_parameters,
    innovation_noise_floor,
    structured_parameter_scale,
)
from glassbox.belief.update import one_step_linearization, usable_one_step_transitions
from glassbox.core.data import Trajectory
from glassbox.core.diagnostics import one_step_innovations
from glassbox.core.dynamics import ModelParams, structured_parameter_names
from glassbox.core.model import ExecutableModel

# One-step Jacobians are cheap next to a fit, but not free. The budget is
# spread evenly across the independent source groups so a corpus with one long
# flight and one short one does not state its information mostly about the long
# one, and the information is the plain sum over the windows actually read.
MAXIMUM_PARAMETER_INFORMATION_WINDOWS = 512


def innovation_noise(
    params: ModelParams,
    flights: Sequence[tuple[Trajectory, np.ndarray | None]],
) -> np.ndarray:
    """Return per-coordinate one-step innovation variance, at or above the floor.

    This is the belief's noise model: how wrong the fitted model's one-step
    predictions were on flights the fit did not see, in the twelve rigid-body
    local coordinates. It is a second moment about zero rather than about the
    mean error, because nothing subtracts that mean at runtime.
    """

    floor = innovation_noise_floor()
    squares: list[np.ndarray] = []
    for trajectory, control_history in flights:
        innovations = one_step_innovations(
            params, trajectory, control_history=control_history
        )
        finite = innovations[np.all(np.isfinite(innovations), axis=1)]
        if len(finite):
            squares.append(np.mean(np.square(finite), axis=0))
    if not squares:
        return floor
    return np.maximum(floor, np.mean(np.asarray(squares), axis=0))


def _balanced_window_indices(
    groups: np.ndarray,
    *,
    maximum_windows: int,
) -> np.ndarray:
    """Spread one window budget evenly across groups, then across each timeline."""

    group_order = tuple(dict.fromkeys(groups.tolist()))
    budget = min(len(groups), max(maximum_windows, len(group_order)))
    members = [np.flatnonzero(groups == group) for group in group_order]
    allocation = np.zeros(len(members), dtype=np.int64)
    remaining = budget
    while remaining > 0:
        progressed = False
        for index, locations in enumerate(members):
            if allocation[index] >= len(locations):
                continue
            allocation[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    selected: list[int] = []
    for locations, count in zip(members, allocation):
        if count == 0:
            continue
        ordinals = ((2 * np.arange(count, dtype=np.int64) + 1) * len(locations)) // (
            2 * count
        )
        selected.extend(int(locations[ordinal]) for ordinal in ordinals)
    return np.asarray(sorted(selected), dtype=np.int64)


def parameter_information(
    model: ExecutableModel,
    trajectories: Sequence[Trajectory],
    groups: Sequence[str | int],
    *,
    innovation_noise: np.ndarray,
    estimable: np.ndarray | None = None,
    maximum_windows: int = MAXIMUM_PARAMETER_INFORMATION_WINDOWS,
    source: str = "training_one_step_information",
) -> ParameterInformation:
    """Accumulate ``J' R^-1 J`` over the training flights' one-step transitions.

    The noise model is one-step innovation covariance, so the windows it
    weights correctly are one-step windows. Whitening a multi-step endpoint by
    it would overstate the information by roughly the horizon and would count
    the same transition once per training horizon; a plain sum over distinct
    one-step transitions is both the honest Fisher information under the
    declared noise and, exactly, what absorbing that telemetry would add.
    """

    names = structured_parameter_names(model.params)
    mask = (
        estimable_structured_parameters(model.params)
        if estimable is None
        else np.asarray(estimable, dtype=bool)
    )
    if len(trajectories) != len(groups):
        raise ValueError("parameter information needs one group per trajectory")
    labels: list[str] = []
    sources: list[tuple[int, int]] = []
    for index, trajectory in enumerate(trajectories):
        usable, _ = usable_one_step_transitions(model, trajectory)
        for start in np.flatnonzero(usable):
            labels.append(str(groups[index]))
            sources.append((index, int(start)))
    precision = np.zeros((len(names), len(names)))
    used = 0
    if sources:
        selected = _balanced_window_indices(
            np.asarray(labels, dtype=object), maximum_windows=maximum_windows
        )
        weight = 1.0 / np.asarray(innovation_noise, dtype=np.float64)
        for index, _ in enumerate(trajectories):
            starts = np.asarray(
                [
                    sources[ordinal][1]
                    for ordinal in selected
                    if sources[ordinal][0] == index
                ],
                dtype=np.int64,
            )
            if not len(starts):
                continue
            _, jacobians = one_step_linearization(model, trajectories[index], starts)
            if not np.all(np.isfinite(jacobians)):
                continue
            jacobians[..., ~mask] = 0.0
            precision += np.einsum(
                "wip,i,wiq->pq", jacobians, weight, jacobians, optimize=True
            )
            used += len(starts)
    precision = 0.5 * (precision + precision.T)
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    precision = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    precision[~mask, :] = 0.0
    precision[:, ~mask] = 0.0
    return ParameterInformation(
        names=names,
        precision=precision,
        scale=structured_parameter_scale(model.params),
        estimable=mask,
        innovation_noise=np.asarray(innovation_noise, dtype=np.float64),
        noise_floor=innovation_noise_floor(),
        effective_count=float(used),
        rank_relative_tolerance=default_rank_relative_tolerance(used * 12, len(names)),
        source=source,
    )
