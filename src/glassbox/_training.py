"""Validated fitting batches and fixed optimizer accounting."""

from dataclasses import dataclass

import jax
import numpy as np


class SequenceFitError(ValueError):
    """A nonfinite or time-limited fit, rather than an admitted revision."""


@dataclass(frozen=True)
class SequenceBatch:
    """Past x[-P:0], u[-P:-1]; future u[0:H-1], target x[1:H]."""

    past_states: np.ndarray
    past_inputs: np.ndarray
    future_inputs: np.ndarray
    future_states: np.ndarray
    dt_s: float

    def __post_init__(self):
        for key in ("past_states", "past_inputs", "future_inputs", "future_states"):
            value = np.array(getattr(self, key), dtype=float, copy=True)
            if value.ndim != 3 or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite [batch,time,channel] array")
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        n, p1, d = self.past_states.shape
        nf, h, u = self.future_inputs.shape
        if (
            n < 3
            or min(p1 - 1, d, h, u) < 1
            or nf != n
            or self.past_inputs.shape != (n, p1 - 1, u)
            or self.future_states.shape != (n, h, d)
            or not np.isfinite(self.dt_s)
            or self.dt_s <= 0
        ):
            raise ValueError("sequence shapes or dt_s are inconsistent")


def trial_parameters(current, proposal, alpha):
    """Canonical fit/replay interpolation; alpha one preserves proposal bytes."""
    if alpha == 1.0:
        return proposal
    return jax.tree.map(lambda old, new: old + alpha * (new - old), current, proposal)


def gradient_metadata(
    training_windows,
    *,
    attempts_started,
    gradient_proposal_calls_returned,
    completed_acceptance_attempts,
):
    """Describe only returned gradient work; unfinished calls remain unknown."""
    counts = (
        completed_acceptance_attempts,
        gradient_proposal_calls_returned,
        attempts_started,
    )
    if (
        type(training_windows) is not int
        or training_windows < 1
        or any(type(value) is not int or value < 0 for value in counts)
        or not counts[0] <= counts[1] <= counts[2]
        or counts[2] - counts[0] > 1
    ):
        raise ValueError("invalid actual full-cache gradient accounting")
    return dict(
        policy="ordered_full_cache",
        loss_scope="full_training_cache_pre_proposal",
        training_windows=training_windows,
        index_dtype="int64",
        sampling_seed=None,
        attempts_started=attempts_started,
        gradient_proposal_calls_returned=gradient_proposal_calls_returned,
        completed_acceptance_attempts=completed_acceptance_attempts,
        known_gradient_window_visits=training_windows
        * gradient_proposal_calls_returned,
        incomplete_gradient_work_unknown=attempts_started
        > gradient_proposal_calls_returned,
    )


def safeguard_metadata(
    *,
    steps,
    accepted_scales,
    objective_calls,
    initial_loss,
    final_loss,
    checkpoint_losses,
    selected_step,
    recipe_id,
    scales,
    wall_time_limit_s,
):
    """Deterministic report independently rebuildable from actual observations."""
    rejected = sum(scale == 0 for scale in accepted_scales)
    longest = current = 0
    for scale in accepted_scales:
        current = current + 1 if scale == 0 else 0
        longest = max(longest, current)
    return dict(
        id=recipe_id,
        acceptance="first_finite_strict_decrease",
        moment_policy="advance_on_every_finite_proposal",
        proposal_attempts=steps,
        completed_attempts=len(accepted_scales),
        accepted_attempts=len(accepted_scales) - rejected,
        rejected_attempts=rejected,
        scales=list(scales),
        accepted_scale_counts=[
            dict(scale=scale, attempts=accepted_scales.count(scale)) for scale in scales
        ],
        maximum_rejection_streak=longest,
        gradient_proposal_calls=steps,
        full_training_objective_calls=objective_calls,
        full_training_objective_calls_max=1 + len(scales) * steps,
        initial_full_training_loss=initial_loss,
        final_full_training_loss=final_loss,
        selected_full_training_loss=next(
            row["full_training_loss"]
            for row in checkpoint_losses
            if row["step"] == selected_step
        ),
        checkpoint_full_training_losses=checkpoint_losses,
        fit_wall_time_limit_s=wall_time_limit_s,
    )


def validate_window_consistency(batch, keys, source_origins):
    """Require one state/command value for each retained recording source row.

    Windows may overlap within a segment. Their history and forecast arrays must
    agree at every shared row, including the history/forecast boundary. Different
    segments in the same recording cannot claim overlapping source rows. The
    check uses only saved observations; it never runs the dynamics model.
    """
    history = batch.past_inputs.shape[1]
    horizon = batch.future_inputs.shape[1]
    records = {}
    for index, key in enumerate(keys):
        records.setdefault(key.recording_id, []).append(index)
    starts = np.asarray(source_origins) - history
    for indices in records.values():
        identities = {}
        segment_ids = np.array(
            [
                identities.setdefault(keys[i].segment_id, len(identities))
                for i in indices
            ]
        )
        for past, future, count in (
            (batch.past_states, batch.future_states, history + horizon + 1),
            (batch.past_inputs, batch.future_inputs, history + horizon),
        ):
            rows = (starts[indices, None] + np.arange(count)).reshape(-1)
            values = np.concatenate((past[indices], future[indices]), axis=1)
            values = values.reshape(-1, values.shape[-1])
            order = np.argsort(rows, kind="stable")
            ordered_rows = rows[order]
            repeats = ordered_rows[1:] == ordered_rows[:-1]
            segments = np.repeat(segment_ids, count)[order]
            if np.any(segments[1:][repeats] != segments[:-1][repeats]):
                raise ValueError("retained segments overlap within a recording")
            ordered_values = values[order]
            if np.any(ordered_values[1:][repeats] != ordered_values[:-1][repeats]):
                raise ValueError(
                    "overlapping windows disagree on recording source rows"
                )
