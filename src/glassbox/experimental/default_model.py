"""One maintained generic learner: fit(recordings), predict, update.

This prototype is not the stable structured ``glassbox.fit``. It predicts
Euclidean observation channels with a single recursive affine-plus-neural model.
There are no caller-selected representations, optimizers or selection policies.
Configuration identity and ordered channel identities are required data facts;
adapters must include units, frame and command/measurement meaning in them.

There is one recipe, ``generic-memory-v4-prototype``. It reads an explicit
100 ms history and a causal memory over a 500 ms in-recording context. A saved
model carries that recipe and ``update`` refits it; any other saved format is
rejected rather than migrated.

Every forecast carries a measured error envelope. The recipe already reserves a
quarter of the supplied recordings as its development role, and the windows cut
from them calibrate a split-conformal half-width per horizon step and per
channel, in that channel's own physical units, at a nominal 90%. It is stored
in the artifact and the report and read back through
:meth:`LearnedDynamics.envelope`. There is no caller option and no way to turn
it off: ``fit``, ``predict`` and ``update`` are unchanged, and ``update``
recalibrates on the same pinned development cache it refits against.

Recordings may also declare, per applied command, the exogenous component the
caller injected into it. That is part of the recordings rather than a caller
option, and this recipe reads none of it: when every recording declares one,
the report records that it was declared and each command channel's excitation
standard deviation as a fraction of that channel's own command range, and the
fit is the same fit either way.

The recipe identifies its own command response from the recordings' own command
variation and holds the fit to it. On the training windows it takes the partial
regression of the one-step next-state change on the applied command given the
observed context -- the same features the affine start is solved on, with that
command's own level columns taken out and used as the regressor -- and the
spread of that slope between the recordings as its standard error. Every
command channel whose response is larger than its own standard error has the
affine block's columns for that channel held to it for every step of training:
its level column carries the whole response and its difference columns carry
zero, so the block's response to that command is the identified one, while the
nonlinear correction and the memory keep their own rows of it and learn around
it. A channel that is not identified that well keeps the ridge's own estimate,
and the report says which channels were held and which were not. ``update``
re-identifies on the merged training cache it refits against.

**The assumption this rests on is structural and stated: the command's
variation given the observed context is exogenous to unobserved disturbance.**
It is an assumption about causality, like the recipe's other ones, not about
any platform: what it asserts is that whatever moves the command beyond what
the observed context explains is not itself a response to a disturbance the
recordings do not show. Where it fails the identified slope is the response
plus that feedback, and the measured standard error does not see the
difference; where a channel's command is explained by the context well enough
that no variation is left, the standard error grows and the channel is not
held.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np

from .arrays import array_fingerprint, load_arrays, save_arrays
from .sequence_collection import SequenceCollection, SequenceWindows, WindowKey
from .sequence_model import (
    SequenceBatch,
    SequenceModel,
    fit_sequence_model,
    initialize_sequence_model,
)

_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
ENVELOPE_COVERAGE = 0.9
"""The nominal coverage of the envelope every forecast carries.

Not a recipe constant: it is the level the envelope claims, which the evidence
manifest declares and measures, rather than a fitting choice. A saved model
records the level it was calibrated at.
"""

RECIPE = {
    "id": "generic-memory-v4-prototype",
    "kind": "filter_mlp",
    "width": 32,
    "memory": 8,
    "delay_s": 0.1,
    "context_s": 0.5,
    "horizon_s": 0.25,
    "training_windows": 384,
    "development_windows": 256,
    "steps": 1000,
    "batch_size": 64,
    "learning_rate": 0.002,
    "ridge_fraction": 0.01,
    "seed": 0,
    "check_every": 100,
    "hold_scale_floor": 0.01,
}
_FORMAT = "glassbox-default-recipe-v4"


def _priority(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _contract(recordings):
    if not isinstance(recordings, SequenceCollection):
        raise TypeError("fit and update require a SequenceCollection of recordings")
    if (
        not recordings.configuration_id
        or not recordings.state_channels
        or not recordings.input_channels
    ):
        raise ValueError(
            "recordings must declare configuration_id and ordered state/input channel identities, including units and frames"
        )
    return dict(
        configuration_id=recordings.configuration_id,
        state_channels=list(recordings.state_channels),
        input_channels=list(recordings.input_channels),
        dt_s=float(recordings.segments[0].dt_s),
    )


def _recording_content(recordings):
    result = {}
    for name in sorted({s.recording_id for s in recordings.segments}):
        segments = sorted(
            (s for s in recordings.segments if s.recording_id == name),
            key=lambda s: s.start_row,
        )
        arrays = {
            f"{i}_{key}": getattr(s, key)
            for i, s in enumerate(segments)
            for key in ("states", "inputs")
        }
        metadata = dict(dt_s=segments[0].dt_s, starts=[s.start_row for s in segments])
        result[name] = array_fingerprint(metadata, arrays)
    if len(set(result.values())) != len(result):
        raise ValueError(
            "duplicate recording content cannot establish a separate validation recording"
        )
    return result


def steps_for(dt_s):
    """Window history (the consumed context), forecast horizon, and explicit delay."""

    def count(seconds):
        return max(1, int(np.rint(seconds / dt_s)))

    return dict(
        history=count(RECIPE["context_s"]),
        horizon=count(RECIPE["horizon_s"]),
        delay=count(RECIPE["delay_s"]),
    )


def _indices(keys, origins, budget):
    by_record = {}
    for i, (key, _origin) in enumerate(zip(keys, origins, strict=True)):
        by_record.setdefault(key.recording_id, []).append(i)
    for indices in by_record.values():
        indices.sort(
            key=lambda i: _priority(
                [keys[i].recording_id, keys[i].segment_id, int(origins[i])]
            )
        )
    result = []
    # Round-robin gives each recording equal slots until it exhausts its data.
    for depth in range(max(map(len, by_record.values()), default=0)):
        for name in sorted(by_record):
            if depth < len(by_record[name]):
                result.append(by_record[name][depth])
            if len(result) == budget:
                return result
    return result


def _excitation(windows, indices=None):
    """The declared excitation of these windows, in the shape ``SequenceWindows`` takes."""
    if not windows.excitation_declared:
        return {}
    return {
        key: getattr(windows, key)
        if indices is None
        else getattr(windows, key)[indices]
        for key in ("past_excitation", "future_excitation")
    }


def _subset(windows, indices):
    return SequenceWindows(
        SequenceBatch(
            **{k: getattr(windows.batch, k)[indices] for k in _ARRAYS},
            dt_s=windows.batch.dt_s,
        ),
        tuple(windows.keys[i] for i in indices),
        tuple(windows.source_origins[i] for i in indices),
        **_excitation(windows, indices),
    )


def _extract(recordings, names, budget):
    steps = steps_for(recordings.segments[0].dt_s)
    p, h = steps["history"], steps["horizon"]
    selected = SequenceCollection(
        tuple(s for s in recordings.segments if s.recording_id in names)
    )
    keys = selected.window_keys(history_steps=p, horizon_steps=h)
    represented = {k.recording_id for k in keys}
    missing = set(names) - represented
    if missing:
        raise ValueError(
            f"recordings have no complete {p}-step history / {h}-step forecast windows: {sorted(missing)}"
        )
    origins = {(s.recording_id, s.segment_id): s.start_row for s in selected.segments}
    source_origins = [origins[k.recording_id, k.segment_id] + k.origin for k in keys]
    chosen = _indices(keys, source_origins, budget)
    if len(chosen) < 3:
        raise ValueError(
            "at least three complete windows are required in each data role"
        )
    return selected.extract([keys[i] for i in chosen], history_steps=p, horizon_steps=h)


def _merge_cache(old, fresh):
    both = old.excitation_declared and fresh.excitation_declared
    merged = SequenceWindows(
        SequenceBatch(
            **{
                k: np.concatenate((getattr(old.batch, k), getattr(fresh.batch, k)))
                for k in _ARRAYS
            },
            dt_s=old.batch.dt_s,
        ),
        old.keys + fresh.keys,
        old.source_origins + fresh.source_origins,
        **(
            {
                key: np.concatenate((getattr(old, key), getattr(fresh, key)))
                for key in ("past_excitation", "future_excitation")
            }
            if both
            else {}
        ),
    )
    return _subset(
        merged,
        _indices(merged.keys, merged.source_origins, RECIPE["training_windows"]),
    )


def _measure(model, windows):
    b = windows.batch
    prediction = np.asarray(
        model.rollout(b.past_states, b.past_inputs, b.future_inputs)
    )
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    result = {}
    ids = np.array([k.recording_id for k in windows.keys])
    for name in sorted(set(ids)):
        mask = ids == name
        error = np.sqrt(np.mean((prediction[mask] - b.future_states[mask]) ** 2, 0))
        reference = np.sqrt(np.mean((hold[mask] - b.future_states[mask]) ** 2, 0))
        result[str(name)] = dict(
            windows=int(mask.sum()),
            channel_rmse=error.tolist(),
            hold_channel_rmse=reference.tolist(),
            worse_than_hold=(error > reference * 1.05 + 1e-12).tolist(),
        )
    return result


def _calibrate(model, windows):
    """Split-conformal half-widths from the windows the fit already holds out.

    For each horizon step and channel, the finite-sample conformal quantile of
    the absolute development-window forecast error: the ``ceil((n+1) * level)``
    smallest of the ``n`` residuals, and the largest of them when that rank
    exceeds ``n``. The result is in the channel's own physical units and covers
    at least the nominal fraction of those residuals by construction.

    The development windows are held out of every gradient step, and they are
    also what selects the training checkpoint. That is a stated approximation of
    exchangeability rather than an independent calibration set, and it is one
    reason measured coverage on a further held-out recording can sit outside the
    envelope's nominal level.
    """
    b = windows.batch
    prediction = np.asarray(
        model.rollout(b.past_states, b.past_inputs, b.future_inputs)
    )
    residual = np.abs(prediction - b.future_states)
    if not np.isfinite(residual).all():
        raise ValueError(
            "a nonfinite development residual cannot calibrate an envelope"
        )
    count = len(residual)
    rank = min(int(np.ceil((count + 1) * ENVELOPE_COVERAGE)), count)
    return np.sort(residual, axis=0)[rank - 1], count, rank


def excitation_fraction(recordings):
    """What the recordings themselves say was injected into their commands.

    ``None`` unless every recording declares its excitation, which is why an
    undeclared fit's report is byte for byte the report it was before this
    existed. When they all do, the answer is one number per command channel:
    the standard deviation of the declared excitation over every applied
    command, as a fraction of that channel's own command range in the same
    recordings. Both quantities are measured from the recordings. There is no
    caller option, no declared range to be told and no threshold here; a
    channel whose command never moves has no range to be excited in and its
    fraction is ``None`` rather than a number over zero.
    """
    if not recordings.excitation_declared:
        return None
    excitation = np.concatenate([s.excitation for s in recordings.segments])
    commands = np.concatenate([s.inputs for s in recordings.segments])
    span = commands.max(axis=0) - commands.min(axis=0)
    deviation = excitation.std(axis=0)
    return [
        None if not width > 0 else float(value / width)
        for value, width in zip(deviation, span, strict=True)
    ]


def _command_design(windows):
    """The affine start's design with the command's level columns taken out.

    One row per (window, horizon step) -- exactly the transitions the affine
    start is solved on. Returns the observed context the slope is taken given:
    the current observation, the explicit history differences, and the command
    differences, in the design's own normalized units; the applied command and
    the observed one-step state change, both in physical units; and the
    recording each row came from.
    """
    b = windows.batch
    p = steps_for(b.dt_s)["delay"]
    context = b.past_inputs.shape[1]
    horizon = b.future_inputs.shape[1]
    x = np.concatenate((b.past_states, b.future_states), 1)
    u = np.concatenate((b.past_inputs, b.future_inputs), 1)
    current = x[:, context:-1]
    xm, xs = current.mean((0, 1)), current.std((0, 1))
    um, us = b.future_inputs.mean((0, 1)), b.future_inputs.std((0, 1))
    xs, us = np.where(xs > 1e-8, xs, 1), np.where(us > 1e-8, us, 1)
    xall, uall = (x - xm) / xs, (u - um) / us
    ids = np.array([k.recording_id for k in windows.keys])
    blocks, commands, changes, sources = [], [], [], []
    for t in range(horizon):
        j = context + t
        state = xall[:, j]
        state_difference = (xall[:, j - p : j] - state[:, None]).reshape(len(state), -1)
        command_difference = (uall[:, j - p : j] - uall[:, j][:, None]).reshape(
            len(state), -1
        )
        blocks.append(np.column_stack((state, state_difference, command_difference)))
        commands.append(u[:, j])
        changes.append(b.future_states[:, t] - current[:, t])
        sources.append(ids)
    return (
        np.concatenate(blocks),
        np.concatenate(commands),
        np.concatenate(changes),
        np.concatenate(sources),
    )


def _partial_slope(block, command, change):
    """Frisch-Waugh: both sides residualized on the context and a constant.

    The result is physical state change per unit command, one row per command
    channel. The least-squares cutoff is numpy's own, referred to the command's
    own variation rather than to what survived the context: a direction of
    command the observed context already accounts for to within floating point
    identifies nothing at all, and is reported as no response rather than
    divided by.
    """
    design = np.column_stack((block, np.ones(len(block))))
    both = np.column_stack((command, change))
    projection, *_ = np.linalg.lstsq(design, both, rcond=None)
    residual = both - design @ projection
    width = command.shape[1]
    left, right = residual[:, :width], residual[:, width:]
    factors, spectrum, directions = np.linalg.svd(left, full_matrices=False)
    scale = np.linalg.svd(command - command.mean(0), compute_uv=False).max()
    surviving = spectrum > np.finfo(float).eps * max(left.shape) * scale
    return directions[surviving].T @ (
        (factors[:, surviving].T @ right) / spectrum[surviving, None]
    )


def _command_response(windows):
    """The one-step command response these windows identify, and how well.

    The estimate is the partial slope above over every training row. Its
    standard error is the spread of the same slope between the recordings the
    windows came from, which is the reading that sees the serial correlation a
    row-wise standard error does not: each recording is a separate run of the
    system. A recording with fewer rows than the context has columns cannot
    state a slope of its own and contributes none; with fewer than two that
    can, there is no spread, nothing is identified and nothing is held.

    A channel is held when its whole response is larger than its own standard
    error in the same units. There is no threshold here beyond that comparison,
    and no constant.
    """
    block, command, change, sources = _command_design(windows)
    response = _partial_slope(block, command, change)
    names = sorted(set(sources.tolist()))
    minimum = block.shape[1] + command.shape[1] + 1
    per_recording = [
        _partial_slope(block[rows], command[rows], change[rows])
        for rows in (sources == name for name in names)
        if rows.sum() > minimum
    ]
    if len(per_recording) < 2:
        spread = None
    else:
        stack = np.stack(per_recording)
        spread = stack.std(0, ddof=1) / np.sqrt(len(stack))
    size = np.linalg.norm(response, axis=1)
    error = (
        np.full(len(size), np.inf) if spread is None else np.linalg.norm(spread, axis=1)
    )
    held = np.isfinite(response).all(1) & (size > error) & (size > 0)
    return dict(
        response=response,
        standard_error=spread,
        held=held,
        size=size,
        error=error,
        recordings=names,
        identifying_recordings=len(per_recording),
        per_recording=per_recording,
        rows=len(block),
    )


def _command_report(identification, contract):
    """What the fit says about the response it held, in the caller's own names."""
    held = identification["held"]
    spread = identification["standard_error"]
    return dict(
        method="partial_regression_of_the_one_step_state_change_on_the_command_given_the_observed_context",
        standard_error="the spread of the same slope between the training recordings",
        assumption="the command's variation given the observed context is exogenous to unobserved disturbance",
        units="physical state change per unit command, per command channel and observation channel",
        rows=identification["rows"],
        recordings=list(identification["recordings"]),
        identifying_recordings=identification["identifying_recordings"],
        channels=list(contract["input_channels"]),
        held=[bool(value) for value in held],
        held_channels=[
            name
            for name, value in zip(contract["input_channels"], held, strict=True)
            if value
        ],
        free_channels=[
            name
            for name, value in zip(contract["input_channels"], held, strict=True)
            if not value
        ],
        response=identification["response"].tolist(),
        standard_error_value=None if spread is None else spread.tolist(),
        response_size=identification["size"].tolist(),
        standard_error_size=[
            None if not np.isfinite(value) else float(value)
            for value in identification["error"]
        ],
    )


def _train(train, development, contract, seen, *, previous=None, excitation=None):
    b = train.batch
    steps = steps_for(b.dt_s)
    memory = dict(memory=RECIPE["memory"], delay_steps=steps["delay"])
    ridge = RECIPE["ridge_fraction"] * len(b.past_states) * b.future_states.shape[1]
    identification = _command_response(train)
    # The pair the model takes is the response and which channels it may be
    # stated for; with no channel identified the fit is the unconstrained one.
    identified = (
        (identification["response"], identification["held"])
        if identification["held"].any()
        else None
    )
    initial = initialize_sequence_model(
        b,
        width=RECIPE["width"],
        seed=RECIPE["seed"],
        ridge=ridge,
        command_response=identified,
        **memory,
    )
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    loss_scale = np.maximum(
        np.sqrt(np.mean((hold - b.future_states) ** 2, 0)),
        RECIPE["hold_scale_floor"] * initial.norms["state_scale"],
    )
    model, optimization = fit_sequence_model(
        b,
        development.batch,
        width=RECIPE["width"],
        ridge=ridge,
        seed=RECIPE["seed"],
        steps=RECIPE["steps"],
        batch_size=RECIPE["batch_size"],
        learning_rate=RECIPE["learning_rate"],
        check_every=RECIPE["check_every"],
        error_scale=loss_scale,
        command_response=identified,
        **memory,
    )
    envelope, calibration_windows, rank = _calibrate(model, development)
    train_u = b.future_inputs.reshape(-1, b.future_inputs.shape[-1])
    report = dict(
        recipe=copy.deepcopy(RECIPE),
        previous_revision=previous,
        history_steps=model.history_steps,
        horizon_steps=b.future_states.shape[1],
        delay_steps=steps["delay"],
        training=train.coverage(),
        development=development.coverage(),
        optimization=optimization,
        command_response=_command_report(identification, contract),
        development_errors=_measure(model, development),
        constant_input_channels=[
            contract["input_channels"][i]
            for i in np.flatnonzero(train_u.std(0) <= 1e-8)
        ],
        envelope=dict(
            nominal_coverage=ENVELOPE_COVERAGE,
            method="split_conformal_absolute_error",
            calibrated_on="development",
            calibration_windows=calibration_windows,
            quantile_rank=rank,
            units="physical, per horizon step and per channel, aligned with predict",
            half_width=envelope.tolist(),
        ),
        evidence_limits=[
            "the held command response assumes the command's variation given the observed context is exogenous to unobserved disturbance; where it is not, the identified slope carries that feedback and the standard error does not see it",
            "the development windows both select the checkpoint and calibrate the envelope, so the envelope's exchangeability is approximate rather than independent",
            "the envelope's coverage is nominal on those windows and measured elsewhere; the harness's evidence tier is where it is measured",
            "worse_than_hold describes measured channel error, not a probability or control-admission rule",
            "support outside observed conditions is not established",
            "Euclidean forecasts do not enforce manifold constraints",
            "updates refit cached windows and keep the original development evidence source; fresh independent evaluation is still needed",
        ],
    )
    if excitation is not None:
        # Only when every recording declared it. Absent, the report is byte for
        # byte the report an undeclared fit has always written, which is what
        # keeps every existing artifact, fingerprint and reference standing.
        report["excitation_declared"] = True
        report["excitation_standard_deviation_fraction"] = excitation
    return LearnedDynamics(model, train, development, contract, seen, report, envelope)


class LearnedDynamics:
    """One recipe with fixed fit/update mechanics and measured error evidence.

    The retained arrays hold at most 384 training and 256 development windows.
    The recording identity ledger grows with updates. Updates require new whole
    recordings, not overlapping chunks of an existing recording, and perform a
    batch refit of the same recipe. This is not a real-time streaming learner
    or an active controller.
    """

    def __init__(self, model, train, development, contract, seen, report, envelope):
        self._model = model
        self._train, self._development = train, development
        self._contract, self._seen, self._report = map(
            copy.deepcopy, (contract, seen, report)
        )
        if self._report["recipe"] != RECIPE:
            raise ValueError("unsupported default recipe version")
        self._envelope = np.array(envelope, dtype=float, copy=True)
        if (
            self._envelope.shape
            != (
                self.horizon_steps,
                len(self._contract["state_channels"]),
            )
            or not np.isfinite(self._envelope).all()
        ):
            raise ValueError(
                "the envelope must hold one finite half-width per horizon step and channel"
            )
        if np.any(self._envelope < 0):
            raise ValueError("an envelope half-width cannot be negative")
        self._envelope.setflags(write=False)

    @property
    def report(self):
        return copy.deepcopy(self._report)

    @property
    def contract(self):
        return copy.deepcopy(self._contract)

    @property
    def recipe(self):
        return copy.deepcopy(self._report["recipe"])

    @property
    def history_steps(self):
        """Observed transitions every forecast consumes; memory starts at rest there."""
        return self._model.history_steps

    @property
    def horizon_steps(self):
        return self._train.batch.future_states.shape[1]

    def predict(self, past_states, past_inputs, future_inputs):
        """JAX-compatible means in declared observation coordinates, without x0.

        Use the most recent required history. Commands must use the fitted time
        grid and coordinate contract; no missing history is padded. Horizons
        beyond this recipe's fitted range are rejected, not silently extrapolated.
        The memory starts at rest at the first consumed observation; earlier
        observations are never implied.
        """
        import jax.numpy as jnp

        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        if x.ndim not in (2, 3) or up.ndim != x.ndim or uf.ndim != x.ndim:
            raise ValueError(
                "predict expects aligned [time,channel] or [batch,time,channel] arrays"
            )
        if x.shape[-2] != up.shape[-2] + 1 or x.shape[-2] < self.history_steps + 1:
            raise ValueError(
                f"predict needs aligned history with at least {self.history_steps + 1} observations and {self.history_steps} inputs"
            )
        if not 1 <= uf.shape[-2] <= self.horizon_steps:
            raise ValueError(
                f"unsupported forecast horizon: expected 1..{self.horizon_steps} steps"
            )
        return self._model.rollout(
            x[..., -self.history_steps - 1 :, :], up[..., -self.history_steps :, :], uf
        )

    def envelope(self, horizon_steps=None):
        """The measured error envelope this forecast carries, in physical units.

        One half-width per horizon step and per declared observation channel,
        aligned row for row and column for column with what :meth:`predict`
        returns over the same horizon. It is a split-conformal quantile of the
        absolute forecast error on the development windows the fit already
        holds out, at the nominal level in :data:`ENVELOPE_COVERAGE`, and it is
        reported with every forecast rather than requested. Horizons beyond the
        fitted range are rejected exactly as ``predict`` rejects them.
        """
        steps = self.horizon_steps if horizon_steps is None else int(horizon_steps)
        if not 1 <= steps <= self.horizon_steps:
            raise ValueError(
                f"unsupported forecast horizon: expected 1..{self.horizon_steps} steps"
            )
        return np.array(self._envelope[:steps], dtype=float)

    def diagnose(self, recordings):
        """Inspect out-of-fit input predictability and extra-history error evidence.

        This offline report does not update the model or qualify it for control.
        At least three distinct, contract-matching diagnostic recordings are needed.
        """
        from .sequence_diagnostics import diagnose

        return diagnose(self, recordings)

    def update(self, recordings):
        """Refit the recipe with fresh recordings; the current revision stays fixed.

        The command response is identified again on the merged training cache
        this refit is solved on, so a revision states the response its own
        windows show rather than carrying its predecessor's forward.
        """
        if _contract(recordings) != self._contract:
            raise ValueError(
                "update configuration, channels or sample interval differ from this model"
            )
        seen = _recording_content(recordings)
        if set(seen) & set(self._seen):
            raise ValueError("update reuses a known recording identity")
        if set(seen.values()) & set(self._seen.values()):
            raise ValueError("update reuses previously observed recording content")
        fresh = _extract(recordings, set(seen), RECIPE["training_windows"])
        training = _merge_cache(self._train, fresh)
        return _train(
            training,
            self._development,
            self._contract,
            {**self._seen, **seen},
            previous=self.fingerprint(),
            excitation=excitation_fraction(recordings),
        )

    def _metadata(self):
        return dict(
            format=_FORMAT,
            recipe=copy.deepcopy(RECIPE),
            model=self._model.metadata(),
            contract=self.contract,
            seen=copy.deepcopy(self._seen),
            report=self.report,
            windows={
                role: dict(
                    keys=[asdict(k) for k in windows.keys],
                    source_origins=list(windows.source_origins),
                )
                for role, windows in (
                    ("train", self._train),
                    ("development", self._development),
                )
            },
        )

    def _arrays(self):
        return {
            "envelope_half_width": self._envelope,
            **self._model.arrays(),
            **{
                f"{role}_{k}": getattr(windows.batch, k)
                for role, windows in (
                    ("train", self._train),
                    ("development", self._development),
                )
                for k in _ARRAYS
            },
        }

    def fingerprint(self):
        return array_fingerprint(self._metadata(), self._arrays())

    def save(self, path):
        """Save model, recipe, evidence, and bounded update caches together."""
        save_arrays(path, self._metadata(), self._arrays())

    @classmethod
    def load(cls, path):
        """Load a saved revision of this recipe, or refuse it.

        A ``v2`` artifact, which carried no envelope, is rejected rather than
        migrated: a forecast without a measured envelope is not a forecast this
        recipe makes, and inventing one on load would be the opposite of
        measuring it.
        """
        meta, arrays = load_arrays(path)
        if meta.get("recipe") != RECIPE or meta.get("format") != _FORMAT:
            raise ValueError("unsupported default recipe version")
        if "envelope_half_width" not in arrays:
            raise ValueError("a saved revision of this recipe carries an envelope")
        model_meta = dict(meta["model"])
        if model_meta.pop("format") != "glassbox-sequence-v1":
            raise ValueError("unsupported sequence format")
        model = SequenceModel(
            **model_meta,
            params={k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
            norms={k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        )
        if model.kind != RECIPE["kind"]:
            raise ValueError("saved model kind differs from its recipe")
        windows = {}
        for role in ("train", "development"):
            w = meta["windows"][role]
            windows[role] = SequenceWindows(
                SequenceBatch(
                    **{k: arrays[f"{role}_{k}"] for k in _ARRAYS}, dt_s=model.dt_s
                ),
                tuple(WindowKey(**k) for k in w["keys"]),
                tuple(w["source_origins"]),
            )
        return cls(
            model,
            windows["train"],
            windows["development"],
            meta["contract"],
            meta["seen"],
            meta["report"],
            arrays["envelope_half_width"],
        )


def fit(recordings):
    """Fit the recipe with automatic recording holdout and window sampling.

    Supply a SequenceCollection carrying configuration and ordered channel facts.
    At least two distinct recordings are required. No architecture, optimizer,
    representation, split-policy, or random-seed options are accepted.
    """
    contract = _contract(recordings)
    seen = _recording_content(recordings)
    names = sorted(seen, key=_priority)
    if len(names) < 2:
        raise ValueError(
            "fit needs at least two distinct recordings for training and development"
        )
    count = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    development = _extract(recordings, names[:count], RECIPE["development_windows"])
    train = _extract(recordings, names[count:], RECIPE["training_windows"])
    return _train(
        train, development, contract, seen, excitation=excitation_fraction(recordings)
    )
