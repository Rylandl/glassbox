"""One maintained generic learner: fit(recordings), predict, update.

This prototype is not the stable structured ``glassbox.fit``. It predicts
Euclidean observation channels with a single recursive affine-plus-neural model.
There are no caller-selected representations, optimizers or selection policies.
Configuration identity and ordered channel identities are required data facts;
adapters must include units, frame and command/measurement meaning in them.

There is one recipe, ``generic-memory-v2-prototype``. It reads an explicit
100 ms history and a causal memory over a 500 ms in-recording context. A saved
model carries that recipe and ``update`` refits it; any other saved format is
rejected rather than migrated.
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
RECIPE = {
    "id": "generic-memory-v2-prototype",
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
_FORMAT = "glassbox-default-recipe-v2"


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


def _subset(windows, indices):
    return SequenceWindows(
        SequenceBatch(
            **{k: getattr(windows.batch, k)[indices] for k in _ARRAYS},
            dt_s=windows.batch.dt_s,
        ),
        tuple(windows.keys[i] for i in indices),
        tuple(windows.source_origins[i] for i in indices),
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


def _train(train, development, contract, seen, *, previous=None):
    b = train.batch
    steps = steps_for(b.dt_s)
    memory = dict(memory=RECIPE["memory"], delay_steps=steps["delay"])
    ridge = RECIPE["ridge_fraction"] * len(b.past_states) * b.future_states.shape[1]
    initial = initialize_sequence_model(
        b,
        width=RECIPE["width"],
        seed=RECIPE["seed"],
        ridge=ridge,
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
        **memory,
    )
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
        development_errors=_measure(model, development),
        constant_input_channels=[
            contract["input_channels"][i]
            for i in np.flatnonzero(train_u.std(0) <= 1e-8)
        ],
        evidence_limits=[
            "development targets select checkpoints and are not independent error calibration",
            "worse_than_hold describes measured channel error, not a probability or control-admission rule",
            "support outside observed conditions is not established",
            "Euclidean forecasts do not enforce manifold constraints",
            "updates refit cached windows and keep the original development evidence source; fresh independent evaluation is still needed",
        ],
    )
    return LearnedDynamics(model, train, development, contract, seen, report)


class LearnedDynamics:
    """One recipe with fixed fit/update mechanics and measured error evidence.

    The retained arrays hold at most 384 training and 256 development windows.
    The recording identity ledger grows with updates. Updates require new whole
    recordings, not overlapping chunks of an existing recording, and perform a
    batch refit of the same recipe. This is not a real-time streaming learner
    or an active controller.
    """

    def __init__(self, model, train, development, contract, seen, report):
        self._model = model
        self._train, self._development = train, development
        self._contract, self._seen, self._report = map(
            copy.deepcopy, (contract, seen, report)
        )
        if self._report["recipe"] != RECIPE:
            raise ValueError("unsupported default recipe version")

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

    def diagnose(self, recordings):
        """Inspect out-of-fit input predictability and extra-history error evidence.

        This offline report does not update the model or qualify it for control.
        At least three distinct, contract-matching diagnostic recordings are needed.
        """
        from .sequence_diagnostics import diagnose

        return diagnose(self, recordings)

    def update(self, recordings):
        """Refit the recipe with fresh recordings; the current revision stays fixed."""
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
        meta, arrays = load_arrays(path)
        if meta.get("recipe") != RECIPE or meta.get("format") != _FORMAT:
            raise ValueError("unsupported default recipe version")
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
    return _train(train, development, contract, seen)
