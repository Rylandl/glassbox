"""Recording-level empirical forecast evidence, separate from model fitting.

Selection sees only caller-designated development targets. A recording identity
prevents long recordings from silently dominating equal-record policies; it does
not establish statistical independence. Guards are empirical, not guarantees.
"""

from dataclasses import dataclass

import numpy as np

POLICIES = ("pooled", "equal_record", "minimax_hold", "guarded_hold")


@dataclass(frozen=True)
class ForecastEvidence:
    names: tuple[str, ...]
    recordings: tuple[str, ...]
    counts: np.ndarray
    group_widths: np.ndarray
    mse: np.ndarray  # [candidate, recording, horizon, group], standardized channel MSE

    def __post_init__(self):
        names, records = tuple(self.names), tuple(self.recordings)
        counts, widths = np.asarray(self.counts), np.asarray(self.group_widths)
        mse = np.asarray(self.mse)
        if (
            not names
            or not records
            or len(set(names)) != len(names)
            or len(set(records)) != len(records)
            or any(not isinstance(n, str) or not n for n in (*names, *records))
            or counts.shape != (len(records),)
            or widths.ndim != 1
            or not len(widths)
            or not np.issubdtype(counts.dtype, np.integer)
            or not np.issubdtype(widths.dtype, np.integer)
            or np.any(counts <= 0)
            or np.any(widths <= 0)
            or mse.ndim != 4
            or mse.shape[:2] != (len(names), len(records))
            or mse.shape[2] < 1
            or mse.shape[3] != len(widths)
            or not np.isfinite(mse).all()
            or np.any(mse < 0)
        ):
            raise ValueError("invalid recording error evidence")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "recordings", records)
        for key, value in (("counts", counts), ("group_widths", widths), ("mse", mse)):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, key, copied)

    @classmethod
    def from_predictions(cls, predictions, target, recording_ids, *, scale, groups):
        target, ids, scale = map(np.asarray, (target, recording_ids, scale))
        groups = tuple(tuple(g) for g in groups)
        flat = [c for g in groups for c in g]
        if (
            target.ndim != 3
            or min(target.shape) < 1
            or ids.shape != (len(target),)
            or ids.dtype.kind not in "US"
            or np.any(ids == "")
            or scale.shape != (target.shape[-1],)
            or not np.isfinite(target).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or not groups
            or any(not g for g in groups)
            or any(
                not isinstance(c, (int, np.integer)) or isinstance(c, bool)
                for c in flat
            )
            or sorted(flat) != list(range(target.shape[-1]))
            or not predictions
            or any(not isinstance(n, str) or not n for n in predictions)
        ):
            raise ValueError(
                "invalid targets, recording identities, scales or group partition"
            )
        names = tuple(predictions)
        records = tuple(dict.fromkeys(ids.tolist()))
        errors = []
        for value in predictions.values():
            p = np.asarray(value)
            if p.shape != target.shape or not np.isfinite(p).all():
                raise ValueError("predictions must be finite and match targets")
            squared = ((p - target) / scale) ** 2
            errors.append(
                np.stack(
                    [
                        np.stack(
                            [
                                squared[ids == r][:, :, list(g)].mean((0, 2))
                                for g in groups
                            ],
                            -1,
                        )
                        for r in records
                    ]
                )
            )
        arrays = [
            np.array([np.sum(ids == r) for r in records]),
            np.array([len(g) for g in groups]),
            np.stack(errors),
        ]
        for value in arrays:
            value.setflags(write=False)
        return cls(names, records, *arrays)

    def choose(
        self, policy, *, recordings=None, reference="hold", maximum_rmse_ratio=1.05
    ):
        """Return an auditable whole-model choice; no per-output model mixing.

        Ratios divide standardized group MSE by max(reference MSE, 1e-12).
        Ties retain prediction insertion order. Subsets support leave-one-record
        validation without supplying excluded outcomes to the decision rule.
        """
        records = self.recordings if recordings is None else tuple(recordings)
        if (
            policy not in POLICIES
            or not records
            or len(set(records)) != len(records)
            or any(r not in self.recordings for r in records)
            or reference not in self.names
            or not np.isfinite(maximum_rmse_ratio)
            or maximum_rmse_ratio < 1
        ):
            raise ValueError(
                "invalid policy, reference, record subset or empirical guard"
            )
        index = [self.recordings.index(r) for r in records]
        mse = self.mse[:, index]
        weights = self.group_widths / self.group_widths.sum()
        by_record = (mse * weights).sum(-1).mean(-1)
        equal = by_record.mean(-1)
        ratio = mse / np.maximum(mse[self.names.index(reference)], 1e-12)
        worst = ratio.max((1, 2, 3))
        eligible = np.ones(len(self.names), dtype=bool)
        if policy == "pooled":
            scores = np.average(by_record, axis=1, weights=self.counts[index])
        elif policy == "minimax_hold":
            scores = worst
        else:
            scores = equal
            if policy == "guarded_hold":
                eligible = worst <= maximum_rmse_ratio**2
        selected = int(np.argmin(np.where(eligible, scores, np.inf)))
        return dict(
            selected=self.names[selected],
            policy=policy,
            recordings=list(records),
            scores=dict(zip(self.names, scores.tolist())),
            eligible=[n for n, ok in zip(self.names, eligible) if ok],
            worst_reference_mse_ratio=dict(zip(self.names, worst.tolist())),
            reference=reference,
            maximum_rmse_ratio=maximum_rmse_ratio,
        )
