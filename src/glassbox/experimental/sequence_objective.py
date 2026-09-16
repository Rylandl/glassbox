"""Generic trajectory loss scales and development-only regression checks.

These are model-selection tools, not uncertainty estimates or safety guarantees.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def relative_error_scale(reference, target, channel_scale, *, minimum_fraction=0.001):
    """Balance each horizon/channel by training-reference RMSE, with a floor.

    Use training predictions/targets only. The floor is a fraction of training
    channel variation; it prevents an almost-exact fit from creating huge weights.
    It is not a measurement-noise estimate or calibrated error bound.
    """
    ref, y, scale = map(np.asarray, (reference, target, channel_scale))
    if (
        ref.ndim != 3
        or ref.shape != y.shape
        or scale.shape != ref.shape[-1:]
        or not all(np.isfinite(a).all() for a in (ref, y, scale))
        or np.any(scale <= 0)
        or not np.isfinite(minimum_fraction)
        or minimum_fraction <= 0
        or not ref.size
    ):
        raise ValueError("invalid reference predictions, targets or channel scales")
    return np.maximum(
        np.sqrt(np.mean((ref - y) ** 2, axis=0)), minimum_fraction * scale
    )


@dataclass(frozen=True)
class SequenceGuard:
    """Limit development RMSE regression vs initialization at selected horizons.

    Horizons count steps from one. Groups contain zero-based output indices.
    A group uses RMS vector error in the supplied physical units, so only group
    comparable channels. Overlapping groups are allowed. This check says nothing
    about unseen recordings; retain its full matrix rather than a single score.
    """

    horizons: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    maximum_ratio: float = 1.05

    def __post_init__(self):
        horizons = tuple(self.horizons)
        groups = tuple(tuple(g) for g in self.groups)
        if (
            not horizons
            or not groups
            or any(not g for g in groups)
            or any(
                not isinstance(h, (int, np.integer)) or isinstance(h, bool) or h < 1
                for h in horizons
            )
            or len(set(horizons)) != len(horizons)
            or any(
                any(
                    not isinstance(c, (int, np.integer)) or isinstance(c, bool) or c < 0
                    for c in g
                )
                or len(set(g)) != len(g)
                for g in groups
            )
            or not np.isfinite(self.maximum_ratio)
            or self.maximum_ratio < 1
        ):
            raise ValueError("invalid guard horizons, groups, or maximum ratio")
        object.__setattr__(self, "horizons", tuple(int(h) for h in horizons))
        object.__setattr__(
            self, "groups", tuple(tuple(int(c) for c in g) for g in groups)
        )

    def validate_shape(self, horizon, channels):
        if (
            max(self.horizons) > horizon
            or max(c for g in self.groups for c in g) >= channels
        ):
            raise ValueError("guard exceeds forecast or output dimensions")

    def errors(self, prediction, target):
        error = prediction - target
        return jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.sqrt(
                            jnp.mean(
                                jnp.sum(error[:, h - 1, jnp.array(group)] ** 2, axis=-1)
                            )
                        )
                        for group in self.groups
                    ]
                )
                for h in self.horizons
            ]
        )

    def accepts(self, errors, reference):
        candidate, baseline = np.asarray(errors), np.asarray(reference)
        shape = (len(self.horizons), len(self.groups))
        if any(
            a.shape != shape or not np.isfinite(a).all() or np.any(a < 0)
            for a in (candidate, baseline)
        ):
            return False
        # Compiled and eager reductions can differ at the last few float32
        # digits. Keep a small relative roundoff allowance in the error units;
        # the absolute term only covers an effectively zero reference error.
        epsilon = max(
            np.finfo(a.dtype if np.issubdtype(a.dtype, np.floating) else float).eps
            for a in (candidate, baseline)
        )
        bound = self.maximum_ratio * baseline
        return bool(np.all(candidate <= bound * (1 + 64 * epsilon) + 1e-12))

    def metadata(self):
        return dict(
            horizons=list(self.horizons),
            groups=[list(g) for g in self.groups],
            maximum_ratio=self.maximum_ratio,
            comparison_roundoff="64 machine eps of the least precise error dtype, plus 1e-12 absolute",
        )
