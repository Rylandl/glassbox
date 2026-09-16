"""Generic affine forecasts fitted separately at explicit future horizons.

Each head predicts a change from the current observation using a command prefix.
There is no recursive state update and no consistency constraint between heads.
These are conditional mean predictions, not uncertainty estimates or a step model.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .structured_regression import array_fingerprint, load_arrays, save_arrays

REPRESENTATIONS = (
    "full",
    "relative",
    "linear_history",
    "quadratic_history",
    "pca95",
    "pca99",
)


def prefix_features(
    past,
    past_inputs,
    future_inputs,
    horizon,
    *,
    use_history,
    representation="full",
    xp=np,
):
    """Current observation/command, optional relative history, then command changes.

    Never inspect commands at or after index ``horizon``. Inputs are ordered
    oldest to newest, as in SequenceBatch. Channels carry no physical semantics.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown forecast representation")
    current, command = past[:, -1], future_inputs[:, 0]
    parts = [command] if representation == "relative" else [current, command]
    if use_history:
        observations = past[:, :-1] - current[:, None]
        inputs = past_inputs - command[:, None]
        if representation in ("linear_history", "quadratic_history"):
            order = 1 if representation == "linear_history" else 2
            p = past_inputs.shape[1]
            if p < order:
                raise ValueError("insufficient history for requested polynomial order")
            times = np.arange(-p, 0, dtype=float) / p
            basis = np.column_stack([times**k for k in range(1, order + 1)])
            inverse = xp.asarray(np.linalg.pinv(basis))
            observations = xp.einsum("kt,ntd->nkd", inverse, observations)
            inputs = xp.einsum("kt,ntu->nku", inverse, inputs)
        parts.extend(
            [observations.reshape(len(past), -1), inputs.reshape(len(past), -1)]
        )
    parts.append(
        (future_inputs[:, 1:horizon] - command[:, None]).reshape(len(past), -1)
    )
    return xp.concatenate(parts, axis=-1)


@dataclass(frozen=True)
class DirectForecast:
    """Independent horizon heads; predict returns rows in ``horizons`` order."""

    dt_s: float
    history_steps: int
    state_width: int
    input_width: int
    horizons: tuple[int, ...]
    use_history: bool
    ridge_fraction: float
    arrays: dict
    representation: str = "full"

    def predict(self, past_states, past_inputs, future_inputs):
        """Return [batch, number of fitted horizons, state] means; JAX compatible."""
        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        single = x.ndim == 2
        if single:
            x, up, uf = x[None], up[None], uf[None]
        if (
            x.ndim != 3
            or up.ndim != 3
            or uf.ndim != 3
            or x.shape[1:] != (self.history_steps + 1, self.state_width)
            or up.shape != (len(x), self.history_steps, self.input_width)
            or uf.shape[0] != len(x)
            or uf.shape[2] != self.input_width
            or uf.shape[1] < max(self.horizons)
        ):
            raise ValueError(
                "forecast shapes do not match fitted history, horizon or channels"
            )
        values = []
        for h in self.horizons:
            features = prefix_features(
                x,
                up,
                uf,
                h,
                use_history=self.use_history,
                representation=self.representation,
                xp=jnp,
            )
            z = (features - self.arrays[f"mean_{h}"]) / self.arrays[f"scale_{h}"]
            if f"projection_{h}" in self.arrays:
                z = z @ self.arrays[f"projection_{h}"]
            values.append(
                x[:, -1] + z @ self.arrays[f"weight_{h}"] + self.arrays[f"bias_{h}"]
            )
        result = jnp.stack(values, axis=1)
        return result[0] if single else result

    def metadata(self):
        result = dict(
            format="glassbox-direct-forecast-v1"
            if self.representation == "full"
            else "glassbox-direct-forecast-v2",
            dt_s=self.dt_s,
            history_steps=self.history_steps,
            state_width=self.state_width,
            input_width=self.input_width,
            horizons=list(self.horizons),
            use_history=self.use_history,
            ridge_fraction=self.ridge_fraction,
        )
        if self.representation != "full":
            result["representation"] = self.representation
        return result

    def fingerprint(self):
        return array_fingerprint(self.metadata(), self.arrays)

    def save(self, path):
        save_arrays(path, self.metadata(), self.arrays)

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        if metadata.pop("format") not in (
            "glassbox-direct-forecast-v1",
            "glassbox-direct-forecast-v2",
        ):
            raise ValueError("unsupported direct forecast format")
        metadata["horizons"] = tuple(metadata["horizons"])
        return cls(**metadata, arrays=arrays)


def fit_direct_forecast(
    batch, horizons, *, use_history=True, ridge_fraction=0.001, representation="full"
):
    """Fit centered affine heads on training windows only.

    Minimize mean squared residual plus ridge_fraction times squared standardized
    feature coefficients, leaving the intercept unpenalized. This penalty does
    not grow weaker just because the same observations are duplicated.
    """
    horizons = tuple(horizons)
    if (
        not horizons
        or any(
            not isinstance(h, (int, np.integer))
            or isinstance(h, bool)
            or h < 1
            or h > batch.future_states.shape[1]
            for h in horizons
        )
        or len(set(horizons)) != len(horizons)
        or not isinstance(use_history, bool)
        or representation not in REPRESENTATIONS
        or (
            not use_history
            and representation in ("relative", "linear_history", "quadratic_history")
        )
        or not np.isfinite(ridge_fraction)
        or ridge_fraction <= 0
    ):
        raise ValueError("invalid forecast horizons, history option or ridge_fraction")
    arrays = {}
    current = batch.past_states[:, -1]
    for h in horizons:
        features = prefix_features(
            batch.past_states,
            batch.past_inputs,
            batch.future_inputs,
            h,
            use_history=use_history,
            representation=representation,
        )
        mean, scale = features.mean(0), features.std(0)
        scale = np.where(scale > 1e-8, scale, 1)
        z = (features - mean) / scale
        if representation.startswith("pca"):
            _, singular, vectors = np.linalg.svd(z, full_matrices=False)
            variance = np.cumsum(singular**2)
            fraction = 0.95 if representation == "pca95" else 0.99
            rank = int(np.searchsorted(variance, fraction * variance[-1])) + 1
            projection = vectors[:rank].T
            arrays.update(
                {f"projection_{h}": projection, f"singular_values_{h}": singular}
            )
            z = z @ projection
        delta = batch.future_states[:, h - 1] - current
        bias = delta.mean(0)
        weight = np.linalg.solve(
            z.T @ z / len(z) + ridge_fraction * np.eye(z.shape[1]),
            z.T @ (delta - bias) / len(z),
        )
        arrays.update(
            {
                f"mean_{h}": mean,
                f"scale_{h}": scale,
                f"bias_{h}": bias,
                f"weight_{h}": weight,
            }
        )
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite direct forecast fit")
    return DirectForecast(
        float(batch.dt_s),
        batch.past_inputs.shape[1],
        current.shape[1],
        batch.future_inputs.shape[2],
        tuple(int(h) for h in horizons),
        use_history,
        float(ridge_fraction),
        arrays,
        representation,
    )
