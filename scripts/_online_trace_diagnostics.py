"""Read-only physical diagnostics of captured, causal online model snapshots."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from glassbox import _dynamics as dynamics
from glassbox.online import _predict

FACTORS = (1, 2, 4, 16, 64)
HEAD_PARTS = (
    "current_linear",
    "delayed_linear",
    "hidden_linear",
    "quadratic",
    "bias",
    "nonlinear",
)
FEATURE_GROUPS = ("motion", "gravity", "issued", "filtered")
QUADRATIC_PARTS = tuple(
    f"{a}:{b}" for i, a in enumerate(FEATURE_GROUPS) for b in FEATURE_GROUPS[i:]
)


def physical_metrics(prediction, truth):
    """Vector/SO(3) errors; one sample gives a norm, multiple samples give RMSE."""
    prediction, truth = (np.asarray(v).reshape(-1, 15) for v in (prediction, truth))
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        return dict(
            velocity_rmse_m_s=float("nan"),
            body_rate_rmse_rad_s=float("nan"),
            orientation_rmse_rad=float("nan"),
        )
    error = prediction - truth
    angles = Rotation.from_matrix(
        truth[:, 6:].reshape(-1, 3, 3).transpose(0, 2, 1)
        @ prediction[:, 6:].reshape(-1, 3, 3)
    ).magnitude()
    return dict(
        velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        body_rate_rmse_rad_s=float(
            np.sqrt(np.mean(np.sum(error[:, 3:6] ** 2, axis=1)))
        ),
        orientation_rmse_rad=float(np.sqrt(np.mean(angles**2))),
    )


@partial(jax.jit, static_argnames=("delay", "dt_s", "factor"))
def _integrate(params, norms, past, inputs, command, *, delay, dt_s, factor):
    applied, history, hidden = dynamics._history(
        params, norms, past, inputs, delay, dt_s
    )
    count = factor * math.ceil(dt_s / dynamics.MAX_SUBSTEP_S)
    duration = dt_s / count
    tau = dynamics.time_constants(params)

    def head(state, filtered):
        return dynamics._head(params, norms, state, command, filtered, history, hidden)[
            0
        ]

    def advance(carry, _):
        state, applied = carry
        rotation = state[:, 6:].reshape(-1, 3, 3)
        first = head(state, applied)
        rotation_half = rotation @ dynamics.rotation_exp(0.5 * duration * state[:, 3:6])
        velocity_half = state[:, :3] + 0.5 * duration * (
            jnp.asarray(dynamics.GRAVITY)
            + jnp.einsum("bij,bj->bi", rotation, first[:, :3])
        )
        omega_half = state[:, 3:6] + 0.5 * duration * first[:, 3:]
        applied_half = command + (applied - command) * jnp.exp(-0.5 * duration / tau)
        midpoint = jnp.concatenate(
            (velocity_half, omega_half, rotation_half.reshape(-1, 9)), axis=1
        )
        middle = head(midpoint, applied_half)
        velocity_next = state[:, :3] + duration * (
            jnp.asarray(dynamics.GRAVITY)
            + jnp.einsum("bij,bj->bi", rotation_half, middle[:, :3])
        )
        following = jnp.concatenate(
            (
                velocity_next,
                state[:, 3:6] + duration * middle[:, 3:],
                (rotation @ dynamics.rotation_exp(duration * omega_half)).reshape(
                    -1, 9
                ),
            ),
            axis=1,
        )
        applied_next = command + (applied - command) * jnp.exp(-duration / tau)
        return (following, applied_next), (
            jnp.stack((state, midpoint, following), axis=1),
            jnp.stack((applied, applied_half, applied_next), axis=1),
        )

    (prediction, _), (states, filtered) = jax.lax.scan(
        advance, (past[:, -1], applied), None, length=count
    )
    return dict(
        prediction=prediction[0],
        states=states[:, 0],
        filtered=filtered[:, 0],
        history=history[0],
        hidden=hidden[0],
        command=command[0],
    )


def integrate_stages(model, past_states, past_inputs, command, factor=1):
    """Return float64 native/refined stages; history/memory stay on observation grid.

    Inputs are unbatched. ``states``/``filtered`` have axes substep, stage
    (start/midpoint/end), coordinate. Ambient JAX precision is restored on exit.
    """
    if type(factor) is not int or factor < 1:
        raise ValueError("refinement factor must be a positive integer")
    with jax.enable_x64(True):
        params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
        values = _integrate(
            params,
            norms,
            jnp.asarray(past_states, dtype=jnp.float64)[None],
            jnp.asarray(past_inputs, dtype=jnp.float64)[None],
            jnp.asarray(command, dtype=jnp.float64)[None],
            delay=model.delay_steps,
            dt_s=model.dt_s,
            factor=factor,
        )
        return jax.tree.map(np.asarray, values)


@jax.jit
def _stage_fields(params, norms, states, filtered, command, history, hidden):
    count, channels = len(states), len(command)
    commands = jnp.broadcast_to(command, (count, channels))
    histories = jnp.broadcast_to(history, (count, *history.shape))
    memories = jnp.broadcast_to(hidden, (count, len(hidden)))
    features = dynamics.current_features(states, commands, filtered, norms)
    sampled = (
        dynamics.sampled_features(features, histories, memories)
        / norms["feature_scale"]
    )
    nonlinear = (
        dynamics.nonlinear_features(
            dynamics.sampled_features(features, histories, memories),
            features.shape[-1],
            history.shape[0],
        )
        / norms["nonlinear_scale"]
    )
    quadratic = dynamics.quadratic_features(features) / norms["quadratic_scale"]
    current, memory = features.shape[1], hidden.shape[0]
    contribution = (
        jnp.stack(
            (
                sampled[:, :current] @ params["linear"][:current],
                sampled[:, current:-memory] @ params["linear"][current:-memory],
                sampled[:, -memory:] @ params["linear"][-memory:],
                quadratic @ params["quadratic"],
                jnp.broadcast_to(params["bias"], (count, 6)),
                jnp.tanh(nonlinear @ params["w1"] + params["b1"]) @ params["w2"],
            ),
            axis=1,
        )
        * norms["output_scale"]
    )
    groups = np.r_[
        np.zeros(6, int), np.ones(3, int), np.full(channels, 2), np.full(channels, 3)
    ]
    left, right = np.triu_indices(current)
    quadratic_parts = (
        jnp.stack(
            [
                (quadratic * jnp.asarray((groups[left] == a) & (groups[right] == b)))
                @ params["quadratic"]
                for a in range(4)
                for b in range(a, 4)
            ],
            axis=1,
        )
        * norms["output_scale"]
    )
    body = dynamics._body_features(states)
    normalized = (body - norms["body_mean"]) / norms["body_scale"]
    support = norms["motion_bound_scale"] / 4
    excess = jnp.maximum(jnp.abs(normalized[:, :6]) - support, 0)
    derivative = 1 - jnp.tanh(excess / (3 * support)) ** 2
    acceleration = dynamics._head(
        params, norms, states, commands, filtered, histories, memories
    )[0]
    rotation = states[:, 6:].reshape(-1, 3, 3)
    return dict(
        body_features=body,
        normalized_body_features=normalized,
        motion_support_ratio=jnp.abs(normalized[:, :6]) / support,
        support_derivative=derivative,
        features=features,
        scaled_sampled_features=sampled,
        scaled_quadratic_features=quadratic,
        head_contributions=contribution,
        quadratic_contributions=quadratic_parts,
        body_acceleration=acceleration[:, :3],
        angular_acceleration=acceleration[:, 3:],
        world_acceleration=jnp.asarray(dynamics.GRAVITY)
        + jnp.einsum("bij,bj->bi", rotation, acceleration[:, :3]),
    )


@jax.jit
def _jacobians(params, norms, states, filtered, command, history, hidden):
    def local(state, applied):
        rotation = state[6:].reshape(3, 3)

        def field(delta):
            changed_rotation = rotation @ dynamics.rotation_exp(delta[6:])
            changed = jnp.concatenate(
                (state[:6] + delta[:6], changed_rotation.reshape(9))
            )
            acceleration = dynamics._head(
                params,
                norms,
                changed[None],
                command[None],
                applied[None],
                history[None],
                hidden[None],
            )[0][0]
            return jnp.concatenate(
                (
                    jnp.asarray(dynamics.GRAVITY) + changed_rotation @ acceleration[:3],
                    acceleration[3:],
                    changed[3:6] - jnp.cross(state[3:6], delta[6:]),
                )
            )

        return jax.jacfwd(field)(jnp.zeros(9))

    return jax.vmap(local)(states, filtered)


@jax.jit
def _angular_head_jacobians(params, norms, states, filtered, command, history, hidden):
    """Angular head derivatives with frozen filter/history/memory context.

    The last axes are head, angular output, and physical local coordinate
    (world velocity, body rate, right attitude tangent). The derivative of the
    hidden-linear head is zero here even when its acceleration is substantial.
    """

    def local(state, applied):
        rotation = state[6:].reshape(3, 3)

        def parts(delta):
            changed = jnp.concatenate(
                (
                    state[:6] + delta[:6],
                    (rotation @ dynamics.rotation_exp(delta[6:])).reshape(9),
                )
            )
            fields = _stage_fields(
                params, norms, changed[None], applied[None], command, history, hidden
            )
            return fields["head_contributions"][0, :, 3:6]

        return jax.jacfwd(parts)(jnp.zeros(9, dtype=state.dtype))

    return jax.vmap(local)(states, filtered)


def _angular_diagnostics(resolutions, jacobian, params, norms, dt_s):
    """Account for rate increments along saved paths and local head derivatives."""
    arrays, reconstruction = {}, {}
    increments = {}
    for factor, values in resolutions.items():
        step = dt_s / len(values["states"])
        increment = step * values["head_contributions"][:, 1, :, 3:6].sum(axis=0)
        change = values["prediction"][3:6] - values["states"][0, 0, 3:6]
        np.testing.assert_allclose(
            increment.sum(axis=0), change, rtol=1e-10, atol=1e-10, equal_nan=False
        )
        increments[factor] = increment
        arrays[f"factor_{factor}_angular_head_increments"] = increment
        reconstruction[str(factor)] = float(
            np.max(np.abs(increment.sum(axis=0) - change))
        )
    native, refined = resolutions[1], resolutions[64]
    difference = increments[1] - increments[64]
    rate_difference = native["prediction"][3:6] - refined["prediction"][3:6]
    np.testing.assert_allclose(
        difference.sum(axis=0), rate_difference, rtol=1e-10, atol=1e-10, equal_nan=False
    )
    arrays["native_minus_refined_angular_head_increments"] = difference
    head_jacobian = np.asarray(
        _angular_head_jacobians(
            params,
            norms,
            jnp.asarray(native["states"][:, :2].reshape(-1, 15)),
            jnp.asarray(native["filtered"][:, :2].reshape(-1, len(native["command"]))),
            jnp.asarray(native["command"]),
            jnp.asarray(native["history"]),
            jnp.asarray(native["hidden"]),
        )
    )
    angular_jacobian = jacobian[:, 3:6, :]
    np.testing.assert_allclose(
        head_jacobian.sum(axis=1),
        angular_jacobian,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=False,
    )
    shaped = head_jacobian.reshape(-1, 2, len(HEAD_PARTS), 3, 9)
    arrays["native_angular_head_jacobians"] = shaped
    block_summary = {}
    for name, start, unit in (
        ("velocity", 0, "(rad/s^2)/(m/s)"),
        ("rate", 3, "(rad/s^2)/(rad/s)"),
        ("attitude", 6, "(rad/s^2)/rad"),
    ):
        block_norm = np.linalg.norm(shaped[..., start : start + 3], axis=(-2, -1))
        arrays[f"native_angular_head_{name}_block_norms"] = block_norm
        block_summary[name] = dict(
            units=unit, maximum_by_head=block_norm.max(axis=(0, 1)).tolist()
        )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite angular increment or head derivative diagnostic")
    summary = dict(
        increment_units="rad/s",
        increment_reconstruction_max_abs=reconstruction,
        native_minus_refined_reconstruction_max_abs=float(
            np.max(np.abs(difference.sum(axis=0) - rate_difference))
        ),
        jacobian_reconstruction_max_abs=float(
            np.max(np.abs(head_jacobian.sum(axis=1) - angular_jacobian))
        ),
        jacobian_coordinate_order=("world_velocity", "body_rate", "right_attitude"),
        jacobian_blocks=block_summary,
        scope="Signed increments account for the integrated path, not causal head error; removing a head changes that path and parts may cancel. Jacobians freeze issued command, filter, history and hidden state. Zero hidden-linear derivative does not imply irrelevant memory; the nonlinear head mixes contexts. No combined norm of unlike coordinate units or stability claim.",
    )
    return summary, arrays


def _decomposition(native, refined, truth, part):
    error = native[part] - truth[part]
    numerical = native[part] - refined[part]
    remainder = refined[part] - truth[part]
    return dict(
        native_error_squared=float(error @ error),
        numerical_difference_squared=float(numerical @ numerical),
        refined_error_squared=float(remainder @ remainder),
        cross_term=float(2 * numerical @ remainder),
        reconstruction_max_abs=float(np.max(np.abs(error - numerical - remainder))),
    )


def _cache_diagnostics(session, model, query, command, query_applied):
    summary, arrays, records = {}, {}, []
    for role in ("bootstrap", "recent"):
        data = getattr(session, "_" + role)
        if not len(data["past_states"]):
            continue
        observed = np.concatenate(
            (data["past_states"], data["future_states"][:, :-1]), axis=1
        )
        commands = np.concatenate((data["past_inputs"], data["future_inputs"]), axis=1)
        body = np.asarray(dynamics._body_features(jnp.asarray(observed))).reshape(-1, 9)
        motion = (body[:, :6] - model.norms["body_mean"][:6]) / model.norms[
            "body_scale"
        ][:6]
        compressed = np.asarray(
            dynamics.supported_motion(
                jnp.asarray(motion), model.norms["motion_bound_scale"]
            )
        )
        issued = commands.reshape(-1, commands.shape[-1])
        tau = 0.001 + np.logaddexp(0.0, model.params["raw_tau"])
        applied = commands[:, 0].copy()
        filtered = np.empty_like(commands)
        for index in range(commands.shape[1]):
            filtered[:, index] = applied
            applied = commands[:, index] + (applied - commands[:, index]) * np.exp(
                -model.dt_s / tau
            )
        features = np.asarray(
            dynamics.current_features(
                jnp.asarray(observed),
                jnp.asarray(commands),
                jnp.asarray(filtered),
                model.norms,
            )
        ).reshape(-1, 9 + 2 * commands.shape[-1])
        features = features / model.norms["feature_scale"][: features.shape[-1]]
        prediction = np.asarray(
            _predict(
                model.params,
                model.norms,
                *[
                    jnp.asarray(data[key])
                    for key in ("past_states", "past_inputs", "future_inputs")
                ],
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
        )
        summary[role] = dict(
            windows=len(data["past_states"]),
            observed_context_rows=len(motion),
            retained_forecast_errors=physical_metrics(
                prediction, data["future_states"]
            ),
        )
        for name, value in dict(
            normalized_motion=motion,
            compressed_motion=compressed,
            commands=issued,
            scaled_current_features=features,
            retained_prediction=prediction,
            retained_truth=data["future_states"],
        ).items():
            arrays[f"cache_{role}_{name}"] = value
        records.append((role, motion, compressed, issued, features))
    query_body = np.asarray(dynamics._body_features(jnp.asarray(query)))
    raw = (query_body[:6] - model.norms["body_mean"][:6]) / model.norms["body_scale"][
        :6
    ]
    compressed_query = np.asarray(
        dynamics.supported_motion(jnp.asarray(raw), model.norms["motion_bound_scale"])
    )
    query_features = np.asarray(
        dynamics.current_features(
            jnp.asarray(query),
            jnp.asarray(command),
            jnp.asarray(query_applied),
            model.norms,
        )
    )
    query_features = (
        query_features / model.norms["feature_scale"][: len(query_features)]
    )
    arrays["query_scaled_current_features"] = query_features
    all_issued, all_weights = [], []
    for role, motion, compressed, issued, features in records:
        normalized = (issued - model.norms["input_mean"]) / model.norms["input_scale"]
        query_command = (command - model.norms["input_mean"]) / model.norms[
            "input_scale"
        ]
        values = dict(
            raw_motion=np.linalg.norm(motion - raw, axis=1),
            compressed_motion=np.linalg.norm(compressed - compressed_query, axis=1),
            command=np.linalg.norm(normalized - query_command, axis=1),
            scaled_current_features=np.linalg.norm(features - query_features, axis=1),
            raw_joint=np.linalg.norm(
                np.concatenate((motion - raw, normalized - query_command), axis=1),
                axis=1,
            ),
            compressed_joint=np.linalg.norm(
                np.concatenate(
                    (compressed - compressed_query, normalized - query_command), axis=1
                ),
                axis=1,
            ),
        )
        summary[role]["nearest_distances"] = {
            name: float(value.min()) for name, value in values.items()
        }
        summary[role]["query_within_motion_axis_ranges"] = (
            (raw >= motion.min(axis=0)) & (raw <= motion.max(axis=0))
        ).tolist()
        summary[role]["query_within_command_axis_ranges"] = (
            (command >= issued.min(axis=0)) & (command <= issued.max(axis=0))
        ).tolist()
        all_issued.append(issued)
        all_weights.append(np.full(len(issued), 1 / (len(records) * len(issued))))
    issued, weights = np.concatenate(all_issued), np.concatenate(all_weights)
    normalized = (issued - model.norms["input_mean"]) / model.norms["input_scale"]
    mean = weights @ issued
    centered = normalized - weights @ normalized
    singular = np.linalg.svd(np.sqrt(weights[:, None]) * centered, compute_uv=False)
    summary["role_balanced_commands"] = dict(
        mean=mean.tolist(),
        standard_deviation=np.sqrt(weights @ ((issued - mean) ** 2)).tolist(),
        span=np.ptp(issued, axis=0).tolist(),
        minimum=issued.min(axis=0).tolist(),
        maximum=issued.max(axis=0).tolist(),
        normalized_singular_values=singular.tolist(),
        normalized_rank=int(
            np.linalg.matrix_rank(np.sqrt(weights[:, None]) * centered)
        ),
        unique_rows=len(np.unique(issued, axis=0)),
        query_normalized=(
            (command - model.norms["input_mean"]) / model.norms["input_scale"]
        ).tolist(),
    )
    arrays["cache_role_balanced_weights"] = weights
    summary["weighting"] = (
        "Equal nonempty roles, windows and times within each role; overlapping contexts retained as in measured-cache conditioning, not independent samples. Motion/command distances use original normalization before/after compression. Full current-feature distance includes gravity and reconstructed filtered commands divided by current feature_scale."
    )
    return summary, arrays


def diagnose_snapshot(
    session,
    past_states,
    past_inputs,
    command,
    truth,
    recorded_prediction,
    *,
    angular=False,
):
    """Inspect one captured causal model without observing, initializing or fitting."""
    before = session.fingerprint()
    model = session.model
    past_states, past_inputs, command, truth, recorded = (
        np.asarray(v)
        for v in (past_states, past_inputs, command, truth, recorded_prediction)
    )
    truth, recorded = truth.reshape(15), recorded.reshape(15)
    with jax.enable_x64(False):
        reproduced = np.asarray(
            session.predict(past_states, past_inputs, command[None])
        )[0]
    if (
        reproduced.dtype != np.float32
        or recorded.dtype != np.float32
        or reproduced.tobytes() != recorded.tobytes()
    ):
        raise ValueError(
            "captured model does not exactly reproduce its recorded float32 prediction"
        )
    arrays = dict(
        recorded_prediction=recorded,
        reproduced_prediction=reproduced,
        revealed_truth=truth,
    )
    summary = dict(
        model_fingerprint=model.fingerprint,
        cursor=session.cursor,
        precision="exact public float32 replay; instrumented dynamics and derivative diagnostics float64",
        head_part_order=HEAD_PARTS,
        quadratic_part_order=QUADRATIC_PARTS,
        stage_order=("start", "midpoint", "end"),
        acceleration_units="body_acceleration is learned non-gravitational specific acceleration (m/s^2); world_acceleration includes gravity; angular_acceleration is body angular acceleration (rad/s^2)",
        resolutions={},
    )
    with jax.enable_x64(True):
        params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
        core = np.asarray(
            _predict(
                params,
                norms,
                jnp.asarray(past_states, dtype=jnp.float64)[None],
                jnp.asarray(past_inputs, dtype=jnp.float64)[None],
                jnp.asarray(command, dtype=jnp.float64)[None, None],
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
        )[0, 0]
        resolutions = {}
        for factor in FACTORS:
            values = integrate_stages(model, past_states, past_inputs, command, factor)
            count = len(values["states"])
            fields = jax.tree.map(
                np.asarray,
                _stage_fields(
                    params,
                    norms,
                    jnp.asarray(values["states"].reshape(-1, 15)),
                    jnp.asarray(values["filtered"].reshape(-1, len(command))),
                    jnp.asarray(command),
                    jnp.asarray(values["history"]),
                    jnp.asarray(values["hidden"]),
                ),
            )
            for name, value in fields.items():
                values[name] = value.reshape(count, 3, *value.shape[1:])
            parts = values["head_contributions"].sum(axis=-2)
            np.testing.assert_allclose(
                parts,
                np.concatenate(
                    (values["body_acceleration"], values["angular_acceleration"]),
                    axis=-1,
                ),
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                values["quadratic_contributions"].sum(axis=-2),
                values["head_contributions"][..., 3, :],
                rtol=1e-10,
                atol=1e-10,
            )
            resolutions[factor] = values
            arrays.update(
                {f"factor_{factor}_{name}": value for name, value in values.items()}
            )
            summary["resolutions"][str(factor)] = dict(
                substeps=count,
                substep_s=model.dt_s / count,
                truth_errors=physical_metrics(values["prediction"], truth),
                difference_from_native=physical_metrics(values["prediction"], core),
                maximum_motion_support_ratio=float(
                    np.max(values["motion_support_ratio"])
                ),
                minimum_support_derivative=float(np.min(values["support_derivative"])),
                maximum_world_acceleration=float(
                    np.max(np.linalg.norm(values["world_acceleration"], axis=-1))
                ),
                maximum_angular_acceleration=float(
                    np.max(np.linalg.norm(values["angular_acceleration"], axis=-1))
                ),
            )
        native = resolutions[1]
        np.testing.assert_allclose(native["prediction"], core, rtol=1e-10, atol=1e-10)
        summary["native_reconstruction_max_abs"] = float(
            np.max(np.abs(native["prediction"] - core))
        )
        summary["float64_vs_recorded_float32"] = physical_metrics(core, recorded)
        summary["convergence_16_to_64"] = physical_metrics(
            resolutions[16]["prediction"], resolutions[64]["prediction"]
        )
        summary["error_decomposition"] = {
            name: _decomposition(core, resolutions[64]["prediction"], truth, part)
            for name, part in (("velocity", slice(0, 3)), ("body_rate", slice(3, 6)))
        }
        arrays["velocity_error_decomposition"] = np.stack(
            (
                core[:3] - truth[:3],
                core[:3] - resolutions[64]["prediction"][:3],
                resolutions[64]["prediction"][:3] - truth[:3],
            )
        )
        arrays["rate_error_decomposition"] = np.stack(
            (
                core[3:6] - truth[3:6],
                core[3:6] - resolutions[64]["prediction"][3:6],
                resolutions[64]["prediction"][3:6] - truth[3:6],
            )
        )
        jacobian = np.asarray(
            _jacobians(
                params,
                norms,
                jnp.asarray(native["states"][:, :2].reshape(-1, 15)),
                jnp.asarray(native["filtered"][:, :2].reshape(-1, len(command))),
                jnp.asarray(command),
                jnp.asarray(native["history"]),
                jnp.asarray(native["hidden"]),
            )
        )
        finite_jacobian = np.isfinite(jacobian).all(axis=(1, 2))
        eigenvalues = np.full((len(jacobian), 9), complex("nan"))
        eigenvalues[finite_jacobian] = np.linalg.eigvals(jacobian[finite_jacobian])
        h = model.dt_s / len(native["states"])
        amplification = np.abs(1 + h * eigenvalues + 0.5 * (h * eigenvalues) ** 2)
        arrays["native_local_jacobians"] = jacobian.reshape(-1, 2, 9, 9)
        arrays["native_local_eigenvalues_real_imag"] = np.stack(
            (eigenvalues.real, eigenvalues.imag), axis=-1
        ).reshape(-1, 2, 9, 2)
        arrays["native_midpoint_linear_amplification"] = amplification.reshape(-1, 2, 9)
        summary["local_derivatives"] = dict(
            minimum_real_eigenvalue=float(eigenvalues.real.min()),
            maximum_real_eigenvalue=float(eigenvalues.real.max()),
            maximum_midpoint_amplification=float(amplification.max()),
            scope="Frozen history/filter/memory local field; moving right-attitude tangent. Midpoint amplification is the scalar stability polynomial of local eigenvalues, not the full Lie-group step derivative or recurrent/path stability.",
        )
        if angular:
            summary["angular"], angular_arrays = _angular_diagnostics(
                resolutions, jacobian, params, norms, model.dt_s
            )
            arrays.update(angular_arrays)
        summary["measured_interval_mean"] = dict(
            world_acceleration=(
                (truth[:3] - past_states[-1, :3]) / model.dt_s
            ).tolist(),
            angular_acceleration=(
                (truth[3:6] - past_states[-1, 3:6]) / model.dt_s
            ).tolist(),
            limitation="Observed secants are interval means, not measured instantaneous derivatives.",
        )
        summary["cache"], cache_arrays = _cache_diagnostics(
            session, model, past_states[-1], command, native["filtered"][0, 0]
        )
        arrays.update(cache_arrays)
    summary["model_calls"] = dict(
        public_float32_predictions=1,
        core_float64_predictions=1,
        instrumented_resolutions=len(FACTORS),
        cache_rollouts=len(
            [r for r in ("bootstrap", "recent") if r in summary["cache"]]
        ),
        derivative_batches=1 + int(angular),
        optimizer_steps=0,
    )
    if session.fingerprint() != before:
        raise RuntimeError("diagnostics mutated the captured session")
    return summary, arrays
