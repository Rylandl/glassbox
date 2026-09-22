"""Independent angular attribution checks on analytic fields, without fitting."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from scipy.spatial.transform import Rotation

from glassbox._dynamics import VehicleSequenceModel

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import _online_trace_diagnostics as diagnostic


def model_fixture(*, active):
    rng = np.random.default_rng(876)
    channels, delay, memory, width = 2, 2, 3, 5
    current = 9 + 2 * channels
    feature = (delay + 1) * current + memory
    shapes = dict(
        linear=(feature, 6),
        quadratic=(current * (current + 1) // 2, 6),
        bias=(6,),
        w1=(feature, width),
        b1=(width,),
        w2=(width, 6),
        memory=(current, memory),
        memory_bias=(memory,),
        raw_tau=(channels,),
        raw_memory_tau=(memory,),
    )
    params = {
        name: rng.normal(0, 0.08, shape) if active else np.zeros(shape)
        for name, shape in shapes.items()
    }
    params["bias"][2] += 9.80665
    params["raw_tau"][:] = [-3, -2.7]
    norms = dict(
        body_mean=np.zeros(9),
        body_scale=np.ones(9),
        motion_bound_scale=np.full(6, 4.0),
        input_mean=np.zeros(channels),
        input_scale=np.ones(channels),
        feature_scale=np.ones(feature),
        quadratic_scale=np.ones(shapes["quadratic"][0]),
        output_scale=np.ones(6),
        state_mean=np.zeros(15),
        state_scale=np.ones(15),
    )
    if active:
        for name in (
            "body_scale",
            "input_scale",
            "feature_scale",
            "quadratic_scale",
            "output_scale",
        ):
            norms[name] = np.exp(rng.normal(0, 0.2, norms[name].shape))
        norms["body_mean"] = rng.normal(0, 0.1, 9)
    return VehicleSequenceModel(0.05, 5, delay, params, norms)


def context_fixture(model):
    t = np.arange(model.history_steps + 1)
    past = np.zeros((len(t), 15))
    past[:, :3] = [3.8, -2.7, 1.9] + t[:, None] * [0.04, -0.03, 0.01]
    past[:, 3:6] = [2.3, -1.7, 0.9]
    past[:, 6:] = (
        Rotation.from_rotvec(t[:, None] * [0.01, -0.02, 0.03] + [0.2, -0.1, 0.3])
        .as_matrix()
        .reshape(-1, 9)
    )
    inputs = np.sin(np.arange(model.history_steps * 2).reshape(-1, 2)) * 0.3
    return past, inputs, np.array([0.2, -0.3])


def numpy_head_parts(model, state, applied, command, history, hidden):
    """Literal NumPy head evaluation, independent of production feature helpers."""
    p, n = model.params, model.norms
    rotation = state[6:].reshape(3, 3)
    body = np.r_[rotation.T @ state[:3], state[3:6], -rotation[2]]
    body = (body - n["body_mean"]) / n["body_scale"]
    support = n["motion_bound_scale"] / 4
    motion = body[:6].copy()
    for axis in range(6):
        if abs(motion[axis]) > support[axis]:
            motion[axis] = (
                np.sign(motion[axis])
                * support[axis]
                * (
                    1
                    + 3
                    * np.tanh((abs(motion[axis]) - support[axis]) / (3 * support[axis]))
                )
            )
    features = np.r_[
        motion,
        body[6:],
        (command - n["input_mean"]) / n["input_scale"],
        (applied - n["input_mean"]) / n["input_scale"],
    ]
    sampled = np.r_[features, (history - features).ravel(), hidden] / n["feature_scale"]
    quadratic = (
        np.array(
            [
                features[i] * features[j]
                for i in range(len(features))
                for j in range(i, len(features))
            ]
        )
        / n["quadratic_scale"]
    )
    count, memory = len(features), len(hidden)
    return (
        np.stack(
            (
                sampled[:count] @ p["linear"][:count],
                sampled[count:-memory] @ p["linear"][count:-memory],
                sampled[-memory:] @ p["linear"][-memory:],
                quadratic @ p["quadratic"],
                p["bias"],
                np.tanh(sampled @ p["w1"] + p["b1"]) @ p["w2"],
            )
        )
        * n["output_scale"]
    )


def test_head_jacobians_match_independent_physical_finite_differences():
    model = model_fixture(active=True)
    past, inputs, command = context_fixture(model)
    stages = diagnostic.integrate_stages(model, past, inputs, command)
    state, applied = stages["states"][0, 0], stages["filtered"][0, 0]
    history, hidden = stages["history"], stages["hidden"]
    with jax.enable_x64(True):
        args = (
            model.params,
            model.norms,
            state[None],
            applied[None],
            command,
            history,
            hidden,
        )
        actual = np.asarray(diagnostic._angular_head_jacobians(*args))[0]
        total = np.asarray(diagnostic._jacobians(*args))[0, 3:6]
        fields = jax.tree.map(np.asarray, diagnostic._stage_fields(*args))
    assert np.max(fields["motion_support_ratio"]) > 1
    assert np.linalg.norm(actual[3]) > 0.01  # Active quadratic and tanh paths.
    assert np.linalg.norm(actual[5]) > 1e-4
    np.testing.assert_allclose(actual.sum(axis=0), total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        fields["head_contributions"][0],
        numpy_head_parts(model, state, applied, command, history, hidden),
        rtol=1e-12,
        atol=1e-12,
    )
    expected = np.zeros((6, 3, 9))
    step = 2e-6
    for axis in range(9):
        values = []
        for sign in (-1, 1):
            changed = state.copy()
            if axis < 6:
                changed[axis] += sign * step
            else:
                tangent = np.eye(3)[axis - 6] * sign * step
                changed[6:] = (
                    state[6:].reshape(3, 3) @ Rotation.from_rotvec(tangent).as_matrix()
                ).ravel()
            values.append(
                numpy_head_parts(model, changed, applied, command, history, hidden)[
                    :, 3:6
                ]
            )
        expected[..., axis] = (values[1] - values[0]) / (2 * step)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-9)


def test_delayed_linear_feedback_sign_and_frozen_hidden_derivative():
    model = model_fixture(active=False)
    params = {k: v.copy() for k, v in model.params.items()}
    current = 13
    direct = np.diag([0.3, -0.2, 0.5])
    first = np.array([[0.7, 0.1, 0], [0, 0.2, 0.1], [0.2, 0, -0.4]])
    second = np.diag([0.2, -0.1, 0.6])
    params["linear"][3:6, 3:6] = direct.T
    params["linear"][current + 3 : current + 6, 3:6] = first.T
    params["linear"][2 * current + 3 : 2 * current + 6, 3:6] = second.T
    params["linear"][-3:, 3:6] = np.eye(3) * 2
    params["bias"][3:6] = [0.1, 0.2, 0.3]
    model = replace(model, params=params)
    state = np.r_[np.zeros(3), [0.1, -0.2, 0.3], np.eye(3).ravel()]
    history = np.full((2, current), 0.4)
    hidden = np.array([0.2, -0.3, 0.1])
    with jax.enable_x64(True):
        actual = np.asarray(
            diagnostic._angular_head_jacobians(
                model.params,
                model.norms,
                state[None],
                np.zeros((1, 2)),
                np.zeros(2),
                history,
                hidden,
            )
        )[0]
    expected = np.zeros((6, 3, 9))
    expected[0, :, 3:6] = direct
    expected[1, :, 3:6] = -first - second
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    parts = numpy_head_parts(model, state, np.zeros(2), np.zeros(2), history, hidden)
    assert np.linalg.norm(parts[2, 3:6]) > 0.1
    assert np.linalg.norm(parts[4, 3:6]) > 0.1
    np.testing.assert_array_equal(actual[[2, 4]], np.zeros((2, 3, 9)))


def test_stiff_linear_rate_field_has_exact_midpoint_polynomial_and_refines():
    model = model_fixture(active=False)
    params = {k: v.copy() for k, v in model.params.items()}
    params["linear"][5, 5] = -100
    norms = dict(model.norms, motion_bound_scale=np.full(6, 40000.0))
    model = replace(model, params=params, norms=norms)
    past = np.zeros((6, 15))
    past[:, 5] = 0.4
    past[:, 6:] = np.eye(3).ravel()
    errors = []
    for factor in diagnostic.FACTORS:
        values = diagnostic.integrate_stages(
            model, past, np.zeros((5, 2)), np.zeros(2), factor
        )
        count = 2 * factor
        h = model.dt_s / count
        expected = 0.4 * (1 - 100 * h + 0.5 * (100 * h) ** 2) ** count
        np.testing.assert_allclose(
            values["prediction"][5], expected, rtol=2e-13, atol=2e-13
        )
        errors.append(abs(expected - 0.4 * np.exp(-100 * model.dt_s)))
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < errors[0] * 1e-5


def test_angular_diagnostics_reconstruct_paths_and_preserve_legacy_output(monkeypatch):
    model = model_fixture(active=True)
    past, inputs, command = context_fixture(model)
    session = SimpleNamespace(
        model=model,
        cursor=19,
        fingerprint=lambda: "unchanged-analytic-session",
        predict=lambda x, u, future: model.rollout(x, u, future),
    )
    with jax.enable_x64(False):
        prediction = np.asarray(session.predict(past, inputs, command[None]))[0]
    monkeypatch.setattr(diagnostic, "_cache_diagnostics", lambda *args: ({}, {}))
    old_summary, old_arrays = diagnostic.diagnose_snapshot(
        session, past, inputs, command, prediction, prediction
    )
    summary, arrays = diagnostic.diagnose_snapshot(
        session, past, inputs, command, prediction, prediction, angular=True
    )
    for name, value in old_arrays.items():
        np.testing.assert_array_equal(arrays[name], value)
    for name, value in old_summary.items():
        if name != "model_calls":
            assert summary[name] == value
    assert old_summary["model_calls"]["derivative_batches"] == 1
    assert summary["model_calls"]["derivative_batches"] == 2
    assert "angular" not in old_summary
    assert arrays["native_angular_head_jacobians"].shape == (2, 2, 6, 3, 9)
    for factor in diagnostic.FACTORS:
        states = arrays[f"factor_{factor}_states"]
        contributions = arrays[f"factor_{factor}_head_contributions"]
        expected = model.dt_s / len(states) * contributions[:, 1, :, 3:6].sum(axis=0)
        actual = arrays[f"factor_{factor}_angular_head_increments"]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(
            actual.sum(axis=0),
            states[-1, 2, 3:6] - past[-1, 3:6],
            rtol=1e-10,
            atol=1e-10,
        )
    delta = arrays["native_minus_refined_angular_head_increments"]
    np.testing.assert_array_equal(
        delta,
        arrays["factor_1_angular_head_increments"]
        - arrays["factor_64_angular_head_increments"],
    )
    np.testing.assert_allclose(
        delta.sum(axis=0), arrays["rate_error_decomposition"][1], rtol=1e-10, atol=1e-10
    )
    heads = arrays["native_angular_head_jacobians"]
    np.testing.assert_allclose(
        heads.sum(axis=2),
        arrays["native_local_jacobians"][..., 3:6, :],
        rtol=1e-10,
        atol=1e-10,
    )
    for name, start in (("velocity", 0), ("rate", 3), ("attitude", 6)):
        expected = np.sqrt(np.sum(heads[..., start : start + 3] ** 2, axis=(-2, -1)))
        np.testing.assert_allclose(
            arrays[f"native_angular_head_{name}_block_norms"], expected, rtol=1e-14
        )
        np.testing.assert_allclose(
            summary["angular"]["jacobian_blocks"][name]["maximum_by_head"],
            expected.max(axis=(0, 1)),
            rtol=1e-14,
        )
