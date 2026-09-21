"""Causal online fitting on independent analytic motion, without simulator calls."""

from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox import (
    STATE_CHANNELS,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
    online,
)
from glassbox._learner_arrays import load_arrays, save_arrays


def stream(commands=3, steps=96, *, change=True, dt=0.05):
    """A delayed linear plant, not the learner's feature/integration equations."""
    rng = np.random.default_rng(915)
    issued = rng.uniform(-0.4, 0.4, (steps, commands))
    response = np.zeros(commands)
    velocity = np.array([0.2, -0.15, 0.1])
    states = np.zeros((steps + 1, 15))
    states[:, 6:] = np.eye(3).ravel()
    states[0, :3] = velocity
    mixing = np.array(
        [
            [1.2, -0.3, 0.2, 0.1, 0.05],
            [0.1, 0.8, -0.2, -0.2, -0.3],
            [-0.2, 0.1, 0.9, 0.3, 0.2],
        ]
    )[:, :commands]
    for index in range(steps):
        delayed = issued[index - 2] if index >= 2 else np.zeros(commands)
        response = 0.65 * response + 0.35 * delayed
        disturbance = np.array([0.4, -0.25, 0.3]) if change and index >= 15 else 0.0
        velocity = velocity + dt * (mixing @ response - 1.5 * velocity + disturbance)
        states[index + 1, :3] = velocity
    return states, issued


def prefix(states, commands, *, offset=73, dt=0.05):
    intervals = max(round(0.5 / dt), round(0.1 / dt) + 1) + round(0.25 / dt)
    return SequenceCollection(
        (
            SequenceSegment(
                "analytic-stream",
                "continuous",
                states[: intervals + 1],
                commands[:intervals],
                dt,
                offset,
            ),
        ),
        "opaque-configuration",
        STATE_CHANNELS,
        tuple(
            f"command-{index} [issued,unitless]" for index in range(commands.shape[1])
        ),
    )


def predict_args(session, states, commands, row):
    history = session.report["history_steps"]
    return (
        states[row - history : row + 1],
        commands[row - history : row],
        commands[row : row + 1],
    )


def snapshot(session):
    return (
        session.fingerprint(),
        deepcopy(session.report),
        session.model.fingerprint,
        session.cursor,
    )


@pytest.mark.parametrize("commands", [3, 4])
def test_same_public_api_initializes_and_learns_arbitrary_command_count(commands):
    states, issued = stream(commands)
    before_precision = jax.config.x64_enabled
    session = OnlineFit(prefix(states, issued))
    assert jax.config.x64_enabled == before_precision
    assert session.cursor == 73 + 15
    assert session.report["history_steps"] == 10
    assert session.report["training_horizon_steps"] == 1
    initial = session.model
    assert initial.params["raw_tau"].shape == (commands,)
    with jax.enable_x64(False):
        args = predict_args(session, states, issued, 15)
        prediction = session.predict(*args)
        assert prediction.shape == (1, 15) and prediction.dtype == np.float32
        np.testing.assert_allclose(
            prediction, initial.rollout(*args), rtol=1e-6, atol=2e-7
        )
        session.observe(session.cursor, issued[15], states[16])
    assert session.cursor == 73 + 16
    assert session.report["observations"] == 1
    assert session.report["gradient_calls"] == 1
    assert session.report["optimizer_steps"] == 1
    assert session.report["cg_iterations"] == 4
    assert session.report["curvature_calls"] == 4
    assert session.report["objective_calls"] == 2
    assert jax.config.x64_enabled == before_precision
    for name, value in initial.norms.items():
        np.testing.assert_array_equal(session.model.norms[name], value)
    for value in session.model.arrays().values():
        assert value.dtype == np.float64 and np.isfinite(value).all()


def test_input_errors_are_rejected_before_optimizer_or_session_mutation(monkeypatch):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    reflection = states[16].copy()
    reflection[6] = -1
    bad_observation = states[16].copy()
    bad_observation[0] = np.inf
    invalid = [
        (session.cursor - 1, issued[15], states[16]),
        (session.cursor + 1, issued[15], states[16]),
        (float(session.cursor), issued[15], states[16]),
        (session.cursor, np.full(3, np.nan), states[16]),
        (session.cursor, issued[15, :2], states[16]),
        (session.cursor, issued[15], states[16, :14]),
        (session.cursor, issued[15], bad_observation),
        (session.cursor, issued[15], reflection),
    ]

    def unexpected(*args, **kwargs):
        pytest.fail("invalid observation reached the optimizer")

    monkeypatch.setattr(online, "_proposal", unexpected)
    before = snapshot(session)
    for args in invalid:
        with pytest.raises((ValueError, TypeError)):
            session.observe(*args)
        assert snapshot(session) == before


def test_duplicate_observation_cannot_train_twice():
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    index = session.cursor
    session.observe(index, issued[15], states[16])
    before = snapshot(session)
    with pytest.raises((ValueError, TypeError)):
        session.observe(index, issued[15], states[16])
    assert snapshot(session) == before


def test_nonfinite_proposal_preserves_parameters_and_ingests_once(monkeypatch):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    initial = session.model.fingerprint

    def nonfinite(params, norms, data, scale, weights, damping, **kwargs):
        proposal = dict(params)
        proposal["bias"] = jnp.full_like(params["bias"], jnp.nan)
        return proposal, jnp.asarray(1.0), jnp.nan, jnp.nan, False

    with monkeypatch.context() as patch:
        patch.setattr(online, "_proposal", nonfinite)
        session.observe(session.cursor, issued[15], states[16])
    assert session.model.fingerprint == initial
    assert session.cursor == 73 + 16
    assert session.report["observations"] == 1
    assert session.report["accepted_proposals"] == 0
    assert session.report["damping"] == 4.0
    session.observe(session.cursor, issued[16], states[17])
    assert session.report["observations"] == 2


def test_saved_resume_preserves_optimizer_history_and_future_revisions(tmp_path):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    for row in range(15, 19):
        session.observe(session.cursor, issued[row], states[row + 1])
    path = tmp_path / "session.npz"
    session.save(path)
    resumed = OnlineFit.load(path)
    assert snapshot(resumed) == snapshot(session)
    for row in range(19, 23):
        args = predict_args(session, states, issued, row)
        np.testing.assert_array_equal(session.predict(*args), resumed.predict(*args))
        session.observe(session.cursor, issued[row], states[row + 1])
        resumed.observe(resumed.cursor, issued[row], states[row + 1])
        assert snapshot(resumed) == snapshot(session)
    assert session.report["accepted_proposals"] > 0


def retained_loss(model, session):
    """Independent equal-role Huber score over actual, unpadded observations."""
    values = []
    with jax.enable_x64(True):
        for windows in (session._bootstrap, session._recent):
            prediction = np.asarray(
                model.rollout(
                    windows["past_states"],
                    windows["past_inputs"],
                    windows["future_inputs"],
                )
            )
            residual = np.abs((prediction - windows["future_states"]) / session._scale)
            values.append(
                np.where(residual <= 1, 0.5 * residual**2, residual - 0.5).mean()
            )
    return float(np.mean(values))


def test_delayed_motion_adaptation_is_causal_and_retention_stays_bounded(tmp_path):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    frozen = session.model
    initial_norms = {name: value.copy() for name, value in frozen.norms.items()}
    initial_fingerprint = frozen.fingerprint
    errors, frozen_errors, archive_sizes = [], [], []
    for row in range(15, 79):
        previous = session.model if row in (15, 46, 78) else None
        args = predict_args(session, states, issued, row)
        before = session.fingerprint()
        prediction = np.asarray(session.predict(*args))
        reference = np.asarray(frozen.rollout(*args))
        assert session.fingerprint() == before  # Prediction must never assimilate.
        errors.append(np.linalg.norm(prediction[0, :3] - states[row + 1, :3]))
        frozen_errors.append(np.linalg.norm(reference[0, :3] - states[row + 1, :3]))
        session.observe(session.cursor, issued[row], states[row + 1])
        report = session.report
        if previous is not None:
            # Score both revisions on exactly the same now-completed full cache.
            old_loss = retained_loss(previous, session)
            new_loss = retained_loss(session.model, session)
            assert new_loss <= old_loss + 1e-9 * max(1.0, old_loss)
        assert report["bootstrap_windows"] <= 32 and report["recent_windows"] <= 32
        assert (
            report["tail_rows"]
            <= report["history_steps"] + report["training_horizon_steps"] + 1
        )
        if row in (46, 78):
            path = tmp_path / f"session-{row}.npz"
            session.save(path)
            _metadata, arrays = load_arrays(path)
            archive_sizes.append(
                (
                    {name: value.shape for name, value in arrays.items()},
                    sum(value.nbytes for value in arrays.values()),
                )
            )
    assert archive_sizes[0] == archive_sizes[1]
    assert session.report["observations"] == 64
    assert session.report["gradient_calls"] == 64
    assert session.report["accepted_proposals"] > 0
    assert session.model.fingerprint != initial_fingerprint
    assert frozen.fingerprint == initial_fingerprint
    for name, expected in initial_norms.items():
        np.testing.assert_array_equal(session.model.norms[name], expected)
    # The changed plant is scored before each corresponding observation is released.
    # This analytic capability assertion is deliberately weaker than the physical gate.
    actual = np.sqrt(np.mean(np.square(errors[-16:])))
    reference = np.sqrt(np.mean(np.square(frozen_errors[-16:])))
    assert actual < 0.95 * reference, (actual, reference)


def test_model_snapshot_cannot_modify_session():
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    before = snapshot(session)
    exported = session.model
    assert not exported.params["bias"].flags.writeable
    exported.params["bias"] = np.full(6, 900.0)
    assert snapshot(session) == before
    report = session.report
    report["observations"] = 900
    assert snapshot(session) == before


def test_prefix_identity_and_contiguity_contract():
    states, issued = stream()
    data = prefix(states, issued)
    with pytest.raises(ValueError):
        OnlineFit(replace(data, state_channels=tuple(reversed(STATE_CHANNELS))))
    short = replace(data.segments[0], states=states[:12], inputs=issued[:11])
    with pytest.raises(ValueError):
        OnlineFit(replace(data, segments=(short,)))
    other = replace(data.segments[0], recording_id="another")
    with pytest.raises(ValueError):
        OnlineFit(replace(data, segments=(data.segments[0], other)))


def test_session_archive_rejects_numeric_and_metadata_tampering(tmp_path):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    path = tmp_path / "session.npz"
    session.save(path)
    with np.load(path, allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    name = next(
        name
        for name, value in raw.items()
        if name != "metadata" and np.issubdtype(value.dtype, np.floating) and value.size
    )
    raw[name] = raw[name].copy()
    raw[name].flat[0] += 1.0
    np.savez_compressed(tmp_path / "altered.npz", **raw)
    with pytest.raises(ValueError, match="fingerprint"):
        OnlineFit.load(tmp_path / "altered.npz")
    metadata, arrays = load_arrays(path)
    metadata["undeclared"] = True
    save_arrays(tmp_path / "repaired.npz", metadata, arrays)
    with pytest.raises(ValueError):
        OnlineFit.load(tmp_path / "repaired.npz")


@pytest.mark.parametrize("commands", [1, 5])
def test_prediction_gradients_causality_and_validation(commands):
    states, issued = stream(commands)
    session = OnlineFit(prefix(states, issued))
    x, up, _ = predict_args(session, states, issued, 15)
    with jax.enable_x64(False):
        future = jnp.asarray(issued[15:18])
        before = session.fingerprint()
        forward = jax.jit(lambda u: session.predict(x, up, u))
        prediction = forward(future)
        gradient = jax.jit(jax.grad(lambda u: jnp.sum(forward(u)[-1, :3])))(future)
        assert gradient.shape == future.shape and np.isfinite(gradient).all()
        assert np.any(np.abs(gradient) > 1e-9)
        np.testing.assert_array_equal(
            prediction[:2], forward(future.at[2].add(0.1))[:2]
        )
        assert session.fingerprint() == before
        with pytest.raises(ValueError):
            session.predict(x, up, np.full((1, commands), np.nan))
        with pytest.raises(ValueError):
            session.predict(x[:-1], up, future)
        with pytest.raises(ValueError):
            session.predict(x, up, np.zeros((1, commands + 1)))


def test_rejection_preserves_parameters_and_damping_survives_resume(
    tmp_path, monkeypatch
):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    initial = session.model.fingerprint
    actual_proposal = online._proposal
    calls = []

    def rejected(*args, **kwargs):
        proposal, current, trial, predicted, finite = actual_proposal(*args, **kwargs)
        assert bool(finite)
        assert np.isfinite([current, trial, predicted]).all()
        calls.append(float(predicted))
        # Exercise the real CG solve but reject a finite, non-improving forecast.
        return proposal, current, current, predicted, finite

    with monkeypatch.context() as patch:
        patch.setattr(online, "_proposal", rejected)
        session.observe(session.cursor, issued[15], states[16])
    assert len(calls) == 1
    assert session.model.fingerprint == initial
    assert session.report["optimizer_steps"] == 1
    assert session.report["objective_calls"] == 2
    assert session.report["cg_iterations"] == 4
    assert session.report["curvature_calls"] == 4
    assert session.report["accepted_proposals"] == 0
    assert session.report["damping"] == 4.0
    path = tmp_path / "rejected.npz"
    session.save(path)
    metadata, arrays = load_arrays(path)
    assert metadata["damping"] == 4.0
    assert not any(name.startswith(("first_", "second_")) for name in arrays)
    resumed = OnlineFit.load(path)
    assert snapshot(resumed) == snapshot(session)
    for row in (16, 17):
        session.observe(session.cursor, issued[row], states[row + 1])
        resumed.observe(resumed.cursor, issued[row], states[row + 1])
        assert snapshot(resumed) == snapshot(session)


@pytest.mark.parametrize(
    "gain,expected_damping,accepted",
    [
        (0.05, 4.0, False),
        (0.2, 4.0, True),
        (0.5, 1.0, True),
        (0.9, 0.5, True),
    ],
)
def test_damping_tracks_actual_to_predicted_decrease(
    monkeypatch, gain, expected_damping, accepted
):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    initial = session.model.fingerprint

    def controlled(params, norms, data, scale, weights, damping, **kwargs):
        proposal = dict(params)
        proposal["bias"] = params["bias"] + 1e-9
        predicted = 0.25
        return proposal, 1.0, 1.0 - gain * predicted, predicted, True

    monkeypatch.setattr(online, "_proposal", controlled)
    session.observe(session.cursor, issued[15], states[16])
    assert session.report["damping"] == expected_damping
    assert session.report["accepted_proposals"] == int(accepted)
    assert (session.model.fingerprint != initial) == accepted
    assert session.cursor == 73 + 16


def test_repaired_archive_cannot_break_causal_or_optimizer_consistency(tmp_path):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    path = tmp_path / "good.npz"
    session.save(path)
    mutations = [
        lambda m, a: m.update(cursor=m["cursor"] + 1),
        lambda m, a: m.update(damping=-1.0),
        lambda m, a: m["counts"].update(cg_iterations=3),
        lambda m, a: a["tail_states"].__setitem__(
            (-1, 0), a["tail_states"][-1, 0] + 0.1
        ),
    ]
    for index, mutate in enumerate(mutations):
        metadata, arrays = load_arrays(path)
        mutate(metadata, arrays)
        changed = tmp_path / f"altered-{index}.npz"
        save_arrays(changed, metadata, arrays)
        with pytest.raises(ValueError):
            OnlineFit.load(changed)


def test_10ms_stream_uses_only_completed_50ms_training_targets(tmp_path):
    states, issued = stream(4, change=False, dt=0.01)
    session = OnlineFit(prefix(states, issued, dt=0.01))
    assert session.report["history_steps"] == 50
    assert session.report["training_horizon_steps"] == 5
    assert session.cursor == 73 + 75
    before = session.fingerprint()
    prediction = session.predict(*predict_args(session, states, issued, 75))
    assert np.isfinite(prediction).all() and session.fingerprint() == before
    session.observe(session.cursor, issued[75], states[76])
    path = tmp_path / "10ms-session.npz"
    session.save(path)
    _metadata, arrays = load_arrays(path)
    # The newest target completes a window that began four observations ago.
    # There is no access to states[77:] or synthetic padding to fill the horizon.
    np.testing.assert_array_equal(arrays["recent_future_states"], states[72:77][None])
    np.testing.assert_array_equal(arrays["recent_future_inputs"], issued[71:76][None])
    np.testing.assert_array_equal(arrays["recent_past_states"], states[21:72][None])
    assert session.report["tail_rows"] == 56
    assert OnlineFit.load(path).fingerprint() == session.fingerprint()


def test_gauss_newton_matches_linear_solve_and_handles_zero_and_large_residuals(
    monkeypatch,
):
    """Check CG/IRLS/trust arithmetic against an independent two-parameter problem."""
    matrix = np.zeros((15, 2))
    matrix[:3] = [[1.0, 0.2], [-0.3, 0.7], [0.2, -0.1]]
    start = np.array([0.1, -0.05])
    weights = np.zeros(64)
    weights[:5] = 0.1
    weights[32] = 0.5
    history = np.zeros((64, 2, 15))
    inputs = np.zeros((64, 1, 3))
    scale = np.ones((1, 15))

    def linear_rollout(params, norms, past, past_inputs, future, delay, dt_s):
        values = jnp.asarray(matrix) @ params["coefficients"]
        return jnp.broadcast_to(values, (64, 1, 15))

    monkeypatch.setattr(online, "_rollout", linear_rollout)
    # Bypass the JIT cache so this deliberately independent residual map is traced.
    solve = online._proposal.__wrapped__
    with jax.enable_x64(True):
        for target, damping in (
            (np.array([0.3, 0.2]), 0.7),
            (start.copy(), 0.7),
            (np.array([30.0, 20.0]), 1e-8),
        ):
            truth = np.broadcast_to(matrix @ target, (64, 1, 15))
            proposal, current, trial, predicted, finite = solve(
                {"coefficients": jnp.asarray(start)},
                {},
                tuple(jnp.asarray(v) for v in (history, inputs, inputs, truth)),
                jnp.asarray(scale),
                jnp.asarray(weights),
                jnp.asarray(damping),
                delay=1,
                dt_s=0.05,
            )
            assert bool(finite) and np.isfinite([current, trial, predicted]).all()
            delta = np.asarray(proposal["coefficients"]) - start
            residual = matrix @ (start - target)
            radius = max(1.0, 0.5 * np.linalg.norm(residual) / np.sqrt(15))
            prediction_change = np.linalg.norm(matrix @ delta) / np.sqrt(15)
            assert prediction_change <= radius * (1 + 1e-10)
            if np.array_equal(target, start):
                np.testing.assert_array_equal(delta, np.zeros(2))
                assert current == trial == predicted == 0
            elif np.max(np.abs(residual)) < 1:
                curvature = matrix.T @ matrix / 15
                gradient = matrix.T @ residual / 15
                expected = np.linalg.solve(curvature + damping * np.eye(2), -gradient)
                np.testing.assert_allclose(delta, expected, atol=1e-12, rtol=1e-10)
                expected_reduction = -gradient @ delta - 0.5 * delta @ curvature @ delta
                assert float(predicted) == pytest.approx(expected_reduction, rel=1e-10)
                assert float(current - trial) == pytest.approx(
                    float(predicted), rel=1e-10
                )
            else:
                assert float(predicted) > 0 and float(trial) < float(current)
                assert prediction_change == pytest.approx(radius, rel=1e-10)
