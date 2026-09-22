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
from glassbox._dynamics import VehicleSequenceModel
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


DYNAMIC_NORMALIZERS = {"feature_scale", "quadratic_scale", "output_scale"}


def assert_normalizers_advance(initial, current):
    for name, value in initial.items():
        if name in DYNAMIC_NORMALIZERS:
            assert np.all(current[name] >= value)
        else:
            np.testing.assert_array_equal(current[name], value)
    np.testing.assert_array_equal(
        current["feature_scale"][-8:], initial["feature_scale"][-8:]
    )


def snapshot(session):
    return (
        session.fingerprint(),
        deepcopy(session.report),
        session.model.fingerprint,
        session.cursor,
    )


def controlled_evidence(current, trial, predicted, finite, *, accepted):
    """Synthetic bounded evidence for tests of the outer session transition."""
    count = 1 if accepted else 5
    losses = np.full(5, np.nan)
    reductions = np.full(5, np.nan)
    losses[:count], reductions[:count] = trial, predicted
    return dict(
        direction_finite=bool(finite),
        trust_shrink=1.0,
        trial_evaluations=count,
        trial_losses=losses,
        trial_predicted=reductions,
        trial_finite=np.r_[np.full(count, finite), np.zeros(5 - count, dtype=bool)],
        selected_alpha=1.0 if accepted else 0.0,
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
    initial_loss_scale = session._scale.copy()
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
    assert session.report["cg_iterations"] == 16
    assert session.report["curvature_calls"] == 16
    assert (
        session.report["objective_calls"]
        == 1 + session.report["last_proposal"]["trial_evaluations"]
    )
    assert jax.config.x64_enabled == before_precision
    assert session.report["conditioning_calls"] == 1
    assert_normalizers_advance(initial.norms, session.model.norms)
    np.testing.assert_array_equal(session._scale, initial_loss_scale)
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
    monkeypatch.setattr(online, "_recondition", unexpected)
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
        return (
            proposal,
            jnp.asarray(1.0),
            jnp.nan,
            jnp.nan,
            False,
            controlled_evidence(1.0, np.nan, np.nan, False, accepted=False),
        )

    with monkeypatch.context() as patch:
        patch.setattr(online, "_proposal", nonfinite)
        session.observe(session.cursor, issued[15], states[16])
    assert session.model.fingerprint == initial
    assert session.cursor == 73 + 16
    assert session.report["observations"] == 1
    assert session.report["accepted_proposals"] == 0
    assert session.report["conditioning_calls"] == 1
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
    assert session.report["conditioning_calls"] == 8


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
            residual = (prediction - windows["future_states"]) / session._scale
            values.append(numpy_group_huber(residual).mean())
    return float(np.mean(values)) + numpy_curvature_prior(
        model, (session._bootstrap, session._recent), session._scale
    )


def test_delayed_motion_adaptation_is_causal_and_retention_stays_bounded(tmp_path):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    frozen = session.model
    initial_norms = {name: value.copy() for name, value in frozen.norms.items()}
    initial_fingerprint = frozen.fingerprint
    errors, frozen_errors, archive_sizes = [], [], []
    preceding_norms = initial_norms
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
        assert_normalizers_advance(preceding_norms, session.model.norms)
        preceding_norms = session.model.norms
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
    assert session.report["conditioning_calls"] == 64
    assert_normalizers_advance(initial_norms, session.model.norms)
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
        proposal, current, trial, predicted, finite, _ = actual_proposal(
            *args, **kwargs
        )
        assert bool(finite)
        assert np.isfinite([current, trial, predicted]).all()
        calls.append(float(predicted))
        # Exercise the real CG solve but reject a finite, non-improving forecast.
        return (
            proposal,
            current,
            current,
            predicted,
            finite,
            controlled_evidence(
                float(current),
                float(current),
                float(predicted),
                bool(finite),
                accepted=False,
            ),
        )

    with monkeypatch.context() as patch:
        patch.setattr(online, "_proposal", rejected)
        session.observe(session.cursor, issued[15], states[16])
    assert len(calls) == 1
    assert session.model.fingerprint == initial
    assert session.report["optimizer_steps"] == 1
    assert (
        session.report["objective_calls"]
        == 1 + session.report["last_proposal"]["trial_evaluations"]
    )
    assert session.report["cg_iterations"] == 16
    assert session.report["curvature_calls"] == 16
    assert session.report["accepted_proposals"] == 0
    assert session.report["conditioning_calls"] == 1
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
        (0.1, 4.0, True),
        (0.2, 4.0, True),
        (0.25, 1.0, True),
        (0.5, 1.0, True),
        (0.75, 1.0, True),
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
        return (
            proposal,
            1.0,
            1.0 - gain * predicted,
            predicted,
            True,
            controlled_evidence(
                1.0, 1.0 - gain * predicted, predicted, True, accepted=accepted
            ),
        )

    monkeypatch.setattr(online, "_proposal", controlled)
    session.observe(session.cursor, issued[15], states[16])
    assert session.report["damping"] == expected_damping
    assert session.report["accepted_proposals"] == int(accepted)
    assert bool(session.report["last_proposal"]["selected_alpha"]) == accepted
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
        lambda m, a: m["counts"].update(conditioning_calls=1),
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


def numpy_group_huber(residual):
    magnitudes = np.stack(
        [
            np.linalg.norm(residual[..., part], axis=-1)
            for part in (slice(0, 3), slice(3, 6), slice(6, 15))
        ],
        axis=-1,
    )
    return np.where(magnitudes <= 1, 0.5 * magnitudes**2, magnitudes - 0.5)


@pytest.mark.parametrize("prior", [np.zeros(2), np.array([0.15, 2.3])])
def test_gauss_newton_matches_linear_solve_and_handles_zero_and_large_residuals(
    monkeypatch,
    prior,
):
    """Compare every group-IRLS proposal to a dense two-parameter solve."""
    matrix = np.random.default_rng(581).normal(0, 0.3, (15, 2))
    start = np.array([0.1, -0.05])
    weights = np.zeros(64)
    weights[:5] = 0.1
    weights[32] = 0.5
    history = np.zeros((64, 2, 15))
    inputs = np.zeros((64, 1, 3))
    scale = np.ones((1, 15))

    def linear_rollout(params, norms, past, past_inputs, future, delay, dt_s):
        return jnp.broadcast_to(
            jnp.asarray(matrix) @ params["coefficients"], (64, 1, 15)
        )

    monkeypatch.setattr(online, "_rollout", linear_rollout)
    monkeypatch.setattr(
        online, "_curvature_diagonal", lambda *args, **kwargs: jnp.asarray(prior)
    )
    solve = online._proposal.__wrapped__
    with jax.enable_x64(True):
        for target, damping in (
            (np.array([0.3, 0.2]), 0.7),
            (start.copy(), 0.7),
            (np.array([30.0, 20.0]), 1e-8),
        ):
            truth = np.broadcast_to(matrix @ target, (64, 1, 15))
            proposal, current, trial, predicted, finite, _ = solve(
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
            irls = np.concatenate(
                [
                    np.full(
                        part.stop - part.start,
                        1 / max(1, np.linalg.norm(residual[part])),
                    )
                    for part in (slice(0, 3), slice(3, 6), slice(6, 15))
                ]
            )
            curvature = matrix.T @ (irls[:, None] * matrix) / 3 + np.diag(prior)
            gradient = matrix.T @ (irls * residual) / 3 + prior * start
            expected = np.linalg.solve(curvature + damping * np.eye(2), -gradient)
            radius = max(1.0, 0.5 * np.linalg.norm(residual) / np.sqrt(3))
            step_size = np.linalg.norm(matrix @ expected) / np.sqrt(3)
            if step_size > radius:
                expected *= radius / step_size
            np.testing.assert_allclose(delta, expected, atol=1e-12, rtol=1e-10)
            expected_reduction = -gradient @ delta - 0.5 * delta @ curvature @ delta
            assert float(predicted) == pytest.approx(
                expected_reduction, abs=1e-12, rel=1e-10
            )
            assert float(current) == pytest.approx(
                numpy_group_huber(residual).mean() + 0.5 * np.dot(prior, start**2),
                abs=1e-12,
            )
            assert float(trial) == pytest.approx(
                numpy_group_huber(residual + matrix @ delta).mean()
                + 0.5 * np.dot(prior, (start + delta) ** 2),
                abs=1e-12,
            )
            assert np.linalg.norm(matrix @ delta) / np.sqrt(3) <= radius * (1 + 1e-10)
            if np.array_equal(target, start) and not np.any(prior):
                np.testing.assert_array_equal(delta, np.zeros(2))
                assert current == trial == predicted == 0
            else:
                assert float(predicted) > 0 and float(trial) < float(current)


def numpy_group_scale(windows):
    origins = windows["past_states"][:, -1]
    targets = windows["future_states"]
    result = np.empty(targets.shape[1:])
    for part, factor in ((slice(0, 3), 1.0), (slice(3, 6), 1.0), (slice(6, 15), 0.5)):
        samples = origins[:, part]
        center = sum(samples) / len(samples)
        spread = np.sqrt(
            sum(factor * np.dot(row - center, row - center) for row in samples)
            / len(samples)
        )
        floor = 0.01 * max(spread, 1e-4)
        for horizon in range(targets.shape[1]):
            energy = sum(
                factor * np.dot(origin - target, origin - target)
                for origin, target in zip(samples, targets[:, horizon, part])
            ) / len(samples)
            result[horizon, part] = max(np.sqrt(energy), floor) / np.sqrt(factor)
    return result


def test_group_scale_matches_raw_vector_spread_and_floors():
    _, bootstrap, _ = conditioning_fixture()
    np.testing.assert_allclose(
        online._scale(bootstrap), numpy_group_scale(bootstrap), rtol=2e-15, atol=1e-16
    )
    unchanged = dict(
        bootstrap, future_states=np.repeat(bootstrap["past_states"][:, -1:], 3, axis=1)
    )
    np.testing.assert_allclose(
        online._scale(unchanged), numpy_group_scale(unchanged), rtol=2e-15, atol=1e-16
    )
    constant = {
        name: np.repeat(value[:1], len(value), axis=0)
        for name, value in unchanged.items()
    }
    expected_floor = np.tile(
        np.r_[np.full(6, 1e-6), np.full(9, np.sqrt(2) * 1e-6)], (3, 1)
    )
    np.testing.assert_allclose(
        online._scale(constant), expected_floor, rtol=2e-15, atol=1e-16
    )


def test_group_scale_and_objective_are_invariant_under_common_frame_rotations():
    from scipy.spatial.transform import Rotation

    _, bootstrap, _ = conditioning_fixture()
    world = Rotation.from_rotvec([0.43, -0.28, 0.71]).as_matrix()
    body = Rotation.from_rotvec([-0.21, 0.65, 0.38]).as_matrix()

    def rotate(states):
        result = states.copy()
        result[..., :3] = states[..., :3] @ world.T
        result[..., 3:6] = states[..., 3:6] @ body.T
        result[..., 6:] = (
            world @ states[..., 6:].reshape(*states.shape[:-1], 3, 3) @ body.T
        ).reshape(*states.shape[:-1], 9)
        return result

    transformed = {
        name: rotate(value) if name.endswith("states") else value
        for name, value in bootstrap.items()
    }
    original_scale, transformed_scale = (
        online._scale(bootstrap),
        online._scale(transformed),
    )
    np.testing.assert_allclose(
        original_scale, transformed_scale, rtol=2e-13, atol=2e-15
    )
    rng = np.random.default_rng(64)
    predicted = bootstrap["future_states"] + rng.normal(
        0, 0.3, bootstrap["future_states"].shape
    )
    original = (predicted - bootstrap["future_states"]) / original_scale
    rotated = (rotate(predicted) - transformed["future_states"]) / transformed_scale
    with jax.enable_x64(True):
        np.testing.assert_allclose(
            online._huber(original), online._huber(rotated), rtol=2e-12, atol=2e-12
        )


def test_radial_huber_gradient_is_finite_at_zero_and_handles_group_outliers():
    examples = np.array(
        [
            np.zeros(15),
            [0.2, -0.1, 0.3, 0, 0, 0, *([0.05] * 9)],
            [3.0, 4.0, 0.0, 0.1, 0.2, 0.0, *([2.0] * 9)],
        ]
    )
    with jax.enable_x64(True):
        for residual in examples:
            value = online._huber(jnp.asarray(residual))
            gradient = jax.grad(lambda r: jnp.sum(online._huber(r)))(
                jnp.asarray(residual)
            )
            expected = residual.copy()
            for part in (slice(0, 3), slice(3, 6), slice(6, 15)):
                expected[part] /= max(1, np.linalg.norm(residual[part]))
            assert np.isfinite(gradient).all()
            np.testing.assert_allclose(
                value, numpy_group_huber(residual), rtol=2e-14, atol=1e-15
            )
            np.testing.assert_allclose(gradient, expected, rtol=2e-14, atol=1e-15)


def conditioning_fixture():
    """Independent smooth motion with unequal cache sizes and active recurrent heads."""
    from scipy.spatial.transform import Rotation

    states, issued = stream()
    original = OnlineFit(prefix(states, issued)).model
    rng = np.random.default_rng(731)
    params = {
        name: rng.normal(0, 0.003, value.shape)
        for name, value in original.params.items()
    }
    params["raw_tau"] = np.array([-3.0, -2.7, -3.3])
    norms = {
        name: np.zeros_like(value) if name.endswith("_mean") else np.ones_like(value)
        for name, value in original.norms.items()
    }
    norms["motion_bound_scale"][:] = 8.0
    model = VehicleSequenceModel(0.05, 10, 2, params, norms)
    t = np.arange(40) * model.dt_s
    states = np.column_stack(
        (
            0.7 + 2.0 * t + 0.4 * t**2,
            -0.2 + 0.6 * t**2,
            0.3 - t,
            0.1 + 0.2 * t,
            -0.15 + 0.1 * t**2,
            0.12 * np.sin(t),
            Rotation.from_rotvec(
                np.column_stack((0.3 * t, -0.2 * t**2, 0.1 * np.sin(t)))
            )
            .as_matrix()
            .reshape(-1, 9),
        )
    )
    commands = np.column_stack(
        (3 * np.sin(4 * t), 4 * np.cos(3 * t), 2 * np.sin(5 * t + 0.7))
    )[:-1]
    bootstrap = online._windows(states, commands, [10, 11], 10, 3)
    recent = online._windows(states, commands, [21, 22, 23, 24, 25], 10, 3)
    return model, bootstrap, recent


def numpy_conditioning_scales(model, roles):
    """Literal per-role/window/time computation without any model feature helpers."""
    role_feature, role_quadratic, role_output = [], [], []
    norms, delay = model.norms, model.delay_steps
    tau = 0.001 + np.logaddexp(0.0, model.params["raw_tau"])
    for windows in roles:
        window_feature, window_quadratic, window_output = [], [], []
        for past, inputs, future, targets in zip(
            *(windows[key] for key in online._FIELDS)
        ):
            observed = np.concatenate((past, targets[:-1]))
            issued = np.concatenate((inputs, future))
            applied = issued[0].copy()
            features = []
            for state, command in zip(observed, issued):
                rotation = state[6:].reshape(3, 3)
                body = np.r_[rotation.T @ state[:3], state[3:6], -rotation[2]]
                body = (body - norms["body_mean"]) / norms["body_scale"]
                support = norms["motion_bound_scale"] / 4
                for axis in range(6):
                    excess = abs(body[axis]) - support[axis]
                    if excess > 0:
                        body[axis] = (
                            np.sign(body[axis])
                            * support[axis]
                            * (1 + 3 * np.tanh(excess / (3 * support[axis])))
                        )
                features.append(
                    np.r_[
                        body,
                        (command - norms["input_mean"]) / norms["input_scale"],
                        (applied - norms["input_mean"]) / norms["input_scale"],
                    ]
                )
                applied = command + (applied - command) * np.exp(-model.dt_s / tau)
            features = np.asarray(features)
            linear, quadratic = [], []
            for time in range(delay, len(features)):
                current = features[time]
                linear.append(
                    np.r_[
                        current,
                        np.concatenate(
                            [
                                features[previous] - current
                                for previous in range(time - delay, time)
                            ]
                        ),
                    ]
                )
                quadratic.append(
                    [
                        current[left] * current[right]
                        for left in range(len(current))
                        for right in range(left, len(current))
                    ]
                )
            output = []
            preceding = past[-1]
            for target in targets:
                output.append(
                    np.r_[
                        preceding[6:].reshape(3, 3).T
                        @ (
                            (target[:3] - preceding[:3]) / model.dt_s - [0, 0, -9.80665]
                        ),
                        (target[3:6] - preceding[3:6]) / model.dt_s,
                    ]
                )
                preceding = target
            window_feature.append(np.mean(np.square(linear), axis=0))
            window_quadratic.append(np.mean(np.square(quadratic), axis=0))
            window_output.append(np.mean(np.square(output), axis=0))
        role_feature.append(np.mean(window_feature, axis=0))
        role_quadratic.append(np.mean(window_quadratic, axis=0))
        role_output.append(np.mean(window_output, axis=0))
    return {
        "feature_scale": np.r_[
            np.maximum(
                norms["feature_scale"][:-8], np.sqrt(np.mean(role_feature, axis=0))
            ),
            norms["feature_scale"][-8:],
        ],
        "quadratic_scale": np.maximum(
            norms["quadratic_scale"], np.sqrt(np.mean(role_quadratic, axis=0))
        ),
        "output_scale": np.maximum(
            norms["output_scale"], np.sqrt(np.mean(role_output, axis=0))
        ),
    }


def test_conditioning_matches_independent_role_window_time_rms_and_never_shrinks():
    model, bootstrap, recent = conditioning_fixture()
    data, weights = online._full_cache(bootstrap, recent)
    expected = numpy_conditioning_scales(model, (bootstrap, recent))
    with jax.enable_x64(True):
        params, norms, finite = online._recondition(
            model.params,
            model.norms,
            data,
            weights,
            delay=model.delay_steps,
            dt_s=model.dt_s,
        )
        assert bool(finite)
        for name, values in expected.items():
            np.testing.assert_allclose(norms[name], values, rtol=2e-13, atol=2e-13)
            assert np.any(np.asarray(norms[name]) > model.norms[name])
        assert_normalizers_advance(model.norms, norms)
        for name in ("b1", "memory_bias", "raw_tau", "raw_memory_tau"):
            np.testing.assert_array_equal(params[name], model.params[name])
        high = {name: np.asarray(value).copy() for name, value in norms.items()}
        for name in DYNAMIC_NORMALIZERS:
            high[name] *= 4
        high["feature_scale"][-8:] = 1
        _, following, finite = online._recondition(
            params,
            high,
            data,
            weights,
            delay=model.delay_steps,
            dt_s=model.dt_s,
        )
        assert bool(finite)
        for name in DYNAMIC_NORMALIZERS:
            np.testing.assert_array_equal(following[name], high[name])


def test_conditioning_preserves_recursive_predictions_and_command_jacobian():
    model, bootstrap, recent = conditioning_fixture()
    assert all(np.all(value != 0) for value in model.params.values())
    data, weights = online._full_cache(bootstrap, recent)
    with jax.enable_x64(True):
        params, norms, finite = online._recondition(
            model.params,
            model.norms,
            data,
            weights,
            delay=model.delay_steps,
            dt_s=model.dt_s,
        )
        assert bool(finite)
        transformed = replace(
            model,
            params={name: np.asarray(value) for name, value in params.items()},
            norms={name: np.asarray(value) for name, value in norms.items()},
        )
        args = tuple(recent[name] for name in online._FIELDS[:3])
        np.testing.assert_allclose(
            transformed.rollout(*args), model.rollout(*args), rtol=2e-11, atol=2e-12
        )
        past, inputs, future = (value[:1] for value in args)
        future = jnp.asarray(future + 0.13)
        before = jax.jacrev(lambda u: model.rollout(past, inputs, u))(future)
        after = jax.jacrev(lambda u: transformed.rollout(past, inputs, u))(future)
        assert np.max(np.abs(before)) > 1e-6
        np.testing.assert_allclose(after, before, rtol=2e-10, atol=2e-12)


def test_invalid_conditioning_retains_original_full_model_and_consumes_once(
    monkeypatch,
):
    states, issued = stream()
    session = OnlineFit(prefix(states, issued))
    initial = session.model
    original_scale = session._scale.copy()
    actual = online._recondition

    def invalid(*args, **kwargs):
        params, norms, _ = actual(*args, **kwargs)
        return params, norms, False

    monkeypatch.setattr(online, "_recondition", invalid)
    session.observe(session.cursor, issued[15], states[16])
    assert session.model.fingerprint == initial.fingerprint
    np.testing.assert_array_equal(session._scale, original_scale)
    assert session.report["conditioning_calls"] == 1
    assert session.report["accepted_proposals"] == 0
    assert session.report["damping"] == 4
    assert session.cursor == 73 + 16


def numpy_curvature_factors(model, roles, scale):
    """Literal physical Hessian weighting, without production feature helpers."""
    norms = model.norms
    role_energy = []
    for windows in roles:
        if not len(windows["past_states"]):
            continue
        samples = []
        for past, inputs, future, targets in zip(
            *(windows[key] for key in online._FIELDS)
        ):
            observed = np.concatenate((past, targets[:-1]))
            issued = np.concatenate((inputs, future))
            energy = []
            for state, command in zip(
                observed[model.delay_steps :], issued[model.delay_steps :]
            ):
                rotation = state[6:].reshape(3, 3)
                motion = (
                    np.r_[rotation.T @ state[:3], state[3:6]] - norms["body_mean"][:6]
                ) / norms["body_scale"][:6]
                support = norms["motion_bound_scale"] / 4
                for axis in range(6):
                    excess = abs(motion[axis]) - support[axis]
                    if excess > 0:
                        motion[axis] = (
                            np.sign(motion[axis])
                            * support[axis]
                            * (1 + 3 * np.tanh(excess / (3 * support[axis])))
                        )
                commands = (command - norms["input_mean"]) / norms["input_scale"]
                energy.append(np.r_[motion, commands] ** 2)
            samples.append(np.mean(energy, axis=0))
        role_energy.append(np.mean(samples, axis=0))
    rms = np.maximum(1, np.sqrt(np.mean(role_energy, axis=0)))
    domain = np.r_[rms[:6], 1 / norms["body_scale"][6:9], rms[6:], rms[6:]]
    factors = []
    for index, (left, right) in enumerate(zip(*np.triu_indices(len(domain)))):
        hessian = 2 if left == right else np.sqrt(2)
        factors.append(
            hessian
            * domain[left]
            * domain[right]
            * model.dt_s
            * norms["output_scale"]
            / (norms["quadratic_scale"][index] * scale[0, :6])
        )
    return domain, np.asarray(factors)


def numpy_curvature_prior(model, roles, scale):
    _, factors = numpy_curvature_factors(model, roles, scale)
    return 0.01 / 4 * np.sum((factors * model.params["quadratic"]) ** 2)


def test_curvature_domain_uses_measured_physical_scales_and_issued_commands():
    from jax.flatten_util import ravel_pytree

    model, bootstrap, recent = conditioning_fixture()
    norms = {name: value.copy() for name, value in model.norms.items()}
    norms["body_scale"] = np.array([0.8, 1.2, 0.5, 1.1, 0.6, 0.9, 0.1, 0.3, 0.002])
    norms["body_mean"] = np.array([0.2, 0.1, 0, -0.1, 0.3, 0.2, 0, 0, -0.99])
    norms["input_scale"] = np.array([0.3, 1.5, 0.7])
    norms["input_mean"] = np.array([0.1, -0.2, 0.3])
    model = replace(model, norms=norms)
    data, weights = online._full_cache(bootstrap, recent)
    scale = online._scale(bootstrap)
    domain, factors = numpy_curvature_factors(model, (bootstrap, recent), scale)
    assert not np.allclose(domain[:6], 1)
    np.testing.assert_array_equal(domain[6:9], [10, 1 / 0.3, 500])
    np.testing.assert_array_equal(domain[9:12], domain[12:])
    _, pooled = numpy_curvature_factors(
        model,
        ({k: np.concatenate((bootstrap[k], recent[k])) for k in bootstrap},),
        scale,
    )
    assert not np.allclose(factors, pooled)  # Distinguishes role from pooled weighting.
    with jax.enable_x64(True):
        flat, unpack = ravel_pytree(model.params)
        actual = online._curvature_diagonal(
            model.params,
            model.norms,
            data,
            scale,
            weights,
            delay=model.delay_steps,
            dt_s=model.dt_s,
        )
        diagonal = unpack(actual)
        np.testing.assert_allclose(
            diagonal["quadratic"], 0.01 / 2 * factors**2, rtol=2e-13, atol=1e-14
        )
        for name in model.params:
            if name != "quadratic":
                np.testing.assert_array_equal(
                    diagonal[name], np.zeros_like(model.params[name])
                )
        assert float(0.5 * jnp.dot(actual, flat**2)) == pytest.approx(
            numpy_curvature_prior(model, (bootstrap, recent), scale), rel=2e-13
        )
        # Neither learned lag nor arbitrary target beyond the current features sets D.
        changed = dict(model.params, raw_tau=np.full(3, 10.0))
        last_target = [v.copy() for v in data]
        last_target[3][:, -1, :6] += 1000
        unchanged = online._curvature_diagonal(
            changed,
            model.norms,
            tuple(last_target),
            scale,
            weights,
            delay=model.delay_steps,
            dt_s=model.dt_s,
        )
        np.testing.assert_array_equal(actual, unchanged)
        zero = dict(model.params, quadratic=np.zeros_like(model.params["quadratic"]))
        affine, _ = ravel_pytree(zero)
        assert float(jnp.dot(actual, affine**2)) == 0


def test_curvature_prior_and_directional_derivative_survive_reconditioning():
    from jax.flatten_util import ravel_pytree

    model, bootstrap, recent = conditioning_fixture()
    data, weights = online._full_cache(bootstrap, recent)
    scale = online._scale(bootstrap)
    rng = np.random.default_rng(115)
    direction = {
        name: rng.normal(0, 0.1, value.shape) for name, value in model.params.items()
    }
    with jax.enable_x64(True):
        params, norms, finite = online._recondition(
            model.params,
            model.norms,
            data,
            weights,
            delay=model.delay_steps,
            dt_s=model.dt_s,
        )
        assert finite
        transformed = replace(
            model,
            params=jax.tree.map(np.asarray, params),
            norms=jax.tree.map(np.asarray, norms),
        )
        sq = np.asarray(norms["quadratic_scale"]) / model.norms["quadratic_scale"]
        so = model.norms["output_scale"] / np.asarray(norms["output_scale"])
        changed_direction = dict(
            direction, quadratic=sq[:, None] * direction["quadratic"] * so
        )
        values, derivatives = [], []
        for core, tangent in ((model, direction), (transformed, changed_direction)):
            flat, _ = ravel_pytree(core.params)
            d, _ = ravel_pytree(tangent)
            diagonal = online._curvature_diagonal(
                core.params,
                core.norms,
                data,
                scale,
                weights,
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
            values.append(float(0.5 * jnp.dot(diagonal, flat**2)))
            derivatives.append(float(jnp.dot(diagonal * flat, d)))
            assert values[-1] == pytest.approx(
                numpy_curvature_prior(core, (bootstrap, recent), scale), rel=3e-13
            )
        assert values[0] == pytest.approx(values[1], rel=3e-13)
        assert derivatives[0] == pytest.approx(derivatives[1], rel=3e-13)


def test_sixteen_pcg_steps_match_independent_truncated_solve(monkeypatch):
    rng = np.random.default_rng(341)
    matrix = rng.normal(0, 0.3, (30, 24))
    start = rng.normal(size=24)
    prior = np.geomspace(1e-4, 1, 24)
    target = np.full(30, 8.0)
    damping = 0.003
    raw = matrix @ start - target
    grouped = raw.reshape(2, 15)
    irls = np.concatenate(
        [
            np.concatenate(
                [
                    np.full(b - a, 1 / max(1, np.linalg.norm(row[a:b])))
                    for a, b in ((0, 3), (3, 6), (6, 15))
                ]
            )
            for row in grouped
        ]
    )
    curvature = matrix.T @ (irls[:, None] * matrix) / 6 + np.diag(prior)
    gradient = matrix.T @ (irls * raw) / 6 + prior * start
    hessian = curvature + damping * np.eye(24)
    delta = np.zeros(24)
    residual = -gradient.copy()
    z = residual / (damping + prior)
    direction = z.copy()
    rho = residual @ z
    for _ in range(16):
        action = hessian @ direction
        alpha = rho / (direction @ action)
        delta += alpha * direction
        residual -= alpha * action
        z = residual / (damping + prior)
        following = residual @ z
        direction = z + (following / rho) * direction
        rho = following
    assert np.linalg.norm(hessian @ delta + gradient) > 1e-7
    radius = max(1, 0.5 * np.linalg.norm(raw) / np.sqrt(6))
    delta *= min(1, radius / (np.linalg.norm(matrix @ delta) / np.sqrt(6)))

    def residual_function(params, norms, data, scale, delay, dt_s):
        return (jnp.asarray(matrix) @ params["coefficients"] - target).reshape(1, 2, 15)

    monkeypatch.setattr(online, "_residual", residual_function)
    monkeypatch.setattr(
        online, "_curvature_diagonal", lambda *a, **kw: jnp.asarray(prior)
    )
    with jax.enable_x64(True):
        proposal, current, trial, predicted, finite, evidence = (
            online._proposal.__wrapped__(
                {"coefficients": jnp.asarray(start)},
                {},
                (),
                np.ones((2, 15)),
                jnp.ones(1),
                damping,
                delay=1,
                dt_s=0.05,
            )
        )
    assert finite
    assert int(evidence["trial_evaluations"]) == 1
    np.testing.assert_allclose(
        np.asarray(proposal["coefficients"]), start + delta, rtol=2e-8, atol=2e-9
    )
    assert float(predicted) == pytest.approx(
        -gradient @ delta - 0.5 * delta @ curvature @ delta, rel=2e-8
    )
    assert float(current) == pytest.approx(
        numpy_group_huber(grouped).mean() + 0.5 * prior @ start**2, rel=2e-12
    )
    assert float(trial) == pytest.approx(
        numpy_group_huber((raw + matrix @ delta).reshape(2, 15)).mean()
        + 0.5 * prior @ (start + delta) ** 2,
        rel=2e-8,
    )
