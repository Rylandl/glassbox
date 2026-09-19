"""Independent centered-normal-equation and unchanged-model contract witnesses."""

import copy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_excited_public_matching import prepared_pair
from test_state_quadratic_model import small_batch

from glassbox._learner_arrays import load_arrays
from glassbox.experimental import affine_anchored_model as anchored
from glassbox.experimental import state_input_model as bilinear
from glassbox.experimental import state_quadratic_model as quadratic
from glassbox.recordings import SequenceWindows, WindowKey


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def coefficients(model):
    return np.vstack(
        [model.params[name] for name in ("linear", "interaction", "autonomous")]
        + [model.params["bias"][None]]
    )


def test_only_joint_rhs_changes_and_existing_affine_solve_is_reused(monkeypatch):
    original = np.linalg.solve
    solves = []

    def observe(system, rhs):
        result = original(system, rhs)
        solves.append((system.copy(), rhs.copy(), result.copy()))
        return result

    monkeypatch.setattr(np.linalg, "solve", observe)
    batch = small_batch(4)
    settings = dict(seed=7, width=4, memory=2, ridge=0.7, delay_steps=1)
    zero = quadratic.initialize_candidate(batch, **settings)
    candidate = anchored.initialize_candidate(batch, **settings)
    assert len(solves) == 4  # Each initializer: one affine solve and one joint solve.
    old_affine, old_joint, new_affine, new_joint = solves
    for old, new in zip(old_affine, new_affine, strict=True):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(old_joint[0], new_joint[0])
    affine_coefficients = old_affine[2]
    feature_rows = len(affine_coefficients) - 1
    product_rows = len(new_joint[1]) - len(affine_coefficients)
    anchor = np.vstack(
        (
            affine_coefficients[:-1],
            np.zeros((product_rows, 3)),
            affine_coefficients[-1:],
        )
    )
    penalty = np.diag(np.r_[np.full(len(anchor) - 1, settings["ridge"]), 0.0])
    expected_rhs = old_joint[1] + penalty @ anchor
    np.testing.assert_array_equal(new_joint[1], expected_rhs)
    np.testing.assert_array_equal(new_joint[1][-1], old_joint[1][-1])
    np.testing.assert_array_equal(
        new_joint[1][feature_rows:], old_joint[1][feature_rows:]
    )
    np.testing.assert_array_equal(
        coefficients(candidate), original(old_joint[0], expected_rhs)
    )
    # The affine coefficients remain jointly optimized, rather than frozen while
    # a smaller product-only system fits residuals.
    assert np.linalg.norm(candidate.params["linear"] - affine_coefficients[:-1]) > 1e-4
    assert np.linalg.norm(candidate.params["interaction"]) > 1e-4
    assert np.linalg.norm(candidate.params["autonomous"]) > 1e-4
    assert np.linalg.norm(coefficients(candidate) - coefficients(zero)) > 1e-4
    assert candidate.norms.keys() == zero.norms.keys()
    assert len(candidate.norms) == 8
    for name, value in zero.norms.items():
        np.testing.assert_array_equal(candidate.norms[name], value)
    for name in ("w1", "b1", "w2", "memory", "memory_bias"):
        np.testing.assert_array_equal(candidate.params[name], zero.params[name])


def test_frozen_product_order_and_target_match_independent_row_design():
    batch = small_batch(4)
    settings = dict(seed=3, width=4, memory=2, ridge=0.7, delay_steps=1)
    affine = anchored.initialize_sequence_model(batch, **settings)
    model = anchored.initialize_candidate(batch, **settings)
    n = model.norms
    physical = np.concatenate((batch.past_states, batch.future_states), 1)
    x = (physical - n["state_mean"]) / n["state_scale"]
    u = (
        np.concatenate((batch.past_inputs, batch.future_inputs), 1) - n["input_mean"]
    ) / n["input_scale"]
    rows, targets = [], []
    for i in range(len(x)):
        for j in range(4, 6):
            z = np.r_[
                x[i, j], u[i, j], x[i, j - 1] - x[i, j], u[i, j - 1] - u[i, j], 0, 0
            ]
            xu = np.asarray(
                [x[i, j, a] * u[i, j, b] for a in range(3) for b in range(2)]
            )
            xx = np.asarray(
                [x[i, j, a] * x[i, j, b] for a in range(3) for b in range(a, 3)]
            )
            rows.append(
                np.r_[
                    z / n["feature_scale"],
                    xu / n["interaction_scale"],
                    xx / n["autonomous_scale"],
                    1,
                ]
            )
            targets.append(
                (physical[i, j + 1] - physical[i, j])
                / n["state_scale"]
                / n["delta_scale"]
            )
    design, target = np.asarray(rows), np.asarray(targets)
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, 0.7), 0.0])
    anchor = np.vstack(
        (affine.params["linear"], np.zeros((12, 3)), affine.params["bias"][None])
    )
    expected = np.linalg.solve(
        design.T @ design + penalty, design.T @ target + penalty @ anchor
    )
    np.testing.assert_allclose(coefficients(model), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(model.params["linear"][-2:], 0)
    np.testing.assert_array_equal(model.params["autonomous"][[2, 4, 5]], 0)


def test_rollout_and_command_derivatives_are_the_unchanged_full_quadratic_model():
    batch = small_batch()
    model = anchored.initialize_candidate(batch, width=4, memory=2, delay_steps=1)
    reference = quadratic.QuadraticSequenceModel(
        model.kind,
        model.dt_s,
        model.history_steps,
        model.params,
        model.norms,
        model.delay_steps,
    )
    inputs = (batch.past_states, batch.past_inputs, batch.future_inputs)
    np.testing.assert_array_equal(model.rollout(*inputs), reference.rollout(*inputs))
    future = jnp.asarray(batch.future_inputs[0])
    functions = [
        lambda value, m=m: m.rollout(batch.past_states[0], batch.past_inputs[0], value)
        for m in (model, reference)
    ]
    jacobians = [jax.jacfwd(function)(future) for function in functions]
    np.testing.assert_array_equal(*jacobians)
    assert np.isfinite(jacobians[0]).all()
    assert anchored._rollout is quadratic._rollout


def test_private_optimizer_preserves_historical_numerics_and_module_identity():
    assert (
        anchored.fit_candidate_sequence.__code__
        == quadratic.fit_candidate_sequence.__code__
    )
    assert anchored.fit_candidate_sequence.__globals__ is anchored._shared.__dict__
    assert anchored._shared is not quadratic._shared
    assert quadratic.RECIPE["id"] == "autonomous-state-quadratic-v1"
    assert bilinear.RECIPE["id"] == "state-input-interaction-v1"
    assert quadratic._shared._FORMAT == "glassbox-state-quadratic-candidate-v1"
    model, report = anchored.fit_candidate_sequence(
        small_batch(),
        small_batch(1),
        steps=2,
        check_every=1,
        width=4,
        memory=2,
        delay_steps=1,
    )
    assert type(model) is anchored.AnchoredQuadraticSequenceModel
    assert report["mechanism"] == anchored.EXPERIMENT
    assert [row["step"] for row in report["trace"]] == [0, 1, 2]
    best = min(report["trace"], key=lambda row: row["validation_rollout_mse"])
    assert report["selected_step"] == best["step"]


def test_initial_parameters_and_norms_do_not_depend_on_development():
    train, dev = small_batch(), small_batch(1)
    settings = dict(steps=0, width=4, memory=2, delay_steps=1)
    first, first_report = anchored.fit_candidate_sequence(train, dev, **settings)
    second, second_report = anchored.fit_candidate_sequence(
        train, replace(dev, future_states=dev.future_states + 0.7), **settings
    )
    for name, value in first.arrays().items():
        np.testing.assert_array_equal(second.arrays()[name], value)
    assert (
        first_report["validation_rollout_mse"]
        != second_report["validation_rollout_mse"]
    )


def test_new_sequence_and_candidate_formats_roundtrip_without_changing_old_formats(
    tmp_path,
):
    batch = small_batch()
    sequence = anchored.initialize_candidate(batch, width=4, memory=2, delay_steps=1)
    windows = SequenceWindows(
        batch, tuple(WindowKey(f"r-{i}", "s", 4) for i in range(24)), (4,) * 24
    )
    contract = dict(
        state_channels=["a", "b", "c"], input_channels=["u", "v"], dt_s=0.05
    )
    candidate = anchored.CandidateDynamics(
        sequence,
        windows,
        windows,
        contract,
        {"recipe": anchored.RECIPE},
        np.ones((2, 3)),
    )
    for name, model, cls, old in (
        (
            "sequence",
            sequence,
            anchored.AnchoredQuadraticSequenceModel,
            quadratic.QuadraticSequenceModel,
        ),
        (
            "candidate",
            candidate,
            anchored.CandidateDynamics,
            quadratic.CandidateDynamics,
        ),
    ):
        path = tmp_path / (name + ".npz")
        model.save(path)
        loaded = cls.load(path)
        assert loaded.fingerprint() == model.fingerprint()
        with pytest.raises(ValueError, match="format"):
            old.load(path)
    inputs = batch.past_states[:2], batch.past_inputs[:2], batch.future_inputs[:2]
    loaded = anchored.CandidateDynamics.load(tmp_path / "candidate.npz")
    np.testing.assert_array_equal(loaded.predict(*inputs), candidate.predict(*inputs))
    np.testing.assert_allclose(
        jax.jit(loaded.predict)(*inputs), candidate.predict(*inputs), atol=1e-12
    )
    assert not hasattr(loaded, "update")
    path = tmp_path / "candidate.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values["param_autonomous"][0, 0] += 0.01
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="fingerprint"):
        load_arrays(path)


def test_wrapper_runs_one_optimizer_and_no_preparation_ridge_solve(
    monkeypatch, tmp_path
):
    public, _, _ = prepared_pair()
    old_fingerprint = public.fingerprint()
    monkeypatch.setattr(
        anchored,
        "FITTING_RECIPE",
        dict(anchored.FITTING_RECIPE, steps=2, check_every=1),
    )
    solve, fit, initialize = (
        np.linalg.solve,
        anchored.fit_candidate_sequence,
        anchored.initialize_sequence_model,
    )
    calls = {"solve": 0, "fit": 0, "affine_initializer": 0}

    def observed_solve(*args, **kwargs):
        calls["solve"] += 1
        return solve(*args, **kwargs)

    def observed_fit(*args, **kwargs):
        calls["fit"] += 1
        return fit(*args, **kwargs)

    def observed_initialize(*args, **kwargs):
        calls["affine_initializer"] += 1
        return initialize(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", observed_solve)
    monkeypatch.setattr(anchored, "fit_candidate_sequence", observed_fit)
    monkeypatch.setattr(anchored, "initialize_sequence_model", observed_initialize)
    candidate = anchored.fit_candidate(public)
    assert calls == {"solve": 2, "fit": 1, "affine_initializer": 1}
    assert public.fingerprint() == old_fingerprint
    assert candidate.report["matching"]["baseline_fingerprint"] == old_fingerprint
    assert (
        candidate._train is public._train
        and candidate._development is public._development
    )
    assert candidate.contract == public.contract
    assert (
        candidate.report["optimization"]["error_scale"]
        == public.report["optimization"]["error_scale"]
    )
    candidate.save(tmp_path / "fit.npz")
    assert (
        anchored.CandidateDynamics.load(tmp_path / "fit.npz").fingerprint()
        == candidate.fingerprint()
    )


@pytest.mark.parametrize("change", ["norm", "scale", "update"])
def test_wrong_preparation_rejected_before_initializer_or_optimizer(
    monkeypatch, change
):
    public, _, _ = prepared_pair()
    if change == "norm":
        norms = copy.deepcopy(public._model.norms)
        norms["state_mean"] += 0.1
        public._model = replace(public._model, norms=norms)
    elif change == "scale":
        public._report["optimization"]["error_scale"][0][0] += 0.1
    else:
        public._report["previous_revision"] = "updated"
    monkeypatch.setattr(np.linalg, "solve", lambda *a: pytest.fail("preparation solve"))
    monkeypatch.setattr(
        anchored, "fit_candidate_sequence", lambda *a, **k: pytest.fail("invalid fit")
    )
    with pytest.raises(ValueError):
        anchored.fit_candidate(public)
