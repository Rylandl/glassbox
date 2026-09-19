"""Full-cache gradient arithmetic and unchanged optimizer boundaries on toy systems."""

import ast
import copy
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_safeguarded_adam_model as inherited_tests
from test_excited_public_matching import prepared_pair
from test_initial_channel_balance_model import settings as base_settings
from test_state_quadratic_model import small_batch

from glassbox.experimental import full_cache_gradient_model as full
from glassbox.experimental import safeguarded_adam_model as safe

same_tree = inherited_tests.same_tree
copied = inherited_tests.copied
phase = inherited_tests.phase


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def settings(batch, **extra):
    return base_settings(batch, batch_size=len(batch.past_states), **extra)


def capture(monkeypatch):
    events = []
    monkeypatch.setattr(
        full, "_observe_attempt", lambda value: events.append(copied(value))
    )
    return events


def scripted_objective(monkeypatch, values):
    remaining = iter(values)
    monkeypatch.setattr(
        full, "make_training_objective", lambda *args: lambda params: next(remaining)
    )


def test_initial_identity_helpers_and_truthful_private_recipe():
    train, dev = small_batch(4), small_batch(5)
    old, old_report = safe.fit_candidate_sequence(
        train, dev, steps=0, **settings(train)
    )
    model, report = full.fit_candidate_sequence(train, dev, steps=0, **settings(train))
    same_tree(model.arrays(), old.arrays())
    for key in ("objective", "trace", "unweighted_development_trace", "error_scale"):
        assert report[key] == old_report[key]
    for name in (
        "initialize_candidate",
        "_rollout",
        "initial_training_forecast",
        "weighting_metadata",
        "make_training_objective",
        "trial_parameters",
    ):
        assert getattr(full, name) is getattr(safe, name)
    assert full.RECIPE["batch_size"] == full.FITTING_RECIPE["batch_size"] == 384
    assert safe.RECIPE["batch_size"] == safe.FITTING_RECIPE["batch_size"] == 64
    assert (
        full.RECIPE["gradient_sampling"]
        == report["gradient_sampling"]
        == "ordered_full_cache"
    )
    assert report["minibatch_seed"] is None
    assert report["gradient"]["known_gradient_window_visits"] == 0
    assert report["safeguard"] == {**old_report["safeguard"], "id": full.EXPERIMENT}
    assert full.SCALES is safe.SCALES and full.FIT_WALL_TIME_LIMIT_S == 7200


def test_inner_adam_operation_order_is_identical_to_pinned_control():
    def update(function):
        tree = ast.parse(inspect.getsource(function))
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "update"
        )

    assert ast.dump(update(full.fit_candidate_sequence)) == ast.dump(
        update(safe.fit_candidate_sequence)
    )


@pytest.mark.parametrize("clipping", [False, True])
def test_full_objective_gradient_once_clip_and_multistep_adam(monkeypatch, clipping):
    train = small_batch(4)  # Three outputs, two inputs, 24 distinct ordered windows.
    args = settings(train)
    if clipping:
        args["error_scale"] *= 1e-4
        monkeypatch.setattr(
            full,
            "initial_training_forecast",
            lambda *unused: train.future_states + args["error_scale"],
        )
    initial = full.initialize_candidate(
        train,
        **{
            key: value
            for key, value in args.items()
            if key not in ("batch_size", "error_scale")
        },
    )
    norms = jax.tree.map(jnp.asarray, initial.norms)
    prediction = full.initial_training_forecast(initial.params, initial.norms, train, 1)
    e0 = np.mean(
        ((prediction - train.future_states) / args["error_scale"]) ** 2, axis=(0, 1)
    )
    inverse = 1.0 / np.maximum(e0, 0.0001)
    weights = jnp.asarray(inverse / inverse.mean())
    normalization = jnp.asarray(args["error_scale"])
    all_data = tuple(jnp.asarray(getattr(train, key)) for key in full._ARRAYS)

    def loss(parameters):
        prediction = safe._rollout(parameters, norms, *all_data[:3], 1)
        return jnp.mean(weights * ((prediction - all_data[3]) / normalization) ** 2)

    @jax.jit
    def reference(parameters, first, second, step):
        value, gradient = jax.value_and_grad(loss)(parameters)
        norm = jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in jax.tree.leaves(gradient)))
        clipped = jax.tree.map(
            lambda leaf: leaf * jnp.minimum(1.0, 5.0 / (norm + 1e-12)), gradient
        )
        first = jax.tree.map(lambda old, leaf: 0.9 * old + 0.1 * leaf, first, clipped)
        second = jax.tree.map(
            lambda old, leaf: 0.999 * old + 0.001 * leaf * leaf, second, clipped
        )
        proposal = jax.tree.map(
            lambda old, m, v: (
                old
                - 1e-4
                * (m / (1 - 0.9**step))
                / (jnp.sqrt(v / (1 - 0.999**step)) + 1e-8)
            ),
            parameters,
            first,
            second,
        )
        return value, norm, clipped, first, second, proposal

    calls = []
    real_value_and_grad = jax.value_and_grad

    def value_and_grad(function, *extra, **kwargs):
        calls.append(function.__name__)
        return real_value_and_grad(function, *extra, **kwargs)

    monkeypatch.setattr(
        full, "jax", SimpleNamespace(**{**vars(jax), "value_and_grad": value_and_grad})
    )
    events = capture(monkeypatch)
    scripted_objective(monkeypatch, [4.0, 3.0, 2.0, 1.0])
    _, report = full.fit_candidate_sequence(
        train, train, steps=3, check_every=1, learning_rate=1e-4, **args
    )
    assert calls == ["loss"]  # One gradient graph, reused once per returned proposal.
    starts, proposals = phase(events, "started"), phase(events, "proposed")
    assert len(proposals) == 3
    for step, (before, observed) in enumerate(zip(starts, proposals, strict=True), 1):
        indices = before["indices"]
        assert indices.dtype == np.dtype("int64")
        np.testing.assert_array_equal(indices, np.arange(len(train.past_states)))
        inputs = jax.tree.map(
            jnp.asarray, (before["params"], before["first"], before["second"])
        )
        value, norm, clipped, first, second, proposal = reference(*inputs, step)
        np.testing.assert_allclose(
            observed["minibatch_loss"], value, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            observed["gradient_norm"], norm, rtol=1e-12, atol=1e-12
        )
        for actual, expected in (
            (observed["clipped_gradient"], clipped),
            (observed["new_first"], first),
            (observed["new_second"], second),
            (observed["proposal"], proposal),
        ):
            same_tree(actual, expected)
        assert (observed["gradient_norm"] > 5) == clipping
    assert report["gradient"] == full.gradient_metadata(
        24,
        attempts_started=3,
        gradient_proposal_calls_returned=3,
        completed_acceptance_attempts=3,
    )


def test_one_initializer_two_solves_one_weight_forecast_no_optimizer_rng(monkeypatch):
    train, dev = small_batch(4), small_batch(5)
    originals = (
        full.initialize_candidate,
        np.linalg.solve,
        full.initial_training_forecast,
        np.random.default_rng,
    )
    counts, seeds = [0, 0, 0], []

    def counted(index, *args, **kwargs):
        counts[index] += 1
        return originals[index](*args, **kwargs)

    def rng(seed):
        seeds.append(seed)
        return originals[3](seed)

    monkeypatch.setattr(
        full, "initialize_candidate", lambda *a, **k: counted(0, *a, **k)
    )
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: counted(1, *a, **k))
    monkeypatch.setattr(
        full, "initial_training_forecast", lambda *a, **k: counted(2, *a, **k)
    )
    monkeypatch.setattr(np.random, "default_rng", rng)
    events = capture(monkeypatch)
    full.fit_candidate_sequence(train, dev, steps=3, check_every=1, **settings(train))
    assert counts == [1, 2, 1] and seeds == [7]
    for row in phase(events, "started"):
        assert row["indices"].dtype == np.dtype("int64")
        np.testing.assert_array_equal(row["indices"], np.arange(24))


def test_full_training_gradient_scalar_stays_distinct_from_acceptance_cache(
    monkeypatch,
):
    train = small_batch(4)
    events = capture(monkeypatch)
    scripted_objective(monkeypatch, [10.0] * 17)
    _, report = full.fit_candidate_sequence(
        train, train, steps=2, check_every=1, **settings(train)
    )
    starts, proposed, completed = (
        phase(events, name) for name in ("started", "proposed", "completed")
    )
    for before, proposal, after in zip(starts, proposed, completed, strict=True):
        same_tree(before["params"], after["params"])
        same_tree(proposal["new_first"], after["first"])
        same_tree(proposal["new_second"], after["second"])
        assert before["current_full_training_loss"] == after["loss"] == 10.0
        assert proposal["minibatch_loss"] != 10.0
        assert after["accepted_scale"] == 0.0
    assert report["safeguard"]["full_training_objective_calls"] == 17
    assert report["safeguard"]["maximum_rejection_streak"] == 2
    assert report["gradient"]["known_gradient_window_visits"] == 48
    assert report["selected_step"] == 0


@pytest.mark.parametrize("batch_size", [0, 23, 25, 64, True, 24.0])
def test_no_silent_subsetting_or_inaccurate_batch_metadata(batch_size):
    train = small_batch(4)
    with pytest.raises(ValueError, match="complete positive training cache"):
        full.fit_candidate_sequence(
            train, train, steps=0, **{**settings(train), "batch_size": batch_size}
        )


@pytest.mark.parametrize(
    "counts,visits,unknown",
    [
        ((0, 0, 0), 0, False),
        ((3, 3, 3), 72, False),
        ((3, 2, 2), 48, True),
        ((3, 3, 2), 72, False),
    ],
)
def test_gradient_work_counts_only_returned_calls(counts, visits, unknown):
    value = full.gradient_metadata(
        24,
        attempts_started=counts[0],
        gradient_proposal_calls_returned=counts[1],
        completed_acceptance_attempts=counts[2],
    )
    assert value["known_gradient_window_visits"] == visits
    assert value["incomplete_gradient_work_unknown"] is unknown
    assert value["sampling_seed"] is None
    assert value["index_dtype"] == "int64"


@pytest.mark.parametrize("counts", [(2, 3, 2), (3, 2, 3), (3, 1, 1), (True, 1, 1)])
def test_impossible_work_prefix_is_rejected(counts):
    with pytest.raises(ValueError, match="actual full-cache gradient accounting"):
        full.gradient_metadata(
            24,
            attempts_started=counts[0],
            gradient_proposal_calls_returned=counts[1],
            completed_acceptance_attempts=counts[2],
        )


@pytest.mark.parametrize(
    "name",
    [
        "test_first_strict_acceptable_scale_and_nonfinite_trials",
        "test_actual_full_training_losses_are_monotone_and_selection_unchanged",
        "test_development_cannot_change_training_proposals_or_acceptance",
        "test_final_time_limit_includes_last_development_evaluation",
        "test_unknown_error_propagates_without_fabricated_attempt_completion",
    ],
)
def test_inherited_acceptance_selection_and_failure_boundaries(monkeypatch, name):
    monkeypatch.setattr(inherited_tests, "safe", full)
    monkeypatch.setattr(inherited_tests, "settings", settings)
    getattr(inherited_tests, name)(monkeypatch)


@pytest.mark.parametrize(
    "field",
    [
        "proposal",
        "first",
        "second",
        "minibatch_loss",
        "gradient_norm",
        "gradient_finite",
    ],
)
def test_nonfinite_returned_proposal_is_failure_not_scale_rejection(monkeypatch, field):
    monkeypatch.setattr(inherited_tests, "safe", full)
    monkeypatch.setattr(inherited_tests, "settings", settings)
    inherited_tests.test_nonfinite_optimizer_state_aborts_with_actual_proposal_prefix(
        monkeypatch, field
    )


@pytest.mark.parametrize(
    "clock,completed", [([0.0, 7201.0], 0), ([0.0, 1.0, 7201.0], 1)]
)
def test_internal_timeout_keeps_real_prefix(monkeypatch, clock, completed):
    monkeypatch.setattr(inherited_tests, "safe", full)
    monkeypatch.setattr(inherited_tests, "settings", settings)
    inherited_tests.test_time_limit_preserves_real_prefix_before_or_after_attempt(
        monkeypatch, clock, completed
    )


@pytest.mark.parametrize("bad_call", [0, 1])
def test_nonfinite_original_development_diagnostic(monkeypatch, bad_call):
    monkeypatch.setattr(inherited_tests, "safe", full)
    monkeypatch.setattr(inherited_tests, "settings", settings)
    inherited_tests.test_nonfinite_original_development_diagnostic_is_numeric_failure(
        monkeypatch, bad_call
    )


def test_wrapper_archive_derivatives_and_historical_globals(monkeypatch, tmp_path):
    public, _, _ = prepared_pair()
    original = public.fingerprint()
    recipes = copy.deepcopy((safe.RECIPE, safe.FITTING_RECIPE))
    monkeypatch.setitem(full.FITTING_RECIPE, "steps", 2)
    monkeypatch.setitem(full.FITTING_RECIPE, "check_every", 1)
    fitted = full.fit_candidate(public)
    assert public.fingerprint() == original
    assert fitted._train is public._train and fitted._development is public._development
    assert type(fitted) is full.CandidateDynamics
    assert type(fitted._model) is full.BalancedQuadraticSequenceModel
    assert fitted.recipe == full.RECIPE
    matching = fitted.report["matching"]
    assert matching["batch_size"] == 384 and matching["minibatch_seed"] is None
    assert matching["gradient_sampling"] == "ordered_full_cache"
    gradient = fitted.report["optimization"]["gradient"]
    assert gradient["known_gradient_window_visits"] == 768
    assert gradient["gradient_proposal_calls_returned"] == 2
    path = tmp_path / "full.npz"
    fitted.save(path)
    loaded = full.CandidateDynamics.load(path)
    assert loaded.fingerprint() == fitted.fingerprint()
    with pytest.raises(ValueError, match="format"):
        safe.CandidateDynamics.load(path)
    data = public._train.batch
    inputs = data.past_states[:2], data.past_inputs[:2], data.future_inputs[:2]
    same_tree(loaded.predict(*inputs), fitted.predict(*inputs))
    np.testing.assert_allclose(
        jax.jit(loaded.predict)(*inputs), fitted.predict(*inputs), atol=1e-12
    )
    derivative = jax.jacfwd(
        lambda commands: loaded.predict(inputs[0][0], inputs[1][0], commands)
    )(jnp.asarray(inputs[2][0]))
    assert np.isfinite(derivative).all()
    assert (safe.RECIPE, safe.FITTING_RECIPE) == recipes
    assert full._wrapper._shared is not safe._wrapper._shared
    assert not hasattr(loaded, "update")
