"""Synthetic checks of the frozen safeguard, optimizer identity and fit boundary."""

import ast
import copy
import inspect
import sys
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_excited_public_matching import prepared_pair
from test_initial_channel_balance_model import settings
from test_state_quadratic_model import small_batch

from glassbox.experimental import affine_anchored_model as anchored
from glassbox.experimental import initial_channel_balance_model as balanced
from glassbox.experimental import safeguarded_adam_model as safe


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def copied(value):
    return jax.tree.map(
        lambda leaf: (
            np.array(leaf, copy=True)
            if isinstance(leaf, (jax.Array, np.ndarray))
            else leaf
        ),
        value,
    )


def same_tree(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()


def capture(monkeypatch):
    events = []
    monkeypatch.setattr(
        safe, "_observe_attempt", lambda state: events.append(copied(state))
    )
    return events


def phase(events, name):
    return [event for event in events if event["phase"] == name]


def scripted_objective(monkeypatch, losses):
    remaining = iter(losses)
    calls = []

    def make(*args):
        def objective(params):
            calls.append(copied(params))
            return next(remaining)

        return objective

    monkeypatch.setattr(safe, "make_training_objective", make)
    return calls


def test_exact_initialization_weights_and_canonical_helper_identity():
    train, dev = small_batch(4), small_batch(5)
    old, old_report = balanced.fit_candidate_sequence(
        train, dev, steps=0, **settings(train)
    )
    new, report = safe.fit_candidate_sequence(train, dev, steps=0, **settings(train))
    same_tree(new.arrays(), old.arrays())
    assert report["objective"] == old_report["objective"]
    assert report["trace"] == old_report["trace"]
    assert (
        report["unweighted_development_trace"]
        == old_report["unweighted_development_trace"]
    )
    assert report["safeguard"]["full_training_objective_calls"] == 1
    assert (
        report["safeguard"]["selected_full_training_loss"]
        == report["safeguard"]["initial_full_training_loss"]
    )
    assert safe.initialize_candidate is anchored.initialize_candidate
    assert safe.initial_training_forecast is balanced.initial_training_forecast
    assert safe.weighting_metadata is balanced.weighting_metadata
    assert safe._rollout is balanced._rollout
    assert safe.SCALES == tuple(2.0**-index for index in range(8))
    assert safe.FIT_WALL_TIME_LIMIT_S == 7200


def test_trial_interpolation_preserves_full_proposal_bytes():
    current = {"a": jnp.array([1e20, -1e20]), "b": jnp.array([[0.5], [-2.0]])}
    proposal = {"a": jnp.array([1.0, -1.0]), "b": jnp.array([[2.0], [0.1]])}
    assert safe.trial_parameters(current, proposal, 1.0) is proposal
    assert not np.array_equal(
        np.asarray(current["a"] + (proposal["a"] - current["a"])), proposal["a"]
    )
    for alpha in safe.SCALES[1:]:
        expected = jax.tree.map(
            lambda p, q, a=alpha: p + a * (q - p), current, proposal
        )
        same_tree(safe.trial_parameters(current, proposal, alpha), expected)


def test_canonical_full_training_evaluator_matches_manual_loss_and_replays():
    train = small_batch(4)
    model = safe.initialize_candidate(
        train, seed=7, width=4, memory=2, ridge=0.7, delay_steps=1
    )
    scale = settings(train)["error_scale"]
    weights = np.array([0.7, 1.1, 1.2])
    evaluate = safe.make_training_objective(model.norms, train, scale, weights, 1)
    actual = float(evaluate(model.params))
    prediction = np.asarray(
        model.rollout(train.past_states, train.past_inputs, train.future_inputs)
    )
    expected = np.mean(weights * ((prediction - train.future_states) / scale) ** 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    replay = safe.make_training_objective(model.norms, train, scale, weights, 1)
    same_tree(evaluate(model.params), replay(model.params))
    changed = safe.make_training_objective(
        model.norms,
        replace(train, future_states=train.future_states + 0.1),
        scale,
        weights,
        1,
    )
    assert float(changed(model.params)) != actual


@pytest.mark.parametrize("clipping", [False, True])
def test_proposals_and_moments_exactly_match_original_adam_arithmetic(
    monkeypatch, clipping
):
    train = small_batch(4)
    args = settings(train)
    if clipping:
        args["error_scale"] = args["error_scale"] * 1e-4

        # Give every toy output the same initial weighting error so the perfectly
        # constant third channel cannot dominate the weights. Both Adam fitters
        # see the same fixture; this isolates active clipping arithmetic.
        def forecast(*values):
            return train.future_states + args["error_scale"]

        monkeypatch.setattr(balanced, "initial_training_forecast", forecast)
        monkeypatch.setattr(safe, "initial_training_forecast", forecast)
    old_snapshots = []
    source = inspect.getsource(balanced.fit_candidate_sequence)
    start = inspect.getsourcelines(balanced.fit_candidate_sequence)[1] - 1
    line = start + next(
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "unweighted_development_trace"
    )

    def trace(frame, event, arg):
        if frame.f_code is balanced.fit_candidate_sequence.__code__:
            if event == "line" and frame.f_lineno == line:
                local = frame.f_locals
                if not old_snapshots or old_snapshots[-1]["i"] != local["i"]:
                    old_snapshots.append(
                        copied(
                            {
                                key: local[key]
                                for key in ("i", "params", "first", "second")
                            }
                        )
                    )
            return trace
        return None

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        balanced.fit_candidate_sequence(
            train,
            train,
            steps=3,
            check_every=1,
            batch_size=7,
            learning_rate=1e-4,
            **args,
        )
    finally:
        sys.settrace(previous)
    events = capture(monkeypatch)
    # Force full acceptance to compare precisely the old Adam trajectory, including
    # its nonzero moment states. The mathematical full objective is tested separately.
    scripted_objective(monkeypatch, [4.0, 3.0, 2.0, 1.0])
    safe.fit_candidate_sequence(
        train,
        train,
        steps=3,
        check_every=1,
        batch_size=7,
        learning_rate=1e-4,
        **args,
    )
    assert len(old_snapshots) == 3
    for old, new in zip(old_snapshots, phase(events, "proposed"), strict=True):
        same_tree(old["params"], new["proposal"])
        same_tree(old["first"], new["new_first"])
        same_tree(old["second"], new["new_second"])
        if clipping:
            assert new["gradient_norm"] > 5
            clipped_norm = np.sqrt(
                sum(np.sum(g * g) for g in new["clipped_gradient"].values())
            )
            assert clipped_norm <= 5 + 1e-12


def test_first_strict_acceptable_scale_and_nonfinite_trials(monkeypatch):
    train = small_batch(4)
    events = capture(monkeypatch)
    calls = scripted_objective(
        monkeypatch, [10.0, 12.0, 10.0, 9.0, np.nan, np.inf, -np.inf, 9.0, 8.0]
    )
    _, report = safe.fit_candidate_sequence(
        train, train, steps=2, check_every=1, **settings(train)
    )
    completed = phase(events, "completed")
    assert [row["accepted_scale"] for row in completed] == [0.25, 0.0625]
    assert [row["accepted_trial_index"] for row in completed] == [2, 4]
    assert len(calls) == 9 == report["safeguard"]["full_training_objective_calls"]
    for start, proposed, done in zip(
        phase(events, "started"), phase(events, "proposed"), completed, strict=True
    ):
        same_tree(
            done["params"],
            safe.trial_parameters(
                start["params"], proposed["proposal"], done["accepted_scale"]
            ),
        )
    assert report["safeguard"]["accepted_attempts"] == 2
    assert report["safeguard"]["full_training_objective_calls_max"] == 17
    assert np.isnan(phase(events, "trial")[3]["loss"])


def test_all_rejected_retains_parameters_loss_and_advances_moments(monkeypatch):
    train = small_batch(4)
    events = capture(monkeypatch)
    calls = scripted_objective(monkeypatch, [10.0] * 17)
    _, report = safe.fit_candidate_sequence(
        train, train, steps=2, check_every=1, **settings(train)
    )
    starts, proposed, completed = (
        phase(events, name) for name in ("started", "proposed", "completed")
    )
    for before, proposal, after in zip(starts, proposed, completed, strict=True):
        same_tree(before["params"], after["params"])
        same_tree(proposal["new_first"], after["first"])
        same_tree(proposal["new_second"], after["second"])
        assert after["accepted_scale"] == 0 and after["accepted_trial_index"] is None
        assert before["current_full_training_loss"] == after["loss"] == 10
    same_tree(starts[1]["first"], completed[0]["first"])
    assert any(np.any(x) for x in jax.tree.leaves(completed[0]["first"]))
    assert report["selected_step"] == 0
    assert len(calls) == 17
    summary = report["safeguard"]
    assert summary["maximum_rejection_streak"] == summary["rejected_attempts"] == 2
    assert summary["accepted_attempts"] == 0
    assert (
        summary["final_full_training_loss"]
        == summary["selected_full_training_loss"]
        == 10
    )
    expected = np.random.default_rng(10007)
    for start in starts:
        np.testing.assert_array_equal(
            start["indices"],
            expected.integers(
                len(train.past_states), size=min(64, len(train.past_states))
            ),
        )


def test_actual_full_training_losses_are_monotone_and_selection_unchanged(monkeypatch):
    train, dev = small_batch(4), small_batch(5)
    events = capture(monkeypatch)
    _, report = safe.fit_candidate_sequence(
        train, dev, steps=5, check_every=1, **settings(train)
    )
    initial = phase(events, "initial_objective")[0]["loss"]
    completed = phase(events, "completed")
    loss = initial
    for row in completed:
        assert row["loss"] <= loss
        assert (row["loss"] < loss) == (row["accepted_scale"] != 0)
        loss = row["loss"]
    assert (
        report["safeguard"]["full_training_objective_calls"]
        == 1 + len(phase(events, "trial"))
        <= 41
    )
    assert (
        report["selected_step"]
        == min(report["trace"], key=lambda row: row["validation_rollout_mse"])["step"]
    )
    assert (
        report["safeguard"]["selected_full_training_loss"]
        == report["safeguard"]["checkpoint_full_training_losses"][
            report["selected_step"]
        ]["full_training_loss"]
    )


def test_development_cannot_change_training_proposals_or_acceptance(monkeypatch):
    train, dev = small_batch(4), small_batch(5)
    first = capture(monkeypatch)
    _, a = safe.fit_candidate_sequence(
        train, dev, steps=2, check_every=1, **settings(train)
    )
    second = capture(monkeypatch)
    _, b = safe.fit_candidate_sequence(
        train,
        replace(dev, future_states=dev.future_states + 0.2),
        steps=2,
        check_every=1,
        **settings(train),
    )
    assert a["objective"] == b["objective"]
    assert a["trace"][0] != b["trace"][0]
    for old, new in zip(first, second, strict=True):
        same_tree(old, new)


@pytest.mark.parametrize("loss", [np.nan, np.inf, -np.inf])
def test_nonfinite_initial_acceptance_loss_is_failure_before_attempts(
    monkeypatch, loss
):
    train = small_batch(4)
    events = capture(monkeypatch)
    scripted_objective(monkeypatch, [loss])
    with pytest.raises(safe.CandidateFitError, match="nonfinite initial full-training"):
        safe.fit_candidate_sequence(train, train, steps=1, **settings(train))
    assert [row["phase"] for row in events] == ["initial_objective"]


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
def test_nonfinite_optimizer_state_aborts_with_actual_proposal_prefix(
    monkeypatch, field
):
    real_jit = jax.jit
    index = {
        "proposal": 0,
        "first": 1,
        "second": 2,
        "minibatch_loss": 3,
        "gradient_norm": 4,
        "gradient_finite": 6,
    }[field]

    def jit(function, *args, **kwargs):
        compiled = real_jit(function, *args, **kwargs)
        if function.__name__ != "update":
            return compiled

        def broken(*values):
            result = list(compiled(*values))
            if index < 3:
                result[index] = jax.tree.map(
                    lambda leaf: jnp.full_like(leaf, np.inf), result[index]
                )
            else:
                result[index] = False if index == 6 else np.inf
            return tuple(result)

        return broken

    monkeypatch.setattr(safe, "jax", SimpleNamespace(**{**vars(jax), "jit": jit}))
    train = small_batch(4)
    events = capture(monkeypatch)
    with pytest.raises(
        safe.CandidateFitError, match="nonfinite Adam proposal at attempt 1"
    ):
        safe.fit_candidate_sequence(train, train, steps=2, **settings(train))
    assert [row["phase"] for row in events] == [
        "initial_objective",
        "started",
        "proposed",
    ]


@pytest.mark.parametrize(
    "clock,completed", [([0.0, 7201.0], 0), ([0.0, 1.0, 7201.0], 1)]
)
def test_time_limit_preserves_real_prefix_before_or_after_attempt(
    monkeypatch, clock, completed
):
    readings = iter(clock)
    monkeypatch.setattr(
        safe, "time", SimpleNamespace(perf_counter=lambda: next(readings))
    )
    train = small_batch(4)
    events = capture(monkeypatch)
    with pytest.raises(safe.CandidateFitError, match=r"^fit_time_limit$"):
        safe.fit_candidate_sequence(
            train, train, steps=2, check_every=1, **settings(train)
        )
    assert len(phase(events, "completed")) == completed
    assert len(phase(events, "started")) == completed
    assert phase(events, "initial_objective")


def test_final_time_limit_includes_last_development_evaluation(monkeypatch):
    readings = iter([0.0, 1.0, 2.0, 7201.0])
    monkeypatch.setattr(
        safe, "time", SimpleNamespace(perf_counter=lambda: next(readings))
    )
    train = small_batch(4)
    events = capture(monkeypatch)
    with pytest.raises(safe.CandidateFitError, match=r"^fit_time_limit$"):
        safe.fit_candidate_sequence(
            train, train, steps=1, check_every=1, **settings(train)
        )
    assert len(phase(events, "completed")) == 1
    assert next(readings, None) is None


@pytest.mark.parametrize("bad_development_call", [0, 1])
def test_nonfinite_original_development_diagnostic_is_numeric_failure(
    monkeypatch, bad_development_call
):
    real_jit = jax.jit
    calls = 0

    def jit(function, *args, **kwargs):
        compiled = real_jit(function, *args, **kwargs)
        if function.__qualname__ != "fit_candidate_sequence.<locals>.evaluate":
            return compiled

        def overflowing_original(*values):
            nonlocal calls
            weighted, original = compiled(*values)
            if calls == bad_development_call:
                original = jnp.asarray(np.inf)
            calls += 1
            return weighted, original

        return overflowing_original

    monkeypatch.setattr(safe, "jax", SimpleNamespace(**{**vars(jax), "jit": jit}))
    train = small_batch(4)
    events = capture(monkeypatch)
    message = (
        "initial recursive validation loss"
        if bad_development_call == 0
        else "nonfinite sequence training at step 1"
    )
    with pytest.raises(safe.CandidateFitError, match=message):
        safe.fit_candidate_sequence(
            train, train, steps=1, check_every=1, **settings(train)
        )
    assert len(phase(events, "completed")) == bad_development_call
    if bad_development_call == 0:
        assert events == []


def test_one_initializer_two_solves_one_weight_forecast_and_draw_per_attempt(
    monkeypatch,
):
    train, dev = small_batch(4), small_batch(5)
    original = (
        safe.initialize_candidate,
        np.linalg.solve,
        safe.initial_training_forecast,
        np.random.default_rng,
    )
    counts, seeds = [0, 0, 0], []

    def count(index, *args, **kwargs):
        counts[index] += 1
        return original[index](*args, **kwargs)

    def rng(seed):
        seeds.append(seed)
        return original[3](seed)

    monkeypatch.setattr(safe, "initialize_candidate", lambda *a, **k: count(0, *a, **k))
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: count(1, *a, **k))
    monkeypatch.setattr(
        safe, "initial_training_forecast", lambda *a, **k: count(2, *a, **k)
    )
    monkeypatch.setattr(np.random, "default_rng", rng)
    events = capture(monkeypatch)
    safe.fit_candidate_sequence(train, dev, steps=3, check_every=1, **settings(train))
    assert counts == [1, 2, 1] and seeds == [7, 10007]
    assert len(phase(events, "proposed")) == 3


def test_wrapper_archive_jax_derivatives_and_historical_globals(monkeypatch, tmp_path):
    public, _, _ = prepared_pair()
    original = public.fingerprint()
    old_recipe = copy.deepcopy(balanced.RECIPE)
    monkeypatch.setitem(safe.FITTING_RECIPE, "steps", 2)
    monkeypatch.setitem(safe.FITTING_RECIPE, "check_every", 1)
    fitted = safe.fit_candidate(public)
    assert public.fingerprint() == original
    assert fitted._train is public._train and fitted._development is public._development
    assert (
        type(fitted) is safe.CandidateDynamics
        and type(fitted._model) is safe.BalancedQuadraticSequenceModel
    )
    assert fitted.recipe == safe.RECIPE
    assert fitted.report["optimization"]["safeguard"]["completed_attempts"] == 2
    path = tmp_path / "guarded.npz"
    fitted.save(path)
    loaded = safe.CandidateDynamics.load(path)
    assert loaded.fingerprint() == fitted.fingerprint()
    with pytest.raises(ValueError, match="format"):
        balanced.CandidateDynamics.load(path)
    data = public._train.batch
    args = data.past_states[:2], data.past_inputs[:2], data.future_inputs[:2]
    same_tree(loaded.predict(*args), fitted.predict(*args))
    np.testing.assert_allclose(
        jax.jit(loaded.predict)(*args), fitted.predict(*args), atol=1e-12
    )
    assert np.isfinite(
        jax.jacfwd(lambda commands: loaded.predict(args[0][0], args[1][0], commands))(
            jnp.asarray(args[2][0])
        )
    ).all()
    assert balanced.RECIPE == old_recipe and balanced.FITTING_RECIPE["steps"] == 1000
    assert safe._wrapper._shared is not balanced._wrapper._shared
    assert not hasattr(loaded, "update")


def test_unknown_error_propagates_without_fabricated_attempt_completion(monkeypatch):
    train = small_batch(4)
    events = capture(monkeypatch)
    count = 0

    def make(*args):
        def objective(params):
            nonlocal count
            count += 1
            if count == 1:
                return 1.0
            raise RuntimeError("injected programming failure")

        return objective

    monkeypatch.setattr(safe, "make_training_objective", make)
    with pytest.raises(RuntimeError, match="programming failure"):
        safe.fit_candidate_sequence(train, train, steps=1, **settings(train))
    assert [row["phase"] for row in events] == [
        "initial_objective",
        "started",
        "proposed",
    ]
