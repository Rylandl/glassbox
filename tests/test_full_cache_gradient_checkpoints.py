"""Full-cache observer tests use only asymmetric small cached toy windows."""

import copy
import json
import sys

import jax
import numpy as np
import pytest
import test_safeguarded_adam_checkpoints as prior
from test_fit_checkpoints import (
    PROVENANCE,
    SETTINGS,
    TOY_RECIPE,
    assert_model_equal,
    toy_scale,
)

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.experimental import full_cache_gradient_checkpoints as checkpoints
from glassbox.experimental import full_cache_gradient_model as model

SETTINGS_FULL = {**SETTINGS, "batch_size": 12}
RECIPE_FULL = {**TOY_RECIPE, "batch_size": 12}


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def source_map():
    paths = (
        "docs/harness/full-cache-gradient-v1.json",
        "src/glassbox/experimental/full_cache_gradient_model.py",
        "src/glassbox/experimental/full_cache_gradient_checkpoints.py",
        "tests/test_full_cache_gradient_model.py",
        "tests/test_full_cache_gradient_checkpoints.py",
    )
    return {
        **prior.source_map(),
        **{path: checkpoints._sha(checkpoints.ROOT / path) for path in paths},
    }


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    balanced = prior.balanced_fitted.__wrapped__(tmp_path_factory)
    historical = prior.fitted.__wrapped__(tmp_path_factory, balanced)
    train, dev = historical["train"], historical["dev"]
    with jax.enable_x64(True):
        solve, value_and_grad = np.linalg.solve, jax.value_and_grad
        calls = dict(solve=0, gradient=0)

        def counted_solve(*args, **kwargs):
            calls["solve"] += 1
            return solve(*args, **kwargs)

        def counted_gradient(*args, **kwargs):
            calls["gradient"] += 1
            return value_and_grad(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(np.linalg, "solve", counted_solve)
            patch.setattr(jax, "value_and_grad", counted_gradient)
            plain, plain_report = model.fit_candidate_sequence(
                train.batch, dev.batch, **SETTINGS_FULL, error_scale=toy_scale(train)
            )
            plain_calls = dict(calls)
            calls.update(solve=0, gradient=0)
            previous = sys.gettrace()
            with checkpoints.capture_fit(
                "candidate",
                PROVENANCE,
                source_sha256=source_map(),
                expected_steps=(0, 1, 2),
            ) as captured:
                observed, report = model.fit_candidate_sequence(
                    train.batch,
                    dev.batch,
                    **SETTINGS_FULL,
                    error_scale=toy_scale(train),
                )
            assert sys.gettrace() is previous
            assert calls == plain_calls == dict(solve=2, gradient=1)
    return dict(
        train=train,
        dev=dev,
        reference=historical["reference"],
        historical=historical,
        plain=plain,
        plain_report=plain_report,
        observed=observed,
        report=report,
        captured=captured,
    )


def save_fitted(path, fitted):
    report = fitted["report"]
    evidence = checkpoints.save_checkpoints(
        path,
        fitted["captured"],
        train=fitted["train"],
        development=fitted["dev"],
        recipe=RECIPE_FULL,
        anchor_reference=fitted["reference"],
        selected_step=report["selected_step"],
        selected_model=fitted["observed"],
    )
    kwargs = dict(
        train=fitted["train"],
        development=fitted["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        anchor_reference=fitted["reference"],
        selected_model=fitted["observed"],
        expectation=dict(
            arm="candidate",
            recipe=RECIPE_FULL,
            expected_steps=[0, 1, 2],
            completed=True,
            failure=None,
            selected_step=report["selected_step"],
            trace=report["trace"],
            unweighted_trace=report["unweighted_development_trace"],
            objective=report["objective"],
            safeguard=report["safeguard"],
            gradient=report["gradient"],
            status="complete",
        ),
    )
    return evidence, kwargs


def reseal(path):
    return prior.reseal(path)


def test_actual_parity_ordered_dtype_and_gradient_loss_identity(fitted):
    assert_model_equal(fitted["plain"], fitted["observed"])
    assert fitted["plain_report"] == fitted["report"]
    state = fitted["captured"].safeguard_observation
    assert len(state["attempts"]) == len(state["proposals"]) == 2
    for row in state["attempts"]:
        assert row["minibatch_indices"] == list(range(12))
        assert row["gradient_indices_dtype"] == np.dtype(np.int64).str
        assert np.isclose(
            row["minibatch_loss"]["value"],
            row["current_full_training_loss"]["value"],
            rtol=1e-10,
            atol=1e-12,
        )
    assert fitted["report"]["gradient"] == checkpoints._gradient(state["attempts"], 12)
    assert fitted["report"]["gradient"]["known_gradient_window_visits"] == 24
    assert all(
        not array.flags.writeable
        for proposal in state["proposals"]
        for array in proposal.values()
    )


def test_full_replay_has_no_gradient_rng_or_solve_and_preserves_historical_behavior(
    tmp_path, fitted, monkeypatch
):
    path = tmp_path / "new"
    evidence, kwargs = save_fitted(path, fitted)

    def forbidden(*args, **kwargs):
        pytest.fail("full-cache replay executed gradient, sampling RNG or solve")

    monkeypatch.setattr(jax, "grad", forbidden)
    monkeypatch.setattr(jax, "value_and_grad", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    result = checkpoints.replay_checkpoints(path, **kwargs)
    assert (
        result["exact"] and result["checkpoints"] == result["training_checkpoints"] == 3
    )
    assert result["gradient"] == evidence["gradient"] == fitted["report"]["gradient"]
    assert (
        result["safeguard"]["full_training_losses_replayed"]
        == 1 + result["safeguard"]["trial_objective_calls"]
    )
    assert not result["gradient_or_adam_reexecuted"]
    monkeypatch.undo()
    previous = checkpoints.previous
    assert previous._model() is prior.model
    assert previous._weighted()._model() is prior.model
    assert previous.FORMAT == "glassbox-safeguarded-adam-checkpoints-v1"
    old_path = tmp_path / "historical"
    _, old_kwargs = prior.save_fitted(old_path, fitted["historical"])
    assert previous.replay_checkpoints(old_path, **old_kwargs)["exact"]


@pytest.mark.parametrize(
    "change",
    (
        "duplicate",
        "missing",
        "reordered",
        "float",
        "bool",
        "dtype32",
        "missing_dtype",
        "returned_loss",
        "count",
        "windows64",
        "seed",
        "scope",
        "index_descriptor",
        "partial",
        "proposal",
        "acceptance",
    ),
)
def test_coherent_current_gradient_and_chain_tamper_rejected(tmp_path, fitted, change):
    path = tmp_path / "checkpoints"
    _, kwargs = save_fitted(path, fitted)
    transcript = json.loads((path / "attempts.json").read_text())
    row = transcript["attempts"][0]
    if change == "duplicate":
        row["minibatch_indices"][1] = 0
    elif change == "missing":
        row["minibatch_indices"].pop()
    elif change == "reordered":
        row["minibatch_indices"][0:2] = [1, 0]
    elif change == "float":
        row["minibatch_indices"][1] = 1.0
    elif change == "bool":
        row["minibatch_indices"][1] = True
    elif change == "dtype32":
        row["gradient_indices_dtype"] = "<i4"
    elif change == "missing_dtype":
        del row["gradient_indices_dtype"]
    elif change == "returned_loss":
        row["minibatch_loss"]["value"] += 0.1
    elif change == "partial":
        transcript["attempts"][-1]["completed"] = False
    elif change == "proposal":
        metadata, arrays = load_arrays(path / "proposals.npz")
        arrays["proposal_bias"][0, 0] += 0.1
        save_arrays(path / "proposals.npz", metadata, arrays)
    elif change == "acceptance":
        row["accepted_scale"] = 0.123
    else:
        key, value = {
            "count": ("known_gradient_window_visits", 128),
            "windows64": ("training_windows", 64),
            "seed": ("sampling_seed", 10000),
            "scope": ("loss_scope", "minibatch"),
            "index_descriptor": ("index_dtype", "int32"),
        }[change]
        transcript["gradient"][key] = value
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["gradient"][key] = value
        (path / "manifest.json").write_text(json.dumps(manifest))
        kwargs["expectation"] = copy.deepcopy(kwargs["expectation"])
        kwargs["expectation"]["gradient"][key] = value
    (path / "attempts.json").write_text(json.dumps(transcript))
    kwargs["expected_sha256"] = reseal(path)
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.check_checkpoint_links(path, **kwargs)


def failed_kwargs(path, fitted, captured, recipe, status):
    failure = captured.failure["message"]
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=fitted["train"],
        development=fitted["dev"],
        recipe=recipe,
        anchor_reference=fitted["reference"],
        failure=failure,
    )
    return dict(
        train=fitted["train"],
        development=fitted["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        anchor_reference=fitted["reference"],
        expectation=dict(
            arm="candidate",
            recipe=recipe,
            expected_steps=[0, 1, 2],
            completed=False,
            failure=failure,
            selected_step=None,
            trace=None,
            unweighted_trace=None,
            objective=None,
            safeguard=None,
            gradient=None,
            status=status,
        ),
    )


def test_preparation_failure_and_prior_tracer_are_preserved(tmp_path, fitted):
    previous = sys.gettrace()

    def tracer(frame, event, argument):
        return tracer

    try:
        sys.settrace(tracer)
        with pytest.raises(RuntimeError, match="preparation unavailable"):
            with checkpoints.capture_fit(
                "candidate",
                PROVENANCE,
                source_sha256=source_map(),
                expected_steps=(0, 1, 2),
            ) as captured:
                raise RuntimeError("preparation unavailable")
        assert sys.gettrace() is tracer
    finally:
        sys.settrace(previous)
    path = tmp_path / "failed"
    kwargs = failed_kwargs(
        path, fitted, captured, RECIPE_FULL, "preparation_unavailable"
    )
    result = checkpoints.replay_checkpoints(path, **kwargs)
    gradient = result["gradient"]
    assert (
        gradient["attempts_started"]
        == gradient["gradient_proposal_calls_returned"]
        == gradient["known_gradient_window_visits"]
        == 0
    )
    assert not gradient["incomplete_gradient_work_unknown"]


@pytest.mark.parametrize(
    "stage", ("before_proposal", "after_first_trial", "nonfinite_proposal")
)
def test_partial_gradient_accounting_distinguishes_unknown_from_returned_work(
    tmp_path, fitted, monkeypatch, stage
):
    original_jit = jax.jit
    settings = {**SETTINGS_FULL, "learning_rate": 1e200}
    recipe = {**RECIPE_FULL, "learning_rate": 1e200}
    calls = []

    def interrupt(function, *args, **kwargs):
        compiled = original_jit(function, *args, **kwargs)
        if function.__name__ == "update" and stage in (
            "before_proposal",
            "nonfinite_proposal",
        ):

            def stopped(*args):
                if stage == "before_proposal":
                    raise model.CandidateFitError(
                        "injected interruption before proposal"
                    )
                proposal, *other = compiled(*args)
                return (jax.tree.map(lambda value: value * np.nan, proposal), *other)

            return stopped
        if (
            function.__qualname__.startswith("make_training_objective.")
            and stage == "after_first_trial"
        ):

            def stopped(*args):
                calls.append(1)
                if len(calls) == 3:
                    raise model.CandidateFitError("injected interruption during trials")
                return compiled(*args)

            return stopped
        return compiled

    monkeypatch.setattr(jax, "jit", interrupt)
    with pytest.raises(ValueError, match=r"injected interruption|nonfinite"):
        with checkpoints.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as captured:
            model.fit_candidate_sequence(
                fitted["train"].batch,
                fitted["dev"].batch,
                **settings,
                error_scale=toy_scale(fitted["train"]),
            )
    monkeypatch.undo()
    path = tmp_path / "interrupted"
    kwargs = failed_kwargs(path, fitted, captured, recipe, "fit_failure")
    result = checkpoints.replay_checkpoints(path, **kwargs)
    gradient = result["gradient"]
    returned = int(stage != "before_proposal")
    assert gradient["attempts_started"] == 1
    assert gradient["gradient_proposal_calls_returned"] == returned
    assert gradient["known_gradient_window_visits"] == 12 * returned
    assert gradient["completed_acceptance_attempts"] == 0
    assert gradient["incomplete_gradient_work_unknown"] == (stage == "before_proposal")
    assert result["safeguard"]["trial_objective_calls"] == int(
        stage == "after_first_trial"
    )


@pytest.mark.parametrize(
    "boundary", ("before_next_checkpoint", "after_final_checkpoint")
)
def test_timeout_prefix_counters(tmp_path, fitted, monkeypatch, boundary):
    class Clock:
        calls = 0

        @classmethod
        def perf_counter(cls):
            cls.calls += 1
            limit = 2 if boundary == "before_next_checkpoint" else 5
            return 0.0 if cls.calls <= limit else 7201.0

    monkeypatch.setattr(model, "time", Clock)
    with pytest.raises(ValueError, match="fit_time_limit"):
        with checkpoints.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as captured:
            model.fit_candidate_sequence(
                fitted["train"].batch,
                fitted["dev"].batch,
                **SETTINGS_FULL,
                error_scale=toy_scale(fitted["train"]),
            )
    monkeypatch.undo()
    path = tmp_path / "timeout"
    kwargs = failed_kwargs(path, fitted, captured, RECIPE_FULL, "fit_failure")
    result = checkpoints.replay_checkpoints(path, **kwargs)
    completed = 1 if boundary == "before_next_checkpoint" else 2
    assert (
        result["gradient"]["attempts_started"]
        == result["gradient"]["gradient_proposal_calls_returned"]
        == result["gradient"]["completed_acceptance_attempts"]
        == completed
    )
    assert result["gradient"]["known_gradient_window_visits"] == 12 * completed
    assert result["checkpoints"] == (1 if completed == 1 else 3)
    assert not result["gradient"]["incomplete_gradient_work_unknown"]


@pytest.mark.parametrize("dtype", (np.int32, np.float64))
def test_marker_requires_actual_integer_dtype_before_serialization(dtype):
    with pytest.raises(checkpoints.CheckpointError, match="not int64"):
        checkpoints._record(
            {}, dict(phase="started", indices=np.arange(12, dtype=dtype))
        )


@pytest.mark.parametrize("difference,accepted", ((5e-13, True), (1e-7, False)))
def test_gradient_loss_diagnostic_tolerance_never_changes_acceptance(
    fitted, difference, accepted
):
    captured = fitted["captured"]
    observed = captured.safeguard_observation
    evidence = copy.deepcopy(
        {key: value for key, value in observed.items() if key != "proposals"}
    )
    evidence["attempts"] = evidence["attempts"][:1]
    row = evidence["attempts"][0]
    original_post_loss = copy.deepcopy(row["cached_post_full_training_loss"])
    original_scale = row["accepted_scale"]
    row["minibatch_loss"]["value"] += difference
    evidence["gradient"] = checkpoints._gradient(evidence["attempts"], 12)

    def validate():
        return checkpoints._chain(
            evidence,
            observed["proposals"][:1],
            captured.weight_observation["arrays"],
            captured.snapshots[:1],
            fitted["train"],
            RECIPE_FULL,
            completed=False,
            selected_step=None,
            safeguard=None,
        )

    if accepted:
        result = validate()
        assert result["completed_attempts"] == 1
        assert row["cached_post_full_training_loss"] == original_post_loss
        assert row["accepted_scale"] == original_scale
    else:
        with pytest.raises(checkpoints.CheckpointError, match="returned loss differs"):
            validate()


def test_checkpoint_observation_binds_exact_returned_loss_mirror(tmp_path, fitted):
    path = tmp_path / "checkpoints"
    _, kwargs = save_fitted(path, fitted)
    transcript = json.loads((path / "attempts.json").read_text())
    transcript["attempts"][0]["minibatch_loss"]["value"] += 5e-13
    (path / "attempts.json").write_text(json.dumps(transcript))
    kwargs["expected_sha256"] = reseal(path)
    with pytest.raises(checkpoints.CheckpointError, match="gradient-loss mirror"):
        checkpoints.check_checkpoint_links(path, **kwargs)
