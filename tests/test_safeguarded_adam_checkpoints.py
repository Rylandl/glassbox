"""Toy-only safeguard observation, replay and coherent evidence alterations."""

import copy
import json
import sys

import jax
import numpy as np
import pytest
import test_initial_channel_balance_checkpoints as prior
from test_fit_checkpoints import (
    PROVENANCE,
    SETTINGS,
    TOY_RECIPE,
    assert_model_equal,
    toy_scale,
)

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.experimental import safeguarded_adam_checkpoints as checkpoints
from glassbox.experimental import safeguarded_adam_model as model


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def source_map():
    paths = (
        "docs/harness/safeguarded-adam-v1.json",
        "src/glassbox/experimental/safeguarded_adam_checkpoints.py",
        "src/glassbox/experimental/safeguarded_adam_model.py",
        "tests/test_safeguarded_adam_checkpoints.py",
        "tests/test_safeguarded_adam_model.py",
    )
    return {
        **prior.source_map(),
        **{path: checkpoints._sha(checkpoints.ROOT / path) for path in paths},
    }


@pytest.fixture(scope="module")
def balanced_fitted(tmp_path_factory):
    return prior.fitted.__wrapped__(tmp_path_factory)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, balanced_fitted):
    train, dev = balanced_fitted["train"], balanced_fitted["dev"]
    reference_root = tmp_path_factory.mktemp("balanced-source") / "checkpoints"
    prior.save_fitted(reference_root, balanced_fitted)
    snapshot = reference_root / "anchored/trajectory/step-0000.npz"
    reference = dict(
        path=snapshot,
        sha256=checkpoints._sha(snapshot),
        source_bundle_sha256="a" * 64,
        source_checkpoint_manifest_sha256=checkpoints._sha(
            reference_root / "manifest.json"
        ),
        source_relative_path="toy/candidate/checkpoints/anchored/trajectory/step-0000.npz",
    )
    with jax.enable_x64(True):
        solve = np.linalg.solve
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return solve(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(np.linalg, "solve", counted)
            plain, plain_report = model.fit_candidate_sequence(
                train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
            )
            plain_calls = len(calls)
            calls.clear()
            previous = sys.gettrace()
            with checkpoints.capture_fit(
                "candidate",
                PROVENANCE,
                source_sha256=source_map(),
                expected_steps=(0, 1, 2),
            ) as captured:
                observed, report = model.fit_candidate_sequence(
                    train.batch, dev.batch, **SETTINGS, error_scale=toy_scale(train)
                )
            assert sys.gettrace() is previous
            assert plain_calls == len(calls) == 2
    return dict(
        train=train,
        dev=dev,
        reference=reference,
        plain=plain,
        plain_report=plain_report,
        observed=observed,
        report=report,
        captured=captured,
    )


def save_fitted(path, fitted):
    evidence = checkpoints.save_checkpoints(
        path,
        fitted["captured"],
        train=fitted["train"],
        development=fitted["dev"],
        recipe=TOY_RECIPE,
        anchor_reference=fitted["reference"],
        selected_step=fitted["report"]["selected_step"],
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
            recipe=TOY_RECIPE,
            expected_steps=[0, 1, 2],
            completed=True,
            failure=None,
            selected_step=fitted["report"]["selected_step"],
            trace=fitted["report"]["trace"],
            unweighted_trace=fitted["report"]["unweighted_development_trace"],
            objective=fitted["report"]["objective"],
            safeguard=fitted["report"]["safeguard"],
            status="complete",
        ),
    )
    return evidence, kwargs


def reseal(path):
    file = path / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["files"] = {
        str(item.relative_to(path)): checkpoints._sha(item)
        for item in path.rglob("*")
        if item.is_file() and item != file
    }
    file.write_text(json.dumps(manifest))
    return checkpoints._sha(file)


def test_actual_fitter_parity_read_only_observation_and_initial_identity(fitted):
    assert_model_equal(fitted["plain"], fitted["observed"])
    assert fitted["plain_report"] == fitted["report"]
    captured = fitted["captured"]
    assert captured.completed and captured.calls == 1
    state = captured.safeguard_observation
    assert len(state["attempts"]) == len(state["proposals"]) == 2
    assert all(row["completed"] for row in state["attempts"])
    assert all(
        not array.flags.writeable
        for proposal in state["proposals"]
        for array in proposal.values()
    )
    assert checkpoints._weighted() is not checkpoints.inherited
    assert checkpoints.inherited._model() is prior.model


def test_exact_full_replay_without_gradient_adam_or_solve(
    tmp_path, fitted, monkeypatch
):
    path = tmp_path / "checkpoints"
    evidence, kwargs = save_fitted(path, fitted)

    def forbidden(*args, **kwargs):
        pytest.fail("replay executed gradient or ridge solve")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(jax, "grad", forbidden)
    monkeypatch.setattr(jax, "value_and_grad", forbidden)
    result = checkpoints.replay_checkpoints(path, **kwargs)
    assert (
        result["exact"] and result["checkpoints"] == result["training_checkpoints"] == 3
    )
    assert not result["gradient_or_adam_reexecuted"]
    chain = result["safeguard"]
    assert chain["attempts"] == chain["completed_attempts"] == 2
    assert chain["checkpoint_chain_bound"] == 3
    assert chain["initial_objective_calls"] == 1
    assert chain["full_training_losses_replayed"] == 1 + chain["trial_objective_calls"]
    assert chain["accepted_attempts"] + chain["rejected_attempts"] == 2
    assert evidence["initial_identity"]["exact"]
    monkeypatch.setattr(
        jax, "jit", lambda *a, **kw: pytest.fail("cheap check compiled forecast")
    )
    assert checkpoints.check_checkpoint_links(path, **kwargs)["exact"]


@pytest.mark.parametrize(
    "change",
    (
        "indices",
        "pre",
        "moment_before",
        "moment_after",
        "scale",
        "acceptance",
        "post",
        "cached",
        "missing_attempt",
        "extra_trial",
        "finite_status",
        "proposal",
        "binding",
        "report",
        "initial",
    ),
)
def test_coherently_resealed_attempt_and_report_changes_rejected(
    tmp_path, fitted, change
):
    path = tmp_path / "checkpoints"
    _, kwargs = save_fitted(path, fitted)
    transcript = json.loads((path / "attempts.json").read_text())
    first = transcript["attempts"][0]
    if change == "indices":
        first["minibatch_indices"][0] += 1
    elif change in ("pre", "moment_before", "moment_after", "post"):
        name = dict(
            pre="pre_parameters_fingerprint",
            moment_before="previous_first_moment_fingerprint",
            moment_after="post_first_moment_fingerprint",
            post="post_parameters_fingerprint",
        )[change]
        first[name] = "0" * 64
    elif change == "scale":
        first["trials"][0]["scale"] = 0.125
    elif change == "acceptance":
        first["accepted_scale"] = 0.123
    elif change == "cached":
        first["cached_post_full_training_loss"]["value"] += 1
    elif change == "missing_attempt":
        transcript["attempts"].pop()
    elif change == "extra_trial":
        first["trials"].append(copy.deepcopy(first["trials"][-1]))
    elif change == "finite_status":
        first["gradient_finite"] = False
    elif change == "proposal":
        file = path / "proposals.npz"
        metadata, arrays = load_arrays(file)
        arrays[sorted(arrays)[0]].flat[0] += 0.01
        save_arrays(file, metadata, arrays)
    elif change == "binding":
        transcript["binding"]["trial_parameters"] = "0" * 64
    elif change == "report":
        kwargs["expectation"] = copy.deepcopy(kwargs["expectation"])
        kwargs["expectation"]["safeguard"]["full_training_objective_calls"] += 1
    elif change == "initial":
        transcript["initial"]["loss"]["value"] += 1
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
            status=status,
        ),
    )


def test_preparation_failure_prefix_and_tracer_restoration(tmp_path, fitted):
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
        path, fitted, captured, TOY_RECIPE, "preparation_unavailable"
    )
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    assert not replay["weighting_available"] and replay["checkpoints"] == 0
    assert (
        replay["safeguard"]["attempts"]
        == replay["safeguard"]["initial_objective_calls"]
        == 0
    )


def test_timeout_after_completed_attempt_keeps_prefix_without_late_checkpoint(
    tmp_path, fitted, monkeypatch
):
    # Clock is confined to the fitter module, avoiding observer timing effects.
    class Clock:
        calls = 0

        @classmethod
        def perf_counter(cls):
            cls.calls += 1
            return 0.0 if cls.calls <= 2 else 7201.0

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
                **SETTINGS,
                error_scale=toy_scale(fitted["train"]),
            )
    monkeypatch.undo()
    assert len(captured.snapshots) == 1
    assert captured.safeguard_observation["attempts"][0]["completed"]
    path = tmp_path / "timeout"
    kwargs = failed_kwargs(path, fitted, captured, TOY_RECIPE, "fit_failure")
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    assert replay["safeguard"]["completed_attempts"] == 1
    assert replay["safeguard"]["checkpoint_chain_bound"] == 1


def test_reference_outer_manifest_is_actual_trust_anchor(tmp_path, fitted):
    path = tmp_path / "checkpoints"
    _, kwargs = save_fitted(path, fitted)
    for field in ("source_checkpoint_manifest_sha256", "sha256"):
        with pytest.raises(checkpoints.CheckpointError):
            checkpoints.check_checkpoint_links(
                path,
                **{
                    **kwargs,
                    "anchor_reference": {**fitted["reference"], field: "0" * 64},
                },
            )


def test_replay_rejects_coherent_rescaling_of_every_acceptance_loss(tmp_path, fitted):
    path = tmp_path / "checkpoints"
    _, kwargs = save_fitted(path, fitted)
    transcript = json.loads((path / "attempts.json").read_text())
    transcript["initial"]["loss"]["value"] *= 2
    for row in transcript["attempts"]:
        for key in ("current_full_training_loss", "cached_post_full_training_loss"):
            row[key]["value"] *= 2
        for trial in row["trials"]:
            if trial["loss"]["status"] == "finite":
                trial["loss"]["value"] *= 2
    kwargs["expectation"] = copy.deepcopy(kwargs["expectation"])
    report = kwargs["expectation"]["safeguard"]
    for key in (
        "initial_full_training_loss",
        "final_full_training_loss",
        "selected_full_training_loss",
    ):
        report[key] *= 2
    for row in report["checkpoint_full_training_losses"]:
        row["full_training_loss"] *= 2
    (path / "attempts.json").write_text(json.dumps(transcript))
    kwargs["expected_sha256"] = reseal(path)
    # The scalar order and every parameter/checkpoint relation still agree.
    assert checkpoints.check_checkpoint_links(path, **kwargs)["exact"]
    with pytest.raises(
        checkpoints.CheckpointError, match="initial full-training objective replay"
    ):
        checkpoints.replay_checkpoints(path, **kwargs)


def test_all_nonfinite_trials_are_rejections_with_finite_proposals(tmp_path, fitted):
    settings = {**SETTINGS, "learning_rate": 1e200}
    recipe = {**TOY_RECIPE, "learning_rate": 1e200}
    with checkpoints.capture_fit(
        "candidate", PROVENANCE, source_sha256=source_map(), expected_steps=(0, 1, 2)
    ) as captured:
        observed, report = model.fit_candidate_sequence(
            fitted["train"].batch,
            fitted["dev"].batch,
            **settings,
            error_scale=toy_scale(fitted["train"]),
        )
    assert report["safeguard"]["rejected_attempts"] == 2
    for row in captured.safeguard_observation["attempts"]:
        assert row["accepted_scale"] == 0
        assert len(row["trials"]) == 8
        assert all(trial["loss"]["status"] != "finite" for trial in row["trials"])
        assert row["pre_parameters_fingerprint"] == row["post_parameters_fingerprint"]
        assert (
            row["previous_first_moment_fingerprint"]
            != row["post_first_moment_fingerprint"]
        )
    path = tmp_path / "rejected"
    evidence = checkpoints.save_checkpoints(
        path,
        captured,
        train=fitted["train"],
        development=fitted["dev"],
        recipe=recipe,
        anchor_reference=fitted["reference"],
        selected_step=report["selected_step"],
        selected_model=observed,
    )
    kwargs = dict(
        train=fitted["train"],
        development=fitted["dev"],
        provenance=PROVENANCE,
        source_sha256=source_map(),
        expected_sha256=evidence["manifest_sha256"],
        anchor_reference=fitted["reference"],
        selected_model=observed,
        expectation=dict(
            arm="candidate",
            recipe=recipe,
            expected_steps=[0, 1, 2],
            completed=True,
            failure=None,
            selected_step=report["selected_step"],
            trace=report["trace"],
            unweighted_trace=report["unweighted_development_trace"],
            objective=report["objective"],
            safeguard=report["safeguard"],
            status="complete",
        ),
    )
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    assert replay["safeguard"]["trial_objective_calls"] == 16
    assert replay["safeguard"]["rejected_attempts"] == 2
    assert "NaN" not in (path / "attempts.json").read_text()
    assert "Infinity" not in (path / "attempts.json").read_text()


@pytest.mark.parametrize("stage", ("weighting", "initial_development", "proposal"))
def test_actual_numerical_failure_prefixes(tmp_path, fitted, monkeypatch, stage):
    original_jit = jax.jit
    scale = toy_scale(fitted["train"])
    if stage == "weighting":
        scale *= 1e-200
    else:

        def inject_failure(function, *args, **kwargs):
            compiled = original_jit(function, *args, **kwargs)
            if function.__name__ == "evaluate" and stage == "initial_development":
                return lambda params: (np.inf, np.inf)
            if function.__name__ == "update" and stage == "proposal":

                def failed_proposal(*arguments):
                    proposal, *other = compiled(*arguments)
                    proposal = jax.tree.map(lambda value: value * np.nan, proposal)
                    return (proposal, *other)

                return failed_proposal
            return compiled

        monkeypatch.setattr(jax, "jit", inject_failure)
    with pytest.raises(ValueError, match="nonfinite"):
        with checkpoints.capture_fit(
            "candidate",
            PROVENANCE,
            source_sha256=source_map(),
            expected_steps=(0, 1, 2),
        ) as captured:
            model.fit_candidate_sequence(
                fitted["train"].batch,
                fitted["dev"].batch,
                **SETTINGS,
                error_scale=scale,
            )
    monkeypatch.undo()
    path = tmp_path / "failed"
    kwargs = failed_kwargs(path, fitted, captured, TOY_RECIPE, "fit_failure")
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    assert replay["checkpoints"] == int(stage == "proposal")
    assert replay["safeguard"]["attempts"] == int(stage == "proposal")
    assert replay["safeguard"]["completed_attempts"] == 0
    if stage == "proposal":
        row = captured.safeguard_observation["attempts"][0]
        assert not row["proposed_parameters_finite"]
        assert row["accepted_scale"] is None and not row["trials"]
        transcript = json.loads((path / "attempts.json").read_text())
        transcript["attempts"][0]["post_first_moment_fingerprint"] = "f" * 64
        (path / "attempts.json").write_text(json.dumps(transcript))
        with pytest.raises(checkpoints.CheckpointError, match="incomplete attempt"):
            checkpoints.check_checkpoint_links(
                path, **{**kwargs, "expected_sha256": reseal(path)}
            )


@pytest.mark.parametrize("stage", ("before_proposal", "after_first_trial"))
def test_incomplete_actual_attempt_prefix_is_replayed_without_fabricated_fields(
    tmp_path, fitted, monkeypatch, stage
):
    original_jit = jax.jit
    settings = {**SETTINGS, "learning_rate": 1e200}
    recipe = {**TOY_RECIPE, "learning_rate": 1e200}
    calls = []

    def interrupt(function, *args, **kwargs):
        compiled = original_jit(function, *args, **kwargs)
        if function.__name__ == "update" and stage == "before_proposal":

            def stopped(*args):
                raise model.CandidateFitError("injected interruption before proposal")

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
    with pytest.raises(ValueError, match="injected interruption"):
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
    replay = checkpoints.replay_checkpoints(path, **kwargs)
    chain = replay["safeguard"]
    assert chain["attempts"] == 1 and chain["completed_attempts"] == 0
    assert chain["trial_objective_calls"] == int(stage == "after_first_trial")
    assert chain["full_training_losses_replayed"] == 1 + int(
        stage == "after_first_trial"
    )
    row = captured.safeguard_observation["attempts"][0]
    assert row["accepted_scale"] is None and row["post_parameters_fingerprint"] is None
    assert (row["proposal_index"] is None) == (stage == "before_proposal")


def test_timeout_after_final_diagnostic_preserves_full_failed_prefix(
    tmp_path, fitted, monkeypatch
):
    class Clock:
        calls = 0

        @classmethod
        def perf_counter(cls):
            cls.calls += 1
            return 0.0 if cls.calls <= 5 else 7201.0

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
                **SETTINGS,
                error_scale=toy_scale(fitted["train"]),
            )
    monkeypatch.undo()
    assert len(captured.snapshots) == 3
    assert all(row["completed"] for row in captured.safeguard_observation["attempts"])
    path = tmp_path / "timeout"
    kwargs = failed_kwargs(path, fitted, captured, TOY_RECIPE, "fit_failure")
    result = checkpoints.replay_checkpoints(path, **kwargs)
    assert result["safeguard"]["completed_attempts"] == 2
    assert result["safeguard"]["checkpoint_chain_bound"] == 3
