"""Source-bound full-cache observations around the pinned safeguard observer.

Only the embedded sampling validator is copied. Historical envelope, numerical
interpolation, weighting and diagnostic helpers retain private unchanged code.
Saved-run replay does not recompute gradients, Adam updates or initializers.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.experimental import safeguarded_adam_checkpoints as previous

inherited = previous.inherited
ROOT = previous.ROOT
CheckpointError = previous.CheckpointError
FORMAT = "glassbox-full-cache-gradient-checkpoints-v1"
PROTOCOL_SHA256 = "fe318d07998e0f05385e421a930f172453641f30e020724550d8811d7757976d"
SCALES = previous.SCALES
_sha, _read, _write = previous._sha, previous._read, previous._write
_fingerprint, _finite = previous._fingerprint, previous._finite
_scalar, _value = previous._scalar, previous._value


def _model():
    return importlib.import_module("glassbox.experimental.full_cache_gradient_model")


def _private(name, path):
    qualified = f"{__package__}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _weighted():
    module = _private("_full_cache_weighted_observer", inherited.__file__)
    module._model = _model
    module.FORMAT = "glassbox-full-cache-gradient-weighted-checkpoints-v1"
    return module


@lru_cache(maxsize=1)
def _engine():
    module = _private("_full_cache_safeguard_observer", previous.__file__)
    module._model, module._weighted = _model, _weighted
    module._sources, module._binding = _sources, _binding
    module._record, module._chain = _record, _chain
    module.FORMAT = FORMAT
    return module


def _sources(values):
    previous._sources(values)
    required = {
        "docs/harness/full-cache-gradient-v1.json",
        "src/glassbox/experimental/full_cache_gradient_model.py",
        "src/glassbox/experimental/full_cache_gradient_checkpoints.py",
        "tests/test_full_cache_gradient_model.py",
        "tests/test_full_cache_gradient_checkpoints.py",
    }
    if not required <= values.keys() or any(
        _sha(ROOT / name) != values[name] for name in required
    ):
        raise CheckpointError("full-cache source inventory differs")
    if values["docs/harness/full-cache-gradient-v1.json"] != PROTOCOL_SHA256:
        raise CheckpointError("full-cache frozen protocol changed")


def _binding():
    return {
        name: inherited._binding(getattr(_model(), name))[1]
        for name in (
            "fit_candidate_sequence",
            "_observe_attempt",
            "make_training_objective",
            "trial_parameters",
            "safeguard_metadata",
            "gradient_metadata",
        )
    }


def _record(evidence, state):
    if state["phase"] == "started":
        indices = np.asarray(state["indices"])
        if indices.ndim != 1 or indices.dtype != np.dtype(np.int64):
            raise CheckpointError("actual full-cache gradient indices are not int64")
        dtype = indices.dtype.str
    previous._record(evidence, state)
    if state["phase"] == "started":
        evidence["attempts"][-1]["gradient_indices_dtype"] = dtype


def _indices(row, windows):
    values = row["minibatch_indices"]
    if (
        row.get("gradient_indices_dtype") != np.dtype(np.int64).str
        or not isinstance(values, list)
        or any(type(value) is not int for value in values)
        or values != list(range(windows))
    ):
        raise CheckpointError(
            "actual full-cache indices/dtype differ from ordered int64 arange"
        )


def _gradient(rows, windows):
    returned = sum(row["proposal_index"] is not None for row in rows)
    return dict(
        policy="ordered_full_cache",
        loss_scope="full_training_cache_pre_proposal",
        training_windows=windows,
        index_dtype="int64",
        sampling_seed=None,
        attempts_started=len(rows),
        gradient_proposal_calls_returned=returned,
        completed_acceptance_attempts=sum(row["completed"] for row in rows),
        known_gradient_window_visits=windows * returned,
        incomplete_gradient_work_unknown=len(rows) > returned,
    )


def _same_gradient(actual, expected):
    if (
        not isinstance(actual, dict)
        or actual.keys() != expected.keys()
        or any(
            type(actual[key]) is not type(value) or actual[key] != value
            for key, value in expected.items()
        )
    ):
        raise CheckpointError(
            "full-cache gradient descriptor or actual work accounting differs"
        )


def _summary(*args):
    return dict(previous._summary(*args), id="full-cache-gradient-v1")


def _chain(evidence, proposals, payload, snapshots, train, recipe, **kwargs):
    windows = len(inherited._base._batch(train).past_states)
    if type(recipe["batch_size"]) is not int or recipe["batch_size"] != windows:
        raise CheckpointError("full-cache fitting batch size differs from actual cache")
    gradient = _gradient(evidence["attempts"], windows)
    _same_gradient(evidence.get("gradient"), gradient)
    result = _accepted_chain(
        evidence, proposals, payload, snapshots, train, recipe, **kwargs
    )
    return dict(result, gradient=gradient)


def _accepted_chain(
    evidence,
    proposals,
    payload,
    snapshots,
    train,
    recipe,
    *,
    completed,
    selected_step,
    safeguard,
    replay=False,
):
    if evidence["binding"] != _binding():
        raise CheckpointError("safeguard capture/helper code binding changed")
    rows, initial = evidence["attempts"], evidence["initial"]
    if not payload:
        if (
            initial is not None
            or rows
            or proposals
            or snapshots
            or completed
            or safeguard is not None
        ):
            raise CheckpointError("attempt evidence before actual weighting")
        return dict(
            attempts=0,
            completed_attempts=0,
            accepted_attempts=0,
            rejected_attempts=0,
            initial_objective_calls=0,
            trial_objective_calls=0,
            checkpoint_chain_bound=0,
            full_training_losses_replayed=0,
            initial_available=False,
        )
    current = {
        name[6:]: value for name, value in payload.items() if name.startswith("param_")
    }
    norms = {
        name[5:]: value for name, value in payload.items() if name.startswith("norm_")
    }
    current = jax.tree.map(jnp.asarray, current)
    by_step = {snapshot["step"]: snapshot for snapshot in snapshots}
    if len(by_step) != len(snapshots):
        raise CheckpointError("duplicate checkpoint state")
    for snapshot in snapshots[:1]:
        if snapshot["step"] != 0 or _fingerprint(snapshot["params"]) != _fingerprint(
            current
        ):
            raise CheckpointError("checkpoint zero differs from actual initial weights")
    if initial is None:
        if (
            rows
            or proposals
            or completed
            or safeguard is not None
            or len(snapshots) > 1
        ):
            raise CheckpointError(
                "missing initial acceptance loss with optimizer evidence"
            )
        return dict(
            attempts=0,
            completed_attempts=0,
            accepted_attempts=0,
            rejected_attempts=0,
            initial_objective_calls=0,
            trial_objective_calls=0,
            checkpoint_chain_bound=len(snapshots),
            full_training_losses_replayed=0,
            initial_available=False,
        )
    if set(initial) != {"loss", "parameters_fingerprint"} or initial[
        "parameters_fingerprint"
    ] != _fingerprint(current):
        raise CheckpointError(
            "initial acceptance parameters differ from weight witness"
        )
    initial_loss = _value(initial["loss"])
    objective = (
        _model().make_training_objective(
            norms,
            inherited._base._batch(train),
            payload["normalization"],
            payload["channel_weights"],
            snapshots[0]["delay_steps"],
        )
        if replay and snapshots
        else None
    )
    replayed = 0
    if replay:
        if objective is None:
            raise CheckpointError(
                "initial acceptance observation lacks initial checkpoint"
            )
        if _scalar(objective(current)) != initial["loss"]:
            raise CheckpointError("initial full-training objective replay differs")
        replayed += 1
    if initial_loss is None:
        if (
            rows
            or proposals
            or completed
            or safeguard is not None
            or len(snapshots) > 1
        ):
            raise CheckpointError("attempts entered after nonfinite initial objective")
        return dict(
            attempts=0,
            completed_attempts=0,
            accepted_attempts=0,
            rejected_attempts=0,
            initial_objective_calls=1,
            trial_objective_calls=0,
            checkpoint_chain_bound=len(snapshots),
            full_training_losses_replayed=replayed,
            initial_available=False,
        )
    current_loss = initial_loss
    batch = inherited._base._batch(train)
    moments = _fingerprint(
        {name: np.zeros_like(value) for name, value in current.items()}
    )
    first, second = moments, moments
    completed_rows, used, losses = [], [], {0: initial_loss}
    bound = int(0 in by_step)
    for index, row in enumerate(rows, 1):
        if (
            row["attempt"] != index
            or index > recipe["steps"]
            or type(row["attempt"]) is not int
            or type(row["completed"]) is not bool
        ):
            raise CheckpointError("attempt roster is not a bounded prefix")
        if not row["completed"] and index != len(rows):
            raise CheckpointError("incomplete attempt is not the final prefix")
        post_fields = (
            "accepted_scale",
            "accepted_trial_index_or_null",
            "post_parameters_fingerprint",
            "post_first_moment_fingerprint",
            "post_second_moment_fingerprint",
            "cached_post_full_training_loss",
        )
        if not row["completed"] and any(row[name] is not None for name in post_fields):
            raise CheckpointError("incomplete attempt fabricates a committed state")
        _indices(row, len(batch.past_states))
        if row["pre_parameters_fingerprint"] != _fingerprint(current):
            raise CheckpointError("actual pre-parameters break accepted state chain")
        if (
            row["previous_first_moment_fingerprint"],
            row["previous_second_moment_fingerprint"],
        ) != (first, second):
            raise CheckpointError("actual Adam moment continuity differs")
        if row["current_full_training_loss"] != _scalar(current_loss):
            raise CheckpointError("cached current acceptance loss differs")
        pi = row["proposal_index"]
        if pi is None:
            proposal_fields = (
                "minibatch_loss",
                "gradient_norm",
                "clipped_gradient_fingerprint",
                "gradient_finite",
                "clipped_gradient_finite",
                "proposed_parameters_finite",
                "new_moments_finite",
                "new_first_moment_fingerprint",
                "new_second_moment_fingerprint",
            )
            if (
                row["trials"]
                or row["completed"]
                or any(row[name] is not None for name in proposal_fields)
            ):
                raise CheckpointError("trial/completion without actual proposal")
            continue
        if type(pi) is not int or pi != len(used) or pi >= len(proposals):
            raise CheckpointError("actual proposal index roster differs")
        used.append(pi)
        proposal = proposals[pi]
        if set(proposal) != set(current) or any(
            value.shape != current[name].shape or value.dtype != current[name].dtype
            for name, value in proposal.items()
        ):
            raise CheckpointError("actual proposal tree contract differs")
        if row["proposed_parameters_finite"] != _finite(proposal):
            raise CheckpointError("actual proposal finite status differs")
        if any(
            type(row[name]) is not bool
            for name in (
                "gradient_finite",
                "clipped_gradient_finite",
                "new_moments_finite",
                "proposed_parameters_finite",
            )
        ):
            raise CheckpointError("proposal finite status is not boolean")
        gradient_loss = _value(row["minibatch_loss"])
        if gradient_loss is not None and not np.isclose(
            gradient_loss, current_loss, rtol=1e-10, atol=1e-12
        ):
            raise CheckpointError(
                "full-gradient returned loss differs from cached pre-proposal objective"
            )
        finite = all(
            (
                row["gradient_finite"],
                row["clipped_gradient_finite"],
                row["new_moments_finite"],
                row["proposed_parameters_finite"],
                _value(row["minibatch_loss"]) is not None,
                _value(row["gradient_norm"]) is not None,
            )
        )
        if not finite:
            if row["trials"] or row["completed"]:
                raise CheckpointError(
                    "nonfinite proposal was treated as a scale rejection"
                )
            continue
        accepted_index, next_params, next_loss = None, current, current_loss
        if len(row["trials"]) > len(SCALES):
            raise CheckpointError("trial evaluation budget exceeded")
        for trial_index, trial in enumerate(row["trials"]):
            if (
                accepted_index is not None
                or trial["index"] != trial_index
                or trial["scale"] != SCALES[trial_index]
            ):
                raise CheckpointError(
                    "trial scales are not first-acceptable frozen prefix"
                )
            actual = _model().trial_parameters(
                current, jax.tree.map(jnp.asarray, proposal), trial["scale"]
            )
            if trial["parameters_fingerprint"] != _fingerprint(actual):
                raise CheckpointError("actual trial interpolation differs")
            value = _value(trial["loss"])
            if replay:
                if _scalar(objective(actual)) != trial["loss"]:
                    raise CheckpointError(
                        "actual full-training trial loss replay differs"
                    )
                replayed += 1
            if value is not None and value < current_loss:
                accepted_index, next_params, next_loss = trial_index, actual, value
        if row["completed"]:
            expected_scale = 0.0 if accepted_index is None else SCALES[accepted_index]
            if accepted_index is None and len(row["trials"]) != len(SCALES):
                raise CheckpointError("complete rejection omitted trial scales")
            if (
                row["accepted_scale"] != expected_scale
                or row["accepted_trial_index_or_null"] != accepted_index
            ):
                raise CheckpointError(
                    "acceptance did not use first finite strict decrease"
                )
            if row["post_parameters_fingerprint"] != _fingerprint(next_params) or row[
                "cached_post_full_training_loss"
            ] != _scalar(next_loss):
                raise CheckpointError(
                    "accepted/rejected post-state or cached loss differs"
                )
            if (
                row["post_first_moment_fingerprint"],
                row["post_second_moment_fingerprint"],
            ) != (
                row["new_first_moment_fingerprint"],
                row["new_second_moment_fingerprint"],
            ):
                raise CheckpointError("finite proposal did not commit new moments")
            first, second = (
                row["post_first_moment_fingerprint"],
                row["post_second_moment_fingerprint"],
            )
            current, current_loss = next_params, next_loss
            losses[index] = current_loss
            completed_rows.append(row)
            if index in by_step:
                if by_step[index]["training_batch_mse"] != gradient_loss:
                    raise CheckpointError(
                        "checkpoint gradient-loss mirror differs from actual returned loss"
                    )
                if _fingerprint(by_step[index]["params"]) != _fingerprint(current):
                    raise CheckpointError(
                        "checkpoint differs from actual accepted parameter chain"
                    )
                bound += 1
        elif any(
            row[name] is not None
            for name in (
                "accepted_scale",
                "post_parameters_fingerprint",
                "cached_post_full_training_loss",
            )
        ):
            raise CheckpointError("incomplete attempt fabricates an accepted state")
    if len(used) != len(proposals) or bound != len(snapshots):
        raise CheckpointError(
            "proposal or checkpoint evidence is outside observed attempt chain"
        )
    if completed:
        if len(completed_rows) != recipe["steps"] or len(rows) != recipe["steps"]:
            raise CheckpointError("completed fit omits proposal attempts")
        expected = _summary(
            rows,
            initial_loss,
            current_loss,
            [(step, losses[step]) for step in by_step],
            selected_step,
            recipe["steps"],
        )
        if safeguard != expected:
            raise CheckpointError(
                "selected-model safeguard report differs from actual attempts"
            )
    elif safeguard is not None:
        raise CheckpointError("failed fit cannot claim a selected safeguard report")
    return dict(
        attempts=len(rows),
        completed_attempts=len(completed_rows),
        accepted_attempts=sum(row["accepted_scale"] > 0 for row in completed_rows),
        rejected_attempts=sum(row["accepted_scale"] == 0 for row in completed_rows),
        initial_objective_calls=1,
        trial_objective_calls=sum(len(row["trials"]) for row in rows),
        checkpoint_chain_bound=bound,
        full_training_losses_replayed=replayed,
        initial_available=True,
    )


def capture_fit(
    arm, provenance, *, source_sha256, expected_steps=tuple(range(0, 1001, 100))
):
    return _engine().capture_fit(
        arm, provenance, source_sha256=source_sha256, expected_steps=expected_steps
    )


def save_checkpoints(
    path,
    captured,
    *,
    train,
    development,
    recipe,
    anchor_reference,
    selected_step=None,
    failure=None,
    selected_model=None,
):
    windows = len(inherited._base._batch(train).past_states)
    rows = captured.safeguard_observation["attempts"]
    for row in rows:
        _indices(row, windows)
    gradient = _gradient(rows, windows)
    captured.safeguard_observation["gradient"] = gradient
    evidence = _engine().save_checkpoints(
        path,
        captured,
        train=train,
        development=development,
        recipe=recipe,
        anchor_reference=anchor_reference,
        selected_step=selected_step,
        failure=failure,
        selected_model=selected_model,
    )
    manifest_path = Path(path) / "manifest.json"
    manifest = _read(manifest_path)
    manifest["gradient"] = gradient
    _write(manifest_path, manifest)
    return dict(evidence, manifest_sha256=_sha(manifest_path), gradient=gradient)


def _check(path, *, expectation, **kwargs):
    if "gradient" not in expectation:
        raise CheckpointError("missing external full-cache gradient descriptor")
    result = _engine()._check(
        path,
        expectation={
            key: value for key, value in expectation.items() if key != "gradient"
        },
        **kwargs,
    )
    gradient = result["safeguard"]["gradient"]
    _same_gradient(_read(Path(path) / "manifest.json").get("gradient"), gradient)
    if expectation["completed"]:
        _same_gradient(expectation["gradient"], gradient)
    elif expectation["gradient"] is not None:
        raise CheckpointError("failed fit cannot claim a selected gradient report")
    return dict(
        result,
        gradient=gradient,
        optimizer_replay_scope="Actual ordered full-cache row witnesses, returned-loss diagnostic consistency, proposal interpolation, canonical acceptance scalar replay and accepted checkpoint chain. Observed gradient/moment fingerprints are not independent gradient/Adam recomputation.",
    )


def check_checkpoint_links(path, **kwargs):
    return _check(path, **kwargs, replay=False)


def replay_checkpoints(path, **kwargs):
    return _check(path, **kwargs, replay=True)
