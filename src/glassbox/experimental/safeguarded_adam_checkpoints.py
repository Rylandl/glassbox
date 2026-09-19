"""Read-only attempt evidence around the pinned weighted checkpoint observer.

Replay interpolates actual saved proposals and evaluates saved training windows.
It never computes gradients, Adam updates, initializers or ridge solves.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox.experimental import initial_channel_balance_checkpoints as inherited

ROOT = inherited.ROOT
CheckpointError = inherited.CheckpointError
FORMAT = "glassbox-safeguarded-adam-checkpoints-v1"
SCALES = tuple(2.0**-index for index in range(8))
PROTOCOL_SHA256 = "40252fccb98101143c45775615a3ef33c73018c374daa48470f19154cf4cd1c4"
_sha, _read, _write, _exact, _copy = (
    inherited._sha,
    inherited._read,
    inherited._write,
    inherited._exact,
    inherited._copy,
)


def _model():
    return importlib.import_module("glassbox.experimental.safeguarded_adam_model")


@lru_cache(maxsize=1)
def _weighted():
    name = f"{__package__}._safeguarded_weighted_observer"
    spec = importlib.util.spec_from_file_location(name, inherited.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._model = _model
    module.FORMAT = "glassbox-safeguarded-adam-weighted-checkpoints-v1"
    return module


def _sources(values):
    required = {
        "docs/harness/safeguarded-adam-v1.json",
        "src/glassbox/experimental/safeguarded_adam_model.py",
        "src/glassbox/experimental/safeguarded_adam_checkpoints.py",
        "tests/test_safeguarded_adam_model.py",
        "tests/test_safeguarded_adam_checkpoints.py",
    }
    if not isinstance(values, dict) or not required <= values.keys():
        raise CheckpointError("safeguard source inventory incomplete")
    for name in required:
        if _sha(ROOT / name) != values[name]:
            raise CheckpointError("safeguard source inventory differs")
    if values["docs/harness/safeguarded-adam-v1.json"] != PROTOCOL_SHA256:
        raise CheckpointError("safeguard frozen protocol changed")


def _binding():
    module = _model()
    return {
        name: inherited._binding(getattr(module, name))[1]
        for name in (
            "fit_candidate_sequence",
            "_observe_attempt",
            "make_training_objective",
            "trial_parameters",
            "safeguard_metadata",
        )
    }


def _fingerprint(tree):
    return array_fingerprint(
        {}, {name: np.asarray(value) for name, value in tree.items()}
    )


def _finite(tree):
    return all(np.isfinite(np.asarray(value)).all() for value in tree.values())


def _scalar(value):
    value = float(value)
    status = (
        "finite"
        if np.isfinite(value)
        else "nan"
        if np.isnan(value)
        else "positive_infinity"
        if value > 0
        else "negative_infinity"
    )
    return dict(status=status, value=value if status == "finite" else None)


def _value(scalar):
    if not isinstance(scalar, dict) or set(scalar) != {"status", "value"}:
        raise CheckpointError("malformed actual scalar observation")
    status, value = scalar["status"], scalar["value"]
    if status == "finite":
        if value is None or not np.isfinite(value) or value < 0:
            raise CheckpointError("invalid finite loss/norm observation")
        return float(value)
    if (
        status not in ("nan", "positive_infinity", "negative_infinity")
        or value is not None
    ):
        raise CheckpointError("nonfinite observation lacks explicit status")
    return None


def _record(evidence, state):
    phase = state["phase"]
    if phase == "initial_objective":
        if evidence["initial"] is not None or evidence["attempts"]:
            raise CheckpointError("duplicate or late initial acceptance objective")
        evidence["initial"] = dict(
            loss=_scalar(state["loss"]),
            parameters_fingerprint=_fingerprint(state["params"]),
        )
        return
    attempt = state["attempt"]
    rows = evidence["attempts"]
    if phase == "started":
        if attempt != len(rows) + 1 or (rows and not rows[-1]["completed"]):
            raise CheckpointError("attempt start is not an ordered prefix")
        rows.append(
            dict(
                attempt=attempt,
                minibatch_indices=np.asarray(state["indices"]).tolist(),
                pre_parameters_fingerprint=_fingerprint(state["params"]),
                previous_first_moment_fingerprint=_fingerprint(state["first"]),
                previous_second_moment_fingerprint=_fingerprint(state["second"]),
                current_full_training_loss=_scalar(state["current_full_training_loss"]),
                proposal_index=None,
                minibatch_loss=None,
                gradient_norm=None,
                clipped_gradient_fingerprint=None,
                gradient_finite=None,
                clipped_gradient_finite=None,
                proposed_parameters_finite=None,
                new_moments_finite=None,
                new_first_moment_fingerprint=None,
                new_second_moment_fingerprint=None,
                trials=[],
                accepted_scale=None,
                accepted_trial_index_or_null=None,
                post_parameters_fingerprint=None,
                post_first_moment_fingerprint=None,
                post_second_moment_fingerprint=None,
                cached_post_full_training_loss=None,
                completed=False,
            )
        )
        return
    if not rows or rows[-1]["attempt"] != attempt or rows[-1]["completed"]:
        raise CheckpointError("attempt observation is outside current prefix")
    row = rows[-1]
    if phase == "proposed":
        if row["proposal_index"] is not None or row["trials"]:
            raise CheckpointError("duplicate actual proposal")
        row.update(
            proposal_index=len(evidence["proposals"]),
            minibatch_loss=_scalar(state["minibatch_loss"]),
            gradient_norm=_scalar(state["gradient_norm"]),
            clipped_gradient_fingerprint=_fingerprint(state["clipped_gradient"]),
            gradient_finite=bool(state["gradient_finite"]),
            clipped_gradient_finite=_finite(state["clipped_gradient"]),
            proposed_parameters_finite=_finite(state["proposal"]),
            new_moments_finite=_finite(state["new_first"])
            and _finite(state["new_second"]),
            new_first_moment_fingerprint=_fingerprint(state["new_first"]),
            new_second_moment_fingerprint=_fingerprint(state["new_second"]),
        )
        evidence["proposals"].append(
            {key: _copy(value) for key, value in state["proposal"].items()}
        )
    elif phase == "trial":
        if row["proposal_index"] is None or state["trial_index"] != len(row["trials"]):
            raise CheckpointError("trial is not an actual ordered proposal prefix")
        row["trials"].append(
            dict(
                index=state["trial_index"],
                scale=float(state["scale"]),
                loss=_scalar(state["loss"]),
                parameters_fingerprint=_fingerprint(state["parameters"]),
            )
        )
    elif phase == "completed":
        row.update(
            accepted_scale=float(state["accepted_scale"]),
            accepted_trial_index_or_null=state["accepted_trial_index"],
            post_parameters_fingerprint=_fingerprint(state["params"]),
            post_first_moment_fingerprint=_fingerprint(state["first"]),
            post_second_moment_fingerprint=_fingerprint(state["second"]),
            cached_post_full_training_loss=_scalar(state["loss"]),
            completed=True,
        )
    else:
        raise CheckpointError("unknown safeguard observation phase")


@contextmanager
def capture_fit(
    arm, provenance, *, source_sha256, expected_steps=tuple(range(0, 1001, 100))
):
    _sources(source_sha256)
    module = _model()
    state = dict(
        binding=_binding(),
        initial=None,
        attempts=[],
        proposals=[],
        observation_seconds=0.0,
    )
    with _weighted().capture_fit(
        arm, provenance, source_sha256=source_sha256, expected_steps=expected_steps
    ) as captured:
        captured.safeguard_observation = state
        previous = sys.gettrace()

        def dispatch(frame, event, argument):
            downstream = previous(frame, event, argument) if previous else None
            if event == "call" and frame.f_code is module._observe_attempt.__code__:
                if (
                    frame.f_back is None
                    or frame.f_back.f_code is not module.fit_candidate_sequence.__code__
                ):
                    raise CheckpointError(
                        "attempt marker was not called by the actual fitter"
                    )
                started = time.perf_counter()
                _record(state, frame.f_locals["state"])
                state["observation_seconds"] += time.perf_counter() - started
            return downstream

        try:
            sys.settrace(dispatch)
            yield captured
            if state["initial"] is None or any(
                not row["completed"] for row in state["attempts"]
            ):
                raise CheckpointError("completed fit has incomplete attempt evidence")
        finally:
            sys.settrace(previous)


def _imported_weights(reference, actual):
    # The reference is the balanced outer witness, not an inferred neighboring file.
    snapshot = Path(reference["path"])
    outer_path = snapshot.parents[2]
    if str(snapshot.relative_to(outer_path)) != "anchored/trajectory/step-0000.npz":
        raise CheckpointError("balanced initial witness has unexpected nested ancestry")
    outer = outer_path / "manifest.json"
    if (
        outer.is_symlink()
        or _sha(outer) != reference["source_checkpoint_manifest_sha256"]
    ):
        raise CheckpointError("balanced outer initial witness anchor differs")
    manifest = _read(outer)
    if (
        manifest["files"].get("anchored/trajectory/step-0000.npz")
        != reference["sha256"]
    ):
        raise CheckpointError("balanced initial snapshot is outside outer manifest")
    path = outer_path / "objective-witness.npz"
    if path.is_symlink() or _sha(path) != manifest["files"].get(
        "objective-witness.npz"
    ):
        raise CheckpointError("balanced objective witness anchor differs")
    metadata, arrays = load_arrays(path)
    if not metadata["witness"]["available"]:
        raise CheckpointError("trusted balanced weighting is unavailable")
    if actual:
        if set(arrays) != set(actual):
            raise CheckpointError(
                "candidate and balanced objective witness array rosters differ"
            )
        for key, value in arrays.items():
            _exact(
                actual[key],
                value,
                "candidate initial weighting differs from balanced reference: " + key,
            )
    return dict(
        source_checkpoint_manifest_sha256=reference[
            "source_checkpoint_manifest_sha256"
        ],
        objective_witness_sha256=_sha(path),
        snapshot_sha256=reference["sha256"],
        exact=bool(actual),
    )


def _common(manifest):
    return {
        key: manifest[key]
        for key in (
            "arm",
            "provenance",
            "source_sha256",
            "train_cache",
            "development_cache",
            "recipe",
        )
    }


def _pack(proposals):
    if not proposals:
        return {}
    keys = set(proposals[0])
    if any(set(value) != keys for value in proposals):
        raise CheckpointError("proposal leaf roster changes")
    return {
        "proposal_" + key: np.stack([value[key] for value in proposals])
        for key in sorted(keys)
    }


def _unpack(payload, count):
    if type(count) is not int or count < 0:
        raise CheckpointError("invalid proposal count")
    if not count:
        if payload:
            raise CheckpointError("unavailable proposals have fabricated arrays")
        return []
    if not payload or any(
        not name.startswith("proposal_") or value.ndim < 1 or value.shape[0] != count
        for name, value in payload.items()
    ):
        raise CheckpointError("proposal archive roster/length differs")
    return [
        {
            name.removeprefix("proposal_"): value[index]
            for name, value in payload.items()
        }
        for index in range(count)
    ]


def _summary(rows, initial_loss, final_loss, checkpoint_losses, selected_step, steps):
    accepted = [row["accepted_scale"] for row in rows]
    streak = maximum = 0
    for value in accepted:
        streak = streak + 1 if value == 0 else 0
        maximum = max(maximum, streak)
    losses = dict(checkpoint_losses)
    return dict(
        id="safeguarded-adam-v1",
        acceptance="first_finite_strict_decrease",
        moment_policy="advance_on_every_finite_proposal",
        proposal_attempts=steps,
        completed_attempts=len(rows),
        accepted_attempts=sum(value > 0 for value in accepted),
        rejected_attempts=sum(value == 0 for value in accepted),
        scales=list(SCALES),
        accepted_scale_counts=[
            dict(scale=scale, attempts=accepted.count(scale)) for scale in SCALES
        ],
        maximum_rejection_streak=maximum,
        gradient_proposal_calls=steps,
        full_training_objective_calls=1 + sum(len(row["trials"]) for row in rows),
        full_training_objective_calls_max=1 + len(SCALES) * steps,
        initial_full_training_loss=initial_loss,
        final_full_training_loss=final_loss,
        selected_full_training_loss=losses[selected_step],
        checkpoint_full_training_losses=[
            dict(step=step, full_training_loss=value)
            for step, value in checkpoint_losses
        ],
        fit_wall_time_limit_s=7200,
    )


def _chain(
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
    rng = np.random.default_rng(recipe["seed"] + 10000)
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
        indices = rng.integers(
            len(batch.past_states),
            size=min(recipe["batch_size"], len(batch.past_states)),
        )
        if row["minibatch_indices"] != indices.tolist():
            raise CheckpointError("actual minibatch indices differ from frozen draw")
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
    path = Path(path)
    if path.exists():
        raise CheckpointError("safeguard checkpoint destination already exists")
    observed = captured.safeguard_observation
    imported = _imported_weights(
        anchor_reference, captured.weight_observation.get("arrays", {})
    )
    evidence = _weighted().save_checkpoints(
        path / "weighted",
        captured,
        train=train,
        development=development,
        recipe=recipe,
        anchor_reference=anchor_reference,
        selected_step=selected_step,
        failure=failure,
        selected_model=selected_model,
    )
    inner = _read(path / "weighted/manifest.json")
    common = _common(inner)
    transcript = {key: value for key, value in observed.items() if key != "proposals"}
    transcript.update(
        format=FORMAT,
        **common,
        proposal_count=len(observed["proposals"]),
        failure=failure,
        completed=failure is None,
        selected_step=selected_step,
        balanced_initial_witness=imported,
    )
    _write(path / "attempts.json", transcript)
    save_arrays(
        path / "proposals.npz",
        dict(format=FORMAT, **common, proposal_count=len(observed["proposals"])),
        _pack(observed["proposals"]),
    )
    manifest = dict(
        format=FORMAT,
        **common,
        weighted_manifest_sha256=evidence["manifest_sha256"],
        attempt_transcript="attempts.json",
        proposals="proposals.npz",
        attempt_observation_seconds=observed["observation_seconds"],
        files={
            str(file.relative_to(path)): _sha(file)
            for file in sorted(path.rglob("*"))
            if file.is_file()
        },
    )
    _write(path / "manifest.json", manifest)
    return dict(
        evidence,
        manifest_sha256=_sha(path / "manifest.json"),
        attempts=len(observed["attempts"]),
        completed_attempts=sum(row["completed"] for row in observed["attempts"]),
        attempt_observation_seconds=observed["observation_seconds"],
    )


def _load(path, expected):
    path = Path(path)
    if _sha(path / "manifest.json") != expected:
        raise CheckpointError("safeguard checkpoint external anchor differs")
    manifest = _read(path / "manifest.json")
    if (
        manifest["format"] != FORMAT
        or manifest["attempt_transcript"] != "attempts.json"
        or manifest["proposals"] != "proposals.npz"
    ):
        raise CheckpointError("safeguard envelope contract differs")
    if any(file.is_symlink() for file in path.rglob("*")):
        raise CheckpointError("safeguard symbolic payload")
    if {
        str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()
    } != set(manifest["files"]) | {"manifest.json"}:
        raise CheckpointError("safeguard payload inventory differs")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or _sha(path / relative) != digest
        ):
            raise CheckpointError("safeguard payload changed")
    if _sha(path / "weighted/manifest.json") != manifest["weighted_manifest_sha256"]:
        raise CheckpointError("safeguard weighted manifest link differs")
    return manifest


def _check(
    path,
    *,
    train,
    development,
    provenance,
    source_sha256,
    expected_sha256,
    expectation,
    anchor_reference,
    selected_model=None,
    replay=False,
):
    _sources(source_sha256)
    path = Path(path)
    manifest = _load(path, expected_sha256)
    if "safeguard" not in expectation:
        raise CheckpointError("missing external safeguard report")
    inner = _read(path / "weighted/manifest.json")
    common = _common(inner)
    if _common(manifest) != common:
        raise CheckpointError("safeguard envelope provenance differs")
    transcript = _read(path / "attempts.json")
    metadata, arrays = load_arrays(path / "proposals.npz")
    count = transcript["proposal_count"]
    if metadata != dict(format=FORMAT, **common, proposal_count=count):
        raise CheckpointError("proposal archive provenance/count differs")
    if any(
        transcript[key] != value
        for key, value in dict(
            format=FORMAT,
            **common,
            failure=expectation["failure"],
            completed=expectation["completed"],
            selected_step=expectation["selected_step"],
        ).items()
    ):
        raise CheckpointError("attempt transcript external outcome/provenance differs")
    _, payload = load_arrays(path / "weighted/objective-witness.npz")
    if transcript["balanced_initial_witness"] != _imported_weights(
        anchor_reference, payload
    ):
        raise CheckpointError("imported initial-weight association differs")
    checker = (
        _weighted().replay_checkpoints if replay else _weighted().check_checkpoint_links
    )
    result = checker(
        path / "weighted",
        train=train,
        development=development,
        provenance=provenance,
        source_sha256=source_sha256,
        expected_sha256=manifest["weighted_manifest_sha256"],
        expectation={
            key: value for key, value in expectation.items() if key != "safeguard"
        },
        anchor_reference=anchor_reference,
        selected_model=selected_model,
    )
    anchored = _read(path / "weighted/anchored/manifest.json")
    snapshots = []
    for row in anchored["checkpoints"]:
        meta, values = load_arrays(path / "weighted/anchored" / row["snapshot"])
        snapshots.append(inherited._base._snapshot_from(meta["snapshot"], values))
    chain = _chain(
        transcript,
        _unpack(arrays, count),
        payload,
        snapshots,
        train,
        expectation["recipe"],
        completed=expectation["completed"],
        selected_step=expectation["selected_step"],
        safeguard=expectation["safeguard"],
        replay=replay,
    )
    return dict(
        result,
        manifest_sha256=expected_sha256,
        safeguard=chain,
        gradient_or_adam_reexecuted=False,
        optimizer_replay_scope="Actual proposal interpolation, full-training scalar replay and accepted checkpoint chain; moments are observed fingerprint continuity only.",
    )


def check_checkpoint_links(path, **kwargs):
    return _check(path, **kwargs, replay=False)


def replay_checkpoints(path, **kwargs):
    return _check(path, **kwargs, replay=True)
