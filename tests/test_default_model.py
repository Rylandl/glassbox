"""Consumer contracts for the single maintained generic learner."""

import copy
import inspect
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.arrays import array_fingerprint
from glassbox.experimental.default_model import (
    ENVELOPE_COVERAGE,
    RECIPE,
    LearnedDynamics,
    fit,
)
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)
from glassbox.experimental.sequence_model import (
    _damped,
    recursion_ceiling,
    recursion_rows,
)


def recording(name, seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(79, 1))
    x = np.zeros((80, 2))
    for i, command in enumerate(u):
        x[i + 1, 0] = 0.95 * x[i, 0] + 0.08 * np.tanh(command[0])
        x[i + 1, 1] = 0.91 * x[i, 1] + 0.06 * command[0]
    return SequenceSegment(name, "whole", x, u, 0.05)


def collection(*segments):
    return SequenceCollection(
        segments,
        configuration_id="toy-v1",
        state_channels=("x0 [unitless]", "x1 [unitless]"),
        input_channels=("u [command,unitless]",),
    )


@pytest.fixture(scope="module")
def fitted():
    return fit(collection(recording("a", 1), recording("b", 2)))


def test_fit_and_update_have_no_tuning_parameters():
    assert list(inspect.signature(fit).parameters) == ["recordings"]
    assert list(inspect.signature(LearnedDynamics.update).parameters) == [
        "self",
        "recordings",
    ]


def test_one_recursive_model_and_prefix_causality(fitted):
    b = fitted._development.batch
    p = fitted.predict(b.past_states[0], b.past_inputs[0], b.future_inputs[0])
    short = fitted.predict(b.past_states[0], b.past_inputs[0], b.future_inputs[0, :2])
    np.testing.assert_allclose(short, p[:2], rtol=1e-5, atol=1e-6)
    jac = jax.jacfwd(lambda u: fitted.predict(b.past_states[0], b.past_inputs[0], u))(
        jnp.asarray(b.future_inputs[0])
    )
    for h in range(p.shape[0]):
        np.testing.assert_array_equal(jac[h, :, h + 1 :], 0)
    np.testing.assert_allclose(
        jax.jit(fitted.predict)(b.past_states, b.past_inputs, b.future_inputs)[0],
        p,
        rtol=1e-5,
        atol=1e-6,
    )


def test_missing_history_and_unsupported_horizon_are_rejected(fitted):
    b = fitted._development.batch
    with pytest.raises(ValueError, match="aligned history"):
        fitted.predict(b.past_states[0, -1:], b.past_inputs[0, :0], b.future_inputs[0])
    with pytest.raises(ValueError, match="unsupported forecast horizon"):
        fitted.predict(
            b.past_states[0],
            b.past_inputs[0],
            np.repeat(b.future_inputs[0, :1], fitted.horizon_steps + 1, axis=0),
        )


def test_roundtrip_retains_recipe_evidence_and_cache(fitted, tmp_path):
    p = tmp_path / "model.npz"
    fitted.save(p)
    restored = LearnedDynamics.load(p)
    assert fitted.fingerprint() == restored.fingerprint()
    assert fitted.report == restored.report
    assert fitted.contract == restored.contract
    np.testing.assert_array_equal(
        fitted._train.batch.future_states, restored._train.batch.future_states
    )
    b = fitted._development.batch
    np.testing.assert_array_equal(
        fitted.predict(b.past_states, b.past_inputs, b.future_inputs),
        restored.predict(b.past_states, b.past_inputs, b.future_inputs),
    )


def test_update_preserves_old_revision_and_reserved_observations(fitted, tmp_path):
    before = fitted.fingerprint()
    path = tmp_path / "saved.npz"
    fitted.save(path)
    update = LearnedDynamics.load(path).update(collection(recording("fresh", 3)))
    assert fitted.fingerprint() == before
    assert update.report["previous_revision"] == before
    assert update.fingerprint() != before
    assert "fresh" in update.report["training"]
    assert len(update._train.keys) <= 384
    assert update._development.keys == fitted._development.keys
    np.testing.assert_array_equal(
        update._development.batch.future_states, fitted._development.batch.future_states
    )


def test_input_record_order_does_not_change_the_recipe_result(fitted):
    reordered = fit(collection(recording("b", 2), recording("a", 1)))
    assert reordered._train.keys == fitted._train.keys
    assert reordered._development.keys == fitted._development.keys
    np.testing.assert_array_equal(
        reordered._train.batch.future_states, fitted._train.batch.future_states
    )
    b = fitted._development.batch
    np.testing.assert_allclose(
        reordered.predict(b.past_states, b.past_inputs, b.future_inputs),
        fitted.predict(b.past_states, b.past_inputs, b.future_inputs),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "change", ["configuration", "units", "dt", "identity", "content"]
)
def test_update_rejects_mismatch_and_reused_evidence(fitted, change):
    c = collection(recording("fresh", 3))
    if change == "configuration":
        c = replace(c, configuration_id="different")
    if change == "units":
        c = replace(c, state_channels=("x0 [different unit]", "x1 [unitless]"))
    if change == "dt":
        c = replace(c, segments=(replace(c.segments[0], dt_s=0.1),))
    if change == "identity":
        c = collection(recording("a", 3))
    if change == "content":
        c = collection(recording("renamed", 1))
    with pytest.raises(ValueError):
        fitted.update(c)


def test_recording_facts_and_independent_roles_are_required():
    with pytest.raises(ValueError, match="configuration_id"):
        fit(SequenceCollection((recording("a", 1), recording("b", 2))))
    with pytest.raises(ValueError, match="at least two"):
        fit(collection(recording("a", 1)))
    with pytest.raises(ValueError, match="duplicate recording content"):
        fit(collection(recording("a", 1), recording("renamed", 1)))


def test_report_and_contract_are_defensive_copies(fitted):
    report = fitted.report
    report["recipe"]["steps"] = 0
    contract = fitted.contract
    contract["state_channels"][0] = "wrong"
    assert fitted.report["recipe"]["steps"] == 1000
    assert fitted.contract["state_channels"][0] == "x0 [unitless]"


def test_default_is_the_versioned_memory_recipe(fitted):
    assert fitted.recipe == RECIPE
    assert fitted.report["recipe"]["id"] == "generic-memory-v4-prototype"
    assert fitted._model.kind == "filter_mlp"
    assert (
        fitted.history_steps,
        fitted.report["delay_steps"],
        fitted.horizon_steps,
    ) == (
        10,
        2,
        5,
    )
    assert fitted._metadata()["format"] == "glassbox-default-recipe-v4"
    assert fitted._train.batch.past_states.shape[1] == 11


def test_longer_supplied_history_is_truncated_to_the_consumed_context(fitted):
    r = recording("a", 1)
    x, u = r.states, r.inputs
    longer = fitted.predict(x[:21], u[:20], u[20:25])
    exact = fitted.predict(x[10:21], u[10:20], u[20:25])
    np.testing.assert_array_equal(longer, exact)
    with pytest.raises(ValueError, match="at least 11 observations"):
        fitted.predict(x[15:21], u[15:20], u[20:25])


def test_only_the_current_format_loads(fitted, tmp_path):
    path = tmp_path / "saved.npz"
    fitted.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files if k != "metadata"}
        meta = json.loads(str(archive["metadata"]))
    meta.pop("fingerprint")
    for change in ("format", "recipe"):
        altered = copy.deepcopy(meta)
        if change == "format":
            altered["format"] = "glassbox-default-recipe-v1"
        else:
            altered["recipe"]["steps"] = 5
        # Re-sign the archive so the version gate, not the fingerprint, decides.
        altered["fingerprint"] = array_fingerprint(altered, arrays)
        other = tmp_path / f"{change}.npz"
        np.savez_compressed(other, metadata=json.dumps(altered), **arrays)
        with pytest.raises(ValueError, match="unsupported default recipe version"):
            LearnedDynamics.load(other)


def test_altered_recipes_are_rejected(fitted):
    report = fitted.report
    report["recipe"]["id"] = "generic-history-v1-prototype"
    with pytest.raises(ValueError, match="recipe version"):
        LearnedDynamics(
            fitted._model,
            fitted._train,
            fitted._development,
            fitted.contract,
            fitted._seen,
            report,
            fitted.envelope(),
        )


def test_a_v2_artifact_carrying_no_envelope_is_rejected(fitted, tmp_path):
    """The format bump is what rejects it; the missing array is the reason for it."""
    path = tmp_path / "saved.npz"
    fitted.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files if k != "metadata"}
        meta = json.loads(str(archive["metadata"]))
    meta.pop("fingerprint")
    v2 = copy.deepcopy(meta)
    v2["format"] = "glassbox-default-recipe-v2"
    v2["recipe"]["id"] = "generic-memory-v2-prototype"
    del v2["report"]["envelope"]
    without = {k: v for k, v in arrays.items() if k != "envelope_half_width"}
    v2["fingerprint"] = array_fingerprint(v2, without)
    other = tmp_path / "v2.npz"
    np.savez_compressed(other, metadata=json.dumps(v2), **without)
    with pytest.raises(ValueError, match="unsupported default recipe version"):
        LearnedDynamics.load(other)
    # And a re-signed current-format archive with the array removed is refused
    # for the reason the bump exists rather than for its version string.
    stripped = copy.deepcopy(meta)
    del stripped["report"]["envelope"]
    stripped["fingerprint"] = array_fingerprint(stripped, without)
    bare = tmp_path / "bare.npz"
    np.savez_compressed(bare, metadata=json.dumps(stripped), **without)
    with pytest.raises(ValueError, match="carries an envelope"):
        LearnedDynamics.load(bare)


def test_the_envelope_is_calibrated_on_the_held_out_development_windows(fitted):
    """Nominal coverage holds on the calibration windows by construction."""
    envelope = fitted.envelope()
    assert envelope.shape == (fitted.horizon_steps, 2)
    assert np.isfinite(envelope).all() and np.all(envelope >= 0)
    np.testing.assert_allclose(envelope, fitted.report["envelope"]["half_width"])
    assert fitted.report["envelope"]["nominal_coverage"] == ENVELOPE_COVERAGE
    assert fitted.report["envelope"]["calibrated_on"] == "development"

    batch = fitted._development.batch
    residual = np.abs(
        np.asarray(
            fitted.predict(batch.past_states, batch.past_inputs, batch.future_inputs)
        )
        - batch.future_states
    )
    covered = (residual <= envelope).mean(axis=0)
    assert covered.min() >= ENVELOPE_COVERAGE
    assert fitted.report["envelope"]["calibration_windows"] == len(residual)
    # The half-width is the conformal rank's own residual, not an interpolation.
    rank = fitted.report["envelope"]["quantile_rank"]
    np.testing.assert_array_equal(envelope, np.sort(residual, axis=0)[rank - 1])


def test_a_shorter_horizon_is_the_prefix_and_a_longer_one_is_refused(fitted):
    full = fitted.envelope()
    np.testing.assert_array_equal(fitted.envelope(2), full[:2])
    assert fitted.envelope(fitted.horizon_steps).shape == full.shape
    for bad in (0, -1, fitted.horizon_steps + 1):
        with pytest.raises(ValueError, match="unsupported forecast horizon"):
            fitted.envelope(bad)
    # It is a copy, so a caller cannot edit the model's own evidence.
    taken = fitted.envelope()
    taken[0, 0] = 1e9
    assert fitted.envelope()[0, 0] != 1e9


def test_an_update_recalibrates_on_the_pinned_development_cache(fitted):
    revised = fitted.update(collection(recording("c", 3), recording("d", 4)))
    assert (
        revised.report["envelope"]["calibration_windows"]
        == (fitted.report["envelope"]["calibration_windows"])
    )
    np.testing.assert_array_equal(
        revised._development.batch.future_states,
        fitted._development.batch.future_states,
    )
    # Same held-out windows, a different fit: a different envelope measured the
    # same way, never the predecessor's carried forward.
    assert not np.array_equal(revised.envelope(), fitted.envelope())
    batch = revised._development.batch
    residual = np.abs(
        np.asarray(
            revised.predict(batch.past_states, batch.past_inputs, batch.future_inputs)
        )
        - batch.future_states
    )
    assert (residual <= revised.envelope()).mean(axis=0).min() >= ENVELOPE_COVERAGE


def recursion_gain(model, batch, unit, seed=0):
    """The gain the bound is written against, measured outside the learner."""
    past = np.asarray(batch.past_states)
    direction = np.random.default_rng(seed).standard_normal((len(past), past.shape[-1]))
    nominal = np.asarray(model.rollout(past, batch.past_inputs, batch.future_inputs))
    moved = past.copy()
    moved[:, -1] = moved[:, -1] + direction * unit
    perturbed = np.asarray(model.rollout(moved, batch.past_inputs, batch.future_inputs))
    deviation = (perturbed - nominal) / unit
    return np.sqrt(np.mean(deviation**2, axis=(0, 2))) / np.sqrt(np.mean(direction**2))


def test_the_fit_bounds_the_recursions_gain_by_the_processes_own(fitted):
    bound = fitted.report["optimization"]["recursion_bound"]
    assert bound["applies"] is True
    unit, ceiling = recursion_ceiling(fitted._train.batch)
    # The ceiling is the process's own motion growth on the training windows,
    # floored at no amplification, and nothing else.
    assert ceiling[0] == pytest.approx(1.0)
    assert (ceiling >= 1.0).all()
    gain = recursion_gain(
        fitted._model, fitted._train.batch, unit, seed=RECIPE["seed"] + 20000
    )
    assert (gain[1:] <= ceiling[1:] + 1e-5).all()
    assert bound["selected_gain_excess"] <= 1e-5


def test_damping_the_recursion_is_what_the_bound_moves(fitted):
    # The factor scales every path from an observed channel back into the next
    # prediction and nothing else, so a model damped to zero returns an error
    # unchanged: the hold-and-shift map's gain is one at every step.
    model = fitted._model
    rows = recursion_rows(
        len(model.norms["state_mean"]),
        len(model.norms["input_mean"]),
        model.delay_steps,
        model.params["memory"].shape[1],
    )
    assert rows.sum() < rows.size
    params = jax.tree.map(np.asarray, _damped(model.params, rows, 0.0))
    frozen = replace(model, params=params)
    unit, _ = recursion_ceiling(fitted._train.batch)
    gain = recursion_gain(frozen, fitted._train.batch, unit)
    np.testing.assert_allclose(gain, np.ones_like(gain), rtol=1e-5, atol=1e-5)
    # The command rows are untouched, so the rows the factor does not scale are
    # exactly the exogenous ones plus the biases.
    for key in ("bias", "b1", "w2", "memory_bias"):
        np.testing.assert_array_equal(params[key], np.asarray(model.params[key]))
    for key in ("linear", "w1", "memory"):
        np.testing.assert_array_equal(
            params[key][~rows], np.asarray(model.params[key])[~rows]
        )


def test_the_bound_is_not_vacuous(fitted):
    # Scaling the recursion back up past what the fit settled on breaks the
    # bound, so the assertion above is a measurement and not an identity.
    model = fitted._model
    rows = recursion_rows(
        len(model.norms["state_mean"]),
        len(model.norms["input_mean"]),
        model.delay_steps,
        model.params["memory"].shape[1],
    )
    unit, ceiling = recursion_ceiling(fitted._train.batch)
    loud = replace(
        model, params=jax.tree.map(np.asarray, _damped(model.params, rows, 4.0))
    )
    gain = recursion_gain(loud, fitted._train.batch, unit)
    assert (gain[1:] > ceiling[1:]).any()
