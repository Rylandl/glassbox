"""Consumer contracts for the single fixed generic learner candidate."""

import inspect
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.default_model import LearnedDynamics, fit
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
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
