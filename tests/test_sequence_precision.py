"""Inference follows ambient precision without retaining converted model arrays."""

import jax
import numpy as np
import pytest

from glassbox._sequence_model import KIND, SequenceModel
from glassbox.experimental.public_v4_numerics import (
    FIXTURES,
    INPUTS,
    NORMS,
    PARAMETERS,
    TOLERANCES,
    artificial_fixture,
    element_check,
)


def fixture(spec):
    meta, arrays = artificial_fixture(spec)
    model = SequenceModel(
        KIND,
        meta["dt_s"],
        meta["history_steps"],
        {k: arrays["param_" + k] for k in PARAMETERS},
        {k: arrays["norm_" + k] for k in NORMS},
        meta["delay_steps"],
    )
    return model, tuple(arrays[k] for k in INPUTS)


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s[0])
@pytest.mark.parametrize("order", [(False, True, False), (True, False, True)])
@pytest.mark.parametrize("wrappers", ["shared", "separate"])
def test_same_model_survives_compiled_precision_transitions(spec, order, wrappers):
    model, values = fixture(spec)
    fingerprint = model.fingerprint()
    owned = tuple((*model.params.values(), *model.norms.values()))
    writable = tuple(a.flags.writeable for a in owned)
    ambient = bool(jax.config.x64_enabled)
    predict, memory = jax.jit(model.rollout), jax.jit(model.memory_state)
    separate = {
        enabled: (jax.jit(model.rollout), jax.jit(model.memory_state))
        for enabled in (False, True)
    }
    repeated = {}
    for enabled in order:
        with jax.enable_x64(enabled):
            if wrappers == "separate":
                predict, memory = separate[enabled]
            y, h = np.asarray(predict(*values)), np.asarray(memory(*values[:2]))
            dtype = np.dtype("float64" if enabled else "float32")
            assert y.dtype == h.dtype == dtype
            tolerance = TOLERANCES["forecast64" if enabled else "paths32"]
            assert element_check(y, model.rollout(*values), tolerance)["passed"]
            assert element_check(h, model.memory_state(*values[:2]), tolerance)[
                "passed"
            ]
            if enabled in repeated:
                for actual, previous in zip((y, h), repeated[enabled], strict=True):
                    np.testing.assert_array_equal(actual, previous)
            repeated[enabled] = y, h
        assert bool(jax.config.x64_enabled) == ambient
    assert model.fingerprint() == fingerprint
    for actual, before, flag in zip(
        (*model.params.values(), *model.norms.values()), owned, writable, strict=True
    ):
        assert actual is before
        assert isinstance(actual, np.ndarray) and actual.dtype == np.float64
        assert actual.flags.writeable == flag


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s[0])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kind", ["float32", "integer", "mixed"])
def test_input_dtype_does_not_quantize_model_or_override_ambient(spec, enabled, kind):
    model, values = fixture(spec)
    if kind == "float32":
        supplied = tuple(v.astype(np.float32) for v in values)
    elif kind == "integer":
        supplied = tuple(np.rint(v).astype(np.int32) for v in values)
    else:
        supplied = (
            values[0].astype(np.float32),
            np.rint(values[1]).astype(np.int32),
            values[2],
        )
    with jax.enable_x64(enabled):
        dtype = np.dtype("float64" if enabled else "float32")
        # Exact values with a floating representation: integer inputs must not
        # cause the learned coefficients themselves to become integers.
        floating = tuple(v.astype(dtype) for v in supplied)
        y = np.asarray(jax.jit(model.rollout)(*supplied))
        h = np.asarray(jax.jit(model.memory_state)(*supplied[:2]))
        assert y.dtype == h.dtype == dtype
        tolerance = TOLERANCES["forecast64" if enabled else "paths32"]
        assert element_check(y, model.rollout(*floating), tolerance)["passed"]
        assert element_check(h, model.memory_state(*floating[:2]), tolerance)["passed"]
        assert np.any(y != np.rint(y))


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s[0])
@pytest.mark.parametrize("order", [(False, True, False), (True, False, True)])
def test_memory_carry_survives_precision_transitions(spec, order):
    model, values = fixture(spec)
    cut = max(model.delay_steps + 1, model.history_steps // 2)
    start = cut - model.delay_steps

    def carried(x, up, uf):
        carry = model.memory_state(x[:, : cut + 1], up[:, :cut])
        memory = model.memory_state(x[:, start:], up[:, start:], memory=carry)
        forecast = model.rollout(x[:, start:], up[:, start:], uf, memory=carry)
        return memory, forecast

    compiled = jax.jit(carried)
    for enabled in order:
        with jax.enable_x64(enabled):
            memory, forecast = map(np.asarray, compiled(*values))
            tolerance = TOLERANCES["forecast64" if enabled else "paths32"]
            assert element_check(memory, model.memory_state(*values[:2]), tolerance)[
                "passed"
            ]
            assert element_check(forecast, model.rollout(*values), tolerance)["passed"]
