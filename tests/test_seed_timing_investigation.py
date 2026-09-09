"""Timing observation must preserve seed arithmetic and survive early rejection."""

import gc
import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_seed_timing")


def test_materialization_preserves_dtype_shape_and_values(experiment):
    recorder = experiment.Recorder()
    value = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    actual = recorder.to_host(value, dtype=float)
    expected = np.asarray(value, dtype=float)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert set(recorder.totals()) == {"seed.materialize", "seed.float_conversion"}


def test_observer_restores_methods_and_records_exception(experiment):
    solver = SimpleNamespace(_seed_observer="previous")
    names = (
        "set_seed",
        "_reject_invalid_request",
        "_initial_latent",
        "_exogenous_forecast",
        "_cold_blocks",
        "_seed_plan",
        "_optimize_plan",
        "_solved_result",
        "_failure_result",
        "evaluate",
        "linearize",
        "finalize",
    )

    def abort(*_args, **_kwargs):
        raise ValueError("early rejection")

    for name in names:
        setattr(solver, name, abort)
    callbacks = list(gc.callbacks)
    with pytest.raises(ValueError, match="early rejection"):
        with experiment.observe_request(solver) as recorder:
            solver._seed_plan()
    assert solver._seed_observer == "previous"
    assert all(getattr(solver, name) is abort for name in names)
    assert "_seed_plan" in recorder.totals()
    assert gc.callbacks == callbacks
    assert not hasattr(solver, "reports")  # Failure timing needs no optimizer report.


def test_traced_seed_matches_untraced_and_preserves_nonfinite_rejection(experiment):
    from investigate_sqp_recovery import GaussNewtonReference

    from glassbox.control.solver import _SolveAbort

    solver = object.__new__(GaussNewtonReference)
    solver.model = SimpleNamespace(values=None)
    solver.work_estimates = None
    solver.legacy_seeding = False
    solver.warm_iterations = 2

    def terms(blocks, *_context):
        return blocks.ravel(), jnp.ones(1)

    def auxiliary(*args):
        values = terms(*args)
        return values, (values, None)

    solver.evaluate = jax.jit(terms)
    solver.linearize = jax.jit(jax.jacfwd(auxiliary, has_aux=True))
    blocks = jnp.array([[0.25, 0.75]])
    args = blocks, None, None, None, SimpleNamespace(states=None), None, None
    plain = solver._seed_plan(*args)
    expected = solver._seed_derivative
    recorder = experiment.Recorder()
    solver._seed_observer = recorder
    traced = solver._seed_plan(*args)
    for a, b in zip(plain, traced):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(
        jax.tree.leaves(expected), jax.tree.leaves(solver._seed_derivative)
    ):
        np.testing.assert_array_equal(a, b)
    assert {
        "seed.finite",
        "seed.cost",
        "seed.gradient",
        "seed.return_arrays",
    } <= recorder.totals().keys()
    # In particular observation must not resurrect an old prepared seed.
    solver.evaluate = lambda *_a: (jnp.array([jnp.nan]), jnp.ones(1))
    with pytest.raises(_SolveAbort, match="no finite SQP seed"):
        solver._seed_plan(*args)
    assert solver._seed_derivative is None
    assert solver._seed_prediction is None


def test_saved_request_replays_recorded_failed_and_successful_inputs(experiment):
    runtime = (
        Path(__file__).parents[1] / "docs/investigations/feedback-recovery-runtime"
    )
    arrays, hashes = experiment.saved_request(runtime)
    state, latent, previous, incoming, seed = arrays
    assert state.shape == (13,)
    assert latent.shape == previous.shape == (4,)
    np.testing.assert_array_equal(previous, incoming[0])
    np.testing.assert_array_equal(seed, np.concatenate((incoming[1:], incoming[-1:])))
    assert (
        hashes["seed_sha256"]
        == "aba4a40d359d46603de8992d64df9e35b2405e25b2600409c3f8d85a9bddf089"
    )


def test_observer_does_not_leave_class_method_aliases_on_instance(experiment):
    names = (
        "set_seed",
        "_reject_invalid_request",
        "_initial_latent",
        "_exogenous_forecast",
        "_cold_blocks",
        "_seed_plan",
        "_optimize_plan",
        "_solved_result",
        "_failure_result",
        "evaluate",
        "linearize",
        "finalize",
    )
    cls = type("Solver", (), dict.fromkeys(names, lambda self: "unchanged"))
    solver = cls()
    solver._seed_observer = None
    before = vars(solver).copy()
    with experiment.observe(solver, experiment.Recorder()):
        assert solver._seed_plan() == "unchanged"
    assert vars(solver) == before
    assert solver._seed_plan() == "unchanged"


def test_gc_audit_separates_collection_duration_from_caller_overlap(experiment):
    audit = importlib.import_module("audit_seed_timing")
    trace = {
        "caller_interval_s": [1.0, 2.0],
        "gc_events": [
            ["start", 0.5, 0.0, 0.0, {"generation": 1}],
            ["stop", 1.25, 0.0, 0.0, {"generation": 1}],
            ["start", 3.0, 0.0, 0.0, {"generation": 0}],
            ["stop", 4.0, 0.0, 0.0, {"generation": 0}],
        ],
    }
    result = audit.gc_summary([trace])
    assert result["collections_overlapping_caller_by_generation"] == {1: 1}
    assert result["collection_duration_s"]["max"] == 0.75
    assert result["overlap_duration_s"]["max"] == 0.25
