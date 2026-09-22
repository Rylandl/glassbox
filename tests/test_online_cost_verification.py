"""Analytic and adversarial checks for the no-model saved profile audit."""

import ast
import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import verify_online_cost as audit


def components():
    p = np.array([1.0, 1.0])
    jacobian = np.array([[1.0, 2.0], [3.0, 4.0]])
    projection = jacobian @ p
    weight = np.array([0.5, 0.25])
    irls = np.array([1.0, 0.5])
    c = weight * irls * projection
    setup = dict(
        flat=np.array([1.0, 2.0]),
        prior=np.array([1.0, 3.0]),
        preconditioner=np.array([2.0, 4.0]),
        raw=np.array([0.3, 0.5]),
        weight=weight,
        root_weight=np.sqrt(weight),
        irls=irls,
        gradient=np.array([-2.0, -4.0]),
    )
    solve = dict(delta=np.array([0.25, 0.5]), finite=np.asarray(True))
    trust = dict(
        delta=np.array([0.25, 0.5]),
        linearized=np.array([1.0, 2.0]),
        radius=np.asarray(1.0),
        shrink=np.asarray(0.5),
        linear_decrease=np.asarray(2.0),
        quadratic_cost=np.asarray(1.0),
        current_loss=np.asarray(2.0),
        direction_finite=np.asarray(True),
    )
    native = {
        "params/x": np.array([1.25, 2.5]),
        "current_loss": np.asarray(2.0),
        "trial_loss": np.asarray(1.0),
        "predicted_reduction": np.asarray(1.0),
        "finite": np.asarray(True),
        "evidence/direction_finite": np.asarray(True),
        "evidence/trust_shrink": np.asarray(0.5),
        "evidence/trial_evaluations": np.asarray(1, dtype=np.int32),
        "evidence/trial_losses": np.array([1.0, np.nan, np.nan, np.nan, np.nan]),
        "evidence/trial_predicted": np.array([1.0, np.nan, np.nan, np.nan, np.nan]),
        "evidence/trial_finite": np.array([True, False, False, False, False]),
        "evidence/selected_alpha": np.asarray(1.0),
    }
    arrays = {}
    for name, values in (
        ("setup", setup),
        ("solve", solve),
        ("trust", trust),
        ("native", native),
    ):
        for key, value in values.items():
            arrays[name + "/" + key] = value.copy()
            arrays["instrumented/" + name + "/" + key] = value.copy()
    arrays.update(
        {
            "conditioning/params/x": setup["flat"].copy(),
            "conditioning/norms/scale": np.array([1.0]),
            "conditioning/finite": np.asarray(True),
            "prior": setup["prior"].copy(),
            "residual": setup["raw"].copy(),
            "linearization/raw": setup["raw"].copy(),
            "linearization/tape_hashes": np.array(["0" * 64], dtype="U64"),
            "linearization/tape_nbytes": np.array([16], dtype=np.int64),
            "cached_jvp": projection,
            "cached_vjp": jacobian.T @ c,
            "cached_curvature": jacobian.T @ c + setup["preconditioner"] * p,
        }
    )
    prepared = {
        "params/x": setup["flat"].copy(),
        "norms/scale": np.array([1.0]),
        "conditioning_finite": np.asarray(True),
        "p": p,
        "c": c,
    }
    return arrays, prepared


def test_analytic_linear_actions_and_prefixes_qualify():
    arrays, prepared = components()
    errors, report = audit.verify_components(arrays, prepared)
    assert max(errors.values()) == 0
    assert report["selected_alpha"] == 1 and report["trial_evaluations"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "cached_jvp",
        "cached_vjp",
        "cached_curvature",
        "residual",
        "prior",
        "setup/gradient",
        "instrumented/trust/current_loss",
    ],
)
def test_changed_float_component_is_retained_as_unqualified(field):
    arrays, prepared = components()
    arrays[field] = arrays[field] + 0.01
    errors, _ = audit.verify_components(arrays, prepared)
    assert max(errors.values()) > 1


@pytest.mark.parametrize(
    "left,right",
    [
        ([np.nan], [0.0]),
        ([np.inf], [-np.inf]),
        ([True], [False]),
        ([1.0000001], [1.0]),
    ],
)
def test_finite_mask_discrete_or_tolerance_mismatch_rejected(left, right):
    with pytest.raises(AssertionError):
        audit.numeric_same(np.asarray(left), np.asarray(right), "fixture")


def test_unused_trial_nan_slot_and_flag_are_qualified():
    arrays, prepared = components()
    arrays["native/evidence/trial_losses"][2] = 0
    arrays["instrumented/native/evidence/trial_losses"][2] = 0
    with pytest.raises(AssertionError):
        audit.verify_components(arrays, prepared)


def test_first_acceptable_rule_is_recomputed():
    arrays, prepared = components()
    for prefix in ("native/", "instrumented/native/"):
        arrays[prefix + "evidence/trial_evaluations"] = np.asarray(2, dtype=np.int32)
        arrays[prefix + "evidence/trial_losses"][1] = 0.5
        arrays[prefix + "evidence/trial_predicted"][1] = 0.75
        arrays[prefix + "evidence/trial_finite"][1] = True
        arrays[prefix + "evidence/selected_alpha"] = np.asarray(0.5)
    with pytest.raises(AssertionError):
        audit.verify_components(arrays, prepared)


def timing_fixture(genuine=True):
    names = audit.SCOPES + (audit.PUBLIC if genuine else ())
    times = {name: np.arange(1, 26, dtype=float) / 100 for name in names}
    if genuine:
        times["public_combined"] = times["public_observe"] + times["public_snapshot"]
    saved = {name: audit.statistics(values[4:]) for name, values in times.items()}
    return times, saved


def test_raw_timings_keep_first_warmup_and_twenty_one_samples():
    times, saved = timing_fixture()
    assert audit.verify_timings(times, saved, True) == 15 * 25
    assert saved["proposal"]["count"] == 21
    assert saved["proposal"]["median_s"] == 0.15
    assert saved["proposal"]["p95_s"] == 0.24


@pytest.mark.parametrize(
    "mutation", ["stats", "count", "nested", "negative", "missing"]
)
def test_timing_tamper_rejected(mutation):
    times, saved = timing_fixture()
    if mutation == "stats":
        saved["solve"]["median_s"] = 123.0
    elif mutation == "count":
        times["prior"] = times["prior"][:-1]
    elif mutation == "nested":
        times["public_combined"][0] += 0.1
    elif mutation == "negative":
        times["proposal"][1] = -1
    else:
        del times["residual"]
    with pytest.raises(AssertionError):
        audit.verify_timings(times, saved, True)


def test_final_probe_cannot_masquerade_as_observe_timing():
    times, saved = timing_fixture()
    with pytest.raises(AssertionError):
        audit.verify_timings(times, saved, False)


def test_import_never_loads_jax_or_glassbox(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"glassbox", "jax", "jaxlib"}:
            raise AssertionError("model runtime imported: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    path = Path(audit.__file__)
    spec = importlib.util.spec_from_file_location("cost_audit_guarded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arrays, prepared = components()
    module.verify_components(arrays, prepared)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            assert not {item.name.split(".")[0] for item in node.names} & {
                "jax",
                "jaxlib",
                "glassbox",
            }
        if isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] not in {"jax", "jaxlib", "glassbox"}


def replay_fixture():
    arrays, prepared = components()
    _, proposal = audit.verify_components(arrays, prepared)
    fixed = dict(initializers=1, ridge_solves=2)
    before = dict(
        fixed,
        observations=0,
        optimizer_steps=0,
        gradient_calls=0,
        objective_calls=0,
        accepted_proposals=0,
        cg_iterations=0,
        curvature_calls=0,
        conditioning_calls=0,
        damping=1.0,
        last_proposal=None,
    )
    after = dict(
        fixed,
        observations=1,
        optimizer_steps=1,
        gradient_calls=1,
        objective_calls=2,
        accepted_proposals=1,
        cg_iterations=16,
        curvature_calls=16,
        conditioning_calls=1,
        damping=0.5,
        last_proposal=proposal,
    )
    records = [
        dict(
            index=index,
            phase="qualification" if index == 0 else "warmup" if index < 4 else "timed",
            model="authentic-after",
            report=after.copy(),
        )
        for index in range(25)
    ]
    return (
        records,
        dict(model="authentic-before", report=before),
        dict(model="authentic-after", report=after),
    )


def test_every_replay_is_bound_to_original_report_and_counters():
    records, before, after = replay_fixture()
    assert audit.verify_replays(records, before, after) == 25
    records[19]["report"]["objective_calls"] += 1
    with pytest.raises(AssertionError):
        audit.verify_replays(records, before, after)


def test_replay_checks_include_first_and_warmup_calls():
    records, before, after = replay_fixture()
    records[0]["model"] = "different"
    with pytest.raises(AssertionError):
        audit.verify_replays(records, before, after)
    records, before, after = replay_fixture()
    records.pop(2)
    with pytest.raises(AssertionError):
        audit.verify_replays(records, before, after)


def test_prefix_source_preserves_sixteen_step_solver_and_arithmetic():
    source = """@jit
 def _proposal(x):
     gradient = x + 1
     delta, _, _, _, finite = jax.lax.fori_loop(0, 16, step, carry)
     direction_finite = finite
     return a, b, c, d, e, f
""".replace("\n ", "\n")
    expected = """@jit
 def _profile_setup_kernel(x):
     gradient = x + 1
     _profile_setup = {'flat': flat, 'prior': prior, 'preconditioner': preconditioner, 'raw': raw, 'weight': weight, 'root_weight': root_weight, 'irls': irls, 'gradient': gradient}
     return _profile_setup
""".replace("\n ", "\n")
    assert audit.prefix_ast(source, "setup") == ast.dump(ast.parse(expected))
    assert audit.prefix_ast(source, "setup") != ast.dump(
        ast.parse(expected.replace("x + 1", "x + 2"))
    )
    with pytest.raises(AssertionError):
        audit.prefix_ast(source.replace("0, 16", "0, 4"), "instrumented")


def test_all_float_failures_are_collected_without_changing_the_threshold():
    arrays, prepared = components()
    arrays["solve/delta"] += 0.01
    arrays["prior"] += 0.01
    errors, report = audit.verify_components(arrays, prepared)
    assert errors["solve"] > 1 and errors["prior"] > 1
    assert errors["instrumented_native"] == 0 and report["selected_alpha"] == 1
    with pytest.raises(AssertionError):
        audit.numeric_same(
            arrays["solve/delta"],
            arrays["instrumented/solve/delta"],
            "still strict by default",
        )


def test_collecting_float_failures_does_not_allow_nonfinite_or_discrete_mismatch():
    arrays, prepared = components()
    arrays["solve/finite"] = np.asarray(False)
    with pytest.raises(AssertionError):
        audit.verify_components(arrays, prepared)
    arrays, prepared = components()
    arrays["solve/delta"][0] = np.nan
    with pytest.raises(AssertionError):
        audit.verify_components(arrays, prepared)
