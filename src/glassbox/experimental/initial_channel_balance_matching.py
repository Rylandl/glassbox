"""Saved same-data and fixed-objective checks without fitting or prediction.

Actual initial parameters and the training forecast are verified against the
prospective weight witness and imported anchored step zero by the observer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .._learner_arrays import array_fingerprint
from . import affine_anchored_matching as inherited
from . import initial_channel_balance_model as balanced
from .state_input_model import _window_fingerprint

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/initial-channel-balance-v1.json"
PROTOCOL_SHA256 = "822a28d5a20596026851ec9e5f64f9c304e09a68b67045b44bc3ee3633d056dc"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protocol():
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen initial-channel-balance protocol changed")
    return json.loads(PROTOCOL.read_text())


def _hex_digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _objective(model, recipe):
    optimization = model.report["optimization"]
    value = optimization["objective"]
    keys = {
        "id",
        "reduction",
        "weighting_data",
        "initial_channel_mse",
        "raw_channel_weights",
        "channel_weights",
        "weight_floor",
        "weight_normalizer",
        "floor_active_channels",
        "initial_parameters_and_norms_fingerprint",
        "initial_training_prediction_fingerprint",
        "fixed_during_training_and_selection",
        "original_hold_scale_preserved",
        "additional_training_weight_forecasts",
    }
    if set(value) != keys or any(
        value[key] != expected
        for key, expected in (
            ("id", balanced.OBJECTIVE),
            ("reduction", "numpy_float64_mean_over_training_windows_and_horizon"),
            ("weighting_data", "initial_recursive_training_predictions_only"),
            ("weight_floor", recipe["hold_scale_floor"] ** 2),
            ("fixed_during_training_and_selection", True),
            ("original_hold_scale_preserved", True),
            ("additional_training_weight_forecasts", 1),
        )
    ):
        raise ValueError("candidate fixed objective identity or floor differs")
    dimension = model._train.batch.future_states.shape[-1]
    mse, raw, weights = (
        np.asarray(value[key], dtype=np.float64)
        for key in ("initial_channel_mse", "raw_channel_weights", "channel_weights")
    )
    if any(
        v.shape != (dimension,) or not np.isfinite(v).all() for v in (mse, raw, weights)
    ) or np.any(mse < 0):
        raise ValueError("candidate initial residual/weight arrays are malformed")
    expected_raw = 1 / np.maximum(mse, recipe["hold_scale_floor"] ** 2)
    normalizer = np.mean(expected_raw)
    expected_weights = expected_raw / normalizer
    if (
        not np.array_equal(raw, expected_raw)
        or value["weight_normalizer"] != normalizer
        or not np.array_equal(weights, expected_weights)
        or np.any(weights <= 0)
        or value["floor_active_channels"]
        != np.flatnonzero(mse < value["weight_floor"]).tolist()
    ):
        raise ValueError("candidate fixed channel weight arithmetic differs")
    for key in (
        "initial_parameters_and_norms_fingerprint",
        "initial_training_prediction_fingerprint",
    ):
        if not _hex_digest(value[key]):
            raise ValueError("candidate initial weighting identity is malformed")
    old_trace = optimization["unweighted_development_trace"]
    expected_steps = [row["step"] for row in optimization["trace"]]
    if [row["step"] for row in old_trace] != expected_steps or any(
        set(row) != {"step", "validation_rollout_mse"}
        or not np.isfinite(row["validation_rollout_mse"])
        or row["validation_rollout_mse"] < 0
        for row in old_trace
    ):
        raise ValueError("candidate unweighted diagnostic trace differs")
    if optimization["selection_objective"] != balanced.OBJECTIVE:
        raise ValueError("candidate checkpoint objective differs")
    initial_selected = optimization["selected_step"] == 0
    if (
        initial_selected
        and array_fingerprint({}, model._model.arrays())
        != value["initial_parameters_and_norms_fingerprint"]
    ):
        raise ValueError(
            "selected initial parameters differ from reported initialization"
        )
    return dict(
        **value,
        weights_fingerprint=array_fingerprint({}, {"channel_weights": weights}),
        selected_initial_parameters_checked=initial_selected,
        actual_initial_arrays_and_prediction_verified_separately_by_observer=True,
    )


def check_matched_models(public, candidate, quadratic, anchored):
    """Check all imported revisions and the saved candidate without any rollout."""
    if (
        type(candidate) is not balanced.CandidateDynamics
        or type(candidate._model) is not balanced.BalancedQuadraticSequenceModel
    ):
        raise TypeError("candidate must be the frozen initial-channel-balance class")
    p = _protocol()
    reference = inherited.check_matched_models(public, anchored, quadratic)
    population = reference["population"]
    recipe = p["fitting"]["generic_recipe"]
    if (
        reference["fitting_recipe"] != recipe
        or reference["roles"] != p["planned_automatic_roles"].get(population)
        or quadratic.report["provenance"]["data_seal"]
        != p["imported_calibration"]["excitation_seal_sha256"].get(population)
    ):
        raise ValueError("candidate imported roles, recipe or data seal differs")
    references = dict(baseline=public, quadratic=quadratic, anchored=anchored)
    for arm, model in references.items():
        if (
            model.fingerprint()
            != p["trusted_comparator_revisions"][population][arm][
                "revision_fingerprint"
            ]
        ):
            raise ValueError("imported reference differs from trusted revision: " + arm)
    sources = dict(reference["source_sha256"])
    for name in (
        p["prior_anchored_protocol"]["path"],
        "src/glassbox/experimental/affine_anchored_model.py",
        "src/glassbox/experimental/affine_anchored_matching.py",
    ):
        expected = p["inherited_source_sha256"][name]
        if _digest(ROOT / name) != expected:
            raise ValueError(
                "inherited matching or initialization source changed: " + name
            )
        sources[name] = expected
    if (
        p["prior_anchored_protocol"]["sha256"]
        != sources[p["prior_anchored_protocol"]["path"]]
    ):
        raise ValueError("prior anchored protocol identity differs")
    expected_recipe = dict(
        anchored.recipe, id=balanced.EXPERIMENT, objective=balanced.OBJECTIVE
    )
    report = candidate.report
    if candidate.recipe != expected_recipe or report["recipe"] != expected_recipe:
        raise ValueError("candidate recipe differs from fixed objective intervention")
    if report.get("previous_revision") is not None:
        raise ValueError("candidate must be an initial fit")
    if candidate.contract != public.contract:
        raise ValueError("candidate consumer contract differs")
    for attribute, role in (("_train", "training"), ("_development", "development")):
        windows = getattr(candidate, attribute)
        if _window_fingerprint(windows) != reference["window_fingerprints"][role]:
            raise ValueError("candidate " + role + " cache differs")
        if report[role] != windows.coverage():
            raise ValueError("candidate cache coverage differs")
    if array_fingerprint({}, candidate._model.norms) != array_fingerprint(
        {}, anchored._model.norms
    ):
        raise ValueError("candidate base/product normalizations differ")
    if {k: v.shape for k, v in candidate._model.params.items()} != {
        k: v.shape for k, v in anchored._model.params.items()
    }:
        raise ValueError("candidate parameter roster or shape differs")
    counts = {
        arm: sum(v.size for v in model._model.params.values())
        for arm, model in {**references, "candidate": candidate}.items()
    }
    if counts != p["fitting"]["mechanism"]["counts"][population]:
        raise ValueError("candidate or reference parameter count differs")
    for key in ("history_steps", "horizon_steps", "delay_steps"):
        if report[key] != anchored.report[key]:
            raise ValueError("candidate timing report differs")
    if (
        candidate._model.kind != anchored._model.kind
        or candidate._model.dt_s != anchored._model.dt_s
        or candidate.history_steps != anchored.history_steps
        or candidate.horizon_steps != anchored.horizon_steps
        or candidate._model.delay_steps != anchored._model.delay_steps
    ):
        raise ValueError("candidate timing or model kind differs")
    optimization = report["optimization"]
    if (
        array_fingerprint({}, {"error_scale": np.asarray(optimization["error_scale"])})
        != reference["loss_scale_fingerprint"]
    ):
        raise ValueError("candidate original hold scale differs")
    if (
        optimization["mechanism"] != balanced.EXPERIMENT
        or optimization["minibatch_seed"] != recipe["seed"] + 10000
    ):
        raise ValueError("candidate mechanism or minibatch seed differs")
    budget = inherited.shared._check_optimization(
        candidate, recipe, counts["candidate"], "balanced"
    )
    if report["matching"] != anchored.report["matching"]:
        raise ValueError(
            "candidate matching report differs from common reference data/budget"
        )
    objective = _objective(candidate, recipe)
    return dict(
        experiment=p["id"],
        protocol_sha256=PROTOCOL_SHA256,
        population=population,
        reference_fingerprints={
            arm: model.fingerprint() for arm, model in references.items()
        },
        candidate_fingerprint=candidate.fingerprint(),
        roles=reference["roles"],
        window_counts=reference["window_counts"],
        window_fingerprints=reference["window_fingerprints"],
        recording_content_fingerprint=reference["recording_content_fingerprint"],
        normalizations_fingerprint=array_fingerprint({}, candidate._model.norms),
        original_loss_scale_fingerprint=reference["loss_scale_fingerprint"],
        shared_initialization_fingerprint=reference[
            "shared_initialization_fingerprint"
        ],
        contract=reference["contract"],
        fitting_recipe=recipe,
        optimizations={**reference["optimizations"], "candidate": budget},
        objective=objective,
        source_sha256=sources,
        objective_source=dict(
            path="src/glassbox/experimental/initial_channel_balance_model.py",
            sha256=_digest(
                ROOT / "src/glassbox/experimental/initial_channel_balance_model.py"
            ),
            forecast_function="initial_training_forecast",
            fit_function="fit_candidate_sequence",
        ),
        exact_same_data=True,
        exact_same_original_hold_scale=True,
        exact_same_numeric_objective=False,
        equal_flops_claimed=False,
        equal_end_to_end_compute_claimed=False,
        verification_scope="Imported trusted revisions, exact caches/norms/original scales/shapes, fixed weight arithmetic and declared optimizer/selection metadata. No fit, initializer, solve or rollout. Actual initial arrays/forecast and weighted trajectory are checked by the separate prospective observer/replay.",
    )
