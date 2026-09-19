"""Read-only input and fitting witnesses for the frozen affine-anchored ablation.

The supplied public/quadratic models are trusted reference revisions; they need
not be freshly fitted arms. Checking them never depends on another fit succeeding.
No optimizer, ridge solve, model rollout or checkpoint reconstruction occurs here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .._learner_arrays import array_fingerprint
from . import affine_anchored_model as anchored
from . import excited_public_matching as shared
from .command_excitation_fit import _training_moments
from .state_input_model import _window_fingerprint

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/affine-anchored-quadratic-v1.json"
PROTOCOL_SHA256 = "d0ec7e1af8ee5877de1a84fb49f2e484070c9d790b5860b2aa2c558b204989f9"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protocol():
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen affine-anchored ablation protocol changed")
    return json.loads(PROTOCOL.read_text())


def _source_mechanics(protocol, reference):
    prior = protocol["prior_architecture_protocol"]
    if _digest(ROOT / prior["path"]) != prior["sha256"]:
        raise ValueError("frozen prior architecture protocol changed")
    path = "src/glassbox/experimental/excited_public_matching.py"
    pins = protocol["inherited_source_sha256"]
    if _digest(ROOT / path) != pins[path]:
        raise ValueError("frozen public/quadratic matching source changed")
    verified = dict(reference["source_mechanics"]["source_sha256"])
    for name, digest in verified.items():
        if name in pins and pins[name] != digest:
            raise ValueError("inherited fitting source identity differs: " + name)
    verified[path] = pins[path]
    return verified


def check_matched_models(public_excited, anchored_candidate, quadratic_excited):
    """Return stable JSON evidence for exact shared inputs and declared mechanics.

    Final model execution and captured optimizer snapshots are replayed separately.
    This check uses cached training moments and deterministic initialization draws,
    not a newly solved initializer or an independent optimizer trajectory.
    """
    if (
        type(anchored_candidate) is not anchored.CandidateDynamics
        or type(anchored_candidate._model)
        is not anchored.AnchoredQuadraticSequenceModel
    ):
        raise TypeError("anchored arm must be the frozen anchored model class")
    protocol = _protocol()
    reference = shared.check_matched_models(public_excited, quadratic_excited)
    source_sha256 = _source_mechanics(protocol, reference)
    recipe = protocol["fitting"]["generic_recipe"]
    population = reference["population"]
    if (
        reference["fitting_recipe"] != recipe
        or reference["roles"] != protocol["planned_automatic_roles"].get(population)
        or quadratic_excited.report["provenance"]["data_seal"]
        != protocol["imported_calibration"]["excitation_seal_sha256"].get(population)
    ):
        raise ValueError("ablation recipe, roles or imported data seal differs")
    trusted = protocol["trusted_comparator_revisions"][population]
    if (
        reference["public_fingerprint"] != trusted["baseline"]["revision_fingerprint"]
        or reference["quadratic_fingerprint"]
        != trusted["quadratic"]["revision_fingerprint"]
    ):
        raise ValueError("public or quadratic reference is not the trusted revision")

    model = anchored_candidate
    report = model.report
    expected_recipe = dict(
        recipe,
        id="affine-anchored-quadratic-v1",
        interaction="current_state_input_outer",
        autonomous="current_state_upper_triangular",
        ridge_prior="existing_affine_initializer",
    )
    if model.recipe != expected_recipe or report["recipe"] != expected_recipe:
        raise ValueError("anchored recipe differs from the frozen ablation recipe")
    if report.get("previous_revision") is not None:
        raise ValueError("anchored candidate must be an initial fit, not an update")
    if model.contract != public_excited.contract:
        raise ValueError("anchored signal/timing contract differs")
    for attribute, role in (("_train", "training"), ("_development", "development")):
        windows = getattr(model, attribute)
        if _window_fingerprint(windows) != reference["window_fingerprints"][role]:
            raise ValueError(
                "anchored " + role + " cache keys, origins, dt or arrays differ"
            )
        if report[role] != windows.coverage():
            raise ValueError("anchored " + role + " coverage report differs from cache")

    train = model._train.batch
    norms, draws, delay, _ = _training_moments(train)
    expected_norms = norms
    if array_fingerprint({}, model._model.norms) != array_fingerprint(
        {}, expected_norms
    ):
        raise ValueError("anchored base or product normalization differs from training")
    initialization_fingerprint = array_fingerprint({}, draws)
    if initialization_fingerprint != reference["shared_initialization_fingerprint"]:
        raise ValueError("anchored shared initialization draws differ")
    expected_shapes = {
        key: value.shape for key, value in quadratic_excited._model.params.items()
    }
    if {
        key: value.shape for key, value in model._model.params.items()
    } != expected_shapes:
        raise ValueError("anchored architecture parameter shapes differ")
    counts = dict(
        baseline=reference["optimizations"]["public"]["parameter_count"],
        candidate=sum(int(np.prod(shape)) for shape in expected_shapes.values()),
        quadratic=reference["optimizations"]["quadratic"]["parameter_count"],
    )
    if counts != protocol["fitting"]["mechanism"]["counts"][population]:
        raise ValueError("architecture parameter counts differ from frozen ablation")
    if (
        model._model.kind != recipe["kind"]
        or model.history_steps != train.past_inputs.shape[1]
        or model.horizon_steps != train.future_states.shape[1]
        or model._model.delay_steps != delay
        or model._model.dt_s != train.dt_s
        or any(
            report[key] != expected
            for key, expected in (
                ("history_steps", model.history_steps),
                ("horizon_steps", model.horizon_steps),
                ("delay_steps", delay),
            )
        )
    ):
        raise ValueError("anchored model timing or kind differs from training cache")
    optimization = report["optimization"]
    scale_fingerprint = array_fingerprint(
        {}, {"error_scale": np.asarray(optimization["error_scale"])}
    )
    if scale_fingerprint != reference["loss_scale_fingerprint"]:
        raise ValueError("anchored numeric loss scale differs from training data")
    if (
        optimization["minibatch_seed"] != recipe["seed"] + 10000
        or optimization["mechanism"] != expected_recipe["id"]
    ):
        raise ValueError("anchored optimizer mechanism or batch seed differs")
    candidate_optimization = shared._check_optimization(
        model, recipe, counts["candidate"], "anchored"
    )
    expected_matching = dict(
        baseline_fingerprint=reference["public_fingerprint"],
        training_windows_fingerprint=reference["window_fingerprints"]["training"],
        development_windows_fingerprint=reference["window_fingerprints"]["development"],
        baseline_norms_fingerprint=reference["base_norms_fingerprint"],
        error_scale_fingerprint=reference["loss_scale_fingerprint"],
        shared_initialization_fingerprint=initialization_fingerprint,
        minibatch_seed=recipe["seed"] + 10000,
        optimizer_steps=recipe["steps"],
        batch_size=recipe["batch_size"],
        baseline_parameter_count=counts["baseline"],
        candidate_parameter_count=counts["candidate"],
        equal_compute=False,
    )
    if report["matching"] != expected_matching:
        raise ValueError(
            "anchored matching report differs from public reference or budget"
        )
    selected_initial = optimization["selected_step"] == 0
    if (
        selected_initial
        and array_fingerprint({}, {key: model._model.params[key] for key in draws})
        != initialization_fingerprint
    ):
        raise ValueError(
            "selected initial anchored shared parameters differ from RNG draws"
        )

    return dict(
        experiment=protocol["id"],
        protocol_sha256=PROTOCOL_SHA256,
        public_fingerprint=reference["public_fingerprint"],
        anchored_fingerprint=model.fingerprint(),
        quadratic_fingerprint=reference["quadratic_fingerprint"],
        population=population,
        roles=reference["roles"],
        window_counts=reference["window_counts"],
        window_fingerprints=reference["window_fingerprints"],
        recording_content_fingerprint=reference["recording_content_fingerprint"],
        base_norms_fingerprint=reference["base_norms_fingerprint"],
        interaction_scale_fingerprint=array_fingerprint(
            {}, {"interaction_scale": norms["interaction_scale"]}
        ),
        autonomous_scale_fingerprint=array_fingerprint(
            {}, {"autonomous_scale": norms["autonomous_scale"]}
        ),
        loss_scale_fingerprint=scale_fingerprint,
        shared_initialization_fingerprint=initialization_fingerprint,
        selected_initial_shared_parameters_checked=selected_initial,
        contract=reference["contract"],
        fitting_recipe=recipe,
        optimizations={
            **reference["optimizations"],
            "anchored": candidate_optimization,
        },
        source_sha256=source_sha256,
        initializer_source=dict(
            path="src/glassbox/experimental/affine_anchored_model.py",
            sha256=_digest(ROOT / "src/glassbox/experimental/affine_anchored_model.py"),
            function="initialize_candidate",
            applied_prior="existing_affine_initializer",
            actual_solve_inputs_verified_separately_by_prospective_witness=True,
        ),
        exact_same_data=True,
        exact_same_numeric_objective=True,
        equal_flops_claimed=False,
        equal_end_to_end_compute_claimed=False,
        verification_scope="Trusted public/quadratic revisions, saved caches, training moments, shared RNG draws, matching metadata, reported budgets/checkpoints and pinned fitting mechanics. No optimizer, ridge solve or rollout is repeated. Captured initialization/checkpoint arrays and model outcomes are replayed separately.",
    )
