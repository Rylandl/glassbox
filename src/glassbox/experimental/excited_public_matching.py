"""Read-only matching evidence for public and quadratic fits on excited data.

This validates saved inputs and declared fitting mechanics, not a repeated
optimizer trajectory. The caller separately anchors the quadratic revision and
its source recordings. No fit, initialization solve, or model rollout occurs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .._learner_arrays import array_fingerprint
from .._sequence_model import SequenceModel
from ..learner import LearnedDynamics, _priority, steps_for
from .command_excitation_fit import _training_moments
from .state_input_model import _window_fingerprint
from .state_quadratic_model import CandidateDynamics, QuadraticSequenceModel

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/excited-public-architecture-v1.json"
PROTOCOL_SHA256 = "6a020ae72a7bfb7a4e0453e0602f788b46d0e1aaf6319197a0743cd57c74fad4"
BASE_NORMS = (
    "state_mean",
    "state_scale",
    "input_mean",
    "input_scale",
    "feature_scale",
    "delta_scale",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protocol():
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen architecture protocol changed")
    return json.loads(PROTOCOL.read_text())


def _source_mechanics(protocol):
    inventory = protocol["sources"]
    if _digest(ROOT / inventory["path"]) != inventory["sha256"]:
        raise ValueError("frozen base source inventory changed")
    pins = {
        **json.loads((ROOT / inventory["path"]).read_text())["glassbox_source_sha256"],
        **protocol["inherited_source_sha256"],
    }
    paths = (
        "src/glassbox/__init__.py",
        "src/glassbox/learner.py",
        "src/glassbox/_sequence_model.py",
        "src/glassbox/_learner_arrays.py",
        "src/glassbox/recordings.py",
        "src/glassbox/experimental/state_input_model.py",
        "src/glassbox/experimental/state_quadratic_model.py",
        "src/glassbox/experimental/command_excitation_fit.py",
    )
    verified = {}
    for path in paths:
        actual = _digest(ROOT / path)
        if actual != pins[path]:
            raise ValueError("frozen fitting source changed: " + path)
        verified[path] = actual

    return dict(
        source_sha256=verified,
        scope="Frozen public and quadratic preparation, optimizer, checkpoint, RNG and serialization implementations; raw source pins preserve prior verified mechanics.",
    )


def _check_optimization(model, recipe, count, label):
    report = model.report
    optimization = report["optimization"]
    expected = dict(
        kind=recipe["kind"],
        steps=recipe["steps"],
        seed=recipe["seed"],
        context_steps=model.history_steps,
        delay_steps=steps_for(model._model.dt_s)["delay"],
        parameter_count=count,
        loss_scale_mode="explicit_horizon_channel",
    )
    if any(optimization.get(key) != value for key, value in expected.items()):
        raise ValueError(label + " optimizer budget or model contract differs")
    trace = optimization["trace"]
    expected_steps = [
        0,
        *range(recipe["check_every"], recipe["steps"] + 1, recipe["check_every"]),
    ]
    if [row["step"] for row in trace] != expected_steps:
        raise ValueError(label + " checkpoint schedule differs")
    losses = np.asarray([row["validation_rollout_mse"] for row in trace], dtype=float)
    batch_losses = np.asarray(
        [row["training_batch_mse"] for row in trace[1:]], dtype=float
    )
    if (
        not np.isfinite(losses).all()
        or np.any(losses < 0)
        or not np.isfinite(batch_losses).all()
        or np.any(batch_losses < 0)
    ):
        raise ValueError(label + " optimization trace is not finite nonnegative")
    best = int(np.argmin(losses))
    if (
        optimization["selected_step"] != trace[best]["step"]
        or optimization["validation_rollout_mse"] != losses[best]
    ):
        raise ValueError(label + " checkpoint selection differs from strict minimum")
    return dict(
        steps=optimization["steps"],
        selected_step=optimization["selected_step"],
        checkpoint_steps=expected_steps,
        parameter_count=count,
    )


def check_matched_models(public_excited, quadratic_excited):
    """Return deterministic exact-input and frozen-mechanics evidence, or reject."""
    if (
        type(public_excited) is not LearnedDynamics
        or type(public_excited._model) is not SequenceModel
    ):
        raise TypeError("public excited arm must be the unchanged public model class")
    if (
        type(quadratic_excited) is not CandidateDynamics
        or type(quadratic_excited._model) is not QuadraticSequenceModel
    ):
        raise TypeError(
            "quadratic excited arm must be the frozen quadratic model class"
        )
    protocol = _protocol()
    mechanics = _source_mechanics(protocol)
    recipe = protocol["fitting"]["generic_recipe"]
    quadratic_recipe = dict(
        recipe,
        id="autonomous-state-quadratic-v1",
        interaction="current_state_input_outer",
        autonomous="current_state_upper_triangular",
    )
    for model, expected in (
        (public_excited, recipe),
        (quadratic_excited, quadratic_recipe),
    ):
        if model.recipe != expected or model.report["recipe"] != expected:
            raise ValueError("model recipe differs from frozen architecture recipe")
    if public_excited.report.get("previous_revision") is not None:
        raise ValueError("public candidate must be an initial fit, not an update")
    if public_excited.contract != quadratic_excited.contract:
        raise ValueError("same-data signal/timing contracts differ")
    provenance = quadratic_excited.report["provenance"]
    seen = public_excited._seen
    if seen != provenance["recording_content_fingerprints"]:
        raise ValueError("public recording contents differ from quadratic provenance")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for value in seen.values()
    ) or len(set(seen.values())) != len(seen):
        raise ValueError("recording content ledger is invalid or contains duplicates")
    names = sorted(seen, key=_priority)
    count = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    roles = {"development": names[:count], "training": names[count:]}
    populations = [
        name
        for name, expected in protocol["planned_automatic_roles"].items()
        if expected == roles
    ]
    if len(populations) != 1 or provenance["roles"] != roles:
        raise ValueError("recording roles differ from frozen admitted calibration")
    population = populations[0]
    if (
        provenance["data_seal"]
        != protocol["imported_calibration"]["excitation_seal_sha256"][population]
    ):
        raise ValueError("quadratic provenance names a different imported data seal")
    window_fingerprints = {}
    for attribute, role, budget in (
        ("_train", "training", recipe["training_windows"]),
        ("_development", "development", recipe["development_windows"]),
    ):
        public, quadratic = (
            getattr(public_excited, attribute),
            getattr(quadratic_excited, attribute),
        )
        a, b = _window_fingerprint(public), _window_fingerprint(quadratic)
        if a != b:
            raise ValueError(role + " cache keys, origins, dt or arrays differ")
        if len(public.keys) != budget or {k.recording_id for k in public.keys} != set(
            roles[role]
        ):
            raise ValueError(role + " cache violates frozen count or parent roster")
        if a != provenance[role + "_windows_fingerprint"]:
            raise ValueError(role + " cache differs from quadratic provenance")
        for model in (public_excited, quadratic_excited):
            if model.report[role] != public.coverage():
                raise ValueError(role + " coverage report differs from cache")
        window_fingerprints[role] = a
    train = public_excited._train.batch
    norms, draws, delay, quadratic_count = _training_moments(train)
    base_norms = {key: norms[key] for key in BASE_NORMS}
    if array_fingerprint({}, public_excited._model.norms) != array_fingerprint(
        {}, base_norms
    ):
        raise ValueError("public shared normalization differs from training cache")
    if array_fingerprint({}, quadratic_excited._model.norms) != array_fingerprint(
        {}, norms
    ):
        raise ValueError("quadratic normalization differs from training cache")
    norms_fingerprint = array_fingerprint({}, norms)
    initialization_fingerprint = array_fingerprint({}, draws)
    if (
        provenance["training_norms_fingerprint"] != norms_fingerprint
        or provenance["shared_initialization_fingerprint"] != initialization_fingerprint
    ):
        raise ValueError("quadratic normalization or initialization provenance differs")
    hold = np.repeat(train.past_states[:, -1:], train.future_states.shape[1], axis=1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - train.future_states) ** 2, axis=0)),
        recipe["hold_scale_floor"] * norms["state_scale"],
    )
    scale_fingerprint = array_fingerprint({}, {"error_scale": scale})
    if provenance["error_scale_fingerprint"] != scale_fingerprint:
        raise ValueError("quadratic loss-scale provenance differs from training data")
    d, m = train.past_states.shape[-1], train.past_inputs.shape[-1]
    width, memory = recipe["width"], recipe["memory"]
    f = (delay + 1) * (d + m) + memory
    shapes = dict(
        linear=(f, d),
        bias=(d,),
        w1=(f, width),
        b1=(width,),
        w2=(width, d),
        memory=(f, memory),
        memory_bias=(memory,),
    )
    public_count = sum(int(np.prod(shape)) for shape in shapes.values())
    optimizations = {}
    for label, model, count, expected_shapes in (
        ("public", public_excited, public_count, shapes),
        (
            "quadratic",
            quadratic_excited,
            quadratic_count,
            dict(shapes, interaction=(d * m, d), autonomous=(d * (d + 1) // 2, d)),
        ),
    ):
        if {
            key: value.shape for key, value in model._model.params.items()
        } != expected_shapes:
            raise ValueError(label + " architecture parameter shapes differ")
        if not np.array_equal(
            np.asarray(model.report["optimization"]["error_scale"]), scale
        ):
            raise ValueError(label + " numeric loss scale differs from training data")
        if (
            model.history_steps != train.past_inputs.shape[1]
            or model.horizon_steps != train.future_states.shape[1]
            or model._model.delay_steps != delay
            or model._model.dt_s != train.dt_s
            or model.contract["dt_s"] != train.dt_s
        ):
            raise ValueError(label + " model timing differs from training cache")
        optimizations[label] = _check_optimization(model, recipe, count, label)
    expected_provenance = dict(
        fitting_recipe=recipe,
        optimizer_steps=recipe["steps"],
        optimizer_fit_calls=1,
        batch_size=recipe["batch_size"],
        minibatch_seed=recipe["seed"] + 10000,
        parameter_count=quadratic_count,
        ridge=recipe["ridge_fraction"]
        * len(train.past_states)
        * train.future_states.shape[1],
    )
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("quadratic fitting budget provenance differs")
    optimization = quadratic_excited.report["optimization"]
    if (
        optimization["minibatch_seed"] != recipe["seed"] + 10000
        or optimization["mechanism"] != quadratic_recipe["id"]
    ):
        raise ValueError("quadratic optimizer mechanism or batch seed differs")
    return dict(
        public_fingerprint=public_excited.fingerprint(),
        quadratic_fingerprint=quadratic_excited.fingerprint(),
        population=population,
        roles=roles,
        window_counts={
            "training": recipe["training_windows"],
            "development": recipe["development_windows"],
        },
        recording_content_fingerprint=array_fingerprint(seen, {}),
        window_fingerprints=window_fingerprints,
        base_norms_fingerprint=array_fingerprint({}, base_norms),
        loss_scale_fingerprint=scale_fingerprint,
        shared_initialization_fingerprint=initialization_fingerprint,
        exact_same_data=True,
        exact_same_numeric_objective=True,
        contract=public_excited.contract,
        fitting_recipe=recipe,
        optimizations=optimizations,
        source_mechanics=mechanics,
        equal_flops_claimed=False,
        verification_scope="Saved fit inputs, training moments, RNG draws, reported budgets/checkpoint selection and pinned source mechanics. No optimizer or ridge solve is repeated; model outcome fidelity is verified separately by saved-prediction replay and trusted comparator anchoring.",
    )
