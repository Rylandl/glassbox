"""Thin private adapter for weighted-objective witnesses and checkpoint replay.

Historical observers and initializer code are loaded unchanged. Extra evidence
is outside their bundle; replay executes predictions and reductions, never a fit
or initializer. Historical reference checkpoints remain historical artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import marshal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.experimental import affine_anchored_checkpoints as inherited

ROOT = inherited.ROOT
CheckpointError = inherited.CheckpointError
FORMAT = "glassbox-initial-channel-balance-checkpoints-v1"
_base = inherited._base
_read, _write, _exact, _copy = (
    inherited._read,
    inherited._write,
    inherited._exact,
    inherited._copy,
)
_sha = _base._sha
WEIGHT_ARRAYS = (
    "initial_training_prediction",
    "initial_channel_mse",
    "raw_channel_weights",
    "channel_weights",
    "normalization",
    "weight_floor",
    "weight_normalizer",
    "fixed_weights",
)
REFERENCE_KEYS = {
    "path",
    "sha256",
    "source_bundle_sha256",
    "source_checkpoint_manifest_sha256",
    "source_relative_path",
}


def _model():
    return importlib.import_module(
        "glassbox.experimental.initial_channel_balance_model"
    )


def _binding(function):
    path = Path(inspect.getsourcefile(function)).resolve()
    source = path.read_text()
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == function.__name__
    )
    code = next(
        item
        for item in compile(source, str(path), "exec").co_consts
        if isinstance(item, CodeType) and item.co_name == function.__name__
    )
    if function.__code__ != code:
        raise CheckpointError("weight construction function differs from source")

    def portable(value):
        return value.replace(
            co_filename=str(path.relative_to(ROOT)),
            co_consts=tuple(
                portable(item) if isinstance(item, CodeType) else item
                for item in value.co_consts
            ),
        )

    return node, dict(
        path=str(path.relative_to(ROOT)),
        function=function.__name__,
        source_sha256=_sha(path),
        code_sha256=hashlib.sha256(marshal.dumps(portable(code))).hexdigest(),
    )


def _weight_binding():
    module = _model()
    node, binding = _binding(module.fit_candidate_sequence)
    points = [
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id == "best"
        and isinstance(item.value, ast.Name)
        and item.value.id == "params"
    ]
    if len(points) != 1:
        raise CheckpointError("weight witness boundary changed")
    return dict(
        fitter=binding,
        weight_line=points[0],
        initial_training_forecast=_binding(module.initial_training_forecast)[1],
        reporting=_binding(module.weighting_metadata)[1],
    )


def _engine(weights=None):
    """Configure only fresh private observer namespaces, never fitting globals."""
    name = f"{__package__}._initial_channel_balance_anchor_observer"
    spec = importlib.util.spec_from_file_location(name, inherited.__file__)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[name] = engine
    spec.loader.exec_module(engine)
    engine.CheckpointError = engine._base.CheckpointError = CheckpointError
    engine._candidate = _model

    def arm(name):
        if name != "candidate":
            raise CheckpointError("only candidate has a current balanced fit")
        module = _model()
        return (
            module.fit_candidate_sequence,
            module.BalancedQuadraticSequenceModel,
            module._rollout,
        )

    prior_sources = engine._sources

    def sources(values):
        required = {
            f"{folder}/{name}{suffix}"
            for folder, suffix in (
                ("src/glassbox/experimental", ".py"),
                ("tests", ".py"),
            )
            for name in (
                ("initial_channel_balance_checkpoints", "initial_channel_balance_model")
                if folder != "tests"
                else (
                    "test_initial_channel_balance_checkpoints",
                    "test_initial_channel_balance_model",
                )
            )
        }
        if not isinstance(values, dict) or not required <= values.keys():
            raise CheckpointError("weighted checkpoint source inventory incomplete")
        return prior_sources(values)

    engine._arm = engine._base._arm = arm
    engine._sources = engine._base._sources = sources
    engine._base._diagnostics = lambda snapshot, data, predict: _diagnostics(
        snapshot, data, predict, weights, observed=True
    )
    return engine


def _initial_metadata(local):
    model = local["model"]
    return dict(
        kind=model.kind,
        dt_s=model.dt_s,
        history_steps=model.history_steps,
        delay_steps=model.delay_steps,
        horizon_steps=local["train"].future_states.shape[1],
        train_fingerprint=_base._batch_fingerprint(local["train"]),
        development_fingerprint=_base._batch_fingerprint(local["validation"]),
        settings={
            key: local[key]
            for key in (
                "seed",
                "steps",
                "batch_size",
                "learning_rate",
                "width",
                "memory",
                "ridge",
                "check_every",
            )
        },
    )


@contextmanager
def capture_fit(
    arm, provenance, *, source_sha256, expected_steps=tuple(range(0, 1001, 100))
):
    engine, module = _engine(), _model()
    binding = _weight_binding()
    state = dict(
        available=False,
        boundary_calls=0,
        forecast_calls=0,
        observation_seconds=0.0,
        binding=binding,
    )
    with engine.capture_fit(
        arm, provenance, source_sha256=source_sha256, expected_steps=expected_steps
    ) as captured:
        captured.weight_observation = state
        previous = sys.gettrace()

        def dispatch(frame, event, argument):
            downstream = previous(frame, event, argument) if previous else None
            if (
                event == "call"
                and frame.f_code is module.initial_training_forecast.__code__
                and frame.f_back.f_code is module.fit_candidate_sequence.__code__
            ):
                state["forecast_calls"] += 1
                if state["forecast_calls"] != 1:
                    raise CheckpointError("multiple initial training weight forecasts")
            if (
                event != "call"
                or frame.f_code is not module.fit_candidate_sequence.__code__
            ):
                return downstream

            def observe(local_frame, local_event, local_argument):
                nonlocal downstream
                started = time.perf_counter()
                if (
                    local_event == "line"
                    and local_frame.f_lineno == binding["weight_line"]
                ):
                    state["boundary_calls"] += 1
                    if state["boundary_calls"] != 1:
                        raise CheckpointError("multiple completed weighting boundaries")
                    local = local_frame.f_locals
                    state.update(
                        available=True,
                        initial=_initial_metadata(local),
                        arrays={
                            **{key: _copy(local[key]) for key in WEIGHT_ARRAYS},
                            **{
                                f"param_{key}": _copy(value)
                                for key, value in local["params"].items()
                            },
                            **{
                                f"norm_{key}": _copy(value)
                                for key, value in local["norms"].items()
                            },
                        },
                    )
                if downstream is not None:
                    downstream = downstream(local_frame, local_event, local_argument)
                if (
                    local_event == "line"
                    and captured.snapshots
                    and local_frame.f_lineno
                    in (
                        captured.binding["initial_line"],
                        captured.binding["checkpoint_line"],
                    )
                ):
                    snapshot, local = captured.snapshots[-1], local_frame.f_locals
                    _exact(
                        local["channel_weights"],
                        state["arrays"]["channel_weights"],
                        "fixed channel weights changed during optimization",
                    )
                    snapshot["observed_unweighted_loss"] = float(
                        local["unweighted_best_initial_loss"]
                        if snapshot["step"] == 0
                        else local["unweighted_val_loss"]
                    )
                    snapshot["channel_weights_fingerprint"] = inherited._fingerprint(
                        local["channel_weights"]
                    )
                state["observation_seconds"] += time.perf_counter() - started
                return observe

            return observe

        try:
            sys.settrace(dispatch)
            yield captured
            if (
                not state["available"]
                or state["forecast_calls"] != 1
                or state["boundary_calls"] != 1
            ):
                raise CheckpointError("completed fit lacks weight construction witness")
        finally:
            sys.settrace(previous)


def _reference(reference):
    if set(reference) != REFERENCE_KEYS:
        raise CheckpointError("incomplete anchored initial snapshot reference")
    path = Path(reference["path"])
    if path.is_symlink() or _sha(path) != reference["sha256"]:
        raise CheckpointError("anchored initial snapshot reference bytes changed")
    binding = {key: value for key, value in reference.items() if key != "path"}
    if any(
        not isinstance(binding[key], str) or len(binding[key]) != 64
        for key in REFERENCE_KEYS - {"path", "source_relative_path"}
    ):
        raise CheckpointError("incomplete anchored initial snapshot hashes")
    relative = Path(binding["source_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise CheckpointError("anchored initial snapshot path is not portable")
    metadata, arrays = load_arrays(path)
    snapshot = _base._snapshot_from(metadata["snapshot"], arrays)
    if snapshot["step"] != 0:
        raise CheckpointError("anchor reference is not the initial checkpoint")
    return binding, snapshot


def _objective(
    state, train, development, recipe, reference, snapshots, *, replay=False
):
    binding, anchor = _reference(reference)
    metadata = {
        key: value
        for key, value in state.items()
        if key not in ("arrays", "observation_seconds")
    }
    payload = state.get("arrays", {})
    if state["binding"] != _weight_binding():
        raise CheckpointError("weight construction source/boundary identity changed")
    if state["forecast_calls"] not in (0, 1) or state["boundary_calls"] != int(
        state["available"]
    ):
        raise CheckpointError("weight construction execution counts differ")
    if not state["available"]:
        if payload or snapshots or "initial" in state:
            raise CheckpointError("unavailable weighting has fabricated observations")
        return (
            dict(**metadata, anchor_reference=binding),
            {},
            dict(available=False, exact=None),
        )
    if state["forecast_calls"] != 1:
        raise CheckpointError("completed weights lack actual initial training forecast")
    required = {
        *WEIGHT_ARRAYS,
        *["param_" + key for key in anchor["params"]],
        *["norm_" + key for key in anchor["norms"]],
    }
    if set(payload) != required:
        raise CheckpointError("mandatory objective witness payload roster differs")
    if any(
        value.dtype != np.dtype("float64") or not np.isfinite(value).all()
        for value in payload.values()
    ):
        raise CheckpointError("objective witness must be finite float64")
    params = {
        key[6:]: value for key, value in payload.items() if key.startswith("param_")
    }
    norms = {
        key[5:]: value for key, value in payload.items() if key.startswith("norm_")
    }
    initial = {
        **state["initial"],
        "params": params,
        "norms": norms,
        "error_scale": payload["normalization"],
    }
    _base._validate_recipe([initial], recipe, train, development)
    for key in (
        "kind",
        "dt_s",
        "history_steps",
        "delay_steps",
        "horizon_steps",
        "train_fingerprint",
        "development_fingerprint",
    ):
        if initial[key] != anchor[key]:
            raise CheckpointError(
                "initial witness differs from trusted anchored contract/cache"
            )
    for group in ("params", "norms"):
        for key, value in initial[group].items():
            _exact(
                value,
                anchor[group][key],
                "actual initial parameters/norms differ from trusted anchored step0",
            )
    _exact(
        initial["error_scale"],
        anchor["error_scale"],
        "original hold scale differs from anchored reference",
    )
    batch = _base._batch(train)
    prediction, scale = payload["initial_training_prediction"], payload["normalization"]
    if (
        prediction.shape != batch.future_states.shape
        or scale.shape != batch.future_states.shape[1:]
    ):
        raise CheckpointError(
            "initial training forecast or normalization shape differs"
        )
    floor = np.asarray(recipe["hold_scale_floor"] ** 2, dtype=np.float64)
    e0 = np.mean(((prediction - batch.future_states) / scale) ** 2, axis=(0, 1))
    raw = 1.0 / np.maximum(e0, floor)
    normalizer = np.asarray(np.mean(raw))
    weights = raw / normalizer
    rebuilt = dict(
        initial_channel_mse=e0,
        raw_channel_weights=raw,
        channel_weights=weights,
        weight_floor=floor,
        weight_normalizer=normalizer,
        fixed_weights=weights,
    )
    for key, value in rebuilt.items():
        _exact(payload[key], value, "stored objective weight reduction differs: " + key)
    if (
        np.any(e0 < 0)
        or np.any(raw <= 0)
        or np.any(weights <= 0)
        or not np.isfinite(weights).all()
        or normalizer <= 0
    ):
        raise CheckpointError("invalid objective weights")
    for snapshot in snapshots:
        _exact(
            snapshot["error_scale"],
            scale,
            "original hold scale changed during trajectory",
        )
        if snapshot["channel_weights_fingerprint"] != inherited._fingerprint(weights):
            raise CheckpointError(
                "trajectory fixed weights differ from witnessed weights"
            )
    if snapshots:
        for group in ("params", "norms"):
            for key, value in initial[group].items():
                _exact(
                    snapshots[0][group][key],
                    value,
                    "checkpoint0 differs from actual weighting initialization",
                )
    if replay:
        replayed = _model().initial_training_forecast(
            params, norms, batch, initial["delay_steps"]
        )
        _exact(
            replayed, prediction, "canonical initial training forecast replay differs"
        )
    return (
        dict(
            **metadata,
            anchor_reference=binding,
            floor_active_channels=(e0 < floor).tolist(),
        ),
        payload,
        dict(
            available=True,
            exact=True,
            arrays=len(params) + len(norms) + 1,
            actual_initial_forecast_replayed=replay,
        ),
    )


def _reported_objective(payload):
    initial = SimpleNamespace(
        arrays=lambda: {
            key: value
            for key, value in payload.items()
            if key.startswith(("param_", "norm_"))
        }
    )
    return _model().weighting_metadata(
        initial,
        payload["initial_training_prediction"],
        payload["initial_channel_mse"],
        payload["raw_channel_weights"],
        payload["channel_weights"],
        payload["weight_floor"],
        payload["weight_normalizer"],
    )


def _diagnostics(snapshot, data, predict, weights, *, observed):
    if weights is None:
        raise CheckpointError("checkpoint diagnostic lacks witnessed weights")
    batch = _base._batch(data)
    prediction = np.asarray(
        predict(
            snapshot["params"],
            snapshot["norms"],
            *[jnp.asarray(getattr(batch, key)) for key in _base.ARRAYS[:3]],
            snapshot["delay_steps"],
        )
    )
    normalized = ((prediction - batch.future_states) / snapshot["error_scale"]) ** 2
    original_channels = np.sum(normalized, axis=(0, 1)) / normalized.size
    weighted_channels = np.sum(weights * normalized, axis=(0, 1)) / normalized.size
    original, weighted = float(original_channels.sum()), float(weighted_channels.sum())
    observed_original = snapshot["observed_unweighted_loss"] if observed else None
    observed_weighted = snapshot["selection_loss"] if observed else None
    for measured, reduced in (
        (observed_original, original),
        (observed_weighted, weighted),
    ):
        if measured is not None and not np.isclose(
            measured, reduced, rtol=1e-10, atol=1e-12
        ):
            raise CheckpointError(
                "checkpoint original/weighted objective decomposition differs"
            )
    metadata, arrays = _base._diagnostics(
        {**snapshot, "selection_loss": original}, data, lambda *args: prediction
    )
    groups = _base.GROUPS.values()
    metadata.update(
        selection_loss=observed_weighted if observed else weighted,
        reconstructed_selection_loss=weighted,
        absolute_loss_difference=None
        if not observed
        else abs(weighted - observed_weighted),
        original_loss=original,
        observed_unweighted_loss=observed_original,
        original_absolute_loss_difference=None
        if not observed
        else abs(original - observed_original),
        original_group_loss_contributions=[
            float(original_channels[a:b].sum()) for a, b in groups
        ],
        group_loss_contributions=[
            float(weighted_channels[a:b].sum()) for a, b in _base.GROUPS.values()
        ],
        objective="fixed initial-training inverse-channel-loss weighting; original hold scale preserved",
        evidence="development observed original/weighted losses"
        if observed
        else "training full-cache diagnostics; not the optimizer minibatch loss",
    )
    arrays.update(
        channel_loss_contributions=weighted_channels,
        original_channel_loss_contributions=original_channels,
    )
    return metadata, arrays


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


def _installed_initialization(path, objective_payload):
    """Bind the two actual boundaries even when initial dev evaluation aborted."""
    if not objective_payload:
        return
    metadata, arrays = load_arrays(Path(path) / "anchored/initializer-witness.npz")
    if (
        not metadata["witness"]["available"]
        or "initial_joint_coefficients" not in arrays
    ):
        raise CheckpointError(
            "completed weighting lacks an installed initializer witness"
        )
    coefficients = np.vstack(
        [objective_payload["param_" + key] for key in inherited.BLOCKS]
    )
    _exact(
        coefficients,
        arrays["initial_joint_coefficients"],
        "weight-boundary parameters differ from actual installed initializer",
    )
    for key, value in objective_payload.items():
        if key.startswith("norm_"):
            _exact(
                value,
                arrays["joint_" + key],
                "weight-boundary normalization differs from actual initializer",
            )


def _selection(snapshots, selected_step):
    rows = [
        dict(
            step=snapshot["step"],
            weighted=snapshot["selection_loss"],
            original=snapshot["observed_unweighted_loss"],
        )
        for snapshot in snapshots
    ]
    return dict(
        rows=rows,
        actual_selected_step=selected_step,
        weighted_minimum_recorded_step=min(rows, key=lambda row: row["weighted"])[
            "step"
        ]
        if rows
        else None,
        original_minimum_recorded_step=min(rows, key=lambda row: row["original"])[
            "step"
        ]
        if rows
        else None,
        scope="Original-loss minimum describes only this weighted trajectory; never an alternate fit, tested checkpoint or replacement selection.",
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
        raise CheckpointError("weighted checkpoint destination already exists")
    witness, payload, initial = _objective(
        captured.weight_observation,
        train,
        development,
        recipe,
        anchor_reference,
        captured.snapshots,
    )
    engine = _engine(payload.get("channel_weights"))
    evidence = engine.save_checkpoints(
        path / "anchored",
        captured,
        train=train,
        development=development,
        recipe=recipe,
        selected_step=selected_step,
        failure=failure,
        selected_model=selected_model,
    )
    _installed_initialization(path, payload)
    inner = _read(path / "anchored/manifest.json")
    common = _common(inner)
    save_arrays(
        path / "objective-witness.npz",
        dict(format=FORMAT, **common, witness=witness),
        payload,
    )
    predict = jax.jit(_model()._rollout, static_argnums=(5,))
    training_rows = []
    started = time.perf_counter()
    for snapshot in captured.snapshots:
        file = f"step-{snapshot['step']:04d}-training.npz"
        metadata, arrays = _diagnostics(
            snapshot, train, predict, payload["channel_weights"], observed=False
        )
        save_arrays(
            path / file,
            dict(format=FORMAT, **common, step=snapshot["step"], diagnostics=metadata),
            arrays,
        )
        training_rows.append(dict(step=snapshot["step"], diagnostics=file))
    _write(
        path / "selection-evidence.json",
        dict(format=FORMAT, **common, **_selection(captured.snapshots, selected_step)),
    )
    manifest = dict(
        format=FORMAT,
        **common,
        anchored_manifest_sha256=evidence["manifest_sha256"],
        objective_witness="objective-witness.npz",
        selection_evidence="selection-evidence.json",
        weighting_available=witness["available"],
        initial_identity=initial,
        training_checkpoints=training_rows,
        weight_observation_seconds=captured.weight_observation["observation_seconds"],
        training_diagnostic_seconds=time.perf_counter() - started,
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
        weighting_available=witness["available"],
        initial_identity=initial,
        training_checkpoints=len(training_rows),
        weight_observation_seconds=manifest["weight_observation_seconds"],
        training_diagnostic_seconds=manifest["training_diagnostic_seconds"],
    )


def _load(path, expected_sha256):
    path = Path(path)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or _sha(path / "manifest.json") != expected_sha256
    ):
        raise CheckpointError("weighted checkpoint external manifest anchor differs")
    manifest = _read(path / "manifest.json")
    if (
        manifest["format"] != FORMAT
        or manifest["objective_witness"] != "objective-witness.npz"
        or manifest["selection_evidence"] != "selection-evidence.json"
    ):
        raise CheckpointError("weighted checkpoint fixed paths or format differ")
    if any(file.is_symlink() for file in path.rglob("*")):
        raise CheckpointError("weighted checkpoint symbolic link")
    files = {str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()}
    if files != set(manifest["files"]) | {"manifest.json"}:
        raise CheckpointError("weighted checkpoint inventory differs")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or _sha(path / relative) != digest
        ):
            raise CheckpointError("weighted checkpoint payload changed")
    if _sha(path / "anchored/manifest.json") != manifest["anchored_manifest_sha256"]:
        raise CheckpointError("weighted/anchored manifest link differs")
    inner = _read(path / "anchored/manifest.json")
    if _common(manifest) != _common(inner):
        raise CheckpointError("weighted/anchored checkpoint contract differs")
    return manifest, inner


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
    path = Path(path)
    manifest, inner = _load(path, expected_sha256)
    if not {"unweighted_trace", "objective"} <= expectation.keys():
        raise CheckpointError(
            "missing external original-development trace or objective"
        )
    common = _common(inner)
    metadata, payload = load_arrays(path / "objective-witness.npz")
    if {key: value for key, value in metadata.items() if key != "witness"} != dict(
        format=FORMAT, **common
    ):
        raise CheckpointError("objective witness provenance differs")
    snapshots = []
    for row in inner["checkpoints"]:
        meta, arrays = load_arrays(path / "anchored" / row["snapshot"])
        snapshots.append(_base._snapshot_from(meta["snapshot"], arrays))
    saved = metadata["witness"]
    state = {
        key: value
        for key, value in saved.items()
        if key not in ("anchor_reference", "floor_active_channels")
    }
    state["arrays"] = payload
    witness, _, initial = _objective(
        state,
        train,
        development,
        expectation["recipe"],
        anchor_reference,
        snapshots,
        replay=replay,
    )
    if witness != saved or manifest["weighting_available"] != witness["available"]:
        raise CheckpointError("objective witness metadata differs")
    # Whether a forecast was reexecuted is fresh replay evidence, not saved status.
    stable_initial = (
        dict(initial, actual_initial_forecast_replayed=False)
        if initial["available"]
        else initial
    )
    if manifest["initial_identity"] != stable_initial:
        raise CheckpointError("initial identity evidence differs")
    original_trace = [
        dict(
            step=snapshot["step"],
            validation_rollout_mse=snapshot["observed_unweighted_loss"],
        )
        for snapshot in snapshots
    ]
    if (
        expectation["completed"] and expectation["unweighted_trace"] != original_trace
    ) or (not expectation["completed"] and expectation["unweighted_trace"] is not None):
        raise CheckpointError(
            "observed original-development trace differs from external report"
        )
    if expectation["completed"] and not witness["available"]:
        raise CheckpointError("completed fit lacks objective witness")
    if expectation["objective"] != (
        _reported_objective(payload) if expectation["completed"] else None
    ):
        raise CheckpointError(
            "selected-model objective metadata differs from actual weight witness"
        )
    if inner["fitter_calls"] == 0 and (state["forecast_calls"] or state["available"]):
        raise CheckpointError("unattempted optimizer has weighting observations")
    engine = _engine(payload.get("channel_weights"))
    checker = engine.replay_checkpoints if replay else engine.check_checkpoint_links
    evidence = checker(
        path / "anchored",
        train=train,
        development=development,
        provenance=provenance,
        source_sha256=source_sha256,
        expected_sha256=manifest["anchored_manifest_sha256"],
        expectation={
            key: value
            for key, value in expectation.items()
            if key not in ("unweighted_trace", "objective")
        },
        selected_model=selected_model,
    )
    _installed_initialization(path, payload)
    expected_rows = [
        dict(
            step=snapshot["step"],
            diagnostics=f"step-{snapshot['step']:04d}-training.npz",
        )
        for snapshot in snapshots
    ]
    if manifest["training_checkpoints"] != expected_rows:
        raise CheckpointError("training checkpoint diagnostic roster differs")
    predict = jax.jit(_model()._rollout, static_argnums=(5,)) if replay else None
    training_arrays = 0
    for snapshot, row in zip(snapshots, expected_rows, strict=True):
        meta, arrays = load_arrays(path / row["diagnostics"])
        forecast = (
            predict if replay else lambda *args, value=arrays["prediction"]: value
        )
        expected_meta, expected_arrays = _diagnostics(
            snapshot, train, forecast, payload["channel_weights"], observed=False
        )
        if meta != dict(
            format=FORMAT, **common, step=snapshot["step"], diagnostics=expected_meta
        ) or set(arrays) != set(expected_arrays):
            raise CheckpointError(
                "training diagnostic metadata or payload roster differs"
            )
        for name, value in expected_arrays.items():
            _exact(value, arrays[name], "training diagnostic replay differs")
            training_arrays += 1
    if _read(path / "selection-evidence.json") != dict(
        format=FORMAT, **common, **_selection(snapshots, expectation["selected_step"])
    ):
        raise CheckpointError("old/weighted selection evidence differs")
    return dict(
        evidence,
        manifest_sha256=expected_sha256,
        weighting_available=witness["available"],
        objective_witness_arrays=len(payload),
        training_checkpoints=len(snapshots),
        training_diagnostic_arrays=training_arrays,
        initial_identity=initial,
        initializers_or_optimizers_reexecuted=False,
    )


def check_checkpoint_links(
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
):
    return _check(
        path,
        train=train,
        development=development,
        provenance=provenance,
        source_sha256=source_sha256,
        expected_sha256=expected_sha256,
        expectation=expectation,
        anchor_reference=anchor_reference,
        selected_model=selected_model,
        replay=False,
    )


def replay_checkpoints(
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
):
    return _check(
        path,
        train=train,
        development=development,
        provenance=provenance,
        source_sha256=source_sha256,
        expected_sha256=expected_sha256,
        expectation=expectation,
        anchor_reference=anchor_reference,
        selected_model=selected_model,
        replay=True,
    )
