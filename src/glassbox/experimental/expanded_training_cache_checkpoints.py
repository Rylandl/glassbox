"""Current-data initialization evidence around the unchanged full-cache replay.

Private adapters change identity associations and envelope names only. The
numerical initializer, optimizer, acceptance chain and diagnostic reductions
remain pinned historical functions. Replay never initializes, solves or draws.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox.experimental import full_cache_gradient_checkpoints as previous

ROOT = previous.ROOT
CheckpointError = previous.CheckpointError
FORMAT = "glassbox-expanded-training-cache-checkpoints-v1"
PROTOCOL_SHA256 = "7f5f67cec628f2d86a2db23f3637a2a24c476d2c6f656e33b7983f1dfbbab65b"
SUBSET = ("w1", "b1", "w2", "memory", "memory_bias")
weighted = previous.inherited
anchored = weighted.inherited
_base = weighted._base
_sha, _read, _write = previous._sha, previous._read, previous._write
_exact, _copy = anchored._exact, anchored._copy


def _model():
    return importlib.import_module(
        "glassbox.experimental.expanded_training_cache_model"
    )


def _sources(values):
    previous._sources(values)
    required = {
        "docs/harness/expanded-training-cache-v1.json",
        "src/glassbox/experimental/expanded_training_cache_model.py",
        "src/glassbox/experimental/expanded_training_cache_checkpoints.py",
        "tests/test_expanded_training_cache_model.py",
        "tests/test_expanded_training_cache_checkpoints.py",
    }
    if not required <= values.keys() or any(
        _sha(ROOT / p) != values[p] for p in required
    ):
        raise CheckpointError("expanded-cache source inventory differs")
    if values["docs/harness/expanded-training-cache-v1.json"] != PROTOCOL_SHA256:
        raise CheckpointError("expanded-cache frozen protocol differs")


def _weighted_read(path):
    value = _read(path)
    # Explicit in-memory compatibility view for the unchanged private helper.
    # The persisted name states within-fit identity, never cross-arm identity.
    if "within_fit_initial_identity" in value:
        value["initial_identity"] = value.pop("within_fit_initial_identity")
    return value


def _weighted_write(path, value):
    value = dict(value)
    if "initial_identity" in value:
        value["within_fit_initial_identity"] = value.pop("initial_identity")
    _write(path, value)


@lru_cache(maxsize=1)
def _weighted():
    module = previous._private("_expanded_weighted_observer", weighted.__file__)
    module._model, module._objective = _model, _objective
    module._read, module._write = _weighted_read, _weighted_write
    module.FORMAT = "glassbox-expanded-training-cache-weighted-checkpoints-v1"
    return module


@lru_cache(maxsize=1)
def _engine():
    module = previous._private("_expanded_full_cache_observer", previous.__file__)
    module._model, module._weighted, module._sources = _model, _weighted, _sources
    module._summary = lambda *args: dict(
        previous._summary(*args), id="expanded-training-cache-v1"
    )
    # Only capture and numerical chain are used; the old persistent envelope is not.
    capture = previous._private("_expanded_attempt_capture", previous.previous.__file__)
    capture._model, capture._weighted = _model, _weighted
    capture._sources, capture._binding = _sources, module._binding
    capture._record = module._record
    module._engine = lambda: capture
    return module


def _subset_description(arrays):
    return {
        key: dict(
            shape=list(arrays[key].shape),
            dtype=arrays[key].dtype.str,
            c_order_bytes_sha256=hashlib.sha256(
                arrays[key].tobytes(order="C")
            ).hexdigest(),
        )
        for key in SUBSET
    }


def _external(context):
    reference = context["external_subset"]
    path = Path(reference["path"])
    if (
        reference["source_arm"] != "candidate"
        or reference["imported_arm"] != "fullcache384"
    ):
        raise CheckpointError("external initializer subset source role differs")
    suffix = "checkpoints/weighted/anchored/trajectory/step-0000.npz"
    for key, arm in (
        ("source_relative_path", "candidate"),
        ("imported_relative_path", "fullcache384"),
    ):
        rel = Path(reference[key])
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or rel.parts[1:] != (arm, *Path(suffix).parts)
        ):
            raise CheckpointError(
                "external initializer subset snapshot association differs"
            )
    if tuple(reference["subset_parameter_names"]) != SUBSET:
        raise CheckpointError("external initializer subset member roster differs")
    if (
        not isinstance(reference["source_bundle_sha256"], str)
        or len(reference["source_bundle_sha256"]) != 64
    ):
        raise CheckpointError("external initializer subset root anchor absent")
    if str(path.relative_to(path.parents[3])) != suffix.removeprefix("checkpoints/"):
        raise CheckpointError("external initializer subset nested path differs")
    if path.is_symlink() or _sha(path) != reference["snapshot_sha256"]:
        raise CheckpointError("external initializer subset snapshot bytes differ")
    for level, name in enumerate(
        ("trajectory", "anchored_checkpoint", "weighted_checkpoint", "checkpoint")
    ):
        directory = path.parents[level]
        manifest_path = directory / "manifest.json"
        if (
            manifest_path.is_symlink()
            or _sha(manifest_path) != reference[f"source_{name}_manifest_sha256"]
        ):
            raise CheckpointError("external initializer subset manifest anchor differs")
        manifest = _read(manifest_path)
        if (
            manifest["files"].get(str(path.relative_to(directory)))
            != reference["snapshot_sha256"]
        ):
            raise CheckpointError(
                "external initializer subset snapshot outside manifest"
            )
        if level:
            child = path.parents[level - 1] / "manifest.json"
            if manifest["files"].get(str(child.relative_to(directory))) != _sha(child):
                raise CheckpointError(
                    "external initializer subset nested manifest association differs"
                )
    metadata, values = load_arrays(path)
    if metadata.get("arm") != "candidate" or metadata["snapshot"]["step"] != 0:
        raise CheckpointError(
            "external initializer subset is not original candidate step0"
        )
    if any("param_" + key not in values for key in SUBSET):
        raise CheckpointError("external initializer subset missing array")
    arrays = {key: values["param_" + key] for key in SUBSET}
    if (
        _subset_description(arrays) != reference["subset_arrays"]
        or array_fingerprint({}, arrays) != reference["subset_array_fingerprint"]
    ):
        raise CheckpointError(
            "external initializer subset values differ from supplied anchor"
        )
    return arrays, metadata


def initialization_context(
    *,
    train,
    development,
    recipe,
    preparation,
    snapshot_path,
    trusted_subset,
    source_bundle_sha256,
):
    """Bind authenticated prepared data and the runner's frozen external subset.

    The caller authenticates the root/import seal and supplies the protocol's
    trusted subset entry. This helper verifies all nested manifest associations.
    """
    result = dict(
        format="glassbox-current-data-initialization-v1",
        current_data=dict(
            train_cache=_base._cache(train),
            development_cache=_base._cache(development),
            recipe=_base._json(recipe),
            preparation=_base._json(preparation),
        ),
        external_subset={
            **_base._json(trusted_subset),
            "path": str(snapshot_path),
            "source_bundle_sha256": source_bundle_sha256,
        },
    )
    _, metadata = _external(result)
    result["reference_training_windows"] = len(
        metadata["train_cache"]["metadata"]["keys"]
    )
    return result


def _context(context, train, development, recipe):
    if (
        set(context)
        != {"format", "current_data", "external_subset", "reference_training_windows"}
        or context["format"] != "glassbox-current-data-initialization-v1"
    ):
        raise CheckpointError("current-data initialization context schema differs")
    data = context["current_data"]
    if (
        set(data) != {"train_cache", "development_cache", "recipe", "preparation"}
        or data["train_cache"] != _base._cache(train)
        or data["development_cache"] != _base._cache(development)
        or data["recipe"] != recipe
    ):
        raise CheckpointError("current-data initialization cache/recipe differs")
    _, meta = _external(context)
    old_n = len(meta["train_cache"]["metadata"]["keys"])
    if (
        type(context["reference_training_windows"]) is not int
        or context["reference_training_windows"] != old_n
        or old_n >= len(_base._batch(train).past_states)
    ):
        raise CheckpointError("current-data training expansion/prefix budget differs")
    if (
        _base._cache(_slice(train, 0, old_n)) != meta["train_cache"]
        or _base._cache(development) != meta["development_cache"]
    ):
        raise CheckpointError(
            "current-data prefix/development differs from trusted cache"
        )
    _base._validate_recipe([], recipe, train, development)
    return {
        **context,
        "external_subset": {
            k: v for k, v in context["external_subset"].items() if k != "path"
        },
    }


def _return_binding():
    return dict(
        initializer=weighted._binding(_model().initialize_candidate)[1],
        caller=weighted._binding(_model().fit_candidate_sequence)[1],
    )


@contextmanager
def capture_fit(
    arm, provenance, *, source_sha256, expected_steps=tuple(range(0, 1001, 100))
):
    module = _model()
    state = dict(
        binding=_return_binding(),
        entered=0,
        returned=0,
        exceptional_unwinds=0,
        available=False,
        observation_seconds=0.0,
        arrays={},
    )
    with _engine().capture_fit(
        arm, provenance, source_sha256=source_sha256, expected_steps=expected_steps
    ) as captured:
        captured.initializer_return = state
        prior = sys.gettrace()

        def dispatch(frame, event, argument):
            downstream = prior(frame, event, argument) if prior else None
            if (
                event != "call"
                or frame.f_code is not module.initialize_candidate.__code__
            ):
                return downstream
            if (
                frame.f_back is None
                or frame.f_back.f_code is not module.fit_candidate_sequence.__code__
            ):
                raise CheckpointError("initializer return observer caller differs")
            state["entered"] += 1
            if state["entered"] != 1:
                raise CheckpointError("multiple actual initializer entries")
            state["train_fingerprint"] = _base._batch_fingerprint(
                frame.f_locals["batch"]
            )

            def observe(local_frame, local_event, value):
                nonlocal downstream
                if downstream is not None:
                    downstream = downstream(local_frame, local_event, value)
                if local_event == "return":
                    started = time.perf_counter()
                    if value is None:
                        state["exceptional_unwinds"] += 1
                    else:
                        state.update(
                            returned=1,
                            available=True,
                            model={
                                key: getattr(value, key)
                                for key in (
                                    "kind",
                                    "dt_s",
                                    "history_steps",
                                    "delay_steps",
                                )
                            },
                            arrays={
                                **{
                                    "param_" + k: _copy(v)
                                    for k, v in value.params.items()
                                },
                                **{
                                    "norm_" + k: _copy(v)
                                    for k, v in value.norms.items()
                                },
                            },
                        )
                    state["observation_seconds"] += time.perf_counter() - started
                return observe

            return observe

        try:
            sys.settrace(dispatch)
            yield captured
            if state["returned"] != 1 or state["exceptional_unwinds"]:
                raise CheckpointError("completed fit lacks normal initializer return")
        finally:
            sys.settrace(prior)


@dataclass(frozen=True)
class _ObjectiveContext:
    context: dict
    returned: dict


def _return_validate(state, context, train, development, recipe, *, completed):
    association = _context(context, train, development, recipe)
    if state["binding"] != _return_binding() or any(
        type(state[k]) is not int
        for k in ("entered", "returned", "exceptional_unwinds")
    ):
        raise CheckpointError("actual initializer return source/count identity differs")
    entered, returned, unwind = (
        state[k] for k in ("entered", "returned", "exceptional_unwinds")
    )
    if (
        entered not in (0, 1)
        or returned not in (0, 1)
        or unwind not in (0, 1)
        or returned + unwind != entered
        or type(state["available"]) is not bool
        or state["available"] != bool(returned)
    ):
        raise CheckpointError("initializer return/unwind prefix differs")
    if completed and returned != 1:
        raise CheckpointError("completed fit lacks initializer return")
    if entered and state.get("train_fingerprint") != _base._batch_fingerprint(train):
        raise CheckpointError("actual initializer return cache differs")
    arrays = state["arrays"]
    if not returned:
        if arrays or "model" in state:
            raise CheckpointError("unavailable initializer fabricates returned arrays")
    else:
        params = {k[6:]: v for k, v in arrays.items() if k.startswith("param_")}
        norms = {k[5:]: v for k, v in arrays.items() if k.startswith("norm_")}
        if set(params) != set(SUBSET) | set(anchored.BLOCKS) or set(norms) != set(
            anchored.BASE_NORMS
        ) | {"interaction_scale", "autonomous_scale"}:
            raise CheckpointError("actual returned initializer array roster differs")
        if set(arrays) != {
            *("param_" + k for k in params),
            *("norm_" + k for k in norms),
        }:
            raise CheckpointError(
                "actual returned initializer full payload roster differs"
            )
        batch = _base._batch(train)
        d, u = batch.future_states.shape[-1], batch.future_inputs.shape[-1]
        delay = max(1, int(np.rint(recipe["delay_s"] / batch.dt_s)))
        f = (delay + 1) * (d + u) + recipe["memory"]
        param_shapes = dict(
            linear=(f, d),
            interaction=(d * u, d),
            autonomous=(d * (d + 1) // 2, d),
            bias=(d,),
            w1=(f, recipe["width"]),
            b1=(recipe["width"],),
            w2=(recipe["width"], d),
            memory=(f, recipe["memory"]),
            memory_bias=(recipe["memory"],),
        )
        norm_shapes = dict(
            state_mean=(d,),
            state_scale=(d,),
            input_mean=(u,),
            input_scale=(u,),
            feature_scale=(f,),
            delta_scale=(d,),
            interaction_scale=(d * u,),
            autonomous_scale=(d * (d + 1) // 2,),
        )
        if any(params[k].shape != shape for k, shape in param_shapes.items()) or any(
            norms[k].shape != shape for k, shape in norm_shapes.items()
        ):
            raise CheckpointError(
                "actual returned initializer parameter/normalization shape differs"
            )
        if any(v.dtype != np.dtype("float64") for v in arrays.values()):
            raise CheckpointError("actual returned initializer dtype differs")
        external, _ = _external(context)
        for key in SUBSET:
            _exact(
                params[key],
                external[key],
                "actual initializer subset differs from external step0: " + key,
            )
        batch = _base._batch(train)

        def count(seconds):
            return max(1, int(np.rint(seconds / batch.dt_s)))

        expected = dict(
            kind=recipe["kind"],
            dt_s=batch.dt_s,
            history_steps=max(count(recipe["context_s"]), count(recipe["delay_s"]) + 1),
            delay_steps=count(recipe["delay_s"]),
        )
        if state["model"] != expected:
            raise CheckpointError("actual initializer return model contract differs")
        if completed and any(not np.isfinite(v).all() for v in arrays.values()):
            raise CheckpointError("completed initializer contains nonfinite arrays")
    return dict(
        **{
            k: v for k, v in state.items() if k not in ("arrays", "observation_seconds")
        },
        current_data_initialization=association,
    )


def _objective(
    state, train, development, recipe, reference, snapshots, *, replay=False
):
    """Bounded replacement of only the old cross-arm initialization association."""
    context, returned = reference.context, reference.returned
    association = _context(context, train, development, recipe)
    metadata = {
        k: v
        for k, v in state.items()
        if k
        not in (
            "arrays",
            "observation_seconds",
            "current_data_initialization",
            "floor_active_channels",
        )
    }
    payload = state.get("arrays", {})
    if (
        state["binding"] != _weighted()._weight_binding()
        or state["forecast_calls"] not in (0, 1)
        or state["boundary_calls"] != int(state["available"])
    ):
        raise CheckpointError("weight source/boundary identity differs")
    if not state["available"]:
        if payload or snapshots or "initial" in state:
            raise CheckpointError("unavailable weighting fabricates observations")
        return (
            dict(**metadata, current_data_initialization=association),
            {},
            dict(available=False, exact=None, scope="within_current_fit"),
        )
    if state["forecast_calls"] != 1 or not returned["available"]:
        raise CheckpointError("weights lack actual initializer return/forecast")
    required = set(weighted.WEIGHT_ARRAYS) | set(returned["arrays"])
    if set(payload) != required or any(
        v.dtype != np.dtype("float64") or not np.isfinite(v).all()
        for v in payload.values()
    ):
        raise CheckpointError("objective witness finite float64 roster differs")
    for key, value in returned["arrays"].items():
        _exact(
            payload[key],
            value,
            "weight boundary differs from actual initializer return: " + key,
        )
    params = {k[6:]: v for k, v in payload.items() if k.startswith("param_")}
    norms = {k[5:]: v for k, v in payload.items() if k.startswith("norm_")}
    initial = dict(
        **state["initial"],
        params=params,
        norms=norms,
        error_scale=payload["normalization"],
    )
    _base._validate_recipe([initial], recipe, train, development)
    if (
        any(initial[k] != returned["model"][k] for k in returned["model"])
        or initial["train_fingerprint"] != _base._batch_fingerprint(train)
        or initial["development_fingerprint"] != _base._batch_fingerprint(development)
    ):
        raise CheckpointError("actual weighting cache/initial model contract differs")
    batch = _base._batch(train)
    prediction, scale = payload["initial_training_prediction"], payload["normalization"]
    if (
        prediction.shape != batch.future_states.shape
        or scale.shape != batch.future_states.shape[1:]
    ):
        raise CheckpointError("initial forecast/normalization shape differs")
    floor = np.asarray(recipe["hold_scale_floor"] ** 2, dtype=np.float64)
    e0 = np.mean(((prediction - batch.future_states) / scale) ** 2, axis=(0, 1))
    raw = 1.0 / np.maximum(e0, floor)
    normalizer = np.asarray(np.mean(raw))
    weights = raw / normalizer
    for key, value in dict(
        initial_channel_mse=e0,
        raw_channel_weights=raw,
        channel_weights=weights,
        weight_floor=floor,
        weight_normalizer=normalizer,
        fixed_weights=weights,
    ).items():
        _exact(
            payload[key], value, "stored current-data weight reduction differs: " + key
        )
    for snapshot in snapshots:
        _exact(snapshot["error_scale"], scale, "hold scale changed during trajectory")
        if snapshot["channel_weights_fingerprint"] != anchored._fingerprint(weights):
            raise CheckpointError("trajectory weights differ from actual weighting")
    if snapshots:
        for key, value in returned["arrays"].items():
            group, name = (
                ("params", key[6:]) if key.startswith("param_") else ("norms", key[5:])
            )
            _exact(
                snapshots[0][group][name],
                value,
                "checkpoint0 differs from actual initializer return",
            )
    if replay:
        actual = _model().initial_training_forecast(
            params, norms, batch, initial["delay_steps"]
        )
        _exact(actual, prediction, "canonical current-data weighting forecast differs")
    return (
        dict(
            **metadata,
            current_data_initialization=association,
            floor_active_channels=(e0 < floor).tolist(),
        ),
        payload,
        dict(
            available=True,
            exact=True,
            arrays=len(params) + len(norms) + 1,
            scope="within_current_fit",
            actual_initial_forecast_replayed=replay,
        ),
    )


def _return_links(path, state):
    """Link real return even if no completed weighting boundary was reached."""
    metadata, arrays = load_arrays(
        Path(path) / "weighted/anchored/initializer-witness.npz"
    )
    witness = metadata["witness"]
    if witness["initializer_calls"] != state["entered"]:
        raise CheckpointError("initializer entry witnesses disagree")
    if not state["available"]:
        return dict(
            available=False, installed_return_link=None, affine_backward_residual=None
        )
    if not witness["available"] or "initial_joint_coefficients" not in arrays:
        raise CheckpointError("normal initializer return lacks installed coefficients")
    payload = state["arrays"]
    _exact(
        np.vstack([payload["param_" + key] for key in anchored.BLOCKS]),
        arrays["initial_joint_coefficients"],
        "initializer return differs from installed joint coefficients",
    )
    for key, value in payload.items():
        if key.startswith("norm_"):
            _exact(
                value,
                arrays["joint_" + key],
                "initializer return differs from installed normalization",
            )
    return dict(available=True, installed_return_link=True)


def _affine_diagnostic(path, train):
    metadata, arrays = load_arrays(
        Path(path) / "weighted/anchored/initializer-witness.npz"
    )
    witness = metadata["witness"]
    if not witness["available"]:
        return dict(available=False, reason="actual_affine_precursor_unavailable")
    norms = {
        k.removeprefix("joint_norm_"): v
        for k, v in arrays.items()
        if k.startswith("joint_norm_")
    }
    rebuilt = anchored._rebuild(
        train, norms, witness["settings"], witness["history_steps"]
    )
    count = len(arrays["affine_linear"])
    design = np.column_stack((rebuilt["D"][:, :count], rebuilt["D"][:, -1]))
    penalty = np.diag(np.r_[np.full(count, witness["settings"]["ridge"]), 0.0])
    system, rhs = design.T @ design + penalty, design.T @ rebuilt["Y"]
    coefficients = np.vstack((arrays["affine_linear"], arrays["affine_bias"]))
    return dict(
        scope="reconstructed affine equations; actual affine matrices were not observed",
        matrices={"A": anchored._fingerprint(system), "B0": anchored._fingerprint(rhs)},
        **anchored._backward(system, coefficients, rhs),
    )


def _snapshots(path):
    outer = _read(Path(path) / "weighted/anchored/manifest.json")
    result = []
    for row in outer["checkpoints"]:
        meta, arrays = load_arrays(Path(path) / "weighted/anchored" / row["snapshot"])
        result.append(_base._snapshot_from(meta["snapshot"], arrays))
    return result


def _slice(data, begin, end):
    batch = replace(
        data.batch, **{key: getattr(data.batch, key)[begin:end] for key in _base.ARRAYS}
    )
    return replace(
        data,
        batch=batch,
        keys=data.keys[begin:end],
        source_origins=data.source_origins[begin:end],
        **{
            key: None if getattr(data, key) is None else getattr(data, key)[begin:end]
            for key in ("past_excitation", "future_excitation")
        },
    )


def _partitions(path, train, context, snapshots, weights):
    """Reduce saved forecasts only. No prediction callback executes model code."""
    n, old_n = len(train.keys), context["reference_training_windows"]
    result = dict(
        scope="training data only; descriptive, never checkpoint selection",
        forecast_calls=0,
        partitions={},
        checkpoints=[],
    )
    for name, begin, end in (
        ("full", 0, n),
        ("original_prefix", 0, old_n),
        ("added", old_n, n),
    ):
        subset = _slice(train, begin, end)
        result["partitions"][name] = dict(
            begin=begin, end=end, windows=end - begin, cache=_base._cache(subset)
        )
    for snapshot in snapshots:
        _, full = load_arrays(
            Path(path) / "weighted" / f"step-{snapshot['step']:04d}-training.npz"
        )
        row = dict(step=snapshot["step"], partitions={})
        for name, info in result["partitions"].items():
            begin, end = info["begin"], info["end"]
            prediction = full["prediction"][begin:end]
            subset = _slice(train, begin, end)
            meta, arrays = weighted._diagnostics(
                snapshot,
                subset,
                lambda *args, value=prediction: value,
                weights,
                observed=False,
            )
            row["partitions"][name] = dict(
                diagnostics=meta,
                arrays={
                    key: value.tolist()
                    for key, value in arrays.items()
                    if key not in ("prediction", "target")
                },
            )
        result["checkpoints"].append(row)
    return result


def _portable_state(state):
    return {
        k: v for k, v in state.items() if k not in ("arrays", "observation_seconds")
    }


def _renamed(result):
    result = dict(result)
    if "initial_identity" in result:
        result["within_fit_initial_identity"] = result.pop("initial_identity")
    return result


def save_checkpoints(
    path,
    captured,
    *,
    train,
    development,
    recipe,
    initialization_context,
    selected_step=None,
    failure=None,
    selected_model=None,
):
    path = Path(path)
    if path.exists():
        raise CheckpointError("expanded-cache checkpoint destination exists")
    state = captured.initializer_return
    return_meta = _return_validate(
        state,
        initialization_context,
        train,
        development,
        recipe,
        completed=failure is None,
    )
    typed = _ObjectiveContext(initialization_context, state)
    result = _weighted().save_checkpoints(
        path / "weighted",
        captured,
        train=train,
        development=development,
        recipe=recipe,
        anchor_reference=typed,
        selected_step=selected_step,
        failure=failure,
        selected_model=selected_model,
    )
    common = previous.previous._common(_read(path / "weighted/manifest.json"))
    save_arrays(
        path / "initializer-return.npz",
        dict(format=FORMAT, **common, witness=return_meta),
        state["arrays"],
    )
    links = _return_links(path, state)
    affine = _affine_diagnostic(path, train)
    observed = captured.safeguard_observation
    gradient = previous._gradient(observed["attempts"], len(train.keys))
    for row in observed["attempts"]:
        previous._indices(row, len(train.keys))
    transcript = {key: value for key, value in observed.items() if key != "proposals"}
    transcript.update(
        format=FORMAT,
        **common,
        gradient=gradient,
        proposal_count=len(observed["proposals"]),
        completed=failure is None,
        failure=failure,
        selected_step=selected_step,
        current_data_initialization=return_meta["current_data_initialization"],
    )
    _write(path / "attempts.json", transcript)
    save_arrays(
        path / "proposals.npz",
        dict(format=FORMAT, **common, proposal_count=len(observed["proposals"])),
        previous.previous._pack(observed["proposals"]),
    )
    partitions = _partitions(
        path,
        train,
        initialization_context,
        captured.snapshots,
        captured.weight_observation.get("arrays", {}).get("channel_weights"),
    )
    _write(
        path / "training-partitions.json", dict(format=FORMAT, **common, **partitions)
    )
    manifest = dict(
        format=FORMAT,
        **common,
        weighted_manifest_sha256=result["manifest_sha256"],
        attempt_transcript="attempts.json",
        proposals="proposals.npz",
        initializer_return="initializer-return.npz",
        training_partitions="training-partitions.json",
        initializer_return_links=links,
        reconstructed_affine_equations=affine,
        gradient=gradient,
        initializer_return_observation_seconds=state["observation_seconds"],
        attempt_observation_seconds=observed["observation_seconds"],
        files={
            str(p.relative_to(path)): _sha(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        },
    )
    _write(path / "manifest.json", manifest)
    return dict(
        _renamed(result),
        manifest_sha256=_sha(path / "manifest.json"),
        gradient=gradient,
        attempts=len(observed["attempts"]),
        completed_attempts=gradient["completed_acceptance_attempts"],
        initializer_return_available=state["available"],
        initializer_return_links=links,
        training_partition_checkpoints=len(captured.snapshots),
    )


@lru_cache(maxsize=1)
def _loader():
    module = previous._private("_expanded_envelope_loader", previous.previous.__file__)
    module.FORMAT = FORMAT
    return module


def _check(
    path,
    *,
    train,
    development,
    provenance,
    source_sha256,
    expected_sha256,
    expectation,
    initialization_context,
    selected_model=None,
    replay=False,
):
    _sources(source_sha256)
    path = Path(path)
    manifest = _loader()._load(path, expected_sha256)
    required = {"gradient", "safeguard"}
    if not required <= expectation.keys():
        raise CheckpointError("missing external gradient/safeguard report")
    common = previous.previous._common(_read(path / "weighted/manifest.json"))
    if previous.previous._common(manifest) != common:
        raise CheckpointError("expanded-cache envelope provenance differs")
    meta, arrays = load_arrays(path / "initializer-return.npz")
    if {key: value for key, value in meta.items() if key != "witness"} != dict(
        format=FORMAT, **common
    ):
        raise CheckpointError("initializer return provenance differs")
    state = {
        key: value
        for key, value in meta["witness"].items()
        if key != "current_data_initialization"
    }
    state["arrays"] = arrays
    actual_meta = _return_validate(
        state,
        initialization_context,
        train,
        development,
        expectation["recipe"],
        completed=expectation["completed"],
    )
    if actual_meta != meta["witness"]:
        raise CheckpointError("current-data initializer association differs")
    links = _return_links(path, state)
    if manifest["initializer_return_links"] != links or manifest[
        "reconstructed_affine_equations"
    ] != _affine_diagnostic(path, train):
        raise CheckpointError("initializer return/affine equation evidence differs")
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
        expectation={k: v for k, v in expectation.items() if k not in required},
        anchor_reference=_ObjectiveContext(initialization_context, state),
        selected_model=selected_model,
    )
    transcript = _read(path / "attempts.json")
    proposal_meta, proposal_arrays = load_arrays(path / "proposals.npz")
    count = transcript["proposal_count"]
    if proposal_meta != dict(format=FORMAT, **common, proposal_count=count) or any(
        transcript.get(key) != value
        for key, value in dict(
            format=FORMAT,
            **common,
            failure=expectation["failure"],
            completed=expectation["completed"],
            selected_step=expectation["selected_step"],
            current_data_initialization=actual_meta["current_data_initialization"],
        ).items()
    ):
        raise CheckpointError("attempt transcript external outcome/provenance differs")
    if "balanced_initial_witness" in transcript:
        raise CheckpointError("obsolete cross-arm initialization association")
    _, payload = load_arrays(path / "weighted/objective-witness.npz")
    snapshots = _snapshots(path)
    chain = _engine()._chain(
        transcript,
        previous.previous._unpack(proposal_arrays, count),
        payload,
        snapshots,
        train,
        expectation["recipe"],
        completed=expectation["completed"],
        selected_step=expectation["selected_step"],
        safeguard=expectation["safeguard"],
        replay=replay,
    )
    gradient = chain["gradient"]
    previous._same_gradient(manifest.get("gradient"), gradient)
    if expectation["completed"]:
        previous._same_gradient(expectation["gradient"], gradient)
    elif expectation["gradient"] is not None:
        raise CheckpointError("failed fit claims selected gradient report")
    partitions = _partitions(
        path, train, initialization_context, snapshots, payload.get("channel_weights")
    )
    if _read(path / "training-partitions.json") != dict(
        format=FORMAT, **common, **partitions
    ):
        raise CheckpointError("saved training partition reductions differ")
    return dict(
        _renamed(result),
        manifest_sha256=expected_sha256,
        gradient=gradient,
        safeguard=chain,
        initializer_return_available=state["available"],
        initializer_return_links=links,
        training_partition_checkpoints=len(snapshots),
        gradient_or_adam_reexecuted=False,
        optimizer_replay_scope="Unchanged full-cache acceptance/interpolation and checkpoint chain; no independent gradient, Adam or solve recomputation.",
    )


def check_checkpoint_links(path, **kwargs):
    return _check(path, **kwargs, replay=False)


def replay_checkpoints(path, **kwargs):
    return _check(path, **kwargs, replay=True)
