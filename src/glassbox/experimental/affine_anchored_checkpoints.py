"""Read-only initializer witnesses around the pinned checkpoint observer.

The historical observer runs in an isolated module namespace. No initializer,
linear solve, optimizer or prediction is added during capture. Witness replay
reconstructs designs and matrix products; it never solves normal equations.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import marshal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import CodeType

import numpy as np

from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox.experimental import fit_checkpoints as historical

_spec = importlib.util.spec_from_file_location(
    f"{__package__}._affine_anchored_checkpoint_shared", historical.__file__
)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
ROOT = _base.ROOT
CheckpointError = _base.CheckpointError
FORMAT = "glassbox-affine-anchored-checkpoints-v1"
BLOCKS = ("linear", "interaction", "autonomous", "bias")
BASE_NORMS = (
    "state_mean",
    "state_scale",
    "input_mean",
    "input_scale",
    "feature_scale",
    "delta_scale",
)
_inherited_sources = _base._sources


def _candidate():
    return importlib.import_module("glassbox.experimental.affine_anchored_model")


def _arm(arm):
    if arm == "candidate":
        module = _candidate()
        return (
            module.fit_candidate_sequence,
            module.AnchoredQuadraticSequenceModel,
            module._rollout,
        )
    return historical._arm(arm)


def _sources(source_sha256):
    required = {
        "src/glassbox/experimental/affine_anchored_checkpoints.py",
        "tests/test_affine_anchored_checkpoints.py",
        "src/glassbox/experimental/affine_anchored_model.py",
        "tests/test_affine_anchored_model.py",
    }
    if not required <= source_sha256.keys():
        raise CheckpointError("anchored checkpoint source map omits required modules")
    return _inherited_sources(source_sha256)


# Bind only the private copy. The historical module and all fitting code stay exact.
_base._arm = _arm
_base._sources = _sources


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def _fingerprint(value):
    value = np.asarray(value)
    return dict(
        dtype=value.dtype.str,
        shape=list(value.shape),
        fingerprint=array_fingerprint({}, {"value": value}),
    )


def _copy(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


def _exact(left, right, message):
    if _fingerprint(left) != _fingerprint(right):
        raise CheckpointError(message)


def _initializer(arm):
    return (
        _candidate().initialize_candidate
        if arm == "candidate"
        else historical.quadratic.initialize_candidate
    )


def _initializer_binding(arm):
    function = _initializer(arm)
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
        raise CheckpointError("initializer code differs from pinned source")
    solves = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "coefficients"
            for target in item.targets
        )
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Attribute)
        and item.value.func.attr == "solve"
    ]
    returns = [item for item in node.body if isinstance(item, ast.Return)]
    if len(solves) != 1 or len(returns) != 1:
        raise CheckpointError("initializer solve/return boundaries changed")

    def portable(value):
        return value.replace(
            co_filename=str(path.relative_to(ROOT)),
            co_consts=tuple(
                portable(item) if isinstance(item, CodeType) else item
                for item in value.co_consts
            ),
        )

    import hashlib

    identity = dict(
        path=str(path.relative_to(ROOT)),
        function=function.__name__,
        source_sha256=_base._sha(path),
        code_sha256=hashlib.sha256(marshal.dumps(portable(code))).hexdigest(),
        solve_line=solves[0].lineno,
        installed_line=returns[0].lineno,
    )
    return function, identity


def _observe_solve(local, arm, identity):
    precursor = local["baseline"]
    design = local["design"]
    base_end = precursor.params["linear"].shape[0]
    interaction_end = base_end + local["products"].shape[-1]
    names = {"D": "design", "Y": "target", "penalty": "penalty"}
    if arm == "candidate":
        names.update(
            A="system", B0="zero_rhs", W_affine="anchor", centered_rhs="anchored_rhs"
        )
    captured = {name: _fingerprint(local[key]) for name, key in names.items()}
    return dict(
        binding=identity,
        arm=arm,
        stage="before_existing_solve",
        captured_fingerprints=captured,
        captured_local_names=names,
        train_fingerprint=_base._batch_fingerprint(local["batch"]),
        settings={
            **{key: local[key] for key in ("seed", "width", "memory", "ridge")},
            "delay_steps": precursor.delay_steps,
        },
        dt_s=precursor.dt_s,
        history_steps=precursor.history_steps,
        horizon_steps=local["batch"].future_states.shape[1],
        block_boundaries=dict(
            linear=[0, base_end],
            interaction=[base_end, interaction_end],
            autonomous=[interaction_end, design.shape[1] - 1],
            bias=[design.shape[1] - 1, design.shape[1]],
        ),
        precursor={
            "linear": _copy(precursor.params["linear"]),
            "bias": _copy(precursor.params["bias"]),
        },
        precursor_norms={key: _copy(value) for key, value in precursor.norms.items()},
        joint_norms={key: _copy(value) for key, value in local["norms"].items()},
        coefficients=None,
    )


@contextmanager
def capture_fit(
    arm, provenance, *, source_sha256, expected_steps=tuple(range(0, 1001, 100))
):
    """Compose read-only initializer observation with the original checkpoint loop."""
    _sources(source_sha256)
    function = _arm(arm)[0]
    initializer, binding = (
        (None, None) if arm == "baseline" else _initializer_binding(arm)
    )
    state = dict(calls=0, witness=None, seconds=0.0)
    previous = sys.gettrace()

    def observe(frame, event, argument):
        if event == "line":
            started = time.perf_counter()
            if frame.f_lineno == binding["solve_line"] and state["witness"] is None:
                state["witness"] = _observe_solve(frame.f_locals, arm, binding)
            elif frame.f_lineno == binding["installed_line"]:
                if state["witness"] is None:
                    raise CheckpointError("initializer installed without solve witness")
                state["witness"]["coefficients"] = _copy(frame.f_locals["coefficients"])
                state["witness"]["stage"] = "joint_coefficients_installed"
            state["seconds"] += time.perf_counter() - started
        return observe

    def dispatch(frame, event, argument):
        if (
            initializer is not None
            and event == "call"
            and frame.f_code is initializer.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is function.__code__
        ):
            state["calls"] += 1
            if state["calls"] != 1:
                raise CheckpointError("multiple optimizer initializer calls")
            return observe
        return previous(frame, event, argument) if previous is not None else None

    try:
        sys.settrace(dispatch)
        with _base.capture_fit(
            arm, provenance, source_sha256=source_sha256, expected_steps=expected_steps
        ) as captured:
            captured.initializer_observation = state
            yield captured
            if arm != "baseline" and (
                state["calls"] != 1
                or state["witness"] is None
                or state["witness"]["coefficients"] is None
            ):
                raise CheckpointError("completed fit missed initializer witness")
    finally:
        sys.settrace(previous)


def _rebuild(train, norms, settings, history):
    """Rebuild the inherited design and target by arithmetic only, never solve."""
    batch = _base._batch(train)
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    x = (physical - norms["state_mean"]) / norms["state_scale"]
    u = np.concatenate((batch.past_inputs, batch.future_inputs), axis=1)
    u = (u - norms["input_mean"]) / norms["input_scale"]
    current, commands = x[:, history:-1], u[:, history:]
    delay = settings["delay_steps"]
    hidden = np.zeros((len(current), settings["memory"]))
    features = np.stack(
        [
            historical.public._features(
                x[:, history + t],
                u[:, history + t],
                x[:, history + t - delay : history + t],
                u[:, history + t - delay : history + t],
                hidden,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        axis=1,
    ).reshape(-1, len(norms["feature_scale"]))
    products = (current[..., :, None] * commands[..., None, :]).reshape(
        len(features), -1
    )
    i, j = np.triu_indices(current.shape[-1])
    autonomous = (current[..., i] * current[..., j]).reshape(len(features), -1)
    for name, values in (("interaction", products), ("autonomous", autonomous)):
        scale = values.std(axis=0)
        _exact(
            norms[f"{name}_scale"],
            np.where(scale > 1e-8, scale, 1.0),
            "product feature normalization differs",
        )
    design = np.column_stack(
        (
            features / norms["feature_scale"],
            products / norms["interaction_scale"],
            autonomous / norms["autonomous_scale"],
            np.ones(len(features)),
        )
    )
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, settings["ridge"]), 0.0])
    delta = (batch.future_states - physical[:, history:-1]) / norms["state_scale"]
    target = (delta / norms["delta_scale"]).reshape(len(features), -1)
    return dict(
        D=design,
        Y=target,
        penalty=penalty,
        A=design.T @ design + penalty,
        B0=design.T @ target,
    )


def _reference(precursor, boundaries):
    d = precursor["linear"].shape[1]
    count = boundaries["bias"][1]
    result = np.zeros((count, d))
    result[: len(precursor["linear"])] = precursor["linear"]
    result[-1] = precursor["bias"]
    return result


def _validate_witness(witness, train, recipe):
    batch = _base._batch(train)

    def count(seconds):
        return max(1, int(np.rint(seconds / batch.dt_s)))

    delay = count(recipe["delay_s"])
    settings = {key: recipe[key] for key in ("seed", "width", "memory")}
    settings.update(
        ridge=recipe["ridge_fraction"]
        * len(batch.past_states)
        * batch.future_states.shape[1],
        delay_steps=delay,
    )
    if (
        witness["settings"] != settings
        or witness["dt_s"] != batch.dt_s
        or witness["history_steps"] != max(count(recipe["context_s"]), delay + 1)
        or witness["horizon_steps"] != count(recipe["horizon_s"])
        or witness["train_fingerprint"] != _base._batch_fingerprint(train)
    ):
        raise CheckpointError("initializer witness differs from expected recipe/cache")
    if set(witness["precursor_norms"]) != set(BASE_NORMS):
        raise CheckpointError("affine precursor normalization roster differs")
    if set(witness["joint_norms"]) != {
        *BASE_NORMS,
        "interaction_scale",
        "autonomous_scale",
    }:
        raise CheckpointError("joint initializer normalization roster differs")
    for key in BASE_NORMS:
        _exact(
            witness["precursor_norms"][key],
            witness["joint_norms"][key],
            "affine precursor and joint normalization differ",
        )
    # Reproduce only deterministic reductions, including failed fits with no
    # checkpoint. No random parameter draw, affine initializer or solve occurs.
    history = witness["history_steps"]
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    controls = np.concatenate((batch.past_inputs, batch.future_inputs), axis=1)
    current = physical[:, history:-1]
    xm, xs = current.mean((0, 1)), current.std((0, 1))
    um, us = batch.future_inputs.mean((0, 1)), batch.future_inputs.std((0, 1))
    xs, us = np.where(xs > 1e-8, xs, 1), np.where(us > 1e-8, us, 1)
    x, u = (physical - xm) / xs, (controls - um) / us
    delta = (batch.future_states - current) / xs
    ds = np.maximum(delta.std((0, 1)), 1e-4)
    hidden = np.zeros((len(current), settings["memory"]))
    features = np.stack(
        [
            historical.public._features(
                x[:, history + t],
                u[:, history + t],
                x[:, history + t - delay : history + t],
                u[:, history + t - delay : history + t],
                hidden,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        axis=1,
    ).reshape(
        -1,
        (delay + 1) * (current.shape[-1] + batch.future_inputs.shape[-1])
        + settings["memory"],
    )
    fs = features.std(0)
    norms = dict(
        state_mean=xm,
        state_scale=xs,
        input_mean=um,
        input_scale=us,
        feature_scale=np.where(fs > 1e-8, fs, 1.0),
        delta_scale=ds,
    )
    for key, value in norms.items():
        _exact(
            witness["precursor_norms"][key],
            value,
            "actual affine precursor normalization differs from trusted cache",
        )
    names = {"D": "design", "Y": "target", "penalty": "penalty"}
    if witness["arm"] == "candidate":
        names.update(
            A="system", B0="zero_rhs", W_affine="anchor", centered_rhs="anchored_rhs"
        )
    if witness["captured_local_names"] != names or set(
        witness["captured_fingerprints"]
    ) != set(names):
        raise CheckpointError("mandatory actual initializer fingerprint roster differs")
    linear = witness["precursor"]["linear"]
    dimension, control = batch.future_states.shape[-1], batch.future_inputs.shape[-1]
    base_end = linear.shape[0]
    interaction_end = base_end + dimension * control
    end = interaction_end + dimension * (dimension + 1) // 2
    boundaries = dict(
        linear=[0, base_end],
        interaction=[base_end, interaction_end],
        autonomous=[interaction_end, end],
        bias=[end, end + 1],
    )
    if witness["block_boundaries"] != boundaries or linear.shape != (
        len(witness["joint_norms"]["feature_scale"]),
        dimension,
    ):
        raise CheckpointError("initializer feature boundaries differ")
    installed = witness["coefficients"] is not None
    if witness["stage"] != (
        "joint_coefficients_installed" if installed else "before_existing_solve"
    ):
        raise CheckpointError("initializer execution stage differs")


def _backward(matrix, coefficients, rhs):
    if coefficients is None:
        return dict(available=False, reason="solve_did_not_return_coefficients")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = float(np.linalg.norm(matrix @ coefficients - rhs))
        denominator = float(
            np.linalg.norm(matrix) * np.linalg.norm(coefficients) + np.linalg.norm(rhs)
        )
    return dict(
        available=bool(np.isfinite([residual, denominator]).all()),
        residual_frobenius=residual if np.isfinite(residual) else None,
        relative_backward_residual=residual / denominator
        if denominator > 0 and np.isfinite([residual, denominator]).all()
        else None,
        scientific_acceptance_threshold=None,
    )


def _witness(captured, train, recipe):
    observed = captured.initializer_observation
    if captured.arm == "baseline":
        if not captured.snapshots:
            return dict(
                available=False,
                reason="public_step_zero_unavailable",
                initializer_calls=0,
            ), {}
        snapshot = captured.snapshots[0]
        linear, bias = snapshot["params"]["linear"], snapshot["params"]["bias"]
        boundaries = dict(
            linear=[0, len(linear)],
            interaction=None,
            autonomous=None,
            bias=[len(linear), len(linear) + 1],
        )
        metadata = dict(
            available=True,
            arm="baseline",
            stage="actual_public_step_zero",
            train_fingerprint=snapshot["train_fingerprint"],
            settings={
                key: snapshot["settings"][key]
                for key in ("seed", "width", "memory", "ridge")
            },
            dt_s=snapshot["dt_s"],
            history_steps=snapshot["history_steps"],
            horizon_steps=snapshot["horizon_steps"],
            block_boundaries=boundaries,
            initializer_calls=0,
            applied_prior="zero",
            bias_penalized=False,
            reference="actual_public_step_zero_affine_coefficients",
        )
        metadata["settings"]["delay_steps"] = snapshot["delay_steps"]
        return metadata, dict(
            affine_linear=linear,
            affine_bias=bias,
            **{
                f"precursor_norm_{key}": value
                for key, value in snapshot["norms"].items()
            },
        )
    witness = observed["witness"]
    if witness is None:
        return dict(
            available=False,
            reason="initializer_witness_unavailable",
            initializer_calls=observed["calls"],
        ), {}
    _validate_witness(witness, train, recipe)
    rebuilt = _rebuild(
        train, witness["joint_norms"], witness["settings"], witness["history_steps"]
    )
    reference = _reference(witness["precursor"], witness["block_boundaries"])
    prior = reference if captured.arm == "candidate" else np.zeros_like(reference)
    rebuilt["W_affine"] = reference
    rebuilt["centered_rhs"] = (
        rebuilt["B0"] + rebuilt["penalty"] @ prior
        if captured.arm == "candidate"
        else rebuilt["B0"]
    )
    for name, value in witness["captured_fingerprints"].items():
        if value != _fingerprint(rebuilt[name]):
            raise CheckpointError(
                f"actual initializer {name} fingerprint differs from cache reconstruction"
            )
    metadata = {
        key: value
        for key, value in witness.items()
        if key not in ("precursor", "precursor_norms", "joint_norms", "coefficients")
    }
    metadata.update(
        available=True,
        initializer_calls=observed["calls"],
        applied_prior="affine" if captured.arm == "candidate" else "zero",
        bias_penalized=False,
        reference="actual_affine_precursor_distinct_from_applied_prior",
        reconstructed_fingerprints={
            key: _fingerprint(value) for key, value in rebuilt.items()
        },
        matrix_capture_scope="candidate D/Y/A/B0/affine-reference/centered-RHS are actual pre-solve fingerprints"
        if captured.arm == "candidate"
        else "historical D/Y/penalty captured; unnamed A/B0 reconstructed after fitting, not captured",
        backward_residual=_backward(
            rebuilt["A"], witness["coefficients"], rebuilt["centered_rhs"]
        ),
    )
    payload = dict(
        affine_linear=witness["precursor"]["linear"],
        affine_bias=witness["precursor"]["bias"],
        common_affine_reference=reference,
        applied_prior=prior,
        **{
            f"precursor_norm_{key}": value
            for key, value in witness["precursor_norms"].items()
        },
        **{f"joint_norm_{key}": value for key, value in witness["joint_norms"].items()},
    )
    if witness["coefficients"] is not None:
        payload["initial_joint_coefficients"] = witness["coefficients"]
    return metadata, payload


def _coefficient_rows(snapshot, witness):
    result = {}
    for block in BLOCKS:
        if block not in snapshot["params"]:
            result[block] = dict(present=False, reason="absent_in_public_architecture")
            continue
        value = snapshot["params"][block]
        reference = (
            witness[f"affine_{block}"]
            if block in ("linear", "bias")
            else np.zeros_like(value)
        )
        norm = float(np.linalg.norm(value))
        anchor_norm = float(np.linalg.norm(reference))
        displacement = float(np.linalg.norm(value - reference))
        result[block] = dict(
            present=True,
            elements=value.size,
            frobenius_norm=norm,
            coefficient_rms=norm / np.sqrt(value.size),
            affine_reference_frobenius=anchor_norm,
            displacement_frobenius=displacement,
            displacement_rms=displacement / np.sqrt(value.size),
            relative_displacement=None
            if anchor_norm == 0
            else displacement / anchor_norm,
            relative_displacement_unavailable_reason="zero_affine_reference"
            if anchor_norm == 0
            else None,
        )
    return dict(
        step=snapshot["step"],
        blocks=result,
        coordinates="normalized model coefficients; coefficient size is not physical importance",
    )


def save_checkpoints(
    path,
    captured,
    *,
    train,
    development,
    recipe,
    selected_step=None,
    failure=None,
    selected_model=None,
):
    path = Path(path)
    if path.exists():
        raise CheckpointError("anchored checkpoint destination already exists")
    path.mkdir(parents=True)
    evidence = _base.save_checkpoints(
        path / "trajectory",
        captured,
        train=train,
        development=development,
        recipe=recipe,
        selected_step=selected_step,
        failure=failure,
        selected_model=selected_model,
    )
    inner = _read(path / "trajectory/manifest.json")
    started = time.perf_counter()
    metadata, payload = _witness(captured, train, recipe)
    common = {
        key: inner[key]
        for key in (
            "arm",
            "provenance",
            "source_sha256",
            "train_cache",
            "development_cache",
            "recipe",
        )
    }
    save_arrays(
        path / "initializer-witness.npz",
        dict(format=FORMAT, **common, witness=metadata),
        payload,
    )
    coefficients = [
        _coefficient_rows(snapshot, payload) for snapshot in captured.snapshots
    ]
    _write(
        path / "coefficient-diagnostics.json",
        dict(
            format=FORMAT,
            **common,
            applied_prior=metadata.get("applied_prior"),
            affine_reference=metadata.get("reference"),
            checkpoints=coefficients,
        ),
    )
    files = {
        str(file.relative_to(path)): _base._sha(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    outer = {
        key: value
        for key, value in inner.items()
        if key not in ("format", "files", "checkpoints")
    }
    outer.update(
        format=FORMAT,
        trajectory_manifest_sha256=evidence["manifest_sha256"],
        initializer_available=metadata["available"],
        initializer_witness="initializer-witness.npz",
        coefficient_diagnostics="coefficient-diagnostics.json",
        files=files,
        initializer_observation_seconds=captured.initializer_observation["seconds"],
        witness_reconstruction_seconds=time.perf_counter() - started,
        checkpoints=[
            dict(
                step=row["step"],
                snapshot="trajectory/" + row["snapshot"],
                diagnostics="trajectory/" + row["diagnostics"],
            )
            for row in inner["checkpoints"]
        ],
    )
    _write(path / "manifest.json", outer)
    return dict(
        evidence,
        manifest_sha256=_base._sha(path / "manifest.json"),
        initializer_available=metadata["available"],
        coefficient_checkpoints=len(coefficients),
        initializer_observation_seconds=outer["initializer_observation_seconds"],
        witness_reconstruction_seconds=outer["witness_reconstruction_seconds"],
    )


def _load_outer(path, expected_sha256=None):
    path = Path(path)
    if (
        expected_sha256 is not None
        and _base._sha(path / "manifest.json") != expected_sha256
    ):
        raise CheckpointError("anchored checkpoint manifest external anchor mismatch")
    outer = _read(path / "manifest.json")
    if outer["format"] != FORMAT:
        raise CheckpointError("anchored checkpoint format differs")
    if (
        outer["initializer_witness"] != "initializer-witness.npz"
        or outer["coefficient_diagnostics"] != "coefficient-diagnostics.json"
        or any(file.is_symlink() for file in path.rglob("*"))
    ):
        raise CheckpointError("anchored checkpoint paths differ")
    inventory = {
        str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()
    }
    if inventory != set(outer["files"]) | {"manifest.json"}:
        raise CheckpointError("anchored checkpoint inventory changed")
    for name, digest in outer["files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or (path / relative).is_symlink()
            or _base._sha(path / relative) != digest
        ):
            raise CheckpointError("anchored checkpoint payload changed")
    inner = _read(path / "trajectory/manifest.json")
    if (
        _base._sha(path / "trajectory/manifest.json")
        != outer["trajectory_manifest_sha256"]
    ):
        raise CheckpointError("nested checkpoint manifest link differs")
    for key, value in inner.items():
        if key not in ("format", "files", "checkpoints") and outer[key] != value:
            raise CheckpointError(
                "outer checkpoint evidence differs from original trajectory"
            )
    expected_rows = [
        dict(
            step=row["step"],
            snapshot="trajectory/" + row["snapshot"],
            diagnostics="trajectory/" + row["diagnostics"],
        )
        for row in inner["checkpoints"]
    ]
    if outer["checkpoints"] != expected_rows:
        raise CheckpointError("outer checkpoint roster differs")
    return outer, inner


def _restore_witness(metadata, payload):
    return dict(
        {
            key: value
            for key, value in metadata.items()
            if key
            in (
                "binding",
                "arm",
                "stage",
                "captured_fingerprints",
                "captured_local_names",
                "train_fingerprint",
                "settings",
                "dt_s",
                "history_steps",
                "horizon_steps",
                "block_boundaries",
            )
        },
        precursor={"linear": payload["affine_linear"], "bias": payload["affine_bias"]},
        precursor_norms={
            key.removeprefix("precursor_norm_"): value
            for key, value in payload.items()
            if key.startswith("precursor_norm_")
        },
        joint_norms={
            key.removeprefix("joint_norm_"): value
            for key, value in payload.items()
            if key.startswith("joint_norm_")
        },
        coefficients=payload.get("initial_joint_coefficients"),
    )


def _check(
    path,
    *,
    train,
    development,
    provenance,
    source_sha256,
    expected_sha256,
    expectation,
    selected_model=None,
    replay=False,
):
    from types import SimpleNamespace

    path = Path(path)
    outer, inner = _load_outer(path, expected_sha256)
    checker = _base.replay_checkpoints if replay else _base.check_checkpoint_links
    evidence = checker(
        path / "trajectory",
        train=train,
        development=development,
        provenance=provenance,
        source_sha256=source_sha256,
        expected_sha256=outer["trajectory_manifest_sha256"],
        expectation=expectation,
        selected_model=selected_model,
    )
    meta, payload = load_arrays(path / outer["initializer_witness"])
    common = {
        key: inner[key]
        for key in (
            "arm",
            "provenance",
            "source_sha256",
            "train_cache",
            "development_cache",
            "recipe",
        )
    }
    if {key: value for key, value in meta.items() if key != "witness"} != dict(
        format=FORMAT, **common
    ):
        raise CheckpointError("initializer witness provenance differs")
    witness = meta["witness"]
    snapshots = []
    for row in inner["checkpoints"]:
        snapshot_meta, snapshot_arrays = load_arrays(
            path / "trajectory" / row["snapshot"]
        )
        snapshots.append(
            _base._snapshot_from(snapshot_meta["snapshot"], snapshot_arrays)
        )
    if outer["initializer_available"] != witness["available"]:
        raise CheckpointError("initializer witness availability differs")
    if inner["completed"] and not witness["available"]:
        raise CheckpointError("completed fit lacks initializer witness")
    if snapshots and not witness["available"]:
        raise CheckpointError("captured trajectory lacks initializer witness")
    if witness["available"] and witness["arm"] != inner["arm"]:
        raise CheckpointError("initializer arm differs")
    if inner["fitter_calls"] == 0 and witness["initializer_calls"] != 0:
        raise CheckpointError("uncalled fitter has initializer observations")
    if witness["initializer_calls"] not in (0, 1):
        raise CheckpointError("initializer observation count differs")
    observed = dict(calls=witness["initializer_calls"], witness=None)
    if inner["arm"] != "baseline" and witness["available"]:
        if witness["binding"] != _initializer_binding(inner["arm"])[1]:
            raise CheckpointError("initializer source identity differs")
        if witness["train_fingerprint"] != _base._batch_fingerprint(train):
            raise CheckpointError("initializer training cache differs")
        if witness["initializer_calls"] != 1:
            raise CheckpointError("initializer call count differs")
        observed["witness"] = _restore_witness(witness, payload)
    rebuilt_meta, rebuilt_payload = _witness(
        SimpleNamespace(
            arm=inner["arm"], snapshots=snapshots, initializer_observation=observed
        ),
        train,
        inner["recipe"],
    )
    if witness != rebuilt_meta or payload.keys() != rebuilt_payload.keys():
        raise CheckpointError("initializer witness reconstruction metadata differs")
    for key, value in rebuilt_payload.items():
        _exact(payload[key], value, "initializer witness reconstruction array differs")
    if snapshots and inner["arm"] != "baseline":
        initial = snapshots[0]
        coefficients = np.vstack([initial["params"][key] for key in BLOCKS])
        _exact(
            coefficients,
            payload["initial_joint_coefficients"],
            "initial checkpoint differs from executed joint solve",
        )
        for key, value in initial["norms"].items():
            _exact(
                value,
                payload[f"joint_norm_{key}"],
                "initializer/checkpoint normalization differs",
            )
    expected_coefficients = dict(
        format=FORMAT,
        **common,
        applied_prior=witness.get("applied_prior"),
        affine_reference=witness.get("reference"),
        checkpoints=[_coefficient_rows(snapshot, payload) for snapshot in snapshots],
    )
    if _read(path / outer["coefficient_diagnostics"]) != expected_coefficients:
        raise CheckpointError("coefficient displacement diagnostics differ")
    return dict(
        evidence,
        manifest_sha256=expected_sha256,
        initializer_available=witness["available"],
        initializer_witness_arrays=len(payload),
        coefficient_checkpoints=len(snapshots),
        normal_equations_resolved=False,
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
        selected_model=selected_model,
        replay=True,
    )


def check_shared_initialization(paths):
    """Compare available actual precursors after individual anchored validation."""
    if set(paths) != {"baseline", "quadratic", "candidate"}:
        raise CheckpointError("shared initializer arm roster differs")
    available, missing, contracts = {}, {}, {}
    for arm, path in paths.items():
        outer, _ = _load_outer(path)
        metadata, payload = load_arrays(Path(path) / outer["initializer_witness"])
        witness = metadata["witness"]
        if outer["arm"] != arm or (witness["available"] and witness["arm"] != arm):
            raise CheckpointError("shared initializer arm identity differs")
        contracts[arm] = {
            key: outer[key]
            for key in ("source_sha256", "train_cache", "development_cache")
        }
        if witness["available"]:
            available[arm] = (witness, payload)
        else:
            missing[arm] = dict(
                reason=witness["reason"],
                failure=outer["failure"],
                fitter_calls=outer["fitter_calls"],
            )
    comparisons = []
    for a, b in (
        ("baseline", "quadratic"),
        ("baseline", "candidate"),
        ("quadratic", "candidate"),
    ):
        if a not in available or b not in available:
            continue
        left, la = available[a]
        right, ra = available[b]
        if contracts[a] != contracts[b]:
            raise CheckpointError("cross-arm initializer source/cache contract differs")
        for key in (
            "train_fingerprint",
            "settings",
            "dt_s",
            "history_steps",
            "horizon_steps",
        ):
            if left[key] != right[key]:
                raise CheckpointError(
                    "cross-arm affine precursor cache/settings differ"
                )
        if left["block_boundaries"]["linear"] != right["block_boundaries"]["linear"]:
            raise CheckpointError("cross-arm affine feature boundary differs")
        names = [
            "affine_linear",
            "affine_bias",
            *[f"precursor_norm_{key}" for key in BASE_NORMS],
        ]
        if "baseline" not in (a, b):
            if left["block_boundaries"] != right["block_boundaries"]:
                raise CheckpointError("cross-arm expanded feature boundaries differ")
            names += [
                "common_affine_reference",
                *[
                    f"joint_norm_{key}"
                    for key in (*BASE_NORMS, "interaction_scale", "autonomous_scale")
                ],
            ]
        for key in names:
            _exact(
                la[key], ra[key], f"cross-arm actual affine precursor differs: {key}"
            )
        comparisons.append(dict(arms=[a, b], exact=True, arrays=len(names)))
    return dict(
        available_arms=sorted(available),
        unavailable=missing,
        comparisons=comparisons,
        all_available_precursors_exact=True,
        additional_fits_or_solves=0,
    )
