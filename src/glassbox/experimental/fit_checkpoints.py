"""Read-only fitting observations and anchored development-only replay.

This module never initializes, fits or selects a model. Python line observation
copies the current parameter tree at the existing finite checkpoint checks.
The external manifest must itself be anchored by the enclosing run manifest.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import marshal
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import CodeType

import jax
import jax.numpy as jnp
import numpy as np

from glassbox import _sequence_model as public
from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox.experimental import state_input_model as bilinear
from glassbox.experimental import state_quadratic_model as quadratic

ROOT = Path(__file__).resolve().parents[3]
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
GROUPS = {
    "velocity_m_s": (0, 3),
    "body_rate_rad_s": (3, 6),
    "rotation_entries": (6, 15),
}
FORMAT = "glassbox-observed-fit-checkpoints-v1"


class CheckpointError(ValueError):
    """Missing, changed or internally inconsistent observation evidence."""


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arm(arm):
    choices = {
        "baseline": (public.fit_sequence_model, public.SequenceModel, public._rollout),
        "candidate": (
            bilinear.fit_candidate_sequence,
            bilinear.BilinearSequenceModel,
            bilinear._rollout,
        ),
        "quadratic": (
            quadratic.fit_candidate_sequence,
            quadratic.QuadraticSequenceModel,
            quadratic._rollout,
        ),
    }
    if arm not in choices:
        raise CheckpointError("unknown checkpoint arm")
    return choices[arm]


def _sources(source_sha256):
    values = _json(source_sha256)
    required = {
        "src/glassbox/experimental/fit_checkpoints.py",
        "tests/test_fit_checkpoints.py",
        "src/glassbox/_sequence_model.py",
        "src/glassbox/experimental/state_input_model.py",
        "src/glassbox/experimental/state_quadratic_model.py",
    }
    if not isinstance(values, dict) or not required <= values.keys():
        raise CheckpointError("checkpoint source map omits required modules")
    for name, digest in values.items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or _sha(ROOT / path) != digest
        ):
            raise CheckpointError(f"checkpoint source identity changed: {name}")
    return values


def _binding(arm):
    function, _, _ = _arm(arm)
    path = Path(inspect.getsourcefile(function)).resolve()
    source = path.read_text()
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function.__name__
    ]
    if len(definitions) != 1:
        raise CheckpointError("fitting function AST identity is ambiguous")
    definition = definitions[0]
    original = next(
        code
        for code in compile(source, str(path), "exec").co_consts
        if isinstance(code, CodeType) and code.co_name == function.__name__
    )
    if function.__code__ != original:
        raise CheckpointError("fitting function code differs from pinned source")
    initial = [
        node.lineno
        for node in ast.walk(definition)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "trace"
            for target in node.targets
        )
    ]
    checkpoints = [
        node.lineno
        for node in ast.walk(definition)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "trace"
        and node.value.func.attr == "append"
    ]
    if len(initial) != 1 or len(checkpoints) != 1:
        raise CheckpointError("frozen checkpoint AST locations changed")

    def portable(code):
        return code.replace(
            co_filename=str(path.relative_to(ROOT)),
            co_consts=tuple(
                portable(value) if isinstance(value, CodeType) else value
                for value in code.co_consts
            ),
        )

    identity = dict(
        path=str(path.relative_to(ROOT)),
        function=function.__name__,
        module=function.__module__,
        source_sha256=_sha(path),
        code_sha256=hashlib.sha256(marshal.dumps(portable(original))).hexdigest(),
        initial_line=initial[0],
        checkpoint_line=checkpoints[0],
    )
    return function, initial[0], checkpoints[0], identity


def _batch(data):
    return getattr(data, "batch", data)


def _batch_fingerprint(data):
    batch = _batch(data)
    return array_fingerprint(
        {"dt_s": batch.dt_s}, {key: getattr(batch, key) for key in ARRAYS}
    )


def _cache(data):
    batch = _batch(data)
    if not hasattr(data, "keys"):
        raise CheckpointError(
            "checkpoint diagnostics require recording window provenance"
        )
    metadata = dict(
        dt_s=batch.dt_s,
        keys=[asdict(k) for k in data.keys],
        source_origins=list(data.source_origins),
    )
    arrays = {key: getattr(batch, key) for key in ARRAYS}
    for key in ("past_excitation", "future_excitation"):
        if getattr(data, key, None) is not None:
            arrays[key] = getattr(data, key)
    return dict(
        batch_fingerprint=_batch_fingerprint(batch),
        window_fingerprint=array_fingerprint(metadata, arrays),
        metadata=metadata,
    )


def _copy(value):
    result = np.array(value, copy=True)
    if not np.isfinite(result).all():
        raise CheckpointError("nonfinite captured checkpoint array")
    result.setflags(write=False)
    return result


@dataclass
class FitCapture:
    arm: str
    provenance: dict
    source_sha256: dict
    expected_steps: tuple
    binding: dict
    snapshots: list = field(default_factory=list)
    calls: int = 0
    failure: dict | None = None
    observation_seconds: float = 0.0
    completed: bool = False

    def observe(self, local, step):
        # Multiline trace.append creates more than one line event at its head.
        if self.snapshots and self.snapshots[-1]["step"] == step:
            return
        start = time.perf_counter()
        if (
            len(self.snapshots) >= len(self.expected_steps)
            or step != self.expected_steps[len(self.snapshots)]
        ):
            raise CheckpointError("unexpected fitting checkpoint order")
        model = local["model"]
        scalar = float(local["best_loss"] if step == 0 else local["val_loss"])
        if not np.isfinite(scalar):
            raise CheckpointError("checkpoint captured before finite loss check")
        self.snapshots.append(
            dict(
                step=step,
                params={k: _copy(v) for k, v in local["params"].items()},
                norms={k: _copy(v) for k, v in local["norms"].items()},
                error_scale=_copy(local["normalization"]),
                selection_loss=scalar,
                training_batch_mse=None if step == 0 else float(local["train_loss"]),
                kind=model.kind,
                dt_s=model.dt_s,
                history_steps=model.history_steps,
                delay_steps=model.delay_steps,
                horizon_steps=local["validation"].future_states.shape[1],
                train_fingerprint=_batch_fingerprint(local["train"]),
                development_fingerprint=_batch_fingerprint(local["validation"]),
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
        )
        self.observation_seconds += time.perf_counter() - start


@contextmanager
def capture_fit(
    arm, provenance, *, source_sha256, expected_steps=tuple(range(0, 1001, 100))
):
    """Observe exactly one unmodified fit; restore the caller's tracer on all exits.

    A raised exception remains raised and preserves partial observations. A
    preparation exception may precede the fitter (zero calls). A normal exit must
    contain one fitter call and the complete requested checkpoint roster.
    """
    steps = tuple(expected_steps)
    if not steps or steps[0] != 0 or tuple(sorted(set(steps))) != steps:
        raise CheckpointError("invalid expected checkpoint roster")
    sources = _sources(source_sha256)
    function, initial_line, checkpoint_line, identity = _binding(arm)
    result = FitCapture(arm, _json(provenance), sources, steps, identity)
    prior = sys.gettrace()

    def observe(frame, event, argument):
        if event == "line":
            if frame.f_lineno == initial_line:
                result.observe(frame.f_locals, 0)
            elif frame.f_lineno == checkpoint_line:
                result.observe(frame.f_locals, int(frame.f_locals["i"]))
        return observe

    def dispatch(frame, event, argument):
        if event == "call" and frame.f_code is function.__code__:
            result.calls += 1
            if result.calls != 1:
                raise CheckpointError(
                    "multiple optimizer calls in one checkpoint capture"
                )
            return observe
        return prior(frame, event, argument) if prior is not None else None

    try:
        sys.settrace(dispatch)
        yield result
        if result.calls != 1 or [s["step"] for s in result.snapshots] != list(steps):
            raise CheckpointError("completed fit missed required checkpoints")
        result.completed = True
    except BaseException as error:
        result.failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        sys.settrace(prior)


def _snapshot_metadata(snapshot):
    return {
        key: value
        for key, value in snapshot.items()
        if key not in ("params", "norms", "error_scale")
    }


def _snapshot_arrays(snapshot):
    return {
        **{f"param_{key}": value for key, value in snapshot["params"].items()},
        **{f"norm_{key}": value for key, value in snapshot["norms"].items()},
        "error_scale": snapshot["error_scale"],
    }


def _snapshot_from(metadata, arrays):
    if any(
        not k.startswith(("param_", "norm_")) and k != "error_scale" for k in arrays
    ):
        raise CheckpointError("unexpected checkpoint array")
    return {
        **metadata,
        "params": {k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
        "norms": {k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        "error_scale": arrays["error_scale"],
    }


def _diagnostics(snapshot, development, predict):
    batch = _batch(development)
    if batch.future_states.shape[-1] != 15:
        raise CheckpointError(
            "physical checkpoint groups require the frozen 15-channel contract"
        )
    prediction = np.asarray(
        predict(
            snapshot["params"],
            snapshot["norms"],
            *[jnp.asarray(getattr(batch, key)) for key in ARRAYS[:3]],
            snapshot["delay_steps"],
        )
    )
    if (
        prediction.shape != batch.future_states.shape
        or not np.isfinite(prediction).all()
    ):
        raise CheckpointError("nonfinite or malformed development checkpoint forecast")
    squared = (prediction - batch.future_states) ** 2
    normalized = squared / snapshot["error_scale"] ** 2
    contribution = np.sum(normalized, axis=(0, 1)) / normalized.size
    reduced = float(np.sum(contribution))
    measured = snapshot["selection_loss"]
    if not np.isclose(reduced, measured, rtol=1e-10, atol=1e-12):
        raise CheckpointError(
            "checkpoint objective decomposition differs from observed selection loss"
        )
    channels = squared.mean(axis=0)
    groups = list(GROUPS)
    endpoint = np.stack(
        [np.sqrt(channels[:, a:b].mean(axis=-1)) for a, b in GROUPS.values()], axis=-1
    )
    prefix = np.stack(
        [
            np.sqrt(
                np.cumsum(channels[:, a:b].mean(axis=-1))
                / np.arange(1, len(channels) + 1)
            )
            for a, b in GROUPS.values()
        ],
        axis=-1,
    )
    horizons = sorted(
        {
            1,
            *[
                round(t / batch.dt_s)
                for t in (0.05, 0.15, 0.25)
                if np.isclose(round(t / batch.dt_s) * batch.dt_s, t, atol=1e-12)
                and 1 <= round(t / batch.dt_s) <= len(channels)
            ],
        }
    )
    parent_ids = sorted({key.recording_id for key in development.keys})
    parents = np.array([key.recording_id for key in development.keys])
    parent_errors = np.stack(
        [
            np.stack(
                [
                    np.sqrt(
                        squared[parents == parent][:, np.array(horizons) - 1, a:b].mean(
                            axis=(0, 2)
                        )
                    )
                    for a, b in GROUPS.values()
                ],
                axis=-1,
            )
            for parent in parent_ids
        ],
        axis=0,
    )
    group_contribution = [float(contribution[a:b].sum()) for a, b in GROUPS.values()]
    metadata = dict(
        groups=groups,
        parent_ids=parent_ids,
        horizon_steps=horizons,
        horizon_s=[float(round(h * batch.dt_s, 12)) for h in horizons],
        selection_loss=measured,
        reconstructed_selection_loss=reduced,
        absolute_loss_difference=abs(reduced - measured),
        group_loss_contributions=group_contribution,
        windows=len(prediction),
        physical_channels=15,
        evidence="development factual only; existing selection unchanged",
    )
    arrays = dict(
        prediction=prediction,
        channel_mse=channels,
        channel_loss_contributions=contribution,
        group_endpoint_rmse=endpoint,
        group_prefix_rmse=prefix,
        parent_endpoint_rmse=parent_errors,
        parent_endpoint_p95=np.percentile(parent_errors, 95, axis=0),
    )
    return metadata, arrays


def _selected(snapshot, model):
    model = getattr(model, "_model", model)
    if model is None:
        return
    for name in ("kind", "dt_s", "history_steps", "delay_steps"):
        if getattr(model, name) != snapshot[name]:
            raise CheckpointError("selected checkpoint model contract differs")
    for name in ("params", "norms"):
        actual = getattr(model, name)
        if actual.keys() != snapshot[name].keys() or any(
            np.asarray(actual[key]).dtype != value.dtype
            or not np.array_equal(actual[key], value)
            for key, value in snapshot[name].items()
        ):
            raise CheckpointError(
                "selected model differs from captured selected checkpoint"
            )


def _validate_snapshots(snapshots, train, development):
    for snapshot in snapshots:
        if snapshot["train_fingerprint"] != _batch_fingerprint(train) or snapshot[
            "development_fingerprint"
        ] != _batch_fingerprint(development):
            raise CheckpointError("captured fitting cache differs from trusted cache")
        batch = _batch(development)
        if (
            snapshot["dt_s"] != batch.dt_s
            or snapshot["history_steps"] != batch.past_inputs.shape[1]
            or snapshot["horizon_steps"] != batch.future_states.shape[1]
            or not np.isfinite(snapshot["selection_loss"])
            or snapshot["error_scale"].shape
            not in ((batch.future_states.shape[-1],), batch.future_states.shape[1:])
            or not np.isfinite(snapshot["error_scale"]).all()
            or np.any(snapshot["error_scale"] <= 0)
        ):
            raise CheckpointError(
                "captured checkpoint timing or objective contract differs"
            )
        for group in ("params", "norms"):
            if not snapshot[group] or any(
                not np.isfinite(a).all() for a in snapshot[group].values()
            ):
                raise CheckpointError("nonfinite or missing checkpoint arrays")


def _validate_recipe(snapshots, recipe, train, development):
    """Pure cache reductions; no initializer, ridge solve or random draws."""
    batch, dev = _batch(train), _batch(development)
    settings = {
        key: recipe[key]
        for key in (
            "seed",
            "steps",
            "batch_size",
            "learning_rate",
            "width",
            "memory",
            "check_every",
        )
    }
    settings["ridge"] = (
        recipe["ridge_fraction"] * len(batch.past_states) * batch.future_states.shape[1]
    )

    def count(seconds):
        return max(1, int(np.rint(seconds / batch.dt_s)))

    delay = count(recipe["delay_s"])
    history = max(count(recipe["context_s"]), delay + 1)
    horizon = count(recipe["horizon_s"])
    if (
        len(batch.past_states) != recipe["training_windows"]
        or len(dev.past_states) != recipe["development_windows"]
        or batch.past_inputs.shape[1] != history
        or batch.future_states.shape[1] != horizon
    ):
        raise CheckpointError("checkpoint caches differ from expected recipe budgets")
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    current = physical[:, history:-1]
    mean, scale = current.mean((0, 1)), current.std((0, 1))
    scale = np.where(scale > 1e-8, scale, 1)
    hold = np.repeat(batch.past_states[:, -1:], horizon, axis=1)
    expected_scale = np.maximum(
        np.sqrt(np.mean((hold - batch.future_states) ** 2, axis=0)),
        recipe["hold_scale_floor"] * scale,
    )
    for snapshot in snapshots:
        if (
            snapshot["settings"] != settings
            or snapshot["kind"] != recipe["kind"]
            or snapshot["delay_steps"] != delay
            or snapshot["history_steps"] != history
            or snapshot["horizon_steps"] != horizon
        ):
            raise CheckpointError(
                "captured fitting settings differ from expected recipe"
            )
        if (
            not np.array_equal(snapshot["norms"]["state_mean"], mean)
            or not np.array_equal(snapshot["norms"]["state_scale"], scale)
            or not np.array_equal(snapshot["error_scale"], expected_scale)
        ):
            raise CheckpointError(
                "checkpoint normalization or loss scale differs from trusted cache"
            )
        first = snapshots[0]
        if snapshot["norms"].keys() != first["norms"].keys() or any(
            snapshot["norms"][key].dtype != value.dtype
            or not np.array_equal(snapshot["norms"][key], value)
            for key, value in first["norms"].items()
        ):
            raise CheckpointError("checkpoint normalization changed during fitting")


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
    """Write external observations; returns the manifest anchor for the run seal."""
    path = Path(path)
    if path.exists():
        raise CheckpointError("checkpoint destination already exists")
    _sources(captured.source_sha256)
    if failure is None and (not captured.completed or captured.failure is not None):
        raise CheckpointError("failed capture needs explicit failure evidence")
    if failure is not None and captured.completed:
        raise CheckpointError("completed capture cannot be relabeled as failed")
    _validate_snapshots(captured.snapshots, train, development)
    _validate_recipe(captured.snapshots, recipe, train, development)
    if failure is None and selected_step not in [s["step"] for s in captured.snapshots]:
        raise CheckpointError("selected checkpoint missing")
    if failure is not None and (
        selected_step is not None or selected_model is not None
    ):
        raise CheckpointError("failed capture cannot contain a selected model")
    path.mkdir(parents=True)
    train_cache, dev_cache = _cache(train), _cache(development)
    common = dict(
        arm=captured.arm,
        provenance=captured.provenance,
        source_sha256=captured.source_sha256,
        binding=captured.binding,
        train_cache=train_cache,
        development_cache=dev_cache,
        recipe=_json(recipe),
    )
    predict = jax.jit(_arm(captured.arm)[2], static_argnums=(5,))
    files, rows = {}, []
    start = time.perf_counter()
    for snapshot in captured.snapshots:
        step = snapshot["step"]
        snapshot_name, diagnostic_name = (
            f"step-{step:04d}.npz",
            f"step-{step:04d}-development.npz",
        )
        metadata = dict(format=FORMAT, **common, snapshot=_snapshot_metadata(snapshot))
        save_arrays(path / snapshot_name, metadata, _snapshot_arrays(snapshot))
        diagnostic_metadata, diagnostic_arrays = _diagnostics(
            snapshot, development, predict
        )
        save_arrays(
            path / diagnostic_name,
            dict(format=FORMAT, **common, step=step, diagnostics=diagnostic_metadata),
            diagnostic_arrays,
        )
        files[snapshot_name], files[diagnostic_name] = (
            _sha(path / snapshot_name),
            _sha(path / diagnostic_name),
        )
        rows.append(
            dict(step=step, snapshot=snapshot_name, diagnostics=diagnostic_name)
        )
        if step == selected_step:
            _selected(snapshot, selected_model)
    manifest = dict(
        format=FORMAT,
        **common,
        expected_steps=list(captured.expected_steps),
        captured_steps=[s["step"] for s in captured.snapshots],
        unavailable_steps=[
            s
            for s in captured.expected_steps
            if s not in [x["step"] for x in captured.snapshots]
        ],
        fitter_calls=captured.calls,
        optimizer_attempted=captured.calls > 0,
        completed=captured.completed,
        failure=_json(failure),
        observed_exception=captured.failure,
        selected_step=selected_step,
        observation_seconds=captured.observation_seconds,
        timing_scope="observation_seconds measures snapshot copying and fingerprints; general Python tracing dispatch overhead remains in fit wall time and is not isolated",
        diagnostic_seconds=time.perf_counter() - start,
        checkpoints=rows,
        files=files,
    )
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return dict(
        manifest_sha256=_sha(path / "manifest.json"),
        checkpoints=len(rows),
        fitter_calls=captured.calls,
        optimizer_attempted=captured.calls > 0,
        unavailable_checkpoints=len(manifest["unavailable_steps"]),
        observation_seconds=manifest["observation_seconds"],
        diagnostic_seconds=manifest["diagnostic_seconds"],
    )


def _check_checkpoints(
    path,
    *,
    train,
    development,
    provenance,
    source_sha256,
    expected_sha256,
    expectation,
    selected_model=None,
    replay,
):
    path = Path(path)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or _sha(path / "manifest.json") != expected_sha256
    ):
        raise CheckpointError("checkpoint manifest external anchor mismatch")
    manifest = json.loads((path / "manifest.json").read_text())
    expected = _json(expectation)
    required = {
        "arm",
        "recipe",
        "expected_steps",
        "completed",
        "failure",
        "selected_step",
        "trace",
        "status",
    }
    if set(expected) != required:
        raise CheckpointError("incomplete external checkpoint expectations")
    for key in required - {"trace", "status"}:
        if manifest[key] != expected[key]:
            raise CheckpointError(f"checkpoint differs from external {key} expectation")
    if expected["completed"] != (expected["status"] == "complete") or expected[
        "status"
    ] not in ("complete", "fit_failure", "preparation_unavailable"):
        raise CheckpointError("checkpoint completion/outcome status mismatch")
    if expected["status"] == "preparation_unavailable" and (
        manifest["fitter_calls"] != 0 or manifest["captured_steps"]
    ):
        raise CheckpointError("unavailable preparation cannot attempt optimizer")
    if not expected["completed"] and expected["trace"] is not None:
        raise CheckpointError("failed arm cannot have a selected-model trace")
    if "arm" in provenance and provenance["arm"] != manifest["arm"]:
        raise CheckpointError("checkpoint arm differs from external provenance")
    if (
        "checkpoint_steps" in provenance
        and provenance["checkpoint_steps"] != manifest["expected_steps"]
    ):
        raise CheckpointError("checkpoint roster differs from external provenance")
    sources = _sources(source_sha256)
    if (
        manifest["format"] != FORMAT
        or manifest["provenance"] != _json(provenance)
        or manifest["source_sha256"] != sources
        or manifest["train_cache"] != _cache(train)
        or manifest["development_cache"] != _cache(development)
        or manifest["binding"] != _binding(manifest["arm"])[3]
    ):
        raise CheckpointError("checkpoint provenance, sources or cache changed")
    names = set(manifest["files"])
    row_names = [
        row[key]
        for row in manifest["checkpoints"]
        for key in ("snapshot", "diagnostics")
    ]
    if len(row_names) != len(set(row_names)) or set(row_names) != names:
        raise CheckpointError("checkpoint row inventory changed")
    if {p.name for p in path.iterdir()} != names | {"manifest.json"}:
        raise CheckpointError("checkpoint bundle inventory changed")
    for name, digest in manifest["files"].items():
        if (
            Path(name).name != name
            or (path / name).is_symlink()
            or _sha(path / name) != digest
        ):
            raise CheckpointError("checkpoint payload seal mismatch")
    roster = manifest["captured_steps"]
    expected = manifest["expected_steps"]
    if (
        roster != expected[: len(roster)]
        or manifest["unavailable_steps"] != expected[len(roster) :]
        or manifest["optimizer_attempted"] != (manifest["fitter_calls"] > 0)
        or manifest["fitter_calls"] not in (0, 1)
        or [row["step"] for row in manifest["checkpoints"]] != roster
    ):
        raise CheckpointError("checkpoint roster or attempt evidence changed")
    if manifest["completed"] and (
        roster != expected
        or manifest["fitter_calls"] != 1
        or manifest["failure"] is not None
        or manifest["observed_exception"] is not None
        or manifest["selected_step"] not in roster
    ):
        raise CheckpointError("completed checkpoint evidence is incomplete")
    if not manifest["completed"] and (
        manifest["failure"] is None
        or manifest["observed_exception"] is None
        or manifest["selected_step"] is not None
    ):
        raise CheckpointError("failed checkpoint evidence is incomplete")
    common = {
        key: manifest[key]
        for key in (
            "arm",
            "provenance",
            "source_sha256",
            "binding",
            "train_cache",
            "development_cache",
            "recipe",
        )
    }
    predict = jax.jit(_arm(manifest["arm"])[2], static_argnums=(5,)) if replay else None
    array_count = 0
    captured_trace = []
    snapshots = []
    for row in manifest["checkpoints"]:
        metadata, arrays = load_arrays(path / row["snapshot"])
        if {key: value for key, value in metadata.items() if key != "snapshot"} != dict(
            format=FORMAT, **common
        ):
            raise CheckpointError("snapshot provenance changed")
        snapshot = _snapshot_from(metadata["snapshot"], arrays)
        if snapshot["step"] != row["step"]:
            raise CheckpointError("snapshot step mismatch")
        _validate_snapshots([snapshot], train, development)
        snapshots.append(snapshot)
        captured_trace.append(
            dict(
                step=snapshot["step"],
                validation_rollout_mse=snapshot["selection_loss"],
                **(
                    {"training_batch_mse": snapshot["training_batch_mse"]}
                    if snapshot["step"] != 0
                    else {}
                ),
            )
        )
        saved_meta, saved_arrays = load_arrays(path / row["diagnostics"])
        if {
            key: value for key, value in saved_meta.items() if key != "diagnostics"
        } != dict(format=FORMAT, **common, step=row["step"]):
            raise CheckpointError("checkpoint diagnostic metadata changed")
        if replay:
            diag_meta, diag_arrays = _diagnostics(snapshot, development, predict)
            if saved_meta["diagnostics"] != diag_meta:
                raise CheckpointError("checkpoint diagnostic metadata changed")
            if saved_arrays.keys() != diag_arrays.keys() or any(
                saved_arrays[key].dtype != value.dtype
                or not np.array_equal(saved_arrays[key], value)
                for key, value in diag_arrays.items()
            ):
                raise CheckpointError("checkpoint diagnostic replay differs")
        if row["step"] == manifest["selected_step"]:
            _selected(snapshot, selected_model)
        array_count += len(arrays) + len(saved_arrays)
    _validate_recipe(snapshots, expectation["recipe"], train, development)
    if manifest["completed"]:
        if captured_trace != expectation["trace"]:
            raise CheckpointError(
                "captured checkpoint trace differs from selected-model report"
            )
        chosen = min(captured_trace, key=lambda row: row["validation_rollout_mse"])[
            "step"
        ]
        if chosen != manifest["selected_step"]:
            raise CheckpointError(
                "selected checkpoint differs from existing minimum-loss rule"
            )
    return dict(
        exact=True,
        manifest_sha256=expected_sha256,
        checkpoints=len(roster),
        **(
            {"replayed_arrays": array_count}
            if replay
            else {"verified_arrays": array_count}
        ),
        fitter_calls=manifest["fitter_calls"],
        optimizer_attempted=manifest["optimizer_attempted"],
        unavailable_checkpoints=len(manifest["unavailable_steps"]),
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
    """Cheap source/cache/bytes/selection checks, without development rollouts."""
    return _check_checkpoints(
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
    """Verify an externally anchored bundle and replay every development forecast."""
    return _check_checkpoints(
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
