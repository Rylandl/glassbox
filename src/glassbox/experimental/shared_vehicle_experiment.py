"""One frozen physical comparison; source-isolated workers and no-fit replay."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def _source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __package__:
    from . import shared_vehicle_data as common
else:
    common = _source(
        Path(__file__).with_name("shared_vehicle_data.py"), "_shared_vehicle_data"
    )
ROOT, PROTOCOL_PATH, PROTOCOL_SHA256 = (
    common.ROOT,
    common.PROTOCOL_PATH,
    common.PROTOCOL_SHA256,
)
SIMULATORS = common.SIMULATORS
read, write, digest, require, sealed = (
    common.read,
    common.write,
    common.digest,
    common.require,
    common.sealed,
)
resolved = common.resolved
FORMAT = "glassbox-shared-vehicle-experiment-v1"
CANDIDATE_FORMAT = common.CANDIDATE_FORMAT
ARMS = ("baseline", "candidate", "hold")


def _data_module():
    return common


def prepare_reference(simulator, protocol):
    """Decode externally pinned caches; never initialize or forecast here."""
    import numpy as np

    from glassbox._learner_arrays import load_arrays
    from glassbox.experimental.public_mean_saved_port import _windows

    entry = protocol["imported_evidence"]["controls"][simulator]
    metadata, values = load_arrays(
        common.anchor(entry["files"]["actual-preparation.npz"])
    )
    cache_meta, cache_values = load_arrays(common.anchor(entry["model"]))
    require(
        metadata["contract"] == cache_meta["contract"]
        and metadata["seen"] == cache_meta["seen"]
        and metadata["windows"]
        == {
            role: dict(row, dt_s=cache_meta["contract"]["dt_s"])
            for role, row in cache_meta["windows"].items()
        },
        "retained cache metadata",
    )
    for key, value in values.items():
        require(
            key in cache_values
            and value.dtype == cache_values[key].dtype
            and np.array_equal(value, cache_values[key]),
            "retained cache bytes",
        )
    train, development = (
        _windows(metadata, values, role, metadata["contract"]["dt_s"])
        for role in ("train", "development")
    )
    require(
        len(train.keys) == 1536 and len(development.keys) == 256, "frozen cache counts"
    )
    train_ids, dev_ids = (
        {k.recording_id for k in w.keys} for w in (train, development)
    )
    require(
        len(train_ids) == 144
        and len(dev_ids) == 24
        and not train_ids.intersection(dev_ids),
        "frozen recording roles",
    )
    require(set(metadata["seen"]) == train_ids | dev_ids, "cache ledger")
    weights = _npz(common.anchor(entry["files"]["weights.npz"]))
    scale, channels = weights["normalization"], weights["channel_weights"]
    require(
        scale.shape == train.batch.future_states.shape[1:] and channels.shape == (15,),
        "retained objective shapes",
    )
    require(
        scale.dtype == channels.dtype == np.float64
        and np.isfinite(scale).all()
        and np.isfinite(channels).all()
        and (scale > 0).all()
        and (channels > 0).all(),
        "retained objective positivity",
    )
    return SimpleNamespace(
        train=train,
        development=development,
        contract=metadata["contract"],
        seen=metadata["seen"],
        normalization=scale,
        channel_weights=channels,
        source=copy.deepcopy(entry),
        metadata=metadata,
        arrays=values,
    )


def _npz(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)), "duplicate archive entries")
        return {key: data[key] for key in data.files}


def _same_arrays(actual, expected, label):
    import numpy as np

    require(set(actual) == set(expected), label + " roster")
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        require(
            a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=True),
            label + ":" + key,
        )


def _tree(params, norms):
    import numpy as np

    return {
        **{"param_" + k: np.asarray(v) for k, v in params.items()},
        **{"norm_" + k: np.asarray(v) for k, v in norms.items()},
    }


class FitCapture:
    """Persist actual callback values, without changing fitting or selection."""

    def __init__(self, output, *, _expected_steps=1000, _check_every=100):
        self.expected_steps, self.check_every = _expected_steps, _check_every
        self.directory = Path(output)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.handle = (self.directory / "work.jsonl").open("x")

    def __call__(self, event):
        import numpy as np

        from glassbox._learner_arrays import array_fingerprint

        phase = event["phase"]
        row = {"phase": phase}
        for name, value in event.items():
            if name == "phase":
                continue
            if value is None or isinstance(value, (str, bool, int, float, np.generic)):
                value = value.item() if isinstance(value, np.generic) else value
                row[name] = (
                    str(value)
                    if isinstance(value, float) and not np.isfinite(value)
                    else value
                )
        if phase == "started":
            indices = np.asarray(event["indices"])
            row.update(indices=indices.tolist(), indices_dtype=indices.dtype.str)
        if phase == "initialized":
            arrays = event["model"].arrays()
            require(
                not (self.directory / "initial.npz").exists(),
                "duplicate initialization",
            )
            np.savez_compressed(self.directory / "initial.npz", **arrays)
            write(self.directory / "initial.json", event["model"].metadata())
            row["array_fingerprint"] = array_fingerprint({}, arrays)
        if phase == "weights":
            arrays = {
                k: np.asarray(event[k])
                for k in (
                    "normalization",
                    "channel_weights",
                    "initial_training_prediction",
                    "initial_channel_mse",
                    "raw_channel_weights",
                    "fixed_weights",
                    "weight_floor",
                    "weight_normalizer",
                )
                if k in event and event[k] is not None
            }
            require(
                {"normalization", "channel_weights"} <= set(arrays),
                "observed objective weights",
            )
            require(not (self.directory / "weights.npz").exists(), "duplicate weights")
            np.savez_compressed(self.directory / "weights.npz", **arrays)
            row["array_fingerprint"] = array_fingerprint({}, arrays)
        if phase == "checkpoint":
            arrays = _tree(event["params"], event["norms"])
            path = self.directory / f"checkpoint-{event['step']:04d}.npz"
            require(not path.exists(), "duplicate checkpoint")
            np.savez_compressed(path, **arrays)
            row["array_fingerprint"] = array_fingerprint({}, arrays)
        self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()

    def finish(self, model):
        self.close()
        work = verify_capture(
            self.directory,
            model,
            _expected_steps=self.expected_steps,
            _check_every=self.check_every,
        )
        if model is not None:
            _diagnostics(self.directory, model)
        write(self.directory / "work-summary.json", work)
        return work


def _checkpoint_core(model, arrays):
    params = {k[6:]: v for k, v in arrays.items() if k.startswith("param_")}
    norms = {k[5:]: v for k, v in arrays.items() if k.startswith("norm_")}
    core = model._model
    fields = dict(
        dt_s=core.dt_s,
        history_steps=core.history_steps,
        delay_steps=core.delay_steps,
        params=params,
        norms=norms,
    )
    if hasattr(core, "kind"):
        fields["kind"] = core.kind
    return type(core)(**fields)


def verify_capture(
    directory, model, *, replay=False, _expected_steps=1000, _check_every=100
):
    import jax
    import numpy as np

    from glassbox._learner_arrays import array_fingerprint

    directory = Path(directory)
    events = [
        json.loads(line) for line in (directory / "work.jsonl").read_text().splitlines()
    ]
    known = (
        "initializer_started",
        "initializer_entered",
        "ridge_solve",
        "finished",
        "calibrated",
        "initialized",
        "weights",
        "checkpoint",
        "initial_objective",
        "started",
        "proposed",
        "trial",
        "completed",
    )
    phases = {phase: [r for r in events if r["phase"] == phase] for phase in known}
    require(sum(map(len, phases.values())) == len(events), "known work phases")
    for phase, path in (("initialized", "initial.npz"), ("weights", "weights.npz")):
        require(
            len(phases[phase]) <= 1
            and (directory / path).exists() == bool(phases[phase]),
            "phase-bound " + phase + " witness",
        )
        if phases[phase]:
            require(
                array_fingerprint({}, _npz(directory / path))
                == phases[phase][0]["array_fingerprint"],
                "actual " + phase + " witness",
            )
    checkpoints = phases["checkpoint"]
    require(
        [r["step"] for r in checkpoints]
        == (
            list(range(0, _expected_steps + 1, _check_every))
            + ([_expected_steps] if _expected_steps % _check_every else [])
        )[: len(checkpoints)],
        "checkpoint prefix",
    )
    for row in checkpoints:
        require(
            array_fingerprint({}, _npz(directory / f"checkpoint-{row['step']:04d}.npz"))
            == row["array_fingerprint"],
            "checkpoint bytes",
        )
    for phase in ("started", "proposed", "completed"):
        require(
            [r["attempt"] for r in phases[phase]]
            == list(range(1, len(phases[phase]) + 1)),
            "attempt prefix " + phase,
        )
    require(
        len(phases["completed"])
        <= len(phases["proposed"])
        <= len(phases["started"])
        <= _expected_steps,
        "partial work counts",
    )
    if model is not None:
        opt = model.report["optimization"]
        require(
            len(phases["initializer_started"]) + len(phases["initializer_entered"]) == 1
            and len(phases["ridge_solve"]) == 2,
            "actual initializer and two ridge solves",
        )
        weights = _npz(directory / "weights.npz")
        reported_weights = (
            opt["channel_weights"]
            if "channel_weights" in opt
            else opt["objective"]["channel_weights"]
        )
        _same_arrays(
            {
                "normalization": weights["normalization"],
                "channel_weights": weights["channel_weights"],
            },
            {
                "normalization": np.asarray(opt["error_scale"], dtype=np.float64),
                "channel_weights": np.asarray(reported_weights, dtype=np.float64),
            },
            "objective report witness",
        )
        if "fixed_weights" in weights:
            _same_arrays(
                {"weights": weights["channel_weights"]},
                {"weights": weights["fixed_weights"]},
                "fixed loss weights",
            )
        require(
            read(directory / "initial.json") == model._model.metadata(),
            "initial geometry metadata",
        )
        gradient = opt["gradient"]
        for key, value in dict(
            attempts_started=len(phases["started"]),
            gradient_proposal_calls_returned=len(phases["proposed"]),
            completed_acceptance_attempts=len(phases["completed"]),
            known_gradient_window_visits=len(model._train.keys)
            * len(phases["proposed"]),
        ).items():
            require(gradient[key] == value, "observed gradient report: " + key)
        require(
            opt["steps"] == _expected_steps
            and opt["batch_size"] == len(model._train.keys),
            "observed fitting budget",
        )
        require(
            all(
                len(phases[k]) == 1
                for k in ("initialized", "weights", "initial_objective")
            ),
            "completed initialization/weight/objective",
        )
        require(
            all(
                len(phases[k]) == _expected_steps
                for k in ("started", "proposed", "completed")
            )
            and len(checkpoints)
            == len(list(range(0, _expected_steps + 1, _check_every)))
            + bool(_expected_steps % _check_every),
            "complete work roster",
        )
        for row in phases["started"]:
            require(
                row["indices"] == list(range(len(model._train.keys)))
                and row["indices_dtype"] == np.dtype(np.int64).str,
                "full-cache gradient indices",
            )
        require(
            [
                {k: row[k] for k in ("step", "validation_rollout_mse")}
                for row in checkpoints
            ]
            == [
                {k: row[k] for k in ("step", "validation_rollout_mse")}
                for row in opt["trace"]
            ],
            "actual selection trace",
        )
        selected = min(checkpoints, key=lambda row: row["validation_rollout_mse"])
        require(
            selected["step"] == opt["selected_step"], "strict first-minimum selection"
        )
        _same_arrays(
            _npz(directory / f"checkpoint-{selected['step']:04d}.npz"),
            model._model.arrays(),
            "selected model",
        )
        _same_arrays(
            _npz(directory / "initial.npz"),
            _npz(directory / "checkpoint-0000.npz"),
            "actual initialization",
        )
        require(
            1 + len(phases["trial"])
            == opt["safeguard"]["full_training_objective_calls"],
            "objective call count",
        )
        _verify_acceptance(phases)
        if replay:
            weights = _npz(directory / "weights.npz")
            with jax.enable_x64(True):
                for row in checkpoints:
                    core = _checkpoint_core(
                        model, _npz(directory / f"checkpoint-{row['step']:04d}.npz")
                    )
                    b = model._development.batch
                    y = np.asarray(
                        core.rollout(b.past_states, b.past_inputs, b.future_inputs)
                    )
                    loss = float(
                        np.mean(
                            weights["channel_weights"]
                            * ((y - b.future_states) / weights["normalization"]) ** 2
                        )
                    )
                    require(
                        np.isclose(
                            loss, row["validation_rollout_mse"], rtol=1e-8, atol=1e-12
                        ),
                        "checkpoint development objective",
                    )
                _diagnostics(directory, model, replay=True)
    return dict(
        completed=model is not None,
        initializer_returns=len(phases["initialized"]),
        ridge_solve_returns=len(phases["ridge_solve"]),
        gradient_calls_returned=len(phases["proposed"]),
        gradient_attempts_started=len(phases["started"]),
        completed_attempts=len(phases["completed"]),
        gradient_window_visits=sum(
            len(r["indices"]) for r in phases["started"][: len(phases["proposed"])]
        ),
        acceptance_objective_calls_returned=len(phases["initial_objective"])
        + len(phases["trial"]),
        checkpoints=len(checkpoints),
        incomplete_work_unknown=len(phases["started"]) > len(phases["completed"]),
        fits_replayed=0,
    )


def _verify_acceptance(phases):
    import numpy as np

    current = float(phases["initial_objective"][0]["loss"])
    grouped = {}
    for row in phases["trial"]:
        grouped.setdefault(row["attempt"], []).append(row)
    for done in phases["completed"]:
        trials = grouped[done["attempt"]]
        require(
            1 <= len(trials) <= 8
            and [r["trial_index"] for r in trials] == list(range(len(trials))),
            "trial order",
        )
        accepted = None
        for i, row in enumerate(trials):
            require(row["scale"] == 2.0**-i, "trial scales")
            loss = float(row["loss"])
            if np.isfinite(loss) and loss < current:
                require(i == len(trials) - 1, "first strict decrease")
                accepted = row
        if accepted is None:
            require(
                len(trials) == 8
                and done["accepted_scale"] == 0
                and done["accepted_trial_index"] is None,
                "rejected proposal",
            )
        else:
            current = float(accepted["loss"])
            require(
                done["accepted_scale"] == accepted["scale"]
                and done["accepted_trial_index"] == accepted["trial_index"],
                "accepted proposal",
            )
        require(done["loss"] == current, "cached full training loss")


def _diagnostics(directory, model, *, replay=False):
    import jax
    import numpy as np

    from glassbox import learner

    directory = Path(directory)
    arrays, summary = {}, {}
    selected = int(model.report["optimization"]["selected_step"])
    weights = _npz(directory / "weights.npz")
    with jax.enable_x64(True):
        for name, step in (("initial", 0), ("selected", selected)):
            core = _checkpoint_core(
                model, _npz(directory / f"checkpoint-{step:04d}.npz")
            )
            for role, windows in (
                ("train", model._train),
                ("development", model._development),
            ):
                b = windows.batch
                y = np.asarray(
                    core.rollout(b.past_states, b.past_inputs, b.future_inputs)
                )
                arrays[name + "_" + role] = y
                error = (y - b.future_states) ** 2
                summary[name + "_" + role] = dict(
                    weighted_loss=float(
                        np.mean(
                            weights["channel_weights"]
                            * error
                            / weights["normalization"] ** 2
                        )
                    ),
                    channel_rmse=np.sqrt(error.mean(axis=(0, 1))).tolist(),
                    group_rmse={
                        g: float(np.sqrt(error[..., sl].mean()))
                        for g, sl in (
                            ("velocity_m_s", slice(0, 3)),
                            ("body_rate_rad_s", slice(3, 6)),
                            ("rotation_entries", slice(6, 15)),
                        )
                    },
                )
        y = arrays["selected_development"]
        b = model._development.batch
        rank = min(int(np.ceil((len(y) + 1) * learner.ENVELOPE_COVERAGE)), len(y))
        envelope = np.sort(np.abs(y - b.future_states), axis=0)[rank - 1]
        _same_arrays(
            {"envelope": envelope},
            {"envelope": model.envelope()},
            "development calibration",
        )
    if replay:
        _same_arrays(
            arrays, _npz(directory / "diagnostics.npz"), "cache forecast replay"
        )
        require(
            summary == read(directory / "diagnostics.json"), "diagnostic reductions"
        )
    else:
        np.savez_compressed(directory / "diagnostics.npz", **arrays)
        write(directory / "diagnostics.json", summary)
    return summary


def verify_candidate(directory, expected_sha256, protocol, *, replay=False):
    from glassbox.experimental.public_mean_flight_fit import _cache_payload
    from glassbox.experimental.shared_vehicle import SharedVehicleDynamics

    directory = Path(directory)
    run = sealed(directory / "run.json", expected_sha256)
    require(
        run["format"] == CANDIDATE_FORMAT
        and run["status"] in ("complete", "fit_failed"),
        "candidate outcome",
    )
    ref = prepare_reference(run["simulator"], protocol)
    require(
        digest(directory / "preparation.npz")
        == ref.source["files"]["actual-preparation.npz"]["sha256"],
        "actual exact preparation",
    )
    model = (
        SharedVehicleDynamics.load(directory / "model.npz")
        if run["status"] == "complete"
        else None
    )
    if model is not None:
        metadata, values = _cache_payload(
            model._train, model._development, model.contract, model._seen
        )
        require(metadata == ref.metadata, "candidate cache metadata")
        _same_arrays(values, ref.arrays, "candidate cache arrays")
        require(model.report == read(directory / "report.json"), "report mirror")
    else:
        require(
            not (directory / "model.npz").exists(), "failed fit has no selected model"
        )
    if (directory / "capture/weights.npz").exists():
        w = _npz(directory / "capture/weights.npz")
        _same_arrays(
            {
                "normalization": w["normalization"],
                "channel_weights": w["channel_weights"],
            },
            {
                "normalization": ref.normalization,
                "channel_weights": ref.channel_weights,
            },
            "frozen objective weights",
        )
    verify_capture(directory / "capture", model, replay=replay)
    return model


def decide(rows, protocol, *, all_available=True, all_input_finite=True):
    import numpy as np

    from glassbox.experimental import state_input_decision as paired
    from glassbox.experimental.excited_bilinear_decision import _angular_repair
    from glassbox.experimental.two_simulator_metrics import aggregate

    p = resolved(protocol)
    require(set(rows) == set(SIMULATORS), "decision simulator roster")
    for values in rows.values():
        require(set(r["arm"] for r in values) == set(ARMS), "decision arm roster")
    summaries = {sim: aggregate(values) for sim, values in rows.items()}
    for arm in ("candidate", "hold"):
        paired.validate_cohorts(
            {
                sim: [
                    dict(r, arm="candidate" if r["arm"] == arm else "baseline")
                    for r in values
                    if r["arm"] in ("baseline", arm)
                ]
                for sim, values in rows.items()
            },
            p,
        )
    comparison = paired.reduce(summaries, p, rows=rows)
    checks = comparison["checks"]
    checks["response_regression"] = checks.pop("response_gain")
    ratios = comparison["weighted_geometric_mean_ratios"]
    valid = all(
        value is not None and np.isfinite(value) and value > 0
        for value in ratios.values()
    )
    joint = float(np.sqrt(ratios["factual"] * ratios["response"])) if valid else None
    checks["joint_progress"] = bool(
        valid
        and joint
        <= p["decision"]["accept_research_candidate"][
            "joint_equal_kind_geometric_mean_ratio_max"
        ]
    )
    comparison["joint_equal_kind_geometric_mean_ratio"] = joint
    comparison["residual_criteria_pass"] = all(checks.values())
    angular_p = copy.deepcopy(p)
    angular_p["decision"]["angular_repair"] = copy.deepcopy(
        p["decision"]["angular_retention"]
    )
    angular_p["decision"]["angular_repair"]["denominator"] = "quadratic"
    angular_summaries = {
        sim: aggregate(
            [
                dict(r, arm="quadratic" if r["arm"] == "baseline" else r["arm"])
                for r in values
                if r["arm"] in ("baseline", "candidate")
            ]
        )
        for sim, values in rows.items()
    }
    angular = _angular_repair(angular_summaries, angular_p, comparison)
    angular.update(
        denominator_arm="baseline",
        baseline_fields_mean="baseline",
        policy=copy.deepcopy(p["decision"]["angular_retention"]),
    )
    require(
        type(all_available) is bool and type(all_input_finite) is bool,
        "boolean model availability/finiteness",
    )
    checks = dict(
        broad_residual_progress=comparison["residual_criteria_pass"],
        angular_retention=angular["residual_criteria_pass"],
        all_models_available=all_available,
        all_input_eligible_predictions_finite=all_input_finite,
    )
    return dict(
        comparison=comparison,
        angular_retention=angular,
        checks=checks,
        residual_criteria_pass=all(checks.values()),
        public_promotion=False,
        scope="Prepared-cache research residual comparison. No public lifecycle, universal improvement, calibration, derivative or controller qualification.",
    )


def _context(context):
    return dict(
        protocol_path=str(Path(context.get("protocol_path", PROTOCOL_PATH)).resolve()),
        protocol_sha256=context.get("protocol_sha256", PROTOCOL_SHA256),
        binding_path=str(Path(context["binding_path"]).resolve()),
        binding_sha256=context["binding_sha256"],
    )


def _authenticate(request):
    return common.authenticate(
        *(
            request[key]
            for key in (
                "protocol_path",
                "protocol_sha256",
                "binding_path",
                "binding_sha256",
            )
        )
    )


def _supervise(stage, simulator, output, *, context, **inputs):
    request = dict(
        _context(context),
        stage=stage,
        simulator=simulator,
        output=str(Path(output).resolve()),
        **inputs,
    )
    protocol, binding = _authenticate(request)
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "request.json", request)
    environment = dict(os.environ)
    environment.pop("JAX_ENABLE_X64", None)
    root = (
        Path(protocol["imported_evidence"]["updated_source"]["root"])
        if stage == "predict" and inputs["arm"] == "baseline"
        else ROOT
    )
    environment.update(PYTHONPATH=str(root / "src"), SCIPY_ARRAY_API="1")
    command = [
        binding["interpreter"],
        str(Path(__file__).resolve()),
        "--worker",
        str(output / "request.json"),
        "--request-sha256",
        digest(output / "request.json"),
    ]
    write(
        output / "command.json",
        dict(
            argv=command,
            cwd=str(ROOT),
            environment_overrides={
                key: environment.get(key)
                for key in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
            },
            hard_timeout_s=14400,
        ),
    )
    result = dict(
        format=CANDIDATE_FORMAT
        if stage == "fit"
        else "glassbox-shared-vehicle-stage-v1",
        stage=stage,
        simulator=simulator,
        status="incomplete",
        protocol_sha256=request["protocol_sha256"],
        binding_sha256=request["binding_sha256"],
        implementation_commit=binding["implementation_commit"],
        request_sha256=digest(output / "request.json"),
    )
    started = time.monotonic()
    try:
        with (output / "worker.log").open("x") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=False,
            )
        result["returncode"] = process.returncode
        require(
            process.returncode == 0,
            "worker failed; retain evidence without automatic retry",
        )
        outcome = read(output / "outcome.json")
        require(
            outcome["status"] in ("complete", "fit_failed")
            and (stage == "fit" or outcome["status"] == "complete"),
            "worker terminal outcome",
        )
        _authenticate(request)
        result["status"] = outcome["status"]
    except BaseException as error:
        result.update(
            status="hard_timeout_incomplete"
            if isinstance(error, subprocess.TimeoutExpired)
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write(
            output / "exit.json",
            {
                key: result.get(key)
                for key in ("status", "returncode", "elapsed_s", "error_type", "error")
            },
        )
        result["files"] = common.inventory(output)
        write(output / "run.json", result)
    return result


def fit_candidate(simulator, output, **context):
    return _supervise("fit", simulator, output, context=context)


def predict(
    simulator,
    output,
    *,
    arm,
    candidate_root,
    candidate_sha256,
    data_root,
    data_sha256,
    **context,
):
    require(arm in ("baseline", "candidate"), "prediction arm")
    return _supervise(
        "predict",
        simulator,
        output,
        context=context,
        arm=arm,
        candidate_root=str(Path(candidate_root).resolve()),
        candidate_sha256=candidate_sha256,
        data_root=str(Path(data_root).resolve()),
        data_sha256=data_sha256,
    )


def evaluate(
    simulator,
    output,
    *,
    baseline_root,
    baseline_sha256,
    prediction_root,
    prediction_sha256,
    data_root,
    data_sha256,
    **context,
):
    return _supervise(
        "score",
        simulator,
        output,
        context=context,
        baseline_root=str(Path(baseline_root).resolve()),
        baseline_sha256=baseline_sha256,
        prediction_root=str(Path(prediction_root).resolve()),
        prediction_sha256=prediction_sha256,
        data_root=str(Path(data_root).resolve()),
        data_sha256=data_sha256,
    )


def _stage(entry, *, simulator=None, stage=None, binding_sha256=None):
    value = sealed(Path(entry["root"]) / "run.json", entry["sha256"])
    require(value["protocol_sha256"] == PROTOCOL_SHA256, "stage protocol association")
    if simulator is not None:
        require(value["simulator"] == simulator, "stage simulator association")
    if stage is not None:
        require(value["stage"] == stage, "stage kind association")
    if binding_sha256 is not None:
        require(value["binding_sha256"] == binding_sha256, "stage source association")
    require(
        value["status"] == "complete"
        or (value["stage"] == "fit" and value["status"] == "fit_failed"),
        "stage terminal state",
    )
    request = read(Path(entry["root"]) / "request.json")
    require(
        digest(Path(entry["root"]) / "request.json") == value["request_sha256"],
        "stage request hash",
    )
    require(
        request["protocol_sha256"] == PROTOCOL_SHA256
        and request["binding_sha256"] == value["binding_sha256"],
        "stage request source association",
    )
    require(
        Path(request["output"]).resolve() == Path(entry["root"]).resolve(),
        "stage request output association",
    )
    require(request["stage"] == value["stage"], "stage request kind association")
    require(
        request["simulator"] == value["simulator"],
        "stage request simulator association",
    )
    return value, request


def _prediction_payload(request, protocol, *, replay=False):
    import jax
    import numpy as np

    from glassbox.experimental import public_mean_physical_evaluation as physical

    require(
        not jax.config.x64_enabled, "physical predictions require ambient default32"
    )
    _data_module().verify_data(
        request["data_root"],
        request["data_sha256"],
        simulator=request["simulator"],
        protocol=protocol,
    )
    candidate = sealed(
        Path(request["candidate_root"]) / "run.json", request["candidate_sha256"]
    )
    require(
        candidate["format"] == CANDIDATE_FORMAT
        and candidate["simulator"] == request["simulator"]
        and candidate["binding_sha256"] == request["binding_sha256"],
        "prediction exact candidate outcome",
    )
    arm = request["arm"]
    if arm == "baseline":
        from glassbox import LearnedDynamics

        model = LearnedDynamics.load(
            common.anchor(
                protocol["imported_evidence"]["controls"][request["simulator"]]["model"]
            )
        )
    else:
        model = verify_candidate(
            request["candidate_root"], request["candidate_sha256"], protocol
        )
    p = resolved(protocol)
    with jax.enable_x64(True):
        physical.reconstruct_truth(request["data_root"], request["simulator"], p)
    require(not jax.config.x64_enabled, "truth reconstruction must restore precision")
    directory = Path(request["output"])
    arrays = 0

    def save(relative, values):
        nonlocal arrays
        path = directory / relative
        if replay:
            actual = physical.arrays(path)
            require(set(values) == set(actual), "prediction replay array roster")
            for key, value in values.items():
                expected = actual[key]
                require(
                    value.dtype == expected.dtype
                    and value.shape == expected.shape
                    and np.array_equal(value, expected, equal_nan=True),
                    "exact prediction replay: " + str(relative) + "/" + key,
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)
        arrays += len(values)

    save(
        "envelope.npz", {arm: np.asarray(model.envelope())} if model is not None else {}
    )
    fn = jax.jit(model.predict) if model is not None else None
    calls, finite = 0, True
    queries = read(Path(request["data_root"]) / "queries.json")
    for query in queries:
        values = physical.arrays(physical.under(request["data_root"], query["path"]))
        predictions = {}
        suffixes = ("", "_factual") if query["kind"] == "response" else ("",)
        for suffix in suffixes:
            predictions[arm + suffix] = (
                np.full(values["target"].shape, np.nan, dtype=np.float32)
                if fn is None
                else physical.predict_query(
                    fn, query, values, dtype=np.float32, factual=bool(suffix)
                )
            )
            command = values["factual_inputs"] if suffix else values["future_inputs"]
            n = (
                physical.finite_command_prefix(command)
                if query["history_eligible"]
                else 0
            )
            if fn is not None:
                calls += bool(n)
                finite = finite and bool(
                    np.isfinite(predictions[arm + suffix][:n]).all()
                )
            require(
                np.isnan(predictions[arm + suffix][n:]).all(),
                "prediction padding follows input eligibility",
            )
        save(physical.query_path(query), predictions)
    outcome = dict(
        status="complete",
        simulator=request["simulator"],
        arm=arm,
        model_available=model is not None,
        model_fingerprint=model.fingerprint() if model is not None else None,
        queries=len(queries),
        native_forecast_calls=calls,
        all_input_eligible_predictions_finite=finite,
        arrays=arrays,
        fits=0,
        initializers=0,
    )
    if replay:
        require(
            outcome == read(directory / "outcome.json"), "prediction outcome replay"
        )
    else:
        write(directory / "outcome.json", outcome)
    return outcome


def _score_payload(request, protocol, *, replay=False):
    import numpy as np

    from glassbox.experimental import independent_training_experiment as inherited
    from glassbox.experimental import public_mean_physical_evaluation as physical

    _data_module().verify_data(
        request["data_root"],
        request["data_sha256"],
        simulator=request["simulator"],
        protocol=protocol,
    )
    directory = Path(request["output"])
    roots = {
        "baseline": Path(request["baseline_root"]),
        "candidate": Path(request["prediction_root"]),
    }
    outcomes = {}
    for arm, root in roots.items():
        _, req = _stage(
            dict(
                root=str(root),
                sha256=request[
                    "baseline_sha256" if arm == "baseline" else "prediction_sha256"
                ],
            ),
            simulator=request["simulator"],
            stage="predict",
            binding_sha256=request["binding_sha256"],
        )
        require(
            req["arm"] == arm
            and req["data_root"] == request["data_root"]
            and req["data_sha256"] == request["data_sha256"],
            "matched prediction truth and arm",
        )
        outcomes[arm] = read(root / "outcome.json")
    require(
        read(roots["baseline"] / "request.json")["candidate_sha256"]
        == read(roots["candidate"] / "request.json")["candidate_sha256"],
        "shared sealed candidate outcomes",
    )
    queries = read(Path(request["data_root"]) / "queries.json")
    finite = True
    calls = {}
    for arm in roots:
        calls[arm] = 0
    envelopes = {}
    for arm, root in roots.items():
        values = physical.arrays(root / "envelope.npz")
        require(
            set(values) == ({arm} if outcomes[arm]["model_available"] else set()),
            "envelope availability",
        )
        envelopes.update(values)

    def save(relative, values):
        path = directory / relative
        if replay:
            actual = physical.arrays(path)
            require(set(actual) == set(values), "joined prediction roster")
            for key, value in values.items():
                require(
                    value.dtype == actual[key].dtype
                    and np.array_equal(value, actual[key], equal_nan=True),
                    "joined predictions reproduce source arms",
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)

    save("envelopes.npz", envelopes)
    for query in queries:
        values = physical.arrays(physical.under(request["data_root"], query["path"]))
        combined = {}
        for arm, root in roots.items():
            predictions = physical.arrays(root / physical.query_path(query))
            suffixes = ("", "_factual") if query["kind"] == "response" else ("",)
            require(
                set(predictions) == {arm + suffix for suffix in suffixes},
                "raw prediction key roster",
            )
            for suffix in suffixes:
                prediction = predictions[arm + suffix]
                require(
                    prediction.dtype == np.float32
                    and prediction.shape == values["target"].shape,
                    "native32 prediction shape/dtype",
                )
                commands = (
                    values["factual_inputs"] if suffix else values["future_inputs"]
                )
                n = (
                    physical.finite_command_prefix(commands)
                    if query["history_eligible"]
                    else 0
                )
                if outcomes[arm]["model_available"]:
                    calls[arm] += bool(n)
                    finite = finite and bool(np.isfinite(prediction[:n]).all())
                else:
                    require(
                        np.isnan(prediction).all(),
                        "unavailable model prediction padding",
                    )
                require(np.isnan(prediction[n:]).all(), "missing input padding")
            combined.update(predictions)
        save(physical.query_path(query), combined)
    metrics = inherited.score(
        request["data_root"], directory, request["simulator"], resolved(protocol)
    )
    for arm in roots:
        require(
            calls[arm] == outcomes[arm]["native_forecast_calls"],
            "actual forecast call count",
        )
    outcome = dict(
        status="complete",
        simulator=request["simulator"],
        candidate_available=outcomes["candidate"]["model_available"],
        all_input_eligible_predictions_finite=finite,
        fits=0,
        initializers=0,
    )
    if replay:
        require(
            metrics == read(directory / "metrics.json"), "raw physical metric replay"
        )
        require(outcome == read(directory / "outcome.json"), "score outcome replay")
    else:
        write(directory / "metrics.json", metrics)
        write(directory / "outcome.json", outcome)
    return outcome


def _checked(stages, protocol, binding_sha256, *, replay_scores=False):
    require(set(stages) == set(SIMULATORS), "complete simulator stages")
    rows = {}
    available, finite = True, True
    candidate_outcomes = {}
    for sim in SIMULATORS:
        require(
            set(stages[sim])
            == {"candidate", "confirmation", "baseline", "prediction", "evaluation"},
            "complete stage roster",
        )
        entry = stages[sim]["candidate"]
        value, _ = _stage(
            entry, simulator=sim, stage="fit", binding_sha256=binding_sha256
        )
        model = verify_candidate(entry["root"], entry["sha256"], protocol)
        available = available and model is not None
        candidate_outcomes[sim] = dict(
            path=str(Path(entry["root"]).resolve() / "run.json"),
            sha256=entry["sha256"],
            status=value["status"],
        )
    for sim, entries in stages.items():
        entry = entries["confirmation"]
        value = sealed(Path(entry["root"]) / "run.json", entry["sha256"])
        require(
            value["status"] == "complete" and value["simulator"] == sim,
            "confirmation terminal association",
        )
        require(
            value["protocol_sha256"] == PROTOCOL_SHA256
            and value.get("binding_sha256", value.get("implementation_sha256"))
            == binding_sha256,
            "confirmation source association",
        )
        data_root = Path(entry["root"]).resolve() / "data"
        data_sha = value["data_sha256"]
        data = _data_module().verify_data(
            data_root, data_sha, simulator=sim, protocol=protocol
        )
        require(
            data["candidate_outcomes"] == candidate_outcomes,
            "confirmation exact two candidate outcomes",
        )
        for name, arm in (("baseline", "baseline"), ("prediction", "candidate")):
            _, request = _stage(
                entries[name],
                simulator=sim,
                stage="predict",
                binding_sha256=binding_sha256,
            )
            require(
                request["arm"] == arm
                and Path(request["data_root"]) == data_root
                and request["data_sha256"] == data_sha,
                "prediction confirmation association",
            )
            require(
                Path(request["candidate_root"]) == Path(entries["candidate"]["root"])
                and request["candidate_sha256"] == entries["candidate"]["sha256"],
                "prediction candidate association",
            )
        _, request = _stage(
            entries["evaluation"],
            simulator=sim,
            stage="score",
            binding_sha256=binding_sha256,
        )
        for field, target in (("baseline", "baseline"), ("prediction", "prediction")):
            require(
                Path(request[field + "_root"]) == Path(entries[target]["root"])
                and request[field + "_sha256"] == entries[target]["sha256"],
                "evaluation prediction association",
            )
        require(
            Path(request["data_root"]) == data_root
            and request["data_sha256"] == data_sha,
            "evaluation confirmation association",
        )
        outcome = _score_payload(request, protocol, replay=True)
        finite = finite and outcome["all_input_eligible_predictions_finite"]
        rows[sim] = read(Path(entries["evaluation"]["root"]) / "metrics.json")["rows"]
    return rows, bool(available), bool(finite)


def finalize(output, *, stages, **context):
    ctx = _context(context)
    protocol, binding = _authenticate(ctx)
    output = Path(output).resolve()
    require(not (output / "run.json").exists(), "experiment already finalized")
    for entries in stages.values():
        for entry in entries.values():
            require(
                Path(entry["root"]).resolve().is_relative_to(output),
                "experiment contains all stage payloads",
            )
    rows, available, finite = _checked(stages, protocol, ctx["binding_sha256"])
    decision = decide(rows, protocol, all_available=available, all_input_finite=finite)
    write(output / "stage-links.json", stages)
    write(output / "decision.json", decision)
    result = dict(
        format=FORMAT,
        status="complete",
        **ctx,
        implementation_commit=binding["implementation_commit"],
        stages=stages,
        decision_sha256=digest(output / "decision.json"),
        residual_criteria_pass=decision["residual_criteria_pass"],
        public_promotion=False,
        files=common.inventory(output),
    )
    write(output / "run.json", result)
    return result


def verify_bundle(output, expected_sha256, **context):
    ctx = _context(context)
    protocol, binding = _authenticate(ctx)
    output = Path(output)
    run = sealed(output / "run.json", expected_sha256)
    require(
        run["format"] == FORMAT and run["status"] == "complete", "experiment identity"
    )
    require(
        all(run[key] == value for key, value in ctx.items())
        and run["implementation_commit"] == binding["implementation_commit"],
        "experiment source identity",
    )
    stages = read(output / "stage-links.json")
    require(stages == run["stages"], "stage links mirror")
    rows, available, finite = _checked(stages, protocol, ctx["binding_sha256"])
    decision = decide(rows, protocol, all_available=available, all_input_finite=finite)
    require(
        decision == read(output / "decision.json")
        and digest(output / "decision.json") == run["decision_sha256"],
        "frozen decision raw reduction",
    )
    require(
        run["residual_criteria_pass"] == decision["residual_criteria_pass"]
        and run["public_promotion"] is False,
        "research verdict mirror",
    )
    return run


def _replay_process(request, destination):
    """Read the original stage; isolate new command/log/outcome evidence."""
    protocol, binding = _authenticate(request)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    request = dict(request, replay_result=str(destination / "outcome.json"))
    write(destination / "request.json", request)
    environment = dict(os.environ)
    environment.pop("JAX_ENABLE_X64", None)
    root = (
        Path(protocol["imported_evidence"]["updated_source"]["root"])
        if request.get("arm") == "baseline"
        else ROOT
    )
    environment.update(PYTHONPATH=str(root / "src"), SCIPY_ARRAY_API="1")
    command = [
        binding["interpreter"],
        str(Path(__file__).resolve()),
        "--worker",
        str(destination / "request.json"),
        "--request-sha256",
        digest(destination / "request.json"),
    ]
    write(
        destination / "command.json",
        dict(
            argv=command,
            cwd=str(ROOT),
            environment_overrides={
                key: environment.get(key)
                for key in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
            },
            hard_timeout_s=14400,
        ),
    )
    started = time.monotonic()
    result = dict(
        status="incomplete",
        fits=0,
        initializers=0,
        protocol_sha256=PROTOCOL_SHA256,
        binding_sha256=request["binding_sha256"],
    )
    try:
        with (destination / "worker.log").open("x") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=False,
            )
        result["returncode"] = process.returncode
        require(process.returncode == 0, "replay worker failed")
        result["outcome"] = read(destination / "outcome.json")
        _authenticate(request)
        result["status"] = "complete"
    except BaseException as error:
        result.update(
            status="hard_timeout_incomplete"
            if isinstance(error, subprocess.TimeoutExpired)
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write(
            destination / "exit.json",
            {
                key: result.get(key)
                for key in ("status", "returncode", "elapsed_s", "error_type", "error")
            },
        )
        write(
            destination / "run.json", dict(result, files=common.inventory(destination))
        )
    return result


def replay(output, expected_sha256, *, simulator, replay_output, **context):
    require(simulator in SIMULATORS, "replay simulator")
    source = verify_bundle(output, expected_sha256, **context)
    ctx = _context(context)
    destination = Path(replay_output).resolve()
    require(
        not destination.is_relative_to(Path(output).resolve()),
        "replay outside original bundle",
    )
    destination.mkdir(parents=True, exist_ok=False)
    stages = source["stages"][simulator]
    data_stage = read(Path(stages["confirmation"]["root"]) / "run.json")
    data_root = Path(stages["confirmation"]["root"]) / "data"
    outcomes = read(data_root / "seal.json")["candidate_outcomes"]
    native = _data_module().collect_confirmation(
        simulator,
        destination / "native",
        candidate_outcomes=outcomes,
        replay_data=data_root,
        expected_data_sha256=data_stage["data_sha256"],
        **ctx,
    )
    candidate = stages["candidate"]
    fit = _replay_process(
        dict(
            ctx,
            stage="replay-fit",
            simulator=simulator,
            output=candidate["root"],
            candidate_root=candidate["root"],
            candidate_sha256=candidate["sha256"],
        ),
        destination / "fit",
    )
    predictions = {}
    for name in ("baseline", "prediction"):
        request = read(Path(stages[name]["root"]) / "request.json")
        request["stage"] = "replay-predict"
        predictions[name] = _replay_process(request, destination / name)
    verify_bundle(output, expected_sha256, **ctx)
    result = dict(
        status="complete",
        simulator=simulator,
        source_run_sha256=expected_sha256,
        native=native,
        fit=fit,
        predictions=predictions,
        fits=0,
        initializers=0,
        exact=True,
    )
    write(destination / "run.json", dict(result, files=common.inventory(destination)))
    return result


def independent_reduce(output, expected_sha256, *, audit_output, **context):
    """NumPy-only scientific reduction after ordinary source/inventory validation.

    Recomputes every raw endpoint/cumulative MSE and all hierarchical means/tails.
    Bootstrap draws, response-direction diagnostics and policy Boolean logic retain
    their explicitly attributed frozen canonical validators, not a second policy.
    """
    import numpy as np

    from glassbox.experimental import public_mean_physical_evaluation as physical
    from glassbox.experimental.independent_training_experiment import (
        _audit_close,
        _audit_hierarchy,
        _audit_hold,
        _audit_scalar,
    )

    source = verify_bundle(output, expected_sha256, **context)
    protocol, _ = _authenticate(_context(context))
    p = resolved(protocol)
    destination = Path(audit_output).resolve()
    require(
        not destination.is_relative_to(Path(output).resolve()),
        "audit must be outside original bundle",
    )
    destination.mkdir(parents=True, exist_ok=False)
    group_columns = {
        "velocity_m_s": slice(0, 3),
        "body_rate_rad_s": slice(3, 6),
        "rotation_entries": slice(6, 15),
    }
    counts = dict(raw_rows=0, summaries=0, parents=0, cells=0, tails=0)
    independent = {}
    for simulator, stages in source["stages"].items():
        data_root, evaluation = (
            Path(stages[k]["root"]) for k in ("confirmation", "evaluation")
        )
        data_root = data_root / "data"
        scored = read(evaluation / "metrics.json")
        indexed = {}
        for row in scored["rows"]:
            indexed.setdefault((row["parent"], row["query"]), []).append(row)
        rows = []
        for query in read(data_root / "queries.json"):
            values = physical.arrays(physical.under(data_root, query["path"]))
            predictions = physical.arrays(evaluation / physical.query_path(query))
            predictions["hold"] = _audit_hold(query, values)
            target = np.asarray(values["target"], dtype=np.float64)
            if query["kind"] == "response":
                predictions["hold_factual"] = _audit_hold(query, values, True)
                predictions = {
                    a: np.asarray(predictions[a], dtype=np.float64)
                    - np.asarray(predictions[a + "_factual"], dtype=np.float64)
                    for a in ARMS
                }
                target = target - np.asarray(values["factual_target"], dtype=np.float64)
            for row in indexed[query["parent"], query["id"]]:
                eligible, finite, mse = _audit_scalar(
                    predictions[row["arm"]],
                    target,
                    values["valid"],
                    row["horizon_steps"],
                    group_columns[row["group"]],
                    row["statistic"],
                )
                require(
                    (eligible, finite)
                    == (row["truth_eligible"], row["prediction_finite"]),
                    "independent truth/finiteness masks differ",
                )
                _audit_close(mse, row["mse"], "raw " + query["id"])
                rows.append(
                    dict(
                        row, mse=mse, truth_eligible=eligible, prediction_finite=finite
                    )
                )
                counts["raw_rows"] += 1
        independent[simulator] = rows
        grouping = (
            "scope",
            "simulator",
            "kind",
            "arm",
            "horizon_s",
            "group",
            "statistic",
        )
        for kind, extra in (
            ("summaries", ()),
            ("parents", ("cell", "parent")),
            ("cells", ("cell",)),
        ):
            keys = grouping + extra
            groups = {}
            for row in rows:
                groups.setdefault(tuple(row[k] for k in keys), []).append(row)
            require(
                len(groups) == len(scored["summary"][kind]),
                "independent aggregate roster",
            )
            for summary in scored["summary"][kind]:
                selected = groups[tuple(summary[k] for k in keys)]
                _audit_close(
                    _audit_hierarchy(selected), summary["available_truth_rmse"], kind
                )
                require(
                    summary["planned"] == len(selected)
                    and summary["truth_eligible"]
                    == sum(r["truth_eligible"] for r in selected)
                    and summary["failed"]
                    == sum(r["truth_eligible"] and r["mse"] is None for r in selected),
                    "independent aggregate availability differs",
                )
                counts[kind] += 1
    decision = read(Path(output) / "decision.json")
    comparison = decision["comparison"]
    policy = p["decision"]["aggregation"]
    all_rows = [r for rows in independent.values() for r in rows]
    comparison_keys = ("simulator", "scope", "kind", "horizon_s", "group")
    tail_keys = ("simulator", "scope", "kind", "arm", "horizon_s", "group", "statistic")
    comparison_groups, tail_groups = {}, {}
    for row in all_rows:
        tail_groups.setdefault(tuple(row[k] for k in tail_keys), []).append(row)
        if row["statistic"] == "endpoint":
            comparison_groups.setdefault(
                tuple(row[k] for k in comparison_keys), []
            ).append(row)
    ratios = []
    for comparison_row in comparison["physical_comparisons"]:
        selected = comparison_groups[tuple(comparison_row[k] for k in comparison_keys)]
        b, c = (
            _audit_hierarchy([r for r in selected if r["arm"] == arm])
            for arm in ("baseline", "candidate")
        )
        floor = policy["floors"][comparison_row["group"]]
        ratio = None if b is None or c is None else max(c, floor) / max(b, floor)
        for name, value in (
            ("baseline_rmse", b),
            ("candidate_rmse", c),
            ("ratio", ratio),
        ):
            _audit_close(value, comparison_row[name], name)
        ratios.append(dict(comparison_row, ratio=ratio))
    for kind, original in comparison["weighted_geometric_mean_ratios"].items():
        selected = [r for r in ratios if r["kind"] == kind]
        weights = np.asarray(
            [
                policy["simulator_weights"][r["simulator"]]
                * policy["scope_weights"][r["scope"]]
                for r in selected
            ]
        )
        value = (
            None
            if any(r["ratio"] is None for r in selected)
            else float(
                np.exp(
                    np.sum(
                        weights / weights.sum() * np.log([r["ratio"] for r in selected])
                    )
                )
            )
        )
        _audit_close(value, original, "weighted " + kind)
    actual_joint = (
        np.sqrt(np.prod(list(comparison["weighted_geometric_mean_ratios"].values())))
        if all(
            v is not None for v in comparison["weighted_geometric_mean_ratios"].values()
        )
        else None
    )
    _audit_close(
        actual_joint,
        comparison["joint_equal_kind_geometric_mean_ratio"],
        "joint forecast/response",
    )
    for tail in comparison["parent_error_tails"]:
        selected = tail_groups[tuple(tail[k] for k in tail_keys)]
        parents = {}
        for row in selected:
            parents.setdefault((row["cell"], row["parent"]), []).append(row)
        values = [_audit_hierarchy(rows) for rows in parents.values()]
        available = [v for v in values if v is not None]
        for name, value in (
            ("p50", float(np.quantile(available, 0.5)) if available else None),
            ("p95", float(np.quantile(available, 0.95)) if available else None),
            ("max", max(available) if available else None),
        ):
            _audit_close(value, tail[name], "parent " + name)
        require(
            tail["planned_parents"] == len(parents)
            and tail["available_parents"] == len(available),
            "independent parent availability",
        )
        counts["tails"] += 1
    result = dict(
        source_run_sha256=expected_sha256,
        passed=True,
        counts=counts,
        comparison_ratios=comparison["weighted_geometric_mean_ratios"],
        fits=0,
        predictions=0,
        initializers=0,
        scope="Independent NumPy raw MSE, matched masks, hierarchy, parent quantiles and aggregate ratios. Frozen canonical validators separately check directions, bootstrap and policy.",
    )
    write(destination / "result.json", result)
    write(destination / "run.json", dict(result, files=common.inventory(destination)))
    sealed(Path(output) / "run.json", expected_sha256)
    return result


def _worker(request_path, request_sha256):
    require(digest(request_path) == request_sha256, "external worker request")
    request = read(request_path)
    protocol, _ = _authenticate(request)
    import jax
    import numpy as np

    import glassbox

    expected = (
        Path(protocol["imported_evidence"]["updated_source"]["root"])
        if request.get("arm") == "baseline"
        else ROOT
    )
    require(
        Path(glassbox.__file__).resolve() == expected / "src/glassbox/__init__.py",
        "isolated worker Glassbox source",
    )
    require(
        not jax.config.x64_enabled and jax.default_backend() == "cpu",
        "ambient32 CPU worker",
    )
    stage = request["stage"]
    if stage == "fit":
        import shutil

        from glassbox._sequence_model import SequenceFitError
        from glassbox.experimental import shared_vehicle

        ref = prepare_reference(request["simulator"], protocol)
        output = Path(request["output"])
        shutil.copy2(
            common.anchor(ref.source["files"]["actual-preparation.npz"]),
            output / "preparation.npz",
        )
        capture = FitCapture(output / "capture")
        try:
            model = shared_vehicle._train(
                ref.train,
                ref.development,
                ref.contract,
                ref.seen,
                error_scale=ref.normalization,
                channel_weights=ref.channel_weights,
                observer=capture,
            )
            model.save(output / "model.npz")
            write(output / "report.json", model.report)
            work = capture.finish(model)
            outcome = dict(
                status="complete",
                simulator=request["simulator"],
                model_fingerprint=model.fingerprint(),
                selected_step=model.report["optimization"]["selected_step"],
                work=work,
                operation="research_prepared_cache_fit",
            )
        except (SequenceFitError, np.linalg.LinAlgError, FloatingPointError) as error:
            capture.close()
            status = (
                "hard_timeout_incomplete"
                if "fit_time_limit" in str(error)
                else "fit_failed"
            )
            outcome = dict(
                status=status,
                simulator=request["simulator"],
                error_type=type(error).__name__,
                error=str(error),
                work=verify_capture(output / "capture", None),
            )
            if status == "hard_timeout_incomplete":
                write(output / "outcome.json", outcome)
                raise TimeoutError(
                    "internal fit deadline; confirmation aborted"
                ) from error
        finally:
            capture.close()
        write(output / "outcome.json", outcome)
    elif stage in ("predict", "replay-predict"):
        outcome = _prediction_payload(
            request, protocol, replay=stage == "replay-predict"
        )
    elif stage == "score":
        outcome = _score_payload(request, protocol)
    elif stage == "replay-fit":
        model = verify_candidate(
            request["candidate_root"],
            request["candidate_sha256"],
            protocol,
            replay=True,
        )
        outcome = dict(
            status="complete", model_available=model is not None, fits=0, initializers=0
        )
    else:
        raise common.IntegrityError("unknown worker stage")
    _authenticate(request)
    if request.get("replay_result"):
        write(request["replay_result"], outcome)
    return outcome


def _reject(function, phrase):
    try:
        function()
    except (ValueError, AssertionError) as error:
        require(phrase in str(error), "alteration reached wrong check: " + str(error))
        return dict(rejected=True, error=str(error))
    raise common.IntegrityError("alteration was accepted")


def alteration_audit(output, expected_sha256, *, audit_output, **context):
    """Exactly the four frozen challenges, on disposable copies only."""
    import shutil

    import numpy as np

    from glassbox._learner_arrays import load_arrays, save_arrays

    source = verify_bundle(output, expected_sha256, **context)
    protocol, _ = _authenticate(_context(context))
    destination = Path(audit_output).resolve()
    require(
        not destination.is_relative_to(Path(output).resolve()),
        "audit outside source bundle",
    )
    destination.mkdir(parents=True, exist_ok=False)
    sim = "crazyflow"
    stages = source["stages"][sim]
    results = []
    candidate = Path(stages["candidate"]["root"])
    if read(candidate / "run.json")["status"] == "complete":
        copied = destination / "SVP-A01"
        shutil.copytree(candidate, copied)
        meta, arrays = load_arrays(copied / "model.npz")
        key = next(k for k in arrays if k.startswith("param_") and arrays[k].size)
        arrays[key] = arrays[key].copy()
        arrays[key].flat[0] = np.nextafter(arrays[key].flat[0], np.inf)
        save_arrays(copied / "model.npz", meta, arrays)
        external = _reject(
            lambda: sealed(copied / "run.json", stages["candidate"]["sha256"]),
            "inventory",
        )
        run = read(copied / "run.json")
        run["files"] = common.inventory(copied)
        (copied / "run.json").unlink()
        write(copied / "run.json", run)
        semantic = _reject(
            lambda: verify_candidate(
                copied, digest(copied / "run.json"), protocol, replay=True
            ),
            "selected model",
        )
        results.append(
            dict(
                id="SVP-A01",
                external=external,
                semantic=semantic,
                clean_validator_passed=True,
            )
        )
    else:
        results.append(
            dict(id="SVP-A01", unavailable="candidate fit_failed; no selected model")
        )
    copied = destination / "SVP-A02"
    shutil.copytree(candidate, copied)
    meta, arrays = load_arrays(copied / "preparation.npz")
    arrays["train_future_inputs"] = arrays["train_future_inputs"].copy()
    arrays["train_future_inputs"].flat[0] = np.nextafter(
        arrays["train_future_inputs"].flat[0], np.inf
    )
    save_arrays(copied / "preparation.npz", meta, arrays)
    external = _reject(
        lambda: sealed(copied / "run.json", stages["candidate"]["sha256"]), "inventory"
    )
    run = read(copied / "run.json")
    run["files"] = common.inventory(copied)
    (copied / "run.json").unlink()
    write(copied / "run.json", run)
    semantic = _reject(
        lambda: verify_candidate(copied, digest(copied / "run.json"), protocol),
        "exact preparation",
    )
    results.append(
        dict(
            id="SVP-A02",
            external=external,
            semantic=semantic,
            clean_validator_passed=True,
        )
    )
    original_data = Path(stages["confirmation"]["root"]) / "data"
    copied = destination / "SVP-A03"
    shutil.copytree(original_data, copied)
    record = next(
        r
        for r in read(copied / "records.json")
        if (copied / (r["prefix"] + ".npz")).exists()
    )
    target = copied / (record["prefix"] + ".npz")
    arrays = _npz(target)
    key = "commands" if "commands" in arrays else "states"
    arrays[key] = arrays[key].copy()
    arrays[key].flat[0] = np.nextafter(arrays[key].flat[0], np.inf)
    np.savez_compressed(target, **arrays)
    original_sha = digest(original_data / "seal.json")
    external = _reject(
        lambda: common.verify_data(copied, original_sha, protocol=protocol), "inventory"
    )
    seal = read(copied / "seal.json")
    seal["files"] = common.inventory(copied, "seal.json")
    (copied / "seal.json").unlink()
    write(copied / "seal.json", seal)
    # The ordinary data validator must accept a repaired local inventory first.
    common.verify_data(copied, digest(copied / "seal.json"), protocol=protocol)
    replay_dir = destination / "SVP-A03-native"
    try:
        common.collect_confirmation(
            sim,
            replay_dir,
            candidate_outcomes=seal["candidate_outcomes"],
            replay_data=copied,
            expected_data_sha256=digest(copied / "seal.json"),
            **_context(context),
        )
    except common.IntegrityError:
        require(
            "fresh array bytes differ" in (replay_dir / "worker.log").read_text(),
            "native alteration reached exact array check",
        )
        semantic = dict(rejected=True, error="fresh array bytes differ")
    else:
        raise common.IntegrityError("native truth alteration accepted")
    results.append(
        dict(
            id="SVP-A03",
            external=external,
            semantic=semantic,
            clean_validator_passed=True,
        )
    )
    decision = copy.deepcopy(read(Path(output) / "decision.json"))
    decision["residual_criteria_pass"] = not decision["residual_criteria_pass"]
    write(destination / "SVP-A04-decision.json", decision)
    rows, available, finite = _checked(
        source["stages"], protocol, context["binding_sha256"]
    )
    fresh = decide(rows, protocol, all_available=available, all_input_finite=finite)
    require(fresh == read(Path(output) / "decision.json"), "clean decision replay")
    semantic = _reject(
        lambda: require(fresh == decision, "frozen decision raw reduction"),
        "frozen decision raw reduction",
    )
    results.append(dict(id="SVP-A04", semantic=semantic, clean_validator_passed=True))
    sealed(Path(output) / "run.json", expected_sha256)
    result = dict(
        source_run_sha256=expected_sha256,
        cases=results,
        passed=all(
            "unavailable" not in row and row["semantic"]["rejected"] for row in results
        ),
        fits=0,
    )
    write(destination / "run.json", dict(result, files=common.inventory(destination)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--request-sha256")
    parser.add_argument(
        "--stage",
        choices=(
            "bind",
            "fit",
            "collect",
            "predict",
            "score",
            "finalize",
            "verify",
            "replay",
            "independent",
            "alter",
        ),
    )
    parser.add_argument(
        "--request", help="Externally anchored JSON with function arguments"
    )
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args.worker, args.request_sha256)
    require(
        args.request is not None and args.request_sha256 is not None,
        "external stage request required",
    )
    require(digest(args.request) == args.request_sha256, "external CLI request")
    request = read(args.request)
    functions = dict(
        bind=common.create_binding,
        fit=fit_candidate,
        collect=common.collect_confirmation,
        predict=predict,
        score=evaluate,
        finalize=finalize,
        verify=verify_bundle,
        replay=replay,
        independent=independent_reduce,
        alter=alteration_audit,
    )
    return functions[args.stage](**request)


if __name__ == "__main__":
    main()
