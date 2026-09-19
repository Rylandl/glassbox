"""Frozen full-cache gradient study, reusing pinned historical harness helpers."""

from __future__ import annotations

import argparse
import importlib.util
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import excited_public_matching as _shared_matching
from glassbox.experimental import initial_channel_balance_matching as _balanced_matching
from glassbox.experimental.excited_bilinear_decision import _angular_repair, _comparison
from glassbox.experimental.initial_channel_balance_decision import _full_cohorts
from glassbox.experimental.state_input_model import _window_fingerprint
from glassbox.experimental.two_simulator_metrics import aggregate

from . import initial_channel_balance_experiment as shared_runner
from . import safeguarded_adam_experiment as previous
from . import state_input_experiment as inherited
from .affine_anchored_experiment import _checkpoint_sources

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/full-cache-gradient-v1.json"
PROTOCOL_SHA256 = "fe318d07998e0f05385e421a930f172453641f30e020724550d8811d7757976d"
SIMULATORS = ("crazyflow", "cascade")
REFERENCE_ARMS = {
    "baseline": "baseline",
    "balanced": "balanced",
    "safeguarded": "candidate",
}
ARMS = (*REFERENCE_ARMS, "candidate")
PREDICTORS = (*ARMS, "hold")
digest, read_json, write_json = previous.digest, previous.read_json, previous.write_json
load_arrays, save_arrays = previous.load_arrays, previous.save_arrays
array_equal, freeze_files, verify_files = (
    previous.array_equal,
    previous.freeze_files,
    previous.verify_files,
)


def delta_protocol():
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen full-cache gradient protocol changed")
    value = read_json(PROTOCOL)
    for entry in (
        value["base_protocol"],
        value["prior_result"],
        value["cohort"]["original_protocol"],
    ):
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(
                "historical protocol or evidence changed: " + entry["path"]
            )
    return value


def _test_roster(rows, shift):
    result = []
    for row in rows:
        if row["role"] != "test":
            continue
        if row["id"].rsplit("/", 1)[-1] != "test-" + str(row["seed"]):
            raise ValueError("historical test identity is malformed")
        seed = row["seed"] + shift
        result.append(
            (row["simulator"], row["id"].rsplit("-", 1)[0] + "-" + str(seed), seed)
        )
    return result


def protocol():
    delta, base = delta_protocol(), previous.protocol()
    value = {
        key: deepcopy(base[key]) for key in delta["resolution"]["inherit_unchanged"]
    }
    value.update(
        {
            key: deepcopy(item)
            for key, item in base.items()
            if key.startswith(("prior_", "previous_"))
        }
    )
    for key in (
        "id",
        "status",
        "base_commit",
        "named_gap",
        "scope",
        "prior_reference_bundle",
        "prior_result",
        "optimizer_evidence",
        "verification",
    ):
        value[key] = deepcopy(delta[key])
    value["counts"], value["recordings"] = (
        deepcopy(base["counts"]),
        deepcopy(base["recordings"]),
    )
    original = read_json(ROOT / delta["cohort"]["original_protocol"]["path"])
    if set(_test_roster(base["recordings"], 0)) != set(
        _test_roster(original["recordings"], 8000000)
    ):
        raise ValueError("resolved base is not the declared preceding test cohort")
    for row in value["recordings"]:
        if row["role"] == "test":
            _, row["id"], row["seed"] = _test_roster(
                [row], delta["cohort"]["test_seed_increment_from_base"]
            )[0]
    roster = _test_roster(value["recordings"], 0)
    if set(roster) != set(
        _test_roster(
            original["recordings"], delta["cohort"]["test_seed_shift_from_original"]
        )
    ):
        raise ValueError("fresh test cohort differs from declared original shift")
    ids, seeds = {r[1] for r in roster}, {r[2] for r in roster}
    if (
        len(roster) != 168
        or len(ids) != 168
        or len({(r[0], r[2]) for r in roster}) != 168
        or len(seeds) != 84
    ):
        raise ValueError("fresh test identity or seed roster differs")
    for shift in delta["cohort"]["prior_test_seed_shifts_from_original"]:
        old = _test_roster(original["recordings"], shift)
        if ids & {r[1] for r in old} or seeds & {r[2] for r in old}:
            raise ValueError(
                "fresh test identities or seeds overlap an inspected cohort"
            )
    value["fresh_confirmation"] = deepcopy(delta["cohort"])
    value["imported_references"] = deepcopy(delta["imported_references"])
    value["trusted_comparator_revisions"] = deepcopy(
        delta["imported_references"]["trusted_comparator_revisions"]
    )
    value["fitting"] = deepcopy(delta["fitting"])
    for key in ("generic_recipe", "normalization", "mechanism"):
        value["fitting"][key] = deepcopy(base["fitting"][key])
    value["fitting"]["arms"] = list(ARMS)
    value["checkpoint_diagnostics"] = {
        **deepcopy(base["checkpoint_diagnostics"]),
        **deepcopy(delta["checkpoint_diagnostics"]),
    }
    diagnostics = value["checkpoint_diagnostics"]
    diagnostics.update(
        source_binding=delta["verification"]["source_binding"],
        initialization=delta["fitting"]["initialization"],
        observer_parity=delta["verification"]["tests"],
        capture=delta["optimizer_evidence"]["observation"],
        replay_integrity=delta["verification"]["replay"],
    )
    value["decision"] = {
        key: deepcopy(base["decision"][key])
        for key in ("aggregation", "metrics", "tails", "uncertainty", "availability")
    }
    value["decision"].update(deepcopy(delta["decision"]))
    value["decision"]["accept_research_candidate"] = deepcopy(
        base["decision"]["accept_research_candidate"]
    )
    value["decision"]["comparisons"] = {
        name: {
            **deepcopy(delta["decision"]["guards_for_every_required_comparison"]),
            **deepcopy(policy),
        }
        for name, policy in delta["decision"]["comparisons"].items()
    }
    value["decision"]["angular_repair"] = deepcopy(
        delta["decision"]["angular_retention"]
    )
    value["decision"]["public_promotion"] = delta["decision"]["qualification"]
    value["inherited_source_sha256"].update(previous._sources())
    value["inherited_source_sha256"][delta["base_protocol"]["path"]] = delta[
        "base_protocol"
    ]["sha256"]
    return value


def _load_private(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _parent():
    """Isolated historical helpers whose dependencies point to this frozen study."""
    module = _load_private(
        "glassbox.experimental._full_cache_parent_runner", previous.__file__
    )
    module.previous = previous
    module.PROTOCOL, module.PROTOCOL_SHA256 = PROTOCOL, PROTOCOL_SHA256
    module.REFERENCE_ARMS, module.ARMS, module.PREDICTORS = (
        REFERENCE_ARMS,
        ARMS,
        PREDICTORS,
    )
    reused = (
        "resolved_digest",
        "resolved_protocol",
        "trusted_references",
        "_reference_sources",
        "_reference_manifest",
        "import_references",
        "check_reference_import",
        "generate",
        "data_ready",
        "fit_provenance",
        "anchor_reference",
        "check_checkpoints",
        "check_links",
        "evaluate",
        "validate_rows",
        "verify_bundle",
        "replay",
    )
    module._native = {name: getattr(module, name) for name in reused}
    for name in (
        *reused,
        "protocol",
        "delta_protocol",
        "_sources",
        "_implementation",
        "_runner",
        "_model_classes",
        "check_sources",
        "compare_candidate",
        "decision",
        "Predictors",
    ):
        setattr(module, name, globals()[name])
    return module


def resolved_digest():
    return _parent()._native["resolved_digest"]()


def resolved_protocol(output, *, create=False):
    return _parent()._native["resolved_protocol"](output, create=create)


def _sources():
    paths = [
        ROOT / folder / (prefix + "full_cache_gradient_" + stem + ".py")
        for stem in ("model", "checkpoints", "experiment")
        for folder, prefix in (("src/glassbox/experimental", ""), ("tests", "test_"))
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("full-cache gradient implementation or tests missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


@lru_cache(maxsize=1)
def _implementation():
    module = _load_private(
        "glassbox.experimental._full_cache_interaction", inherited.__file__
    )
    module.PROTOCOL, module.PROTOCOL_SHA256, module.protocol = (
        PROTOCOL,
        PROTOCOL_SHA256,
        protocol,
    )
    module.ARMS, module.PREDICTORS = ARMS, PREDICTORS
    private_flight = module.flight()
    original_read, original_check = (
        private_flight.read_json,
        private_flight.check_sources,
    )
    private_flight.read_json = lambda path: (
        protocol() if Path(path) == PROTOCOL else original_read(path)
    )

    def check(p):
        if p != protocol():
            raise ValueError("resolved protocol identity differs")
        runtime = original_check(p)
        for path, expected in p["inherited_source_sha256"].items():
            if digest(ROOT / path) != expected:
                raise ValueError("inherited source changed: " + path)
        runtime["full_cache_gradient_implementation"] = _sources()
        runtime["checkpoint_source_sha256"] = _checkpoint_sources(
            p, {**_sources(), str(PROTOCOL.relative_to(ROOT)): PROTOCOL_SHA256}
        )
        runtime["resolved_protocol_sha256"] = resolved_digest()
        return runtime

    private_flight.check_sources = check
    module._original_predict = module.Predictors.predict
    module.check_links = lambda *a, **kw: check_links(*a, **kw)
    module.Predictors = lambda *a: Predictors(*a)
    module.decision = lambda *a: decision(*a)
    return module


@lru_cache(maxsize=1)
def _runner():
    module = _load_private(
        "glassbox.experimental._full_cache_collection_runner", shared_runner.__file__
    )
    module.PROTOCOL, module.PROTOCOL_SHA256, module.protocol = (
        PROTOCOL,
        PROTOCOL_SHA256,
        protocol,
    )
    module.REFERENCE_ARMS, module.ARMS, module.PREDICTORS = (
        REFERENCE_ARMS,
        ARMS,
        PREDICTORS,
    )
    for name in (
        "_implementation",
        "_model_classes",
        "check_sources",
        "check_reference_import",
        "check_checkpoints",
        "fit_provenance",
        "compare_candidate",
    ):
        setattr(module, name, globals()[name])
    return module


def flight():
    return _implementation().flight()


def check_sources():
    return _implementation().check_sources()


def _model_classes():
    from ..learner import LearnedDynamics
    from .full_cache_gradient_model import CandidateDynamics as FullCache
    from .initial_channel_balance_model import CandidateDynamics as Balanced
    from .safeguarded_adam_model import CandidateDynamics as Safeguarded

    return dict(
        baseline=LearnedDynamics,
        balanced=Balanced,
        safeguarded=Safeguarded,
        candidate=FullCache,
    )


def trusted_references():
    return _parent()._native["trusted_references"]()


def _reference_sources(simulator):
    return _parent()._native["_reference_sources"](simulator)


def _reference_manifest(*args):
    return {
        **_parent()._native["_reference_manifest"](*args),
        "format": "glassbox-full-cache-reference-import-v1",
    }


def import_references(simulator, output, runtime):
    return _parent()._native["import_references"](simulator, output, runtime)


def check_reference_import(simulator, output, runtime):
    return _parent()._native["check_reference_import"](simulator, output, runtime)


def generate(simulator, output):
    return _parent()._native["generate"](simulator, output)


def data_ready(output):
    return _parent()._native["data_ready"](output)


def fit_provenance(simulator, arm, output, runtime):
    return {
        **_parent()._native["fit_provenance"](simulator, arm, output, runtime),
        "optimizer": "ordered_full_cache_gradient_safeguarded_adam",
        "gradient_sampling": "ordered_full_cache",
        "gradient_training_windows": 384,
    }


def anchor_reference(simulator, output):
    return _parent()._native["anchor_reference"](simulator, output)


def check_checkpoints(output, simulator, arm, *, replay=False):
    if arm not in ARMS:
        raise ValueError("unknown checkpoint arm")
    if arm in REFERENCE_ARMS:
        return _parent()._native["check_checkpoints"](
            output, simulator, arm, replay=replay
        )
    from . import full_cache_gradient_checkpoints as checkpoints
    from .full_cache_gradient_model import RECIPE

    base = Path(output) / simulator
    directory = base / arm
    outcome = read_json(directory / "outcome.json")
    model = (
        _model_classes()[arm].load(directory / "model.npz")
        if outcome["model_available"]
        else None
    )
    expectation = _runner()._outcome_expectation(outcome, model, arm, RECIPE)
    optimization = None if model is None else model.report["optimization"]
    for expected, key in (
        ("unweighted_trace", "unweighted_development_trace"),
        ("objective", "objective"),
        ("safeguard", "safeguard"),
        ("gradient", "gradient"),
    ):
        expectation[expected] = None if optimization is None else optimization[key]
    reference = _model_classes()["baseline"].load(base / "baseline/model.npz")
    runtime = check_sources()
    checker = (
        checkpoints.replay_checkpoints if replay else checkpoints.check_checkpoint_links
    )
    return checker(
        directory / "checkpoints",
        expected_sha256=outcome["checkpoint_manifest_sha256"],
        expectation=expectation,
        train=reference._train,
        development=reference._development,
        provenance=fit_provenance(simulator, arm, output, runtime),
        source_sha256=runtime["checkpoint_source_sha256"],
        selected_model=None if model is None else model._model,
        anchor_reference=anchor_reference(simulator, output),
    )


def fit_arm(simulator, arm, output):
    from ..learner import RECIPE as PUBLIC_RECIPE
    from .full_cache_gradient_checkpoints import capture_fit, save_checkpoints
    from .full_cache_gradient_model import RECIPE, CandidateFitError, fit_candidate

    if arm != "candidate" or PUBLIC_RECIPE != protocol()["fitting"]["generic_recipe"]:
        raise ValueError("only frozen candidate fitting is permitted")
    runtime = data_ready(output)
    base = Path(output) / simulator
    for reference_arm in REFERENCE_ARMS:
        check_checkpoints(output, simulator, reference_arm)
    directory = base / arm
    directory.mkdir(parents=True, exist_ok=False)
    provenance = fit_provenance(simulator, arm, output, runtime)
    write_json(directory / "start.json", provenance)
    reference = _model_classes()["baseline"].load(base / "baseline/model.npz")
    start = time.perf_counter()
    model, reason, status = None, None, "complete"
    capture = capture_fit(
        arm,
        provenance,
        source_sha256=runtime["checkpoint_source_sha256"],
        expected_steps=tuple(protocol()["checkpoint_diagnostics"]["steps"]),
    )
    try:
        with capture as captured:
            model = fit_candidate(reference)
    except CandidateFitError as error:
        reason, status = str(error), "fit_failure"
    fit_wall_time = time.perf_counter() - start
    if model is not None:
        write_json(
            directory / "matching.json", compare_candidate(model, simulator, output)
        )
        model.save(directory / "model.npz")
        write_json(directory / "report.json", model.report)
        write_json(directory / "contract.json", model.contract)
    started = time.perf_counter()
    evidence = save_checkpoints(
        directory / "checkpoints",
        captured,
        train=reference._train,
        development=reference._development,
        recipe=RECIPE,
        selected_step=None
        if model is None
        else model.report["optimization"]["selected_step"],
        failure=reason,
        selected_model=None if model is None else model._model,
        anchor_reference=anchor_reference(simulator, output),
    )
    write_json(
        directory / "outcome.json",
        dict(
            status=status,
            model_available=model is not None,
            reason=reason,
            checkpoint_manifest_sha256=evidence["manifest_sha256"],
        ),
    )
    freeze_files(
        directory,
        "seal.json",
        dict(
            wall_time_s=time.perf_counter() - start,
            fit_and_capture_wall_time_s=fit_wall_time,
            checkpoint_diagnostics_wall_time_s=time.perf_counter() - started,
            peak_memory=None,
            peak_memory_scope="not measured by the in-process fitter; no estimate",
        ),
    )
    print(f"fit complete {simulator}/{arm}: status={status}", flush=True)


class Predictors:
    def __init__(self, *args):
        self._instance = _runner().Predictors(*args)

    def predict(self, *args):
        return self._instance.predict(*args)

    def envelope(self, *args):
        return self._instance.envelope(*args)


def check_links(output, simulator, *, evaluation=False):
    return _parent()._native["check_links"](output, simulator, evaluation=evaluation)


def evaluate(simulator, output, *, replay=False):
    return _parent()._native["evaluate"](simulator, output, replay=replay)


def validate_rows(simulator, output, rows):
    return _parent()._native["validate_rows"](simulator, output, rows)


def compare_candidate(model, simulator, output):
    base, classes = Path(output) / simulator, _model_classes()
    return check_matched_models(
        classes["baseline"].load(base / "baseline/model.npz"),
        model,
        classes["balanced"].load(base / "balanced/model.npz"),
        classes["safeguarded"].load(base / "safeguarded/model.npz"),
        simulator,
    )


def decision(output):
    summaries, rows, optimizations = {}, {}, {}
    for simulator in SIMULATORS:
        check_links(output, simulator, evaluation=True)
        base = Path(output) / simulator
        summaries[simulator] = read_json(base / "evaluation/summary.json")
        rows[simulator] = read_json(base / "evaluation/rows.json")
        validate_rows(simulator, output, rows[simulator])
        outcome = read_json(base / "candidate/outcome.json")
        optimizations[simulator] = (
            read_json(base / "candidate/report.json")["optimization"]
            if outcome["model_available"]
            else None
        )
    return reduce_full_cache(
        summaries, protocol(), rows=rows, optimization=optimizations["crazyflow"]
    )


def finalize(output):
    output = Path(output)
    if (output / "run.json").exists() or (output / "decision.json").exists():
        raise FileExistsError("frozen result already exists")
    write_json(output / "decision.json", decision(output))
    sha = freeze_files(
        output,
        "run.json",
        dict(
            protocol_sha256=PROTOCOL_SHA256,
            resolved_protocol_sha256=resolved_protocol(output),
            runtime=check_sources(),
            scope="Frozen full-cache gradients: two fresh fits, six byte-exact imported reference fits",
        ),
    )
    print("trusted bundle SHA256: " + sha, flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    return _parent()._native["verify_bundle"](output, expected_sha256)


def replay(simulator, output, expected_sha256):
    return _parent()._native["replay"](simulator, output, expected_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "fit", "evaluate", "finalize", "replay")
    )
    parser.add_argument("simulator", choices=SIMULATORS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="candidate")
    parser.add_argument("--expected-bundle-sha")
    args = parser.parse_args()
    if args.stage == "fit":
        fit_arm(args.simulator, args.arm, args.output)
    elif args.stage == "finalize":
        finalize(args.output)
    elif args.stage == "replay":
        if not args.expected_bundle_sha:
            parser.error("replay requires --expected-bundle-sha")
        replay(args.simulator, args.output, args.expected_bundle_sha)
    else:
        globals()[args.stage](args.simulator, args.output)


_FULL_CACHE_PAIRS = {
    "full_cache_public_progress": ("candidate", "baseline"),
    "balanced_progress": ("candidate", "balanced"),
    "safeguarded_progress": ("candidate", "safeguarded"),
}


def _check_gradient(optimization, p, count):
    """Completed-model reported-work contract; actual prefixes live in observer."""
    policy = p["fitting"]["optimizer"]
    steps = policy["proposal_attempts"]
    expected = dict(
        policy="ordered_full_cache",
        loss_scope="full_training_cache_pre_proposal",
        training_windows=count,
        index_dtype="int64",
        sampling_seed=None,
        attempts_started=steps,
        gradient_proposal_calls_returned=steps,
        completed_acceptance_attempts=steps,
        known_gradient_window_visits=count * steps,
        incomplete_gradient_work_unknown=False,
    )
    if (
        optimization.get("gradient") != expected
        or any(
            optimization.get(key) != value
            for key, value in (
                ("minibatch_seed", None),
                ("gradient_sampling", "ordered_full_cache"),
                ("mechanism", p["id"]),
            )
        )
        or "minibatch_seed" not in optimization
    ):
        raise ValueError("candidate full-cache gradient identity or accounting differs")
    for key in (
        "training_windows",
        "attempts_started",
        "gradient_proposal_calls_returned",
        "completed_acceptance_attempts",
        "known_gradient_window_visits",
    ):
        if type(optimization["gradient"][key]) is not int:
            raise ValueError("candidate gradient counters must be integers")
    if type(optimization["gradient"]["incomplete_gradient_work_unknown"]) is not bool:
        raise ValueError("candidate gradient unknown-work flag must be boolean")
    return deepcopy(expected)


def check_matched_models(
    public, candidate, balanced, safeguarded, simulator, *, candidate_module=None
):
    """Trust anchored references and compare data/mechanics without a quadratic arm."""
    if candidate_module is None:
        from glassbox.experimental import full_cache_gradient_model as candidate_module
    if (
        type(candidate) is not candidate_module.CandidateDynamics
        or type(candidate._model) is not candidate_module.BalancedQuadraticSequenceModel
    ):
        raise TypeError("candidate must be the frozen full-cache-gradient class")
    p = protocol()
    recipe = p["fitting"]["generic_recipe"]
    references = dict(baseline=public, balanced=balanced, safeguarded=safeguarded)
    models = {**references, "candidate": candidate}
    anchors = p["imported_references"]["trusted_comparator_revisions"][simulator]
    for arm, model in references.items():
        if model.fingerprint() != anchors[arm]["revision_fingerprint"]:
            raise ValueError("imported reference differs from trusted revision: " + arm)
        if model.recipe.get("batch_size") != recipe["batch_size"]:
            raise ValueError("imported reference batch recipe differs")
    if public.recipe != recipe or public.report["recipe"] != recipe:
        raise ValueError("public fitting recipe differs")
    if (
        candidate.recipe != candidate_module.RECIPE
        or candidate.report["recipe"] != candidate_module.RECIPE
    ):
        raise ValueError("candidate full-cache recipe differs")
    expected_recipe = dict(
        safeguarded.recipe,
        id=p["id"],
        batch_size=384,
        gradient_sampling="ordered_full_cache",
    )
    if candidate.recipe != expected_recipe:
        raise ValueError(
            "candidate recipe changes more than full-cache gradient policy"
        )
    if any(model.contract != public.contract for model in models.values()):
        raise ValueError("matched consumer contracts differ")
    if any(
        model.report.get("previous_revision") is not None for model in models.values()
    ):
        raise ValueError("matching requires initial fits")
    roles = p["planned_automatic_roles"][simulator]
    if set(public._seen) != set(roles["training"]) | set(roles["development"]):
        raise ValueError("public recording ledger differs from admitted roles")
    windows = {}
    for attribute, role, expected_count in (
        ("_train", "training", recipe["training_windows"]),
        ("_development", "development", recipe["development_windows"]),
    ):
        cache = getattr(public, attribute)
        if len(cache.keys) != expected_count or {
            x.recording_id for x in cache.keys
        } != set(roles[role]):
            raise ValueError("public cache count or parent roster differs")
        expected = _window_fingerprint(cache)
        for arm, model in models.items():
            current = getattr(model, attribute)
            if (
                _window_fingerprint(current) != expected
                or model.report[role] != current.coverage()
            ):
                raise ValueError(arm + " " + role + " cache or coverage differs")
        windows[role] = expected
    norms = array_fingerprint({}, balanced._model.norms)
    shapes = {k: value.shape for k, value in balanced._model.params.items()}
    for arm, model in models.items():
        expected_norms = balanced._model.norms
        if arm == "baseline":
            expected_norms = {
                key: expected_norms[key] for key in _shared_matching.BASE_NORMS
            }
        if array_fingerprint({}, model._model.norms) != array_fingerprint(
            {}, expected_norms
        ):
            raise ValueError(arm + " feature norms differ")
        if (
            arm != "baseline"
            and {k: v.shape for k, v in model._model.params.items()} != shapes
        ):
            raise ValueError(arm + " parameter roster or shapes differ")
        if any(
            model.report[key] != balanced.report[key]
            for key in ("history_steps", "horizon_steps", "delay_steps")
        ):
            raise ValueError(arm + " timing report differs")
        if (
            any(
                getattr(model._model, key) != getattr(balanced._model, key)
                for key in ("kind", "dt_s", "delay_steps")
            )
            or model.history_steps != balanced.history_steps
            or model.horizon_steps != balanced.horizon_steps
        ):
            raise ValueError(arm + " model timing or kind differs")
    train = public._train.batch
    hold = np.repeat(train.past_states[:, -1:], train.future_states.shape[1], axis=1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - train.future_states) ** 2, axis=0)),
        recipe["hold_scale_floor"] * balanced._model.norms["state_scale"],
    )
    scale_hash = array_fingerprint({}, {"error_scale": scale})
    for arm, model in models.items():
        if (
            array_fingerprint(
                {},
                {
                    "error_scale": np.asarray(
                        model.report["optimization"]["error_scale"]
                    )
                },
            )
            != scale_hash
        ):
            raise ValueError(arm + " original hold scale differs")
    fitting_recipe = dict(recipe, batch_size=384)
    expected_matching = dict(
        balanced.report["matching"],
        batch_size=384,
        minibatch_seed=None,
        gradient_sampling="ordered_full_cache",
    )
    if candidate.report["matching"] != expected_matching:
        raise ValueError("candidate preparation changes more than gradient metadata")
    count = sum(v.size for v in candidate._model.params.values())
    budget = _shared_matching._check_optimization(
        candidate, fitting_recipe, count, "full-cache"
    )
    objective = _balanced_matching._objective(candidate, fitting_recipe)
    for arm in ("balanced", "safeguarded"):
        _balanced_matching._objective(models[arm], recipe)
        if (
            candidate.report["optimization"]["objective"]
            != models[arm].report["optimization"]["objective"]
        ):
            raise ValueError("candidate fixed objective differs from trusted " + arm)
    optimization = candidate.report["optimization"]
    gradient = _check_gradient(optimization, p, len(public._train.keys))
    safeguard = previous._check_safeguard(optimization, p)
    return dict(
        experiment=p["id"],
        population=simulator,
        candidate_fingerprint=candidate.fingerprint(),
        reference_fingerprints={
            arm: model.fingerprint() for arm, model in references.items()
        },
        window_fingerprints=windows,
        roles=deepcopy(roles),
        normalizations_fingerprint=norms,
        original_loss_scale_fingerprint=scale_hash,
        parameter_count=count,
        contract=candidate.contract,
        fitting_recipe=fitting_recipe,
        optimizer=budget,
        objective=objective,
        safeguard=safeguard,
        gradient=gradient,
        exact_same_data=True,
        exact_same_numeric_objective=True,
        actual_initial_identity_verified_separately_by_observer=True,
        equal_flops_claimed=False,
        equal_end_to_end_compute_claimed=False,
        scope="Anchored references, exact caches/norms/scales/shapes and declared objective/gradient budget. Actual initialization, weights, indices and proposal work require the observer; no fitting or prediction.",
    )


def optimization_progress(optimization, p):
    if optimization is not None:
        _check_gradient(
            optimization,
            p,
            p["fitting"]["optimizer"]["gradient_rows"]["benchmark_training_windows"],
        )
    return previous.optimization_progress(optimization, p)


def reduce_full_cache(summaries, p, *, rows, optimization):
    """Frozen physical gates and canonical optimization diagnostic remain separate."""
    if (
        rows is None
        or set(summaries) != set(rows)
        or any(aggregate(values) != summaries[sim] for sim, values in rows.items())
    ):
        raise ValueError("summary differs from full metric rows")
    if set(p["decision"]["comparisons"]) != set(_FULL_CACHE_PAIRS):
        raise ValueError("frozen full-cache-gradient comparison roster differs")
    if any(
        set(row["arm"] for row in values)
        != {"baseline", "safeguarded", "balanced", "candidate", "hold"}
        for values in rows.values()
    ):
        raise ValueError("frozen full-cache-gradient five-arm roster differs")
    # Reuse the unchanged five-arm roster checker through a private role view.
    cohort_rows = {
        sim: [
            dict(
                row,
                arm={"balanced": "anchored", "safeguarded": "quadratic"}.get(
                    row["arm"], row["arm"]
                ),
            )
            for row in values
        ]
        for sim, values in rows.items()
    }
    cohorts = _full_cohorts(cohort_rows, p)
    for cohort in cohorts.values():
        cohort["arms"]["balanced"] = cohort["arms"].pop("anchored")
        cohort["arms"]["safeguarded"] = cohort["arms"].pop("quadratic")
    comparisons = {
        name: _comparison(rows, p, name, *arms)
        for name, arms in _FULL_CACHE_PAIRS.items()
    }
    target = p["decision"]["angular_retention"]
    if (target["numerator"], target["denominator"]) != ("candidate", "balanced"):
        raise ValueError("angular retention must use imported balanced denominator")
    view = deepcopy(p)
    view["decision"]["angular_repair"] = dict(target, denominator="quadratic")
    angular_rows = {
        sim: [
            dict(row, arm="quadratic" if row["arm"] == "balanced" else "candidate")
            for row in values
            if row["arm"] in ("candidate", "balanced")
        ]
        for sim, values in rows.items()
    }
    angular = _angular_repair(
        {sim: aggregate(values) for sim, values in angular_rows.items()},
        view,
        comparisons["balanced_progress"],
    )
    angular.update(
        denominator_arm="balanced",
        baseline_fields_mean="balanced",
        policy=deepcopy(target),
    )
    hashes = {
        name: result["bootstrap"].get("parent_draws_sha256")
        for name, result in comparisons.items()
    }
    hashes["angular_retention"] = angular["bootstrap"].get("parent_draws_sha256")
    available = {value for value in hashes.values() if value is not None}
    if len(available) > 1:
        raise ValueError("pairwise bootstrap parent draws differ")
    checks = {
        name: result["residual_criteria_pass"] for name, result in comparisons.items()
    }
    checks.update(
        angular_retention=angular["residual_criteria_pass"],
        finite_eligible_predictions=all(
            result["checks"]["finite_eligible_predictions"]
            for result in comparisons.values()
        ),
        matching_planned_queries_and_truth=all(
            result["checks"]["matching_planned_queries_and_truth"]
            for result in comparisons.values()
        ),
    )
    diagnostic = optimization_progress(optimization, p)
    residual = all(checks.values())
    return dict(
        protocol=p["id"],
        comparisons=comparisons,
        **{
            name: result["residual_criteria_pass"]
            for name, result in comparisons.items()
        },
        angular_retention=checks["angular_retention"],
        angular_retention_comparison=angular,
        diagnostic_optimization_progress=diagnostic["passed"],
        optimization_diagnostic=diagnostic,
        residual_criteria_pass=residual,
        checks=checks,
        cohorts=cohorts,
        bootstrap_parent_draws_sha256=next(iter(available), None),
        bootstrap_comparison_hashes=hashes,
        available_bootstrap_parent_draws_match=bool(available),
        shared_bootstrap_parent_draws_verified=all(hashes.values())
        and len(available) == 1,
        verification_status="requires_integrity_replay_tamper_and_focused_tests",
        public_promotion=False,
        qualification=p["decision"]["public_promotion"],
    )


if __name__ == "__main__":
    main()
