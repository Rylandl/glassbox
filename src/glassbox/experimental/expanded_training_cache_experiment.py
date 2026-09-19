"""Frozen training-cache expansion with authenticated current-data preparation."""

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
from glassbox.learner import _subset

from . import full_cache_gradient_experiment as previous
from . import initial_channel_balance_experiment as shared_runner
from . import safeguarded_adam_experiment as _historical_helpers
from . import state_input_experiment as inherited
from .affine_anchored_experiment import _checkpoint_sources

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/expanded-training-cache-v1.json"
PROTOCOL_SHA256 = "7f5f67cec628f2d86a2db23f3637a2a24c476d2c6f656e33b7983f1dfbbab65b"
SIMULATORS = ("crazyflow", "cascade")
REFERENCE_ARMS = {
    "baseline": "baseline",
    "balanced": "balanced",
    "fullcache384": "candidate",
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
        raise ValueError("frozen expanded-cache protocol changed")
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
        "data_support",
        "initialization_evidence",
        "data_independent_initial_subset",
    ):
        value[key] = deepcopy(delta[key])
    value["counts"], value["recordings"] = (
        deepcopy(base["counts"]),
        deepcopy(base["recordings"]),
    )
    original = read_json(ROOT / delta["cohort"]["original_protocol"]["path"])
    if set(_test_roster(base["recordings"], 0)) != set(
        _test_roster(original["recordings"], 9000000)
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
    for key in ("generic_recipe", "mechanism"):
        value["fitting"][key] = deepcopy(base["fitting"][key])
    replacements = delta["resolution"]["replace_current_view_fields"]
    value["fitting"]["normalization"] = replacements["fitting.normalization"]
    value["fitting"]["mechanism"]["initial_prediction"] = replacements[
        "fitting.mechanism.initial_prediction"
    ]
    value["fitting"]["mechanism"]["initialization"] = delta["fitting"]["initialization"]
    value["fitting"].pop("trusted_initial_snapshots", None)
    value["fitting"]["arms"] = list(ARMS)
    value["checkpoint_diagnostics"] = {
        **deepcopy(base["checkpoint_diagnostics"]),
        **deepcopy(delta["checkpoint_diagnostics"]),
    }
    diagnostics = value["checkpoint_diagnostics"]
    diagnostics.update(
        source_binding=delta["verification"]["source_binding"],
        initialization=replacements["checkpoint_diagnostics.initialization"],
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
        delta["decision"]["guards_for_every_required_comparison"]
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
        "glassbox.experimental._expanded_cache_parent_runner",
        _historical_helpers.__file__,
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
        ROOT / folder / (prefix + "expanded_training_cache_" + stem + ".py")
        for stem in ("model", "checkpoints", "experiment")
        for folder, prefix in (("src/glassbox/experimental", ""), ("tests", "test_"))
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("expanded-cache implementation or tests missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


@lru_cache(maxsize=1)
def _implementation():
    module = _load_private(
        "glassbox.experimental._expanded_cache_interaction", inherited.__file__
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
        runtime["expanded_training_cache_implementation"] = _sources()
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
        "glassbox.experimental._expanded_cache_collection_runner",
        shared_runner.__file__,
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
    from .expanded_training_cache_model import CandidateDynamics as Expanded
    from .full_cache_gradient_model import CandidateDynamics as FullCache
    from .initial_channel_balance_model import CandidateDynamics as Balanced

    return dict(
        baseline=LearnedDynamics,
        balanced=Balanced,
        fullcache384=FullCache,
        candidate=Expanded,
    )


def trusted_references():
    return _parent()._native["trusted_references"]()


def _reference_sources(simulator):
    return _parent()._native["_reference_sources"](simulator)


def _reference_manifest(*args):
    return {
        **_parent()._native["_reference_manifest"](*args),
        "format": "glassbox-expanded-cache-reference-import-v1",
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
        "gradient_training_windows": 1536,
        "fitting_collection": "authenticated_recordings_expanded1536_exact384prefix",
    }


def _prepare(simulator, output):
    from .expanded_training_cache_model import prepare

    base, runtime = Path(output) / simulator, check_sources()
    _runner().check_import(simulator, output, runtime)
    check_reference_import(simulator, output, runtime)
    roles = read_json(base / "excitation/roles.json")
    if roles != protocol()["planned_automatic_roles"][simulator]:
        raise ValueError("candidate preparation roles differ from frozen roles")
    collection, _ = flight().collection_and_trajectories(base / "excitation")
    reference = _model_classes()["baseline"].load(base / "baseline/model.npz")
    seals = dict(
        common_data_seal_sha256=digest(base / "data/seal.json"),
        excitation_seal_sha256=digest(base / "excitation/seal.json"),
        excitation_reuse_sha256=digest(base / "reuse.json"),
        reference_reuse_sha256=digest(base / "reference-reuse.json"),
    )
    return prepare(collection, reference, roles=roles, data_seal=seals)


def initialization_context(simulator, output, prepared):
    from . import expanded_training_cache_checkpoints as checkpoints
    from .expanded_training_cache_model import RECIPE

    delta = delta_protocol()
    entry = delta["data_independent_initial_subset"]["trusted_snapshots"][simulator]
    return checkpoints.initialization_context(
        train=prepared.train,
        development=prepared.development,
        recipe=RECIPE,
        preparation=prepared.provenance,
        snapshot_path=Path(output) / entry["imported_relative_path"],
        trusted_subset=entry,
        source_bundle_sha256=delta["prior_reference_bundle"]["manifest_sha256"],
    )


def check_checkpoints(output, simulator, arm, *, replay=False):
    if arm not in ARMS:
        raise ValueError("unknown checkpoint arm")
    if arm in REFERENCE_ARMS:
        return _parent()._native["check_checkpoints"](
            output, simulator, arm, replay=replay
        )
    from . import expanded_training_cache_checkpoints as checkpoints
    from .expanded_training_cache_model import (
        RECIPE,
        PreparationUnavailable,
        verify_preparation,
    )

    directory = Path(output) / simulator / arm
    outcome = read_json(directory / "outcome.json")
    try:
        prepared = _prepare(simulator, output)
    except PreparationUnavailable as error:
        expected = dict(
            status="fit_failure",
            model_available=False,
            reason=str(error),
            failure_stage="preparation",
            checkpoint_manifest_sha256=None,
        )
        if (
            outcome != expected
            or read_json(directory / "preparation-unavailable.json")
            != error.diagnostics
            or any(
                (directory / name).exists()
                for name in (
                    "checkpoints",
                    "model.npz",
                    "preparation.npz",
                    "report.json",
                    "matching.json",
                    "contract.json",
                )
            )
        ):
            raise ValueError("preparation-unavailable evidence differs") from error
        return dict(
            status="preparation_unavailable",
            reason=str(error),
            diagnostics=error.diagnostics,
            fit_entered=False,
        )
    if outcome.get("failure_stage") == "preparation":
        raise ValueError("preparation failure claimed despite complete support")
    verify_preparation(directory / "preparation.npz", prepared)
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
    runtime = check_sources()
    checker = (
        checkpoints.replay_checkpoints if replay else checkpoints.check_checkpoint_links
    )
    return checker(
        directory / "checkpoints",
        expected_sha256=outcome["checkpoint_manifest_sha256"],
        expectation=expectation,
        train=prepared.train,
        development=prepared.development,
        provenance=fit_provenance(simulator, arm, output, runtime),
        source_sha256=runtime["checkpoint_source_sha256"],
        selected_model=None if model is None else model._model,
        initialization_context=initialization_context(simulator, output, prepared),
    )


def fit_arm(simulator, arm, output):
    from ..learner import RECIPE as PUBLIC_RECIPE
    from .expanded_training_cache_checkpoints import capture_fit, save_checkpoints
    from .expanded_training_cache_model import (
        RECIPE,
        CandidateFitError,
        PreparationUnavailable,
        fit_candidate,
        save_preparation,
    )

    if arm != "candidate" or PUBLIC_RECIPE != protocol()["fitting"]["generic_recipe"]:
        raise ValueError("only frozen candidate fitting is permitted")
    runtime = data_ready(output)
    for reference_arm in REFERENCE_ARMS:
        check_checkpoints(output, simulator, reference_arm)
    directory = Path(output) / simulator / arm
    directory.mkdir(parents=True, exist_ok=False)
    provenance = fit_provenance(simulator, arm, output, runtime)
    write_json(directory / "start.json", provenance)
    started = time.perf_counter()
    try:
        prepared = _prepare(simulator, output)
    except PreparationUnavailable as error:
        write_json(directory / "preparation-unavailable.json", error.diagnostics)
        write_json(
            directory / "outcome.json",
            dict(
                status="fit_failure",
                model_available=False,
                reason=str(error),
                failure_stage="preparation",
                checkpoint_manifest_sha256=None,
            ),
        )
        freeze_files(
            directory,
            "seal.json",
            dict(
                wall_time_s=time.perf_counter() - started,
                preparation_wall_time_s=time.perf_counter() - started,
                fit_and_capture_wall_time_s=None,
                checkpoint_diagnostics_wall_time_s=None,
                peak_memory=None,
                peak_memory_scope="not measured; fit not entered",
            ),
        )
        print(f"preparation unavailable {simulator}/{arm}: {error}", flush=True)
        return
    save_preparation(directory / "preparation.npz", prepared)
    context = initialization_context(simulator, output, prepared)
    preparation_time = time.perf_counter() - started
    model, reason, status = None, None, "complete"
    capture = capture_fit(
        arm,
        provenance,
        source_sha256=runtime["checkpoint_source_sha256"],
        expected_steps=tuple(protocol()["checkpoint_diagnostics"]["steps"]),
    )
    fit_started = time.perf_counter()
    try:
        with capture as captured:
            model = fit_candidate(prepared)
    except CandidateFitError as error:
        reason, status = str(error), "fit_failure"
    fit_time = time.perf_counter() - fit_started
    if model is not None:
        write_json(
            directory / "matching.json", compare_candidate(model, simulator, output)
        )
        model.save(directory / "model.npz")
        write_json(directory / "report.json", model.report)
        write_json(directory / "contract.json", model.contract)
    diagnostic_started = time.perf_counter()
    evidence = save_checkpoints(
        directory / "checkpoints",
        captured,
        train=prepared.train,
        development=prepared.development,
        recipe=RECIPE,
        selected_step=None
        if model is None
        else model.report["optimization"]["selected_step"],
        failure=reason,
        selected_model=None if model is None else model._model,
        initialization_context=context,
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
            wall_time_s=time.perf_counter() - started,
            preparation_wall_time_s=preparation_time,
            fit_and_capture_and_calibration_wall_time_s=fit_time,
            fit_and_capture_wall_time_s=(
                None
                if model is None
                else model.report["timing"]["fit_and_capture_wall_time_s"]
            ),
            calibration_wall_time_s=(
                None
                if model is None
                else model.report["timing"]["calibration_wall_time_s"]
            ),
            checkpoint_diagnostics_wall_time_s=time.perf_counter() - diagnostic_started,
            peak_memory=None,
            peak_memory_scope="not measured by in-process fitter; no estimate",
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
        classes["fullcache384"].load(base / "fullcache384/model.npz"),
        simulator,
        prepared=_prepare(simulator, output),
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
    return reduce_expanded_cache(
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
            scope="Frozen expanded training cache: two fresh1536 fits, six byte-exact imported384 reference fits",
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


_EXPANDED_CACHE_PAIRS = {
    "expanded_public_progress": ("candidate", "baseline"),
    "balanced_progress": ("candidate", "balanced"),
    "data_support_progress": ("candidate", "fullcache384"),
}
_check_gradient = previous._check_gradient
optimization_progress = previous.optimization_progress


def _same_expanded_windows(left, right):
    if _window_fingerprint(left) != _window_fingerprint(right):
        return False
    if left.excitation_declared != right.excitation_declared:
        return False
    return not left.excitation_declared or array_fingerprint(
        {}, {k: getattr(left, k) for k in ("past_excitation", "future_excitation")}
    ) == array_fingerprint(
        {}, {k: getattr(right, k) for k in ("past_excitation", "future_excitation")}
    )


def check_matched_models(
    public,
    candidate,
    balanced,
    fullcache384,
    simulator,
    *,
    prepared,
    candidate_module=None,
):
    """Validate new-data preparation and anchored references without identity fiction."""
    if candidate_module is None:
        from glassbox.experimental import (
            expanded_training_cache_model as candidate_module,
        )
    if (
        type(candidate) is not candidate_module.CandidateDynamics
        or type(candidate._model) is not candidate_module.BalancedQuadraticSequenceModel
    ):
        raise TypeError("candidate must be the frozen expanded-training-cache class")
    p = protocol()
    recipe, support = p["fitting"]["generic_recipe"], p["data_support"]
    fitting = dict(
        recipe,
        training_windows=support["training_windows"],
        batch_size=support["training_windows"],
    )
    references = dict(baseline=public, balanced=balanced, fullcache384=fullcache384)
    models = {**references, "candidate": candidate}
    anchors = p["imported_references"]["trusted_comparator_revisions"][simulator]
    for arm, model in references.items():
        if model.fingerprint() != anchors[arm]["revision_fingerprint"]:
            raise ValueError("imported reference differs from trusted revision: " + arm)
        if (
            model.recipe["training_windows"] != support["reference_training_windows"]
            or model.recipe["development_windows"] != support["development_windows"]
        ):
            raise ValueError("imported reference cache recipe differs")
    if public.recipe != recipe or public.report["recipe"] != recipe:
        raise ValueError("public fitting recipe differs")
    expected = dict(fullcache384.recipe, **p["fitting"]["candidate_recipe_overrides"])
    if (
        candidate.recipe != expected
        or candidate.recipe != candidate_module.RECIPE
        or candidate.report["recipe"] != expected
        or candidate_module.FITTING_RECIPE != fitting
    ):
        raise ValueError("candidate changes more than the declared cache recipe")
    if any(model.contract != public.contract for model in models.values()):
        raise ValueError("matched signal/timing contracts differ")
    if any(
        model.report.get("previous_revision") is not None for model in models.values()
    ):
        raise ValueError("matching requires initial fits")
    roles = p["planned_automatic_roles"][simulator]
    provenance = prepared.provenance
    if (
        prepared.contract != public.contract
        or provenance["recording_content_fingerprints"] != public._seen
        or provenance["roles"] != roles
        or provenance["public_reference_fingerprint"] != public.fingerprint()
        or candidate.report.get("preparation") != provenance
        or set(public._seen) != set(roles["training"]) | set(roles["development"])
        or set(roles["training"]) & set(roles["development"])
    ):
        raise ValueError("prepared source ledger, roles, contract or report differs")
    for attribute, role, n in (
        ("_train", "training", support["reference_training_windows"]),
        ("_development", "development", support["development_windows"]),
    ):
        cache = getattr(public, attribute)
        if len(cache.keys) != n or {k.recording_id for k in cache.keys} != set(
            roles[role]
        ):
            raise ValueError("reference cache count or parent roster differs")
        for arm, model in references.items():
            current = getattr(model, attribute)
            if (
                not _same_expanded_windows(current, cache)
                or model.report[role] != current.coverage()
            ):
                raise ValueError(arm + " reference cache or coverage differs")
    for cache, role, count in (
        (prepared.train, "training", support["training_windows"]),
        (prepared.development, "development", support["development_windows"]),
    ):
        identities = list(zip(cache.keys, cache.source_origins, strict=True))
        if (
            len(cache.keys) != count
            or len(set(identities)) != count
            or {k.recording_id for k in cache.keys} != set(roles[role])
        ):
            raise ValueError("prepared cache count, uniqueness or roles differ")
        actual = candidate._train if role == "training" else candidate._development
        if (
            not _same_expanded_windows(actual, cache)
            or candidate.report[role] != cache.coverage()
        ):
            raise ValueError("candidate " + role + " cache or coverage differs")
    prefix = _subset(prepared.train, np.arange(support["reference_training_windows"]))
    if not _same_expanded_windows(prefix, public._train) or not _same_expanded_windows(
        prepared.development, public._development
    ):
        raise ValueError("expanded training prefix or fixed development cache differs")
    shapes = {k: v.shape for k, v in fullcache384._model.params.items()}
    for arm, model in models.items():
        if (
            arm != "baseline"
            and {k: v.shape for k, v in model._model.params.items()} != shapes
        ):
            raise ValueError(arm + " quadratic parameter roster/shapes differ")
        if (
            any(
                model.report[k] != public.report[k]
                for k in ("history_steps", "horizon_steps", "delay_steps")
            )
            or any(
                getattr(model._model, k) != getattr(public._model, k)
                for k in ("kind", "dt_s", "delay_steps")
            )
            or model.history_steps != public.history_steps
            or model.horizon_steps != public.horizon_steps
        ):
            raise ValueError(arm + " model timing or kind differs")
    batch = prepared.train.batch
    norms = candidate_module.training_norms(batch)
    if array_fingerprint({}, norms) != array_fingerprint(
        {}, prepared.norms
    ) or array_fingerprint({}, norms) != array_fingerprint({}, candidate._model.norms):
        raise ValueError(
            "candidate normalization differs from actual expanded training data"
        )
    hold = np.repeat(batch.past_states[:, -1:], batch.future_states.shape[1], axis=1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - batch.future_states) ** 2, axis=0)),
        recipe["hold_scale_floor"] * norms["state_scale"],
    )
    scale_hash = array_fingerprint({}, {"error_scale": scale})
    if (
        array_fingerprint({}, {"error_scale": prepared.error_scale}) != scale_hash
        or array_fingerprint(
            {},
            {
                "error_scale": np.asarray(
                    candidate.report["optimization"]["error_scale"]
                )
            },
        )
        != scale_hash
        or prepared.ridge
        != recipe["ridge_fraction"]
        * len(batch.past_states)
        * batch.future_states.shape[1]
        or prepared.delay != candidate._model.delay_steps
    ):
        raise ValueError("candidate expanded loss scale, ridge or delay differs")
    count = sum(v.size for v in candidate._model.params.values())
    preparation_facts = dict(
        intervention=p["id"],
        training_windows_fingerprint=_window_fingerprint(prepared.train),
        development_windows_fingerprint=_window_fingerprint(prepared.development),
        reference_training_windows_fingerprint=_window_fingerprint(public._train),
        training_norms_fingerprint=array_fingerprint({}, norms),
        error_scale_fingerprint=scale_hash,
        fitting_recipe=fitting,
        ridge=prepared.ridge,
        delay=prepared.delay,
        parameter_count=count,
        training_windows=len(prepared.train.keys),
        development_windows=len(prepared.development.keys),
        reference_training_windows=len(public._train.keys),
        added_training_windows=len(prepared.train.keys) - len(public._train.keys),
        same_recordings=True,
        same_development_cache=True,
        training_prefix_exact=True,
        same_training_cache=False,
        same_numeric_objective=False,
        same_all_initial_parameters_required=False,
        equal_compute=False,
    )
    if any(provenance.get(k) != v for k, v in preparation_facts.items()):
        raise ValueError("preparation facts differ from actual expanded cache")
    budget = _shared_matching._check_optimization(
        candidate, fitting, count, "expanded-cache"
    )
    objective = _balanced_matching._objective(candidate, fitting)
    for model in (balanced, fullcache384):
        _balanced_matching._objective(model, recipe)
    optimization = candidate.report["optimization"]
    gradient = _check_gradient(optimization, p, len(batch.past_states))
    safeguard = previous.previous._check_safeguard(optimization, p)
    return dict(
        experiment=p["id"],
        population=simulator,
        candidate_fingerprint=candidate.fingerprint(),
        reference_fingerprints={
            arm: model.fingerprint() for arm, model in references.items()
        },
        roles=deepcopy(roles),
        preparation=deepcopy(provenance),
        window_fingerprints=dict(
            reference_training=_window_fingerprint(public._train),
            training=_window_fingerprint(prepared.train),
            development=_window_fingerprint(prepared.development),
        ),
        normalizations_fingerprint=array_fingerprint({}, norms),
        reference_normalizations_fingerprint=array_fingerprint(
            {}, fullcache384._model.norms
        ),
        original_loss_scale_fingerprint=scale_hash,
        parameter_count=count,
        contract=candidate.contract,
        fitting_recipe=fitting,
        optimizer=budget,
        objective=objective,
        safeguard=safeguard,
        gradient=gradient,
        same_recordings=True,
        same_development_cache=True,
        training_prefix_exact=True,
        same_objective_formula_arms=["candidate", "balanced", "fullcache384"],
        same_training_cache=False,
        same_numeric_objective=False,
        same_all_initial_parameters_required=False,
        equal_compute=False,
        external_five_array_subset_verified_separately_by_observer=True,
        scope="Authenticated reconstructed expanded cache and fixed development; training-derived norms/scales and formulas, not cross-arm numerical identity. Actual initialization/weights/gradient/acceptance evidence is separately source-bound; no fitting or prediction.",
    )


def reduce_expanded_cache(summaries, p, *, rows, optimization):
    """Frozen physical gates and canonical optimization diagnostic remain separate."""
    if (
        rows is None
        or set(summaries) != set(rows)
        or any(aggregate(values) != summaries[sim] for sim, values in rows.items())
    ):
        raise ValueError("summary differs from full metric rows")
    if set(p["decision"]["comparisons"]) != set(_EXPANDED_CACHE_PAIRS):
        raise ValueError("frozen expanded-training-cache comparison roster differs")
    if any(
        set(row["arm"] for row in values)
        != {"baseline", "fullcache384", "balanced", "candidate", "hold"}
        for values in rows.values()
    ):
        raise ValueError("frozen expanded-training-cache five-arm roster differs")
    # Reuse the unchanged five-arm roster checker through a private role view.
    cohort_rows = {
        sim: [
            dict(
                row,
                arm={"balanced": "anchored", "fullcache384": "quadratic"}.get(
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
        cohort["arms"]["fullcache384"] = cohort["arms"].pop("quadratic")
    comparisons = {
        name: _comparison(rows, p, name, *arms)
        for name, arms in _EXPANDED_CACHE_PAIRS.items()
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
