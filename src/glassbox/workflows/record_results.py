"""The manifest of recorded artifacts and the runner that regenerates them.

Every quantitative claim in the documentation is the literal content of one
artifact under ``docs/results/``, and this module owns the one manifest that
says, for each of them, which tier it belongs to, which inputs it consumes,
which optional extra it needs, and the exact ``glassbox`` steps that reproduce
it. Nothing here writes documentation prose: after regenerating an artifact,
update the prose on its page by hand from the new JSON.

Two tiers. The ``local`` tier runs in this repository with nothing downloaded
and is what continuous integration checks. The ``corpus`` tier is the
maintainer job: it needs a pinned public corpus on disk, several hours of
fitting, and for the Cascade rows the ``cascade`` extra.

The five corpus validation artifacts share one chain, ``corpus prepare`` then
``fit`` then ``evaluate``, and one assembled contract: the corpus with its
citation, pinned version and file digests; the scoring policy the evaluation
ran under; the fit that produced the model, with its split, its optimization
record and its parameter evidence; the results and the baseline they are
scored against; and the source fingerprint of the code that produced them.
The chain is a function of a :class:`RecordingPlan`, so the same manifest
describes a recording, a check against the committed artifact, and a smoke run
on a tiny fit budget whose output is marked ``smoke`` and can never be
mistaken for a recording.

Every step is a ``glassbox`` subcommand run in-process through
:func:`glassbox.cli.main`, except the assembly steps, which have no subcommand
form and call this module's own writers.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import math
import platform as platform_module
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import glassbox
import glassbox.cli as cli
from glassbox.io.corpus import REFERENCE_CORPORA

_REPO_ROOT = Path(__file__).resolve().parents[3]

_EXTRA_MODULES = {"cascade": "cascade", "px4": "pyulog", "ros": "rosbags"}

LOCAL_TIER = "local"
CORPUS_TIER = "corpus"
TIERS = (LOCAL_TIER, CORPUS_TIER)

VALIDATION_ARTIFACT_TYPE = "glassbox_corpus_validation"
VALIDATION_FORMAT_VERSION = 1
VALIDATION_METHOD_VERSION = 1

VALIDATION_SOURCE_FILES = (
    "belief/belief.py",
    "belief/belief_io.py",
    "belief/forecast_error.py",
    "belief/information.py",
    "belief/parameter_evidence.py",
    "core/data.py",
    "core/dynamics.py",
    "core/geometry.py",
    "core/identification.py",
    "core/metrics.py",
    "core/model.py",
    "core/model_io.py",
    "fitting.py",
    "io/corpus.py",
    "workflows/evaluate.py",
    "workflows/holdout.py",
)
"""The modules a corpus validation number passes through, fingerprinted.

The corpus adapter that produced the trajectories is not in this list because
the corpus block already pins the inputs by digest: a different adapter output
is a different corpus, and the pinned files say which one was read.
"""

DEFAULT_TOLERANCE = (1e-5, 1e-7)
"""Relative and absolute tolerance ``--check`` allows on a recorded float."""


class StepFailed(RuntimeError):
    """Raised by a step that could not complete; carries a human message."""


class UnknownArtifactNames(Exception):
    """Raised when ``--only`` names an artifact absent from the manifest."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        super().__init__(f"unknown artifact name(s): {', '.join(self.names)}")


def _expand_globs(argv: Sequence[str]) -> list[str]:
    """Expand any glob-pattern token, relative to the current directory.

    Steps are written exactly as the documented shell command, including glob
    patterns such as ``artifacts/x8_reference/canonical/training/*.npz`` that a
    shell would normally expand before ``glassbox`` ever saw them. Dispatching
    in-process bypasses the shell, so expansion happens here, lazily, only when
    a step actually runs and the files a previous step wrote exist.
    """

    expanded: list[str] = []
    for token in argv:
        if "*" in token or "?" in token:
            matches = sorted(glob.glob(token))
            if not matches:
                raise StepFailed(f"no files matched {token!r}")
            expanded.extend(matches)
        else:
            expanded.append(token)
    return expanded


@dataclass(frozen=True)
class CliStep:
    """One ``glassbox`` subcommand invocation, run in-process."""

    argv: tuple[str, ...]

    def describe(self) -> str:
        return "glassbox " + " ".join(self.argv)

    def run(self) -> None:
        argv = _expand_globs(self.argv)
        try:
            code = cli.main(argv)
        except SystemExit as exit_signal:
            code = exit_signal.code
        if code not in (None, 0):
            raise StepFailed(f"{self.describe()} failed (exit {code!r})")


@dataclass(frozen=True)
class PythonStep:
    """A regeneration step with no ``glassbox`` subcommand form."""

    description: str
    action: Callable[[], None]

    def describe(self) -> str:
        return self.description

    def run(self) -> None:
        self.action()


Step = CliStep | PythonStep


@dataclass(frozen=True)
class RecordingPlan:
    """Where one run reads its corpora, writes its outputs, and how far it fits.

    The default plan is the recording: prepared corpora under ``artifacts/``,
    artifacts under ``docs/results/``, every pinned file downloaded and
    verified on demand, and each fit run for the budget its experiment page
    documents.

    ``source_root`` reuses already-verified upstream files from a tree laid out
    with one directory per corpus, so a run that writes elsewhere does not
    download the corpora a second time. ``fit_steps`` and ``fold_limit``
    replace the documented optimization budget and the number of held-out folds
    with a smaller one. ``smoke`` records in every artifact it produces that
    those budgets were cut, so a smoke output can never be read as evidence.
    """

    corpus_root: Path = Path("artifacts")
    results_root: Path = Path("docs/results")
    source_root: Path | None = None
    fit_steps: int | None = None
    fold_limit: int | None = None
    smoke: bool = False

    def work(self, directory: str) -> Path:
        """The directory one corpus's prepared data and reports live in."""

        return self.corpus_root / directory

    def raw(self, directory: str) -> str:
        """The ``--raw`` argument that reuses an already-verified source tree."""

        if self.source_root is None:
            return ""
        return f"--raw {self.source_root / directory / 'raw'}"

    def steps(self) -> str:
        """The ``--steps`` argument that overrides a documented fit budget."""

        return "" if self.fit_steps is None else f"--steps {self.fit_steps}"

    def folds(self) -> str:
        """The ``--limit-folds`` argument that shortens a holdout run."""

        return "" if self.fold_limit is None else f"--limit-folds {self.fold_limit}"

    def output(self, filename: str) -> Path:
        return self.results_root / filename


@dataclass(frozen=True)
class ArtifactSpec:
    """One artifact under ``docs/results/`` and how to regenerate it."""

    name: str
    output: str
    steps: tuple[Step, ...] = ()
    extra: str | None = None
    inputs: tuple[str, ...] = ()
    tier: str = LOCAL_TIER
    doc_page: str = ""
    volatile: tuple[str, ...] = ()
    """Path patterns whose values vary with the host or the source tree.

    ``--check`` excludes them, and the artifact's pinned test uses the same
    table as its ``ignore`` collection, so the two agree by construction.
    """

    tolerance: tuple[float, float] = DEFAULT_TOLERANCE
    awaiting_first_record: str | None = None
    """Set while this entry's first recorded run has not been committed yet."""

    @property
    def recorded(self) -> bool:
        """Whether this entry's output is committed under ``docs/results/``."""

        return self.awaiting_first_record is None

    @property
    def doc_path(self) -> str:
        """The file :attr:`doc_page` names, without its section anchor.

        One page carries the prose for every artifact, so an entry names the
        section a re-record has to update and not only the file.
        """

        return self.doc_page.partition("#")[0]


def _cli(command: str) -> CliStep:
    """One step, written the way its experiment page documents the command.

    The string is split on whitespace, the way a shell splits a command line
    with no quoting, so a manifest entry reads as the command it documents. No
    path a run writes to may contain whitespace, and the command line refuses
    one that does.
    """

    return CliStep(tuple(command.split()))


# ---------------------------------------------------------------------------
# Provenance shared by every assembled artifact.


def source_fingerprint(relative_paths: Sequence[str]) -> str:
    """Digest the maintained source modules that produce one artifact.

    Every path is written relative to the ``glassbox`` package root, so the
    root is resolved from the installed package rather than from this module's
    own depth inside it.
    """

    source_root = Path(glassbox.__file__).resolve().parent
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        digest.update(relative_path.encode("utf-8"))
        digest.update((source_root / relative_path).read_bytes())
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    import jax

    return {
        "platform": platform_module.platform(),
        "python": platform_module.python_version(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
    }


# ---------------------------------------------------------------------------
# The corpus validation contract.

_FIT_SPEC_KEYS = (
    "training_horizons_s",
    "training_horizon_steps",
    "evaluation_horizons_s",
    "optimization_steps_per_model",
    "learning_rate",
    "endpoint_weight",
    "stability_regularization",
    "model_class",
    "model_family",
    "platform",
    "ablations",
    "fixed_response_time_constant_s",
    "training_flight_weighting",
    "multirotor_thrust_command_offset",
    "angular_control_coupling",
    "training_windows",
    "training_windows_by_horizon",
    "training_window_selection",
    "diagnostics",
)

_SPLIT_KEYS = (
    "mode",
    "holdout",
    "independent_source_group_holdout",
    "training_source_groups",
    "validation_source_groups",
)

_PROTOCOL_KEYS = (
    "protocol",
    "baseline",
    "stride",
    "floors",
    "scoring",
    "evaluation",
    "holdout_label",
    "independent_holdout",
    "can_promote_model",
)

_BASELINE_KEYS = (
    "baseline_metrics",
    "baseline_per_trajectory",
    "kinematic_persistence",
    "hold_state",
)


def _flight_paths(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce a fit report's per-flight table to identity and duration.

    A fit report records every training and validation flight in full, with its
    typed spec, labels and provenance. The artifact needs to say which flights
    were trained on and which were held out, not to restate the corpus, so only
    the path and the duration survive.
    """

    items = list(summaries)
    return {
        "count": len(items),
        "duration_s": sum(float(item["duration_s"]) for item in items),
        "paths": [str(item["path"]) for item in items],
    }


def _fit_arm(report: Mapping[str, Any]) -> dict[str, Any]:
    """The contract's fit block for one fitted model, from its fit report."""

    configuration = report["configuration"]
    split = report["split"]
    learned = report["models"]["learned_lag"]
    return {
        "spec": {
            key: configuration[key] for key in _FIT_SPEC_KEYS if key in configuration
        },
        "split": {
            **{key: split[key] for key in _SPLIT_KEYS if key in split},
            "training": _flight_paths(split["training_flights"]),
            "validation": _flight_paths(split["validation_flights"]),
        },
        "fit": learned["fit"],
        "parameter_evidence": learned["parameter_evidence"],
    }


def corpus_block(name: str) -> dict[str, Any]:
    """What the registry pins about one corpus, digests included."""

    corpus = REFERENCE_CORPORA[name]
    return {
        "name": corpus.name,
        "summary": corpus.summary,
        "citation": {
            "doi_or_url": corpus.citation.doi_or_url,
            "license": corpus.citation.license,
            "pinned_version": corpus.citation.pinned_version,
        },
        "evaluation_split": corpus.validation_split,
        "files": [
            {
                "relative_path": item.relative_path,
                "size_bytes": item.size_bytes,
                "algorithm": item.algorithm,
                "digest": item.digest,
            }
            for item in corpus.files
        ],
    }


def _fit_arms(
    evaluation: Mapping[str, Any], fit_reports: Mapping[str, Path]
) -> dict[str, Any]:
    """Read one fit block per arm the evaluation scored.

    A leave-one-label-out summary has no separate fit reports: each fold is its
    own fit, and the summary names the report it wrote, so the folds are the
    arms.
    """

    if "per_fold" in evaluation:
        return {
            value: _fit_arm(json.loads(Path(fold["report"]).read_text()))
            for value, fold in evaluation["per_fold"].items()
        }
    return {
        name: _fit_arm(json.loads(path.read_text()))
        for name, path in fit_reports.items()
    }


def assemble_corpus_validation_report(
    corpus: str,
    evaluation_report: Path,
    fit_reports: Mapping[str, Path] | None = None,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Assemble one corpus validation artifact from the chain's own reports.

    The artifact is a re-keying of machine output, never a transcription: the
    corpus block comes from the registry, the protocol and results from the
    evaluation report, and the fit blocks from the fit reports that produced
    the scored models. What the evaluation report says about itself, its own
    ``format_version`` included, stays under ``results``; the corpus block it
    carried is replaced by the registry's, which pins every file by digest.
    """

    evaluation = json.loads(evaluation_report.read_text())
    hoisted = {*_PROTOCOL_KEYS, *_BASELINE_KEYS, "corpus"}
    return {
        "artifact_type": VALIDATION_ARTIFACT_TYPE,
        "format_version": VALIDATION_FORMAT_VERSION,
        "smoke": smoke,
        "corpus": corpus_block(corpus),
        "protocol": {
            key: evaluation[key] for key in _PROTOCOL_KEYS if key in evaluation
        },
        "fit": _fit_arms(evaluation, dict(fit_reports or {})),
        "results": {
            key: value for key, value in evaluation.items() if key not in hoisted
        },
        "baseline": {
            "name": evaluation["baseline"],
            **{key: evaluation[key] for key in _BASELINE_KEYS if key in evaluation},
        },
        "implementation": {
            "method_version": VALIDATION_METHOD_VERSION,
            "source_files": list(VALIDATION_SOURCE_FILES),
            "source_sha256": source_fingerprint(VALIDATION_SOURCE_FILES),
        },
        "environment": _environment(),
    }


def write_corpus_validation_report(
    output_path: Path,
    corpus: str,
    evaluation_report: Path,
    fit_reports: Mapping[str, Path] | None = None,
    *,
    smoke: bool = False,
) -> None:
    document = assemble_corpus_validation_report(
        corpus, evaluation_report, fit_reports, smoke=smoke
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}")


# ---------------------------------------------------------------------------
# The Cascade X8 assembly, which combines several per-step reports.


def _strip_per_trajectory(value: Any) -> Any:
    """Drop every ``per_trajectory`` field; it is development-scale detail."""

    if isinstance(value, dict):
        return {
            key: _strip_per_trajectory(item)
            for key, item in value.items()
            if key != "per_trajectory"
        }
    if isinstance(value, list):
        return [_strip_per_trajectory(item) for item in value]
    return value


_CASCADE_DIAGNOSTIC_NAMES = (
    "diagnostic_cg005_w04_i1",
    "diagnostic_cg005_w04_i3.5",
    "diagnostic_skywalker_x8_cg005_w04",
    "diagnostic_skywalker_x8_panels_cg005_w04",
)


def assemble_cascade_x8_validation_report(
    x8_cascade_dir: Path, x8_reference_dir: Path
) -> dict[str, Any]:
    """Assemble ``cascade-x8-validation-results.json`` in its recorded shape.

    Ported from the one-off script used to produce the checked-in artifact.
    It takes the table-variant Cascade evaluation report, strips
    per-trajectory detail, and folds in the Glassbox reference model scores
    (from the X8 benchmark report), the residual regression diagnostics
    (aircraft/channels/configuration only, from each ``diagnose-cascade``
    report), and the component-panel comparison when that report is present.
    """

    with (x8_cascade_dir / "cascade_report.json").open() as handle:
        cascade_report = json.load(handle)
    with (x8_reference_dir / "benchmark_report.json").open() as handle:
        benchmark_report = json.load(handle)

    document = _strip_per_trajectory(cascade_report)
    document["glassbox_reference_models"] = {
        name: {
            "aggregate": _strip_per_trajectory(entry["aggregate"]),
            "score_vs_kinematic_persistence": entry["score_vs_kinematic_persistence"],
        }
        for name, entry in benchmark_report["models"].items()
    }

    document["residual_diagnostics"] = {}
    for diagnostic_name in _CASCADE_DIAGNOSTIC_NAMES:
        with (x8_cascade_dir / f"{diagnostic_name}.json").open() as handle:
            diagnostic = json.load(handle)
        document["residual_diagnostics"][diagnostic_name] = {
            key: diagnostic[key] for key in ("aircraft", "channels", "configuration")
        }

    panels_path = x8_cascade_dir / "cascade_panels_report.json"
    if panels_path.exists():
        with panels_path.open() as handle:
            panels_report = json.load(handle)
        best = panels_report["best_model"]
        document["component_panels"] = {
            "table_best_variant": panels_report["models"][cascade_report["best_model"]][
                "score_vs_kinematic_persistence"
            ],
            "best_model": best,
            "primary_model": panels_report["primary_model"],
            "scores_vs_kinematic_persistence": {
                name: entry["score_vs_kinematic_persistence"]
                for name, entry in panels_report["models"].items()
            },
            "best_model_aggregate": _strip_per_trajectory(
                panels_report["models"][best]["aggregate"]
            ),
        }

    return dict(sorted(document.items()))


def write_cascade_x8_validation_report(
    output_path: Path, x8_cascade_dir: Path, x8_reference_dir: Path
) -> None:
    document = assemble_cascade_x8_validation_report(x8_cascade_dir, x8_reference_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}")


def _copy_diagnostic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(f"copied {source} to {destination}")


# ---------------------------------------------------------------------------
# The manifest.

ADAPTIVE_RECOVERY_VOLATILE = (
    "environment",
    "git_revision",
    "implementation.source_files",
    "implementation.source_sha256",
    "recovery[*].prewarm_wall_time_s",
    "recovery[*].solve_time_median_s",
    "recovery[*].solve_time_p90_s",
    "recovery[*].solve_time_maximum_s",
)
"""Host- and source-dependent paths in ``adaptive-recovery-results.json``.

The source digest records which sources produced the numbers; it changes with
any edit, including one that leaves every number alone, so it is provenance
rather than a freshness gate and does not take part in a comparison.
"""

NMPC_ACCEPTANCE_VOLATILE = (
    "environment",
    "summary.post_jit_solve_time_s",
    "scenarios[*].warmup_solve_time_s",
    "scenarios[*].solve_time_median_s",
    "scenarios[*].solve_time_p90_s",
    "scenarios[*].solve_time_maximum_s",
)

VALIDATION_VOLATILE = (
    "environment",
    "implementation.source_files",
    "implementation.source_sha256",
    "fit.*.fit.wall_time_s",
)


@dataclass(frozen=True)
class FitArm:
    """One fitted model in a corpus chain: its report file and its command."""

    report: str
    command: str


def _corpus_validation(
    plan: RecordingPlan,
    *,
    name: str,
    corpus: str,
    directory: str,
    doc_page: str,
    fits: Mapping[str, FitArm],
    evaluation: str,
    evaluation_report: str,
    awaiting_first_record: str | None = None,
) -> ArtifactSpec:
    """One corpus validation entry: prepare, fit each arm, evaluate, assemble.

    ``fits`` maps each model the evaluation scores to the ``glassbox fit`` that
    produced it, ``evaluation`` is the one ``glassbox evaluate`` that scores
    them, and ``evaluation_report`` is the file that evaluation writes,
    relative to the corpus directory. Every path comes from ``plan``, so a
    recording, a check and a smoke run are the same chain pointed at different
    directories.
    """

    work = plan.work(directory)
    output = plan.output(f"{name}.json")
    reports = {arm: work / item.report for arm, item in fits.items()}
    steps: list[Step] = [
        _cli(f"corpus prepare {corpus} {work} {plan.raw(directory)}"),
        *(_cli(f"fit {item.command} {plan.steps()}") for item in fits.values()),
        _cli(f"evaluate {evaluation}"),
        PythonStep(
            f"assemble {output} from {work}",
            lambda: write_corpus_validation_report(
                output, corpus, work / evaluation_report, reports, smoke=plan.smoke
            ),
        ),
    ]
    return ArtifactSpec(
        name=name,
        output=str(output),
        steps=tuple(steps),
        extra=REFERENCE_CORPORA[corpus].extra,
        inputs=(
            f"corpus {corpus} pinned at "
            f"{REFERENCE_CORPORA[corpus].citation.pinned_version}",
        ),
        tier=CORPUS_TIER,
        doc_page=doc_page,
        volatile=VALIDATION_VOLATILE,
        awaiting_first_record=awaiting_first_record,
    )


_PENDING_CORPUS_RECORD = "awaiting the maintainer's first corpus run"

RECORDING = RecordingPlan()
"""The default plan: the corpora under ``artifacts``, the artifacts committed."""


def build_manifest(plan: RecordingPlan = RECORDING) -> tuple[ArtifactSpec, ...]:
    """Build the eight-artifact manifest for one recording plan."""

    nanodrone = plan.work("nanodrone")
    arp = plan.work("arp_reference")
    idf = plan.work("idf_reference")
    x8 = plan.work("x8_reference")
    cascade = plan.work("x8_cascade")
    epfl = plan.work("epfl_topoplane")
    recovery = plan.output("adaptive-recovery-results.json")
    nmpc = plan.output("nmpc-acceptance-results.json")
    cascade_artifact = plan.output("cascade-x8-validation-results.json")
    x8_split = "--holdout-label benchmark_split=validation"
    x8_training = f"{x8}/canonical/training/*.npz {x8}/canonical/validation/*.npz"
    cascade_sweep = (
        "--cg-shifts 0,0.03,0.05 --inertia-scales 1,2,3.5 "
        "--vertical-wind-fractions 0.25,0.5,1"
    )
    cascade_diagnostic = (
        f"benchmark cascade-x8 {cascade} --diagnose --split all "
        "--cg-shift 0.05 --vertical-wind-fraction 0.4"
    )

    return (
        ArtifactSpec(
            name="adaptive-recovery-results",
            output=str(recovery),
            steps=(_cli(f"benchmark recovery --output {recovery}"),),
            inputs=("synthetic scenarios generated in-process",),
            tier=LOCAL_TIER,
            doc_page="docs/validation.md#adaptive-recovery",
            volatile=ADAPTIVE_RECOVERY_VOLATILE,
        ),
        ArtifactSpec(
            name="nmpc-acceptance-results",
            output=str(nmpc),
            steps=(_cli(f"benchmark nmpc --output {nmpc}"),),
            inputs=("synthetic scenarios generated in-process",),
            tier=LOCAL_TIER,
            doc_page="docs/validation.md#nmpc-acceptance",
            volatile=NMPC_ACCEPTANCE_VOLATILE,
        ),
        _corpus_validation(
            plan,
            name="validation-nanodrone-results",
            corpus="nanodrone",
            directory="nanodrone",
            doc_page="docs/validation.md#nano-quadrotor",
            fits={
                "structured_residual": FitArm(
                    "report.json",
                    f"{nanodrone}/canonical/train/*.npz "
                    f"{nanodrone}/canonical/test/*.npz "
                    "--holdout-profile melon "
                    "--training-horizons 0.1,0.5,1.0 "
                    "--evaluation-horizons 0.1,0.5,1.0 "
                    "--model-class structured_residual "
                    f"--model {nanodrone}/model.json "
                    f"--report {nanodrone}/report.json",
                )
            },
            evaluation=(
                f"{nanodrone}/model.json {nanodrone}/canonical/test/*.npz "
                "--protocol nanodrone --corpus nanodrone "
                f"--report {nanodrone}/benchmark_report.json"
            ),
            evaluation_report="benchmark_report.json",
        ),
        _corpus_validation(
            plan,
            name="validation-arp-results",
            corpus="arp",
            directory="arp_reference",
            doc_page="docs/validation.md#arp",
            fits={
                "structured": FitArm(
                    "report.json",
                    f"{arp}/canonical/*.npz --holdout-count 1 "
                    "--training-horizons 0.1,0.5,2.0 "
                    f"--model {arp}/model.json --report {arp}/report.json",
                )
            },
            evaluation=(
                f"{arp}/model.json {arp}/canonical/log_66*.npz "
                "--protocol windowed --corpus arp "
                f"--report {arp}/benchmark_report.json"
            ),
            evaluation_report="benchmark_report.json",
        ),
        _corpus_validation(
            plan,
            name="validation-idf-results",
            corpus="idf",
            directory="idf_reference",
            doc_page="docs/validation.md#idf-ds",
            fits={},
            evaluation=(
                f"--hold-out source_group {idf}/canonical/*.npz "
                "--model-class structured_residual --output-dir "
                f"{idf}/source_benchmark_structured_residual "
                f"{plan.steps()} {plan.folds()}"
            ),
            evaluation_report="source_benchmark_structured_residual/summary.json",
            awaiting_first_record=_PENDING_CORPUS_RECORD,
        ),
        _corpus_validation(
            plan,
            name="validation-x8-results",
            corpus="x8",
            directory="x8_reference",
            doc_page="docs/validation.md#skywalker-x8",
            fits={
                "structured": FitArm(
                    "structured_report.json",
                    f"{x8_training} {x8_split} --training-horizons 0.1,0.5,2.0 "
                    f"--model {x8}/structured_model.json "
                    f"--report {x8}/structured_report.json",
                ),
                "structured_residual": FitArm(
                    "residual_report.json",
                    f"{x8_training} {x8_split} --training-horizons 0.1,0.5,2.0 "
                    "--model-class structured_residual "
                    f"--model {x8}/residual_model.json "
                    f"--report {x8}/residual_report.json",
                ),
            },
            evaluation=(
                f"{x8}/canonical/validation/*.npz --protocol x8 --corpus x8 "
                f"--model structured={x8}/structured_model.json "
                f"--model structured_residual={x8}/residual_model.json "
                f"--report {x8}/benchmark_report.json"
            ),
            evaluation_report="benchmark_report.json",
        ),
        _corpus_validation(
            plan,
            name="validation-epfl-results",
            corpus="epfl",
            directory="epfl_topoplane",
            doc_page="docs/validation.md#epfl-topoplane2",
            fits={
                "structured": FitArm(
                    "structured_report.json",
                    f"{epfl}/canonical/*.npz "
                    "--evaluation-horizons 0.2,0.5,1,2 --training-horizons 0.2,1,2 "
                    "--holdout-count 2 "
                    f"--model {epfl}/structured_model.json "
                    f"--report {epfl}/structured_report.json",
                ),
                "structured_residual": FitArm(
                    "residual_report.json",
                    f"{epfl}/canonical/*.npz "
                    "--evaluation-horizons 0.2,0.5,1,2 --training-horizons 0.2,1,2 "
                    "--holdout-count 2 --model-class structured_residual "
                    f"--model {epfl}/residual_model.json "
                    f"--report {epfl}/residual_report.json",
                ),
            },
            evaluation=(
                "--fit-reports "
                f"structured={epfl}/structured_report.json "
                f"structured_residual={epfl}/residual_report.json "
                "--horizons 0.2,0.5,1,2 --score-horizons 0.5,1,2 "
                f"--report {epfl}/characterization_report.json"
            ),
            evaluation_report="characterization_report.json",
        ),
        ArtifactSpec(
            name="cascade-x8-validation-results",
            output=str(cascade_artifact),
            steps=(
                _cli(f"corpus prepare x8 {x8} {plan.raw('x8_reference')}"),
                _cli(f"corpus prepare x8 {cascade} --raw {x8}/raw"),
                _cli(
                    f"fit {x8_training} {x8_split} --training-horizons 0.1,0.5,2.0 "
                    f"--model {x8}/structured_model.json "
                    f"--report {x8}/structured_report.json {plan.steps()}"
                ),
                _cli(
                    f"fit {x8_training} {x8_split} --training-horizons 0.1,0.5,2.0 "
                    "--model-class structured_residual "
                    f"--model {x8}/residual_model.json "
                    f"--report {x8}/residual_report.json {plan.steps()}"
                ),
                _cli(
                    f"evaluate {x8}/canonical/validation/*.npz "
                    "--protocol x8 --corpus x8 "
                    f"--model structured={x8}/structured_model.json "
                    f"--model structured_residual={x8}/residual_model.json "
                    f"--report {x8}/benchmark_report.json"
                ),
                _cli(
                    f"benchmark cascade-x8 {cascade} "
                    f"--output {cascade}/cascade_report.json "
                    f"--reference-report {x8}/benchmark_report.json {cascade_sweep}"
                ),
                _cli(
                    f"benchmark cascade-x8 {cascade} --aircraft skywalker_x8_panels "
                    f"--output {cascade}/cascade_panels_report.json "
                    f"--reference-report {x8}/benchmark_report.json {cascade_sweep}"
                ),
                _cli(
                    f"{cascade_diagnostic} --inertia-scale 1 --mass 3.364 "
                    f"--output {cascade}/diagnostic_cg005_w04_i1.json"
                ),
                _cli(
                    f"{cascade_diagnostic} --inertia-scale 3.5 --mass 3.364 "
                    f"--output {cascade}/diagnostic_cg005_w04_i3.5.json"
                ),
                _cli(
                    f"benchmark cascade-x8 {cascade} --diagnose "
                    "--aircraft skywalker_x8_panels --split all --cg-shift 0.05 "
                    "--vertical-wind-fraction 0.4 --inertia-scale 1 --mass 3.364 "
                    f"--output {cascade}/diagnostic_skywalker_x8_panels_cg005_w04.json"
                ),
                PythonStep(
                    f"copy {cascade}/diagnostic_cg005_w04_i1.json to "
                    f"{cascade}/diagnostic_skywalker_x8_cg005_w04.json",
                    lambda: _copy_diagnostic(
                        cascade / "diagnostic_cg005_w04_i1.json",
                        cascade / "diagnostic_skywalker_x8_cg005_w04.json",
                    ),
                ),
                PythonStep(
                    f"assemble {cascade_artifact} from {cascade} and {x8}",
                    lambda: write_cascade_x8_validation_report(
                        cascade_artifact, cascade, x8
                    ),
                ),
            ),
            extra="cascade",
            inputs=("corpus x8", "the Cascade fixed-wing simulator"),
            tier=CORPUS_TIER,
            doc_page="docs/validation.md#cascade-x8",
        ),
    )


MANIFEST: tuple[ArtifactSpec, ...] = build_manifest()


# ---------------------------------------------------------------------------
# Selection and running.


def _importable(module_name: str) -> bool:
    """Return whether an optional extra is installed, without importing it.

    Listing must not trigger an optional extra's import-time side effects; a
    real import only happens when a step that needs the extra is dispatched,
    and its actionable error surfaces there.
    """

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def missing_requirements(spec: ArtifactSpec) -> list[str]:
    """Human-readable reasons ``spec`` cannot run right now, if any."""

    if spec.extra is not None and not _importable(_EXTRA_MODULES[spec.extra]):
        return [f"needs the optional '{spec.extra}' extra"]
    return []


def resolve_names(
    manifest: Sequence[ArtifactSpec], names: Sequence[str]
) -> list[ArtifactSpec]:
    """Return the named manifest entries, or say which names are unknown."""

    by_name = {spec.name: spec for spec in manifest}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise UnknownArtifactNames(unknown)
    return [by_name[name] for name in names]


def select_artifacts(
    manifest: Sequence[ArtifactSpec],
    *,
    only: Sequence[str] | None = None,
    tier: str = LOCAL_TIER,
) -> list[ArtifactSpec]:
    """Choose which manifest entries a run or dry run would act on.

    ``only`` selects specific artifacts by name and, being explicit, bypasses
    the tier gate; each selected entry can still be skipped individually for a
    missing extra or missing local data. Otherwise every artifact in ``tier``
    is selected: the ``local`` tier is what runs in this repository
    with no download, and the ``corpus`` tier is the maintainer job that needs
    a pinned corpus on disk.
    """

    if only:
        return resolve_names(manifest, only)
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose one of {', '.join(TIERS)}")
    return [spec for spec in manifest if spec.tier == tier]


def run_selected(selected: Sequence[ArtifactSpec]) -> list[tuple[ArtifactSpec, str]]:
    """Run every selected artifact's steps in order; stop on the first failure.

    Returns one ``(spec, status)`` pair per selected artifact that was
    attempted, where ``status`` is ``"ok"`` or ``"skipped: <reason>"``. A step
    failure raises :class:`SystemExit` immediately instead of continuing to
    the next artifact.
    """

    results: list[tuple[ArtifactSpec, str]] = []
    for spec in selected:
        problems = missing_requirements(spec)
        if problems:
            reason = "; ".join(problems)
            print(f"skipping {spec.name}: {reason}")
            results.append((spec, f"skipped: {reason}"))
            continue
        print(f"regenerating {spec.name} -> {spec.output}")
        for index, step in enumerate(spec.steps, start=1):
            print(f"  [{index}/{len(spec.steps)}] {step.describe()}")
            try:
                step.run()
            except StepFailed as error:
                print(
                    f"FAILED at step {index}/{len(spec.steps)} of {spec.name}: {error}",
                    file=sys.stderr,
                )
                raise SystemExit(1) from error
        results.append((spec, "ok"))
    return results


# ---------------------------------------------------------------------------
# The freshness check.

_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _pattern(pattern: str) -> re.Pattern[str]:
    compiled = _PATTERN_CACHE.get(pattern)
    if compiled is None:
        body = re.escape(pattern).replace("\\*", ".*")
        compiled = re.compile(rf"(?:.*\.)?{body}\Z")
        _PATTERN_CACHE[pattern] = compiled
    return compiled


def matches_path(patterns: Sequence[str], path: str) -> bool:
    """Whether one dotted path matches any pattern.

    A pattern matches a path when it globs the whole path or a dot-separated
    suffix of it, so ``"environment"`` covers that block wherever it appears
    while ``"scenarios[*].solve_time_p90_s"`` stays specific. ``*`` is the only
    metacharacter; brackets are literal. This is the same rule the recorded
    tests use in ``tests/_recorded.py``.
    """

    return any(_pattern(pattern).match(path) is not None for pattern in patterns)


def recorded_differences(
    actual: Any,
    recorded: Any,
    *,
    volatile: Sequence[str] = (),
    tolerance: tuple[float, float] = DEFAULT_TOLERANCE,
    path: str = "report",
) -> list[str]:
    """Every way a fresh artifact differs from the committed one.

    Volatile paths are skipped entirely. Booleans, integers, strings and
    ``None`` must compare equal; a float may differ by the relative and
    absolute tolerance the artifact declares.
    """

    differences: list[str] = []
    _compare(actual, recorded, path, volatile, tolerance, differences)
    return differences


def _compare(
    actual: Any,
    recorded: Any,
    path: str,
    volatile: Sequence[str],
    tolerance: tuple[float, float],
    differences: list[str],
) -> None:
    if matches_path(volatile, path):
        return
    if isinstance(recorded, dict):
        if not isinstance(actual, dict):
            differences.append(f"{path}: recorded a mapping, produced {type(actual)}")
            return
        for key in sorted(set(recorded) ^ set(actual)):
            if not matches_path(volatile, f"{path}.{key}"):
                side = (
                    "recorded but not produced"
                    if key in recorded
                    else "produced but not recorded"
                )
                differences.append(f"{path}.{key}: {side}")
        for key in recorded:
            if key in actual:
                _compare(
                    actual[key],
                    recorded[key],
                    f"{path}.{key}",
                    volatile,
                    tolerance,
                    differences,
                )
        return
    if isinstance(recorded, list):
        if not isinstance(actual, list):
            differences.append(f"{path}: recorded a list, produced {type(actual)}")
            return
        if len(actual) != len(recorded):
            differences.append(
                f"{path}: recorded {len(recorded)} items, produced {len(actual)}"
            )
            return
        for index, (left, right) in enumerate(zip(actual, recorded)):
            _compare(left, right, f"{path}[{index}]", volatile, tolerance, differences)
        return
    numbers = (int, float)
    if (
        isinstance(recorded, numbers)
        and isinstance(actual, numbers)
        and not isinstance(recorded, bool)
        and not isinstance(actual, bool)
    ):
        if isinstance(recorded, int) and isinstance(actual, int):
            if actual != recorded:
                differences.append(
                    f"{path}: recorded {recorded!r}, produced {actual!r}"
                )
            return
        if not math.isfinite(recorded) or not math.isfinite(actual):
            if repr(actual) != repr(recorded):
                differences.append(
                    f"{path}: recorded {recorded!r}, produced {actual!r}"
                )
            return
        relative, absolute = tolerance
        if abs(actual - recorded) > relative * abs(recorded) + absolute:
            differences.append(
                f"{path}: recorded {recorded!r}, produced {actual!r} "
                f"(allowed {relative:g} relative plus {absolute:g} absolute)"
            )
        return
    if actual != recorded:
        differences.append(f"{path}: recorded {recorded!r}, produced {actual!r}")


@dataclass(frozen=True)
class CheckResult:
    """What the check found for one artifact."""

    spec: ArtifactSpec
    status: str
    differences: tuple[str, ...] = ()


def check_selected(
    selected: Sequence[ArtifactSpec], produced_root: Path
) -> list[CheckResult]:
    """Compare freshly produced artifacts against the committed ones.

    ``produced_root`` is where the run that produced them wrote. Each fresh
    artifact is compared against the file of the same name under
    ``docs/results/``, excluding the entry's volatile paths.
    """

    results: list[CheckResult] = []
    for spec in selected:
        filename = Path(spec.output).name
        fresh = produced_root / filename
        committed = _REPO_ROOT / "docs" / "results" / filename
        if not spec.recorded:
            results.append(CheckResult(spec, f"skipped: {spec.awaiting_first_record}"))
            continue
        if not fresh.exists():
            results.append(CheckResult(spec, "skipped: nothing produced"))
            continue
        differences = recorded_differences(
            json.loads(fresh.read_text()),
            json.loads(committed.read_text()),
            volatile=spec.volatile,
            tolerance=spec.tolerance,
        )
        results.append(
            CheckResult(
                spec,
                "ok" if not differences else "differs",
                tuple(differences),
            )
        )
    return results
