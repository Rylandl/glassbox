"""The manifest of recorded artifacts and the runner that regenerates them.

Recorded results are produced by many different CLI leaves and one bespoke
assembly step, and the command that regenerates each one lives on whatever
experiment or concept page cites it. This module owns one manifest naming
every artifact under ``docs/results/``: whether it can be regenerated in this
repository, the exact ``glassbox`` argv steps that reproduce it, what optional
extra and local data it needs, and how long it takes.

Every step is a ``glassbox`` subcommand run in-process through
:func:`glassbox.cli.main`, except the Cascade X8 assembly step, which has no
subcommand form and instead calls :func:`assemble_cascade_x8_validation_report`
directly. Nothing here writes documentation prose: after regenerating an
artifact, update the prose on its page by hand from the new JSON.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import glassbox.cli as cli

_REPO_ROOT = Path(__file__).resolve().parents[3]

_EXTRA_MODULES = {"cascade": "cascade"}

LOCAL_TIER = "local"
CORPUS_TIER = "corpus"
TIERS = (LOCAL_TIER, CORPUS_TIER)


class StepFailed(RuntimeError):
    """Raised by a step that could not complete; carries a human message."""


class UnknownArtifactNames(Exception):
    """Raised when ``--only`` names an artifact absent from the manifest."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        super().__init__(f"unknown artifact name(s): {', '.join(self.names)}")


def _expand_globs(argv: Sequence[str]) -> list[str]:
    """Expand any glob-pattern token, relative to the current directory.

    Steps are written exactly as the documented shell command, including
    glob patterns such as ``artifacts/x8_reference/canonical/training/*.npz``
    that a shell would normally expand before ``glassbox`` ever saw them.
    Dispatching in-process bypasses the shell, so expansion happens here,
    lazily, only when a step actually runs.
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
class ArtifactSpec:
    """One artifact under ``docs/results/`` and how to regenerate it."""

    name: str
    output: str
    steps: tuple[Step, ...] = ()
    extra: str | None = None
    required_data: tuple[str, ...] = ()
    tier: str = "local"
    doc_page: str = ""
    unavailable_reason: str | None = None

    @property
    def regenerable(self) -> bool:
        return self.unavailable_reason is None


def _cli(*argv: str) -> CliStep:
    return CliStep(argv)


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


_X8_FIT_TRAJECTORIES = (
    "artifacts/x8_reference/canonical/training/*.npz",
    "artifacts/x8_reference/canonical/validation/*.npz",
)

MANIFEST: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        name="adaptive-recovery-results",
        output="docs/results/adaptive-recovery-results.json",
        steps=(
            _cli(
                "benchmark",
                "recovery",
                "--output",
                "docs/results/adaptive-recovery-results.json",
            ),
        ),
        tier="local",
        doc_page="docs/experiments/adaptive-recovery.md",
    ),
    ArtifactSpec(
        name="nmpc-acceptance-results",
        output="docs/results/nmpc-acceptance-results.json",
        steps=(
            _cli(
                "benchmark",
                "nmpc",
                "--output",
                "docs/results/nmpc-acceptance-results.json",
            ),
        ),
        tier="local",
        doc_page="docs/concepts/nmpc.md",
    ),
    ArtifactSpec(
        name="cascade-x8-validation-results",
        output="docs/results/cascade-x8-validation-results.json",
        steps=(
            _cli("corpus", "prepare", "x8", "artifacts/x8_reference"),
            _cli(
                "corpus",
                "prepare",
                "x8",
                "artifacts/x8_cascade",
                "--raw",
                "artifacts/x8_reference/raw",
            ),
            _cli(
                "fit",
                *_X8_FIT_TRAJECTORIES,
                "--holdout-label",
                "benchmark_split=validation",
                "--training-horizons",
                "0.1,0.5,2.0",
                "--model",
                "artifacts/x8_reference/structured_model.json",
                "--report",
                "artifacts/x8_reference/structured_report.json",
            ),
            _cli(
                "fit",
                *_X8_FIT_TRAJECTORIES,
                "--holdout-label",
                "benchmark_split=validation",
                "--training-horizons",
                "0.1,0.5,2.0",
                "--model-class",
                "structured_residual",
                "--model",
                "artifacts/x8_reference/residual_model.json",
                "--report",
                "artifacts/x8_reference/residual_report.json",
            ),
            _cli(
                "evaluate",
                "artifacts/x8_reference/canonical/validation/*.npz",
                "--protocol",
                "x8",
                "--corpus",
                "x8",
                "--model",
                "structured=artifacts/x8_reference/structured_model.json",
                "--model",
                "structured_residual=artifacts/x8_reference/residual_model.json",
                "--report",
                "artifacts/x8_reference/benchmark_report.json",
            ),
            _cli(
                "benchmark",
                "cascade-x8",
                "artifacts/x8_cascade",
                "--output",
                "artifacts/x8_cascade/cascade_report.json",
                "--reference-report",
                "artifacts/x8_reference/benchmark_report.json",
                "--cg-shifts",
                "0,0.03,0.05",
                "--inertia-scales",
                "1,2,3.5",
                "--vertical-wind-fractions",
                "0.25,0.5,1",
            ),
            _cli(
                "benchmark",
                "cascade-x8",
                "artifacts/x8_cascade",
                "--aircraft",
                "skywalker_x8_panels",
                "--output",
                "artifacts/x8_cascade/cascade_panels_report.json",
                "--reference-report",
                "artifacts/x8_reference/benchmark_report.json",
                "--cg-shifts",
                "0,0.03,0.05",
                "--inertia-scales",
                "1,2,3.5",
                "--vertical-wind-fractions",
                "0.25,0.5,1",
            ),
            _cli(
                "benchmark",
                "cascade-x8",
                "artifacts/x8_cascade",
                "--diagnose",
                "--split",
                "all",
                "--cg-shift",
                "0.05",
                "--vertical-wind-fraction",
                "0.4",
                "--inertia-scale",
                "1",
                "--mass",
                "3.364",
                "--output",
                "artifacts/x8_cascade/diagnostic_cg005_w04_i1.json",
            ),
            _cli(
                "benchmark",
                "cascade-x8",
                "artifacts/x8_cascade",
                "--diagnose",
                "--split",
                "all",
                "--cg-shift",
                "0.05",
                "--vertical-wind-fraction",
                "0.4",
                "--inertia-scale",
                "3.5",
                "--mass",
                "3.364",
                "--output",
                "artifacts/x8_cascade/diagnostic_cg005_w04_i3.5.json",
            ),
            _cli(
                "benchmark",
                "cascade-x8",
                "artifacts/x8_cascade",
                "--diagnose",
                "--aircraft",
                "skywalker_x8_panels",
                "--split",
                "all",
                "--cg-shift",
                "0.05",
                "--vertical-wind-fraction",
                "0.4",
                "--inertia-scale",
                "1",
                "--mass",
                "3.364",
                "--output",
                "artifacts/x8_cascade/diagnostic_skywalker_x8_panels_cg005_w04.json",
            ),
            PythonStep(
                "copy artifacts/x8_cascade/diagnostic_cg005_w04_i1.json to "
                "artifacts/x8_cascade/diagnostic_skywalker_x8_cg005_w04.json",
                lambda: _copy_diagnostic(
                    _REPO_ROOT / "artifacts/x8_cascade/diagnostic_cg005_w04_i1.json",
                    _REPO_ROOT
                    / "artifacts/x8_cascade/diagnostic_skywalker_x8_cg005_w04.json",
                ),
            ),
            PythonStep(
                "assemble docs/results/cascade-x8-validation-results.json from "
                "artifacts/x8_cascade and artifacts/x8_reference",
                lambda: write_cascade_x8_validation_report(
                    _REPO_ROOT / "docs/results/cascade-x8-validation-results.json",
                    _REPO_ROOT / "artifacts/x8_cascade",
                    _REPO_ROOT / "artifacts/x8_reference",
                ),
            ),
        ),
        extra="cascade",
        required_data=("artifacts/x8_reference/raw",),
        tier="corpus",
        doc_page="docs/experiments/cascade-x8-validation.md",
    ),
    ArtifactSpec(
        name="multirotor-profile-results",
        output="docs/results/multirotor-profile-results.json",
        tier="corpus",
        doc_page="docs/experiments/px4-sitl-multirotor.md",
        unavailable_reason=(
            "PX4 SITL multirotor corpus rerun; source ULogs are not checked into "
            "the repo"
        ),
    ),
)


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

    if not spec.regenerable:
        return [f"not regenerable: {spec.unavailable_reason}"]
    problems: list[str] = []
    if spec.extra is not None and not _importable(_EXTRA_MODULES[spec.extra]):
        problems.append(f"needs the optional '{spec.extra}' extra")
    for relative in spec.required_data:
        if not (_REPO_ROOT / relative).exists():
            problems.append(f"needs data at {relative}")
    return problems


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
    missing extra or missing local data. Otherwise every regenerable artifact
    in ``tier`` is selected: the ``local`` tier is what runs in this repository
    with no download, and the ``corpus`` tier is the maintainer job that needs
    a pinned corpus on disk.
    """

    if only:
        return resolve_names(manifest, only)
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose one of {', '.join(TIERS)}")
    return [spec for spec in manifest if spec.regenerable and spec.tier == tier]


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
