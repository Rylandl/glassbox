"""Regenerate the recorded artifacts under ``docs/results/`` from one manifest.

The local tier runs without downloads. The corpus tier needs pinned datasets
and the relevant extras. Missing inputs are reported as skips.

Use ``--check`` to compare fresh runs with recorded results, or ``--smoke DIR``
to exercise a corpus chain with a small fit budget outside the repository.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

from glassbox.workflows.record_results import (
    MANIFEST,
    TIERS,
    ArtifactSpec,
    CheckResult,
    RecordingPlan,
    UnknownArtifactNames,
    build_manifest,
    check_selected,
    missing_requirements,
    resolve_names,
    run_selected,
    select_artifacts,
)

SMOKE_FIT_STEPS = 2
SMOKE_FOLD_LIMIT = 2


def _status_text(spec: ArtifactSpec) -> str:
    problems = missing_requirements(spec)
    if problems:
        return f"blocked: {'; '.join(problems)}"
    if not spec.recorded:
        return f"pending: {spec.awaiting_first_record}"
    return "recorded"


def _print_manifest(manifest: Sequence[ArtifactSpec]) -> None:
    name_width = max((len(spec.name) for spec in manifest), default=0)
    tier_width = max((len(spec.tier) for spec in manifest), default=0)
    for spec in manifest:
        print(
            f"{spec.name:<{name_width}}  "
            f"{spec.tier:<{tier_width}}  "
            f"{_status_text(spec)}  "
            f"[{spec.output}]"
        )
        print(f"{'':<{name_width}}  inputs: {', '.join(spec.inputs) or 'none'}")


def _print_dry_run(selected: Sequence[ArtifactSpec]) -> None:
    for spec in selected:
        problems = missing_requirements(spec)
        print(f"{spec.name} ({spec.tier}) -> {spec.output}")
        print(f"  inputs: {', '.join(spec.inputs) or 'none'}")
        if problems:
            print(f"  skipped: {'; '.join(problems)}")
            continue
        for index, step in enumerate(spec.steps, start=1):
            print(f"  {index}. {step.describe()}")


def _print_summary(results: Sequence[tuple[ArtifactSpec, str]]) -> None:
    if not results:
        print("nothing selected")
        return
    print()
    print("summary:")
    name_width = max(len(spec.name) for spec, _ in results)
    for spec, status in results:
        print(f"  {spec.name:<{name_width}}  {status:<8}  {spec.doc_page}")
    print()
    print(
        "Doc prose numbers are hand-written and are not updated by this command. "
        "For each artifact regenerated above, update the numbers on its listed "
        "page from the new JSON."
    )


def _print_check(results: Sequence[CheckResult]) -> int:
    if not results:
        print("nothing selected")
        return 0
    name_width = max(len(item.spec.name) for item in results)
    for item in results:
        print(f"  {item.spec.name:<{name_width}}  {item.status}")
        for difference in item.differences:
            print(f"    {difference}")
    failed = [item for item in results if item.status == "differs"]
    if failed:
        print()
        print(
            f"{len(failed)} artifact(s) no longer reproduce. Re-record them with "
            "'glassbox record-results' and update the prose on their pages, or "
            "fix the change that moved the numbers."
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="regenerate only these artifacts by name; overrides the tier gate",
    )
    parser.add_argument(
        "--tier",
        choices=TIERS,
        help="regenerate one tier; local by default when running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the steps that would run, without running them",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the full manifest with tier, inputs and status, and exit",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "regenerate into a temporary directory and compare against the "
            "committed artifacts; exits non-zero when one no longer reproduces"
        ),
    )
    parser.add_argument(
        "--smoke",
        type=Path,
        metavar="DIR",
        help=(
            "exercise the corpus tier on a tiny fit budget, writing everything "
            "under DIR; the artifacts are marked as smoke output"
        ),
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        metavar="N",
        help=f"smoke fit budget in optimizer steps (default {SMOKE_FIT_STEPS})",
    )
    parser.add_argument(
        "--limit-folds",
        type=int,
        metavar="N",
        help=f"smoke holdout budget in folds (default {SMOKE_FOLD_LIMIT})",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        metavar="DIR",
        help="reuse verified corpus sources from DIR instead of downloading them",
    )
    return parser


def _smoke_plan(args: argparse.Namespace) -> RecordingPlan:
    return RecordingPlan(
        corpus_root=args.smoke,
        results_root=args.smoke,
        source_root=args.source_root,
        fit_steps=SMOKE_FIT_STEPS if args.fit_steps is None else args.fit_steps,
        fold_limit=SMOKE_FOLD_LIMIT if args.limit_folds is None else args.limit_folds,
        smoke=True,
    )


def main(argv: Sequence[str] | None = None) -> int | None:
    parser = build_parser()
    args = parser.parse_args(argv)

    for option, value in (("--smoke", args.smoke), ("--source-root", args.source_root)):
        if value is not None and any(character.isspace() for character in str(value)):
            parser.error(f"{option} cannot contain whitespace: {value}")
    if args.smoke is not None:
        local = [
            spec.name
            for spec in MANIFEST
            if spec.tier == "local"
            and (args.tier == "local" or (args.only and spec.name in args.only))
        ]
        if local:
            parser.error(
                "--smoke shortens a corpus fit and marks its output; the local "
                f"tier has neither: {', '.join(local)}"
            )
    if args.smoke is None and not (args.fit_steps is None and args.limit_folds is None):
        parser.error("--fit-steps and --limit-folds only shorten a --smoke run")
    if args.smoke is not None and args.check:
        parser.error("--check compares against the committed artifacts")

    if args.list:
        try:
            shown = resolve_names(MANIFEST, args.only) if args.only else MANIFEST
        except UnknownArtifactNames as error:
            parser.error(str(error))
        _print_manifest(shown)
        return None

    if args.check:
        return _run_check(args, parser)

    plan = RecordingPlan(source_root=args.source_root)
    tier = args.tier
    if args.smoke is not None:
        plan = _smoke_plan(args)
        tier = tier or "corpus"
    manifest = build_manifest(plan)

    try:
        if args.dry_run and not args.only and tier is None:
            selected = list(manifest)
        elif tier is None:
            selected = select_artifacts(manifest, only=args.only)
        else:
            selected = select_artifacts(manifest, only=args.only, tier=tier)
    except UnknownArtifactNames as error:
        parser.error(str(error))

    if args.dry_run:
        _print_dry_run(selected)
        return None

    _print_summary(run_selected(selected))
    return None


def _run_check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Regenerate into a temporary directory and compare with the committed set."""

    with tempfile.TemporaryDirectory(prefix="glassbox-check-") as directory:
        produced = Path(directory)
        plan = RecordingPlan(results_root=produced, source_root=args.source_root)
        manifest = build_manifest(plan)
        try:
            selected = select_artifacts(
                manifest,
                only=args.only,
                tier=args.tier or "local",
            )
        except UnknownArtifactNames as error:
            parser.error(str(error))
        run_selected([spec for spec in selected if spec.recorded])
        print()
        print("check:")
        return _print_check(check_selected(selected, produced))


if __name__ == "__main__":
    raise SystemExit(main())
