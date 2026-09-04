"""Regenerate the recorded artifacts under ``docs/results/`` from one manifest.

Every number quoted in the documentation is the literal content of one
artifact, and this command is how each of those artifacts is produced. The
manifest names, for every artifact, the exact ``glassbox`` steps that
reproduce it, the optional extra and local data it needs, and the page whose
prose cites it.

``--tier local`` is what runs in this repository with nothing downloaded, and
is what continuous integration runs. ``--tier corpus`` is the maintainer job
that needs a pinned corpus on disk and, for the Cascade rows, the ``cascade``
extra; an artifact whose requirements are not met is skipped by name and
reason rather than failing the run. ``--only NAME`` regenerates artifacts by
name and ignores the tier gate.

``--list`` prints the whole manifest with each artifact's tier and status.
``--dry-run`` prints the steps that would run; with neither ``--only`` nor
``--tier`` it prints the plan for every artifact in the manifest.

This command writes JSON, never prose. After regenerating an artifact, update
the numbers on its page by hand from the new file.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from glassbox.workflows.record_results import (
    MANIFEST,
    TIERS,
    ArtifactSpec,
    UnknownArtifactNames,
    missing_requirements,
    resolve_names,
    run_selected,
    select_artifacts,
)


def _status_text(spec: ArtifactSpec) -> str:
    problems = missing_requirements(spec)
    if not spec.regenerable:
        return f"not regenerable: {spec.unavailable_reason}"
    if problems:
        return f"blocked: {'; '.join(problems)}"
    return "regenerable"


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


def _print_dry_run(selected: Sequence[ArtifactSpec]) -> None:
    for spec in selected:
        problems = missing_requirements(spec)
        print(f"{spec.name} ({spec.tier}) -> {spec.output}")
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
        help="print the full manifest with tier and status, and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        try:
            shown = resolve_names(MANIFEST, args.only) if args.only else MANIFEST
        except UnknownArtifactNames as error:
            parser.error(str(error))
        _print_manifest(shown)
        return

    try:
        if args.dry_run and not args.only and args.tier is None:
            selected = list(MANIFEST)
        elif args.tier is None:
            selected = select_artifacts(MANIFEST, only=args.only)
        else:
            selected = select_artifacts(MANIFEST, only=args.only, tier=args.tier)
    except UnknownArtifactNames as error:
        parser.error(str(error))

    if args.dry_run:
        _print_dry_run(selected)
        return

    _print_summary(run_selected(selected))


if __name__ == "__main__":
    main()
