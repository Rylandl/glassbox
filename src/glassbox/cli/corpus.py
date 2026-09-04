"""Download and convert the pinned public reference corpora.

``list`` prints the registry: every corpus's name, the optional extra its
parser needs, the published scoring protocol its evaluation uses, and how it is
cited. It reads no files and needs no extra, so it answers "what can Glassbox
ingest" on a bare install.

``fetch NAME DIR`` downloads that corpus's pinned files under ``DIR`` and
verifies each one's size and digest. A file already there that matches is kept
without any network access; one that does not match is an error, and
``--overwrite`` is the only way past it. There is no way to skip verification:
a corpus that does not verify is a different corpus, and a number measured on
it is not comparable to the published one.

``prepare NAME DIR`` fetches into ``DIR/raw`` and converts every recording into
canonical trajectory NPZ files under ``DIR/canonical``, preserving the upstream
train/test split as subdirectories where the corpus publishes one. ``--raw``
reuses an already-verified source tree elsewhere, so preparing a second
canonical copy of one corpus does not download it twice. Corpora whose
recordings carry a published audit also write ``DIR/corpus_report.json``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from glassbox.io.corpus import REFERENCE_CORPORA


def _list(args: argparse.Namespace) -> None:
    width = max(len(name) for name in REFERENCE_CORPORA)
    for name, corpus in REFERENCE_CORPORA.items():
        extra = f"[{corpus.extra}]" if corpus.extra else ""
        protocol = corpus.protocol or "-"
        print(f"{name:<{width}}  {protocol:<9} {extra:<7} {corpus.summary}")
        print(
            f"{'':<{width}}  {corpus.citation.doi_or_url}  "
            f"{corpus.citation.license}  pinned at {corpus.citation.pinned_version}"
        )
        if corpus.validation_split is not None:
            print(f"{'':<{width}}  evaluation split: {corpus.validation_split}")


def _fetch(args: argparse.Namespace) -> None:
    corpus = REFERENCE_CORPORA[args.name]
    paths = corpus.fetch(args.directory, overwrite=args.overwrite)
    print(
        f"verified {len(paths)} pinned {corpus.name} files "
        f"({corpus.citation.pinned_version}) under {args.directory}"
    )


def _prepare(args: argparse.Namespace) -> None:
    corpus = REFERENCE_CORPORA[args.name]
    prepared = corpus.prepare(
        args.directory, overwrite=args.overwrite, raw_root=args.raw
    )
    print(
        f"prepared {corpus.name} ({corpus.citation.pinned_version}): "
        f"{len(prepared.sources)} verified sources in {prepared.raw_root}, "
        f"{len(prepared.trajectories)} trajectories "
        f"({prepared.duration_s:.1f}s) in {prepared.canonical_root}"
    )
    if prepared.report_path is not None:
        print(f"wrote corpus audit {prepared.report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="print every pinned corpus, its extra, protocol, and citation"
    )
    list_parser.set_defaults(handler=_list)

    fetch_parser = subparsers.add_parser(
        "fetch", help="download and verify one corpus's pinned files"
    )
    fetch_parser.add_argument("name", choices=tuple(REFERENCE_CORPORA))
    fetch_parser.add_argument("directory", type=Path)
    fetch_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing file that does not match its pinned digest",
    )
    fetch_parser.set_defaults(handler=_fetch)

    prepare_parser = subparsers.add_parser(
        "prepare", help="fetch one corpus and convert it to canonical NPZ files"
    )
    prepare_parser.add_argument("name", choices=tuple(REFERENCE_CORPORA))
    prepare_parser.add_argument("directory", type=Path)
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing source file that does not match its pinned digest",
    )
    prepare_parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="verified source tree to reuse instead of DIRECTORY/raw",
    )
    prepare_parser.set_defaults(handler=_prepare)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
