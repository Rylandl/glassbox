"""One registry of the pinned reference corpora and how to obtain them.

Every public corpus Glassbox can ingest is described here once: who published
it, which immutable files it consists of, which adapter parses them, which
optional extra that adapter needs, and which scoring protocol the published
evaluation uses. Fetching and preparing are then the same two operations for
all of them, implemented once over :mod:`glassbox.io.pinned_download`, instead
of five copies of the same download-and-convert loop.

An entry names its parser as a module path rather than importing it, because
``glassbox corpus list`` has to render in an environment where the ``px4`` and
``ros`` extras are absent. The named module supplies two attributes:

``PINNED_FILES``
    the corpus's immutable file table as a tuple of :class:`PinnedFile`.
``<adapter_type>``
    a frozen adapter class, constructed with no arguments, exposing
    ``inspect(path) -> Mapping[str, Any]`` and either ``load(path) ->
    Trajectory`` for a one-recording-one-trajectory corpus or ``load_all(path)
    -> tuple[Trajectory, ...]`` for a corpus whose recordings are split into
    segments. That concrete class, not a protocol, is the adapter contract:
    :attr:`ReferenceCorpus.adapter` returns an instance of it.

An entry may also name a ``unpack`` function that turns one pinned archive into
adapter inputs, a ``report`` function that audits the prepared corpus, and a
``validator`` that checks a set of trajectories against the corpus's published
evaluation split before a score is computed.

Verification is not optional. A pinned file whose size or digest does not match
is an error, on both the download and the parse path: a corpus that does not
verify is a different corpus, and a number measured on it is not comparable to
the published one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from glassbox.core.data import Trajectory, load_trajectory_npz, save_trajectory_npz
from glassbox.io.pinned_download import download_verified

__all__ = [
    "REFERENCE_CORPORA",
    "Citation",
    "PinnedFile",
    "PreparedCorpus",
    "ReferenceCorpus",
]


@dataclass(frozen=True)
class Citation:
    """How a corpus is published, licensed, and pinned."""

    doi_or_url: str
    license: str
    pinned_version: str


@dataclass(frozen=True)
class PinnedFile:
    """One immutable upstream file, verified by size and digest."""

    url: str
    relative_path: str
    size_bytes: int
    digest: str
    algorithm: str = "sha256"
    converted: bool = True
    """Whether :meth:`ReferenceCorpus.prepare` feeds this file to the adapter.

    A pinned README carries the corpus's own documentation and a pinned archive
    is unpacked first; neither is a recording.
    """

    @property
    def filename(self) -> str:
        return Path(self.relative_path).name


@dataclass(frozen=True)
class PreparedCorpus:
    """What one ``prepare`` produced, for the command line to report."""

    corpus: str
    raw_root: Path
    canonical_root: Path
    sources: tuple[Path, ...]
    trajectories: tuple[Path, ...]
    duration_s: float
    report_path: Path | None = None


@dataclass(frozen=True)
class ReferenceCorpus:
    """One pinned public flight corpus and the two commands that obtain it."""

    name: str
    citation: Citation
    source: str
    adapter_type: str
    extra: str | None = None
    protocol: str | None = None
    validation_split: str | None = None
    validator: str | None = None
    unpack: str | None = None
    report: str | None = None
    segments: bool = False
    by_split: bool = False
    stem: str = "{stem}"
    user_agent: str = "glassbox-corpus/1"
    timeout_s: float = 60.0
    summary: str = ""

    @property
    def files(self) -> tuple[PinnedFile, ...]:
        """The corpus's pinned file table, read from its parser module."""

        return tuple(self._module().PINNED_FILES)

    @property
    def adapter(self) -> Any:
        """A fresh instance of this corpus's concrete adapter class."""

        return getattr(self._module(), self.adapter_type)()

    def _module(self) -> Any:
        return import_module(self.source)

    def fetch(
        self, destination: str | Path, *, overwrite: bool = False
    ) -> tuple[Path, ...]:
        """Download and verify every pinned file under ``destination``.

        A file already present that matches its pinned size and digest is kept
        without any network access; one that does not match is an error unless
        ``overwrite`` is set.
        """

        root = Path(destination)
        return tuple(
            download_verified(
                item.url,
                root / item.relative_path,
                size_bytes=item.size_bytes,
                digest=item.digest,
                algorithm=item.algorithm,
                user_agent=self.user_agent,
                overwrite=overwrite,
                timeout_s=self.timeout_s,
                existing_mismatch_message=(
                    f"existing file does not match the pinned {self.name} "
                    f"corpus: {root / item.relative_path}"
                ),
                size_mismatch_message=(
                    f"downloaded size mismatch for {item.relative_path}"
                ),
                digest_mismatch_message=(
                    f"downloaded checksum mismatch for {item.relative_path}"
                ),
            )
            for item in self.files
        )

    def prepare(
        self,
        destination: str | Path,
        *,
        overwrite: bool = False,
        raw_root: str | Path | None = None,
    ) -> PreparedCorpus:
        """Fetch the corpus and convert it to canonical trajectory NPZ files.

        Sources land under ``destination/raw`` and canonical trajectories under
        ``destination/canonical``. ``raw_root`` reuses an already-verified
        source tree elsewhere, so a second canonical copy of one corpus does
        not download it twice.
        """

        root = Path(destination)
        raw = root / "raw" if raw_root is None else Path(raw_root)
        canonical = root / "canonical"
        fetched = self.fetch(raw, overwrite=overwrite)
        sources = self._adapter_inputs(raw, fetched, overwrite=overwrite)

        adapter = self.adapter
        outputs: list[Path] = []
        duration_s = 0.0
        for order, source in enumerate(sources, start=1):
            trajectories = (
                tuple(adapter.load_all(source))
                if self.segments
                else (adapter.load(source),)
            )
            for index, trajectory in enumerate(trajectories, start=1):
                stem = self.stem.format(
                    stem=source.stem, index=index, order=order, split=source.parent.name
                )
                directory = (
                    canonical / source.parent.name if self.by_split else canonical
                )
                output = directory / f"{stem}.npz"
                save_trajectory_npz(trajectory, output)
                duration_s += float(trajectory.time_s[-1] - trajectory.time_s[0])
                outputs.append(output)

        report_path = None
        if self.report is not None:
            report_path = root / "corpus_report.json"
            report = getattr(self._module(), self.report)(outputs, sources[0].parent)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        return PreparedCorpus(
            corpus=self.name,
            raw_root=raw,
            canonical_root=canonical,
            sources=tuple(sources),
            trajectories=tuple(outputs),
            duration_s=duration_s,
            report_path=report_path,
        )

    def _adapter_inputs(
        self, raw: Path, fetched: Sequence[Path], *, overwrite: bool
    ) -> tuple[Path, ...]:
        """Return the files the adapter parses, unpacking an archive first."""

        if self.unpack is not None:
            archive = next(
                raw / item.relative_path for item in self.files if not item.converted
            )
            return tuple(
                getattr(self._module(), self.unpack)(
                    archive, raw / "ulogs", overwrite=overwrite
                )
            )
        return tuple(path for path, item in zip(fetched, self.files) if item.converted)

    def load_evaluation_trajectories(
        self, paths: Sequence[str | Path]
    ) -> tuple[list[Path], list[Trajectory]]:
        """Load trajectories and check them against the published split.

        The published protocol names the flights a score may be taken on. A
        trajectory that is not one of them is a different measurement, not a
        worse number, so the check runs before the score.
        """

        resolved = [Path(path) for path in paths]
        if not resolved:
            raise ValueError(f"at least one {self.name} trajectory is required")
        trajectories = [load_trajectory_npz(path) for path in resolved]
        if self.validator is not None:
            getattr(self._module(), self.validator)(trajectories)
        return resolved, trajectories


REFERENCE_CORPORA: Mapping[str, ReferenceCorpus] = {
    corpus.name: corpus
    for corpus in (
        ReferenceCorpus(
            name="nanodrone",
            summary="IDSIA Nano-Quadrotor system-identification benchmark",
            citation=Citation(
                doi_or_url="10.1016/j.conengprac.2026.106871",
                license="upstream repository terms",
                pinned_version="2d921b57d166fe2debe08a5d39bd07297c5abc39",
            ),
            source="glassbox.io.nanodrone_reference",
            adapter_type="NanoDroneBenchmarkAdapter",
            protocol="nanodrone",
            validation_split="the three Melon recordings of the official test split",
            validator="validate_benchmark_test_trajectories",
            by_split=True,
            user_agent="glassbox-nanodrone-adapter/1",
        ),
        ReferenceCorpus(
            name="arp",
            summary="ARP Laboratory large-quadrotor system-identification ULogs",
            citation=Citation(
                doi_or_url="https://arxiv.org/abs/2404.07837",
                license="MIT",
                pinned_version="2d267dd07b4262f579ee223d20b26a6dc9d17147",
            ),
            source="glassbox.io.arp_reference",
            adapter_type="ARPReferenceAdapter",
            extra="px4",
            user_agent="glassbox-arp-reference-adapter/1",
        ),
        ReferenceCorpus(
            name="idf",
            summary="IDF-DS Holybro Pixhawk fixed-wing telemetry sessions",
            citation=Citation(
                doi_or_url="10.5281/zenodo.16992976",
                license="CC-BY-4.0",
                pinned_version="zenodo record 16992976",
            ),
            source="glassbox.io.idf_reference",
            adapter_type="IDFFixedWingAdapter",
            extra="px4",
            unpack="unpack_pinned_ulogs",
            report="idf_corpus_report",
            segments=True,
            # The pinned table is in session order, so {order} is the session
            # number; tests/test_corpus.py pins that correspondence.
            stem="idf_session_{order:02d}_{stem}_segment_{index:02d}",
            user_agent="glassbox-idf-reference-adapter/1",
        ),
        ReferenceCorpus(
            name="x8",
            summary="NTNU Skywalker X8 flying-wing system-identification campaign",
            citation=Citation(
                doi_or_url="10.18710/U4TLYV",
                license="CC0-1.0",
                pinned_version="1.0",
            ),
            source="glassbox.io.x8_reference",
            adapter_type="X8ReferenceAdapter",
            protocol="x8",
            validation_split="the upstream validation maneuvers",
            validator="validate_validation_trajectories",
            by_split=True,
            user_agent="glassbox-skywalker-x8-adapter/1",
        ),
        ReferenceCorpus(
            name="epfl",
            summary="EPFL TOPOPlane2 conventional fixed-wing navigation flight",
            citation=Citation(
                doi_or_url="10.5281/zenodo.10337559",
                license="CC-BY-4.0",
                pinned_version="v1",
            ),
            source="glassbox.io.epfl_reference",
            adapter_type="EPFLTopoplaneAdapter",
            extra="ros",
            segments=True,
            stem="topoplane2_segment_{index:02d}",
            user_agent="glassbox-epfl-topoplane-adapter/1",
            timeout_s=600.0,
        ),
    )
}
