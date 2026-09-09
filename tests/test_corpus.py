"""Contracts for the pinned reference-corpus registry.

The registry is the one place that says what Glassbox can ingest, so these
tests cover the listing, the pinned-file table, the refusal to accept a file
that does not verify, and one ``prepare`` per corpus over the same fixtures the
adapter tests parse. The adapter tests keep the parsing coverage; what is
pinned here is the plumbing they no longer own.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_arp_reference import _base_trajectory
from test_epfl_reference import _streams
from test_idf_reference import _trajectory as _idf_trajectory
from test_nanodrone_benchmark import _write_fixture as _write_nanodrone_csv
from test_x8_reference import _write_fixture as _write_x8_csv

import glassbox.io.arp_reference as arp_module
import glassbox.io.epfl_reference as epfl_module
import glassbox.io.idf_reference as idf_module
import glassbox.io.nanodrone_reference as nanodrone_module
import glassbox.io.x8_reference as x8_module
from glassbox.core.data import load_trajectory_npz
from glassbox.io.corpus import REFERENCE_CORPORA, PinnedFile

_CORPUS_NAMES = ("nanodrone", "arp", "idf", "x8", "epfl")


def _pin(path: Path, relative_path: str, *, algorithm: str, **extra) -> PinnedFile:
    """Pin an on-disk fixture so ``fetch`` reuses it without any network."""

    digest = hashlib.new(algorithm, path.read_bytes()).hexdigest()
    return PinnedFile(
        url=f"https://example.invalid/{relative_path}",
        relative_path=relative_path,
        size_bytes=path.stat().st_size,
        digest=digest,
        algorithm=algorithm,
        **extra,
    )


# ---------------------------------------------------------------------------
# The registry itself.


def test_registry_names_every_corpus_with_its_citation_and_protocol() -> None:
    assert tuple(REFERENCE_CORPORA) == _CORPUS_NAMES
    for name, corpus in REFERENCE_CORPORA.items():
        assert corpus.name == name
        assert corpus.summary
        assert corpus.citation.doi_or_url
        assert corpus.citation.license
        assert corpus.citation.pinned_version
        assert corpus.extra in (None, "px4", "ros")
        assert corpus.protocol in (None, "nanodrone", "x8")


def test_a_corpus_with_a_published_split_names_it_and_can_check_it() -> None:
    """The evaluation split contract belongs to the corpus, not the scorer."""

    for corpus in REFERENCE_CORPORA.values():
        assert (corpus.validation_split is None) == (corpus.validator is None)
        if corpus.validator is not None:
            assert corpus.protocol is not None
            assert callable(getattr(corpus._module(), corpus.validator))


@pytest.mark.parametrize("name", _CORPUS_NAMES)
def test_every_pinned_file_carries_a_url_size_and_digest(name: str) -> None:
    corpus = REFERENCE_CORPORA[name]
    assert corpus.files
    for item in corpus.files:
        assert item.url.startswith("https://")
        assert item.relative_path and not item.relative_path.startswith("/")
        assert item.size_bytes > 0
        assert item.algorithm in ("sha256", "md5")
        assert len(item.digest) == (64 if item.algorithm == "sha256" else 32)
    # Every corpus either ships recordings or ships one archive it unpacks.
    assert any(item.converted for item in corpus.files) or corpus.unpack is not None


def test_the_idf_pinned_table_is_in_session_order() -> None:
    """``prepare`` names IDF segments by table position, which is the session."""

    sessions = tuple(recording.session for recording in idf_module.IDF_RECORDINGS)
    assert sessions == tuple(range(1, len(sessions) + 1))


# ---------------------------------------------------------------------------
# Fetch: a pinned corpus that does not verify is an error.


def test_fetch_downloads_and_verifies_every_pinned_file(tmp_path, monkeypatch) -> None:
    payload = b"pinned corpus bytes"
    monkeypatch.setattr(
        nanodrone_module,
        "PINNED_FILES",
        (
            PinnedFile(
                url="https://example.invalid/data/train/one.csv",
                relative_path="data/train/one.csv",
                size_bytes=len(payload),
                digest=hashlib.sha256(payload).hexdigest(),
            ),
        ),
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: io.BytesIO(payload)
    )

    paths = REFERENCE_CORPORA["nanodrone"].fetch(tmp_path)

    assert paths == (tmp_path / "data" / "train" / "one.csv",)
    assert paths[0].read_bytes() == payload


def test_fetch_rejects_a_download_whose_digest_does_not_match(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        nanodrone_module,
        "PINNED_FILES",
        (
            PinnedFile(
                url="https://example.invalid/data/train/one.csv",
                relative_path="data/train/one.csv",
                size_bytes=4,
                digest="0" * 64,
            ),
        ),
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"beef")
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        REFERENCE_CORPORA["nanodrone"].fetch(tmp_path)
    assert not (tmp_path / "data" / "train" / "one.csv").exists()


def test_fetch_refuses_an_existing_file_that_does_not_match_its_pin(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "data" / "train" / "one.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"a different recording")
    monkeypatch.setattr(
        nanodrone_module,
        "PINNED_FILES",
        (
            PinnedFile(
                url="https://example.invalid/data/train/one.csv",
                relative_path="data/train/one.csv",
                size_bytes=4,
                digest="0" * 64,
            ),
        ),
    )

    with pytest.raises(FileExistsError, match="pinned nanodrone corpus"):
        REFERENCE_CORPORA["nanodrone"].fetch(tmp_path)


# ---------------------------------------------------------------------------
# Prepare: one verified source tree in, one canonical trajectory tree out.


def _prepare_nanodrone(tmp_path, monkeypatch):
    relative = "data/train/chirp_20251017_run1.csv"
    source = _write_nanodrone_csv(tmp_path / "raw" / relative)
    recording = nanodrone_module.BenchmarkRecording(
        relative,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        source.stat().st_size,
    )
    monkeypatch.setattr(nanodrone_module, "BENCHMARK_RECORDINGS", (recording,))
    monkeypatch.setattr(
        nanodrone_module, "_RECORDING_BY_FILENAME", {recording.filename: recording}
    )
    monkeypatch.setattr(
        nanodrone_module,
        "PINNED_FILES",
        (_pin(source, relative, algorithm="sha256"),),
    )
    return ("canonical/train/chirp_20251017_run1.npz",)


def _prepare_arp(tmp_path, monkeypatch):
    relative = "logs_large/log_63_2024-1-8-16-37-54.ulg"
    source = tmp_path / "raw" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pinned ARP ULog fixture")
    recording = arp_module.ARPRecording(
        relative,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        source.stat().st_size,
        1,
    )
    monkeypatch.setattr(arp_module, "ARP_RECORDINGS", (recording,))
    monkeypatch.setattr(
        arp_module, "_RECORDING_BY_FILENAME", {recording.filename: recording}
    )
    monkeypatch.setattr(
        arp_module, "PINNED_FILES", (_pin(source, relative, algorithm="sha256"),)
    )
    monkeypatch.setattr(
        arp_module, "load_px4_trajectory", lambda path, config: _base_trajectory()
    )
    return ("canonical/log_63_2024-1-8-16-37-54.npz",)


def _prepare_idf(tmp_path, monkeypatch):
    payload = b"raw ULog fixture"
    recording = idf_module.IDFRecording(
        "reference.ulg", len(payload), zlib.crc32(payload), 1, "2025-08-21", (1,)
    )
    monkeypatch.setattr(idf_module, "IDF_RECORDINGS", (recording,))
    monkeypatch.setattr(
        idf_module, "_RECORDING_BY_FILENAME", {recording.filename: recording}
    )
    archive = tmp_path / "raw" / idf_module.IDF_ARCHIVE_FILENAME
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(recording.archive_path, payload)
    monkeypatch.setattr(
        idf_module,
        "PINNED_FILES",
        (
            _pin(
                archive,
                idf_module.IDF_ARCHIVE_FILENAME,
                algorithm="md5",
                converted=False,
            ),
        ),
    )
    monkeypatch.setattr(
        idf_module,
        "load_px4_trajectories",
        lambda path, config: (_idf_trajectory(20), _idf_trajectory(30)),
    )
    monkeypatch.setattr(
        idf_module,
        "ULog",
        lambda *_args, **_kwargs: SimpleNamespace(
            start_timestamp=0, last_timestamp=1_000_000
        ),
    )
    return (
        "canonical/idf_session_01_reference_segment_01.npz",
        "canonical/idf_session_01_reference_segment_02.npz",
    )


def _prepare_x8(tmp_path, monkeypatch):
    relative = "training/lateral_121_1.csv"
    source = tmp_path / "raw" / relative
    source.parent.mkdir(parents=True)
    _write_x8_csv(source)
    recording = x8_module.X8Recording(
        filename="lateral_121_1.csv",
        split="training",
        file_id=20,
        size_bytes=source.stat().st_size,
        md5=hashlib.md5(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(x8_module, "X8_RECORDINGS", (recording,))
    monkeypatch.setattr(
        x8_module, "_RECORDING_BY_FILENAME", {recording.filename: recording}
    )
    monkeypatch.setattr(
        x8_module, "PINNED_FILES", (_pin(source, relative, algorithm="md5"),)
    )
    return ("canonical/training/lateral_121_1.npz",)


def _prepare_epfl(tmp_path, monkeypatch):
    source = tmp_path / "raw" / epfl_module.TOPOPLANE_FILENAME
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pinned TOPOPlane2 fixture")
    monkeypatch.setattr(epfl_module, "TOPOPLANE_SIZE_BYTES", source.stat().st_size)
    monkeypatch.setattr(
        epfl_module, "TOPOPLANE_MD5", hashlib.md5(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        epfl_module,
        "PINNED_FILES",
        (_pin(source, epfl_module.TOPOPLANE_FILENAME, algorithm="md5"),),
    )
    monkeypatch.setattr(epfl_module, "_read_topoplane_streams", lambda path: _streams())
    return ("canonical/topoplane2_segment_01.npz",)


_PREPARERS = {
    "nanodrone": _prepare_nanodrone,
    "arp": _prepare_arp,
    "idf": _prepare_idf,
    "x8": _prepare_x8,
    "epfl": _prepare_epfl,
}


@pytest.mark.parametrize("name", _CORPUS_NAMES)
def test_prepare_writes_the_canonical_tree_the_corpus_declares(
    name: str, tmp_path, monkeypatch
) -> None:
    expected = _PREPARERS[name](tmp_path, monkeypatch)
    corpus = REFERENCE_CORPORA[name]

    prepared = corpus.prepare(tmp_path)

    assert prepared.corpus == name
    assert prepared.raw_root == tmp_path / "raw"
    assert prepared.canonical_root == tmp_path / "canonical"
    assert prepared.trajectories == tuple(tmp_path / item for item in expected)
    assert prepared.duration_s > 0.0
    for output in prepared.trajectories:
        assert load_trajectory_npz(output).spec is not None
    assert (prepared.report_path is not None) == (corpus.report is not None)
    if prepared.report_path is not None:
        assert prepared.report_path.read_text().endswith("\n")


def test_prepare_reuses_a_verified_source_tree_elsewhere(tmp_path, monkeypatch) -> None:
    """A second canonical copy of one corpus does not download it twice."""

    _prepare_nanodrone(tmp_path, monkeypatch)
    second = tmp_path / "second"

    prepared = REFERENCE_CORPORA["nanodrone"].prepare(second, raw_root=tmp_path / "raw")

    assert prepared.raw_root == tmp_path / "raw"
    assert prepared.trajectories == (
        second / "canonical" / "train" / "chirp_20251017_run1.npz",
    )
