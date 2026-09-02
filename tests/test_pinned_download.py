from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pytest

from glassbox.io.pinned_download import download_verified, file_digest

PAYLOAD = b"glassbox pinned reference payload\n" * 64
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
MD5 = hashlib.md5(PAYLOAD).hexdigest()


def _serve(monkeypatch, payload: bytes) -> list[tuple[str, float]]:
    calls: list[tuple[str, float]] = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_file_digest_matches_hashlib_for_both_pinned_algorithms(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(PAYLOAD)

    assert file_digest(source) == SHA256
    assert file_digest(source, algorithm="md5") == MD5


def test_download_replaces_target_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _serve(monkeypatch, PAYLOAD)
    target = tmp_path / "nested" / "payload.bin"

    result = download_verified(
        "https://example.invalid/payload.bin",
        target,
        size_bytes=len(PAYLOAD),
        digest=SHA256,
        user_agent="glassbox-test/1",
        timeout_s=12.0,
    )

    assert result == target
    assert target.read_bytes() == PAYLOAD
    assert calls == [("https://example.invalid/payload.bin", 12.0)]
    assert sorted(item.name for item in target.parent.iterdir()) == ["payload.bin"]


def test_matching_target_is_reused_without_any_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _serve(monkeypatch, PAYLOAD)
    target = tmp_path / "payload.bin"
    target.write_bytes(PAYLOAD)

    result = download_verified(
        "https://example.invalid/payload.bin",
        target,
        size_bytes=len(PAYLOAD),
        digest=SHA256,
        user_agent="glassbox-test/1",
    )

    assert result == target
    assert calls == []


def test_mismatched_target_is_refused_unless_overwrite_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _serve(monkeypatch, PAYLOAD)
    target = tmp_path / "payload.bin"
    target.write_bytes(b"stale contents")

    with pytest.raises(FileExistsError, match="pinned test source"):
        download_verified(
            "https://example.invalid/payload.bin",
            target,
            size_bytes=len(PAYLOAD),
            digest=SHA256,
            user_agent="glassbox-test/1",
            existing_mismatch_message="existing file does not match pinned test source",
        )

    assert calls == []
    assert target.read_bytes() == b"stale contents"

    result = download_verified(
        "https://example.invalid/payload.bin",
        target,
        size_bytes=len(PAYLOAD),
        digest=SHA256,
        user_agent="glassbox-test/1",
        overwrite=True,
    )

    assert result.read_bytes() == PAYLOAD
    assert len(calls) == 1


def test_checksum_mismatch_rejects_the_download_and_keeps_the_target_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupted = PAYLOAD[:-1] + b"X"
    _serve(monkeypatch, corrupted)
    target = tmp_path / "payload.bin"

    with pytest.raises(ValueError, match="pinned test digest"):
        download_verified(
            "https://example.invalid/payload.bin",
            target,
            size_bytes=len(corrupted),
            digest=SHA256,
            user_agent="glassbox-test/1",
            digest_mismatch_message="downloaded pinned test digest mismatch",
        )

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_size_mismatch_rejects_the_download_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, PAYLOAD)
    target = tmp_path / "payload.bin"

    with pytest.raises(ValueError, match="pinned test size"):
        download_verified(
            "https://example.invalid/payload.bin",
            target,
            size_bytes=len(PAYLOAD) + 1,
            digest=SHA256,
            user_agent="glassbox-test/1",
            size_mismatch_message="downloaded pinned test size mismatch",
        )

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
