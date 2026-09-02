"""One pinned download-and-verify routine for every reference-corpus adapter.

Each public dataset adapter fetches immutable upstream files whose size and
digest are pinned in source. The retrieval policy is identical everywhere: skip
a file that already matches, refuse to clobber a mismatched one unless the
caller opted in, stream the download to a sibling temporary file, verify size
and digest before publishing, and replace the target atomically so a failed or
interrupted fetch never leaves a partial artifact behind.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

_DIGEST_CHUNK_BYTES = 1024 * 1024


def file_digest(path: str | Path, *, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file, read in fixed-size chunks.

    The chunked read keeps multi-gigabyte reference archives out of memory.
    ``algorithm`` is any name :mod:`hashlib` accepts; the pinned corpora use
    ``sha256`` for the media-server snapshots and ``md5`` for the Dataverse and
    Zenodo archives, which publish only MD5.
    """

    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(_DIGEST_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    url: str,
    target: str | Path,
    *,
    size_bytes: int,
    digest: str,
    algorithm: str = "sha256",
    user_agent: str,
    overwrite: bool = False,
    timeout_s: float = 60.0,
    existing_mismatch_message: str | None = None,
    size_mismatch_message: str | None = None,
    digest_mismatch_message: str | None = None,
) -> Path:
    """Fetch ``url`` to ``target`` only if it verifies against the pinned pin.

    An existing target whose size and digest already match is returned without
    any network access. An existing target that does not match raises
    :class:`FileExistsError` unless ``overwrite`` is set. The download itself
    goes to a temporary file in the target's own directory and is moved into
    place with :func:`os.replace` only after both checks pass, so the target is
    either absent or complete and verified.

    The three message parameters override the wording each adapter reports; the
    defaults name the target file.
    """

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if (
            target_path.stat().st_size == size_bytes
            and file_digest(target_path, algorithm=algorithm) == digest
        ):
            return target_path
        if not overwrite:
            raise FileExistsError(
                existing_mismatch_message
                if existing_mismatch_message is not None
                else f"existing file does not match the pinned source: {target_path}"
            )

    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".download",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                shutil.copyfileobj(response, temporary)
        if temporary_path.stat().st_size != size_bytes:
            raise ValueError(
                size_mismatch_message
                if size_mismatch_message is not None
                else f"downloaded size mismatch for {target_path.name}"
            )
        if file_digest(temporary_path, algorithm=algorithm) != digest:
            raise ValueError(
                digest_mismatch_message
                if digest_mismatch_message is not None
                else f"downloaded checksum mismatch for {target_path.name}"
            )
        os.replace(temporary_path, target_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target_path
