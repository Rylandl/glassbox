"""Fingerprinted array archives owned by the learner.

One compressed ``.npz`` holds JSON metadata beside named arrays. The stored
fingerprint covers the metadata and every array's name, dtype, shape and bytes,
so loading an altered archive fails instead of returning a plausible model.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def array_fingerprint(metadata, arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_arrays(path, metadata, arrays):
    meta = {**metadata, "fingerprint": array_fingerprint(metadata, arrays)}
    np.savez_compressed(path, metadata=json.dumps(meta), **arrays)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        arrays = {k: archive[k] for k in archive.files if k != "metadata"}
    identity = meta.pop("fingerprint")
    if identity != array_fingerprint(meta, arrays):
        raise ValueError("model fingerprint mismatch")
    return meta, arrays
