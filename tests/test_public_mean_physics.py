"""Prospective fresh-cohort identity and seal checks; no simulations or fits."""

import copy
import json
from pathlib import Path

import pytest

from glassbox.experimental import public_mean_physics as physics

ROOT = Path(__file__).resolve().parents[1]


def original():
    return json.loads((ROOT / "docs/harness/two-simulator-flight-v1.json").read_text())


def test_fresh_roster_preserves_all_conditions_and_shifts_only_test_identities():
    p = original()
    before = copy.deepcopy(p)
    result = physics.fresh_test_roster(p)
    sources = [r for r in p["recordings"] if r["role"] == "test"]
    assert p == before
    assert len(result) == 168
    for source, actual in zip(sources, result, strict=True):
        assert actual["seed"] == source["seed"] + 11000000
        assert actual["id"].endswith("test-" + str(actual["seed"]))
        assert {k: v for k, v in actual.items() if k not in ("seed", "id")} == {
            k: v for k, v in source.items() if k not in ("seed", "id")
        }


@pytest.mark.parametrize(
    "defect", ["omitted", "duplicate", "malformed", "wrong_simulator"]
)
def test_roster_rejects_missing_duplicate_or_changed_identities(defect):
    p = original()
    index = next(i for i, r in enumerate(p["recordings"]) if r["role"] == "test")
    if defect == "omitted":
        del p["recordings"][index]
    elif defect == "duplicate":
        p["recordings"].append(copy.deepcopy(p["recordings"][index]))
    elif defect == "malformed":
        p["recordings"][index]["id"] += "-suffix"
    else:
        p["recordings"][index]["simulator"] = "other"
    with pytest.raises(ValueError):
        physics.fresh_test_roster(p)


def test_roster_refuses_previously_inspected_seed_shift():
    with pytest.raises(ValueError, match="frozen physical seed shift"):
        physics.fresh_test_roster(original(), shift=10000000)


def test_physical_seal_detects_payload_and_extra_file_changes(tmp_path):
    physics.write(tmp_path / "records.json", [{"id": i} for i in range(84)])
    physics.write(
        tmp_path / "seal.json",
        {
            "format": "glassbox-public-mean-physics-v1",
            "protocol_sha256": physics.PROTOCOL_SHA256,
            "files": physics._inventory(tmp_path),
        },
    )
    expected = physics.digest(tmp_path / "seal.json")
    physics.verify_data(tmp_path, expected)
    (tmp_path / "unplanned.txt").write_text("extra")
    with pytest.raises(ValueError, match="physical payload inventory"):
        physics.verify_data(tmp_path, expected)
    (tmp_path / "unplanned.txt").unlink()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/seal.json").write_text("{}")
    with pytest.raises(ValueError, match="physical payload inventory"):
        physics.verify_data(tmp_path, expected)
    (tmp_path / "nested/seal.json").unlink()
    (tmp_path / "records.json").write_text("[]")
    with pytest.raises(ValueError, match="physical payload inventory"):
        physics.verify_data(tmp_path, expected)


def test_physical_seal_requires_external_anchor(tmp_path):
    physics.write(tmp_path / "seal.json", {})
    with pytest.raises(ValueError, match="external data seal"):
        physics.verify_data(tmp_path, "0" * 64)
