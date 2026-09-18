"""Inventory and evidence-boundary tests; no simulator or learner fitting."""

import copy
import json
from pathlib import Path

import pytest

from glassbox.experimental import jsbsim_onboarding as harness


def test_frozen_inventory_counts_and_legacy_candidates():
    protocol, inventory = harness.load_spec()
    assert protocol["scope"].startswith("No fitting")
    assert len(inventory["directories"]) == 61
    assert len(inventory["entries"]) == 66
    assert sum(e["root_tag"] == "fdm_config" for e in inventory["entries"]) == 64
    assert (
        sum(e["selected_initialization"] is not None for e in inventory["entries"])
        == 39
    )
    assert (
        next(e for e in inventory["entries"] if e["id"] == "737/737.xml")[
            "selected_initialization"
        ]
        == "aircraft/737/cruise_init.xml"
    )
    assert "LM" in inventory["directories"]
    assert not any(e["id"].startswith("LM/") for e in inventory["entries"])
    assert len({harness.case_slug(e["id"]) for e in inventory["entries"]}) == 66


def test_branch_roster_includes_factual_then_both_directions():
    roster = harness.branch_roster(2)
    assert [x["label"] for x in roster] == [
        "factual",
        "c000_lower",
        "c000_upper",
        "c001_lower",
        "c001_upper",
    ]
    assert [x["command_index"] for x in roster] == [None, 0, 0, 1, 1]


def test_discovery_does_not_skip_legacy_or_trimmed_initializations(tmp_path):
    directory = tmp_path / "aircraft" / "vehicle"
    directory.mkdir(parents=True)
    (directory / "variant.xml").write_text("<FDM_CONFIG/>")
    (directory / "reset00.xml").write_text("<initialize><trim>1</trim></initialize>")
    (directory / "cruise.xml").write_text(
        "<initialize><running>-1</running></initialize>"
    )
    (entry,) = harness.discover(tmp_path)
    assert entry["root_tag"] == "FDM_CONFIG"
    assert entry["selected_initialization"] == "aircraft/vehicle/cruise.xml"
    assert entry["initializations"][1]["requests_trim"] is True
    (directory / "reset00.xml").write_text("<initialize/>")
    (entry,) = harness.discover(tmp_path)
    assert entry["selected_initialization"] == "aircraft/vehicle/reset00.xml"


def test_asset_roster_rejects_added_removed_changed_files(tmp_path):
    for part in ("aircraft", "engine", "systems"):
        (tmp_path / part).mkdir()
    (tmp_path / "aircraft" / "demo.xml").write_text("<fdm_config/>")
    inventory = dict(
        assets={"aircraft/demo.xml": harness.digest(tmp_path / "aircraft/demo.xml")},
        entries=harness.discover(tmp_path),
        directories=[],
    )
    harness.check_assets(tmp_path, inventory)
    (tmp_path / "engine" / "extra.xml").write_text("<engine/>")
    with pytest.raises(ValueError, match="roster"):
        harness.check_assets(tmp_path, inventory)
    (tmp_path / "engine" / "extra.xml").unlink()
    (tmp_path / "aircraft" / "demo.xml").write_text('<fdm_config name="changed"/>')
    with pytest.raises(ValueError, match="roster"):
        harness.check_assets(tmp_path, inventory)
    (tmp_path / "aircraft" / "demo.xml").unlink()
    with pytest.raises(ValueError, match="roster"):
        harness.check_assets(tmp_path, inventory)


def test_asset_symlinks_cannot_escape_pinned_root(tmp_path):
    for part in ("aircraft", "engine", "systems"):
        (tmp_path / part).mkdir()
    (tmp_path / "aircraft" / "outside.xml").symlink_to(Path(__file__))
    with pytest.raises(ValueError, match="symlink"):
        harness.check_assets(tmp_path, dict(assets={}, entries=[], directories=[]))


def test_changed_protocol_is_rejected_even_if_parseable(tmp_path, monkeypatch):
    modified = tmp_path / "protocol.json"
    value = json.loads(harness.PROTOCOL.read_text())
    value["recording"]["transitions"] += 1
    modified.write_text(json.dumps(value))
    monkeypatch.setattr(harness, "PROTOCOL", modified)
    with pytest.raises(ValueError, match="frozen protocol"):
        harness.load_spec()


def test_discovered_initialization_must_equal_pinned_choice(tmp_path):
    for part in ("aircraft", "engine", "systems"):
        (tmp_path / part).mkdir()
    (tmp_path / "aircraft" / "demo.xml").write_text("<fdm_config/>")
    actual = harness.discover(tmp_path)
    wrong = copy.deepcopy(actual)
    wrong[0]["selected_initialization"] = "made-up.xml"
    inventory = dict(
        assets={"aircraft/demo.xml": harness.digest(tmp_path / "aircraft/demo.xml")},
        entries=wrong,
        directories=[],
    )
    with pytest.raises(ValueError, match="rediscovered"):
        harness.check_assets(tmp_path, inventory)


def test_atomic_checkpoint_keeps_partial_arrays_with_matching_metadata(tmp_path):
    import numpy as np

    case = {
        "id": "vehicle.xml",
        "runs": {"parent": {"status": "running"}},
        "branches": [],
    }
    arrays = {"parent__observations": np.array([[1.0, -0.0, 3.0]])}
    harness.checkpoint(tmp_path, case, arrays)
    harness.write_json(
        tmp_path / "progress.json",
        dict(label="parent", failed_interval=2, failed_substep=3),
    )
    restored, values = harness.restore_checkpoint(tmp_path, {"id": "vehicle.xml"})
    assert restored["runs"] == case["runs"]
    assert restored["interrupted_progress"]["failed_substep"] == 3
    assert (
        values["parent__observations"].tobytes()
        == arrays["parent__observations"].tobytes()
    )
    # A killed writer's incomplete next checkpoint does not replace its last complete one.
    (tmp_path / "checkpoint.tmp").write_bytes(b"incomplete")
    again, _ = harness.restore_checkpoint(tmp_path, {"id": "vehicle.xml"})
    assert again == restored


def test_full_verify_requires_an_external_seal_before_simulation(tmp_path):
    (tmp_path / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="trusted external"):
        harness.verify(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="trusted external"):
        harness.verify(tmp_path, tmp_path, expected_run_sha256="0" * 64)
