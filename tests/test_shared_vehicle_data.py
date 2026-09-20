"""Input/source boundaries and deterministic roster; no native physics."""

from pathlib import Path

import pytest

from glassbox.experimental import shared_vehicle_data as subject


def test_frozen_roster_exact_plus_14m_and_exclusions():
    p = subject.read_protocol()
    rows = subject.roster(p)
    original = subject.read(subject.anchor(p["dependencies"]["original_flight"]))
    source = [r for r in original["recordings"] if r["role"] == "test"]
    assert rows == [
        dict(
            r,
            seed=r["seed"] + 14000000,
            id=r["id"].rsplit("-", 1)[0] + "-" + str(r["seed"] + 14000000),
        )
        for r in source
    ]
    pairs = {(r["simulator"], r["seed"]) for r in rows}
    assert len(pairs) == len({r["id"] for r in rows}) == 168
    for shift in range(0, 14000000, 1000000):
        assert pairs.isdisjoint((r["simulator"], r["seed"] + shift) for r in source)
    assert pairs.isdisjoint(
        (r["simulator"], r["seed"])
        for r in original["recordings"]
        if r["role"] != "test"
    )


def test_views_preserve_fixture_and_return_fresh_objects():
    p = subject.read(subject.PROTOCOL_PATH)
    view = subject.physical_view(p)
    original = subject.read(subject.anchor(p["dependencies"]["original_flight"]))
    assert {k: v for k, v in view.items() if k != "recordings"} == {
        k: v for k, v in original.items() if k != "recordings"
    }
    resolved = subject.resolved(p)
    assert resolved["decision"] == p["decision"]
    resolved["decision"].clear()
    assert subject.resolved(p)["decision"] == p["decision"]


def test_external_anchor_inventory_and_exclusive_writer(tmp_path):
    subject.write(tmp_path / "payload.json", {"x": 1})
    subject.write(tmp_path / "run.json", {"files": subject.inventory(tmp_path)})
    sha = subject.digest(tmp_path / "run.json")
    subject.sealed(tmp_path / "run.json", sha)
    with pytest.raises(FileExistsError):
        subject.write(tmp_path / "payload.json", {})
    (tmp_path / "payload.json").write_text("changed")
    with pytest.raises(ValueError, match="inventory"):
        subject.sealed(tmp_path / "run.json", sha)
    with pytest.raises(ValueError, match="external"):
        subject.sealed(tmp_path / "run.json", "0" * 64)
    (tmp_path / "symlink").symlink_to(tmp_path / "payload.json")
    with pytest.raises(ValueError, match="symlink"):
        subject.inventory(tmp_path)


def test_changed_protocol_cannot_self_authorize(tmp_path):
    p = subject.read(subject.PROTOCOL_PATH)
    p["architecture"]["width"] = 99
    subject.write(tmp_path / "changed.json", p)
    with pytest.raises(ValueError, match="frozen protocol"):
        subject.read_protocol(
            tmp_path / "changed.json", subject.digest(tmp_path / "changed.json")
        )


def candidate(tmp_path, sim, status="complete"):
    path = tmp_path / sim
    path.mkdir()
    subject.write(
        path / "run.json",
        dict(
            format=subject.CANDIDATE_FORMAT,
            simulator=sim,
            status=status,
            protocol_sha256=subject.PROTOCOL_SHA256,
            binding_sha256="bound",
            files={},
        ),
    )
    return dict(
        path=str(path / "run.json"),
        sha256=subject.digest(path / "run.json"),
        status=status,
    )


@pytest.mark.parametrize(
    "status", ["complete", "fit_failed", "hard_timeout_incomplete", "failed"]
)
def test_only_declared_candidate_outcomes_allow_collection(tmp_path, status):
    entries = {s: candidate(tmp_path, s, status) for s in subject.SIMULATORS}
    if status in ("complete", "fit_failed"):
        assert (
            subject.candidate_prerequisites(entries, binding_sha256="bound") == entries
        )
    else:
        with pytest.raises(ValueError, match="complete or fit_failed"):
            subject.candidate_prerequisites(entries, binding_sha256="bound")


def test_both_candidates_required_and_source_bound(tmp_path):
    rows = {s: candidate(tmp_path, s) for s in subject.SIMULATORS}
    with pytest.raises(ValueError, match="both candidate"):
        subject.candidate_prerequisites(
            {"crazyflow": rows["crazyflow"]}, binding_sha256="bound"
        )
    with pytest.raises(ValueError, match="source association"):
        subject.candidate_prerequisites(rows, binding_sha256="wrong")


def test_source_identity_rejects_dirty_or_changed_file(tmp_path, monkeypatch):
    (tmp_path / "test.py").write_text("saved")
    monkeypatch.setattr(
        subject, "_git", lambda root, *args: "commit" if args[0] == "rev-parse" else ""
    )
    identity = subject._source_identity(tmp_path, "commit", ["test.py"])
    (tmp_path / "test.py").write_text("altered")
    with pytest.raises(ValueError, match="bound source bytes"):
        subject._source_identity(tmp_path, "commit", identity["files"])
    monkeypatch.setattr(
        subject,
        "_git",
        lambda root, *args: "commit" if args[0] == "rev-parse" else " M test.py",
    )
    with pytest.raises(ValueError, match="clean"):
        subject._source_identity(tmp_path, "commit", ["test.py"])


def test_standalone_collector_passes_resolved_fresh_protocol_to_native_helpers(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from glassbox.experimental import expanded_training_cache_experiment as historical

    entry = dict(
        id="crazyflow/p/test-14200000",
        seed=14200000,
        simulator="crazyflow",
        cell="p",
        role="test",
    )
    p = dict(
        cells={s: [dict(id="p", group="primary")] for s in subject.SIMULATORS},
        generation={s: dict(dt_s=0.01) for s in subject.SIMULATORS},
        recordings=[entry],
    )
    calls = []

    class Fixture:
        spec = SimpleNamespace(to_dict=lambda: {"channels": []})

        def config(self):
            return {"dt_s": 0.01}

        def generate(self, row, cell):
            calls.append((row, cell))
            return None, {"setup_failure": "toy"}

    def fixture_for(sim, got):
        assert got == p and sim == "crazyflow"
        return Fixture()

    flight = SimpleNamespace(
        fixture_for=fixture_for, make_queries=lambda *args: ([], {}, {})
    )
    monkeypatch.setattr(historical, "check_sources", lambda: {"runtime": "toy"})
    monkeypatch.setattr(historical, "protocol", lambda: p)
    monkeypatch.setattr(historical, "flight", lambda: flight)
    monkeypatch.setattr(
        subject, "_worker_identity", lambda req: ({}, {"oracle": {"source": "frozen"}})
    )
    monkeypatch.setattr(subject, "physical_view", lambda protocol: p)
    monkeypatch.setattr(
        subject, "candidate_prerequisites", lambda anchors, **kw: anchors
    )
    request = dict(
        stage="collect",
        simulator="crazyflow",
        candidate_outcomes={},
        binding_sha256="bound",
        directory=str(tmp_path / "data"),
    )
    subject.write(tmp_path / "request.json", request)
    result = subject._worker(
        tmp_path / "request.json", subject.digest(tmp_path / "request.json")
    )
    assert calls[0][0] == entry
    assert result["test_seed_shift_from_original"] == 14000000
    assert (
        result["parents"] == 1 and result["training_collected"] == result["fits"] == 0
    )
    assert (
        subject.read(tmp_path / "data/records.json")[0]["validity"]["reason"]
        == "setup_failure"
    )


def test_native_source_binding_is_hash_only_and_pinned():
    p = subject.read(subject.PROTOCOL_PATH)
    sources = subject.simulator_identity(p)
    assert Path(sources["crazyflow"]["package_root"]).name == "crazyflow"
    assert Path(sources["cascade"]["package_root"]).name == "cascade"
    assert sources["crazyflow"]["files"] and sources["cascade"]["files"]
