"""No-fit checks of the frozen generic API migration and its replay defenses."""

import copy
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from glassbox import LearnedDynamics
from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import api_migration as migration
from glassbox.experimental import qualification


@pytest.fixture(scope="module")
def plan():
    return migration.frozen_plan()[0]


@pytest.fixture(scope="module")
def artifact(plan):
    root = Path(plan["saved_evidence"]["root_hint"])
    item = next(
        item
        for item in plan["saved_evidence"]["models"]
        if "hidden_input_delay-101/model.npz" in item["path"]
    )
    path = root / item["path"]
    if not path.exists():
        pytest.skip("the pinned migration model is not present on this host")
    assert (
        migration.sha256(path.read_bytes())
        == plan["saved_evidence"]["sha256"][item["path"]]
    )
    return path


def rewrite_inventory(directory):
    migration.write_json(
        directory / "files.json",
        {
            path.relative_to(directory).as_posix(): migration.sha256(path.read_bytes())
            for path in directory.rglob("*")
            if path.is_file() and path.name != "files.json"
        },
    )


def test_manifest_pins_recipe_surface_and_complete_evidence(plan):
    from glassbox.learner import ENVELOPE_COVERAGE, RECIPE

    assert RECIPE == plan["recipe"]
    assert ENVELOPE_COVERAGE == plan["artifact"]["nominal_coverage"]
    assert plan["saved_evidence"]["counts"] == {"models": 33, "forecasts": 60}
    models = plan["saved_evidence"]["models"]
    assert len({item["path"] for item in models}) == 33
    assert sum(len(item["forecasts"]) for item in models) == 60
    assert migration.public_contract()["accepted"]


def test_manifest_edit_is_rejected(tmp_path, monkeypatch, plan):
    altered = copy.deepcopy(plan)
    altered["saved_evidence"]["prediction_tolerance"]["atol"] = 1.0
    path = tmp_path / "manifest.json"
    migration.write_json(path, altered)
    monkeypatch.setattr(migration, "MANIFEST_PATH", path)
    with pytest.raises(ValueError, match="migration manifest"):
        migration.frozen_plan()


def test_qualification_keeps_original_plan_and_all_fifteen_source_anchors(plan):
    qualified, raw = qualification.frozen_plan()
    assert len(qualified["source_sha256"]) == 15
    assert migration.sha256(raw) == plan["qualification_compatibility"]["plan_sha256"]
    for name, contract in plan["qualification_compatibility"]["files"].items():
        assert contract["original_sha256"] == qualified["source_sha256"][name]
        assert migration.qualification_source_matches(
            migration.ROOT, name, qualified["source_sha256"][name]
        )


@pytest.mark.parametrize(
    "name",
    [
        "src/glassbox/experimental/harness.py",
        "src/glassbox/experimental/learned_plan.py",
        "examples/cascade_accuracy.py",
    ],
)
def test_qualification_rejects_every_nonimport_statement_change(tmp_path, plan, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text((migration.ROOT / name).read_text() + "\npass\n")
    expected = plan["qualification_compatibility"]["files"][name]["original_sha256"]
    assert not migration.qualification_source_matches(tmp_path, name, expected)


def test_qualification_rejects_unlisted_sources_and_import_targets(tmp_path, plan):
    assert not migration.qualification_source_matches(
        tmp_path, "src/glassbox/control/solver.py", "0" * 64
    )
    name = "src/glassbox/experimental/harness.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text(
        (migration.ROOT / name)
        .read_text()
        .replace("from ..learner import", "from ..fitting import")
    )
    expected = plan["qualification_compatibility"]["files"][name]["original_sha256"]
    assert not migration.qualification_source_matches(tmp_path, name, expected)


@pytest.mark.parametrize(
    "field", ["param_linear", "train_future_states", "envelope_half_width", "metadata"]
)
def test_public_load_rejects_each_frozen_tamper_case(artifact, field):
    data = migration.archive_values(artifact)
    if field == "metadata":
        metadata = json.loads(str(data[field]))
        metadata["contract"]["configuration_id"] += "-altered"
        data[field] = json.dumps(metadata)
    else:
        data[field] = data[field].copy()
        data[field].flat[0] += 1.0
    altered = io.BytesIO()
    np.savez_compressed(altered, **data)
    altered.seek(0)
    with pytest.raises(ValueError, match="fingerprint"):
        LearnedDynamics.load(altered)


@pytest.mark.parametrize("field", ["format", "recipe"])
def test_resigning_an_unsupported_model_does_not_bypass_version_gate(artifact, field):
    data = migration.archive_values(artifact)
    metadata = json.loads(str(data.pop("metadata")))
    metadata.pop("fingerprint")
    if field == "format":
        metadata[field] = "unsupported-format"
    else:
        metadata[field]["steps"] += 1
    metadata["fingerprint"] = array_fingerprint(metadata, data)
    altered = io.BytesIO()
    np.savez_compressed(altered, metadata=json.dumps(metadata), **data)
    altered.seek(0)
    with pytest.raises(ValueError, match="recipe version"):
        LearnedDynamics.load(altered)


def fake_fixture(directory):
    """Only the receipt boundary: model content is checked by verify separately."""
    directory.mkdir()
    _plan, raw = migration.frozen_plan()
    for name in (*migration.FIXTURE_MODELS, "predictions.npz"):
        (directory / name).write_bytes(b"model content is irrelevant to roster tests")
    report = dict(
        kind="fixture",
        implementation="public",
        environment=migration.environment(),
        fitted_fingerprint="a" * 64,
        updated_fingerprint="b" * 64,
        model_files=list(migration.FIXTURE_MODELS),
    )
    migration.seal(directory, report, raw)
    return report


def test_fixture_receipt_requires_static_files_and_report_fields(tmp_path):
    directory = tmp_path / "fixture"
    report = fake_fixture(directory)
    assert migration.checked_run(directory) == report
    report["model_files"] = []
    migration.write_json(directory / "report.json", report)
    rewrite_inventory(directory)
    with pytest.raises(ValueError, match="roster"):
        migration.checked_run(directory)


@pytest.mark.parametrize(
    "change",
    [
        "omit_file",
        "extra_file",
        "extra_report_field",
        "omit_report_field",
        "implementation",
        "environment",
    ],
)
def test_rewritten_receipt_cannot_hide_file_or_summary_mutations(tmp_path, change):
    directory = tmp_path / "fixture"
    report = fake_fixture(directory)
    if change == "omit_file":
        (directory / migration.FIXTURE_MODELS[0]).unlink()
    elif change == "extra_file":
        (directory / "extra.npz").write_bytes(b"extra")
    elif change == "extra_report_field":
        report["new_claim"] = "accepted"
    elif change == "omit_report_field":
        report.pop("fitted_fingerprint")
    elif change == "implementation":
        report["implementation"] = "unknown"
    else:
        report["environment"]["x64"] = False
    migration.write_json(directory / "report.json", report)
    rewrite_inventory(directory)
    with pytest.raises(ValueError):
        migration.checked_run(directory)


def test_frozen_evidence_hash_does_not_trust_a_run_inventory(
    tmp_path, monkeypatch, artifact, plan
):
    selected = copy.deepcopy(plan)
    selected["saved_evidence"]["sha256"] = {
        "model.npz": migration.sha256(artifact.read_bytes())
    }
    (tmp_path / "model.npz").write_bytes(artifact.read_bytes() + b"altered")
    migration.write_json(
        tmp_path / "files.json",
        {"model.npz": migration.sha256((tmp_path / "model.npz").read_bytes())},
    )
    with pytest.raises(ValueError, match="frozen migration evidence"):
        migration.checked_evidence(tmp_path, selected)


def test_a_saved_fixture_replays_and_rejects_forged_metrics_when_available(
    tmp_path, plan
):
    source = Path(plan["saved_evidence"]["root_hint"]) / "generic-public-api-candidate"
    if not source.exists():
        pytest.skip("the post-freeze fixture capture is not present on this host")
    copied = tmp_path / "captured"
    shutil.copytree(source, copied)
    assert migration.verify(copied)["no_fit"]
    predictions = migration.archive_values(copied / "predictions.npz")
    predictions["updated_batch"] = predictions["updated_batch"].copy()
    predictions["updated_batch"].flat[0] += 1.0
    np.savez_compressed(copied / "predictions.npz", **predictions)
    rewrite_inventory(copied)
    with pytest.raises(ValueError, match="array values differ"):
        migration.verify(copied)
