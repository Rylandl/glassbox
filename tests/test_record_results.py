"""Tests for the ``glassbox record-results`` manifest and runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import glassbox.cli as cli
from glassbox.workflows import record_results
from glassbox.workflows.record_results import (
    MANIFEST,
    ArtifactSpec,
    PythonStep,
    StepFailed,
    UnknownArtifactNames,
    assemble_cascade_x8_validation_report,
    missing_requirements,
    run_selected,
    select_artifacts,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "docs" / "results"
_X8_CASCADE_DIR = _REPO_ROOT / "artifacts" / "x8_cascade"
_X8_REFERENCE_DIR = _REPO_ROOT / "artifacts" / "x8_reference"

_OPTIONAL_MODULES = ("crazyflow", "cascade")


def _block_optional_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _OPTIONAL_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------
# Manifest completeness: every artifact under docs/results/ is named exactly
# once, and every regenerable entry's output already exists.


def test_manifest_covers_every_recorded_artifact_exactly_once() -> None:
    on_disk = sorted(path.name for path in _RESULTS_DIR.glob("*.json"))
    manifest_files = [Path(spec.output).name for spec in MANIFEST]

    assert sorted(manifest_files) == on_disk
    assert len(manifest_files) == len(set(manifest_files))


def test_every_regenerable_entrys_output_exists() -> None:
    for spec in MANIFEST:
        if spec.regenerable:
            assert (_REPO_ROOT / spec.output).exists(), spec.name


def test_every_manifest_entry_names_a_doc_page() -> None:
    for spec in MANIFEST:
        assert spec.doc_page, spec.name
        assert (_REPO_ROOT / spec.doc_page).exists(), spec.name


def test_not_regenerable_entries_carry_a_reason_and_no_steps() -> None:
    for spec in MANIFEST:
        if not spec.regenerable:
            assert spec.unavailable_reason
            assert spec.steps == ()


def test_regenerable_entries_declare_a_duration_class() -> None:
    for spec in MANIFEST:
        if spec.regenerable:
            assert spec.duration in ("fast", "slow")
            assert spec.steps


# ---------------------------------------------------------------------------
# --list and --dry-run must work with no optional extra installed and must
# not execute anything.


def test_list_runs_without_any_optional_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_optional_extras(monkeypatch)

    record_results.main(["--list"])

    stdout = capsys.readouterr().out
    for spec in MANIFEST:
        assert spec.name in stdout
    assert "needs the optional 'crazyflow' extra" in stdout
    assert "needs the optional 'cascade' extra" in stdout


def test_dry_run_runs_without_any_optional_extra_and_executes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_optional_extras(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(list(argv)))

    record_results.main(["--dry-run", "--include-slow"])

    assert calls == []
    stdout = capsys.readouterr().out
    assert "adaptive-recovery-results" in stdout
    assert "glassbox adaptive-recovery --output" in stdout


def test_dry_run_reports_blocked_entries_without_executing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_optional_extras(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(list(argv)))

    record_results.main(["--dry-run", "--only", "crazyflow-bootstrap-results"])

    assert calls == []
    stdout = capsys.readouterr().out
    assert "crazyflow-bootstrap-results" in stdout
    assert "skipped" in stdout
    assert "needs the optional 'crazyflow' extra" in stdout


def test_only_with_an_unknown_name_exits_with_status_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        record_results.main(["--only", "not-a-real-artifact"])

    assert excinfo.value.code == 2
    assert "unknown artifact name(s): not-a-real-artifact" in capsys.readouterr().err


def test_select_artifacts_defaults_to_fast_regenerable_only() -> None:
    selected = select_artifacts(MANIFEST, only=None, include_slow=False)

    assert selected
    assert all(spec.regenerable for spec in selected)
    assert all(spec.duration == "fast" for spec in selected)
    assert "cascade-x8-validation-results" not in {spec.name for spec in selected}


def test_select_artifacts_include_slow_adds_slow_entries() -> None:
    selected = select_artifacts(MANIFEST, only=None, include_slow=True)

    assert "cascade-x8-validation-results" in {spec.name for spec in selected}


def test_select_artifacts_only_bypasses_the_duration_gate() -> None:
    selected = select_artifacts(
        MANIFEST, only=["cascade-x8-validation-results"], include_slow=False
    )

    assert [spec.name for spec in selected] == ["cascade-x8-validation-results"]


def test_select_artifacts_only_unknown_name_raises() -> None:
    with pytest.raises(UnknownArtifactNames):
        select_artifacts(MANIFEST, only=["nope"], include_slow=False)


# ---------------------------------------------------------------------------
# The runner: steps run in order, and a failure stops the whole run without
# continuing on to later steps or later artifacts.


def test_run_selected_calls_steps_in_order_and_stops_on_first_failure() -> None:
    calls: list[str] = []

    def _step(name: str, *, fail: bool = False) -> PythonStep:
        def action() -> None:
            calls.append(name)
            if fail:
                raise StepFailed(f"{name} failed")

        return PythonStep(name, action)

    first_artifact = ArtifactSpec(
        name="fake-first",
        output="docs/results/does-not-exist-first.json",
        steps=(_step("first.a"), _step("first.b", fail=True), _step("first.c")),
        duration="fast",
        doc_page="docs/README.md",
    )
    second_artifact = ArtifactSpec(
        name="fake-second",
        output="docs/results/does-not-exist-second.json",
        steps=(_step("second.a"),),
        duration="fast",
        doc_page="docs/README.md",
    )

    with pytest.raises(SystemExit) as excinfo:
        run_selected([first_artifact, second_artifact])

    assert excinfo.value.code == 1
    # The failing step's own action ran (that's how it raised); nothing after
    # it, in the same artifact or the next one, was reached.
    assert calls == ["first.a", "first.b"]


def test_run_selected_skips_entries_with_unmet_requirements_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    blocked = ArtifactSpec(
        name="fake-blocked",
        output="docs/results/does-not-exist-blocked.json",
        steps=(PythonStep("unreachable", lambda: calls.append("unreachable")),),
        extra="crazyflow",
        duration="fast",
        doc_page="docs/README.md",
    )
    runnable = ArtifactSpec(
        name="fake-runnable",
        output="docs/results/does-not-exist-runnable.json",
        steps=(PythonStep("ran", lambda: calls.append("ran")),),
        duration="fast",
        doc_page="docs/README.md",
    )

    monkeypatch.setattr(record_results, "_importable", lambda name: False)
    results = run_selected([blocked, runnable])

    assert calls == ["ran"]
    statuses = {spec.name: status for spec, status in results}
    assert statuses["fake-blocked"].startswith("skipped:")
    assert statuses["fake-runnable"] == "ok"


def test_missing_requirements_reports_missing_data_path(tmp_path: Path) -> None:
    spec = ArtifactSpec(
        name="fake-needs-data",
        output="docs/results/does-not-exist.json",
        steps=(),
        required_data=("this/path/does/not/exist",),
        duration="fast",
        doc_page="docs/README.md",
    )

    problems = missing_requirements(spec)

    assert any("needs data at" in problem for problem in problems)


def test_missing_requirements_empty_for_a_fully_available_entry() -> None:
    spec = ArtifactSpec(
        name="fake-available",
        output="docs/results/adaptive-recovery-results.json",
        steps=(),
        duration="fast",
        doc_page="docs/README.md",
    )

    assert missing_requirements(spec) == []


def test_not_regenerable_entry_reports_its_reason() -> None:
    spec = next(spec for spec in MANIFEST if not spec.regenerable)

    problems = missing_requirements(spec)

    assert problems == [f"not regenerable: {spec.unavailable_reason}"]


# ---------------------------------------------------------------------------
# Cascade X8 assembly: reproduces the checked-in artifact from the raw
# per-step reports when they are present locally, and is skipped otherwise.


_CASCADE_ARTIFACTS_PRESENT = (_X8_CASCADE_DIR / "cascade_report.json").exists() and (
    _X8_REFERENCE_DIR / "benchmark_report.json"
).exists()


@pytest.mark.skipif(
    not _CASCADE_ARTIFACTS_PRESENT,
    reason="artifacts/x8_cascade and artifacts/x8_reference reports are not present locally",
)
def test_cascade_assembly_reproduces_the_recorded_artifact() -> None:
    produced = assemble_cascade_x8_validation_report(_X8_CASCADE_DIR, _X8_REFERENCE_DIR)
    recorded = json.loads(
        (_RESULTS_DIR / "cascade-x8-validation-results.json").read_text()
    )

    # No field in this artifact is host- or timing-dependent (unlike
    # adaptive-recovery-results.json, it carries no environment block or
    # source hash), so nothing is excluded from the comparison.
    assert produced == recorded


@pytest.mark.skipif(
    not _CASCADE_ARTIFACTS_PRESENT,
    reason="artifacts/x8_cascade and artifacts/x8_reference reports are not present locally",
)
def test_cascade_assembly_drops_per_trajectory_detail() -> None:
    produced = assemble_cascade_x8_validation_report(_X8_CASCADE_DIR, _X8_REFERENCE_DIR)

    def _has_per_trajectory(value: object) -> bool:
        if isinstance(value, dict):
            return "per_trajectory" in value or any(
                _has_per_trajectory(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(_has_per_trajectory(item) for item in value)
        return False

    assert not _has_per_trajectory(produced)


@pytest.mark.skipif(
    not _CASCADE_ARTIFACTS_PRESENT,
    reason="artifacts/x8_cascade and artifacts/x8_reference reports are not present locally",
)
def test_cascade_assembly_adds_reference_models_and_diagnostics() -> None:
    produced = assemble_cascade_x8_validation_report(_X8_CASCADE_DIR, _X8_REFERENCE_DIR)

    assert set(produced["glassbox_reference_models"]) == {
        "structured",
        "structured_residual",
    }
    assert set(produced["residual_diagnostics"]) == {
        "diagnostic_cg005_w04_i1",
        "diagnostic_cg005_w04_i3.5",
        "diagnostic_skywalker_x8_cg005_w04",
        "diagnostic_skywalker_x8_panels_cg005_w04",
    }
    for diagnostic in produced["residual_diagnostics"].values():
        assert set(diagnostic) == {"aircraft", "channels", "configuration"}
