"""Tests for the ``glassbox record-results`` manifest and runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import glassbox.cli as cli
import glassbox.workflows.record_results as manifest_module
from glassbox.cli import record_results
from glassbox.workflows.record_results import (
    MANIFEST,
    TIERS,
    VALIDATION_SOURCE_FILES,
    ArtifactSpec,
    PythonStep,
    RecordingPlan,
    StepFailed,
    UnknownArtifactNames,
    assemble_cascade_x8_validation_report,
    assemble_corpus_validation_report,
    build_manifest,
    check_selected,
    matches_path,
    missing_requirements,
    recorded_differences,
    run_selected,
    select_artifacts,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "docs" / "results"
_X8_CASCADE_DIR = _REPO_ROOT / "artifacts" / "x8_cascade"
_X8_REFERENCE_DIR = _REPO_ROOT / "artifacts" / "x8_reference"

_OPTIONAL_MODULES = ("cascade", "pyulog", "rosbags")


def _block_optional_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _OPTIONAL_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------
# Manifest completeness: every artifact under docs/results/ is named exactly
# once, and every already-recorded entry's output is on disk.


def test_manifest_covers_every_recorded_artifact_exactly_once() -> None:
    on_disk = sorted(path.name for path in _RESULTS_DIR.glob("*.json"))
    manifest_files = [Path(spec.output).name for spec in MANIFEST]
    recorded_files = [Path(spec.output).name for spec in MANIFEST if spec.recorded]

    assert sorted(recorded_files) == on_disk
    assert len(manifest_files) == len(set(manifest_files))


def test_every_recorded_entrys_output_exists() -> None:
    for spec in MANIFEST:
        if spec.recorded:
            assert (_REPO_ROOT / spec.output).exists(), spec.name


def test_every_manifest_entry_is_recorded() -> None:
    """Phase 3's job was to leave nothing declared but unrecorded.

    Every claim in the documentation is now backed by an artifact that one
    command produced and that is committed beside it. An entry awaiting its
    first recording is a legitimate transient state, tested below on a spec of
    its own, but it is not a state the manifest is allowed to rest in.
    """

    unrecorded = [spec.name for spec in MANIFEST if not spec.recorded]

    assert not unrecorded, f"declared but never recorded: {', '.join(unrecorded)}"


def test_every_manifest_entry_names_a_doc_page_and_its_section() -> None:
    for spec in MANIFEST:
        assert spec.doc_page, spec.name
        assert (_REPO_ROOT / spec.doc_path).exists(), spec.name
        anchor = spec.doc_page.partition("#")[2]
        if anchor:
            headings = (_REPO_ROOT / spec.doc_path).read_text().splitlines()
            slugs = {
                line.lstrip("# ").lower().replace(" ", "-")
                for line in headings
                if line.startswith("#")
            }
            assert anchor in slugs, f"{spec.name} -> {spec.doc_page}"


def test_every_entry_declares_a_tier_its_inputs_and_its_steps() -> None:
    for spec in MANIFEST:
        assert spec.tier in TIERS, spec.name
        assert spec.inputs, spec.name
        assert spec.steps, spec.name


def test_every_corpus_validation_entry_declares_its_adapters_extra() -> None:
    from glassbox.io.corpus import REFERENCE_CORPORA

    for name, corpus in REFERENCE_CORPORA.items():
        spec = next(
            item for item in MANIFEST if item.name == f"validation-{name}-results"
        )
        assert spec.extra == corpus.extra, spec.name
        assert spec.inputs == (
            f"corpus {name} pinned at {corpus.citation.pinned_version}",
        )


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
    assert "needs the optional 'cascade' extra" in stdout


def test_dry_run_runs_without_any_optional_extra_and_executes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_optional_extras(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(list(argv)))

    record_results.main(["--dry-run"])

    assert calls == []
    stdout = capsys.readouterr().out
    for spec in MANIFEST:
        assert spec.name in stdout
    assert "glassbox benchmark recovery --output" in stdout


def test_dry_run_reports_blocked_entries_without_executing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_optional_extras(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(list(argv)))

    record_results.main(["--dry-run", "--only", "cascade-x8-validation-results"])

    assert calls == []
    stdout = capsys.readouterr().out
    assert "cascade-x8-validation-results" in stdout
    assert "skipped" in stdout
    assert "needs the optional 'cascade' extra" in stdout


def test_only_with_an_unknown_name_exits_with_status_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        record_results.main(["--only", "not-a-real-artifact"])

    assert excinfo.value.code == 2
    assert "unknown artifact name(s): not-a-real-artifact" in capsys.readouterr().err


def test_select_artifacts_defaults_to_the_local_tier() -> None:
    selected = select_artifacts(MANIFEST)

    assert selected
    assert all(spec.tier == "local" for spec in selected)
    assert "cascade-x8-validation-results" not in {spec.name for spec in selected}


def test_select_artifacts_corpus_tier_selects_the_maintainer_job() -> None:
    selected = select_artifacts(MANIFEST, tier="corpus")

    assert "cascade-x8-validation-results" in {spec.name for spec in selected}
    assert "adaptive-recovery-results" not in {spec.name for spec in selected}


def test_select_artifacts_only_bypasses_the_tier_gate() -> None:
    selected = select_artifacts(MANIFEST, only=["cascade-x8-validation-results"])

    assert [spec.name for spec in selected] == ["cascade-x8-validation-results"]


def test_select_artifacts_only_unknown_name_raises() -> None:
    with pytest.raises(UnknownArtifactNames):
        select_artifacts(MANIFEST, only=["nope"])


def test_select_artifacts_rejects_an_unknown_tier() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        select_artifacts(MANIFEST, tier="everything")


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
        tier="local",
        doc_page="docs/README.md",
    )
    second_artifact = ArtifactSpec(
        name="fake-second",
        output="docs/results/does-not-exist-second.json",
        steps=(_step("second.a"),),
        tier="local",
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
        extra="cascade",
        tier="local",
        doc_page="docs/README.md",
    )
    runnable = ArtifactSpec(
        name="fake-runnable",
        output="docs/results/does-not-exist-runnable.json",
        steps=(PythonStep("ran", lambda: calls.append("ran")),),
        tier="local",
        doc_page="docs/README.md",
    )

    monkeypatch.setattr(manifest_module, "_importable", lambda name: False)
    results = run_selected([blocked, runnable])

    assert calls == ["ran"]
    statuses = {spec.name: status for spec, status in results}
    assert statuses["fake-blocked"].startswith("skipped:")
    assert statuses["fake-runnable"] == "ok"


def test_missing_requirements_empty_for_a_fully_available_entry() -> None:
    spec = ArtifactSpec(
        name="fake-available",
        output="docs/results/adaptive-recovery-results.json",
        steps=(),
        tier="local",
        doc_page="docs/README.md",
    )

    assert missing_requirements(spec) == []


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
    # source hash), so nothing is excluded from the comparison. It is compared
    # under the manifest entry's own tolerance rather than for equality: the
    # reference-model scores it folds in come from the X8 fits, which the
    # corpus tier refits, and a refit moves a geometric mean's last bit.
    spec = next(
        item for item in MANIFEST if item.name == "cascade-x8-validation-results"
    )
    assert not recorded_differences(produced, recorded, tolerance=spec.tolerance)


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


# ---------------------------------------------------------------------------
# The recording plan: one manifest, pointed at different directories, with or
# without a shortened fit budget.


def test_a_smoke_plan_redirects_every_path_and_shortens_the_fit(
    tmp_path: Path,
) -> None:
    plan = RecordingPlan(
        corpus_root=tmp_path / "work",
        results_root=tmp_path / "out",
        source_root=tmp_path / "sources",
        fit_steps=2,
        fold_limit=2,
        smoke=True,
    )

    manifest = build_manifest(plan)

    spec = next(item for item in manifest if item.name == "validation-x8-results")
    assert spec.output.startswith(str(tmp_path / "out"))
    commands = [step.describe() for step in spec.steps]
    assert str(tmp_path / "sources" / "x8_reference" / "raw") in commands[0]
    assert all(str(tmp_path / "work") in command for command in commands[1:])
    assert sum("--steps 2" in command for command in commands) == 2

    folds = next(item for item in manifest if item.name == "validation-idf-results")
    holdout = next(
        step.describe() for step in folds.steps if "--hold-out" in step.describe()
    )
    assert "--limit-folds 2" in holdout
    assert "--steps 2" in holdout


def test_the_default_plan_writes_the_committed_paths() -> None:
    for spec in MANIFEST:
        assert spec.output.startswith("docs/results/"), spec.name
        for step in spec.steps:
            assert "--raw " not in step.describe() or "artifacts/" in step.describe()


# ---------------------------------------------------------------------------
# The freshness comparison behind --check.


def test_matches_path_globs_a_whole_path_or_a_suffix() -> None:
    assert matches_path(["environment"], "report.environment")
    assert matches_path(
        ["scenarios[*].solve_time_p90_s"], "report.scenarios[3].solve_time_p90_s"
    )
    assert not matches_path(["environment"], "report.environment_scale")


def test_recorded_differences_is_empty_for_an_identical_document() -> None:
    document = {"a": 1.0, "b": [{"c": "text"}], "d": True}

    assert recorded_differences(document, document) == []


def test_recorded_differences_names_every_moved_value() -> None:
    recorded = {"a": 1.0, "b": {"c": 2.0}, "d": "left"}
    produced = {"a": 1.0, "b": {"c": 2.5}, "d": "right"}

    differences = recorded_differences(produced, recorded)

    assert len(differences) == 2
    assert any("report.b.c" in item for item in differences)
    assert any("report.d" in item for item in differences)


def test_recorded_differences_skips_volatile_paths() -> None:
    recorded = {"environment": {"python": "3.12.1"}, "value": 1.0}
    produced = {"environment": {"python": "3.13.0"}, "value": 1.0}

    assert recorded_differences(produced, recorded, volatile=("environment",)) == []


def test_recorded_differences_reports_a_missing_or_extra_key() -> None:
    differences = recorded_differences({"a": 1}, {"a": 1, "b": 2})

    assert differences == ["report.b: recorded but not produced"]


def test_recorded_differences_allows_a_float_inside_the_tolerance() -> None:
    assert recorded_differences({"a": 1.0 + 1e-9}, {"a": 1.0}) == []
    assert recorded_differences({"a": 1.001}, {"a": 1.0})


# ---------------------------------------------------------------------------
# The corpus validation contract: one artifact assembled from the chain's own
# reports, with nothing transcribed by hand.


def _fit_report() -> dict[str, object]:
    return {
        "format_version": 1,
        "split": {
            "mode": "leave_labeled_out",
            "holdout": {"rule": "label", "key": "benchmark_split"},
            "independent_source_group_holdout": True,
            "training_source_groups": ["a"],
            "validation_source_groups": ["b"],
            "training_flights": [{"path": "train.npz", "duration_s": 10.0}],
            "validation_flights": [{"path": "held.npz", "duration_s": 4.0}],
        },
        "configuration": {
            "model_class": "structured",
            "learning_rate": 0.02,
            "optimization_steps_per_model": 400,
            "training_windows_per_flight_by_horizon": {"0.1s": {"train.npz": 12}},
        },
        "models": {
            "learned_lag": {
                "fit": {"final_loss": 0.5, "wall_time_s": 1.0},
                "parameter_evidence": {"numerical_rank": 6},
            }
        },
    }


def _evaluation_report() -> dict[str, object]:
    return {
        "format_version": 1,
        "protocol": "windowed",
        "baseline": "kinematic_persistence",
        "stride": "one_horizon",
        "scoring": {"name": "windowed"},
        "independent_holdout": True,
        "can_promote_model": True,
        "corpus": {"name": "overwritten by the registry block"},
        "dataset": {"trajectory_count": 1},
        "model": {"horizon_rollouts": {}},
        "baseline_metrics": {"horizon_rollouts": {}},
        "score_vs_baseline": 0.4,
    }


def test_a_corpus_validation_artifact_carries_the_whole_contract(
    tmp_path: Path,
) -> None:
    fit_report = tmp_path / "report.json"
    fit_report.write_text(json.dumps(_fit_report()))
    evaluation_report = tmp_path / "benchmark_report.json"
    evaluation_report.write_text(json.dumps(_evaluation_report()))

    document = assemble_corpus_validation_report(
        "x8", evaluation_report, {"structured": fit_report}
    )

    assert set(document) == {
        "artifact_type",
        "format_version",
        "smoke",
        "corpus",
        "protocol",
        "fit",
        "results",
        "baseline",
        "implementation",
        "environment",
    }
    assert document["artifact_type"] == "glassbox_corpus_validation"
    assert document["smoke"] is False

    # The corpus block is the registry's, pinned by digest, not the one the
    # evaluation report carried.
    corpus = document["corpus"]
    assert corpus["name"] == "x8"
    assert corpus["citation"]["pinned_version"] == "1.0"
    assert corpus["files"] and all(item["digest"] for item in corpus["files"])

    # The protocol is the whole policy, and the results are what is left of the
    # evaluation report once the protocol, baseline and corpus are hoisted out.
    assert document["protocol"]["scoring"] == {"name": "windowed"}
    assert set(document["results"]) == {
        "format_version",
        "dataset",
        "model",
        "score_vs_baseline",
    }
    assert document["baseline"]["name"] == "kinematic_persistence"

    # One fit block per scored model: the specification, the split reduced to
    # identity and duration, the optimization record, the parameter evidence.
    arm = document["fit"]["structured"]
    assert set(arm) == {"spec", "split", "fit", "parameter_evidence"}
    assert arm["spec"]["model_class"] == "structured"
    assert "training_windows_per_flight_by_horizon" not in arm["spec"]
    assert arm["split"]["validation"] == {
        "count": 1,
        "duration_s": 4.0,
        "paths": ["held.npz"],
    }
    assert arm["parameter_evidence"] == {"numerical_rank": 6}
    assert document["implementation"]["source_files"] == list(VALIDATION_SOURCE_FILES)


def test_a_smoke_assembly_says_so(tmp_path: Path) -> None:
    fit_report = tmp_path / "report.json"
    fit_report.write_text(json.dumps(_fit_report()))
    evaluation_report = tmp_path / "benchmark_report.json"
    evaluation_report.write_text(json.dumps(_evaluation_report()))

    document = assemble_corpus_validation_report(
        "x8", evaluation_report, {"structured": fit_report}, smoke=True
    )

    assert document["smoke"] is True


def test_a_leave_one_out_summary_takes_its_fit_arms_from_the_folds(
    tmp_path: Path,
) -> None:
    fold_report = tmp_path / "fold_01_a_report.json"
    fold_report.write_text(json.dumps(_fit_report()))
    summary = {
        "format_version": 1,
        "protocol": "windowed",
        "baseline": "kinematic_persistence",
        "evaluation": "leave_one_source_group_out",
        "holdout_label": "source_group",
        "independent_holdout": True,
        "aggregate": {},
        "per_fold": {"session_a": {"report": str(fold_report)}},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))

    document = assemble_corpus_validation_report("idf", summary_path)

    assert set(document["fit"]) == {"session_a"}
    assert document["fit"]["session_a"]["fit"]["final_loss"] == 0.5
    assert document["protocol"]["evaluation"] == "leave_one_source_group_out"


def test_smoke_refuses_a_local_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        record_results.main(
            ["--smoke", "/tmp/glassbox-smoke", "--only", "adaptive-recovery-results"]
        )

    assert excinfo.value.code == 2
    assert "the local tier has neither" in capsys.readouterr().err


def test_smoke_refuses_a_path_with_whitespace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        record_results.main(["--smoke", "/tmp/glassbox smoke"])

    assert excinfo.value.code == 2
    assert "cannot contain whitespace" in capsys.readouterr().err


def test_a_shortened_budget_needs_a_smoke_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        record_results.main(["--fit-steps", "2"])

    assert excinfo.value.code == 2
    assert "only shorten a --smoke run" in capsys.readouterr().err


_VALIDATION_CONTRACT = {
    "artifact_type",
    "format_version",
    "smoke",
    "corpus",
    "protocol",
    "fit",
    "results",
    "baseline",
    "implementation",
    "environment",
}


def test_a_recorded_corpus_validation_artifact_satisfies_the_contract() -> None:
    """Guard the committed corpus artifacts, once the maintainer job records them.

    Nothing in the test suite can regenerate one: each is a multi-hour fit on a
    downloaded corpus. What a test can do is hold the committed file to the
    contract the assembly writes, and refuse a smoke output committed by
    mistake.
    """

    for spec in MANIFEST:
        if not spec.name.startswith("validation-") or not spec.recorded:
            continue
        document = json.loads((_REPO_ROOT / spec.output).read_text())
        assert set(document) == _VALIDATION_CONTRACT, spec.name
        assert document["artifact_type"] == "glassbox_corpus_validation", spec.name
        assert document["smoke"] is False, spec.name
        assert document["corpus"]["files"], spec.name
        assert document["fit"], spec.name


# ---------------------------------------------------------------------------
# The one transient state an entry can be in: declared, not yet recorded. No
# entry is in it today, so its behavior is pinned on a spec written here.


def _pending_spec() -> ArtifactSpec:
    return ArtifactSpec(
        name="fake-pending",
        output="docs/results/not-recorded-yet.json",
        steps=(PythonStep("unreachable", lambda: None),),
        inputs=("corpus fake",),
        tier="corpus",
        doc_page="docs/validation.md",
        awaiting_first_record="the maintainer job has not run yet",
    )


def test_a_pending_entry_is_not_recorded() -> None:
    assert not _pending_spec().recorded
    assert all(spec.recorded for spec in MANIFEST)


def test_check_skips_a_pending_entry_instead_of_failing_on_a_missing_file(
    tmp_path: Path,
) -> None:
    """A declared entry with no committed artifact is not a check failure.

    Comparing against a file that was never recorded would report a difference
    that says nothing about the code, so the check names the entry and moves
    on. The reason it prints is the entry's own.
    """

    results = check_selected([_pending_spec()], tmp_path)

    assert [item.status for item in results] == [
        "skipped: the maintainer job has not run yet"
    ]
    assert not results[0].differences


def test_list_reports_a_pending_entry_as_pending(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(record_results, "MANIFEST", (_pending_spec(),))

    record_results.main(["--list"])

    stdout = capsys.readouterr().out
    assert "pending: the maintainer job has not run yet" in stdout
