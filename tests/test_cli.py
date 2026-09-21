"""The installed CLI uses only recording archives and the public model API."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import glassbox.cli as cli
from glassbox import STATE_CHANNELS, SequenceCollection, SequenceSegment
from glassbox.io.recordings import save_recordings


def archive(path, recording_id):
    states = np.zeros((5, 15))
    states[:, 6:] = np.eye(3).ravel()
    data = SequenceCollection(
        (SequenceSegment(recording_id, "whole", states, np.zeros((4, 2)), 0.1),),
        "cli-fixture",
        STATE_CHANNELS,
        ("command-a [issued]", "command-b [issued]"),
    )
    save_recordings(data, path)
    return path


def test_help_contains_only_supported_commands(capsys):
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "{fit,evaluate}" in output
    assert "corpus" not in output
    assert "benchmark" not in output


def test_fit_loads_compatible_archives_and_writes_model_report(
    tmp_path, monkeypatch, capsys
):
    paths = [archive(tmp_path / f"{name}.npz", name) for name in ("a", "b")]
    received = []

    def fitting(recordings):
        received.append(recordings)
        return SimpleNamespace(
            report={"fixture": True},
            save=lambda path: Path(path).write_bytes(b"model fixture"),
        )

    monkeypatch.setattr(cli, "fit", fitting)
    model, report = tmp_path / "out/model.npz", tmp_path / "out/fit.json"
    cli.main(
        [
            "fit",
            *(str(path) for path in paths),
            "--model",
            str(model),
            "--report",
            str(report),
        ]
    )
    assert [segment.recording_id for segment in received[0].segments] == ["a", "b"]
    assert model.read_bytes() == b"model fixture"
    assert json.loads(report.read_text()) == {"fixture": True}
    assert "wrote model" in capsys.readouterr().out


def test_evaluate_prints_or_saves_report(tmp_path, monkeypatch, capsys):
    path = archive(tmp_path / "held-out.npz", "held-out")
    model = object()
    monkeypatch.setattr(
        cli, "LearnedDynamics", SimpleNamespace(load=lambda path: model)
    )

    def scoring(actual_model, recordings):
        assert actual_model is model
        assert recordings.segments[0].recording_id == "held-out"
        return {"rmse": [0.1], "coverage": None}

    monkeypatch.setattr(cli, "evaluate", scoring)
    args = ["evaluate", "model.npz", str(path)]
    cli.main(args)
    expected = {"rmse": [0.1], "coverage": None}
    assert json.loads(capsys.readouterr().out) == expected
    report = tmp_path / "reports/evaluation.json"
    cli.main([*args, "--report", str(report)])
    assert json.loads(report.read_text()) == expected


def test_invalid_recording_reports_an_actionable_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as result:
        cli.main(
            [
                "fit",
                str(tmp_path / "missing.npz"),
                "--model",
                str(tmp_path / "model.npz"),
            ]
        )
    assert result.value.code == 2
    assert "missing.npz" in capsys.readouterr().err
    assert not (tmp_path / "model.npz").exists()


@pytest.mark.parametrize("command", ["extract", "corpus", "benchmark", "synthetic"])
def test_removed_commands_are_rejected(command):
    with pytest.raises(SystemExit) as result:
        cli.main([command])
    assert result.value.code == 2
