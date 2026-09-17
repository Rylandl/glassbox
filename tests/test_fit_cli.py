"""The public command line fits and evaluates the same generic recipe."""

from __future__ import annotations

import json

import numpy as np
import pytest

from glassbox import LearnedDynamics, SequenceCollection, SequenceSegment, cli
from glassbox.io.recordings import load_recordings, save_recordings
from glassbox.workflows.forecast import evaluate


def recordings(name, seed):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(79, 1))
    states = np.zeros((80, 2))
    for i, command in enumerate(inputs):
        states[i + 1] = states[i] * [0.95, 0.91] + [
            0.08 * np.tanh(command[0]),
            0.06 * command[0],
        ]
    return SequenceCollection(
        (SequenceSegment(name, "whole", states, inputs, 0.05),),
        configuration_id="cli-test",
        state_channels=("x0 [unitless]", "x1 [unitless]"),
        input_channels=("u [command,unitless]",),
    )


def test_generic_cli_fits_saves_and_evaluates_the_public_model(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"recording-{i}.npz"
        save_recordings(recordings(f"recording-{i}", i), path)
        paths.append(str(path))
    model_path = tmp_path / "model-without-extension"
    report_path = tmp_path / "fit.json"
    cli.main(
        ["fit", *paths[:2], "--model", str(model_path), "--report", str(report_path)]
    )
    assert model_path.exists() and not model_path.with_suffix(".npz").exists()
    model = LearnedDynamics.load(model_path)
    assert model.report == json.loads(report_path.read_text())
    assert model.recipe["id"] == "generic-memory-v3-prototype"
    before = model.fingerprint()
    evaluation_path = tmp_path / "evaluation.json"
    cli.main(["evaluate", str(model_path), paths[2], "--report", str(evaluation_path)])
    actual = json.loads(evaluation_path.read_text())
    assert actual == evaluate(model, load_recordings(paths[2]))
    assert (
        actual["aggregate"]["windows"] == 80 - model.history_steps - model.horizon_steps
    )
    assert model.fingerprint() == before
    with pytest.raises(SystemExit) as rejected:
        cli.main(["evaluate", str(model_path), paths[0]])
    assert rejected.value.code == 2


@pytest.mark.parametrize(
    "flag",
    [
        "--steps",
        "--model-class",
        "--holdout-count",
        "--learning-rate",
        "--train-fraction",
    ],
)
def test_fit_cli_rejects_removed_recipe_options(flag):
    with pytest.raises(SystemExit) as exc:
        cli.main(["fit", "recordings.npz", "--model", "model.npz", flag, "1"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "flag", ["--horizons", "--protocol", "--model", "--hold-out", "--fit-reports"]
)
def test_evaluate_cli_rejects_removed_selectors(flag):
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "model.npz", "recordings.npz", flag, "1"])
    assert exc.value.code == 2


def test_cli_contract_is_only_input_and_output_paths():
    from glassbox.cli.evaluate import build_parser as evaluate_parser
    from glassbox.cli.fit import build_parser as fit_parser

    assert {action.dest for action in fit_parser()._actions} == {
        "help",
        "recordings",
        "model",
        "report",
    }
    assert {action.dest for action in evaluate_parser()._actions} == {
        "help",
        "model",
        "recordings",
        "report",
    }
