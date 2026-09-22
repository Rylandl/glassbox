"""Synthetic checks of passive recording, causal inputs and exclusive outputs."""

import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "run_dart", Path(__file__).parents[1] / "scripts/run_dart.py"
)
trial = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trial)


def test_journal_records_before_call_and_preserves_return_identity(tmp_path):
    journal = trial.Journal(tmp_path)
    value = np.array([1.0, 2.0], dtype=np.float32)
    returned = (np.asarray(3.0, dtype=np.float32), value)

    def function(argument):
        events = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text().splitlines()
        ]
        assert len(events) == 1 and events[0]["phase"] == "started"
        with (tmp_path / "arrays.bin").open("rb") as stream:
            stream.seek(events[0]["arrays"]["flat"])
            actual = np.load(stream, allow_pickle=False)
        assert actual.dtype == value.dtype and np.array_equal(actual, value)
        assert argument is value
        return returned

    result = journal.call(
        "gradient",
        3,
        dict(flat=value),
        function,
        (value,),
        lambda result: dict(objective=result[0], gradient=result[1]),
    )
    journal.close()
    assert result is returned
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [e["phase"] for e in events] == ["started", "returned"]
    with (tmp_path / "arrays.bin").open("rb") as stream:
        for name, expected in dict(objective=returned[0], gradient=returned[1]).items():
            stream.seek(events[-1]["arrays"][name])
            actual = np.load(stream, allow_pickle=False)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            np.testing.assert_array_equal(actual, expected)
    assert journal.counts["gradient"] == dict(
        started=1, returned=1, raised=0, nonfinite=0
    )


def test_nonfinite_intermediate_is_recorded_without_changing_planner_control_flow(
    tmp_path,
):
    journal = trial.Journal(tmp_path)
    value = np.array([np.nan], dtype=np.float32)
    assert (
        journal.call(
            "gradient", 0, {}, lambda: value, (), lambda result: dict(result=result)
        )
        is value
    )
    journal.close()
    assert journal.counts["gradient"]["nonfinite"] == 1
    assert journal.counts["gradient"]["returned"] == 1


def test_raised_callback_closes_its_record_and_retains_inputs(tmp_path):
    journal = trial.Journal(tmp_path)

    def fail():
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        journal.call(
            "native",
            0,
            dict(state=np.zeros(17)),
            fail,
            (),
            lambda result: dict(state=result),
        )
    journal.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [e["phase"] for e in events] == ["started", "raised"]
    assert journal.counts["native"]["raised"] == 1


def test_initial_uses_only_observed13_and_issued_history():
    states = np.zeros((52, 17), dtype=np.float32)
    states[:, 6] = 1
    states[:, 13:] = 42
    commands = np.arange(51 * 4, dtype=np.float64).reshape(51, 4)
    mode = jax.config.x64_enabled
    current, past, issued = trial.initial(states, commands)
    assert jax.config.x64_enabled == mode
    assert current.shape == (13,) and past.shape == (51, 15) and issued.shape == (50, 4)
    assert current.dtype == past.dtype == issued.dtype == np.float32
    np.testing.assert_array_equal(past[:, 6:], np.tile(np.eye(3).reshape(9), (51, 1)))
    np.testing.assert_array_equal(issued, commands[-50:].astype(np.float32))
    states[:, 13:] = -900
    for actual, expected in zip(
        trial.initial(states, commands), (current, past, issued), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("steps", [1, 3])
def test_shift_seed_keeps_exact_command_dtype_and_repeats_final_command(steps):
    commands = np.arange(120 * 4, dtype=np.float64).reshape(120, 4)
    shifted = trial.shift_seed(commands, steps)
    assert shifted.dtype == commands.dtype and shifted.shape == commands.shape
    np.testing.assert_array_equal(shifted[:-steps], commands[steps:])
    np.testing.assert_array_equal(
        shifted[-steps:], np.repeat(commands[-1:], steps, axis=0)
    )


def test_prefix_scores_exclude_commands_replaced_by_later_replans():
    states = np.zeros((4, 17), dtype=np.float32)
    states[:, 6] = 1
    commands = np.zeros((3, 4), dtype=np.float64)
    commands[1:] = 1
    prediction = states[:, :13].copy()
    prediction[1, 1] = 0.0005
    prediction[2:, 1] = 100
    selected = [
        dict(origin=0, inputs_commands=np.zeros((3, 4)), result_states=prediction)
    ]
    target = dict(
        normal=[-1, 0, 0], center_m=[0.65, 0, 1], body_contact_offset_m=[0, 0, 0.04]
    )
    result = trial.matched_prefixes(
        selected, dict(states=states, commands=commands), 1, target
    )
    assert len(result["rows"]) == 1
    assert set(result["summary"]) == {"10ms"}
    assert result["summary"]["10ms"]["contact_position_m"][
        "rmse_norm"
    ] == pytest.approx(0.0005)
    with pytest.raises(ValueError, match="prefixes differ"):
        trial.matched_prefixes(
            selected, dict(states=states, commands=commands), 3, target
        )


def test_existing_output_is_untouched(tmp_path):
    marker = tmp_path / "existing"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        trial.run(Path("absent"), Path("absent"), tmp_path, Path("absent"))
    assert marker.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [marker]


def test_failure_before_science_is_preserved_and_cannot_be_retried(
    tmp_path, monkeypatch
):
    output = tmp_path / "run"

    def fail(*args):
        raise ValueError("bad source")

    monkeypatch.setattr(trial, "_run", fail)
    with pytest.raises(ValueError, match="bad source"):
        trial.run(tmp_path, tmp_path, output, tmp_path)
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        trial.run(tmp_path, tmp_path, output, tmp_path)


def test_array_comparison_retains_shape_dtype_and_numeric_differences():
    value = np.array([1.0, 2.0], dtype=np.float32)
    assert trial.array_comparison(value, value)["exact"]
    assert not trial.array_comparison(value, value.astype(np.float64))["exact"]
    assert trial.array_comparison(value + 0.5, value)["max_abs_difference"] == 0.5


def test_model_binding_rejects_changed_or_relative_revision(tmp_path):
    model = tmp_path / "revision.npz"
    model.write_bytes(b"frozen model")
    protocol = {"model": {"path": str(model), "sha256": trial.digest(model)}}
    assert trial.selected_model(tmp_path, protocol) == model
    model.write_bytes(b"different model")
    with pytest.raises(ValueError, match="frozen model identity"):
        trial.selected_model(tmp_path, protocol)
    protocol["model"]["path"] = "revision.npz"
    with pytest.raises(ValueError, match="frozen model identity"):
        trial.selected_model(tmp_path, protocol)


def test_historical_protocol_selects_its_original_revision(tmp_path):
    assert trial.selected_model(tmp_path, {}) == tmp_path / "models/dart.npz"
