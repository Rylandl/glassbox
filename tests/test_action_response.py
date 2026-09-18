"""Action-response diagnostics preserve direction, failed slots and evidence."""

import copy
import io
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental import action_response as q

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "docs/harness/action-response-v1.json").read_text())
LOWER = np.array([0.0, -0.3, -0.5], dtype=np.float32)
UPPER = np.array([1.0, 0.4, 0.5], dtype=np.float32)


def npz_bytes(**values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def json_bytes(value):
    return json.dumps(value, allow_nan=False).encode()


def test_probes_preserve_baseline_and_isolate_exact_physical_channel_changes():
    baseline = np.array(
        [[0.2, -0.1, 0.3], [0.7, 0.2, -0.1], [0.0, 0.4, -0.5]],
        dtype=np.float32,
    )
    original = baseline.copy()
    probes = q.command_probes(baseline, LOWER, UPPER)
    assert probes.shape == (7, 3, 3)
    assert probes.dtype == np.float32
    np.testing.assert_array_equal(baseline, original)
    np.testing.assert_array_equal(probes[0], baseline)
    for channel in range(3):
        for sign, bound in enumerate((LOWER, UPPER)):
            index = 1 + 2 * channel + sign
            others = [value for value in range(3) if value != channel]
            np.testing.assert_array_equal(
                probes[index, :, others], baseline[:, others].T
            )
            expected = np.float32(0.95) * baseline[:, channel]
            expected += np.float32(0.05) * bound[channel]
            np.testing.assert_array_equal(probes[index, :, channel], expected)
    assert np.all(probes >= LOWER)
    assert np.all(probes <= UPPER)


@pytest.mark.parametrize("bound,sign", [(LOWER, 0), (UPPER, 1)])
def test_bound_probes_remain_present_when_the_whole_sequence_is_inactive(bound, sign):
    baseline = np.tile(bound, (5, 1))
    probes = q.command_probes(baseline, LOWER, UPPER)
    assert probes.shape == (7, 5, 3)
    for channel in range(3):
        np.testing.assert_array_equal(probes[1 + 2 * channel + sign], baseline)
        assert np.any(probes[1 + 2 * channel + 1 - sign] != baseline)


def test_equal_interpolation_fraction_is_not_equal_physical_command_size():
    baseline = np.tile((LOWER + UPPER) / 2, (5, 1))
    probes = q.command_probes(baseline, LOWER, UPPER)
    maximum_changes = [np.max(np.abs(probes[2 + 2 * j] - baseline)) for j in range(3)]
    np.testing.assert_allclose(maximum_changes, [0.025, 0.0175, 0.025], atol=1e-7)


@pytest.mark.parametrize("defect", ["low", "high", "nan", "reversed_bounds", "shape"])
def test_invalid_commands_are_rejected_instead_of_silently_clipped(defect):
    baseline = np.tile((LOWER + UPPER) / 2, (5, 1))
    lower, upper = LOWER.copy(), UPPER.copy()
    if defect == "low":
        baseline[0, 0] = -0.01
    elif defect == "high":
        baseline[0, 1] = 0.41
    elif defect == "nan":
        baseline[0, 2] = np.nan
    elif defect == "reversed_bounds":
        lower[0] = 2
    else:
        baseline = baseline[:, :2]
    with pytest.raises((ValueError, AssertionError)):
        q.command_probes(baseline, lower, upper)


@pytest.mark.parametrize(
    "prediction,cosine,gain,ratio,error,negative",
    [
        ([-1.0, 0.0], -1.0, -1.0, 1.0, 2.0, 1.0),
        ([2.0, 0.0], 1.0, 2.0, 2.0, 1.0, 0.0),
        ([0.0, 1.0], 0.0, 0.0, 1.0, np.sqrt(2), 0.0),
    ],
)
def test_direction_reversal_is_separate_from_magnitude_error(
    prediction, cosine, gain, ratio, error, negative
):
    result = q.response_statistics(np.array([prediction]), np.array([[1.0, 0.0]]))
    assert result["total_count"] == result["valid_count"] == 1
    assert result["invalid_count"] == result["weak_true_count"] == 0
    assert result["normalizable_count"] == result["cosine_count"] == 1
    for name, value in (
        ("cosine", cosine),
        ("projected_gain", gain),
        ("magnitude_ratio", ratio),
        ("relative_response_error", error),
    ):
        assert result[name] == dict(count=1, median=value, p10=value, p90=value)
    assert result["rms_vector_error"] == pytest.approx(error)
    assert result["negative_projected_gain_fraction"] == negative


def test_zero_prediction_retains_zero_gain_and_unit_relative_error_without_cosine():
    result = q.response_statistics(np.zeros((3, 2)), np.tile([3.0, 4.0], (3, 1)))
    assert result["normalizable_count"] == 3
    assert result["weak_predicted_count"] == 3
    assert result["cosine_count"] == result["cosine"]["count"] == 0
    assert result["cosine"]["median"] is None
    assert result["magnitude_ratio"]["median"] == 0
    assert result["projected_gain"]["median"] == 0
    assert result["relative_response_error"]["median"] == 1
    assert result["rms_vector_error"] == 5
    assert result["negative_projected_gain_fraction"] == 0


def test_weak_truth_is_retained_in_absolute_errors_but_not_normalized_denominators():
    truth = np.array([[0.0, 0.0], [0.5e-6, 0.0], [1e-6, 0.0], [2e-6, 0.0]])
    result = q.response_statistics(np.zeros_like(truth), truth)
    assert result["total_count"] == result["valid_count"] == 4
    assert result["weak_true_count"] == 3
    assert result["weak_predicted_count"] == 4
    assert result["normalizable_count"] == 1
    assert result["cosine_count"] == 0
    assert result["true_response_norm"]["count"] == 4
    assert result["magnitude_ratio"]["count"] == 1
    assert result["rms_vector_error"] == pytest.approx(np.sqrt(5.25e-12 / 4))


def test_invalid_vectors_have_explicit_denominators_and_never_partial_component_credit():
    truth = np.array([[1.0, 0.0], [0.0, 0.0], [np.nan, 0.0], [1.0, 0.0]])
    predicted = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [np.inf, 0.0]])
    result = q.response_statistics(predicted, truth)
    assert result["total_count"] == 4
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 2
    assert result["weak_true_count"] == 1
    assert result["normalizable_count"] == result["cosine_count"] == 1
    assert result["true_response_norm"]["count"] == 2
    assert result["rms_vector_error"] == 0
    absolute = q.absolute_statistics(predicted, truth)
    assert absolute == dict(
        total_count=4,
        valid_count=2,
        invalid_count=2,
        component_rmse=0.0,
        mean_vector_error=0.0,
    )


def test_completely_invalid_population_is_not_reported_as_zero_error():
    predicted, truth = np.full((2, 3), np.nan), np.ones((2, 3))
    response = q.response_statistics(predicted, truth)
    absolute = q.absolute_statistics(predicted, truth)
    assert response["total_count"] == absolute["total_count"] == 2
    assert response["invalid_count"] == absolute["invalid_count"] == 2
    assert response["valid_count"] == absolute["valid_count"] == 0
    assert response["rms_vector_error"] is None
    assert response["negative_projected_gain_fraction"] is None
    assert absolute["component_rmse"] is absolute["mean_vector_error"] is None
    for name in (
        "cosine",
        "magnitude_ratio",
        "projected_gain",
        "relative_response_error",
    ):
        assert response[name] == dict(count=0, median=None, p10=None, p90=None)
    json.dumps(response, allow_nan=False)
    json.dumps(absolute, allow_nan=False)


def test_component_rmse_and_mean_vector_error_are_distinct_quantities():
    predicted = np.array([[[3.0, 4.0, 0.0]], [[0.0, 0.0, 0.0]]])
    result = q.absolute_statistics(predicted, np.zeros_like(predicted))
    assert result["total_count"] == 2
    assert result["component_rmse"] == pytest.approx(np.sqrt(25 / 6))
    assert result["mean_vector_error"] == 2.5


def test_full_horizon_vector_keeps_early_error_when_endpoint_response_is_correct():
    truth = np.zeros((2, 5, 3))
    truth[:, :, 0] = 1.0
    predicted = truth.copy()
    predicted[:, 0, 0] = -1.0
    endpoint = q.response_statistics(predicted[:, -1], truth[:, -1])
    full = q.response_statistics(predicted.reshape(2, -1), truth.reshape(2, -1))
    assert endpoint["rms_vector_error"] == 0
    assert endpoint["projected_gain"]["median"] == 1
    assert full["rms_vector_error"] == 2
    assert full["projected_gain"]["median"] == pytest.approx(0.6)
    assert full["relative_response_error"]["median"] == pytest.approx(2 / np.sqrt(5))


def test_full_horizon_weakness_uses_the_flattened_norm_without_scaling_the_threshold():
    truth = np.full((1, 5, 1), 0.75e-6)
    per_step = q.response_statistics(truth, truth)
    full = q.response_statistics(truth.reshape(1, -1), truth.reshape(1, -1))
    assert per_step["weak_true_count"] == 5
    assert per_step["normalizable_count"] == 0
    assert full["weak_true_count"] == 0
    assert full["normalizable_count"] == 1
    assert full["cosine"]["median"] == pytest.approx(1)


@pytest.mark.parametrize("function", ["absolute_statistics", "response_statistics"])
def test_metric_pairs_must_have_exact_matching_shapes(function):
    with pytest.raises((ValueError, AssertionError)):
        getattr(q, function)(np.ones((2, 3)), np.ones((1, 3)))


def trial_fixture():
    """All frozen slots, with deliberately different direction and gain by group."""
    count = len(PLAN["population"]["origins"])
    baseline = np.tile((LOWER + UPPER) / 2, (5, 1))
    future = np.tile(q.command_probes(baseline, LOWER, UPPER), (count, 1, 1, 1))
    delta = future - future[:, :1]
    reference = np.zeros((count, 7, 5, 15), dtype=np.float32)
    # Every probe has a nonzero response in each group; physical groups stay separate.
    effect = np.sum(delta, axis=-1) * np.arange(1, 6, dtype=np.float32)
    for column in (0, 3, 6):
        reference[..., column] = effect
    learned = reference.copy()
    learned[..., 0] *= 2
    learned[..., 3] *= -1
    learned[..., 6] *= 0
    # Constant forecast bias affects the absolute metric but must cancel in response.
    learned[..., 0] += 3
    learned[..., 3] += 4
    canonical = np.zeros((count, 7, 5, 13), dtype=np.float32)
    canonical[..., 6] = 1
    replayed = np.zeros((321, 13), dtype=np.float64)
    replayed[:, 6] = 1
    trial = dict(
        origins=np.asarray(PLAN["population"]["origins"], dtype=np.int64),
        observations=np.zeros((count, 11, 15), dtype=np.float32),
        past_commands=np.tile(baseline[0], (count, 10, 1)),
        future_commands=future,
        command_deltas=delta,
        learned=learned,
        reference=reference,
        canonical=canonical,
        learner_valid=np.ones((count, 7), dtype=bool),
        reference_valid=np.ones((count, 7), dtype=bool),
        replayed_states=replayed,
    )
    return [copy.deepcopy(trial) for _ in range(4)], [[] for _ in range(4)]


def test_report_separates_units_horizons_probe_signs_and_own_baseline_offsets():
    trials, failures = trial_fixture()
    result = q.report(PLAN, trials, failures, {"fingerprint": "fixture"})
    assert result["no_fit"]
    assert not any(
        result[name]
        for name in ("learner_changed", "controller_changed", "application_qualified")
    )
    assert len(result["trials"]) == 4
    assert result["horizon_seconds"] == pytest.approx([0.05, 0.1, 0.15, 0.2, 0.25])
    pooled = result["pooled"]
    assert pooled["origins"] == 124
    assert pooled["forecast_slots"] == pooled["learner_valid"] == 868
    assert pooled["learner_invalid"] == pooled["reference_invalid"] == 0
    assert set(pooled["groups"]) == {"velocity", "body_rate", "rotation_entries"}
    for group, gain, cosine, units in (
        ("velocity", 2, 1, "m/s"),
        ("body_rate", -1, -1, "rad/s"),
        ("rotation_entries", 0, None, "unitless"),
    ):
        data = pooled["groups"][group]
        assert data["units"] == units
        assert set(data["response"]) == {"pooled_six_probes", *q.PROBES[1:]}
        for name, response in data["response"].items():
            assert len(response["by_horizon"]) == 5
            expected = 744 if name == "pooled_six_probes" else 124
            for value in [*response["by_horizon"], response["full_horizon"]]:
                assert value["total_count"] == value["valid_count"] == expected
                assert value["projected_gain"]["median"] == pytest.approx(
                    gain, abs=2e-5
                )
                if cosine is None:
                    assert value["cosine"]["median"] is None
                    assert value["cosine_count"] == 0
                else:
                    assert value["cosine"]["median"] == pytest.approx(cosine, abs=2e-5)
        for baseline in data["absolute"]["baseline"]:
            assert baseline["total_count"] == 124
        for all_sequences in data["absolute"]["all_sequences"]:
            assert all_sequences["total_count"] == 868
    velocity = pooled["groups"]["velocity"]
    assert velocity["absolute"]["baseline"][0]["component_rmse"] == pytest.approx(
        np.sqrt(3)
    )
    assert velocity["absolute"]["baseline"][0]["mean_vector_error"] == 3
    error = [
        x["rms_vector_error"]
        for x in velocity["response"]["pooled_six_probes"]["by_horizon"]
    ]
    assert all(second > first for first, second in pairwise(error))
    for trial in result["trials"]:
        assert trial["origins"] == 31
        assert trial["forecast_slots"] == 217


@pytest.mark.parametrize("probe_index,invalid_responses", [(0, 6), (1, 1)])
def test_failed_baseline_or_probe_propagates_into_response_denominators(
    probe_index, invalid_responses
):
    trials, failures = trial_fixture()
    trials[0]["learned"][0, probe_index] = np.nan
    trials[0]["learner_valid"][0, probe_index] = False
    failures[0].append(
        dict(
            origin=10,
            probe=q.PROBES[probe_index],
            predictor="learner",
            kind="nonfinite",
        )
    )
    result = q.report(PLAN, trials, failures, {})
    pooled = result["pooled"]
    assert pooled["forecast_slots"] == 868
    assert pooled["learner_valid"] == 867
    assert pooled["learner_invalid"] == 1
    assert result["trials"][0]["failures"] == failures[0]
    for group in pooled["groups"].values():
        for score in group["absolute"]["all_sequences"]:
            assert score["total_count"] == 868
            assert score["valid_count"] == 867
            assert score["invalid_count"] == 1
        response = group["response"]["pooled_six_probes"]
        for score in [*response["by_horizon"], response["full_horizon"]]:
            assert score["total_count"] == 744
            assert score["invalid_count"] == invalid_responses
            assert score["valid_count"] == 744 - invalid_responses


def test_action_size_counts_inactive_probes_without_excluding_their_forecast_slots():
    trials, failures = trial_fixture()
    for trial in trials:
        trial["future_commands"][:] = q.command_probes(
            np.tile(UPPER, (5, 1)), LOWER, UPPER
        )
        trial["command_deltas"] = (
            trial["future_commands"] - trial["future_commands"][:, :1]
        )
    result = q.report(PLAN, trials, failures, {})["pooled"]
    assert result["forecast_slots"] == 868
    size = result["action_size"]
    assert size["pooled_six_probes"]["sequences"] == 744
    assert size["pooled_six_probes"]["fully_inactive_sequences"] == 372
    np.testing.assert_allclose(
        size["pooled_six_probes"]["maximum_absolute_by_channel"],
        [0.05, 0.035, 0.05],
        atol=1e-7,
    )
    for channel in range(3):
        inactive = size[f"command_{channel}_plus"]
        assert inactive["fully_inactive_sequences"] == inactive["sequences"] == 124
        assert inactive["rms_by_channel"] == [0.0] * 3
        active = size[f"command_{channel}_minus"]
        assert active["fully_inactive_sequences"] == 0
        assert sum(value > 0 for value in active["rms_by_channel"]) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "missing_trial",
        "missing_origin",
        "duplicate_origin",
        "wrong_origin",
        "extra_array",
        "missing_array",
        "wrong_dtype",
        "nonfinite_history",
        "wrong_delta",
        "nonfinite_valid_forecast",
        "unrecorded_failure",
        "partial_failure",
        "fake_failure",
    ],
)
def test_report_rejects_missing_slots_and_inconsistent_evidence(defect):
    trials, failures = trial_fixture()
    trial = trials[0]
    if defect == "missing_trial":
        trials.pop()
    elif defect == "missing_origin":
        trial["origins"] = trial["origins"][:-1]
    elif defect == "duplicate_origin":
        trial["origins"][1] = trial["origins"][0]
    elif defect == "wrong_origin":
        trial["origins"][0] += 1
    elif defect == "extra_array":
        trial["posthoc"] = np.array([True])
    elif defect == "missing_array":
        trial.pop("past_commands")
    elif defect == "wrong_dtype":
        trial["future_commands"] = trial["future_commands"].astype(np.float64)
    elif defect == "nonfinite_history":
        trial["observations"][0, 0, 0] = np.nan
    elif defect == "wrong_delta":
        trial["command_deltas"][0, 1, 0, 0] += 0.01
    elif defect == "nonfinite_valid_forecast":
        trial["learned"][0, 0, 0, 0] = np.nan
    elif defect in {"unrecorded_failure", "partial_failure"}:
        trial["learned"][0, 1] = np.nan
        trial["learner_valid"][0, 1] = False
        if defect == "partial_failure":
            trial["learned"][0, 1, 0, 0] = 1
            failures[0].append(
                dict(
                    origin=10, probe=q.PROBES[1], predictor="learner", kind="nonfinite"
                )
            )
    else:
        failures[0].append(
            dict(origin=10, probe=q.PROBES[1], predictor="learner", kind="nonfinite")
        )
    with pytest.raises((ValueError, AssertionError)):
        q.report(PLAN, trials, failures, {})


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """Public run and verifier; fresh numerical evidence never reads saved output."""
    plan = copy.deepcopy(PLAN)
    pinned = b"synthetic pinned model bytes"
    plan["input_paths"] = {"control/generic.npz": "generic.npz"}
    plan["input_sha256"] = {"control/generic.npz": q.digest(pinned)}
    raw = json_bytes(plan)
    trials, failures = trial_fixture()
    metadata = dict(fingerprint="fixed-fixture", recipe={"id": "fixture"})
    calls = []

    def fresh(proposed_plan, inputs):
        q.exact(proposed_plan, plan)
        assert inputs == {"control/generic.npz": pinned}
        calls.append("fresh numerical reconstruction")
        return copy.deepcopy(trials), copy.deepcopy(failures), copy.deepcopy(metadata)

    monkeypatch.setattr(q, "frozen_plan", lambda: (plan, raw))
    monkeypatch.setattr(q, "check_environment", lambda _: {"python": "test-double"})
    monkeypatch.setattr(q, "probe", fresh)
    artifact_root = tmp_path / "inputs"
    artifact_root.mkdir()
    (artifact_root / "generic.npz").write_bytes(pinned)
    output = tmp_path / "run"
    q.run(artifact_root, output)
    result = q.verify(output)
    assert result["verified_trials"] == 4
    assert result["verified_origins"] == 124
    assert result["verified_forecast_sequences"] == 868
    assert result["exact_physical_replay"] and result["exact_learner_replay"]
    assert len(calls) == 2
    return plan, output, calls


def rewrite_report_and_hashes(plan, directory, *, report_mutation=None):
    trials = [
        q.arrays((directory / f"trial-{i}/arrays.npz").read_bytes()) for i in range(4)
    ]
    failures = [
        json.loads((directory / f"trial-{i}/failures.json").read_bytes())
        for i in range(4)
    ]
    metadata = json.loads((directory / "model.json").read_bytes())
    result = q.report(plan, trials, failures, metadata)
    if report_mutation is not None:
        report_mutation(result)
    (directory / "report.json").write_bytes(json_bytes(result))
    rewrite_hashes(plan, directory)


def rewrite_hashes(plan, directory):
    value = dict(
        plan_sha256=q.PLAN_SHA256,
        files={
            name: q.digest((directory / name).read_bytes())
            for name in sorted(q.artifact_names(plan) - {"run.json"})
        },
    )
    (directory / "run.json").write_bytes(json_bytes(value))


@pytest.mark.parametrize(
    "defect", ["history", "future_command", "learned", "reference", "model"]
)
def test_fresh_replay_rejects_rehashed_changes_with_recomputed_summaries(
    saved_run, defect
):
    plan, output, calls = saved_run
    path = output / "trial-0/arrays.npz"
    arrays = q.arrays(path.read_bytes())
    if defect == "history":
        arrays["observations"][0, 0, 0] += 0.01
    elif defect == "future_command":
        arrays["future_commands"][0, 1, 0, 0] += 0.001
        arrays["command_deltas"] = (
            arrays["future_commands"] - arrays["future_commands"][:, :1]
        )
    elif defect in {"learned", "reference"}:
        arrays[defect][0, 1, 0, 0] += 0.01
    else:
        metadata = json.loads((output / "model.json").read_bytes())
        metadata["fingerprint"] = "forged-but-rehashed"
        (output / "model.json").write_bytes(json_bytes(metadata))
    path.write_bytes(npz_bytes(**arrays))
    rewrite_report_and_hashes(plan, output)
    before = len(calls)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
    assert len(calls) == before + 1


def test_rehashed_response_summary_is_recomputed_before_replay(saved_run):
    plan, output, calls = saved_run

    def tamper(result):
        result["pooled"]["groups"]["body_rate"]["response"]["pooled_six_probes"][
            "full_horizon"
        ]["projected_gain"]["median"] = 1.0

    rewrite_report_and_hashes(plan, output, report_mutation=tamper)
    before = len(calls)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
    assert len(calls) == before


@pytest.mark.parametrize(
    "defect",
    ["pinned_input", "manifest", "environment", "extra_file", "missing_file", "hash"],
)
def test_public_verify_rejects_pins_and_roster_before_numerical_replay(
    saved_run, defect
):
    plan, output, calls = saved_run
    if defect == "pinned_input":
        (output / "inputs/control/generic.npz").write_bytes(b"other model bytes")
    elif defect == "manifest":
        with (output / "manifest.json").open("ab") as stream:
            stream.write(b" ")
    elif defect == "environment":
        (output / "environment.json").write_bytes(b"{}")
    elif defect == "extra_file":
        (output / "unfrozen.json").write_bytes(b"{}")
    elif defect == "missing_file":
        (output / "trial-3/failures.json").unlink()
    else:
        with (output / "model.json").open("ab") as stream:
            stream.write(b" ")
    if defect not in {"missing_file", "hash"}:
        rewrite_hashes(plan, output)
    before = len(calls)
    with pytest.raises((ValueError, AssertionError)):
        q.verify(output)
    assert len(calls) == before
