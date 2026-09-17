"""Temporal evidence, revision selection, and accounting in streaming refinement."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.forecast_error import ForecastErrorEnvelope
from glassbox.core.data import trajectory_content_digest, trajectory_segment
from glassbox.core.fixedwing_synthetic import true_fixed_wing_parameters
from glassbox.core.metrics import state_rmse_metrics
from glassbox.core.model import (
    ExecutableModel,
    ModelValidityEnvelope,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import true_parameters
from glassbox.workflows.refinement import ModelRefiner


@pytest.fixture(scope="module", params=["multirotor", "fixedwing"])
def fixture(request, quadrotor_flight, fixedwing_flight):
    fixedwing = request.param == "fixedwing"
    flight = (fixedwing_flight if fixedwing else quadrotor_flight)(21, 0.4)
    params = true_fixed_wing_parameters() if fixedwing else true_parameters()
    runtime = replace(
        runtime_spec_from_trajectory(flight),
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(100.0, 100.0, 100.0),
        ),
    )
    belief = DynamicsBelief(
        ExecutableModel(params, flight.spec, runtime),
        forecast_error=ForecastErrorEnvelope(
            horizons_s=(0.1,),
            tangent_covariance=np.eye(12)[None],
            raw_sample_count=(10,),
            effective_sample_count=(10.0,),
            independent_group_count=(1,),
        ),
        provenance={"update_count": 5, "parameter_distance_since_measurement": 2.0},
    )
    return belief, flight


def test_matches_manual_score_then_learn_and_keeps_active_pinned(fixture):
    initial, flight = fixture
    refiner = ModelRefiner(initial)
    pinned = refiner.active
    manual = initial
    for start in (0, 5, 10, 15):
        block = trajectory_segment(flight, start, start + 5)
        expected = manual.rollout(
            block.states[0],
            block.controls,
            command_history=block.control_prefix,
            exogenous=block.exogenous[:-1],
        )
        result = refiner.observe(
            block, recording_id="flight", start_interval=np.int64(start)
        )
        np.testing.assert_array_equal(
            result.candidate_score.prediction.states, expected.states
        )
        assert result.candidate_score.rmse == state_rmse_metrics(
            np.asarray(expected.states)[1:], block.states[1:]
        )
        manual, update = manual.absorb(block)
        assert result.update.to_dict() == update.to_dict()
        np.testing.assert_array_equal(
            refiner.candidate.belief.information.precision, manual.information.precision
        )
        assert refiner.active is pinned
        assert result.active_score.revision is pinned
        assert result.candidate_after.belief.forecast_error is initial.forecast_error
        assert result.candidate_after.belief.support is initial.support
        assert result.candidate_after.belief.parameter_distance_since_measurement >= 2.0
        json.dumps(result.to_dict(), allow_nan=False)
    assert refiner.candidate.belief.update_count == 9
    # Prefix commands initialize actuators but do not add information twice.
    assert refiner.candidate.belief.information.effective_count == len(flight.controls)
    assert initial.update_count == 5


def test_both_forecasts_precede_absorption(fixture, monkeypatch):
    belief, flight = fixture
    refiner = ModelRefiner(belief)
    first = refiner.observe(
        trajectory_segment(flight, 0, 5), recording_id="f", start_interval=0
    )
    events = []
    rollout, absorb = DynamicsBelief.rollout, DynamicsBelief.absorb

    def tracked_rollout(self, *args, **kwargs):
        events.append(("predict", self))
        return rollout(self, *args, **kwargs)

    def tracked_absorb(self, *args, **kwargs):
        events.append(("learn", self))
        return absorb(self, *args, **kwargs)

    monkeypatch.setattr(DynamicsBelief, "rollout", tracked_rollout)
    monkeypatch.setattr(DynamicsBelief, "absorb", tracked_absorb)
    refiner.observe(
        trajectory_segment(flight, 5, 10), recording_id="f", start_interval=5
    )
    assert [kind for kind, _ in events] == ["predict", "predict", "learn"]
    assert events[0][1] is belief
    assert events[1][1] is events[2][1] is first.candidate_after.belief


def test_adopts_scored_revision_not_unscored_successor_and_rejects_stale_decision(
    fixture,
):
    belief, flight = fixture
    refiner = ModelRefiner(belief)
    original = refiner.active
    first = refiner.observe(
        trajectory_segment(flight, 0, 5), recording_id="f", start_interval=0
    )
    with pytest.raises(ValueError, match="scored"):
        refiner.adopt(
            first.candidate_after.revision_id,
            expected_active_revision=original.revision_id,
            reason="too soon",
        )
    second = refiner.observe(
        trajectory_segment(flight, 5, 10), recording_id="f", start_interval=5
    )
    evaluated = second.candidate_score.revision
    adoption = refiner.adopt(
        evaluated.revision_id,
        expected_active_revision=original.revision_id,
        reason="offline decision",
    )
    assert refiner.active is first.candidate_after is evaluated
    assert refiner.candidate is second.candidate_after
    assert adoption.after_block_count == 2
    assert refiner.adoptions == (adoption,)
    with pytest.raises(ValueError, match="active revision changed"):
        refiner.adopt(
            original.revision_id,
            expected_active_revision=original.revision_id,
            reason="stale",
        )
    refiner.adopt(
        original.revision_id,
        expected_active_revision=evaluated.revision_id,
        reason="explicit rollback",
    )
    assert refiner.active is original


@pytest.mark.parametrize("start", [-1, 0, 4, 6])
def test_rejects_repeated_overlapping_or_missing_intervals(fixture, start):
    belief, flight = fixture
    refiner = ModelRefiner(belief)
    refiner.observe(
        trajectory_segment(flight, 0, 5), recording_id="f", start_interval=0
    )
    candidate = refiner.candidate
    with pytest.raises(ValueError, match="expected start_interval 5"):
        refiner.observe(
            trajectory_segment(flight, 5, 10), recording_id="f", start_interval=start
        )
    assert refiner.candidate is candidate
    assert len(refiner.results) == 1


@pytest.mark.parametrize("change", ["configuration", "source", "bounds", "timing"])
def test_contract_failure_does_not_consume_block(fixture, change):
    belief, flight = fixture
    block = trajectory_segment(flight, 0, 5)
    if change == "configuration":
        bad = replace(
            block,
            spec=replace(
                block.spec,
                vehicle=replace(block.spec.vehicle, configuration_id="other"),
            ),
        )
    elif change == "source":
        bad = replace(
            block, spec=replace(block.spec, observation_source="other-estimator")
        )
    elif change == "bounds":
        channels = list(block.spec.channels)
        channels[0] = replace(channels[0], maximum=2.0)
        bad = replace(block, spec=replace(block.spec, channels=tuple(channels)))
    else:
        bad = replace(block, time_s=block.time_s * 2)
    refiner = ModelRefiner(belief)
    with pytest.raises(ValueError, match=r"contract|sample period"):
        refiner.observe(bad, recording_id="f", start_interval=0)
    assert not refiner.results
    assert refiner.candidate.belief is belief
    assert refiner.observe(block, recording_id="f", start_interval=0).update.absorbed


@pytest.mark.parametrize("change", ["prefix", "state"])
def test_preserves_history_and_shared_endpoint(fixture, change):
    belief, flight = fixture
    refiner = ModelRefiner(belief)
    refiner.observe(
        trajectory_segment(flight, 0, 5), recording_id="f", start_interval=0
    )
    block = trajectory_segment(flight, 5, 10)
    if change == "prefix":
        bad = replace(block, control_prefix=block.control_prefix[-1:])
    else:
        states = block.states.copy()
        states[0, 0] += 0.01
        bad = replace(block, states=states)
    with pytest.raises(ValueError, match=r"prefix|endpoint"):
        refiner.observe(bad, recording_id="f", start_interval=5)
    assert len(refiner.results) == 1
    assert refiner.observe(block, recording_id="f", start_interval=5).update.absorbed


def test_exact_content_alias_and_fit_reserved_content_are_rejected(fixture):
    belief, flight = fixture
    block = trajectory_segment(flight, 0, 5)
    refiner = ModelRefiner(belief)
    refiner.observe(block, recording_id="f", start_interval=0)
    alias = replace(block, labels={"name": "other"}, provenance={"path": "renamed.npz"})
    with pytest.raises(ValueError, match="already seen"):
        refiner.observe(alias, recording_id="renamed", start_interval=0)
    for role in ("training", "forecast_error_calibration"):
        reserved = replace(
            belief,
            provenance={
                "data_identity": {
                    "algorithm": "trajectory_sha256_v1",
                    role: [trajectory_content_digest(block)],
                }
            },
        )
        with pytest.raises(ValueError, match="reserved"):
            ModelRefiner(reserved).observe(alias, recording_id="f", start_interval=0)


def test_failed_computation_leaves_session_retryable(fixture, monkeypatch):
    belief, flight = fixture
    refiner = ModelRefiner(belief)
    block = trajectory_segment(flight, 0, 5)

    def fail(self, telemetry):
        raise RuntimeError("worker interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(DynamicsBelief, "absorb", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            refiner.observe(block, recording_id="f", start_interval=0)
    assert not refiner.results
    assert refiner.candidate is refiner.active
    with pytest.raises(ValueError, match="scored"):
        refiner.adopt(
            refiner.active.revision_id,
            expected_active_revision=refiner.active.revision_id,
            reason="incomplete scoring",
        )
    assert refiner.observe(block, recording_id="f", start_interval=0).update.absorbed


def test_refused_update_is_reported_and_scored_block_is_consumed(fixture):
    belief, flight = fixture
    states = flight.states.copy()
    states[:, 3:6] = 1000.0  # Outside the fixture's observed velocity support.
    block = trajectory_segment(replace(flight, states=states), 0, 5)
    refiner = ModelRefiner(belief)
    result = refiner.observe(block, recording_id="f", start_interval=0)
    assert not result.update.absorbed
    assert result.update.reason
    assert result.candidate_after is refiner.active
    assert len(refiner.results) == 1
    with pytest.raises(ValueError, match="expected start_interval 5"):
        refiner.observe(block, recording_id="f", start_interval=0)


def test_nonfinite_forecast_is_explicit_and_truth_can_still_be_absorbed(
    fixture, monkeypatch
):
    belief, flight = fixture
    rollout = DynamicsBelief.rollout

    def nonfinite(self, *args, **kwargs):
        prediction = rollout(self, *args, **kwargs)
        states = np.asarray(prediction.states).copy()
        states[1, 0] = np.nan
        return replace(prediction, states=states)

    monkeypatch.setattr(DynamicsBelief, "rollout", nonfinite)
    result = ModelRefiner(belief).observe(
        trajectory_segment(flight, 0, 5), recording_id="f", start_interval=0
    )
    assert result.update.absorbed
    assert result.candidate_score.rmse is None
    assert not result.candidate_score.to_dict()["finite_state_forecast"]
    json.dumps(result.to_dict(), allow_nan=False)


def test_rejects_model_inputs_that_are_not_the_telemetry_command_space(fixture):
    belief, _ = fixture

    class SquaredMap:
        command_channels = belief.input_spec.controls
        model_control_size = len(command_channels)

        def model_control(self, command):
            return command**2

    mapped = replace(belief, model=replace(belief.model, actuation=SquaredMap()))
    with pytest.raises(ValueError, match="direct command model"):
        ModelRefiner(mapped)
    spec = replace(
        belief.input_spec,
        channels=tuple(
            replace(channel, semantic="measured_actuation")
            if channel.kind == "control"
            else channel
            for channel in belief.input_spec.channels
        ),
    )
    measured = replace(
        belief, model=replace(belief.model, input_spec=spec, actuation=None)
    )
    assert measured.model.actuation is None
    with pytest.raises(ValueError, match="direct command model"):
        ModelRefiner(measured)


def test_recording_boundaries_reset_interval_cursor_and_preserve_external_prefix(
    fixture,
):
    belief, flight = fixture
    # A subrecording may start with known commands from before its own index zero.
    first_recording = trajectory_segment(flight, 0, 10)
    second_recording = trajectory_segment(flight, 10, 20)
    refiner = ModelRefiner(belief)
    for recording_id, recording in (
        ("first", first_recording),
        ("second", second_recording),
    ):
        for start in (0, 5):
            result = refiner.observe(
                trajectory_segment(recording, start, start + 5),
                recording_id=recording_id,
                start_interval=start,
            )
            assert result.update.window_count == 5
    assert refiner.candidate.belief.information.effective_count == 20


def test_invalid_command_does_not_consume_block(fixture):
    belief, flight = fixture
    block = trajectory_segment(flight, 0, 5)
    controls = block.controls.copy()
    controls[2, 0] = 2.0
    refiner = ModelRefiner(belief)
    with pytest.raises(ValueError, match="bounds"):
        refiner.observe(
            replace(block, controls=controls), recording_id="f", start_interval=0
        )
    assert refiner.candidate.belief is belief
    assert not refiner.results


def test_bounded_retention_preserves_pending_adoption_and_unique_ids(fixture):
    belief, flight = fixture
    refiner = ModelRefiner(belief, history_steps=3, retained_blocks=1, max_recordings=1)
    identities = set()
    held = None
    for start in range(0, 20, 2):
        block = trajectory_segment(flight, start, start + 2)
        if block.control_prefix is not None:
            block = replace(block, control_prefix=block.control_prefix[-3:])
        result = refiner.observe(block, recording_id="f", start_interval=start)
        identities.add(result.candidate_after.revision_id)
        if start == 2:
            held = result.candidate_score.revision
            refiner.hold_for_adoption(held.revision_id)
        assert len(refiner.results) == 1
        assert refiner.retained_revision_count <= 4
        assert result.command_history_truncated == (start > 3)
    assert len(identities) == 10
    assert refiner.block_count == 10
    adoption = refiner.adopt(
        held.revision_id,
        expected_active_revision=refiner.active.revision_id,
        reason="prepared decision",
    )
    refiner.release_adoption_hold()
    assert adoption.after_block_count == 10
    assert refiner.active is held
    with pytest.raises(ValueError, match="expected start_interval 20"):
        refiner.observe(
            trajectory_segment(flight, 0, 2), recording_id="f", start_interval=0
        )
    with pytest.raises(ValueError, match="recording retention"):
        refiner.observe(
            trajectory_segment(flight, 0, 2), recording_id="alias", start_interval=0
        )


def test_explicit_gaps_add_no_parameter_information(fixture):
    belief, flight = fixture
    refiner = ModelRefiner(belief, history_steps=3, retained_blocks=2, max_recordings=1)
    refiner.skip(
        recording_id="f", start_interval=0, stop_interval=4, reason="startup history"
    )
    for start in (4, 12):
        if start == 12:
            refiner.skip(
                recording_id="f",
                start_interval=6,
                stop_interval=12,
                reason="queue overflow",
            )
        block = trajectory_segment(flight, start, start + 2)
        block = replace(block, control_prefix=block.control_prefix[-3:])
        assert (
            refiner.observe(
                block, recording_id="f", start_interval=start
            ).update.window_count
            == 2
        )
    assert refiner.skipped_interval_count == 10
    assert refiner.candidate.belief.information.effective_count == 4
    with pytest.raises(ValueError, match="gap must advance"):
        refiner.skip(
            recording_id="f", start_interval=0, stop_interval=4, reason="duplicate"
        )


def test_mean_only_rollout_preserves_predictions_and_explicitly_omits_covariance(
    fixture, monkeypatch
):
    belief, flight = fixture
    belief, _ = belief.absorb(trajectory_segment(flight, 0, 5))
    block = trajectory_segment(flight, 5, 10)
    kwargs = dict(command_history=block.control_prefix, exogenous=block.exogenous[:-1])
    full = belief.rollout(block.states[0], block.controls, **kwargs)
    assert full.parameter_information_rank > 0
    assert full.parameter_covariance is not None

    def unexpected(*args, **kwargs):
        raise AssertionError("mean-only scoring computed parameter covariance")

    monkeypatch.setattr(DynamicsBelief, "_parameter_covariance", unexpected)
    mean = belief.rollout(
        block.states[0], block.controls, propagate_parameter_covariance=False, **kwargs
    )
    np.testing.assert_array_equal(mean.states, full.states)
    np.testing.assert_array_equal(mean.latent_states, full.latent_states)
    np.testing.assert_array_equal(
        mean.forecast_error_covariance, full.forecast_error_covariance
    )
    assert mean.parameter_covariance is None
    assert mean.parameter_information_rank == full.parameter_information_rank
    assert mean.uncertainty_available
    absent = replace(belief, forecast_error=None).rollout(
        block.states[0], block.controls, propagate_parameter_covariance=False, **kwargs
    )
    assert not absent.uncertainty_available
