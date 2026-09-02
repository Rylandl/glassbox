from __future__ import annotations

import io
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

# Crazyflow must establish SciPy's array-API process contract before any JAX
# linear algebra test can lazily import SciPy. Keep the optional dependency
# optional while making the combined test order deterministic when installed.
try:
    import crazyflow as _crazyflow
except (ImportError, RuntimeError):
    # Crazyflow raises RuntimeError instead of ImportError when SciPy was
    # imported before it without SCIPY_ARRAY_API=1; treat both as "not
    # installed" here since either way the optional extra is unusable.
    _crazyflow = None

import jax
import numpy as np
import pytest
from _recorded import assert_recorded_close, recorded_result

from glassbox.belief.belief import (
    DynamicsBelief,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.control.flight_supervisor import SupervisorMode, SupervisorReason
from glassbox.control.nmpc import NMPCController
from glassbox.core.dynamics import MOTOR_MIXER
from glassbox.core.runtime import runtime_spec_from_trajectory
from glassbox.core.synthetic import generate_trajectory, resting_state, true_parameters
from glassbox.integrations.crazyflow import (
    CrazyflowPlant,
    CrazyflowPlantConfig,
    CrazyflowUnavailableError,
    canonical_state_to_crazyflow,
    crazyflow_state_to_canonical,
    crazyflow_to_glassbox_motors,
    glassbox_to_crazyflow_motors,
    motor_rpm_from_thrust,
    motor_thrust_from_rpm,
)
from glassbox.integrations.crazyflow_animation import (
    CrazyflowAnimationConfig,
    _interpolate_sample,
    _storyboard,
    _throw_storyboard,
    render_crazyflow_throw_trace,
)
from glassbox.integrations.crazyflow_bootstrap import (
    run_crazyflow_bootstrap_benchmark,
    run_crazyflow_bootstrap_trial,
)
from glassbox.integrations.crazyflow_fleet import (
    FLEET_LOG_ARM_LENGTH_RATIOS,
    _configuration_direction_prior,
)
from glassbox.integrations.crazyflow_online import (
    _belief_from_worker_update,
    _eligible_recovery_evidence_start,
    _initialize_adaptation_worker,
    _online_recovery_trajectory,
    _post_update_controller_template,
    _run_isolated_belief_update,
    _set_process_suspended,
)
from glassbox.integrations.crazyflow_prototype import run_crazyflow_prototype
from glassbox.integrations.crazyflow_supervisor_campaign import (
    _SUPERVISOR_FAULT_EXPECTATIONS,
    _simulate_supervisor_fault_campaign,
)
from glassbox.integrations.crazyflow_telemetry import (
    _crazyflow_solver_policy,
    generate_crazyflow_trajectory,
)
from glassbox.integrations.crazyflow_throw import (
    CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS,
    run_crazyflow_throw_campaign,
    run_crazyflow_throw_trial,
)
from glassbox.integrations.crazyflow_throw_study import (
    CRAZYFLOW_THROW_STUDY_CASES,
    CrazyflowStudyTrace,
    format_study_table,
    run_crazyflow_throw_study,
)

CRAZYFLOW_MIXER = np.asarray(
    (
        (-1.0, -1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0, 1.0),
    )
)


def test_motor_permutation_preserves_the_canonical_mixer() -> None:
    canonical = np.asarray((0.11, 0.23, 0.37, 0.41))
    crazyflow = glassbox_to_crazyflow_motors(canonical)

    np.testing.assert_allclose(
        CRAZYFLOW_MIXER @ crazyflow,
        np.asarray(MOTOR_MIXER) @ canonical,
    )
    np.testing.assert_allclose(crazyflow_to_glassbox_motors(crazyflow), canonical)


def test_state_conversion_round_trips_quaternion_storage() -> None:
    state = resting_state()
    state[0:3] = (1.0, -2.0, 3.0)
    state[3:6] = (0.1, 0.2, -0.3)
    state[6:10] = np.asarray((0.8, 0.2, -0.4, 0.4))
    state[6:10] /= np.linalg.norm(state[6:10])
    state[10:13] = (0.5, -0.6, 0.7)

    converted = canonical_state_to_crazyflow(state)
    recovered = crazyflow_state_to_canonical(
        pos=converted["pos"],
        vel=converted["vel"],
        quat_xyzw=converted["quat"],
        ang_vel=converted["ang_vel"],
    )

    np.testing.assert_allclose(recovered, state)


def test_quadratic_motor_thrust_conversion_round_trips() -> None:
    coefficients = np.asarray((0.0, -3.133427287299859e-7, 4.407354891648379e-10))
    thrust = np.asarray((0.0, 0.04, 0.11, 0.20))

    rpm = motor_rpm_from_thrust(thrust, coefficients)

    assert np.all(rpm >= 0.0)
    np.testing.assert_allclose(
        motor_thrust_from_rpm(rpm, coefficients),
        thrust,
        rtol=1e-10,
        atol=1e-12,
    )


def test_plant_config_requires_an_integral_control_interval() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CrazyflowPlantConfig(
            simulation_frequency_hz=500,
            control_frequency_hz=60,
        )


def test_plant_wraps_scipy_array_api_runtime_error(monkeypatch) -> None:
    """CrazyflowPlant must translate crazyflow's SciPy-array-API RuntimeError.

    ``crazyflow/__init__.py`` raises RuntimeError, not ImportError, when
    SciPy was imported before it without SCIPY_ARRAY_API=1. Simulate that
    with a poisoned meta-path finder so the guard is exercised regardless of
    whether the real optional extra is installed in this environment.
    """

    class _PoisonedFinder:
        @staticmethod
        def find_spec(name, path, target=None):
            if name == "crazyflow":
                raise RuntimeError("set SCIPY_ARRAY_API=1 before importing SciPy")
            return None

    monkeypatch.delitem(sys.modules, "crazyflow", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_PoisonedFinder(), *sys.meta_path])

    with pytest.raises(CrazyflowUnavailableError, match="SCIPY_ARRAY_API"):
        CrazyflowPlant()


@pytest.mark.crazyflow
def test_cold_motor_reset_reports_exact_zero_normalized_thrust() -> None:
    pytest.importorskip("crazyflow")
    plant = CrazyflowPlant()
    state = resting_state()
    state[2] = 1.0
    try:
        sample = plant.reset(
            state,
            applied_motor_thrust_fraction=np.zeros(4),
        )
    finally:
        plant.close()

    np.testing.assert_array_equal(
        sample.applied_motor_thrust_fraction,
        np.zeros(4),
    )


def test_animation_interpolation_normalizes_quaternion_and_aligns_command() -> None:
    timestamps = np.asarray((0.0, 1.0))
    states = np.zeros((2, 13), dtype=np.float64)
    states[:, 6] = 1.0
    states[1, 0:3] = (1.0, 2.0, 3.0)
    states[1, 6:10] = (0.0, 1.0, 0.0, 0.0)
    commands = np.asarray(((0.2, 0.3, 0.4, 0.5), (0.6, 0.7, 0.8, 0.9)))

    state, command, index = _interpolate_sample(
        timestamps,
        states,
        commands,
        0.5,
    )

    assert index == 0
    np.testing.assert_allclose(state[0:3], (0.5, 1.0, 1.5))
    assert np.linalg.norm(state[6:10]) == pytest.approx(1.0)
    np.testing.assert_allclose(command, (0.4, 0.5, 0.6, 0.7))


def test_animation_config_requires_even_video_dimensions() -> None:
    with pytest.raises(ValueError, match="even"):
        CrazyflowAnimationConfig(width=1279)


def test_crazyflow_prototype_uses_fixed_deeper_backtracking() -> None:
    policy = _crazyflow_solver_policy()

    assert policy.horizon_steps == 30
    assert policy.block_count == 10
    assert policy.maximum_iterations == 6
    assert policy.line_search_steps == 16


def test_fixed_supervisor_campaign_covers_every_typed_reason_once() -> None:
    names = [name for name, _, _ in _SUPERVISOR_FAULT_EXPECTATIONS]
    expected_reasons = [
        reason for _, _, reason in _SUPERVISOR_FAULT_EXPECTATIONS if reason is not None
    ]

    assert len(names) == len(set(names))
    assert names[0] == "nominal"
    assert _SUPERVISOR_FAULT_EXPECTATIONS[0][1:] == (
        SupervisorMode.NOMINAL,
        None,
    )
    assert len(expected_reasons) == len(set(expected_reasons))
    assert set(expected_reasons) == set(SupervisorReason)


def test_configuration_prior_does_not_promote_unexplained_fit_scatter() -> None:
    trajectory = generate_trajectory(seed=0, duration_s=0.2)
    runtime_spec = runtime_spec_from_trajectory(trajectory)
    params = true_parameters()
    names = structured_parameter_names(params)
    base = np.asarray(structured_parameter_vector(params), dtype=np.float64)
    direction = np.zeros_like(base)
    direction[names.index("log_angular_accel[0]")] = -1.0
    direction[names.index("log_angular_accel[1]")] = -1.0
    fit_scatter = np.zeros_like(base)
    fit_scatter[names.index("log_linear_drag")] = 1.0
    coordinates = np.asarray(FLEET_LOG_ARM_LENGTH_RATIOS)
    centered_square = np.square(coordinates) - np.mean(np.square(coordinates))
    members = [
        DynamicsBelief(
            with_structured_parameter_vector(
                params,
                base + coordinate * direction + 0.2 * scatter * fit_scatter,
            ),
            trajectory.spec,
            runtime_spec,
        )
        for coordinate, scatter in zip(coordinates, centered_square)
    ]

    prior, unprojected_rank = _configuration_direction_prior(
        members,
        FLEET_LOG_ARM_LENGTH_RATIOS,
    )

    assert unprojected_rank == 2
    assert prior.empirical_rank == 1
    assert np.linalg.matrix_rank(prior.between_member_covariance) == 1


def test_online_recovery_prefix_preserves_state_control_alignment() -> None:
    source = generate_trajectory(seed=9, duration_s=0.2)
    belief = DynamicsBelief(
        true_parameters(),
        source.spec,
        runtime_spec_from_trajectory(source),
    )
    states = [state.copy() for state in source.states[:6]]
    controls = [control.copy() for control in source.controls[:5]]
    observations = [np.zeros(4, dtype=np.float64) for _ in states]

    evidence = _online_recovery_trajectory(
        belief,
        states,
        controls,
        observations,
    )

    assert evidence.states.shape == (6, 13)
    assert evidence.controls.shape == (5, 4)
    assert evidence.observations.shape == (6, 4)
    assert evidence.spec.observation_roles == (
        "applied_front_left_motor_thrust_fraction",
        "applied_front_right_motor_thrust_fraction",
        "applied_rear_right_motor_thrust_fraction",
        "applied_rear_left_motor_thrust_fraction",
    )
    assert evidence.labels["source_group"] == "unknown-target-online-recovery"
    assert not evidence.provenance["telemetry_only"][
        "hidden_physical_parameters_supplied_to_glassbox"
    ]
    assert "arm_length_ratio" not in evidence.labels


def test_online_evidence_waits_for_a_full_valid_contiguous_window() -> None:
    validity = [1.2, 0.95, 0.8]

    assert _eligible_recovery_evidence_start(validity, evidence_steps=2) is None

    validity.append(0.7)

    assert _eligible_recovery_evidence_start(validity, evidence_steps=2) == 1
    assert _eligible_recovery_evidence_start([0.7, np.nan, 0.6], 2) is None


def test_post_update_controller_template_advances_only_update_semantics() -> None:
    source = generate_trajectory(seed=10, duration_s=0.2)
    params = true_parameters()
    names = structured_parameter_names(params)
    parameter_belief = LocalGaussianParameterBelief(
        parameter_names=names,
        covariance=np.eye(len(names)) * 1e-3,
        source="test fleet",
        evidence_count=3,
        effective_sample_count=3.0,
    )
    belief = DynamicsBelief(
        params,
        source.spec,
        runtime_spec_from_trajectory(source),
        parameter_belief=parameter_belief,
    )

    template = _post_update_controller_template(belief)

    assert template.params is belief.params
    assert template.parameter_belief.update_count == 1
    assert template.parameter_belief.evidence_count == 4
    assert template.parameter_belief.effective_sample_count == 4.0
    np.testing.assert_array_equal(
        template.parameter_belief.covariance,
        belief.parameter_belief.covariance,
    )


def test_adaptation_worker_crosses_spawn_ipc_as_numerical_payload() -> None:
    telemetry = generate_trajectory(seed=11, duration_s=0.2)
    belief = DynamicsBelief(
        true_parameters(),
        telemetry.spec,
        runtime_spec_from_trajectory(telemetry),
    )

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_adaptation_worker,
    ) as executor:
        update = executor.submit(
            _run_isolated_belief_update,
            belief,
            telemetry,
        ).result(timeout=60.0)

    restored = _belief_from_worker_update(belief, update)
    assert update.process_id != os.getpid()
    assert update.parameter_leaves
    assert all(isinstance(leaf, np.ndarray) for leaf in update.parameter_leaves)
    assert isinstance(update.parameter_belief_payload, dict)
    assert isinstance(update.update_report, dict)
    assert update.update_wall_time_s > 0.0
    assert update.update_cpu_time_s > 0.0
    for restored_leaf, update_leaf in zip(
        jax.tree_util.tree_leaves(restored.params),
        update.parameter_leaves,
        strict=True,
    ):
        np.testing.assert_allclose(restored_leaf, update_leaf)
    assert restored.predictive_error is belief.predictive_error


def test_process_temporal_partition_waits_for_stop_acknowledgement() -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=time.sleep,
        args=(0.1,),
    )
    process.start()
    try:
        assert process.pid is not None
        assert _set_process_suspended(process.pid, suspended=True)
        time.sleep(0.02)
        assert process.is_alive()
        assert _set_process_suspended(process.pid, suspended=False)
        process.join(timeout=5.0)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            _set_process_suspended(process.pid, suspended=False)
            process.terminate()
            process.join(timeout=5.0)


@pytest.mark.crazyflow
def test_pinned_crazyflow_plant_hovers_and_changes_arm_configuration() -> None:
    pytest.importorskip("crazyflow")
    plant = CrazyflowPlant()
    try:
        state = resting_state()
        state[2] = 1.0
        hover = np.full(4, plant.hover_motor_thrust_fraction)
        initial = plant.reset(state, applied_motor_thrust_fraction=hover)
        plant.set_arm_length_ratio(1.2)
        samples = [plant.step(hover) for _ in range(5)]
    finally:
        plant.close()

    assert plant.crazyflow_version == "0.3.2"
    assert plant.arm_length_ratio == pytest.approx(1.2)
    assert samples[-1].time_s == pytest.approx(0.1)
    assert abs(samples[-1].state[2] - initial.state[2]) < 2e-3
    assert np.linalg.norm(samples[-1].state[3:6]) < 0.05
    assert np.all(np.isfinite(samples[-1].state))


@pytest.mark.crazyflow
def test_crazyflow_trajectory_keeps_hidden_configuration_out_of_telemetry() -> None:
    pytest.importorskip("crazyflow")
    trajectory = generate_crazyflow_trajectory(
        seed=5,
        duration_s=0.1,
        arm_length_ratio=1.2,
        source_group="unknown-target",
        configuration_id="unknown-target",
    )

    assert trajectory.states.shape == (6, 13)
    assert trajectory.controls.shape == (5, 4)
    assert trajectory.observations.shape == (6, 4)
    assert trajectory.spec.vehicle.configuration_id == "unknown-target"
    assert trajectory.spec.vehicle.fixed_states == {}
    assert "arm_length_ratio" not in trajectory.labels
    assert trajectory.spec.observation_roles == (
        "applied_front_left_motor_thrust_fraction",
        "applied_front_right_motor_thrust_fraction",
        "applied_rear_right_motor_thrust_fraction",
        "applied_rear_left_motor_thrust_fraction",
    )
    assert np.all(np.isfinite(trajectory.states))
    assert np.min(trajectory.controls) >= 0.0
    assert np.max(trajectory.controls) <= 1.0


@pytest.mark.crazyflow
def test_integrated_supervisor_fault_campaign_keeps_hidden_plant_bounded() -> None:
    pytest.importorskip("crazyflow")
    trajectory = generate_trajectory(seed=13, duration_s=0.2)
    controller = NMPCController(
        DynamicsBelief(
            true_parameters(),
            trajectory.spec,
            runtime_spec_from_trajectory(trajectory),
        ),
        _policy=_crazyflow_solver_policy(),
    )

    report = _simulate_supervisor_fault_campaign(controller)

    assert report["case_count"] == len(_SUPERVISOR_FAULT_EXPECTATIONS)
    assert report["fault_count"] == len(SupervisorReason)
    assert report["all_typed_reasons_covered"]
    assert report["all_cases_passed"]
    assert report["all_supervised_commands_finite"]
    assert report["all_supervised_commands_bounded"]
    assert report["all_true_plant_steps_finite"]


@pytest.mark.crazyflow
def test_fast_crazyflow_contract_report_remains_non_claiming() -> None:
    pytest.importorskip("crazyflow")
    report, baseline, modified = run_crazyflow_prototype(duration_s=0.1)

    assert report["artifact_type"] == "glassbox_crazyflow_hidden_plant_prototype"
    assert report["semantics"]["diagnostic_only"]
    assert not report["semantics"]["flight_safety_claim"]
    assert not report["semantics"]["throw_to_recover_claim"]
    assert not report["semantics"]["hidden_physical_parameters_supplied_to_glassbox"]
    assert report["observations"]["all_telemetry_finite"]
    assert report["observations"]["all_commands_bounded"]
    assert baseline.spec == modified.spec


# Recorded-tier policy for docs/results/crazyflow-bootstrap-results.json.
_BOOTSTRAP_TOLERANCES = {
    # The arrest trial is a ~100-step closed loop: short enough that its
    # metrics track the recorded run closely, long enough that last-bit
    # differences are visible, so pin them at 1e-4 relative.
    "*": 1e-4,
}
_BOOTSTRAP_EXACT = (
    # Literal configuration constants; nothing in the loop can move them.
    "configuration.hidden_arm_length_ratio",
    "configuration.excitation_maximum_fraction_of_command_span",
    "configuration.provisional_identifier.*",
    "configuration.identifier.*",
)
_BOOTSTRAP_IGNORE = (
    # Wall clock: a property of the host, not of the method.
    "timing.*",
    "*wall_time_s",
    # The conjunction of every observation, including threshold flags that
    # sit close enough to their bound to flip on last-bit differences.
    "observations.gate_passed",
)


@pytest.mark.crazyflow
def test_no_prior_bootstrap_identifies_hover_and_arrests_rates() -> None:
    pytest.importorskip("crazyflow")

    report = run_crazyflow_bootstrap_benchmark()

    # Contract tier: the claims the concepts page makes about this run.
    assert report["semantics"]["airframe_parameter_prior_used"] is False
    assert report["semantics"]["canonical_motor_mixer_supplied_to_identifier"] is False
    assert report["semantics"]["hover_command_supplied_to_identifier"] is False
    assert report["identification"]["command_evidence_rank"] == 4
    assert report["identification"]["angular_effect_rank"] == 3
    assert report["identification"]["ready"]
    assert report["provisional_identification"]["ready"] is False
    arrest = report["velocity_attitude_rate_arrest"]
    assert arrest["velocity_reduction_ratio"] < 0.20
    assert arrest["rate_reduction_ratio"] < 0.20
    assert arrest["terminal_tilt_rad"] < 0.05
    assert arrest["commands_finite"]
    assert arrest["commands_bounded"]
    assert report["observations"]["all_recovery_values_finite"]
    assert report["observations"]["all_recovery_commands_bounded"]
    # ``gate_passed`` is the conjunction of every observation, so it inherits
    # the sensitivity of the loosest one: recorded, never pinned.
    assert isinstance(report["observations"]["gate_passed"], bool)

    # Recorded tier.
    assert_recorded_close(
        report,
        recorded_result("crazyflow-bootstrap-results.json"),
        tolerances=_BOOTSTRAP_TOLERANCES,
        exact=_BOOTSTRAP_EXACT,
        ignore=_BOOTSTRAP_IGNORE,
    )


@pytest.mark.crazyflow
def test_no_prior_bootstrap_trace_is_state_aligned_for_animation() -> None:
    pytest.importorskip("crazyflow")

    run = run_crazyflow_bootstrap_trial()
    trace = run.trace
    moments = _storyboard(trace, CrazyflowAnimationConfig())

    assert trace.evidence_states.shape == (29, 13)
    assert trace.evidence_applied_motor_commands.shape == (29, 4)
    assert trace.recovery_states.shape == (101, 13)
    assert trace.recovery_applied_motor_commands.shape == (101, 4)
    assert trace.provisional_interval_count == 24
    assert moments[0].phase == "evidence"
    assert moments[-1].phase == "recovery"
    assert "STABILIZED" in moments[-1].status


# Recorded-tier policy for docs/results/crazyflow-throw-results.json.
_THROW_TOLERANCES = {
    # 900 closed-loop steps at 10 ms: the run is deterministic but amplifies
    # last-bit differences, so continuous metrics get 1e-3 relative.
    "*": 1e-3,
}
_THROW_EXACT = (
    # The release state and every solver, controller and identifier setting
    # are literal constants, or offline float64 functions of them.
    "configuration.*",
)
_THROW_IGNORE = (
    # Wall clock: a property of the host, not of the method.
    "*wall_time_s",
    # Everything below is downstream of which interval the candidate belief
    # happened to freeze on, which moves with last-bit trajectory noise.
    "timing.first_supported_control_time_s",
    "timing.certified_belief_time_s",
    "timing.time_from_enable_to_first_supported_control_s",
    "timing.time_from_enable_to_certified_belief_s",
    "identification.validated_predictive_belief.*",
    "identification.terminal_working_belief.*",
    "identification.initial_admission_validation.candidate_interval_count",
    "identification.initial_admission_validation.*rmse*",
    "identification.initial_admission_validation.*improvement*",
    "identification.initial_admission_validation.model_movement_fraction",
    "identification.last_validation.*",
    "identification.validation_history",
    "identification.accepted_update_count",
    "identification.rejected_update_count",
    "identification.accepted_replacement_count",
    "identification.pending_proposal_at_trial_end",
    # Counts and extrema taken over the same chaotic 900-step loop.
    "command_objective.nonzero_information_action_count",
    "command_objective.maximum_estimated_information_gain",
    "command_objective.minimum_objective_value",
    "command_objective.maximum_objective_value",
    # The replacement criterion flips on last-bit differences (see the
    # experiment page), and the gate is the conjunction that contains it.
    "observations.at_least_one_belief_replacement_committed",
    "observations.gate_passed",
)


@pytest.mark.crazyflow
def test_continuous_throw_fits_and_arrests_without_a_post_release_reset() -> None:
    pytest.importorskip("crazyflow")

    run = run_crazyflow_throw_trial()
    report = run.report
    trace = run.trace

    # Contract tier: the claims the experiment page makes about this run.
    assert report["semantics"]["motors_cold_at_release"]
    assert report["semantics"]["continuous_after_release"]
    assert report["semantics"]["simulator_reset_after_release"] is False
    assert report["semantics"]["physical_hand_contact_modeled"] is False
    assert report["semantics"]["flight_safety_claim"] is False
    assert report["semantics"]["continuous_identification_during_control"]
    assert report["semantics"]["separate_evidence_collection_phase"] is False
    assert report["semantics"][
        "initial_control_belief_uses_disjoint_predictive_validation"
    ]
    assert report["semantics"]["post_admission_candidate_replacement_implemented"]
    assert report["semantics"]["information_term_targets_weak_information_directions"]
    assert report["semantics"]["single_belief_space_command_objective"]
    assert report["semantics"]["controller_tier_count"] == 1
    assert report["semantics"]["fallback_controller_implemented"] is False
    assert report["semantics"]["safety_net_controller_implemented"] is False
    observations = report["observations"]
    configuration = report["configuration"]
    enable_index = configuration["model_enable_delay_step_count"]
    online_step_count = configuration["online_step_count"]
    sample_count = enable_index + online_step_count + 1
    # Every flight-quality observation the experiment page claims.
    assert observations["pre_enable_commands_exactly_zero"]
    assert observations["no_reset_after_release"]
    assert observations["every_action_used_one_command_objective"]
    assert observations["no_fallback_or_safety_net_controller"]
    assert observations["working_belief_updated_for_every_post_enable_interval"]
    assert observations["initial_admission_scored_on_future_intervals"]
    assert observations["first_supported_action_began_before_model_validation"]
    assert observations["validated_command_evidence_rank_is_four"]
    assert observations["validated_angular_effect_rank_is_three"]
    assert observations["hover_error_below_2_percent"]
    assert observations["terminal_velocity_below_1_percent_of_release"]
    assert observations["terminal_rate_below_2_percent_of_release"]
    assert observations["terminal_speed_below_0_02_m_s"]
    assert observations["terminal_rate_below_0_03_rad_s"]
    assert observations["terminal_tilt_below_0_01_rad"]
    assert observations["sustained_hover_exceeds_3_s"]
    assert observations["minimum_altitude_above_1_m"]
    assert observations["all_values_finite"]
    assert observations["all_commands_bounded"]
    # Sample indices are structural, not fixed: the admission interval moves
    # with last-bit trajectory noise, but its arithmetic does not.
    assert trace.model_enable_sample_index == enable_index
    assert (
        trace.model_enable_sample_index
        < trace.first_supported_control_sample_index
        < trace.certified_belief_sample_index
        < sample_count
    )
    assert trace.states.shape == (sample_count, 13)
    assert trace.applied_motor_commands.shape == (sample_count, 4)
    assert trace.requested_motor_commands.shape == (sample_count - 1, 4)
    assert trace.command_objective_values.shape == (sample_count,)
    assert trace.information_action_fractions.shape == (sample_count,)
    assert trace.estimated_information_gains.shape == (sample_count,)
    np.testing.assert_allclose(trace.applied_motor_commands[0], 0.0, atol=2e-11)
    np.testing.assert_allclose(
        trace.applied_motor_commands[: trace.model_enable_sample_index + 1],
        0.0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        trace.requested_motor_commands[: trace.model_enable_sample_index],
        0.0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        np.diff(trace.timestamps_s),
        configuration["sample_period_s"],
        atol=1e-12,
    )
    identification = report["identification"]
    assert identification["working_update_count"] == online_step_count
    certified = identification["validated_predictive_belief"]
    assert certified["command_evidence_rank"] == 4
    assert certified["angular_effect_rank"] == 3
    assert certified["method"] == "recursive_rank_supported_multirotor_bootstrap_v1"
    assert certified["airframe_parameter_prior_used"] is False
    assert certified["canonical_motor_mixer_assumed"] is False
    initial_validation = identification["initial_admission_validation"]
    assert initial_validation["accepted"]
    assert initial_validation["initial_admission"]
    assert (
        initial_validation["validation_interval_count"]
        == configuration["identifier"]["validation_interval_count"]
    )
    # The candidate is frozen after its own interval count and admitted that
    # many future intervals later, wherever the freeze lands.
    assert (
        trace.certified_belief_sample_index - trace.model_enable_sample_index
        == initial_validation["candidate_interval_count"]
        + initial_validation["validation_interval_count"]
    )
    timing = report["timing"]
    assert timing["certified_belief_time_s"] == pytest.approx(
        trace.certified_belief_sample_index * configuration["sample_period_s"]
    )
    assert timing["first_supported_control_time_s"] == pytest.approx(
        trace.first_supported_control_sample_index * configuration["sample_period_s"]
    )
    assert report["command_objective"]["controller_tier_count"] == 1
    assert report["command_objective"]["fallback_or_safety_net_controller"] is False
    assert report["command_objective"]["post_enable_command_count"] == online_step_count
    continuous_throw = report["continuous_throw"]
    assert continuous_throw["commands_finite"]
    assert continuous_throw["commands_bounded"]
    assert continuous_throw["pre_enable_commands_exactly_zero"]
    assert continuous_throw["terminal_velocity_norm_m_s"] < 0.02
    assert continuous_throw["terminal_angular_rate_norm_rad_s"] < 0.03
    assert continuous_throw["terminal_tilt_rad"] < 0.01
    assert continuous_throw["sustained_hover_duration_s"] > 3.0
    # The experiment page claims the release height is the altitude floor;
    # the simulator reports altitude in float32.
    assert (
        continuous_throw["minimum_altitude_m"]
        >= configuration["release_height_m"] - 1e-6
    )
    # Chaotic derived quantities: reported and recorded, never pinned to a
    # value. Print them so a run still shows what it did.
    chaotic = {
        "first_supported_control_sample_index": (
            trace.first_supported_control_sample_index
        ),
        "certified_belief_sample_index": trace.certified_belief_sample_index,
        "candidate_interval_count": initial_validation["candidate_interval_count"],
        "accepted_update_count": identification["accepted_update_count"],
        "rejected_update_count": identification["rejected_update_count"],
        "accepted_replacement_count": identification["accepted_replacement_count"],
        "nonzero_information_action_count": (
            report["command_objective"]["nonzero_information_action_count"]
        ),
    }
    print("throw chaotic derived quantities:", chaotic)
    assert all(isinstance(value, int) for value in chaotic.values()), chaotic
    assert isinstance(observations["at_least_one_belief_replacement_committed"], bool)
    assert isinstance(observations["gate_passed"], bool)
    moments = _throw_storyboard(trace, CrazyflowAnimationConfig())
    throw_only_moments = _throw_storyboard(
        trace,
        CrazyflowAnimationConfig(),
        throw_only=True,
    )
    assert "UNPOWERED THROW" in moments[0].status
    assert any("LEARNING WHILE ARRESTING" in moment.status for moment in moments)
    assert "LEARNED CONTROL" in moments[-1].status
    assert all("SEPARATE" not in moment.status for moment in moments)
    assert len(moments) == 300
    assert len(throw_only_moments) == 30
    assert throw_only_moments[-1].simulation_time_s == pytest.approx(29.0 / 30.0)
    assert all("SYSTEM OFF" in moment.status for moment in throw_only_moments)
    assert np.all(np.diff([moment.simulation_time_s for moment in moments]) > 0.0)

    # Recorded tier.
    assert_recorded_close(
        report,
        recorded_result("crazyflow-throw-results.json"),
        tolerances=_THROW_TOLERANCES,
        exact=_THROW_EXACT,
        ignore=_THROW_IGNORE,
    )


# Recorded-tier policy for docs/results/crazyflow-throw-campaign-results.json.
_CAMPAIGN_TOLERANCES = {
    # Five instances of the same 900-step closed loop as the canonical throw,
    # so the same 1e-3 relative policy applies to their metrics.
    "*": 1e-3,
}
_CAMPAIGN_EXACT = (
    # The scenario table is the campaign's literal input.
    "cases[*].scenario.*",
)
_CAMPAIGN_IGNORE = (
    # Per-case gates carry the replacement criterion, which flips on last-bit
    # differences, so which cases pass and why is recorded, not pinned.
    "cases[*].gate_passed",
    "cases[*].failed_observations",
    "aggregate.passing_case_count",
    "aggregate.failing_case_count",
    "aggregate.pass_fraction",
    # Admission timing and candidate accounting follow the same chaotic
    # freeze interval as the canonical throw.
    "cases[*].timing.*",
    "cases[*].identification.*",
)


@pytest.mark.crazyflow
def test_continuous_throw_campaign_retains_successes_and_failed_gates() -> None:
    pytest.importorskip("crazyflow")

    report = run_crazyflow_throw_campaign()

    # Contract tier: the campaign is fixed, keeps its failures, and stays
    # bounded; which cases pass is a recorded outcome, not a claim.
    assert report["case_count"] == len(CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS) == 5
    assert report["semantics"]["fixed_scenarios"]
    assert report["semantics"]["held_out_after_controller_tuning"] is False
    assert report["semantics"]["failed_gates_retained"]
    aggregate = report["aggregate"]
    assert aggregate["all_commands_finite_and_bounded"]
    assert all(
        np.isfinite(value) for value in aggregate.values() if isinstance(value, float)
    )
    assert (
        aggregate["passing_case_count"] + aggregate["failing_case_count"]
        == report["case_count"]
    )
    assert aggregate["pass_fraction"] == pytest.approx(
        aggregate["passing_case_count"] / report["case_count"]
    )
    by_name = {case["scenario"]["name"]: case for case in report["cases"]}
    assert set(by_name) == {
        scenario.name for scenario in CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS
    }
    for name, case in by_name.items():
        recovery = case["recovery"]
        assert recovery["commands_finite"], name
        assert recovery["commands_bounded"], name
        assert all(
            np.isfinite(value)
            for value in recovery.values()
            if isinstance(value, float)
        ), name
        assert recovery["terminal_tilt_rad"] < 0.05, name
        assert isinstance(case["gate_passed"], bool), name
        assert all(isinstance(item, str) for item in case["failed_observations"]), name
        # A gate is exactly its list of failed observations.
        assert case["gate_passed"] == (case["failed_observations"] == []), name
    print(
        "campaign chaotic derived quantities:",
        {name: case["failed_observations"] for name, case in by_name.items()},
    )

    # Recorded tier.
    assert_recorded_close(
        report,
        recorded_result("crazyflow-throw-campaign-results.json"),
        tolerances=_CAMPAIGN_TOLERANCES,
        exact=_CAMPAIGN_EXACT,
        ignore=_CAMPAIGN_IGNORE,
    )


@pytest.mark.crazyflow
def test_throw_study_reports_both_control_models_for_the_canonical_case() -> None:
    pytest.importorskip("crazyflow")

    canonical = next(
        case for case in CRAZYFLOW_THROW_STUDY_CASES if case.name == "canonical"
    )
    report = run_crazyflow_throw_study((canonical,))

    assert report["artifact_type"] == "glassbox_crazyflow_throw_control_model_study"
    assert report["control_models"] == ["certified", "working"]
    assert report["case_count"] == 1
    modes = report["cases"][0]["modes"]
    assert set(modes) == {"certified", "working"}
    for name, metrics in modes.items():
        assert metrics["control_model"] == name
        assert metrics["flight"]["non_finite_value_count"] == 0, name
        assert metrics["flight"]["command_bound_violation_count"] == 0, name
        assert metrics["readiness"]["control_model_time_s"] is not None, name
        assert metrics["identification"]["working_interval_count"] == 900, name
    # Certified mode flies a snapshot the transaction admitted; working mode
    # flies the working belief from the moment its own support holds, which is
    # earlier, and records what the transaction would have done instead.
    certified = modes["certified"]
    working = modes["working"]
    assert (
        working["readiness"]["control_model_time_s"]
        == working["readiness"]["readiness_time_s"]
    )
    assert (
        working["readiness"]["control_model_time_s"]
        < certified["readiness"]["control_model_time_s"]
    )
    assert certified["identification"]["certified_interval_count"] is not None
    assert "shadow_rejection_reasons" in working["identification"]
    assert working["identification"]["shadow_accepted_update_count"] >= 1
    difference = report["cases"][0]["difference_working_minus_certified"]
    assert set(difference) == {
        "flight",
        "readiness",
        "stability",
        "hover_estimate",
        "flown_model_error",
    }
    assert difference["flight"]["terminal_speed_m_s"] == pytest.approx(
        working["flight"]["terminal_speed_m_s"]
        - certified["flight"]["terminal_speed_m_s"]
    )
    table = format_study_table(report)
    assert table.splitlines()[0].startswith("| case")
    assert len(table.splitlines()) == 4
    json.dumps(report, allow_nan=False)


def test_render_crazyflow_throw_trace_accepts_an_uncertified_dual_arm_trace(
    tmp_path,
) -> None:
    """The renderer must not assume every arm certifies or even validates.

    A dual-control arm never certifies a belief at all, so its
    ``CrazyflowStudyTrace`` carries ``certified_belief_sample_index=None`` and
    ``validated=False``.  This builds one by hand — no plant simulation, no
    ffmpeg — and checks the storyboard still produces the three honest
    phases, and that the renderer (with the plant, its rendering, and the
    ffmpeg encoder all mocked out) accepts the trace and names the arm
    correctly, in well under a second.
    """

    sample_count = 40
    sample_period_s = 0.01
    timestamps = np.arange(sample_count, dtype=np.float64) * sample_period_s
    states = np.zeros((sample_count, 13), dtype=np.float64)
    states[:, 6] = 1.0  # identity quaternion; everything else stays at rest.
    applied = np.zeros((sample_count, 4), dtype=np.float64)
    requested = np.zeros((sample_count - 1, 4), dtype=np.float64)
    # Model enable at sample 10 (t=0.10 s); command evidence reaches rank four
    # at sample 25 (t=0.25 s); never certified.
    ranks = np.zeros(sample_count, dtype=np.int64)
    ranks[25:] = 4
    interval_counts = np.zeros(sample_count, dtype=np.int64)
    interval_counts[11:] = np.arange(1, sample_count - 10)

    trace = CrazyflowStudyTrace(
        arm="dual_control_nmpc_pass2b",
        case_name="canonical",
        sample_period_s=sample_period_s,
        model_enable_sample_index=10,
        first_supported_control_sample_index=12,
        command_rank_four_sample_index=25,
        certified_belief_sample_index=None,
        validated=False,
        timestamps_s=timestamps,
        states=states,
        applied_motor_commands=applied,
        requested_motor_commands=requested,
        working_interval_counts=interval_counts,
        command_evidence_ranks=ranks,
    )

    config = CrazyflowAnimationConfig(width=640, height=360, frames_per_second=10)
    moments = _throw_storyboard(trace, config)
    assert moments[0].phase == "unpowered"
    assert any(moment.phase == "learning" for moment in moments)
    assert moments[-1].phase == "learned"
    assert all("SUSTAINED HOVER" not in moment.status for moment in moments)
    assert all("VALIDATED" not in moment.status for moment in moments)

    class _FakePlant:
        def close(self) -> None:
            pass

    class _FakeEncoder:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()

    def _fake_render_plant_frame(*_args, width, height, **_kwargs):
        return np.zeros((height, width, 3), dtype=np.uint8)

    with (
        patch(
            "glassbox.integrations.crazyflow_animation.CrazyflowPlant",
            return_value=_FakePlant(),
        ),
        patch(
            "glassbox.integrations.crazyflow_animation._render_plant_frame",
            side_effect=_fake_render_plant_frame,
        ),
        patch(
            "glassbox.integrations.crazyflow_animation._start_encoder",
            return_value=_FakeEncoder(),
        ),
        patch("glassbox.integrations.crazyflow_animation._finish_encoder"),
    ):
        summary = render_crazyflow_throw_trace(
            trace,
            tmp_path / "dual-control.mp4",
            config=config,
        )

    assert summary["arm"] == "dual_control_nmpc_pass2b"
    assert summary["case_name"] == "canonical"
    assert summary["frame_count"] == len(moments)
    assert summary["throw_only"] is False
