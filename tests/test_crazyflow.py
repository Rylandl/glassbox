from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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

from glassbox.belief import (
    DynamicsBelief,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.dynamics import MOTOR_MIXER
from glassbox.flight_supervisor import SupervisorMode, SupervisorReason
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
)
from glassbox.integrations.crazyflow_bootstrap import (
    run_crazyflow_bootstrap_benchmark,
    run_crazyflow_bootstrap_trial,
)
from glassbox.integrations.crazyflow_prototype import (
    _SUPERVISOR_FAULT_EXPECTATIONS,
    FLEET_LOG_ARM_LENGTH_RATIOS,
    _belief_from_worker_update,
    _configuration_direction_prior,
    _crazyflow_solver_policy,
    _eligible_recovery_evidence_start,
    _initialize_adaptation_worker,
    _online_recovery_trajectory,
    _post_update_controller_template,
    _run_isolated_belief_update,
    _set_process_suspended,
    _simulate_supervisor_fault_campaign,
    generate_crazyflow_trajectory,
    run_crazyflow_prototype,
)
from glassbox.integrations.crazyflow_throw import (
    CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS,
    run_crazyflow_throw_campaign,
    run_crazyflow_throw_trial,
)
from glassbox.nmpc import NMPCController
from glassbox.runtime import runtime_spec_from_trajectory
from glassbox.synthetic import generate_trajectory, resting_state, true_parameters

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


@pytest.mark.crazyflow
def test_no_prior_bootstrap_identifies_hover_and_arrests_rates() -> None:
    pytest.importorskip("crazyflow")

    report = run_crazyflow_bootstrap_benchmark()

    assert report["semantics"]["airframe_parameter_prior_used"] is False
    assert report["semantics"]["canonical_motor_mixer_supplied_to_identifier"] is False
    assert report["semantics"]["hover_command_supplied_to_identifier"] is False
    assert report["identification"]["command_evidence_rank"] == 4
    assert report["identification"]["angular_effect_rank"] == 3
    assert report["identification"]["ready"]
    assert report["observations"]["gate_passed"]
    assert report["velocity_attitude_rate_arrest"]["velocity_reduction_ratio"] < 0.20
    assert report["velocity_attitude_rate_arrest"]["rate_reduction_ratio"] < 0.20
    assert report["velocity_attitude_rate_arrest"]["terminal_tilt_rad"] < 0.05

    recorded = json.loads(
        (
            Path(__file__).parents[1] / "docs/results/crazyflow-bootstrap-results.json"
        ).read_text()
    )
    assert recorded["observations"] == report["observations"]
    assert recorded["evaluation_only"]["estimated_hover_motor_command"] == (
        pytest.approx(
            report["evaluation_only"]["estimated_hover_motor_command"],
            rel=1e-6,
        )
    )
    assert recorded["velocity_attitude_rate_arrest"][
        "velocity_reduction_ratio"
    ] == pytest.approx(
        report["velocity_attitude_rate_arrest"]["velocity_reduction_ratio"],
        rel=1e-6,
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


@pytest.mark.crazyflow
def test_continuous_throw_fits_and_arrests_without_a_post_release_reset() -> None:
    pytest.importorskip("crazyflow")

    run = run_crazyflow_throw_trial()
    report = run.report
    trace = run.trace

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
    # The recorded run passes every flight-quality observation; the
    # post-admission replacement criterion is recorded as failed (see the
    # experiment page), so the gate is pinned to the record rather than asserted.
    failed = [
        name for name, passed in report["observations"].items() if passed is False
    ]
    assert failed == ["at_least_one_belief_replacement_committed", "gate_passed"]
    assert trace.model_enable_sample_index == 100
    assert trace.first_supported_control_sample_index == 103
    assert trace.certified_belief_sample_index == 288
    assert trace.states.shape == (1001, 13)
    assert trace.applied_motor_commands.shape == (1001, 4)
    assert trace.requested_motor_commands.shape == (1000, 4)
    assert trace.command_objective_values.shape == (1001,)
    assert trace.information_action_fractions.shape == (1001,)
    assert trace.estimated_information_gains.shape == (1001,)
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
    np.testing.assert_allclose(np.diff(trace.timestamps_s), 0.01, atol=1e-12)
    assert report["identification"]["working_update_count"] == 900
    certified = report["identification"]["validated_predictive_belief"]
    assert certified["command_evidence_rank"] == 4
    assert certified["angular_effect_rank"] == 3
    initial_validation = report["identification"]["initial_admission_validation"]
    assert initial_validation["candidate_interval_count"] == 172
    assert initial_validation["validation_interval_count"] == 16
    assert initial_validation["accepted"]
    assert report["identification"]["accepted_update_count"] == 1
    assert report["identification"]["accepted_replacement_count"] == 0
    assert report["command_objective"]["controller_tier_count"] == 1
    assert report["command_objective"]["fallback_or_safety_net_controller"] is False
    assert report["command_objective"]["nonzero_information_action_count"] > 0
    assert report["continuous_throw"]["terminal_to_release_velocity_ratio"] < 0.01
    assert report["continuous_throw"]["terminal_to_release_rate_ratio"] < 0.02
    assert report["continuous_throw"]["terminal_tilt_rad"] < 0.01
    assert report["continuous_throw"]["terminal_velocity_norm_m_s"] < 0.02
    assert report["continuous_throw"]["terminal_angular_rate_norm_rad_s"] < 0.03
    assert abs(report["continuous_throw"]["terminal_vertical_velocity_m_s"]) < 0.01
    assert report["continuous_throw"]["sustained_hover_duration_s"] > 5.0
    moments = _throw_storyboard(trace, CrazyflowAnimationConfig())
    throw_only_moments = _throw_storyboard(
        trace,
        CrazyflowAnimationConfig(),
        throw_only=True,
    )
    assert "UNPOWERED THROW" in moments[0].status
    assert any("LEARNING WHILE ARRESTING" in moment.status for moment in moments)
    assert "SUSTAINED HOVER" in moments[-1].status
    assert all("SEPARATE" not in moment.status for moment in moments)
    assert len(moments) == 300
    assert len(throw_only_moments) == 30
    assert throw_only_moments[-1].simulation_time_s == pytest.approx(29.0 / 30.0)
    assert all("SYSTEM OFF" in moment.status for moment in throw_only_moments)
    assert np.all(np.diff([moment.simulation_time_s for moment in moments]) > 0.0)

    recorded = json.loads(
        (Path(__file__).parents[1] / "docs/results/crazyflow-throw-results.json").read_text()
    )
    assert recorded["observations"] == report["observations"]
    assert recorded["continuous_throw"][
        "terminal_to_release_velocity_ratio"
    ] == pytest.approx(
        report["continuous_throw"]["terminal_to_release_velocity_ratio"],
        rel=1e-6,
    )


@pytest.mark.crazyflow
def test_continuous_throw_campaign_retains_successes_and_failed_gates() -> None:
    pytest.importorskip("crazyflow")

    report = run_crazyflow_throw_campaign()

    assert report["case_count"] == len(CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS) == 5
    assert report["semantics"]["held_out_after_controller_tuning"] is False
    assert report["semantics"]["failed_gates_retained"]
    assert report["aggregate"]["passing_case_count"] == 2
    assert report["aggregate"]["failing_case_count"] == 3
    assert report["aggregate"]["pass_fraction"] == pytest.approx(0.4)
    assert report["aggregate"]["all_commands_finite_and_bounded"]
    by_name = {case["scenario"]["name"]: case for case in report["cases"]}
    assert by_name["shorter_arms_high_release"]["gate_passed"]
    assert by_name["milder_low_energy_release"]["gate_passed"]
    assert by_name["canonical"]["failed_observations"] == [
        "at_least_one_belief_replacement_committed"
    ]
    assert by_name["long_arms_cross_axis_tumble"]["failed_observations"] == [
        "at_least_one_belief_replacement_committed",
        "terminal_vertical_speed_below_0_01_m_s",
    ]
    assert by_name["reversed_tumble"]["failed_observations"] == [
        "terminal_vertical_speed_below_0_01_m_s",
        "minimum_altitude_above_1_m",
    ]

    recorded = json.loads(
        (
            Path(__file__).parents[1] / "docs/results/crazyflow-throw-campaign-results.json"
        ).read_text()
    )
    assert recorded["aggregate"] == report["aggregate"]
