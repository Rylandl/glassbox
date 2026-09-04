from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import plan_model
from glassbox.control.identifier import (
    TRANSITION_AGGREGATION_STEPS,
    BootstrapEvidence,
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
)
from glassbox.control.plan import SafetyEnvelope, SolveStatus, TrackingTolerances
from glassbox.control.solver import BoundedShootingSolver
from glassbox.core.dynamics import (
    GRAVITY_M_S2,
    MOTOR_MIXER,
    BootstrapMultirotorParams,
    structured_parameter_names,
)

LEVEL_STATE = np.concatenate((np.zeros(6), (1.0, 0.0, 0.0, 0.0), np.zeros(3)))


def _excitation(interval_count: int, *, collective_only: bool = False) -> np.ndarray:
    if collective_only:
        phase = np.arange(interval_count, dtype=np.float64)
        collective = 0.5 + 0.1 * np.sin(0.9 * phase + 0.2)
        return np.repeat(collective[:, None], 4, axis=1)
    generator = np.random.default_rng(7)
    return 0.5 + np.clip(
        0.09 * generator.standard_normal((interval_count, 4)),
        -0.16,
        0.16,
    )


def _linear_hidden_plant(
    commands: np.ndarray,
    *,
    sample_period_s: float = 0.02,
    effect_scales: np.ndarray | None = None,
    acceleration_noise_m_s2: float = 0.0,
    angular_noise_rad_s2: float = 0.0,
    noise_seed: int = 3,
    thrust_effect: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thrust_effect = (
        np.asarray((4.8, 5.0, 5.2, 4.9))
        if thrust_effect is None
        else np.asarray(thrust_effect, dtype=np.float64)
    )
    angular_effect = np.diag((34.0, 31.0, 11.0)) @ np.asarray(MOTOR_MIXER)
    angular_rate_coefficient = -np.diag((0.4, 0.5, 0.25))
    thrust_intercept = 0.08
    states = np.zeros((len(commands) + 1, 13), dtype=np.float64)
    states[:, 6] = 1.0
    timestamps = np.arange(len(states), dtype=np.float64) * sample_period_s
    scales = (
        np.ones(len(commands), dtype=np.float64)
        if effect_scales is None
        else np.asarray(effect_scales, dtype=np.float64)
    )
    if scales.shape != (len(commands),):
        raise ValueError("effect_scales must be interval-aligned")
    noise = np.random.default_rng(noise_seed)
    for index, command in enumerate(commands):
        velocity = states[index, 3:6]
        angular_velocity = states[index, 10:13]
        body_specific_force_z = (
            scales[index] * (thrust_effect @ command)
            + thrust_intercept
            - 0.12 * velocity[2]
        )
        acceleration = np.asarray((0.0, 0.0, body_specific_force_z - GRAVITY_M_S2))
        acceleration = acceleration + acceleration_noise_m_s2 * noise.standard_normal(3)
        angular_acceleration = (
            scales[index] * (angular_effect @ command)
            + angular_rate_coefficient @ angular_velocity
            + angular_noise_rad_s2 * noise.standard_normal(3)
        )
        states[index + 1, 0:3] = (
            states[index, 0:3]
            + sample_period_s * velocity
            + 0.5 * sample_period_s**2 * acceleration
        )
        states[index + 1, 3:6] = velocity + sample_period_s * acceleration
        states[index + 1, 10:13] = (
            angular_velocity + sample_period_s * angular_acceleration
        )
    return timestamps, states, thrust_effect, angular_effect


def _update_recursive_identifier(
    commands: np.ndarray,
    config: RecursiveBootstrapConfig | None = None,
    **plant: float,
) -> tuple[RecursiveBootstrapIdentifier, np.ndarray, np.ndarray]:
    timestamps, states, thrust_effect, angular_effect = _linear_hidden_plant(
        commands,
        **plant,
    )
    identifier = RecursiveBootstrapIdentifier(config)
    for index, command in enumerate(commands):
        identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )
    return identifier, thrust_effect, angular_effect


def test_recursive_bootstrap_updates_every_interval_and_recovers_the_map() -> None:
    commands = _excitation(80)

    identifier, thrust_effect, angular_effect = _update_recursive_identifier(commands)
    evidence = identifier.evidence
    params = identifier.belief.model.params

    assert evidence.interval_count == len(commands)
    assert evidence.command_evidence_rank == 4
    assert evidence.angular_effect_rank == 3
    assert identifier.working_belief_supported
    np.testing.assert_allclose(
        params.collective_acceleration_per_command,
        thrust_effect,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        params.angular_acceleration_per_command,
        angular_effect,
        atol=1e-8,
    )
    recorded = evidence.to_dict()
    assert recorded["airframe_parameter_prior_used"] is False
    assert recorded["canonical_motor_mixer_assumed"] is False
    assert recorded["collective_target"] == "integrated"
    assert recorded["transition_aggregation_steps"] == TRANSITION_AGGREGATION_STEPS


def test_recursive_evidence_exposes_supported_covariance_and_information() -> None:
    identifier, _, _ = _update_recursive_identifier(_excitation(80))
    evidence = identifier.evidence

    assert evidence.normalized_command_information.shape == (4, 4)
    assert evidence.supported_collective_effect_covariance.shape == (4, 4)
    assert evidence.supported_angular_effect_covariance.shape == (3, 4, 4)
    assert np.all(np.linalg.eigvalsh(evidence.normalized_command_information) >= -1e-10)
    assert np.all(
        np.linalg.eigvalsh(evidence.supported_collective_effect_covariance) >= -1e-10
    )
    assert np.all(
        np.linalg.eigvalsh(evidence.supported_angular_effect_covariance) >= -1e-10
    )
    assert evidence.minimum_supported_information_singular_value > 0.0
    assert 0.0 < evidence.information_authority <= 1.0
    assert evidence.collective_effect_signal_to_noise > 0.0
    assert np.all(evidence.angular_effect_signal_to_noise > 0.0)
    assert evidence.to_dict()["effect_covariance_scope"] == "supported_subspace_only"


def test_recursive_authority_tracks_information_not_elapsed_interval_count() -> None:
    generator = np.random.default_rng(3)
    patterns = generator.standard_normal((80, 4))
    config = RecursiveBootstrapConfig(
        minimum_normalized_command_rms=0.0001,
        minimum_information_singular_value=0.01,
        full_authority_information_singular_value=0.5,
    )
    summaries = []
    for amplitude in (0.0035, 0.09):
        commands = 0.5 + amplitude * patterns
        timestamps, states, _, _ = _linear_hidden_plant(commands)
        identifier = RecursiveBootstrapIdentifier(config)
        for index, command in enumerate(commands):
            identifier.update(
                states[index],
                states[index + 1],
                command,
                timestamps[index + 1] - timestamps[index],
            )
        summaries.append(identifier.evidence)

    weak, strong = summaries
    assert weak.interval_count == strong.interval_count == 80
    assert weak.command_evidence_rank == strong.command_evidence_rank == 4
    assert weak.information_authority < 0.05
    assert strong.information_authority > 0.8
    assert weak.collective_authority < strong.collective_authority
    assert np.max(weak.angular_axis_authority) < np.min(strong.angular_axis_authority)


def test_recursive_rank_deficiency_never_claims_unobserved_axes() -> None:
    commands = _excitation(80, collective_only=True)

    identifier, _, _ = _update_recursive_identifier(commands)
    evidence = identifier.evidence

    assert evidence.command_evidence_rank <= 1
    assert evidence.angular_effect_rank < 3
    assert not identifier.working_belief_supported
    assert np.max(evidence.angular_axis_authority) < 0.26


def test_recursive_lateral_velocity_coefficient_stays_near_zero_under_noise() -> None:
    commands = _excitation(200)

    clean, _, _ = _update_recursive_identifier(commands)
    noisy, _, _ = _update_recursive_identifier(
        commands,
        acceleration_noise_m_s2=0.01,
        angular_noise_rad_s2=0.01,
    )

    assert clean.evidence.collective_nuisance_rank == 2
    assert noisy.evidence.collective_nuisance_rank == 2
    np.testing.assert_allclose(
        clean.belief.model.params.collective_velocity_coefficient,
        (0.0, 0.0, -0.12),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        noisy.belief.model.params.collective_velocity_coefficient[:2],
        0.0,
        atol=0.05,
    )
    assert noisy.belief.model.params.collective_velocity_coefficient[
        2
    ] == pytest.approx(-0.12, abs=0.05)
    assert noisy.evidence.to_dict()["collective_nuisance_rank"] == 2
    assert noisy.evidence.to_dict()["angular_nuisance_rank"] == 7


def test_recursive_update_refuses_a_non_finite_sample_and_keeps_its_belief() -> None:
    commands = _excitation(60)
    identifier, _, _ = _update_recursive_identifier(commands)
    before = identifier.belief
    corrupted = np.zeros(13, dtype=np.float64)
    corrupted[6] = 1.0
    corrupted[3] = np.nan

    returned = identifier.update(corrupted, np.zeros(13), np.full(4, 0.5), 0.02)

    assert returned is before
    assert identifier.belief is before
    assert identifier.rejected_sample_count == 1
    report = identifier.last_sample_report
    assert not report.accepted
    assert report.reason == "previous_state_not_finite"
    assert report.interval_count == identifier.evidence.interval_count
    assert report.to_dict()["accepted"] is False

    for _ in range(TRANSITION_AGGREGATION_STEPS):
        identifier.update(LEVEL_STATE, LEVEL_STATE, np.full(4, 0.5), 0.02)

    assert identifier.evidence.interval_count == len(commands) + (
        TRANSITION_AGGREGATION_STEPS
    )
    assert identifier.last_sample_report.accepted


def test_recursive_update_accepts_rounding_width_bound_overshoot() -> None:
    identifier = RecursiveBootstrapIdentifier()

    for _ in range(TRANSITION_AGGREGATION_STEPS):
        identifier.update(LEVEL_STATE, LEVEL_STATE, np.full(4, 1.0 + 1e-7), 0.02)

    assert identifier.evidence.interval_count == TRANSITION_AGGREGATION_STEPS
    assert identifier.last_sample_report.accepted

    identifier.update(LEVEL_STATE, LEVEL_STATE, np.full(4, 1.01), 0.02)

    assert identifier.evidence.interval_count == TRANSITION_AGGREGATION_STEPS
    assert identifier.last_sample_report.reason == "applied_command_outside_bounds"


def test_recursive_evidence_exposes_the_accumulated_regression_grams() -> None:
    """The whole evidence, not only the summaries the fits reduce it to.

    A planner whose plan moves the nuisance regressors as well as the commands
    cannot work from the command-block summaries: it needs the Gram in the
    identifier's own feature order.  Exposing it is only honest if it is
    demonstrably the same evidence the reported command information is derived
    from, so the Schur complement is recomputed here through the identifier's
    own nuisance pseudo-inverse rule and compared, and the Loewner ordering the
    complement must satisfy is checked alongside it.
    """

    empty = RecursiveBootstrapIdentifier().evidence
    assert empty.collective_information.shape == (8, 8)
    assert empty.angular_information.shape == (11, 11)
    assert np.all(empty.collective_information == 0.0)
    assert np.all(empty.angular_information == 0.0)
    assert not empty.collective_information.flags.writeable
    assert not empty.angular_information.flags.writeable

    identifier, _, _ = _update_recursive_identifier(_excitation(80))
    evidence = identifier.evidence
    collective = evidence.collective_information
    angular = evidence.angular_information

    assert np.allclose(collective, collective.T)
    assert np.allclose(angular, angular.T)
    assert np.all(np.linalg.eigvalsh(collective) >= -1e-8)
    assert np.all(np.linalg.eigvalsh(angular) >= -1e-8)
    assert np.trace(angular[:4, :4]) > 0.0

    nuisance_inverse, _ = RecursiveBootstrapIdentifier._nuisance_inverse(
        angular[4:, 4:],
        relative_tolerance=identifier.config.nuisance_rank_relative_tolerance,
    )
    complement = (
        angular[:4, :4] - angular[:4, 4:] @ nuisance_inverse @ angular[:4, 4:].T
    )
    complement = 0.5 * (complement + complement.T)
    assert np.allclose(complement, evidence.normalized_command_information, atol=1e-8)
    # Residualizing can only remove information, never add it.
    assert np.all(
        np.linalg.eigvalsh(angular[:4, :4] - evidence.normalized_command_information)
        >= -1e-8
    )

    recorded = evidence.to_dict()
    assert np.allclose(recorded["collective_information"], collective)
    assert np.allclose(recorded["angular_information"], angular)


def test_transitions_are_assimilated_in_windows_of_two() -> None:
    """One sample per window, the window's mean, standing for its transitions.

    Only every second transition changes the belief, the interval count still
    counts transitions, and the accumulated Gram is the window length times the
    outer product of the window's mean features, so the support thresholds and
    the residual floor stay per transition.
    """

    assert TRANSITION_AGGREGATION_STEPS == 2

    rng = np.random.default_rng(3)
    dt = 0.01

    def transition() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        previous = np.zeros(13)
        previous[6] = 1.0
        previous[3:6] = rng.normal(scale=0.1, size=3)
        previous[10:13] = rng.normal(scale=0.2, size=3)
        current = previous.copy()
        current[3:6] += rng.normal(scale=0.02, size=3)
        current[10:13] += rng.normal(scale=0.05, size=3)
        command = np.clip(0.5 + 0.1 * rng.normal(size=4), 0.0, 1.0)
        return previous, current, command

    transitions = [transition() for _ in range(8)]
    identifier = RecursiveBootstrapIdentifier()
    for index, (previous, current, command) in enumerate(transitions):
        before = identifier.belief
        after = identifier.update(previous, current, command, dt)
        if (index + 1) % TRANSITION_AGGREGATION_STEPS:
            assert after is before
            assert identifier.last_sample_report.reason == "sample_buffered"
        else:
            assert after is not before
            assert identifier.evidence.interval_count == index + 1
    assert identifier.evidence.interval_count == len(transitions)
    assert identifier.belief.information.effective_count == float(len(transitions))

    # The aggregated angular Gram over the first window is the window length
    # times the outer product of that window's mean features.
    window = [
        identifier._sample_features(previous, current, command, dt)
        for previous, current, command in transitions[:TRANSITION_AGGREGATION_STEPS]
    ]
    mean_angular = np.mean([entry.angular_features for entry in window], axis=0)
    single = RecursiveBootstrapIdentifier()
    for previous, current, command in transitions[:TRANSITION_AGGREGATION_STEPS]:
        single.update(previous, current, command, dt)
    assert np.allclose(
        single.evidence.angular_information,
        TRANSITION_AGGREGATION_STEPS * np.outer(mean_angular, mean_angular),
    )


def test_the_collective_map_is_fit_on_the_integrated_target() -> None:
    """The cumulative collective regression is the least-squares form for noise.

    With white noise on the measured velocity, the identifier's collective map
    is closer to the truth than the per-interval least-squares fit computed
    here from the same transitions, because the integrated target carries the
    velocity noise once per row instead of divided by the interval.  The
    exported collective Gram is the equivalent per-transition one, scaled so
    that dividing it by the reported residual, which is the declared floor,
    gives the integrated precision; it is therefore not the plain accumulated
    outer products the angular regression uses.
    """

    dt = 0.01
    true_map = np.asarray((4.6, 4.7, 4.5, 4.6))
    intercept = -0.4
    noise = 0.02
    local = np.random.default_rng(3)
    velocity = np.zeros(3)
    transitions = []
    for step in range(60):
        command = np.clip(
            0.5 + 0.1 * (1 if (step // 5) % 2 else -1) + 0.05 * local.normal(size=4),
            0.0,
            1.0,
        )
        force = true_map @ command + intercept
        previous = np.zeros(13)
        previous[6] = 1.0
        previous[3:6] = velocity + noise * local.normal(size=3)
        velocity = velocity + dt * np.asarray((0.0, 0.0, force - GRAVITY_M_S2))
        current = np.zeros(13)
        current[6] = 1.0
        current[3:6] = velocity + noise * local.normal(size=3)
        transitions.append((previous, current, command))

    identifier = RecursiveBootstrapIdentifier()
    rows = []
    targets = []
    for previous, current, command in transitions:
        identifier.update(previous, current, command, dt)
        rows.append(np.concatenate((command, previous[3:6], np.ones(1))))
        targets.append((current[5] - previous[5]) / dt + GRAVITY_M_S2)
    per_interval = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)[0]

    integrated_error = abs(
        np.sum(identifier.belief.model.params.collective_acceleration_per_command)
        - true_map.sum()
    )
    per_interval_error = abs(np.sum(per_interval[:4]) - true_map.sum())
    assert integrated_error < per_interval_error
    assert (
        identifier.evidence.collective_residual_std_m_s2
        == RecursiveBootstrapConfig().collective_residual_std_floor_m_s2
    )
    # The collective Gram is the rescaled integrated one, not the accumulated
    # per-transition outer products the angular Gram is.
    assert identifier.evidence.collective_information[7, 7] != pytest.approx(
        identifier.evidence.angular_information[10, 10]
    )


# The bootstrap parameterization's names, in order, are the coordinate system
# every information state over this family is stated in, and the evidence
# summary is what the dual-control controller in glassbox-throw reads field by
# field.  Both are downstream contracts: a rename, a reorder or a removal has
# to be made here on purpose, and mirrored there.
BOOTSTRAP_PARAMETER_NAMES = (
    *(f"collective_acceleration_per_command[{motor}]" for motor in range(4)),
    *(f"collective_velocity_coefficient[{axis}]" for axis in range(3)),
    "collective_intercept_m_s2",
    *(
        f"angular_acceleration_per_command[{axis},{motor}]"
        for axis in range(3)
        for motor in range(4)
    ),
    *(
        f"angular_rate_coefficient[{axis},{other}]"
        for axis in range(3)
        for other in range(3)
    ),
    *(
        f"angular_rate_product_coefficient[{axis},{pair}]"
        for axis in range(3)
        for pair in range(3)
    ),
    *(f"angular_intercept_rad_s2[{axis}]" for axis in range(3)),
)

BOOTSTRAP_EVIDENCE_FIELDS = (
    "interval_count",
    "collective_information",
    "angular_information",
    "collective_residual_std_m_s2",
    "angular_residual_std_rad_s2",
    "normalized_command_support_projector",
    "normalized_command_singular_values",
    "normalized_command_information",
    "angular_output_support_projector",
    "supported_collective_effect_covariance",
    "supported_angular_effect_covariance",
    "command_evidence_rank",
    "angular_effect_rank",
    "collective_nuisance_rank",
    "angular_nuisance_rank",
    "collective_support_fraction",
    "minimum_supported_information_singular_value",
    "information_authority",
    "collective_effect_signal_to_noise",
    "angular_effect_signal_to_noise",
    "collective_authority",
    "angular_axis_authority",
    "exploration_completion",
    "hover_command",
)


def test_the_bootstrap_family_and_its_evidence_are_downstream_contracts() -> None:
    identifier, _, _ = _update_recursive_identifier(_excitation(20))
    params = identifier.belief.model.params

    assert isinstance(params, BootstrapMultirotorParams)
    assert structured_parameter_names(params) == BOOTSTRAP_PARAMETER_NAMES
    assert len(BOOTSTRAP_PARAMETER_NAMES) == 41
    assert identifier.belief.information.names == BOOTSTRAP_PARAMETER_NAMES
    assert (
        tuple(field.name for field in fields(BootstrapEvidence))
        == BOOTSTRAP_EVIDENCE_FIELDS
    )


def test_the_identifier_produces_a_dynamics_belief_over_the_bootstrap_family() -> None:
    """The identifier's product is the belief a fit produces, not a second type.

    The model is executable over the declared command box, the information is
    the identifier's own Grams stated over the structured parameters, and there
    is no forecast-error envelope, because nothing held any evidence out.
    """

    identifier, thrust_effect, _ = _update_recursive_identifier(_excitation(80))
    belief = identifier.belief

    assert isinstance(belief, DynamicsBelief)
    assert belief.forecast_error is None
    assert belief.maximum_error_horizon_s is None
    assert belief.model.input_spec.vehicle.family == "multirotor_bootstrap"
    assert belief.sample_period_s == identifier.config.sample_period_s
    np.testing.assert_allclose(belief.model.command_minimum, 0.0)
    np.testing.assert_allclose(belief.model.command_maximum, 1.0)

    information = belief.information
    assert information.effective_count == 80.0
    assert information.resolved_rank() > 0
    assert np.all(np.linalg.eigvalsh(information.precision) >= -1e-6)
    covariance = information.covariance()
    assert covariance.shape == (41, 41)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-9)
    assert information.authority(information.resolved_subspace()[:, 0]) > 0.0

    # The model predicts what the identifier's own maps predict.
    hover = identifier.evidence.hover_command
    assert hover is not None
    np.testing.assert_allclose(
        belief.model.params.hover_command(),
        hover,
        rtol=1e-6,
    )
    next_state, next_latent = belief.model.transition(LEVEL_STATE, hover, hover, None)
    np.testing.assert_allclose(next_latent, hover, rtol=1e-6)
    # A hover command holds the vehicle up: the vertical velocity barely moves.
    assert abs(float(next_state[5])) < 1e-6
    assert float(
        np.sum(belief.model.params.collective_acceleration_per_command)
    ) == pytest.approx(float(np.sum(thrust_effect)), rel=1e-6)

    assert belief.provenance["source"] == "recursive_bootstrap_identifier"
    assert belief.provenance["interval_count"] == 80
    assert belief.provenance["bootstrap_evidence"]["command_evidence_rank"] == 4


def test_a_bootstrap_belief_round_trips_through_the_artifact(tmp_path) -> None:
    identifier, _, _ = _update_recursive_identifier(_excitation(40))
    belief = identifier.belief
    path = tmp_path / "bootstrap-belief.json"

    belief.save(path)
    restored = DynamicsBelief.load(path)

    assert isinstance(restored.model.params, BootstrapMultirotorParams)
    for name in ("collective_acceleration_per_command", "angular_rate_coefficient"):
        np.testing.assert_allclose(
            np.asarray(getattr(restored.model.params, name)),
            np.asarray(getattr(belief.model.params, name)),
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        restored.information.precision, belief.information.precision, rtol=1e-12
    )
    assert restored.information.names == belief.information.names
    assert restored.forecast_error is None
    assert restored.model.input_spec.vehicle.family == "multirotor_bootstrap"


def _scripted_plant() -> tuple[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Two hundred transitions of one hidden linear plant, noise on the readback.

    This is the sequence the pinned estimates below were measured on. It moves
    all four motors independently, excites both rate and rate-product terms,
    and puts white noise on the measured velocity and body rate, so every part
    of the estimator is loaded.
    """

    sample_period_s = 0.01
    generator = np.random.default_rng(20260903)
    block = generator.standard_normal((200, 4))
    ramp = np.sin(0.31 * np.arange(200))[:, None] * np.asarray(
        (0.05, -0.04, 0.03, -0.02)
    )
    commands = np.clip(0.5 + 0.08 * block + ramp, 0.0, 1.0)

    thrust_effect = np.asarray((4.8, 5.0, 5.2, 4.9))
    thrust_intercept = 0.08
    velocity_coefficient = np.asarray((0.0, 0.0, -0.12))
    angular_effect = np.diag((34.0, 31.0, 11.0)) @ np.asarray(MOTOR_MIXER)
    angular_rate_coefficient = -np.diag((0.4, 0.5, 0.25))
    angular_rate_product_coefficient = np.asarray(
        ((0.3, -0.2, 0.1), (0.05, 0.4, -0.15), (-0.1, 0.25, 0.2))
    )
    angular_intercept = np.asarray((0.02, -0.03, 0.01))

    noise = np.random.default_rng(20260904)
    states = np.zeros((201, 13), dtype=np.float64)
    states[:, 6] = 1.0
    for index, command in enumerate(commands):
        velocity = states[index, 3:6]
        angular_velocity = states[index, 10:13]
        rate_products = np.asarray(
            (
                angular_velocity[0] * angular_velocity[1],
                angular_velocity[0] * angular_velocity[2],
                angular_velocity[1] * angular_velocity[2],
            )
        )
        specific_force_z = (
            thrust_effect @ command + thrust_intercept + velocity_coefficient @ velocity
        )
        acceleration = np.asarray((0.0, 0.0, specific_force_z - GRAVITY_M_S2))
        angular_acceleration = (
            angular_effect @ command
            + angular_rate_coefficient @ angular_velocity
            + angular_rate_product_coefficient @ rate_products
            + angular_intercept
        )
        states[index + 1, 0:3] = (
            states[index, 0:3]
            + sample_period_s * velocity
            + 0.5 * sample_period_s**2 * acceleration
        )
        states[index + 1, 3:6] = velocity + sample_period_s * acceleration
        states[index + 1, 6:10] = (1.0, 0.0, 0.0, 0.0)
        states[index + 1, 10:13] = (
            angular_velocity + sample_period_s * angular_acceleration
        )
    measured = states.copy()
    measured[:, 3:6] += 0.004 * noise.standard_normal((201, 3))
    measured[:, 10:13] += 0.01 * noise.standard_normal((201, 3))
    return sample_period_s, [
        (measured[index], measured[index + 1], commands[index]) for index in range(200)
    ]


def _scripted_identifier() -> RecursiveBootstrapIdentifier:
    sample_period_s, transitions = _scripted_plant()
    identifier = RecursiveBootstrapIdentifier()
    for previous, current, command in transitions:
        identifier.update(previous, current, command, sample_period_s)
    return identifier


def test_the_scripted_plant_estimates_are_pinned() -> None:
    """What the estimator computes, held fixed across container changes.

    These are the two hundredth interval's estimates: the maps, the intercepts,
    the hover command, both accumulated Grams, the residual scales and every
    rank and authority. The container around them changed at this commit and
    the numbers did not, which is what this pins.
    """

    identifier = _scripted_identifier()
    params = identifier.belief.model.params
    evidence = identifier.evidence

    np.testing.assert_allclose(
        params.collective_acceleration_per_command,
        (4.727309360698963, 5.239587007309757, 5.099312954764219, 4.791217957595088),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        params.collective_velocity_coefficient,
        (0.00012675058623564522, -0.003769984446337915, -0.10551079688093523),
        rtol=1e-12,
    )
    assert float(params.collective_intercept_m_s2) == pytest.approx(
        0.10019540162772955, rel=1e-12
    )
    np.testing.assert_allclose(
        params.angular_acceleration_per_command[0],
        (
            35.765664558133395,
            -31.423062815174827,
            -36.93245865659684,
            34.90298993506994,
        ),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        params.angular_intercept_rad_s2,
        (-1.4191461859189711, -2.4452076997306156, -1.2719091564257448),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        evidence.hover_command,
        np.full(4, 0.4888072589327078),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        evidence.angular_residual_std_rad_s2,
        (0.7633902499641877, 0.7143143806083629, 0.712011432222831),
        rtol=1e-12,
    )
    assert evidence.collective_residual_std_m_s2 == 0.05
    assert evidence.interval_count == 200
    assert identifier.belief.information.effective_count == 200.0
    assert evidence.command_evidence_rank == 4
    assert evidence.angular_effect_rank == 3
    assert evidence.collective_nuisance_rank == 2
    assert evidence.angular_nuisance_rank == 7
    assert evidence.information_authority == pytest.approx(1.0, rel=1e-12)
    assert evidence.collective_authority == pytest.approx(0.9999999999999996, rel=1e-12)
    np.testing.assert_allclose(evidence.angular_axis_authority, 1.0, rtol=1e-12)
    assert evidence.exploration_completion == pytest.approx(1.0, rel=1e-12)
    assert float(np.trace(evidence.collective_information)) == pytest.approx(
        11099.438987179006, rel=1e-12
    )
    assert float(np.trace(evidence.angular_information)) == pytest.approx(
        303.38838980129236, rel=1e-12
    )


def _bootstrap_solver(
    belief: DynamicsBelief,
) -> BoundedShootingSolver:
    plan = plan_model(
        belief,
        TrackingTolerances.for_platform(belief.model.input_spec.vehicle.family),
        SafetyEnvelope(maximum_speed_m_s=8.0, maximum_angular_velocity_rad_s=8.0),
    )
    return BoundedShootingSolver(plan, plan.policy)


def test_a_bootstrap_belief_drives_the_bounded_solver() -> None:
    """One loop, two beliefs: the solver never learns which family it is on.

    The belief the identifier built from nothing goes through the same
    ``plan_model`` a fitted belief does, and the same solver holds a hover
    reference with it. The spread it charges is the tangent covariance of the
    directions its own Grams resolved.
    """

    identifier = _scripted_identifier()
    belief = identifier.belief
    solver = _bootstrap_solver(belief)
    hover = identifier.evidence.hover_command
    assert hover is not None

    result = solver.solve(LEVEL_STATE, solver.hold_reference(LEVEL_STATE), hover)

    assert result.command_usable
    assert result.status in {
        SolveStatus.CONVERGED,
        SolveStatus.ITERATION_LIMIT,
        SolveStatus.STALLED,
    }
    assert np.all(result.command >= np.asarray(belief.model.command_minimum))
    assert np.all(result.command <= np.asarray(belief.model.command_maximum))
    assert np.isfinite(result.diagnostics.final_objective)
    assert result.diagnostics.final_objective <= result.diagnostics.initial_objective
    assert np.all(np.isfinite(result.predicted_states))
    # A belief with resolved directions charges a positive predicted spread.
    assert belief.information.resolved_rank() > 0
    assert (
        result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation
        > (0.0)
    )


def test_a_rank_zero_bootstrap_belief_still_solves() -> None:
    """A fresh identifier's belief plans, with zero spread and a bounded hold.

    Nothing is resolved, so the covariance is exactly zero and the objective is
    the point objective. The zero map makes every command equivalent, so the
    solver returns the previous command rather than inventing one.
    """

    belief = RecursiveBootstrapIdentifier().belief
    assert belief.information.resolved_rank() == 0
    assert np.all(belief.information.covariance() == 0.0)
    assert not belief.uncertainty_available

    solver = _bootstrap_solver(belief)
    previous_command = np.full(4, 0.4)
    result = solver.solve(
        LEVEL_STATE, solver.hold_reference(LEVEL_STATE), previous_command
    )

    assert result.command_usable
    assert np.isfinite(result.diagnostics.final_objective)
    assert (
        result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation
        == 0.0
    )
    np.testing.assert_allclose(result.command, previous_command, atol=1e-9)
    assert np.all(result.command >= 0.0)
    assert np.all(result.command <= 1.0)
