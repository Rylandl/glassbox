from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from glassbox.control.identifier import (
    RecursiveBootstrapBelief,
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
)
from glassbox.core.dynamics import GRAVITY_M_S2, MOTOR_MIXER


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
    belief = identifier.belief

    assert belief.interval_count == len(commands)
    assert belief.command_evidence_rank == 4
    assert belief.angular_effect_rank == 3
    assert identifier.working_belief_supported
    np.testing.assert_allclose(
        belief.collective_acceleration_per_command,
        thrust_effect,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        belief.angular_acceleration_per_command,
        angular_effect,
        atol=1e-8,
    )
    assert belief.to_dict()["airframe_parameter_prior_used"] is False
    assert belief.to_dict()["canonical_motor_mixer_assumed"] is False


def test_recursive_belief_exposes_supported_covariance_and_information() -> None:
    identifier, _, _ = _update_recursive_identifier(_excitation(80))
    belief = identifier.belief

    assert belief.normalized_command_information.shape == (4, 4)
    assert belief.supported_collective_effect_covariance.shape == (4, 4)
    assert belief.supported_angular_effect_covariance.shape == (3, 4, 4)
    assert np.all(np.linalg.eigvalsh(belief.normalized_command_information) >= -1e-10)
    assert np.all(
        np.linalg.eigvalsh(belief.supported_collective_effect_covariance) >= -1e-10
    )
    assert np.all(
        np.linalg.eigvalsh(belief.supported_angular_effect_covariance) >= -1e-10
    )
    assert belief.minimum_supported_information_singular_value > 0.0
    assert 0.0 < belief.information_authority <= 1.0
    assert belief.collective_effect_signal_to_noise > 0.0
    assert np.all(belief.angular_effect_signal_to_noise > 0.0)
    assert belief.to_dict()["effect_covariance_scope"] == "supported_subspace_only"


def test_recursive_authority_tracks_information_not_elapsed_interval_count() -> None:
    generator = np.random.default_rng(3)
    patterns = generator.standard_normal((80, 4))
    config = RecursiveBootstrapConfig(
        minimum_normalized_command_rms=0.0001,
        minimum_information_singular_value=0.01,
        full_authority_information_singular_value=0.5,
    )
    beliefs = []
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
        beliefs.append(identifier.belief)

    weak, strong = beliefs
    assert weak.interval_count == strong.interval_count == 80
    assert weak.command_evidence_rank == strong.command_evidence_rank == 4
    assert weak.information_authority < 0.05
    assert strong.information_authority == pytest.approx(1.0)
    assert weak.collective_authority < strong.collective_authority
    assert np.max(weak.angular_axis_authority) < np.min(strong.angular_axis_authority)


def test_recursive_rank_deficiency_never_claims_unobserved_axes() -> None:
    commands = _excitation(80, collective_only=True)

    identifier, _, _ = _update_recursive_identifier(commands)
    belief = identifier.belief

    assert belief.command_evidence_rank <= 1
    assert belief.angular_effect_rank < 3
    assert not identifier.working_belief_supported
    assert np.max(belief.angular_axis_authority) < 0.26


def test_recursive_lateral_velocity_coefficient_stays_near_zero_under_noise() -> None:
    commands = _excitation(200)

    clean, _, _ = _update_recursive_identifier(commands)
    noisy, _, _ = _update_recursive_identifier(
        commands,
        acceleration_noise_m_s2=0.01,
        angular_noise_rad_s2=0.01,
    )

    assert clean.belief.collective_nuisance_rank == 2
    assert noisy.belief.collective_nuisance_rank == 2
    np.testing.assert_allclose(
        clean.belief.collective_velocity_coefficient,
        (0.0, 0.0, -0.12),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        noisy.belief.collective_velocity_coefficient[:2],
        0.0,
        atol=0.05,
    )
    assert noisy.belief.collective_velocity_coefficient[2] == pytest.approx(
        -0.12,
        abs=0.05,
    )
    assert noisy.belief.to_dict()["collective_nuisance_rank"] == 2
    assert noisy.belief.to_dict()["angular_nuisance_rank"] == 7


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
    assert report.interval_count == before.interval_count
    assert report.to_dict()["accepted"] is False

    healthy = identifier.update(
        np.concatenate((np.zeros(6), (1.0, 0.0, 0.0, 0.0), np.zeros(3))),
        np.concatenate((np.zeros(6), (1.0, 0.0, 0.0, 0.0), np.zeros(3))),
        np.full(4, 0.5),
        0.02,
    )

    assert healthy.interval_count == before.interval_count + 1
    assert identifier.last_sample_report.accepted


def test_recursive_update_accepts_rounding_width_bound_overshoot() -> None:
    identifier = RecursiveBootstrapIdentifier()
    state = np.concatenate((np.zeros(6), (1.0, 0.0, 0.0, 0.0), np.zeros(3)))

    belief = identifier.update(state, state, np.full(4, 1.0 + 1e-7), 0.02)

    assert belief.interval_count == 1
    assert identifier.last_sample_report.accepted

    refused = identifier.update(state, state, np.full(4, 1.01), 0.02)

    assert refused.interval_count == 1
    assert identifier.last_sample_report.reason == "applied_command_outside_bounds"


def test_recursive_belief_exposes_the_accumulated_regression_grams() -> None:
    """The whole evidence, not only the summaries the fits reduce it to.

    A planner whose plan moves the nuisance regressors as well as the commands
    cannot work from the command-block summaries: it needs the Gram in the
    identifier's own feature order.  Exposing it is only honest if it is
    demonstrably the same evidence the reported command information is derived
    from, so the Schur complement is recomputed here through the identifier's
    own nuisance pseudo-inverse rule and compared, and the Loewner ordering the
    complement must satisfy is checked alongside it.
    """

    empty = RecursiveBootstrapIdentifier().belief
    assert empty.collective_information.shape == (8, 8)
    assert empty.angular_information.shape == (11, 11)
    assert np.all(empty.collective_information == 0.0)
    assert np.all(empty.angular_information == 0.0)
    assert not empty.collective_information.flags.writeable
    assert not empty.angular_information.flags.writeable

    identifier, _, _ = _update_recursive_identifier(_excitation(80))
    belief = identifier.belief
    collective = belief.collective_information
    angular = belief.angular_information

    assert np.allclose(collective, collective.T)
    assert np.allclose(angular, angular.T)
    assert np.all(np.linalg.eigvalsh(collective) >= -1e-8)
    assert np.all(np.linalg.eigvalsh(angular) >= -1e-8)
    # Both regressions read the same normalized command off the same samples,
    # so their command blocks are the same accumulated outer products.
    assert np.allclose(collective[:4, :4], angular[:4, :4])
    assert np.trace(angular[:4, :4]) > 0.0

    nuisance_inverse, _ = RecursiveBootstrapIdentifier._nuisance_inverse(
        angular[4:, 4:],
        relative_tolerance=identifier.config.nuisance_rank_relative_tolerance,
    )
    complement = (
        angular[:4, :4] - angular[:4, 4:] @ nuisance_inverse @ angular[:4, 4:].T
    )
    complement = 0.5 * (complement + complement.T)
    assert np.allclose(complement, belief.normalized_command_information, atol=1e-8)
    # Residualizing can only remove information, never add it.
    assert np.all(
        np.linalg.eigvalsh(angular[:4, :4] - belief.normalized_command_information)
        >= -1e-8
    )

    recorded = belief.to_dict()
    assert np.allclose(recorded["collective_information"], collective)
    assert np.allclose(recorded["angular_information"], angular)


def test_transition_aggregation_assimilates_window_means_weighted_by_the_window() -> (
    None
):
    """One sample per window, the window's mean, standing for its transitions.

    At a window of one the identifier is bit-for-bit the identifier as it
    was.  At a window of three, only every third transition changes the
    belief, the interval count still counts transitions, and the accumulated
    Gram is three times the outer product of the window's mean features, so
    the support thresholds and the residual floor stay per transition.
    """

    import numpy as np

    from glassbox.control.identifier import (
        RecursiveBootstrapConfig,
        RecursiveBootstrapIdentifier,
    )

    with pytest.raises(ValueError):
        RecursiveBootstrapConfig(transition_aggregation_steps=0)

    rng = np.random.default_rng(3)
    dt = 0.01

    def transition(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        previous = np.zeros(13)
        previous[6] = 1.0
        previous[3:6] = rng.normal(scale=0.1, size=3)
        previous[10:13] = rng.normal(scale=0.2, size=3)
        current = previous.copy()
        current[3:6] += rng.normal(scale=0.02, size=3)
        current[10:13] += rng.normal(scale=0.05, size=3)
        command = np.clip(0.5 + 0.1 * rng.normal(size=4), 0.0, 1.0)
        return previous, current, command

    transitions = [transition(k) for k in range(9)]
    plain = RecursiveBootstrapIdentifier()
    reference = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(transition_aggregation_steps=1)
    )
    windowed = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(transition_aggregation_steps=3)
    )
    for index, (previous, current, command) in enumerate(transitions):
        a = plain.update(previous, current, command, dt).to_dict()
        b = reference.update(previous, current, command, dt).to_dict()
        a.pop("update_wall_time_s")
        b.pop("update_wall_time_s")
        assert a == b
        before = windowed.belief
        after = windowed.update(previous, current, command, dt)
        if (index + 1) % 3:
            assert after is before
            assert windowed.last_sample_report.reason == "sample_buffered"
        else:
            assert after is not before
            assert after.interval_count == index + 1
    assert windowed.belief.interval_count == 9
    assert plain.belief.interval_count == 9

    # The aggregated Gram is the window length times the outer product of the
    # window's mean features: check the first window directly.
    features = [
        windowed._sample_features(previous, current, command, dt)
        for previous, current, command in transitions[:3]
    ]
    mean_force = np.mean([f.force_features for f in features], axis=0)
    single = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(transition_aggregation_steps=3)
    )
    for previous, current, command in transitions[:3]:
        single.update(previous, current, command, dt)
    assert np.allclose(
        single.belief.collective_information, 3.0 * np.outer(mean_force, mean_force)
    )


def test_the_integrated_collective_fit_is_honest_under_velocity_noise() -> None:
    """The cumulative collective regression is the least-squares form for measurement noise.

    Off, the identifier is bit-for-bit as it was.  On, with white noise on the
    measured velocity, the collective map's coefficient error after a burst of
    transitions is smaller than the per-interval fit's and the collective
    authority is higher, because the integrated target carries the velocity
    noise once per row instead of divided by the interval.  The exported
    collective Gram is the equivalent per-transition one, scaled so that
    dividing it by the reported residual, the declared floor, gives the
    integrated precision.
    """

    import numpy as np

    from glassbox.control.identifier import (
        RecursiveBootstrapConfig,
        RecursiveBootstrapIdentifier,
    )
    from glassbox.core.dynamics import GRAVITY_M_S2

    dt = 0.01
    true_map = np.asarray((4.6, 4.7, 4.5, 4.6))
    intercept = -0.4
    noise = 0.02

    def flight(seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        local = np.random.default_rng(seed)
        velocity = np.zeros(3)
        transitions = []
        for k in range(60):
            command = np.clip(
                0.5 + 0.1 * (1 if (k // 5) % 2 else -1) + 0.05 * local.normal(size=4),
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
        return transitions

    transitions = flight(3)
    plain = RecursiveBootstrapIdentifier()
    off = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(integrated_collective=False)
    )
    on = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(integrated_collective=True)
    )
    first_half_authority: dict[str, int | None] = {"off": None, "on": None}
    for index, (previous, current, command) in enumerate(transitions):
        a = plain.update(previous, current, command, dt).to_dict()
        b = off.update(previous, current, command, dt).to_dict()
        a.pop("update_wall_time_s")
        b.pop("update_wall_time_s")
        assert a == b
        on.update(previous, current, command, dt)
        for name, identifier in (("off", off), ("on", on)):
            if (
                first_half_authority[name] is None
                and identifier.belief.collective_authority >= 0.5
            ):
                first_half_authority[name] = index

    per_step_error = abs(
        np.sum(off.belief.collective_acceleration_per_command) - true_map.sum()
    )
    integrated_error = abs(
        np.sum(on.belief.collective_acceleration_per_command) - true_map.sum()
    )
    assert integrated_error < per_step_error
    # The collective authority reaches half no later under the integrated fit.
    assert first_half_authority["on"] is not None
    assert first_half_authority["off"] is None or (
        first_half_authority["on"] <= first_half_authority["off"]
    )
    assert (
        on.belief.collective_residual_std_m_s2
        == RecursiveBootstrapConfig().collective_residual_std_floor_m_s2
    )
    assert not np.allclose(
        on.belief.collective_information, off.belief.collective_information
    )
    assert np.allclose(on.belief.angular_information, off.belief.angular_information)


# The dual-control controller in glassbox-throw reads this belief field by
# field, so the field list is a downstream contract: a rename or removal has to
# be made here on purpose, and mirrored there.  The count dropped from
# thirty-nine to thirty-three at this commit, when the staged-regressor and
# collective-sign switches and their six bookkeeping fields were deleted.
RECURSIVE_BOOTSTRAP_BELIEF_FIELDS = (
    "interval_count",
    "effective_interval_count",
    "collective_acceleration_per_command",
    "collective_velocity_coefficient",
    "collective_intercept_m_s2",
    "angular_acceleration_per_command",
    "angular_rate_coefficient",
    "angular_rate_product_coefficient",
    "angular_intercept_rad_s2",
    "normalized_command_support_projector",
    "normalized_command_singular_values",
    "normalized_command_information",
    "supported_collective_effect_covariance",
    "supported_angular_effect_covariance",
    "collective_information",
    "angular_information",
    "command_evidence_rank",
    "angular_effect_rank",
    "collective_nuisance_rank",
    "angular_nuisance_rank",
    "angular_output_support_projector",
    "collective_support_fraction",
    "minimum_supported_information_singular_value",
    "information_authority",
    "collective_effect_signal_to_noise",
    "angular_effect_signal_to_noise",
    "collective_residual_std_m_s2",
    "angular_residual_std_rad_s2",
    "exploration_completion",
    "collective_authority",
    "angular_axis_authority",
    "hover_command",
    "update_wall_time_s",
)


def test_recursive_bootstrap_belief_fields_are_a_downstream_contract() -> None:
    names = tuple(field.name for field in fields(RecursiveBootstrapBelief))
    assert names == RECURSIVE_BOOTSTRAP_BELIEF_FIELDS
