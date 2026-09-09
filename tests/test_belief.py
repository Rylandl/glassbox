from __future__ import annotations

import json
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.forecast_error import (
    EmpiricalErrorSample,
    ForecastErrorEnvelope,
)
from glassbox.belief.information import (
    ParameterInformation,
    estimable_structured_parameters,
)
from glassbox.belief.parameter_evidence import innovation_noise, parameter_information
from glassbox.core.data import Channel
from glassbox.core.dynamics import (
    ResidualDynamicsParams,
    initial_residual_parameters,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.fixedwing_synthetic import true_fixed_wing_parameters
from glassbox.core.metrics import one_step_innovations
from glassbox.core.model import (
    ExecutableModel,
    ModelValidityEnvelope,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import true_parameters


def _envelope(value: float = 0.0) -> ForecastErrorEnvelope:
    first = np.full((3, 12), value, dtype=np.float64)
    second = np.full((2, 12), 2.0 * value, dtype=np.float64)
    return ForecastErrorEnvelope.from_samples(
        {
            0.1: (
                EmpiricalErrorSample(first, "group-a", "flight-a"),
                EmpiricalErrorSample(second, "group-b", "flight-b"),
            ),
            0.2: (
                EmpiricalErrorSample(first, "group-a", "flight-a"),
                EmpiricalErrorSample(second, "group-b", "flight-b"),
            ),
        }
    )


def _nonsingular_envelope(scale: float = 0.02) -> ForecastErrorEnvelope:
    errors = scale * np.concatenate((np.eye(12), -np.eye(12)), axis=0)
    samples = (
        EmpiricalErrorSample(errors, "group-a", "flight-a"),
        EmpiricalErrorSample(errors, "group-b", "flight-b"),
    )
    return ForecastErrorEnvelope.from_samples({0.1: samples, 0.2: samples})


def _member_information(params, *, spread: float = 0.2) -> ParameterInformation:
    center = np.asarray(structured_parameter_vector(params))
    positive = center.copy()
    negative = center.copy()
    positive[0] += spread
    negative[0] -= spread
    return ParameterInformation.seeded_from_members(
        params,
        (
            with_structured_parameter_vector(params, jnp.asarray(positive)),
            with_structured_parameter_vector(params, jnp.asarray(negative)),
        ),
        source="independent_vehicle_members",
    )


def test_forecast_envelope_balances_complete_source_groups() -> None:
    envelope = _envelope(1.0)

    assert envelope.raw_sample_count == (5, 5)
    assert envelope.effective_sample_count[0] == pytest.approx(4.8)
    assert envelope.independent_group_count == (2, 2)
    assert np.min(np.linalg.eigvalsh(envelope.tangent_covariance[0])) >= -1e-10
    # The envelope is the uncentered second moment, because nothing subtracts
    # the mean error from a runtime forecast: half the samples err by 1 and
    # half by 2, so the group-balanced second moment is 2.5 rather than the
    # 0.25 a covariance about the mean of 1.5 would report.
    np.testing.assert_allclose(envelope.tangent_covariance[0], 2.5)
    np.testing.assert_allclose(
        envelope.covariance_at(0.05),
        0.5 * envelope.tangent_covariance[0],
        atol=1e-7,
    )


def test_structured_parameter_block_is_generic_and_leaves_residual_fixed() -> None:
    residual = initial_residual_parameters(true_fixed_wing_parameters(), hidden_units=3)
    vector = np.asarray(structured_parameter_vector(residual))
    changed = vector.copy()
    changed[0] += 0.1

    updated = with_structured_parameter_vector(residual, jnp.asarray(changed))

    assert isinstance(updated, ResidualDynamicsParams)
    np.testing.assert_allclose(structured_parameter_vector(updated), changed)
    np.testing.assert_allclose(updated.hidden_weights, residual.hidden_weights)
    np.testing.assert_allclose(updated.output_weights, residual.output_weights)


def test_belief_round_trip_and_runtime_forecast(tmp_path, quadrotor_flight) -> None:
    trajectory = quadrotor_flight(4, 0.3)
    belief = DynamicsBelief(
        model=ExecutableModel(
            true_parameters(), trajectory.spec, runtime_spec_from_trajectory(trajectory)
        ),
        forecast_error=_envelope(0.01),
        provenance={"fixture": True},
    )
    path = tmp_path / "belief.json"

    belief.save(path)
    restored = DynamicsBelief.load(path)
    commands = jnp.asarray(trajectory.controls[:5])
    forecast = restored.rollout(jnp.asarray(trajectory.states[0]), commands)
    nominal_from_legacy_loader = ExecutableModel.load(path)

    assert restored.provenance == {"fixture": True}
    assert restored.forecast_error_available
    assert restored.maximum_error_horizon_s == pytest.approx(0.2)
    assert restored.support == restored.runtime_spec.validity_envelope
    assert forecast.uncertainty_available
    assert forecast.forecast_error_horizon_supported
    assert forecast.states.shape == (6, 13)
    assert forecast.forecast_error_covariance.shape == (6, 12, 12)
    assert forecast.validity_utilization.shape == (6, 6)
    # A point belief has empirical errors but no resolved parameter information.
    assert forecast.parameter_information_rank == 0
    np.testing.assert_array_equal(
        forecast.parameter_covariance,
        np.zeros_like(forecast.parameter_covariance),
    )
    assert nominal_from_legacy_loader.command_size == restored.model.command_size


def test_runtime_rollout_enforces_declared_command_bounds(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(5, 0.3)
    belief = DynamicsBelief(
        model=ExecutableModel(
            true_parameters(), trajectory.spec, runtime_spec_from_trajectory(trajectory)
        ),
        forecast_error=_envelope(0.01),
    )
    initial_state = jnp.asarray(trajectory.states[0])
    commands = jnp.asarray(trajectory.controls[:5])
    channel_name = trajectory.spec.control_names[2]

    unbounded = commands.at[3, 2].set(7.5)
    with pytest.raises(ValueError, match=f"{channel_name!r}=7.5 outside"):
        belief.rollout(initial_state, unbounded)
    with pytest.raises(ValueError, match="command history lies outside"):
        belief.rollout(
            initial_state,
            commands,
            command_history=commands.at[0, 2].set(-3.0),
        )

    # A command 1e-9 outside the bound is clipped onto it. The value is carried
    # in float64 because float32 could not represent the violation.
    slack = np.array(trajectory.controls[:5], dtype=np.float64)
    slack[3, 2] = 1.0 + 1e-9
    exact = np.array(trajectory.controls[:5], dtype=np.float64)
    exact[3, 2] = 1.0
    clipped = belief.rollout(initial_state, slack)
    bounded = belief.rollout(initial_state, exact)

    np.testing.assert_array_equal(clipped.states, bounded.states)
    np.testing.assert_array_equal(clipped.commands, bounded.commands)


@dataclass(frozen=True)
class _SquaredActuation:
    command_channels: tuple[Channel, ...]
    model_control_size: int = 4

    def model_control(self, command):
        return jnp.square(command)


@pytest.mark.parametrize("family", ["quadrotor", "fixedwing"])
def test_forecast_continuation_preserves_actuation_and_wind(
    family, quadrotor_flight, fixedwing_flight
) -> None:
    generate, params = (
        (quadrotor_flight, true_parameters())
        if family == "quadrotor"
        else (fixedwing_flight, true_fixed_wing_parameters())
    )
    trajectory = generate(7, 0.3)
    wind = Channel(
        name="wind",
        role="wind_north",
        semantic="world_wind_velocity",
        unit="m/s",
        kind="exogenous",
        frame="NWU",
    )
    model = ExecutableModel(
        params,
        replace(trajectory.spec, channels=(*trajectory.spec.channels, wind)),
        runtime_spec_from_trajectory(trajectory),
        actuation=_SquaredActuation(trajectory.spec.controls),
    )
    belief = DynamicsBelief(model)
    commands = jnp.asarray(trajectory.controls[:6])
    history = jnp.asarray(trajectory.controls[6:10])
    context = jnp.linspace(-3.0, 4.0, len(commands))[:, None]

    full = belief.rollout(
        trajectory.states[0], commands, command_history=history, exogenous=context
    )
    prefix = belief.rollout(
        trajectory.states[0],
        commands[:3],
        command_history=history,
        exogenous=context[:3],
    )
    suffix = belief.rollout(
        prefix.states[-1],
        commands[3:],
        initial_latent_state=prefix.latent_states[-1],
        exogenous=context[3:],
    )

    np.testing.assert_array_equal(
        full.latent_states[0], model.initial_latent_state(history)
    )
    np.testing.assert_allclose(full.states[:4], prefix.states, atol=1e-6)
    np.testing.assert_allclose(full.states[3:], suffix.states, atol=1e-6)
    np.testing.assert_allclose(full.latent_states[3:], suffix.latent_states, atol=1e-6)
    calm = belief.rollout(trajectory.states[0], commands, command_history=history)
    assert np.max(np.abs(np.asarray(full.states - calm.states))) > 1e-5


def test_resolved_information_propagates_through_the_rollout(
    tmp_path, quadrotor_flight
) -> None:
    trajectory = quadrotor_flight(7, 0.3)
    params = true_parameters()
    belief = DynamicsBelief(
        model=ExecutableModel(
            params, trajectory.spec, runtime_spec_from_trajectory(trajectory)
        ),
        information=_member_information(params),
        forecast_error=_nonsingular_envelope(),
    )
    path = tmp_path / "information-belief.json"
    belief.save(path)

    restored = DynamicsBelief.load(path)
    prediction = restored.rollout(
        jnp.asarray(trajectory.states[0]),
        jnp.asarray(trajectory.controls[:5]),
    )

    assert restored.information.resolved_rank() == 1
    np.testing.assert_allclose(
        restored.information.covariance(),
        belief.information.covariance(),
        atol=1e-12,
    )
    assert prediction.parameter_information_rank == 1
    assert np.max(prediction.parameter_covariance) > 0.0
    uninformed = replace(restored, information=None).rollout(
        jnp.asarray(trajectory.states[0]),
        jnp.asarray(trajectory.controls[:5]),
    )
    np.testing.assert_array_equal(prediction.states, uninformed.states)
    np.testing.assert_allclose(
        prediction.forecast_error_covariance,
        uninformed.forecast_error_covariance,
    )
    assert not uninformed.parameter_information_complete
    np.testing.assert_array_equal(uninformed.parameter_covariance, 0.0)


def test_information_leaves_unresolved_directions_at_exactly_zero() -> None:
    params = true_parameters()
    information = ParameterInformation.unknown(params)
    precision = np.zeros_like(information.precision)
    precision[0, 0] = 4.0
    resolved = information.with_precision(precision, effective_count=8.0)

    covariance = resolved.covariance()
    subspace = resolved.resolved_subspace()

    assert resolved.resolved_rank() == 1
    assert subspace.shape == (len(resolved.names), 1)
    assert covariance[0, 0] == pytest.approx(0.25)
    np.testing.assert_array_equal(covariance[1:, :], 0.0)
    np.testing.assert_array_equal(covariance[:, 1:], 0.0)


def test_authority_is_one_on_the_best_direction_and_zero_off_the_subspace() -> None:
    params = true_parameters()
    information = ParameterInformation.unknown(params)
    precision = np.zeros_like(information.precision)
    precision[0, 0] = 4.0
    precision[2, 2] = 1.0
    resolved = information.with_precision(precision, effective_count=8.0)
    size = len(resolved.names)

    best = np.zeros(size)
    best[0] = 1.0
    weaker = np.zeros(size)
    weaker[2] = 1.0
    unresolved = np.zeros(size)
    unresolved[3] = 1.0

    assert resolved.authority(best) == pytest.approx(1.0)
    assert resolved.authority(weaker) == pytest.approx(0.25)
    assert resolved.authority(unresolved) == 0.0
    assert ParameterInformation.unknown(params).authority(best) == 0.0


def test_information_gain_is_reported_only_where_the_belief_already_resolves() -> None:
    params = true_parameters()
    information = ParameterInformation.unknown(params)
    precision = np.zeros_like(information.precision)
    precision[0, 0] = 4.0
    resolved = information.with_precision(precision, effective_count=8.0)
    increment = np.zeros_like(precision)
    increment[0, 0] = 12.0

    # Four to sixteen along the resolved direction is one halving of the
    # standard deviation, which is 0.5 * log(4) nats.
    assert resolved.information_gain_nats(increment) == pytest.approx(0.5 * np.log(4.0))
    # A rank-zero belief has no direction to state a finite gain along; the
    # rank change is what reports its progress.
    assert information.information_gain_nats(increment) == 0.0


def test_training_information_uses_only_estimable_structured_coordinates(
    quadrotor_flight,
) -> None:
    trajectories = tuple(
        replace(quadrotor_flight(seed, 0.3), labels={"source_group": group})
        for seed, group in ((1, "group-a"), (2, "group-b"))
    )
    params = true_parameters()
    model = ExecutableModel(
        params,
        trajectories[0].spec,
        _permissive_runtime_spec(trajectories[0]),
    )
    estimable = estimable_structured_parameters(params, diagonal_angular_control=True)

    information = parameter_information(
        model,
        trajectories,
        ("group-a", "group-b"),
        innovation_noise=innovation_noise(
            one_step_innovations(params, item) for item in trajectories
        ),
        estimable=estimable,
    )

    assert information.estimable_count == 9
    assert 0 < information.resolved_rank() <= information.estimable_count
    assert information.effective_count > 0.0
    assert np.all(np.isfinite(information.precision))
    assert np.allclose(information.precision[~estimable], 0.0)


def test_training_information_is_vehicle_family_generic(fixedwing_flight) -> None:
    trajectories = tuple(fixedwing_flight(seed, 0.3) for seed in (1, 2))
    params = true_fixed_wing_parameters()
    model = ExecutableModel(
        params,
        trajectories[0].spec,
        _permissive_runtime_spec(trajectories[0]),
    )
    fixed_response = estimable_structured_parameters(params, fixed_response_time=True)

    information = parameter_information(
        model,
        trajectories,
        ("fixedwing-a", "fixedwing-b"),
        innovation_noise=innovation_noise(
            one_step_innovations(params, item) for item in trajectories
        ),
    )

    assert information.estimable_count == len(structured_parameter_names(params))
    assert 0 < information.resolved_rank() <= information.estimable_count
    assert np.count_nonzero(fixed_response) == information.estimable_count - 1


def test_information_coerces_numpy_scalar_tolerance_to_json_native_float() -> None:
    information = replace(
        ParameterInformation.unknown(true_parameters()),
        rank_relative_tolerance=np.float32(1e-5),
    )

    assert type(information.rank_relative_tolerance) is float
    payload = json.loads(json.dumps(information.to_dict()))
    assert payload["rank_relative_tolerance"] == pytest.approx(1e-5)


def _permissive_runtime_spec(trajectory):
    return replace(
        runtime_spec_from_trajectory(trajectory),
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(100.0, 100.0, 100.0),
        ),
    )


def test_evidence_dataclasses_own_immutable_array_inputs() -> None:
    errors = np.zeros((3, 12))
    errors[0, 0] = 0.1
    sample = EmpiricalErrorSample(errors, "group-a", "flight-a")
    errors[0, 0] = 99.0

    assert sample.errors[0, 0] == 0.1
    assert not sample.errors.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        sample.errors[0, 0] = 1.0

    covariance = 0.01 * np.eye(12)[None, :, :]
    envelope = ForecastErrorEnvelope(
        horizons_s=(0.1,),
        tangent_covariance=covariance,
        raw_sample_count=(4,),
        effective_sample_count=(4.0,),
        independent_group_count=(2,),
    )
    covariance[0, 0, 0] = 99.0

    assert envelope.tangent_covariance[0, 0, 0] == pytest.approx(0.01)
    assert not envelope.tangent_covariance.flags.writeable

    params = true_parameters()
    names = structured_parameter_names(params)
    precision = np.zeros((len(names), len(names)))
    precision[0, 0] = 4.0
    scale = np.ones(len(names))
    mask = np.ones(len(names), dtype=bool)
    noise = np.full(12, 1e-4)
    information = ParameterInformation(
        names=names,
        precision=precision,
        scale=scale,
        estimable=mask,
        innovation_noise=noise,
        noise_floor=np.full(12, 1e-8),
        effective_count=8.0,
    )
    precision[0, 0] = 99.0
    scale[0] = 99.0
    mask[1] = False
    noise[0] = 99.0

    assert information.precision[0, 0] == pytest.approx(4.0)
    assert information.scale[0] != 99.0
    assert bool(information.estimable[1])
    assert information.innovation_noise[0] == pytest.approx(1e-4)
    for array in (
        information.precision,
        information.scale,
        information.estimable,
        information.innovation_noise,
        information.noise_floor,
    ):
        assert not array.flags.writeable


def test_information_refuses_precision_outside_the_estimable_mask() -> None:
    params = true_parameters()
    names = structured_parameter_names(params)
    precision = np.zeros((len(names), len(names)))
    precision[1, 1] = 4.0
    mask = np.ones(len(names), dtype=bool)
    mask[1] = False

    with pytest.raises(ValueError, match="outside the estimable mask"):
        ParameterInformation(
            names=names,
            precision=precision,
            scale=np.ones(len(names)),
            estimable=mask,
            innovation_noise=np.full(12, 1e-4),
            noise_floor=np.full(12, 1e-8),
            effective_count=1.0,
        )


def test_information_refuses_noise_below_its_declared_floor() -> None:
    params = true_parameters()
    names = structured_parameter_names(params)

    with pytest.raises(ValueError, match="below its declared floor"):
        ParameterInformation(
            names=names,
            precision=np.zeros((len(names), len(names))),
            scale=np.ones(len(names)),
            estimable=np.ones(len(names), dtype=bool),
            innovation_noise=np.full(12, 1e-9),
            noise_floor=np.full(12, 1e-8),
            effective_count=0.0,
        )
