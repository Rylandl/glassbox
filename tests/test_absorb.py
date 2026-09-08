"""The recursive information update, and the properties that bound its step."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.information import ParameterInformation, innovation_noise_floor
from glassbox.belief.parameter_evidence import parameter_information
from glassbox.belief.update import one_step_linearization
from glassbox.core.data import Trajectory, trajectory_segment
from glassbox.core.dynamics import (
    rollout_with_latent,
    step_with_latent,
    structured_parameter_names,
    structured_parameter_vector,
    with_response_time_constant,
    with_structured_parameter_vector,
)
from glassbox.core.fixedwing_synthetic import true_fixed_wing_parameters
from glassbox.core.geometry import state_plus_tangent
from glassbox.core.metrics import one_step_innovations
from glassbox.core.model import (
    ExecutableModel,
    ModelValidityEnvelope,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import generate_trajectory, true_parameters
from glassbox.fitting import FitSpec, Holdout, fit

# Sixty-four seeds and a four-sigma band. The step under the null has
# covariance ``P dL P = P - P L P``, which is at most the posterior covariance
# ``P`` and therefore at most the prior covariance, so the mean over ``S``
# independent seeds has standard deviation at most one prior sigma over
# ``sqrt(S)``. Four of those is a two-sided level near 6e-5 per direction, so
# the whole property fails by chance well under once in a thousand runs. The
# constant is derived from that inequality, not from what a run happened to
# produce.
NULL_SEED_COUNT = 64
NULL_STEP_SIGMA_MULTIPLE = 4.0


@pytest.mark.parametrize("family", ["multirotor", "fixedwing"])
def test_one_step_evidence_uses_full_history_and_prefix(
    family,
    quadrotor_flight,
    fixedwing_flight,
) -> None:
    with jax.enable_x64(True):
        template = (
            quadrotor_flight(8, 2.5)
            if family == "multirotor"
            else fixedwing_flight(8, 2.5)
        )
        params = with_response_time_constant(
            true_parameters()
            if family == "multirotor"
            else true_fixed_wing_parameters(),
            2.0,
        )
        controls = np.array(template.controls)
        controls[20:] = controls[20]
        states, _ = rollout_with_latent(
            params,
            jnp.asarray(template.states[0]),
            jnp.asarray(controls),
            template.nominal_dt_s,
            control_roles=template.spec.control_roles,
            exogenous=jnp.asarray(template.exogenous[:-1]),
            exogenous_roles=template.spec.exogenous_roles,
        )
        full = replace(template, controls=controls, states=np.asarray(states))
        segment = trajectory_segment(full, 45, len(controls))
        starts = np.array([0, 50, 75])
        model = ExecutableModel(params, full.spec, _permissive_runtime_spec(full))
        errors, jacobians = one_step_linearization(model, segment, starts)
        full_errors, full_jacobians = one_step_linearization(model, full, starts + 45)
        np.testing.assert_allclose(errors, 0.0, atol=1e-10)
        np.testing.assert_allclose(errors, full_errors, atol=1e-10)
        np.testing.assert_allclose(jacobians, full_jacobians, rtol=1e-10, atol=1e-10)

        center = np.asarray(structured_parameter_vector(params))
        tau_index = next(
            i
            for i, name in enumerate(structured_parameter_names(params))
            if "time_constant" in name
        )
        epsilon = 1e-4
        delta = np.zeros_like(center)
        delta[tau_index] = epsilon
        measured = []
        for vector in (center - delta, center + delta):
            candidate = with_structured_parameter_vector(params, jnp.asarray(vector))
            measured.append(one_step_innovations(candidate, full)[starts + 45])
        derivative = -(measured[1] - measured[0]) / (2.0 * epsilon)
        assert np.linalg.norm(derivative) > 1e-6
        np.testing.assert_allclose(
            jacobians[..., tau_index],
            derivative,
            rtol=1e-5,
            atol=1e-8,
        )


def test_absorption_and_fit_evidence_accept_jax_float64(quadrotor_flight) -> None:
    with jax.enable_x64(True):
        trajectory = quadrotor_flight(19, 0.2)
        belief = _belief(trajectory)
        updated, result = belief.absorb(trajectory)
        assert result.absorbed
        assert updated.information.effective_count == len(trajectory.controls)
        evidence = parameter_information(
            belief.model,
            [trajectory],
            ["flight"],
            innovation_noise=belief.information.innovation_noise,
        )
        np.testing.assert_allclose(
            evidence.precision, updated.information.precision, rtol=1e-8, atol=1e-7
        )


def _permissive_runtime_spec(trajectory: Trajectory):
    return replace(
        runtime_spec_from_trajectory(trajectory),
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(100.0, 100.0, 100.0),
        ),
    )


def _shifted(params, index: int, delta: float):
    vector = np.asarray(structured_parameter_vector(params), dtype=np.float64).copy()
    vector[index] += delta
    return with_structured_parameter_vector(params, jnp.asarray(vector))


def _belief(
    trajectory: Trajectory,
    params=None,
    *,
    precision: np.ndarray | None = None,
    innovation_noise: np.ndarray | None = None,
) -> DynamicsBelief:
    resolved = true_parameters() if params is None else params
    information = ParameterInformation.unknown(
        resolved, innovation_noise=innovation_noise
    )
    if precision is not None:
        information = information.with_precision(precision, effective_count=1.0)
    return DynamicsBelief(
        model=ExecutableModel(
            resolved, trajectory.spec, _permissive_runtime_spec(trajectory)
        ),
        information=information,
    )


def _diagonal_precision(params, entries: dict[int, float]) -> np.ndarray:
    size = len(structured_parameter_names(params))
    precision = np.zeros((size, size))
    for index, value in entries.items():
        precision[index, index] = value
    return precision


def _noisy_transitions(
    params,
    trajectory: Trajectory,
    noise: np.ndarray,
    *,
    seed: int,
) -> Trajectory:
    """Re-simulate the plant with i.i.d. one-step innovation noise of ``noise``.

    Every state after the first is the model's own prediction from the measured
    previous state, displaced along the tangent by a fresh draw. The innovation
    a belief at the true parameters measures on this telemetry is therefore
    exactly that draw, independent across transitions, which is the null the
    step-size property is stated under.
    """

    generator = np.random.default_rng(seed)
    states = np.array(trajectory.states, dtype=np.float64)
    latent = jnp.asarray(trajectory.controls[0])
    deviation = np.sqrt(noise)
    for index in range(len(trajectory.controls)):
        predicted, latent = step_with_latent(
            params,
            jnp.asarray(states[index]),
            latent,
            jnp.asarray(trajectory.controls[index]),
            trajectory.nominal_dt_s,
            trajectory.spec.control_roles,
            jnp.asarray(trajectory.exogenous[index]),
            trajectory.spec.exogenous_roles,
        )
        draw = deviation * generator.standard_normal(12)
        states[index + 1] = np.asarray(state_plus_tangent(predicted, jnp.asarray(draw)))
    return replace(trajectory, states=states)


def test_a_direction_the_evidence_cannot_resolve_receives_no_step(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(31, 0.6)
    params = _shifted(true_parameters(), 0, 0.2)
    belief = _belief(trajectory, params)
    before = np.asarray(structured_parameter_vector(belief.params), dtype=np.float64)

    updated, result = belief.absorb(trajectory)
    step = (
        np.asarray(structured_parameter_vector(updated.params), dtype=np.float64)
        - before
    )

    assert result.absorbed
    # A coordinate the fitter holds fixed is not merely unexcited: the update
    # has no business moving it and moves it by exactly zero, not nearly zero.
    estimable = belief.information.estimable
    np.testing.assert_array_equal(step[~estimable], 0.0)
    # A direction the posterior does not resolve carries exactly no variance,
    # so the pseudo-inverse annihilates it and no score along it can produce a
    # step. That is an algebraic identity on the covariance, checked here at
    # machine precision.
    information = updated.information
    indices, eigenvalues, eigenvectors, threshold = information._normalized_spectrum()
    unresolved = eigenvalues <= threshold
    assert np.any(unresolved)
    basis = np.zeros((len(information.names), int(np.count_nonzero(unresolved))))
    basis[indices] = eigenvectors[:, unresolved]
    normalized_basis = basis / information.scale[:, None]
    covariance = information.covariance()
    np.testing.assert_allclose(
        covariance @ normalized_basis,
        0.0,
        atol=1e-12 * float(np.max(np.abs(covariance))),
    )
    # The realized step follows, to the resolution the float32 parameter block
    # can store: the belief's coefficients are float32, so a step of order one
    # round-trips with about a hundred nanounits of storage noise and nothing
    # smaller than that is observable through the parameters.
    storage = float(
        np.finfo(np.float32).eps
        * np.max(np.abs(structured_parameter_vector(updated.params)))
    )
    projection = normalized_basis.T @ step
    assert np.max(np.abs(projection)) < 10.0 * storage


def test_a_well_resolved_direction_moves_less_than_a_poorly_resolved_one(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(32, 0.6)
    params = _shifted(true_parameters(), 0, 0.15)
    confident = _belief(
        trajectory, params, precision=_diagonal_precision(params, {0: 1e7})
    )
    uncertain = _belief(
        trajectory, params, precision=_diagonal_precision(params, {0: 1e2})
    )
    before = np.asarray(structured_parameter_vector(params), dtype=np.float64)

    confident_updated, _ = confident.absorb(trajectory)
    uncertain_updated, _ = uncertain.absorb(trajectory)
    confident_step = abs(
        float(structured_parameter_vector(confident_updated.params)[0]) - before[0]
    )
    uncertain_step = abs(
        float(structured_parameter_vector(uncertain_updated.params)[0]) - before[0]
    )

    assert (
        confident.information.covariance()[0, 0]
        < (uncertain.information.covariance()[0, 0])
    )
    assert confident_step < uncertain_step


def test_information_accumulates_and_never_decays(quadrotor_flight) -> None:
    first_block = quadrotor_flight(33, 0.6)
    second_block = quadrotor_flight(34, 0.6)
    belief = _belief(first_block, _shifted(true_parameters(), 0, 0.1))

    once, first = belief.absorb(first_block)
    twice, second = once.absorb(second_block)

    for earlier, later in ((belief, once), (once, twice)):
        increment = later.information.precision - earlier.information.precision
        assert np.min(np.linalg.eigvalsh(0.5 * (increment + increment.T))) >= -1e-9
        assert later.information.resolved_rank() >= earlier.information.resolved_rank()
        assert later.information.effective_count > earlier.information.effective_count
    assert first.absorbed and second.absorbed
    assert twice.update_count == 2
    assert twice.parameter_distance_since_measurement > (
        once.parameter_distance_since_measurement
    )


def test_the_step_under_the_null_stays_inside_its_own_prior_sigma(
    quadrotor_flight,
) -> None:
    """The pinned step-size property that replaces the null-acceptance rate.

    At the true parameters, observed through i.i.d. one-step noise of exactly
    the declared innovation covariance, the update has nothing to learn and
    must not drift. It is not asked to accept nothing; it is asked to take a
    step whose size is the one its own posterior claims, which is what makes
    the reported covariance an honest statement rather than a label.
    """

    base = quadrotor_flight(35, 1.0)
    params = true_parameters()
    noise = 25.0 * innovation_noise_floor()
    estimable = ParameterInformation.unknown(params).estimable
    prior = _diagonal_precision(
        params, dict.fromkeys(np.flatnonzero(estimable).tolist(), 1e4)
    )
    reference = _belief(base, params, precision=prior, innovation_noise=noise)
    center = np.asarray(structured_parameter_vector(params), dtype=np.float64)
    directions = reference.information.resolved_subspace()
    assert directions.shape[1] > 0
    prior_sigma = np.sqrt(
        np.maximum(
            np.einsum(
                "ip,ij,jp->p",
                directions,
                reference.information.covariance(),
                directions,
                optimize=True,
            ),
            0.0,
        )
    )
    normalizer = np.einsum("ip,ip->p", directions, directions)

    steps = []
    for seed in range(NULL_SEED_COUNT):
        telemetry = _noisy_transitions(params, base, noise, seed=1000 + seed)
        updated, result = reference.absorb(telemetry)
        assert result.absorbed
        step = (
            np.asarray(structured_parameter_vector(updated.params), dtype=np.float64)
            - center
        )
        steps.append((directions.T @ step) / normalizer)

    mean_step = np.mean(np.asarray(steps), axis=0)
    standardized = np.abs(mean_step) / prior_sigma
    bound = NULL_STEP_SIGMA_MULTIPLE / np.sqrt(NULL_SEED_COUNT)

    assert float(np.max(standardized)) < bound


def test_realized_noise_rises_only_by_what_the_step_could_not_explain(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(36, 0.6)
    params = true_parameters()
    floor = innovation_noise_floor()

    # A small parameter shift is explained almost exactly by the first step, so
    # the error it produced is not evidence of irreducible noise and the floor
    # stands. An empty belief does not get to record its own ignorance as noise.
    explained, explained_result = _belief(trajectory, _shifted(params, 0, 0.01)).absorb(
        trajectory
    )
    np.testing.assert_allclose(explained.information.innovation_noise, floor)

    # Noise the parameters cannot absorb is a different matter: it raises the
    # floor on the coordinates that actually carried it.
    noise = np.concatenate((np.full(6, 1e-8), np.full(3, 1e-4), np.full(3, 1e-6)))
    noisy = _noisy_transitions(params, trajectory, noise, seed=7)
    _, raised_result = _belief(noisy, params).absorb(noisy)
    raised, _ = _belief(noisy, params).absorb(noisy)

    assert explained_result.absorbed and raised_result.absorbed
    assert np.all(raised.information.innovation_noise >= floor)
    assert np.all(raised.information.innovation_noise[6:9] > 10.0 * floor[6:9])
    assert np.max(raised.information.innovation_noise[0:3] / floor[0:3]) < 10.0


def test_absorb_refuses_telemetry_outside_the_validity_envelope(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(37, 0.4)
    belief = _belief(trajectory)
    tight = replace(
        belief.model,
        runtime_spec=replace(
            belief.model.runtime_spec,
            validity_envelope=ModelValidityEnvelope(
                body_velocity_center_m_s=(50.0, 50.0, 50.0),
                body_velocity_half_width_m_s=(1e-6, 1e-6, 1e-6),
                angular_velocity_center_rad_s=(50.0, 50.0, 50.0),
                angular_velocity_half_width_rad_s=(1e-6, 1e-6, 1e-6),
            ),
        ),
    )
    bounded = replace(belief, model=tight)

    unchanged, result = bounded.absorb(trajectory)

    assert not result.absorbed
    assert "validity envelope" in result.reason
    assert result.window_count == 0
    assert result.maximum_validity_utilization > 1.0
    np.testing.assert_array_equal(
        structured_parameter_vector(unchanged.params),
        structured_parameter_vector(bounded.params),
    )


def test_absorb_refuses_non_finite_telemetry(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(38, 0.4)
    belief = _belief(trajectory)
    # A canonical Trajectory cannot be constructed with a non-finite sample,
    # so the guard is reached by writing past the constructor. It exists
    # because telemetry reaches absorb from a live link as well as from a file.
    states = np.array(trajectory.states, dtype=np.float64)
    states[:, 3] = np.nan
    object.__setattr__(trajectory, "states", states)

    _, result = belief.absorb(trajectory)

    assert not result.absorbed
    assert "finite" in result.reason


def test_absorb_refuses_a_telemetry_period_the_model_does_not_run_at(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(39, 0.4)
    belief = _belief(trajectory)
    retimed = replace(trajectory, time_s=2.0 * np.asarray(trajectory.time_s))

    _, result = belief.absorb(retimed)

    assert not result.absorbed
    assert "sample period" in result.reason


def test_absorb_does_not_require_actionable_control_semantics(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(40, 0.4)
    physical_spec = replace(
        trajectory.spec,
        channels=tuple(
            replace(
                channel,
                semantic="squared_rotor_speed_ratio",
                minimum=None,
                maximum=None,
            )
            for channel in trajectory.spec.channels
        ),
    )
    telemetry = replace(trajectory, spec=physical_spec)
    belief = _belief(telemetry, _shifted(true_parameters(), 0, 0.1))

    assert belief.model.actuation is None
    _, result = belief.absorb(telemetry)

    assert result.absorbed


def test_fit_then_absorb_recovers_a_changed_configuration(tmp_path) -> None:
    """The round trip the product claims: fit a vehicle, then fly a different one."""

    from glassbox.core.data import save_trajectory_npz

    params = true_parameters()
    paths = []
    for index, seed in enumerate((51, 52, 53)):
        path = tmp_path / f"flight{index}.npz"
        save_trajectory_npz(
            generate_trajectory(seed=seed, duration_s=4.0, dt_s=0.02, params=params),
            path,
        )
        paths.append(path)
    outcome = fit(
        paths,
        FitSpec(
            horizons_s=(0.1,),
            evaluation_horizons_s=(0.1,),
            steps=120,
            holdout=Holdout.by_group(1),
        ),
    )
    belief = outcome.belief
    changed = _shifted(_shifted(belief.params, 2, -0.2), 3, -0.2)
    telemetry = generate_trajectory(seed=61, duration_s=2.0, dt_s=0.02, params=changed)

    updated, result = belief.absorb(telemetry)

    before = one_step_innovations(belief.params, telemetry)
    after = one_step_innovations(updated.params, telemetry)
    assert result.absorbed
    # The fit hands back a belief whose training evidence already resolved
    # some directions. Absorbing telemetry from the changed vehicle adds to
    # that evidence rather than replacing it: the effective count grows and no
    # resolved direction is given back.
    assert belief.information.resolved_rank() > 0
    assert updated.information.resolved_rank() >= belief.information.resolved_rank()
    assert updated.information.effective_count > belief.information.effective_count
    assert result.innovation_rms_after < result.innovation_rms_before
    assert np.sqrt(np.mean(np.square(after))) < np.sqrt(np.mean(np.square(before)))


def test_absorb_is_a_pure_function_of_the_belief_it_is_given(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(41, 0.4)
    belief = _belief(trajectory, _shifted(true_parameters(), 0, 0.1))
    original = np.asarray(structured_parameter_vector(belief.params), dtype=np.float64)
    original_precision = np.array(belief.information.precision, copy=True)

    first, _ = belief.absorb(trajectory)
    second, _ = belief.absorb(trajectory)

    np.testing.assert_array_equal(structured_parameter_vector(belief.params), original)
    np.testing.assert_array_equal(belief.information.precision, original_precision)
    np.testing.assert_array_equal(
        structured_parameter_vector(first.params),
        structured_parameter_vector(second.params),
    )
    assert jax.tree_util.tree_structure(first.params) == (
        jax.tree_util.tree_structure(belief.params)
    )
