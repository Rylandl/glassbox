"""Fixture contracts with synthetic dynamics; no real trims or simulator trials."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.two_simulator_cascade import (
    CONFIGURATION_ID,
    Fixture,
    FixtureSetupError,
    _pack,
    _targets,
    _wind_nwu,
)


@pytest.fixture
def protocol():
    path = Path(__file__).parents[1] / "docs/harness/two-simulator-flight-v1.json"
    return json.loads(path.read_text())


@pytest.fixture
def fake_physics(monkeypatch):
    """Patch physical and controller calls before the fixture captures them."""
    pytest.importorskip("cascade")
    import cascade.analysis
    import cascade.control
    import cascade.integration
    from cascade.state import (
        ActuatorState,
        AeroState,
        AircraftState,
        ControlInput,
        RigidBodyState,
    )

    calls = []

    def trim(model, condition, environment, **kwargs):
        calls.append((condition, environment, kwargs))
        heading = condition.heading_rad
        state = AircraftState(
            RigidBodyState(
                jnp.asarray([0.0, 0.0, -condition.altitude_m]),
                jnp.asarray([0.0, 0.0, np.sin(heading / 2), np.cos(heading / 2)]),
                jnp.asarray(
                    [
                        condition.airspeed_m_s * np.cos(heading),
                        condition.airspeed_m_s * np.sin(heading),
                        0.0,
                    ]
                )
                + environment.wind,
                jnp.zeros(3),
            ),
            ActuatorState(jnp.asarray([0.31, -0.17]), jnp.asarray([123.0])),
            AeroState(jnp.asarray([0.23, 0.81])),
        )
        return SimpleNamespace(
            state=state,
            control=ControlInput(jnp.asarray([0.45]), jnp.asarray([0.04, -0.12])),
            decision=jnp.asarray([0.0, 0.023, 0.0, 0.45, 0.04, -0.12]),
            residual=jnp.zeros(6),
            scaled_residual=jnp.zeros(6),
            success=True,
            optimizer_success=True,
            cost=0.0,
            optimality=0.0,
            evaluations=1,
            message="synthetic test trim",
            angle_of_attack_rad=0.023,
            sideslip_rad=0.0,
        )

    def step(model, state, control, environment, dt):
        native_command = jnp.concatenate((control.propeller, control.channel))
        return state._replace(
            rigid_body=state.rigid_body._replace(
                position=state.rigid_body.position + dt * state.rigid_body.velocity,
                velocity=state.rigid_body.velocity
                + dt * (native_command + environment.wind),
                angular_velocity=state.rigid_body.angular_velocity
                + dt * native_command,
            ),
            actuators=state.actuators._replace(
                surface_deflection=state.actuators.surface_deflection
                + dt * control.channel,
                propeller_speed=state.actuators.propeller_speed
                + dt * control.propeller,
            ),
            aero=state.aero._replace(
                separation=state.aero.separation + dt * jnp.asarray([0.1, -0.1])
            ),
        )

    def pilot(controller, state, setpoint, aircraft, environment, dt):
        assert dt == 0.05
        control = ControlInput(
            controller.guidance.throttle_trim[None], jnp.asarray([0.03, -0.07])
        )
        return control, state._replace(step_index=state.step_index + 1)

    monkeypatch.setattr(cascade.analysis, "trim_straight_flight", trim)
    monkeypatch.setattr(cascade.integration, "rk4_step", step)
    monkeypatch.setattr(cascade.control, "cascade_step", pilot)
    with jax.enable_x64(True):
        yield calls


@pytest.fixture
def fixture(protocol, fake_physics):
    return Fixture(protocol)


def condition(protocol, wind="calm", heading=0):
    cell = copy.deepcopy(protocol["cells"]["cascade"][0])
    cell.update(wind=wind, heading_nwu_deg=heading)
    return cell


def entry(cell, seed=104):
    return {
        "id": f"mock-cascade-{seed}",
        "simulator": "cascade",
        "cell": cell["id"],
        "seed": seed,
        "role": "test",
    }


def identical(left, right):
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()


def test_configuration_and_native_state_layout(fixture):
    config = fixture.config()
    assert config["configuration_id"] == CONFIGURATION_ID
    assert fixture.spec.vehicle.family == "fixedwing"
    assert fixture.spec.vehicle.configuration_id == CONFIGURATION_ID
    assert config["command_names"] == ["throttle", "elevator", "aileron"]
    assert config["native_command_names"] == ["throttle", "aileron", "elevator"]
    assert [c.unit for c in fixture.spec.controls] == ["1", "rad", "rad"]
    assert [c.minimum for c in fixture.spec.controls] == [0.0, -0.35, -0.35]
    assert not fixture.spec.exogenous
    assert len(config["full_state_names"]) == 18
    assert config["actuator_output_names"] == [
        "surface_left_elevon_rad",
        "surface_right_elevon_rad",
        "propeller_throttle_rad_s",
    ]
    assert len(config["source_identity"]["runtime_assets"]) == 66
    json.dumps(config, allow_nan=False)
    vector = np.arange(18, dtype=np.float64) / 19
    identical(np.asarray(_pack(fixture._unpack(jnp.asarray(vector)))), vector)


@pytest.mark.parametrize("wind", ["calm", "constant", "gust"])
def test_exact_saved_factual_branch_preserves_all_hidden_state(fixture, protocol, wind):
    cell = condition(protocol, wind=wind, heading=90)
    arrays, metadata = fixture.generate(entry(cell), cell)
    assert arrays["full_states"].shape == (61, 18)
    assert arrays["commands"].shape == (60, 3)
    assert arrays["integration_wind"].shape == (60, 20, 3)
    assert arrays["control_prefix"].shape == (10, 3)
    assert metadata["simulator_rng_used"] is False
    json.dumps(metadata, allow_nan=False)
    for origin in (20, 40):
        branch, branch_metadata = fixture.branch(
            arrays["full_states"][origin],
            arrays["commands"][origin : origin + 5],
            origin,
            cell,
        )
        for key in ("time_s", "states", "full_states", "actuator_outputs", "wind"):
            identical(branch[key], arrays[key][origin : origin + 6])
        for key in ("commands", "integration_wind"):
            identical(branch[key], arrays[key][origin : origin + 5])
        assert branch_metadata["origin_index"] == origin
    assert not np.array_equal(
        arrays["full_states"][20, 13:], arrays["full_states"][0, 13:]
    )


def test_recorded_channel_order_is_converted_before_native_integration(
    fixture, protocol
):
    cell = condition(protocol)
    trim, _, _ = fixture._trim(cell, 0)
    vector = np.asarray(_pack(trim.state))
    result, _ = fixture.branch(vector, np.asarray([[0.6, 0.2, -0.1]]), 0, cell)
    np.testing.assert_allclose(
        result["full_states"][-1, 10:13], 0.05 * np.asarray([0.6, -0.1, 0.2])
    )
    np.testing.assert_allclose(
        result["states"][-1, 10:13], 0.05 * np.asarray([0.6, 0.1, -0.2])
    )
    np.testing.assert_allclose(
        result["actuator_outputs"][-1, :2],
        vector[13:15] + 0.05 * np.asarray([-0.1, 0.2]),
    )
    identical(result["commands"], np.asarray([[0.6, 0.2, -0.1]]))


def test_pilot_draws_trim_offset_and_dither_follow_declared_order(
    fixture, protocol, fake_physics
):
    cell = condition(protocol, wind="constant", heading=-90)
    first, metadata = fixture.generate(entry(cell, seed=85), cell)
    replay, replay_metadata = fixture.generate(entry(cell, seed=85), cell)
    assert len(fake_physics) == 1
    for key in first:
        identical(first[key], replay[key])
    assert metadata == replay_metadata
    phases = np.random.Generator(np.random.PCG64(85)).uniform(0, 2 * np.pi, 3)
    np.testing.assert_array_equal(metadata["phases_rad"], phases)
    assert metadata["pilot"]["rate_period"] == 1
    assert metadata["pilot"]["attitude_period"] == 1
    assert metadata["pilot"]["guidance_period"] == 2
    np.testing.assert_array_equal(
        first["control_prefix"], np.repeat([[0.45, -0.12, 0.04]], 10, axis=0)
    )
    np.testing.assert_allclose(
        first["pilot_raw_commands"], np.repeat([[0.45, -0.07, 0.03]], 60, axis=0)
    )
    np.testing.assert_allclose(first["commands"][0], [0.45, -0.19, 0.07])
    for index in (0, 7, 59):
        t = index / 20
        expected_dither = (
            min(1, t)
            * 0.5
            * np.asarray([0.035, 0.025, 0.02])
            * np.sin(np.asarray([1.3, 2.1, 1.7]) * t + phases)
        )
        np.testing.assert_array_equal(first["requested_dither"][index], expected_dither)
        expected = (
            np.asarray([0.45, -0.07, 0.03]) + [0.0, -0.12, 0.04] + expected_dither
        )
        np.testing.assert_array_equal(
            first["commands"][index], np.clip(expected, fixture.lower, fixture.upper)
        )
    # The synthetic trim already included +2 m/s northward wind; generation must not add it again.
    np.testing.assert_allclose(first["states"][0, 3:6], [2.0, -16.0, 0.0], atol=1e-14)
    requested_condition, environment, kwargs = fake_physics[0]
    assert requested_condition.heading_rad == np.pi / 2
    assert kwargs == {"residual_tolerance": 1e-4, "max_evaluations": 300}
    np.testing.assert_array_equal(environment.gravity, [0.0, 0.0, 9.81])


def test_trim_cache_key_is_speed_heading_and_initial_wind(
    fixture, protocol, fake_physics
):
    calm = condition(protocol)
    fixture._trim(calm, 0)
    fixture._trim(condition(protocol, wind="gust"), 2)
    assert len(fake_physics) == 1
    fixture._trim(condition(protocol, wind="constant"), 1)
    fixture._trim(condition(protocol, heading=90), 0)
    changed_speed = condition(protocol)
    changed_speed["speed_m_s"] = 18
    fixture._trim(changed_speed, 0)
    assert len(fake_physics) == 4


@pytest.mark.parametrize(
    "reason", ["unsuccessful_trim", "nonfinite_trim", "trim_outside_command_bounds"]
)
def test_failed_trim_is_json_safe_cached_and_never_repaired(fixture, protocol, reason):
    original = fixture._trim_function
    count = 0

    def bad_trim(*args, **kwargs):
        nonlocal count
        count += 1
        trim = original(*args, **kwargs)
        if reason == "unsuccessful_trim":
            trim.success = False
        elif reason == "nonfinite_trim":
            trim.control = trim.control._replace(propeller=jnp.asarray([np.nan]))
        else:
            trim.control = trim.control._replace(channel=jnp.asarray([0.36, -0.12]))
        return trim

    fixture._trim_function = bad_trim
    cell = condition(protocol)
    details = []
    for _ in range(2):
        with pytest.raises(FixtureSetupError) as error:
            fixture.generate(entry(cell), cell)
        assert error.value.detail["reason"] == reason
        json.dumps(error.value.detail, allow_nan=False)
        details.append(error.value.detail)
    assert count == 1
    assert details[0] == details[1]


def test_unexpected_trim_programming_error_propagates(fixture, protocol):
    def broken(*args, **kwargs):
        raise TypeError("programming defect")

    fixture._trim_function = broken
    cell = condition(protocol)
    with pytest.raises(TypeError, match="programming defect"):
        fixture.generate(entry(cell), cell)
    assert not fixture._trim_cache


def test_gust_uses_global_integer_substep_clock(fixture, protocol):
    cell = condition(protocol, wind="gust")
    trim, _, _ = fixture._trim(cell, 2)
    result, _ = fixture.branch(
        np.asarray(_pack(trim.state)), np.asarray([[0.4, 0.0, 0.0]]), 27, cell
    )
    indices = 27 * 20 + np.arange(20)
    np.testing.assert_array_equal(
        result["integration_wind"][0], np.asarray(fixture._wind(indices, 2))
    )
    t = indices / 400.0
    np.testing.assert_allclose(
        result["integration_wind"][0, :, 0],
        2 * np.sin(np.pi * (t - 0.5) / 2) ** 2,
        atol=1e-15,
    )
    assert np.ptp(result["integration_wind"][0, :, 0]) > 0
    assert result["time_s"].tolist() == [27 / 20, 28 / 20]
    np.testing.assert_array_equal(
        np.asarray(_wind_nwu(jnp.asarray([0, 200, 1000, 1200]), 2)), np.zeros((4, 3))
    )


def test_raw_invalid_tail_is_retained_for_runner_validity(fixture, protocol):
    original = fixture._advance

    def become_invalid(vector, command, index, code):
        result, wind = original(vector, command, index, code)
        return result.at[17].set(jnp.nan) if index >= 2 else result, wind

    fixture._advance = become_invalid
    cell = condition(protocol)
    arrays, _ = fixture.generate(entry(cell), cell)
    assert arrays["full_states"].shape == (61, 18)
    assert np.isfinite(arrays["full_states"][:3]).all()
    assert np.isnan(arrays["full_states"][3:, 17]).all()


def test_branch_rejects_wrong_contract_without_reset(fixture, protocol):
    cell = condition(protocol)
    vector = np.zeros(18)
    tape = np.asarray([[0.4, 0.0, 0.0]])
    with pytest.raises(ValueError, match="integer index"):
        fixture.branch(vector, tape, 1.0, cell)
    with pytest.raises(ValueError, match="full physical state"):
        fixture.branch(vector[:-1], tape, 1, cell)
    with pytest.raises(ValueError, match="declared bounds"):
        fixture.branch(vector, np.asarray([[0.4, 0.4, 0.0]]), 1, cell)
    with pytest.raises(ValueError, match="three columns"):
        fixture.branch(vector, np.ones((1, 2)), 1, cell)


def test_condition_targets_are_air_relative_and_heading_is_ned():
    cell = {"speed_m_s": 18.0, "heading_nwu_deg": 90}
    np.testing.assert_array_equal(
        _targets(cell, 1, np.zeros(3), 0), [18, 100, -np.pi / 2]
    )


def test_wrong_source_root_or_float32_runtime_fails_before_physics(
    protocol, fake_physics
):
    changed = copy.deepcopy(protocol)
    changed["generation"]["cascade"]["source_root"] = (
        "/private/tmp/not-the-pinned-cascade"
    )
    with pytest.raises(ValueError, match="source archive"):
        Fixture(changed)
    with jax.enable_x64(False), pytest.raises(ValueError, match="float64 runtime"):
        Fixture(protocol)
    assert not fake_physics
