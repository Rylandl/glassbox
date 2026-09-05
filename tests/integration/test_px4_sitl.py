from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from glassbox.control.fitted import NMPCController, default_solver_policy
from glassbox.core.model import ExecutableModel
from glassbox.integrations.px4 import (
    PX4HILActuatorSource,
    PX4MavlinkStateSource,
)
from glassbox.integrations.px4_nmpc_shadow import px4_shadow_link, run_px4_nmpc_shadow

PX4_SITL_IMAGE = (
    "px4io/px4-sitl@"
    "sha256:01866d912ac22ca6119a996b830cf628a6d47dfb60fdccc41cd9f44b62935a44"
)
RUN_PX4_SITL = os.environ.get("GLASSBOX_RUN_PX4_SITL") == "1"
RUN_PX4_FLIGHT_SHADOW = os.environ.get("GLASSBOX_RUN_PX4_FLIGHT_SHADOW") == "1"
SIH_QUADX_CANONICAL_MOTOR_INDICES = (2, 0, 3, 1)
FLIGHT_SHADOW_PROFILES = (
    "vertical_steps",
    "lateral_steps",
    "yaw_steps",
    "combined",
)

pytestmark = [
    pytest.mark.px4_sitl,
    pytest.mark.skipif(
        not RUN_PX4_SITL,
        reason="set GLASSBOX_RUN_PX4_SITL=1 to launch the pinned PX4 SIH image",
    ),
]


@dataclass(frozen=True)
class PX4SITLFixture:
    container_name: str
    state_source: PX4MavlinkStateSource


@pytest.fixture
def px4_sitl() -> Iterator[PX4SITLFixture]:
    if shutil.which("docker") is None:
        pytest.fail("the opt-in PX4 SITL test requires Docker")
    daemon = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if daemon.returncode != 0:
        pytest.fail(f"Docker is unavailable: {daemon.stderr.strip()}")

    container_name = f"glassbox-px4-sitl-{uuid.uuid4().hex[:10]}"
    launched = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "--add-host",
            "host.docker.internal:host-gateway",
            "--env",
            "PX4_SIM_MODEL=sihsim_quadx",
            PX4_SITL_IMAGE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if launched.returncode != 0:
        pytest.fail(f"could not launch PX4 SIH: {launched.stderr.strip()}")

    source: PX4MavlinkStateSource | None = None
    try:
        # The onboard telemetry link leaves the GCS link free for the explicit
        # profile driver used only by the flown shadow test below.
        source = PX4MavlinkStateSource.connect(
            "udpin:0.0.0.0:14540",
            heartbeat_timeout_s=20.0,
        )
        yield PX4SITLFixture(container_name, source)
    finally:
        if source is not None:
            source.close()
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            text=True,
            check=False,
        )


def test_pinned_px4_sih_emits_canonical_read_only_state(
    px4_sitl: PX4SITLFixture,
) -> None:
    px4_state_source = px4_sitl.state_source
    samples = [px4_state_source.next_sample(timeout_s=2.0) for _ in range(10)]
    states = np.asarray([sample.state for sample in samples])

    assert states.shape == (10, 13)
    assert np.all(np.isfinite(states))
    np.testing.assert_allclose(
        np.linalg.norm(states[:, 6:10], axis=1),
        np.ones(10),
        atol=1e-9,
    )
    assert max(sample.message_skew_s for sample in samples) <= 0.10
    assert max(sample.maximum_receive_age_s for sample in samples) <= 0.25
    assert all(
        np.isfinite(sample.estimated_source_clock_lag_s)
        and sample.estimated_source_clock_lag_s >= 0.0
        for sample in samples
    )
    boot_times = np.asarray(
        [sample.position_time_boot_ms for sample in samples], dtype=np.int64
    )
    boot_advances_ms = np.diff(boot_times) % (2**32)
    assert np.all((boot_advances_ms > 0) & (boot_advances_ms < 1_000))
    assert not hasattr(px4_state_source, "send")


@pytest.mark.skipif(
    RUN_PX4_FLIGHT_SHADOW,
    reason="the flown shadow path obtains its applied command from telemetry",
)
def test_eligible_artifact_runs_complete_nmpc_shadow_path_when_provided(
    px4_sitl: PX4SITLFixture,
) -> None:
    px4_state_source = px4_sitl.state_source
    model_path = os.environ.get("GLASSBOX_PX4_NMPC_MODEL")
    command_text = os.environ.get("GLASSBOX_PX4_NMPC_COMMAND")
    if model_path is None and command_text is None:
        pytest.skip("set the PX4 NMPC model and command variables for the full path")
    if model_path is None or command_text is None:
        pytest.fail(
            "GLASSBOX_PX4_NMPC_MODEL and GLASSBOX_PX4_NMPC_COMMAND must be set together"
        )

    model = ExecutableModel.load(model_path)
    if model.input_spec.vehicle.family != "multirotor":
        pytest.fail("the maintained PX4 SIH fixture is a multirotor")
    try:
        previous_command = np.asarray([float(item) for item in command_text.split(",")])
    except ValueError:
        pytest.fail("GLASSBOX_PX4_NMPC_COMMAND must be comma-separated numbers")

    link = px4_shadow_link(
        px4_state_source,
        model,
        previous_command=previous_command,
    )
    lines: list[str] = []
    summary = run_px4_nmpc_shadow(
        link,
        NMPCController(
            model,
            policy=replace(
                default_solver_policy(model), allow_unresolved_parameters=True
            ),
        ),
        steps=3,
        write_line=lines.append,
    )
    samples = [json.loads(line) for line in lines]

    assert link.writable is False
    assert summary.written_command_count == 0
    assert summary.steps == 3
    assert len(samples) == 3
    assert all(
        sample["diagnostics"]["maximum_command_bound_violation"] <= 1e-6
        for sample in samples
    )
    for sample in samples:
        if sample["solve_time_s"] > summary.interval_s:
            assert sample["status"] == "deadline_exceeded"
            assert sample["used_fallback"]


def _attitude_span_deg(states: np.ndarray) -> float:
    quaternions = states[:, 6:10]
    reference = quaternions[0]
    absolute_dots = np.abs(quaternions @ reference)
    angles_rad = 2.0 * np.arccos(np.clip(absolute_dots, 0.0, 1.0))
    return float(np.rad2deg(np.max(angles_rad)))


def _assert_profile_excitation(profile: str, states: np.ndarray) -> None:
    vertical_span_m = float(np.ptp(states[:, 2]))
    horizontal_span_m = float(np.max(np.ptp(states[:, :2], axis=0)))
    attitude_span_deg = _attitude_span_deg(states)

    if profile == "vertical_steps":
        assert vertical_span_m > 0.20
    elif profile == "lateral_steps":
        assert horizontal_span_m > 0.20
    elif profile == "yaw_steps":
        assert attitude_span_deg > 10.0
    elif profile == "combined":
        assert vertical_span_m > 0.10
        assert horizontal_span_m > 0.20
        assert attitude_span_deg > 10.0
    else:  # pragma: no cover - the parameter list is maintained above
        raise AssertionError(f"unsupported flight-shadow profile: {profile}")


@pytest.mark.parametrize("profile", FLIGHT_SHADOW_PROFILES)
def test_flown_profile_pairs_real_applied_commands_with_shadow_solver(
    px4_sitl: PX4SITLFixture,
    profile: str,
) -> None:
    if not RUN_PX4_FLIGHT_SHADOW:
        pytest.skip(
            "set GLASSBOX_RUN_PX4_FLIGHT_SHADOW=1 to arm the disposable SIH fixture"
        )
    model_path = os.environ.get("GLASSBOX_PX4_NMPC_MODEL")
    if model_path is None:
        pytest.fail("the flown shadow test requires GLASSBOX_PX4_NMPC_MODEL")

    model = ExecutableModel.load(model_path)
    expected_roles = (
        "motor_front_left",
        "motor_front_right",
        "motor_rear_right",
        "motor_rear_left",
    )
    roles = tuple(channel.role for channel in model.actuation.command_channels)
    if model.input_spec.vehicle.family != "multirotor" or roles != expected_roles:
        pytest.fail(
            "the maintained SIH quadx fixture requires canonical quadrotor motor roles"
        )

    driver: subprocess.Popen[str] | None = None
    with PX4HILActuatorSource.connect(
        command_indices=SIH_QUADX_CANONICAL_MOTOR_INDICES,
        heartbeat_timeout_s=20.0,
    ) as actuator_source:
        # Wait for five seconds of PX4 boot time before requesting an ordinary
        # takeoff, matching the readiness margin used by dataset recording.
        boot_deadline = time.monotonic() + 15.0
        actuator_sample = actuator_source.next_sample(timeout_s=2.0)
        while (
            actuator_sample.source_time_us < 5_000_000
            and time.monotonic() < boot_deadline
        ):
            actuator_sample = actuator_source.next_sample(timeout_s=2.0)
        if actuator_sample.source_time_us < 5_000_000:
            pytest.fail("PX4 SIH did not reach the takeoff readiness margin")

        takeoff = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                "/root",
                px4_sitl.container_name,
                "/opt/px4/bin/px4-commander",
                "takeoff",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if takeoff.returncode != 0:
            pytest.fail(f"PX4 takeoff command failed: {takeoff.stderr.strip()}")

        armed_deadline = time.monotonic() + 10.0
        while not actuator_sample.armed and time.monotonic() < armed_deadline:
            actuator_sample = actuator_source.next_sample(timeout_s=2.0)
        if not actuator_sample.armed:
            pytest.fail("PX4 SIH did not arm after its normal takeoff command")

        airborne_deadline = time.monotonic() + 10.0
        state_sample = px4_sitl.state_source.next_sample(timeout_s=2.0)
        while state_sample.state[2] < 0.5 and time.monotonic() < airborne_deadline:
            state_sample = px4_sitl.state_source.next_sample(timeout_s=2.0)
        if state_sample.state[2] < 0.5:
            pytest.fail("PX4 SIH did not become airborne")

        try:
            driver = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "glassbox.io.sitl_profile",
                    profile,
                    "--condition",
                    "high",
                ],
                cwd=Path(__file__).parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            link = px4_shadow_link(
                px4_sitl.state_source,
                model,
                applied_command_source=actuator_source,
            )
            lines: list[str] = []
            summary = run_px4_nmpc_shadow(
                link,
                NMPCController(
                    model,
                    policy=replace(
                        default_solver_policy(model), allow_unresolved_parameters=True
                    ),
                ),
                steps=160,
                write_line=lines.append,
            )
            output, _ = driver.communicate(timeout=50.0)
            if driver.returncode != 0:
                pytest.fail(f"PX4 profile driver failed:\n{output}")
        finally:
            if driver is not None and driver.poll() is None:
                driver.terminate()
                try:
                    driver.communicate(timeout=5.0)
                except subprocess.TimeoutExpired:
                    driver.kill()
                    driver.communicate(timeout=5.0)

    samples = [json.loads(line) for line in lines]
    states = np.asarray([sample["state"] for sample in samples])
    applied_commands = np.asarray([sample["applied_command"] for sample in samples])
    assert link.writable is False
    assert summary.written_command_count == 0
    assert link.applied_command_source_kind == "telemetry"
    assert summary.steps == len(samples) == 160
    assert all(sample["armed"] is True for sample in samples)
    assert summary.maximum_applied_command_skew_s <= min(
        0.10, model.runtime_spec.sample_period_s
    )
    assert summary.maximum_receive_age_s <= 0.25
    assert np.max(np.ptp(applied_commands, axis=0)) > 0.02
    _assert_profile_excitation(profile, states)
    assert all(
        sample["diagnostics"]["maximum_command_bound_violation"] <= 1e-6
        for sample in samples
    )
