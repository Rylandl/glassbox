from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from glassbox.integrations.px4 import (
    PX4HILActuatorSource,
    PX4MavlinkStateSource,
)
from glassbox.integrations.px4_nmpc_shadow import run_px4_nmpc_shadow
from glassbox.runtime import RuntimeDynamicsModel

PX4_SITL_IMAGE = (
    "px4io/px4-sitl@"
    "sha256:01866d912ac22ca6119a996b830cf628a6d47dfb60fdccc41cd9f44b62935a44"
)
RUN_PX4_SITL = os.environ.get("GLASSBOX_RUN_PX4_SITL") == "1"
RUN_PX4_FLIGHT_SHADOW = os.environ.get("GLASSBOX_RUN_PX4_FLIGHT_SHADOW") == "1"
SIH_QUADX_CANONICAL_MOTOR_INDICES = (2, 0, 3, 1)

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


@pytest.fixture(scope="module")
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


def test_eligible_artifact_runs_complete_nmpc_shadow_path_when_provided(
    px4_sitl: PX4SITLFixture,
) -> None:
    px4_state_source = px4_sitl.state_source
    model_path = os.environ.get("GLASSBOX_PX4_NMPC_MODEL")
    command_text = os.environ.get("GLASSBOX_PX4_NMPC_COMMAND")
    if model_path is None and command_text is None:
        pytest.skip("set the PX4 NMPC model and command variables for the full path")
    if command_text is None and RUN_PX4_FLIGHT_SHADOW:
        pytest.skip("the flown shadow path obtains its applied command from telemetry")
    if model_path is None or command_text is None:
        pytest.fail(
            "GLASSBOX_PX4_NMPC_MODEL and GLASSBOX_PX4_NMPC_COMMAND must be set together"
        )

    model = RuntimeDynamicsModel.load(model_path)
    if model.input_spec.vehicle.family != "multirotor":
        pytest.fail("the maintained PX4 SIH fixture is a multirotor")
    try:
        previous_command = np.asarray([float(item) for item in command_text.split(",")])
    except ValueError:
        pytest.fail("GLASSBOX_PX4_NMPC_COMMAND must be comma-separated numbers")

    report = run_px4_nmpc_shadow(
        px4_state_source,
        model,
        previous_command,
        sample_count=3,
    )

    assert report["commands_transmitted"] is False
    assert report["summary"]["sample_count"] == 3
    assert all(
        sample["maximum_command_bound_violation"] <= 1e-6
        for sample in report["samples"]
    )
    for sample in report["samples"]:
        if sample["solve_time_s"] > report["model_sample_period_s"]:
            assert sample["status"] == "deadline_exceeded"
            assert sample["used_fallback"]


def test_flown_profile_pairs_real_applied_commands_with_shadow_solver(
    px4_sitl: PX4SITLFixture,
) -> None:
    if not RUN_PX4_FLIGHT_SHADOW:
        pytest.skip(
            "set GLASSBOX_RUN_PX4_FLIGHT_SHADOW=1 to arm the disposable SIH fixture"
        )
    model_path = os.environ.get("GLASSBOX_PX4_NMPC_MODEL")
    if model_path is None:
        pytest.fail("the flown shadow test requires GLASSBOX_PX4_NMPC_MODEL")

    model = RuntimeDynamicsModel.load(model_path)
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
                    "glassbox.sitl_profile",
                    "vertical_steps",
                    "--condition",
                    "high",
                ],
                cwd=Path(__file__).parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            report = run_px4_nmpc_shadow(
                px4_sitl.state_source,
                model,
                applied_command_source=actuator_source,
                sample_count=80,
            )
            output, _ = driver.communicate(timeout=35.0)
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

    states = np.asarray([sample["state"] for sample in report["samples"]])
    command_range = np.asarray(report["summary"]["applied_command_peak_to_peak"])
    assert report["commands_transmitted"] is False
    assert report["applied_command_source"] == "telemetry"
    assert report["summary"]["all_applied_command_samples_armed"] is True
    assert report["summary"]["maximum_applied_command_state_skew_s"] <= min(
        0.10, model.runtime_spec.sample_period_s
    )
    assert report["summary"]["maximum_applied_command_receive_age_s"] <= 0.25
    assert np.max(command_range) > 0.02
    assert np.ptp(states[:, 2]) > 0.20
    assert all(
        sample["maximum_command_bound_violation"] <= 1e-6
        for sample in report["samples"]
    )
