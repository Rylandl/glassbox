from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from glassbox.integrations.px4 import PX4MavlinkStateSource
from glassbox.integrations.px4_nmpc_shadow import run_px4_nmpc_shadow
from glassbox.runtime import RuntimeDynamicsModel

PX4_SITL_IMAGE = (
    "px4io/px4-sitl@"
    "sha256:01866d912ac22ca6119a996b830cf628a6d47dfb60fdccc41cd9f44b62935a44"
)
RUN_PX4_SITL = os.environ.get("GLASSBOX_RUN_PX4_SITL") == "1"

pytestmark = [
    pytest.mark.px4_sitl,
    pytest.mark.skipif(
        not RUN_PX4_SITL,
        reason="set GLASSBOX_RUN_PX4_SITL=1 to launch the pinned PX4 SIH image",
    ),
]


@pytest.fixture(scope="module")
def px4_state_source() -> Iterator[PX4MavlinkStateSource]:
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
        source = PX4MavlinkStateSource.connect(heartbeat_timeout_s=20.0)
        yield source
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
    px4_state_source: PX4MavlinkStateSource,
) -> None:
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
    px4_state_source: PX4MavlinkStateSource,
) -> None:
    model_path = os.environ.get("GLASSBOX_PX4_NMPC_MODEL")
    command_text = os.environ.get("GLASSBOX_PX4_NMPC_COMMAND")
    if model_path is None and command_text is None:
        pytest.skip("set the PX4 NMPC model and command variables for the full path")
    if model_path is None or command_text is None:
        pytest.fail(
            "GLASSBOX_PX4_NMPC_MODEL and GLASSBOX_PX4_NMPC_COMMAND must be set together"
        )

    model = RuntimeDynamicsModel.load(model_path)
    if model.input_spec.vehicle.family != "multirotor":
        pytest.fail("the maintained PX4 SIH fixture is a multirotor")
    try:
        previous_command = np.asarray(
            [float(item) for item in command_text.split(",")]
        )
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
