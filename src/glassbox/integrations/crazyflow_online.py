"""Online belief adaptation against the hidden Crazyflow plant.

Adaptation runs in a separate process so a slow fit cannot stall the control
loop. This module owns that worker and its inter-process contract, the
reconstruction of a belief from a worker result, and the closed-loop recovery
that partitions evidence in time between the controller and the fit.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import signal
import time
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.belief.belief import (
    DynamicsBelief,
    LocalGaussianParameterBelief,
    RuntimeDynamicsBelief,
    parameter_belief_from_dict,
)
from glassbox.control.flight_supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisorMode,
    SupervisorReason,
)
from glassbox.control.nmpc import NMPCController
from glassbox.core.data import Trajectory
from glassbox.core.dynamics import (
    ModelParams,
)
from glassbox.core.geometry import quaternion_from_euler, rigid_body_local_error
from glassbox.core.synthetic import resting_state
from glassbox.integrations.crazyflow import CrazyflowPlant
from glassbox.integrations.crazyflow_fleet import (
    FLEET_LOG_ARM_LENGTH_RATIOS,
)
from glassbox.integrations.crazyflow_telemetry import (
    DEFAULT_ARM_LENGTH_RATIO,
    _applied_motor_observation_channels,
    _crazyflow_solver_policy,
    generate_crazyflow_trajectory,
    prewarm_controller,
)

ONLINE_EVIDENCE_DURATION_S = 0.8
ONLINE_POST_INSTALL_DURATION_S = 1.0
ONLINE_MAXIMUM_DURATION_S = 7.5
ONLINE_INTEGRATED_PREWARM_STEPS = 10


@dataclass(frozen=True)
class _PreparedOnlineController:
    """Candidate belief/controller pair prepared away from the control loop."""

    update_report: dict[str, Any]
    update_wall_time_s: float
    controller_validation_wall_time_s: float | None
    total_wall_time_s: float
    validation_statuses: tuple[str, ...]
    ready_for_install: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class _WorkerBeliefUpdate:
    """Pickle-safe numerical result returned by the isolated update process."""

    parameter_leaves: tuple[np.ndarray, ...]
    parameter_belief_payload: dict[str, Any]
    provenance: dict[str, Any]
    predictive_error_parameter_update_count: int
    update_report: dict[str, Any]
    update_wall_time_s: float
    update_cpu_time_s: float
    process_id: int
    process_niceness: int | None


def _initialize_adaptation_worker() -> None:
    """Lower worker scheduling priority without touching the control process."""

    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass


def _process_niceness() -> int | None:
    if not hasattr(os, "nice"):
        return None
    try:
        return os.nice(0)
    except OSError:
        return None


def _set_process_suspended(process_id: int, *, suspended: bool) -> bool:
    """Best-effort POSIX temporal partition for the adaptation worker."""

    process_signal = getattr(signal, "SIGSTOP" if suspended else "SIGCONT", None)
    if process_signal is None:
        return False
    try:
        os.kill(process_id, process_signal)
    except (OSError, ProcessLookupError):
        return False
    if suspended and all(
        hasattr(os, name)
        for name in ("waitid", "P_PID", "WSTOPPED", "WEXITED", "WNOWAIT")
    ):
        try:
            status = os.waitid(
                os.P_PID,
                process_id,
                os.WSTOPPED | os.WEXITED | os.WNOWAIT,
            )
        except (ChildProcessError, OSError):
            return False
        return status is not None and status.si_code == getattr(
            os,
            "CLD_STOPPED",
            status.si_code,
        )
    return True


def _run_isolated_belief_update(
    belief: DynamicsBelief,
    telemetry: Trajectory,
) -> _WorkerBeliefUpdate:
    started_at = time.perf_counter()
    cpu_started_at = time.process_time()
    updated, update_report = belief.update(telemetry)
    update_cpu_time_s = time.process_time() - cpu_started_at
    return _WorkerBeliefUpdate(
        parameter_leaves=tuple(
            np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(updated.params)
        ),
        parameter_belief_payload=updated.parameter_belief.to_dict(),
        provenance=dict(updated.provenance),
        predictive_error_parameter_update_count=(
            updated.predictive_error_parameter_update_count
        ),
        update_report=update_report.to_dict(),
        update_wall_time_s=time.perf_counter() - started_at,
        update_cpu_time_s=update_cpu_time_s,
        process_id=os.getpid(),
        process_niceness=_process_niceness(),
    )


def _belief_from_worker_update(
    belief: DynamicsBelief,
    update: _WorkerBeliefUpdate,
) -> DynamicsBelief:
    return replace(
        belief,
        params=_parameters_from_worker_update(belief.params, update),
        parameter_belief=parameter_belief_from_dict(update.parameter_belief_payload),
        predictive_error_parameter_update_count=(
            update.predictive_error_parameter_update_count
        ),
        provenance=update.provenance,
    )


def _parameters_from_worker_update(
    template: ModelParams,
    update: _WorkerBeliefUpdate,
) -> ModelParams:
    template_leaves, structure = jax.tree_util.tree_flatten(template)
    if len(template_leaves) != len(update.parameter_leaves):
        raise ValueError("isolated update parameter structure changed")
    leaves = []
    for template_leaf, candidate_leaf in zip(
        template_leaves,
        update.parameter_leaves,
        strict=True,
    ):
        template_array = np.asarray(template_leaf)
        candidate_array = np.asarray(candidate_leaf)
        if candidate_array.shape != template_array.shape:
            raise ValueError("isolated update parameter shape changed")
        if candidate_array.dtype != template_array.dtype:
            raise ValueError("isolated update parameter dtype changed")
        if not np.all(np.isfinite(candidate_array)):
            raise ValueError("isolated update parameters must be finite")
        leaves.append(candidate_array)
    return jax.tree_util.tree_unflatten(structure, leaves)


def _runtime_belief_from_worker_update(
    template: NMPCController,
    update: _WorkerBeliefUpdate,
) -> RuntimeDynamicsBelief:
    """Bind numerical worker output to the prevalidated runtime template."""

    parameter_belief = template.belief.parameter_belief
    if update.parameter_belief_payload != parameter_belief.to_dict():
        raise ValueError("isolated update parameter uncertainty changed")
    if (
        update.predictive_error_parameter_update_count
        != template.belief.predictive_error_parameter_update_count
    ):
        raise ValueError("isolated update predictive-error semantics changed")
    nominal = template.model.rebind_parameters(
        _parameters_from_worker_update(template.model.params, update)
    )
    return replace(template.belief, nominal=nominal)


def _post_update_controller_template(belief: DynamicsBelief) -> DynamicsBelief:
    """Create the exact non-mean runtime semantics expected after one update."""

    parameter_belief = belief.parameter_belief
    if not isinstance(parameter_belief, LocalGaussianParameterBelief):
        raise ValueError("online controller template requires parameter uncertainty")
    return replace(
        belief,
        parameter_belief=replace(
            parameter_belief,
            evidence_count=parameter_belief.evidence_count + 1,
            effective_sample_count=parameter_belief.effective_sample_count + 1.0,
            update_count=parameter_belief.update_count + 1,
        ),
    )


def _recovery_initial_state(belief: DynamicsBelief) -> np.ndarray:
    state = resting_state()
    state[0:3] = (0.10, -0.08, 1.12)
    state[6:10] = quaternion_from_euler(0.16, -0.12, 0.08)
    envelope = belief.runtime_spec.validity_envelope
    state[3:6] = np.asarray(envelope.body_velocity_center_m_s) + 0.15 * np.asarray(
        envelope.body_velocity_half_width_m_s
    ) * np.asarray((1.0, -1.0, -1.0))
    state[10:13] = np.asarray(
        envelope.angular_velocity_center_rad_s
    ) + 0.25 * np.asarray(envelope.angular_velocity_half_width_rad_s) * np.asarray(
        (1.0, -1.0, 1.0)
    )
    return state


def _online_recovery_trajectory(
    belief: DynamicsBelief,
    states: list[np.ndarray],
    controls: list[np.ndarray],
    applied_motor_thrust: list[np.ndarray],
) -> Trajectory:
    """Package a state-aligned prefix without exposing the hidden configuration."""

    if len(states) != len(controls) + 1:
        raise ValueError("online recovery states must bracket every control")
    if len(applied_motor_thrust) != len(states):
        raise ValueError("applied motor observations must align with recovery states")
    if len(controls) < 1:
        raise ValueError("online recovery evidence needs at least one transition")
    sample_period_s = belief.runtime_spec.sample_period_s
    return Trajectory(
        time_s=np.arange(len(states), dtype=np.float64) * sample_period_s,
        states=np.asarray(states, dtype=np.float64),
        controls=np.asarray(controls, dtype=np.float64),
        observations=np.asarray(applied_motor_thrust, dtype=np.float64),
        spec=replace(
            belief.input_spec,
            observations=_applied_motor_observation_channels(),
        ),
        labels={
            "source_group": "unknown-target-online-recovery",
            "evidence_role": "stale-controller-recovery-prefix",
        },
        provenance={
            "collection": {
                "controller_model": "prechange_stale_belief",
                "state_aligned_applied_actuator_observations": True,
            },
            "telemetry_only": {
                "hidden_physical_parameters_supplied_to_glassbox": False,
                "target_configuration_label_supplied_to_glassbox": False,
            },
        },
    )


def _normalized_rms(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _eligible_recovery_evidence_start(
    state_validity: list[float],
    evidence_steps: int,
) -> int | None:
    """Return the first usable streaming-window start under the fixed gate."""

    if evidence_steps < 1:
        raise ValueError("evidence_steps must be positive")
    start = len(state_validity) - (evidence_steps + 1)
    if start < 0:
        return None
    recent = state_validity[start:]
    if not all(np.isfinite(value) for value in recent):
        return None
    return start if max(recent) <= 1.0 else None


def _simulate_online_adaptation_recovery(
    belief: DynamicsBelief,
    initial_state: np.ndarray,
) -> dict[str, Any]:
    """Continue stale control while preparing and atomically installing an update."""

    plant = CrazyflowPlant()
    plant.set_arm_length_ratio(DEFAULT_ARM_LENGTH_RATIO)
    hover = np.full(4, plant.hover_motor_thrust_fraction, dtype=np.float64)
    reference_state = resting_state()
    reference_state[2] = 1.0
    stale_controller = NMPCController(belief, _policy=_crazyflow_solver_policy())
    reference = stale_controller.hold_reference(jnp.asarray(reference_state))

    prewarm_started_at = time.perf_counter()
    cold, warm = prewarm_controller(stale_controller, reference, initial_state, hover)
    stale_prewarm_wall_time_s = time.perf_counter() - prewarm_started_at
    if not cold.command_usable or not warm.command_usable:
        plant.close()
        raise RuntimeError(
            "stale controller could not be prewarmed for online recovery"
        )

    controller_template = NMPCController(
        _post_update_controller_template(belief),
        _policy=_crazyflow_solver_policy(),
    )
    template_reference = controller_template.hold_reference(
        jnp.asarray(reference_state)
    )
    template_prewarm_started_at = time.perf_counter()
    template_cold, template_warm = prewarm_controller(
        controller_template, template_reference, initial_state, hover
    )
    template_hover_cold, template_hover_warm = prewarm_controller(
        controller_template, template_reference, reference_state, hover
    )
    candidate_template_prewarm_wall_time_s = (
        time.perf_counter() - template_prewarm_started_at
    )
    if not all(
        result.command_usable
        for result in (
            template_cold,
            template_warm,
            template_hover_cold,
            template_hover_warm,
        )
    ):
        plant.close()
        raise RuntimeError("post-update controller template could not be prewarmed")

    worker_prewarm_telemetry = generate_crazyflow_trajectory(
        seed=299,
        duration_s=ONLINE_EVIDENCE_DURATION_S,
        arm_length_ratio=math.exp(FLEET_LOG_ARM_LENGTH_RATIOS[-1]),
        source_group="known-fleet-adaptation-worker-prewarm",
        configuration_id="known_fleet_worker_prewarm",
    )
    worker_environment = {
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1 "
            "--xla_force_host_platform_device_count=1"
        ),
        "NPROC": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NUM_INTRA_THREADS": "1",
        "NUM_INTER_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    previous_environment = {name: os.environ.get(name) for name in worker_environment}
    worker_prewarm_started_at = time.perf_counter()
    executor: ProcessPoolExecutor | None = None
    try:
        os.environ.update(worker_environment)
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_adaptation_worker,
        )
        worker_prewarm = executor.submit(
            _run_isolated_belief_update,
            belief,
            worker_prewarm_telemetry,
        ).result()
    except Exception:
        plant.close()
        if executor is not None:
            executor.shutdown(wait=True)
        raise
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    worker_prewarm_wall_time_s = time.perf_counter() - worker_prewarm_started_at
    assert executor is not None

    # Prewarm the integrated controller/plant path before auditing deadlines.
    integrated_prewarm_started_at = time.perf_counter()
    prewarm_sample = plant.reset(
        initial_state,
        applied_motor_thrust_fraction=hover,
    )
    prewarm_previous = hover
    prewarm_warm_start = warm.warm_start
    for _ in range(ONLINE_INTEGRATED_PREWARM_STEPS):
        prewarm_result = stale_controller.solve(
            jnp.asarray(prewarm_sample.state),
            reference,
            jnp.asarray(prewarm_previous),
            applied_command=jnp.asarray(prewarm_sample.applied_motor_thrust_fraction),
            warm_start=prewarm_warm_start,
        )
        jax.block_until_ready(prewarm_result.command)
        if not prewarm_result.command_usable:
            plant.close()
            executor.shutdown(wait=True)
            raise RuntimeError("integrated stale control prewarm was not usable")
        prewarm_previous = np.asarray(prewarm_result.command)
        prewarm_warm_start = prewarm_result.warm_start
        prewarm_sample = plant.step(prewarm_previous)
    integrated_prewarm_wall_time_s = time.perf_counter() - integrated_prewarm_started_at
    sample = plant.reset(initial_state, applied_motor_thrust_fraction=hover)

    sample_period_s = plant.sample_period_s
    supervisor_config = MultirotorSupervisorConfig(
        command_minimum=tuple(
            float(value) for value in stale_controller.model.command_minimum
        ),
        command_maximum=tuple(
            float(value) for value in stale_controller.model.command_maximum
        ),
        collective_hold_command=tuple(float(value) for value in hover),
        maximum_state_age_s=2.0 * sample_period_s,
        maximum_command_age_s=sample_period_s,
    )
    supervisor = MultirotorFlightSupervisor(supervisor_config)
    evidence_steps = round(ONLINE_EVIDENCE_DURATION_S / sample_period_s)
    maximum_steps = round(ONLINE_MAXIMUM_DURATION_S / sample_period_s)
    post_install_steps = round(ONLINE_POST_INSTALL_DURATION_S / sample_period_s)
    states = [np.asarray(sample.state)]
    controls: list[np.ndarray] = []
    applied = [np.asarray(sample.applied_motor_thrust_fraction)]
    state_validity = [
        float(
            np.max(
                np.asarray(
                    stale_controller.model.validity_utilization(
                        jnp.asarray(sample.state)
                    )
                )
            )
        )
    ]
    control_modes: list[str] = []
    solve_times: list[float] = []
    outer_step_times: list[float] = []
    solve_statuses: list[str] = []
    solve_messages: list[str] = []
    support_modes: list[str] = []
    predicted_validity: list[float] = []
    support_validity: list[float] = []
    loop_lag: list[float] = []
    fallback_count = 0
    support_applied = 0
    nonfinite_diagnostic_count = 0
    outer_deadline_miss_count = 0
    outer_deadline_miss_steps: list[int] = []
    outer_deadline_miss_details: list[dict[str, Any]] = []
    absolute_deadline_miss_steps: list[int] = []
    absolute_completion_lateness: list[float] = []
    supervisor_modes: list[str] = []
    supervisor_reasons: list[str] = []
    supervisor_decision_times: list[float] = []
    supervisor_intervention_count = 0
    supervisor_fault_injected = False
    supervisor_fault_step: int | None = None
    supervisor_fault_response: dict[str, Any] | None = None
    previous = hover
    active_controller = stale_controller
    active_reference = reference
    active_warm_start = warm.warm_start
    active_mode = "stale"
    submission_wall_time: float | None = None
    submission_sim_time_s: float | None = None
    evidence_start_sim_time_s: float | None = None
    evidence_maximum_validity: float | None = None
    candidate_ready_wall_time_s: float | None = None
    candidate_resolved_sim_time_s: float | None = None
    install_wall_time_s: float | None = None
    install_sim_time_s: float | None = None
    install_step: int | None = None
    prepared: _PreparedOnlineController | None = None
    worker_update: _WorkerBeliefUpdate | None = None
    candidate_future: Future[_WorkerBeliefUpdate] | None = None
    future_consumed = False
    worker_error: str | None = None
    worker_result_wall_time_s: float | None = None
    pending_controller: NMPCController | None = None
    pending_reference: Any | None = None
    worker_suspend_count = 0
    worker_resume_count = 0
    temporal_partition_available = hasattr(signal, "SIGSTOP") and hasattr(
        signal, "SIGCONT"
    )
    control_loop_started_at = time.perf_counter()
    control_loop_ended_at = control_loop_started_at
    try:
        for index in range(maximum_steps):
            step_started_at = time.perf_counter()
            simulation_time_s = index * sample_period_s
            worker_suspended = False
            candidate_deferred_for_tick = False
            worker_resolution_wall_time_s = 0.0
            if (
                candidate_future is not None
                and not future_consumed
                and not candidate_future.done()
            ):
                worker_suspended = _set_process_suspended(
                    worker_prewarm.process_id,
                    suspended=True,
                )
                worker_suspend_count += int(worker_suspended)
            if (
                candidate_future is not None
                and not future_consumed
                and candidate_future.done()
            ):
                worker_resolution_started_at = time.perf_counter()
                future_consumed = True
                worker_result_wall_time_s = (
                    time.perf_counter() - control_loop_started_at
                )
                candidate_resolved_sim_time_s = simulation_time_s
                try:
                    worker_update = candidate_future.result()
                    if not bool(worker_update.update_report.get("applied", False)):
                        reason = str(
                            worker_update.update_report.get(
                                "reason",
                                "isolated belief update was not applied",
                            )
                        )
                        prepared = _PreparedOnlineController(
                            update_report=worker_update.update_report,
                            update_wall_time_s=worker_update.update_wall_time_s,
                            controller_validation_wall_time_s=None,
                            total_wall_time_s=worker_update.update_wall_time_s,
                            validation_statuses=(),
                            ready_for_install=False,
                            rejection_reason=reason,
                        )
                        candidate_ready_wall_time_s = worker_result_wall_time_s
                    else:
                        updated_belief = _runtime_belief_from_worker_update(
                            controller_template,
                            worker_update,
                        )
                        pending_controller = controller_template.rebind_belief(
                            updated_belief
                        )
                        pending_reference = pending_controller.hold_reference(
                            jnp.asarray(reference_state)
                        )
                        candidate_deferred_for_tick = True
                except Exception as error:
                    worker_error = f"{type(error).__name__}: {error}"
                    reason = f"isolated update failed: {worker_error}"
                    update_wall_time_s = (
                        0.0
                        if worker_update is None
                        else worker_update.update_wall_time_s
                    )
                    prepared = _PreparedOnlineController(
                        update_report={"applied": False, "reason": reason},
                        update_wall_time_s=update_wall_time_s,
                        controller_validation_wall_time_s=None,
                        total_wall_time_s=update_wall_time_s,
                        validation_statuses=(),
                        ready_for_install=False,
                        rejection_reason=reason,
                    )
                    candidate_ready_wall_time_s = worker_result_wall_time_s
                worker_resolution_wall_time_s = (
                    time.perf_counter() - worker_resolution_started_at
                )

            if install_step is not None and index - install_step >= post_install_steps:
                break
            if (
                future_consumed
                and pending_controller is None
                and install_step is None
                and submission_sim_time_s is not None
                and simulation_time_s - submission_sim_time_s
                >= ONLINE_POST_INSTALL_DURATION_S
            ):
                break

            validating_candidate = (
                pending_controller is not None and not candidate_deferred_for_tick
            )
            controller_for_step = (
                pending_controller if validating_candidate else active_controller
            )
            reference_for_step = (
                pending_reference if validating_candidate else active_reference
            )
            warm_start_for_step = (
                template_hover_warm.warm_start
                if validating_candidate
                else active_warm_start
            )
            saved_active_warm_start = active_warm_start
            validation_started_at = time.perf_counter()
            result = controller_for_step.solve(
                jnp.asarray(states[-1]),
                reference_for_step,
                jnp.asarray(previous),
                applied_command=jnp.asarray(sample.applied_motor_thrust_fraction),
                warm_start=warm_start_for_step,
                deadline_s=sample_period_s,
            )
            command_mode = active_mode
            if validating_candidate:
                validation_wall_time_s = time.perf_counter() - validation_started_at
                assert worker_update is not None
                if result.command_usable:
                    active_controller = controller_for_step
                    active_reference = reference_for_step
                    active_warm_start = result.warm_start
                    active_mode = "adapted"
                    command_mode = "adapted"
                    install_step = index
                    install_sim_time_s = simulation_time_s
                    install_wall_time_s = time.perf_counter() - control_loop_started_at
                    candidate_ready_wall_time_s = install_wall_time_s
                    prepared = _PreparedOnlineController(
                        update_report=worker_update.update_report,
                        update_wall_time_s=worker_update.update_wall_time_s,
                        controller_validation_wall_time_s=validation_wall_time_s,
                        total_wall_time_s=(
                            worker_update.update_wall_time_s + validation_wall_time_s
                        ),
                        validation_statuses=(result.status.value,),
                        ready_for_install=True,
                        rejection_reason=None,
                    )
                else:
                    active_warm_start = saved_active_warm_start
                    command_mode = "candidate_rejected"
                    reason = (
                        "first adapted solve was not command-usable: "
                        f"{result.status.value}: {result.message}"
                    )
                    prepared = _PreparedOnlineController(
                        update_report=worker_update.update_report,
                        update_wall_time_s=worker_update.update_wall_time_s,
                        controller_validation_wall_time_s=validation_wall_time_s,
                        total_wall_time_s=(
                            worker_update.update_wall_time_s + validation_wall_time_s
                        ),
                        validation_statuses=(result.status.value,),
                        ready_for_install=False,
                        rejection_reason=reason,
                    )
                    candidate_ready_wall_time_s = (
                        time.perf_counter() - control_loop_started_at
                    )
                pending_controller = None
                pending_reference = None
            else:
                active_warm_start = result.warm_start
            command_generated_at_s = time.perf_counter()
            fault_this_step = (
                install_step is not None
                and not supervisor_fault_injected
                and index - install_step == 10
            )
            supervisor_command_timestamp_s = command_generated_at_s
            if fault_this_step:
                supervisor_command_timestamp_s -= (
                    2.0 * supervisor_config.maximum_command_age_s
                )
            supervisor_started_at = time.perf_counter()
            supervised = supervisor.supervise(
                state=states[-1],
                state_received_at_s=step_started_at,
                candidate_command=result.command,
                command_generated_at_s=supervisor_command_timestamp_s,
                now_s=time.perf_counter(),
                controller_command_usable=result.command_usable,
                previous_applied_command=sample.applied_motor_thrust_fraction,
            )
            supervisor_decision_times.append(
                time.perf_counter() - supervisor_started_at
            )
            supervisor_modes.append(supervised.mode.value)
            supervisor_reasons.extend(reason.value for reason in supervised.reasons)
            supervisor_intervention_count += int(supervised.intervened)
            if fault_this_step:
                supervisor_fault_injected = True
                supervisor_fault_step = index
                supervisor_fault_response = supervised.to_dict()
            command = np.asarray(supervised.command)
            sample = plant.step(command)
            controls.append(command)
            states.append(np.asarray(sample.state))
            applied.append(np.asarray(sample.applied_motor_thrust_fraction))
            state_validity.append(
                float(
                    np.max(
                        np.asarray(
                            stale_controller.model.validity_utilization(
                                jnp.asarray(sample.state)
                            )
                        )
                    )
                )
            )
            control_modes.append(command_mode)
            previous = command
            solve_times.append(result.diagnostics.solve_time_s)
            solve_statuses.append(result.status.value)
            solve_messages.append(result.message)
            support_modes.append(result.diagnostics.support_filter_mode.value)
            support_applied += int(result.diagnostics.support_filter_applied)
            fallback_count += int(result.used_fallback)
            diagnostic_values = (
                result.diagnostics.maximum_validity_utilization,
                result.diagnostics.support_horizon_maximum_robust_validity_utilization,
            )
            nonfinite_diagnostic_count += sum(
                not np.isfinite(value) for value in diagnostic_values
            )
            if np.isfinite(diagnostic_values[0]):
                predicted_validity.append(float(diagnostic_values[0]))
            if np.isfinite(diagnostic_values[1]):
                support_validity.append(float(diagnostic_values[1]))

            eligible_start = _eligible_recovery_evidence_start(
                state_validity,
                evidence_steps,
            )
            if candidate_future is None and eligible_start is not None:
                recent_validity = state_validity[eligible_start:]
                evidence = _online_recovery_trajectory(
                    belief,
                    states[eligible_start:],
                    controls[eligible_start:],
                    applied[eligible_start:],
                )
                submission_wall_time = time.perf_counter()
                submission_sim_time_s = len(controls) * sample_period_s
                evidence_start_sim_time_s = eligible_start * sample_period_s
                evidence_maximum_validity = max(recent_validity)
                candidate_future = executor.submit(
                    _run_isolated_belief_update,
                    belief,
                    evidence,
                )
                worker_suspended = _set_process_suspended(
                    worker_prewarm.process_id,
                    suspended=True,
                )
                worker_suspend_count += int(worker_suspended)

            step_elapsed_s = time.perf_counter() - step_started_at
            outer_step_times.append(step_elapsed_s)
            if step_elapsed_s > sample_period_s:
                outer_deadline_miss_count += 1
                outer_deadline_miss_steps.append(index)
                outer_deadline_miss_details.append(
                    {
                        "step": index,
                        "outer_step_time_s": step_elapsed_s,
                        "solver_reported_time_s": result.diagnostics.solve_time_s,
                        "solver_status": result.status.value,
                        "control_mode": command_mode,
                        "worker_suspended": worker_suspended,
                        "worker_resolution_wall_time_s": (
                            worker_resolution_wall_time_s
                        ),
                        "validating_candidate": validating_candidate,
                    }
                )

            command_deadline = control_loop_started_at + (
                len(controls) * sample_period_s
            )
            absolute_lateness_s = max(
                0.0,
                time.perf_counter() - command_deadline,
            )
            absolute_completion_lateness.append(absolute_lateness_s)
            if absolute_lateness_s > 0.0:
                absolute_deadline_miss_steps.append(index)

            if worker_suspended:
                resumed = _set_process_suspended(
                    worker_prewarm.process_id,
                    suspended=False,
                )
                worker_resume_count += int(resumed)

            target_wall_time = control_loop_started_at + (
                len(controls) * sample_period_s
            )
            remaining_s = target_wall_time - time.perf_counter()
            if remaining_s > 0.0:
                time.sleep(remaining_s)
                loop_lag.append(0.0)
            else:
                loop_lag.append(-remaining_s)
    finally:
        control_loop_ended_at = time.perf_counter()
        plant.close()
        _set_process_suspended(worker_prewarm.process_id, suspended=False)
        executor.shutdown(wait=True)

    control_loop_wall_time_s = control_loop_ended_at - control_loop_started_at
    completed_after_control_loop = False
    if candidate_future is not None and not future_consumed:
        completed_after_control_loop = True
        future_consumed = True
        try:
            worker_update = candidate_future.result()
            reason = "isolated update completed after the fixed control window"
            prepared = _PreparedOnlineController(
                update_report=worker_update.update_report,
                update_wall_time_s=worker_update.update_wall_time_s,
                controller_validation_wall_time_s=None,
                total_wall_time_s=worker_update.update_wall_time_s,
                validation_statuses=(),
                ready_for_install=False,
                rejection_reason=reason,
            )
        except Exception as error:
            worker_error = f"{type(error).__name__}: {error}"
            reason = f"isolated update failed: {worker_error}"
            prepared = _PreparedOnlineController(
                update_report={"applied": False, "reason": reason},
                update_wall_time_s=0.0,
                controller_validation_wall_time_s=None,
                total_wall_time_s=0.0,
                validation_statuses=(),
                ready_for_install=False,
                rejection_reason=reason,
            )
    if prepared is None:
        reason = (
            "no contiguous evidence window remained inside the learned validity "
            "envelope"
        )
        prepared = _PreparedOnlineController(
            update_report={"applied": False, "reason": reason},
            update_wall_time_s=0.0,
            controller_validation_wall_time_s=None,
            total_wall_time_s=0.0,
            validation_statuses=(),
            ready_for_install=False,
            rejection_reason=reason,
        )

    states_array = np.asarray(states)
    commands_array = np.asarray(controls)
    references = np.repeat(reference_state[None, :], len(states_array), axis=0)
    errors = np.asarray(
        jax.vmap(rigid_body_local_error)(
            jnp.asarray(references),
            jnp.asarray(states_array),
        )
    )
    scale = np.asarray(stale_controller.tolerances.local_state_scale)
    normalized = errors / scale
    pre_install_stop = len(states_array) if install_step is None else install_step + 1
    pre_install_normalized = normalized[:pre_install_stop]
    post_install_normalized = (
        np.empty((0, normalized.shape[1]))
        if install_step is None
        else normalized[install_step:]
    )
    actual_validity = np.asarray(state_validity)
    minimum = np.asarray(stale_controller.model.command_minimum)
    maximum = np.asarray(stale_controller.model.command_maximum)
    command_violation = max(
        float(np.max(minimum - commands_array)),
        float(np.max(commands_array - maximum)),
        0.0,
    )
    submission_to_ready_s = None
    if submission_wall_time is not None:
        if candidate_ready_wall_time_s is not None:
            submission_to_ready_s = (
                control_loop_started_at
                + candidate_ready_wall_time_s
                - submission_wall_time
            )
    submission_to_install_s = (
        None
        if submission_wall_time is None or install_wall_time_s is None
        else (control_loop_started_at + install_wall_time_s - submission_wall_time)
    )
    stale_after_submission_count = 0
    if submission_sim_time_s is not None:
        submission_step = round(submission_sim_time_s / sample_period_s)
        stale_after_submission_count = sum(
            mode == "stale" for mode in control_modes[submission_step:]
        )
    return {
        "condition": "online_stale_to_adapted",
        "evidence": {
            "duration_s": ONLINE_EVIDENCE_DURATION_S,
            "sample_count": evidence_steps + 1,
            "submitted": submission_wall_time is not None,
            "eligibility_rule": (
                "first contiguous window with validity utilization <= 1.0"
            ),
            "start_sim_time_s": evidence_start_sim_time_s,
            "maximum_validity_utilization": evidence_maximum_validity,
            "excluded_prefix_state_count": (
                None
                if evidence_start_sim_time_s is None
                else round(evidence_start_sim_time_s / sample_period_s)
            ),
            "excluded_prefix_maximum_validity_utilization": (
                None
                if evidence_start_sim_time_s in (None, 0.0)
                else max(
                    state_validity[: round(evidence_start_sim_time_s / sample_period_s)]
                )
            ),
            "source_group": "unknown-target-online-recovery",
            "collected_under_controller": "stale",
            "hidden_target_configuration_supplied": False,
            "applied_actuator_state_retained": True,
        },
        "update": prepared.update_report,
        "candidate": {
            "ready_for_install": prepared.ready_for_install,
            "shared_precompiled_parameterized_kernels": True,
            "isolated_adaptation_process": True,
            "rejection_reason": prepared.rejection_reason,
            "validation_statuses": list(prepared.validation_statuses),
            "completed_after_control_loop": completed_after_control_loop,
            "worker_process_id": (
                None if worker_update is None else worker_update.process_id
            ),
            "worker_process_niceness": (
                None if worker_update is None else worker_update.process_niceness
            ),
            "worker_error": worker_error,
        },
        "supervisor": {
            "included": True,
            "independent_of_fitted_dynamics": True,
            "configuration": supervisor_config.to_dict(),
            "mode_counts": dict(sorted(Counter(supervisor_modes).items())),
            "reason_counts": dict(sorted(Counter(supervisor_reasons).items())),
            "intervention_count": supervisor_intervention_count,
            "maximum_decision_wall_time_s": max(
                supervisor_decision_times,
                default=0.0,
            ),
            "fault_injection": {
                "injected": supervisor_fault_injected,
                "type": "stale_adapted_command_timestamp",
                "step": supervisor_fault_step,
                "expected_mode": SupervisorMode.RATE_ARREST.value,
                "expected_reason": SupervisorReason.COMMAND_STALE.value,
                "response": supervisor_fault_response,
                "handled": bool(
                    supervisor_fault_response is not None
                    and supervisor_fault_response["mode"]
                    == SupervisorMode.RATE_ARREST.value
                    and SupervisorReason.COMMAND_STALE.value
                    in supervisor_fault_response["reasons"]
                ),
            },
        },
        "isolation": {
            "start_method": "spawn",
            "numerical_payload_only": True,
            "control_process_id": os.getpid(),
            "worker_process_id": worker_prewarm.process_id,
            "separate_process": worker_prewarm.process_id != os.getpid(),
            "worker_process_niceness": worker_prewarm.process_niceness,
            "lower_scheduling_priority_requested": True,
            "temporal_partition_available": temporal_partition_available,
            "worker_suspend_count": worker_suspend_count,
            "worker_resume_count": worker_resume_count,
            "worker_runs_only_in_control_slack": temporal_partition_available,
            "worker_environment": worker_environment,
            "prewarm_known_fleet_only": True,
            "prewarm_hidden_target_configuration_supplied": False,
            "prewarm_update": worker_prewarm.update_report,
        },
        "handoff": {
            "atomic_install_at_control_boundary": install_step is not None,
            "submission_sim_time_s": submission_sim_time_s,
            "candidate_resolved_sim_time_s": candidate_resolved_sim_time_s,
            "install_sim_time_s": install_sim_time_s,
            "install_step": install_step,
            "stale_steps_after_submission": stale_after_submission_count,
            "control_continued_during_candidate_preparation": (
                stale_after_submission_count > 0
            ),
        },
        "timing": {
            "control_period_s": sample_period_s,
            "stale_controller_prewarm_wall_time_s": stale_prewarm_wall_time_s,
            "candidate_template_prewarm_wall_time_s": (
                candidate_template_prewarm_wall_time_s
            ),
            "worker_process_prewarm_wall_time_s": worker_prewarm_wall_time_s,
            "worker_prewarm_update_wall_time_s": (worker_prewarm.update_wall_time_s),
            "worker_prewarm_update_cpu_time_s": (worker_prewarm.update_cpu_time_s),
            "integrated_control_plant_prewarm_wall_time_s": (
                integrated_prewarm_wall_time_s
            ),
            "belief_update_wall_time_s": prepared.update_wall_time_s,
            "belief_update_cpu_time_s": (
                None if worker_update is None else worker_update.update_cpu_time_s
            ),
            "candidate_first_solve_validation_wall_time_s": (
                prepared.controller_validation_wall_time_s
            ),
            "submission_to_worker_result_wall_time_s": (
                None
                if submission_wall_time is None or worker_result_wall_time_s is None
                else (
                    control_loop_started_at
                    + worker_result_wall_time_s
                    - submission_wall_time
                )
            ),
            "submission_to_candidate_ready_wall_time_s": submission_to_ready_s,
            "submission_to_install_wall_time_s": submission_to_install_s,
            "control_loop_wall_time_s": control_loop_wall_time_s,
            "outer_control_step_median_s": float(np.median(outer_step_times)),
            "outer_control_step_p90_s": float(np.quantile(outer_step_times, 0.9)),
            "outer_control_step_maximum_s": max(outer_step_times),
            "solver_reported_median_s": float(np.median(solve_times)),
            "solver_reported_p90_s": float(np.quantile(solve_times, 0.9)),
            "solver_reported_maximum_s": max(solve_times),
            "outer_deadline_miss_count": outer_deadline_miss_count,
            "outer_deadline_miss_steps": outer_deadline_miss_steps,
            "outer_deadline_miss_details": outer_deadline_miss_details,
            "absolute_deadline_miss_count": len(absolute_deadline_miss_steps),
            "absolute_deadline_miss_steps": absolute_deadline_miss_steps,
            "maximum_absolute_completion_lateness_s": max(
                absolute_completion_lateness,
                default=0.0,
            ),
            "maximum_schedule_lag_s": max(loop_lag, default=0.0),
        },
        "control_mode_counts": dict(sorted(Counter(control_modes).items())),
        "solve_status_counts": dict(sorted(Counter(solve_statuses).items())),
        "solve_message_counts": dict(sorted(Counter(solve_messages).items())),
        "support_filter_mode_counts": dict(sorted(Counter(support_modes).items())),
        "support_filter_applied_count": support_applied,
        "fallback_count": fallback_count,
        "nonfinite_diagnostic_count": nonfinite_diagnostic_count,
        "normalized_tracking_rms": _normalized_rms(normalized),
        "pre_install_normalized_tracking_rms": _normalized_rms(pre_install_normalized),
        "post_install_normalized_tracking_rms": _normalized_rms(
            post_install_normalized
        ),
        "pre_install_normalized_attitude_rate_rms": _normalized_rms(
            pre_install_normalized[:, 6:12]
        ),
        "post_install_normalized_attitude_rate_rms": _normalized_rms(
            post_install_normalized[:, 6:12]
        ),
        "terminal_attitude_error_rad": float(np.linalg.norm(errors[-1, 6:9])),
        "terminal_angular_velocity_error_rad_s": float(
            np.linalg.norm(errors[-1, 9:12])
        ),
        "maximum_actual_validity_utilization": float(np.max(actual_validity)),
        "maximum_predicted_validity_utilization": (
            max(predicted_validity) if predicted_validity else None
        ),
        "maximum_support_horizon_robust_validity_utilization": (
            max(support_validity) if support_validity else None
        ),
        "maximum_command_bound_violation": command_violation,
        "finite": bool(
            np.all(np.isfinite(states_array))
            and np.all(np.isfinite(commands_array))
            and np.all(np.isfinite(normalized))
        ),
    }
