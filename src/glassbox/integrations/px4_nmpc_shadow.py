"""PX4 telemetry-to-NMPC shadow runner that never transmits commands."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.integrations.px4 import (
    PX4AppliedCommandSample,
    PX4MavlinkStateSource,
    PX4StateSample,
    PX4TelemetryError,
    px4_boot_time_skew_s,
)
from glassbox.nmpc import NMPCController, NMPCWarmStart
from glassbox.runtime import RuntimeDynamicsModel
from glassbox.streaming_evaluation import StreamingOneStepEvaluator

_BOOT_TIME_MODULUS_MS = 2**32
_MAXIMUM_APPLIED_COMMAND_STATE_SKEW_S = 0.10


class AppliedCommandSource(Protocol):
    """Read-only source of canonical commands actually applied by the vehicle."""

    def sample_nearest(
        self,
        time_boot_ms: int,
        *,
        timeout_s: float = 1.0,
    ) -> PX4AppliedCommandSample: ...


def _command(value: str, *, expected_size: int) -> np.ndarray:
    try:
        command = np.asarray([float(item) for item in value.split(",")])
    except ValueError as error:
        raise ValueError("previous command must be comma-separated numbers") from error
    if command.shape != (expected_size,) or not np.all(np.isfinite(command)):
        raise ValueError(f"previous command must contain {expected_size} finite values")
    return command


def _held_reference(controller: NMPCController, state: np.ndarray):
    exogenous = np.zeros(controller.model.exogenous_size, dtype=np.float64)
    return controller.hold_reference(state, exogenous=exogenous)


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _optional_max(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return max(values) if values else None


def _source_clock_advance_s(first_ms: int, last_ms: int) -> float | None:
    advance_ms = (
        last_ms - first_ms + _BOOT_TIME_MODULUS_MS // 2
    ) % _BOOT_TIME_MODULUS_MS - _BOOT_TIME_MODULUS_MS // 2
    return advance_ms * 1e-3 if advance_ms >= 0 else None


def _clock_ratio(source_advance_s: float | None, host_elapsed_s: float) -> float | None:
    if source_advance_s is None or host_elapsed_s <= 0.0:
        return None
    return source_advance_s / host_elapsed_s


def _solve_row(
    controller: NMPCController,
    sample: PX4StateSample,
    applied_command: np.ndarray,
    warm_start: NMPCWarmStart | None,
    *,
    applied_sample: PX4AppliedCommandSample | None = None,
    deadline_s: float | None = None,
) -> tuple[dict[str, Any], NMPCWarmStart | None]:
    applied_command_state_skew_s = (
        None
        if applied_sample is None
        else px4_boot_time_skew_s(
            sample.position_time_boot_ms,
            (applied_sample.source_time_us // 1_000) % _BOOT_TIME_MODULUS_MS,
        )
    )
    maximum_applied_command_state_skew_s = min(
        _MAXIMUM_APPLIED_COMMAND_STATE_SKEW_S,
        controller.model.runtime_spec.sample_period_s,
    )
    if (
        applied_command_state_skew_s is not None
        and applied_command_state_skew_s > maximum_applied_command_state_skew_s + 1e-12
    ):
        raise PX4TelemetryError(
            "PX4 state and applied-command telemetry exceed the "
            f"{maximum_applied_command_state_skew_s * 1_000:g} ms alignment limit"
        )
    result = controller.solve(
        jnp.asarray(sample.state),
        _held_reference(controller, sample.state),
        jnp.asarray(applied_command),
        applied_command=jnp.asarray(applied_command),
        warm_start=warm_start,
        deadline_s=deadline_s,
    )
    current_validity = np.asarray(
        controller.model.validity_utilization(
            jnp.asarray(sample.state),
            jnp.zeros(controller.model.exogenous_size),
        )
    )
    row = {
        "position_time_boot_ms": sample.position_time_boot_ms,
        "attitude_time_boot_ms": sample.attitude_time_boot_ms,
        "message_skew_s": sample.message_skew_s,
        "maximum_receive_age_s": sample.maximum_receive_age_s,
        "estimated_source_clock_lag_s": sample.estimated_source_clock_lag_s,
        "state": sample.state.tolist(),
        "current_maximum_validity_utilization": float(np.max(current_validity)),
        "status": result.status.value,
        "command_usable": result.command_usable,
        "used_fallback": result.used_fallback,
        "solve_time_s": result.diagnostics.solve_time_s,
        "iterations": result.diagnostics.iterations,
        "maximum_predicted_validity_utilization": _finite_or_none(
            result.diagnostics.maximum_validity_utilization
        ),
        "maximum_command_bound_violation": (
            result.diagnostics.maximum_command_bound_violation
        ),
        "support_filter_mode": result.diagnostics.support_filter_mode.value,
        "support_filter_applied": result.diagnostics.support_filter_applied,
        "support_command_fraction": result.diagnostics.support_command_fraction,
        "next_step_mean_validity_utilization": _finite_or_none(
            result.diagnostics.next_step_mean_validity_utilization
        ),
        "next_step_robust_validity_utilization": _finite_or_none(
            result.diagnostics.next_step_robust_validity_utilization
        ),
        "current_angular_rate_energy": _finite_or_none(
            result.diagnostics.current_angular_rate_energy
        ),
        "next_step_angular_rate_energy": _finite_or_none(
            result.diagnostics.next_step_angular_rate_energy
        ),
        "support_horizon_s": result.diagnostics.support_horizon_s,
        "support_horizon_maximum_robust_validity_utilization": _finite_or_none(
            result.diagnostics.support_horizon_maximum_robust_validity_utilization
        ),
        "support_horizon_terminal_robust_validity_utilization": _finite_or_none(
            result.diagnostics.support_horizon_terminal_robust_validity_utilization
        ),
        "support_horizon_terminal_angular_rate_energy": _finite_or_none(
            result.diagnostics.support_horizon_terminal_angular_rate_energy
        ),
        "applied_command": applied_command.tolist(),
        "applied_command_source_time_us": (
            None if applied_sample is None else applied_sample.source_time_us
        ),
        "applied_command_state_skew_s": applied_command_state_skew_s,
        "applied_command_receive_age_s": (
            None if applied_sample is None else applied_sample.receive_age_s
        ),
        "applied_command_armed": (
            None if applied_sample is None else applied_sample.armed
        ),
        "applied_command_mav_mode": (
            None if applied_sample is None else applied_sample.mav_mode
        ),
        "shadow_command": np.asarray(result.command).tolist(),
    }
    return row, result.warm_start


def _validated_command(model: RuntimeDynamicsModel, command: np.ndarray) -> np.ndarray:
    command = np.asarray(command, dtype=np.float64)
    expected_shape = (model.command_size,)
    if command.shape != expected_shape or not np.all(np.isfinite(command)):
        raise ValueError(
            f"applied command must have shape {expected_shape} and be finite"
        )
    minimum = np.asarray(model.command_minimum)
    maximum = np.asarray(model.command_maximum)
    if np.any(command < minimum) or np.any(command > maximum):
        raise ValueError("applied command must lie inside the artifact bounds")
    return command


def _next_applied_command(
    source: AppliedCommandSource,
    model: RuntimeDynamicsModel,
    *,
    time_boot_ms: int,
    timeout_s: float,
) -> tuple[np.ndarray, PX4AppliedCommandSample]:
    sample = source.sample_nearest(time_boot_ms, timeout_s=timeout_s)
    return _validated_command(model, sample.command), sample


def run_px4_nmpc_shadow(
    source: PX4MavlinkStateSource,
    model: RuntimeDynamicsModel,
    previous_command: np.ndarray | None = None,
    *,
    applied_command_source: AppliedCommandSource | None = None,
    sample_count: int = 10,
    telemetry_timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Run an artifact against live PX4 state without sending its commands."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if applied_command_source is None and previous_command is None:
        raise ValueError(
            "provide either a fixed previous_command or an applied_command_source"
        )
    if applied_command_source is not None and previous_command is not None:
        raise ValueError(
            "previous_command and applied_command_source are mutually exclusive"
        )
    fixed_command = (
        None
        if previous_command is None
        else _validated_command(model, previous_command)
    )

    controller = NMPCController(model)
    one_step_evaluator = StreamingOneStepEvaluator(model)
    first_sample = source.next_sample(timeout_s=telemetry_timeout_s)
    if applied_command_source is None:
        if fixed_command is None:  # pragma: no cover - guarded above
            raise RuntimeError("fixed applied command was not initialized")
        first_command = fixed_command
        first_applied_sample = None
    else:
        first_command, first_applied_sample = _next_applied_command(
            applied_command_source,
            model,
            time_boot_ms=first_sample.position_time_boot_ms,
            timeout_s=telemetry_timeout_s,
        )
    one_step_evaluator.observe(
        first_sample.state,
        first_command,
        elapsed_s=0.0,
    )
    audit_elapsed_s = 0.0
    audit_position_boot_ms = first_sample.position_time_boot_ms
    first_sample_host_s = time.monotonic()
    cold_row, warm_start = _solve_row(
        controller,
        first_sample,
        first_command,
        None,
        applied_sample=first_applied_sample,
    )
    warm_row, warm_start = _solve_row(
        controller,
        first_sample,
        first_command,
        warm_start,
        applied_sample=first_applied_sample,
    )

    rows: list[dict[str, Any]] = []
    sample_host_times_s: list[float] = []
    model_period_s = model.runtime_spec.sample_period_s
    for _ in range(sample_count):
        sample = source.next_sample(timeout_s=telemetry_timeout_s)
        if applied_command_source is None:
            if fixed_command is None:  # pragma: no cover - guarded above
                raise RuntimeError("fixed applied command was not initialized")
            applied_command = fixed_command
            applied_sample = None
        else:
            applied_command, applied_sample = _next_applied_command(
                applied_command_source,
                model,
                time_boot_ms=sample.position_time_boot_ms,
                timeout_s=telemetry_timeout_s,
            )
        sample_host_s = time.monotonic()
        audit_advance_s = _source_clock_advance_s(
            audit_position_boot_ms,
            sample.position_time_boot_ms,
        )
        if audit_advance_s is not None:
            audit_elapsed_s += audit_advance_s
        one_step_audit = one_step_evaluator.observe(
            sample.state,
            applied_command,
            elapsed_s=audit_elapsed_s,
        )
        audit_position_boot_ms = sample.position_time_boot_ms
        row, warm_start = _solve_row(
            controller,
            sample,
            applied_command,
            warm_start,
            applied_sample=applied_sample,
            deadline_s=model_period_s,
        )
        row["one_step_model_audit"] = one_step_audit
        row["sample_host_elapsed_s"] = sample_host_s - first_sample_host_s
        rows.append(row)
        sample_host_times_s.append(sample_host_s)

    solve_times = np.asarray([row["solve_time_s"] for row in rows])
    warmup_host_elapsed_s = sample_host_times_s[0] - first_sample_host_s
    warmup_source_advance_s = _source_clock_advance_s(
        first_sample.position_time_boot_ms,
        int(rows[0]["position_time_boot_ms"]),
    )
    sample_host_elapsed_s = sample_host_times_s[-1] - sample_host_times_s[0]
    sample_source_advance_s = _source_clock_advance_s(
        int(rows[0]["position_time_boot_ms"]),
        int(rows[-1]["position_time_boot_ms"]),
    )
    applied_commands = np.asarray([row["applied_command"] for row in rows])
    return {
        "schema_version": 5,
        "mode": "read_only_shadow",
        "commands_transmitted": False,
        "applied_command_source": (
            "fixed" if applied_command_source is None else "telemetry"
        ),
        "platform": model.input_spec.vehicle.family,
        "model_sample_period_s": model_period_s,
        "prediction_horizon_s": controller.prediction_horizon_s,
        "command_roles": [channel.role for channel in model.actuation.command_channels],
        "initial_applied_command": first_command.tolist(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "warmup": {"cold": cold_row, "warm": warm_row},
        "samples": rows,
        "summary": {
            "sample_count": len(rows),
            "usable_command_count": sum(row["command_usable"] for row in rows),
            "fallback_count": sum(row["used_fallback"] for row in rows),
            "support_filter_mode_counts": dict(
                sorted(Counter(row["support_filter_mode"] for row in rows).items())
            ),
            "support_filter_applied_count": sum(
                row["support_filter_applied"] for row in rows
            ),
            "support_best_effort_count": sum(
                row["support_filter_mode"].endswith("best_effort") for row in rows
            ),
            "model_period_deadline_miss_count": int(
                np.count_nonzero(solve_times > model_period_s)
            ),
            "solve_time_median_s": float(np.median(solve_times)),
            "solve_time_p90_s": float(np.quantile(solve_times, 0.9)),
            "solve_time_maximum_s": float(np.max(solve_times)),
            "maximum_message_skew_s": max(row["message_skew_s"] for row in rows),
            "maximum_estimated_source_clock_lag_s": max(
                row["estimated_source_clock_lag_s"] for row in rows
            ),
            "maximum_applied_command_state_skew_s": _optional_max(
                rows, "applied_command_state_skew_s"
            ),
            "maximum_applied_command_receive_age_s": _optional_max(
                rows, "applied_command_receive_age_s"
            ),
            "all_applied_command_samples_armed": (
                None
                if applied_command_source is None
                else all(row["applied_command_armed"] for row in rows)
            ),
            "applied_command_peak_to_peak": np.ptp(applied_commands, axis=0).tolist(),
            "one_step_model_audit": one_step_evaluator.summary(),
            "warmup_host_elapsed_s": warmup_host_elapsed_s,
            "warmup_source_clock_advance_s": warmup_source_advance_s,
            "warmup_source_clock_realtime_ratio": _clock_ratio(
                warmup_source_advance_s, warmup_host_elapsed_s
            ),
            "sample_host_elapsed_s": sample_host_elapsed_s,
            "sample_source_clock_advance_s": sample_source_advance_s,
            "sample_source_clock_realtime_ratio": _clock_ratio(
                sample_source_advance_s, sample_host_elapsed_s
            ),
            "maximum_current_validity_utilization": max(
                row["current_maximum_validity_utilization"] for row in rows
            ),
            "maximum_predicted_validity_utilization": _optional_max(
                rows, "maximum_predicted_validity_utilization"
            ),
            "maximum_next_step_robust_validity_utilization": _optional_max(
                rows, "next_step_robust_validity_utilization"
            ),
            "maximum_support_horizon_robust_validity_utilization": _optional_max(
                rows,
                "support_horizon_maximum_robust_validity_utilization",
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fitted Glassbox model against passive PX4 MAVLink telemetry. "
            "No command messages are transmitted."
        )
    )
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--previous-command",
        required=True,
        help="comma-separated command currently applied to the vehicle",
    )
    parser.add_argument(
        "--connection",
        default="udpin:0.0.0.0:14550",
        help="passive pymavlink connection string",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    model = RuntimeDynamicsModel.load(args.model)
    previous_command = _command(args.previous_command, expected_size=model.command_size)
    with PX4MavlinkStateSource.connect(args.connection) as source:
        report = run_px4_nmpc_shadow(
            source,
            model,
            previous_command,
            sample_count=args.samples,
        )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
