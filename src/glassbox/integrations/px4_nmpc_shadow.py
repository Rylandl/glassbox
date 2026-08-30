"""PX4 telemetry-to-NMPC shadow runner that never transmits commands."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.integrations.px4 import PX4MavlinkStateSource, PX4StateSample
from glassbox.nmpc import NMPCController, NMPCWarmStart
from glassbox.runtime import RuntimeDynamicsModel

_BOOT_TIME_MODULUS_MS = 2**32


def _command(value: str, *, expected_size: int) -> np.ndarray:
    try:
        command = np.asarray([float(item) for item in value.split(",")])
    except ValueError as error:
        raise ValueError("previous command must be comma-separated numbers") from error
    if command.shape != (expected_size,) or not np.all(np.isfinite(command)):
        raise ValueError(
            f"previous command must contain {expected_size} finite values"
        )
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
        (last_ms - first_ms + _BOOT_TIME_MODULUS_MS // 2)
        % _BOOT_TIME_MODULUS_MS
        - _BOOT_TIME_MODULUS_MS // 2
    )
    return advance_ms * 1e-3 if advance_ms >= 0 else None


def _clock_ratio(source_advance_s: float | None, host_elapsed_s: float) -> float | None:
    if source_advance_s is None or host_elapsed_s <= 0.0:
        return None
    return source_advance_s / host_elapsed_s


def _solve_row(
    controller: NMPCController,
    sample: PX4StateSample,
    previous_command: np.ndarray,
    warm_start: NMPCWarmStart | None,
    *,
    deadline_s: float | None = None,
) -> tuple[dict[str, Any], NMPCWarmStart | None]:
    result = controller.solve(
        jnp.asarray(sample.state),
        _held_reference(controller, sample.state),
        jnp.asarray(previous_command),
        applied_command=jnp.asarray(previous_command),
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
        "shadow_command": np.asarray(result.command).tolist(),
    }
    return row, result.warm_start


def run_px4_nmpc_shadow(
    source: PX4MavlinkStateSource,
    model: RuntimeDynamicsModel,
    previous_command: np.ndarray,
    *,
    sample_count: int = 10,
    telemetry_timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Run an artifact against live PX4 state without sending its commands."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    previous_command = np.asarray(previous_command, dtype=np.float64)
    expected_shape = (model.command_size,)
    if previous_command.shape != expected_shape or not np.all(
        np.isfinite(previous_command)
    ):
        raise ValueError(
            f"previous_command must have shape {expected_shape} and be finite"
        )
    minimum = np.asarray(model.command_minimum)
    maximum = np.asarray(model.command_maximum)
    if np.any(previous_command < minimum) or np.any(previous_command > maximum):
        raise ValueError("previous_command must lie inside the artifact bounds")

    controller = NMPCController(model)
    first_sample = source.next_sample(timeout_s=telemetry_timeout_s)
    first_sample_host_s = time.monotonic()
    cold_row, warm_start = _solve_row(
        controller, first_sample, previous_command, None
    )
    warm_row, warm_start = _solve_row(
        controller, first_sample, previous_command, warm_start
    )

    rows: list[dict[str, Any]] = []
    sample_host_times_s: list[float] = []
    model_period_s = model.runtime_spec.sample_period_s
    for _ in range(sample_count):
        sample = source.next_sample(timeout_s=telemetry_timeout_s)
        sample_host_s = time.monotonic()
        row, warm_start = _solve_row(
            controller,
            sample,
            previous_command,
            warm_start,
            deadline_s=model_period_s,
        )
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
    return {
        "schema_version": 2,
        "mode": "read_only_shadow",
        "commands_transmitted": False,
        "platform": model.input_spec.vehicle.family,
        "model_sample_period_s": model_period_s,
        "prediction_horizon_s": controller.prediction_horizon_s,
        "command_roles": [
            channel.role for channel in model.actuation.command_channels
        ],
        "previous_applied_command": previous_command.tolist(),
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
            "model_period_deadline_miss_count": int(
                np.count_nonzero(solve_times > model_period_s)
            ),
            "solve_time_median_s": float(np.median(solve_times)),
            "solve_time_p90_s": float(np.quantile(solve_times, 0.9)),
            "solve_time_maximum_s": float(np.max(solve_times)),
            "maximum_message_skew_s": max(
                row["message_skew_s"] for row in rows
            ),
            "maximum_estimated_source_clock_lag_s": max(
                row["estimated_source_clock_lag_s"] for row in rows
            ),
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
    previous_command = _command(
        args.previous_command, expected_size=model.command_size
    )
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
