"""Command-line synthetic parameter-recovery demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from glassbox.data import trajectory_windows
from glassbox.evaluation import parameter_dict, rollout_metrics
from glassbox.identification import fit_dynamics
from glassbox.synthetic import (
    generate_trajectory,
    initial_parameter_guess,
    true_parameters,
)


def run_demo(
    *,
    train_flights: int = 3,
    duration_s: float = 6.0,
    dt_s: float = 0.02,
    horizon: int = 25,
    steps: int = 400,
) -> dict[str, Any]:
    """Run synthetic recovery and return parameters, losses, and test metrics."""

    if train_flights < 1:
        raise ValueError("train_flights must be positive")

    training = [
        generate_trajectory(seed=seed, duration_s=duration_s, dt_s=dt_s)
        for seed in range(train_flights)
    ]
    held_out = generate_trajectory(seed=10_000, duration_s=duration_s, dt_s=dt_s)
    windows = trajectory_windows(training, horizon=horizon, stride=horizon)

    initial_params = initial_parameter_guess()
    start_time = perf_counter()
    fit = fit_dynamics(windows, initial_params, steps=steps)
    fit_time_s = perf_counter() - start_time

    return {
        "configuration": {
            "train_flights": train_flights,
            "duration_s": duration_s,
            "dt_s": dt_s,
            "rollout_horizon_steps": horizon,
            "optimization_steps": steps,
            "training_windows": len(windows.initial_states),
        },
        "parameters": {
            "true": parameter_dict(true_parameters()),
            "initial": parameter_dict(initial_params),
            "fitted": parameter_dict(fit.params),
        },
        "fit": {
            "initial_loss": fit.initial_loss,
            "final_loss": fit.final_loss,
            "loss_reduction": fit.initial_loss / fit.final_loss,
            "wall_time_s": fit_time_s,
        },
        "held_out_rollout": {
            "initial": rollout_metrics(initial_params, held_out),
            "fitted": rollout_metrics(fit.params, held_out),
            "true": rollout_metrics(true_parameters(), held_out),
        },
    }


def _print_summary(result: dict[str, Any]) -> None:
    fit = result["fit"]
    held_out = result["held_out_rollout"]
    print("Glassbox synthetic recovery")
    print(
        f"loss: {fit['initial_loss']:.6g} -> {fit['final_loss']:.6g} "
        f"({fit['loss_reduction']:.1f}x reduction)"
    )
    print(f"fit wall time: {fit['wall_time_s']:.2f} s")
    print("held-out full-flight rollout:")
    for name in ("initial", "fitted", "true"):
        metrics = held_out[name]
        print(
            f"  {name:7s}  position={metrics['position_rmse_m']:.4f} m  "
            f"attitude={metrics['attitude_rmse_deg']:.3f} deg  "
            f"velocity={metrics['velocity_rmse_m_s']:.4f} m/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-flights", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--json", type=Path, help="optional output path for full results"
    )
    args = parser.parse_args()

    result = run_demo(
        train_flights=args.train_flights,
        duration_s=args.duration_s,
        dt_s=args.dt_s,
        horizon=args.horizon,
        steps=args.steps,
    )
    _print_summary(result)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
