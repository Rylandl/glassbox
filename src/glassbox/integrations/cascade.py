"""Optional Cascade fixed-wing plant and predictor for simulation and validation experiments.

Cascade (``cascade-flight``) is a differentiable fixed-wing flight-dynamics core. Its canonical
state boundary is the same NWU/FLU scalar-first 13-vector Glassbox uses, so no frame conversion
happens in this module; the schema strings are compared at import time. Cascade's Skywalker X8 is
assembled from the published NTNU model and is used here as an independent physics predictor,
fitted to nothing in the flight campaign, against the untouched validation maneuvers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.data import (
    RIGID_BODY_STATE_SCHEMA,
    Trajectory,
    TrajectoryWindows,
    trajectory_windows,
)
from glassbox.evaluation import (
    _state_error_metrics,
    kinematic_persistence_windowed_metrics,
)
from glassbox.x8_evaluation import (
    X8_EVALUATION_HORIZONS_S,
    _aggregate_horizons,
    _geometric_ratio,
    _horizon_steps,
    _validation_trajectories,
)
from glassbox.x8_reference import (
    X8_REFERENCE_DOI,
    X8_REFERENCE_NAME,
    X8_REFERENCE_VERSION,
)

X8_CONTROL_ROLES = ("throttle", "roll", "pitch")
# Gryte et al. 2018 report the pitch triple about a nominal CG 0.44 m aft of the nose and warn
# that a 30 mm shift moves the trim alpha from 11 to 3.25 degrees. The NTNU repository's
# "flight-tuned" triple (0.02275, -0.4629, -0.2292) is exactly the wind-tunnel triple moved 30 mm
# forward by the lever-arm transform C_m' = C_m - (dx / c) C_L, so CG shift is the one axis.
DEFAULT_CG_SHIFTS_M = (0.0, 0.02, 0.03, 0.05)


class CascadeUnavailableError(RuntimeError):
    """Raised when the optional Cascade dependency is unavailable or incompatible."""


def _require_cascade() -> Any:
    try:
        import cascade
        from cascade.canonical import CANONICAL_STATE_SCHEMA
    except ImportError as error:
        raise CascadeUnavailableError(
            "install the optional simulator with `uv sync --extra cascade`"
        ) from error
    if CANONICAL_STATE_SCHEMA != RIGID_BODY_STATE_SCHEMA:
        raise CascadeUnavailableError(
            f"Cascade canonical schema {CANONICAL_STATE_SCHEMA!r} does not match "
            f"{RIGID_BODY_STATE_SCHEMA!r}"
        )
    return cascade


@dataclass(frozen=True)
class CascadePlantConfig:
    """Fixed execution contract for one Cascade plant."""

    aircraft: str = "skywalker_x8"
    simulation_frequency_hz: int = 400
    control_frequency_hz: int = 40
    density_kg_m3: float = 1.225

    def __post_init__(self) -> None:
        if self.aircraft not in ("skywalker_x8", "aerobatic_reference"):
            raise ValueError("aircraft must be 'skywalker_x8' or 'aerobatic_reference'")


class CascadePlant:
    """Single-world Cascade plant hidden behind canonical telemetry.

    ``reset``, ``step`` and ``snapshot`` return Cascade's ``PlantSample`` whose ``state`` is the
    canonical 13-vector, ``commanded_control`` and ``applied_control`` follow ``control_names``
    (propellers first, then the specification's channels in their own units), and
    ``wind_nwu_m_s`` is the held wind.
    """

    def __init__(
        self,
        config: CascadePlantConfig | None = None,
        *,
        spec: Any | None = None,
        model: Any | None = None,
    ) -> None:
        cascade = _require_cascade()
        self.config = CascadePlantConfig() if config is None else config
        if spec is None:
            loaders = {
                "skywalker_x8": cascade.skywalker_x8_spec,
                "aerobatic_reference": cascade.aerobatic_reference_spec,
            }
            spec = loaders[self.config.aircraft]()
        self.spec = spec
        self._plant = cascade.Plant(
            spec,
            cascade.PlantConfig(
                simulation_frequency_hz=self.config.simulation_frequency_hz,
                control_frequency_hz=self.config.control_frequency_hz,
                density_kg_m3=self.config.density_kg_m3,
            ),
            model=model,
        )
        self.control_names: tuple[str, ...] = self._plant.control_names

    @property
    def model(self) -> Any:
        return self._plant.model

    @property
    def sample_period_s(self) -> float:
        return self._plant.sample_period_s

    def reset(self, state: Any, *, applied_control: Any | None = None, wind_nwu: Any | None = None):
        return self._plant.reset(state, applied_control=applied_control, wind_nwu=wind_nwu)

    def step(self, command: Any, *, wind_nwu: Any | None = None):
        return self._plant.step(command, wind_nwu=wind_nwu)

    def snapshot(self):
        return self._plant.snapshot()


@dataclass(frozen=True)
class X8Variant:
    """One published-parameter choice for the Cascade X8 plus a vertical-wind fraction."""

    label: str
    cg_shift_forward_m: float
    mass_kg: float
    yaw_damping: float
    vertical_wind_fraction: float
    primary: bool


def shift_center_of_gravity(body: Any, shift_forward_m: float, chord_m: float) -> Any:
    """Move the pitching-moment reference forward with the lever-arm transform.

    Every lift term produces a moment about the new point: ``C_m' = C_m - (dx / c) C_L`` applied
    to the constant, alpha, pitch-rate, and elevator coefficients alike (Reinhardt et al. 2022,
    Eq. 18, for a pure longitudinal shift and small angles).
    """

    ratio = shift_forward_m / chord_m
    pitch = replace(
        body.pitch,
        zero=body.pitch.zero - ratio * body.lift.zero,
        alpha_rad=body.pitch.alpha_rad - ratio * body.lift.alpha_rad,
        q=body.pitch.q - ratio * body.lift.q,
        elevator_rad=body.pitch.elevator_rad - ratio * body.lift.elevator_rad,
    )
    return replace(body, pitch=pitch)


def x8_variant_models(
    *,
    cg_shifts_forward_m: Sequence[float] = DEFAULT_CG_SHIFTS_M,
    masses_kg: Sequence[float] = (3.364, 4.0),
    yaw_damping: Sequence[float] = (-0.012, -0.072),
    vertical_wind_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[list[X8Variant], list[Any]]:
    """Build the Cascade X8 variant grid. The primary variant is the published model as-is."""

    cascade = _require_cascade()

    base = cascade.skywalker_x8_spec()
    variants: list[X8Variant] = []
    models: list[Any] = []
    for shift in cg_shifts_forward_m:
        for mass in masses_kg:
            for cnr in yaw_damping:
                body = shift_center_of_gravity(base.body, float(shift), base.reference_chord_m)
                body = replace(body, yaw=replace(body.yaw, r=float(cnr)))
                spec = replace(base, mass_kg=float(mass), body=body)
                model = spec.to_model()
                for fraction in vertical_wind_fractions:
                    primary = (
                        float(shift) == 0.0
                        and float(mass) == base.mass_kg
                        and float(cnr) == base.body.yaw.r
                        and float(fraction) == 1.0
                    )
                    variants.append(
                        X8Variant(
                            label=(
                                f"cg=+{shift:g}m,mass={mass:g}kg,cnr={cnr:g},"
                                f"vertical_wind={fraction:g}"
                            ),
                            cg_shift_forward_m=float(shift),
                            mass_kg=float(mass),
                            yaw_damping=float(cnr),
                            vertical_wind_fraction=float(fraction),
                            primary=primary,
                        )
                    )
                    models.append(model)
    return variants, models


def _check_control_layout(windows: TrajectoryWindows) -> None:
    if windows.control_roles != X8_CONTROL_ROLES:
        raise ValueError(
            f"Cascade X8 prediction expects control roles {X8_CONTROL_ROLES}, "
            f"got {windows.control_roles}"
        )


def _initial_wind_nwu(windows: TrajectoryWindows) -> np.ndarray:
    """Return (windows, 3) NWU wind held through each window, zero when untyped."""

    count = windows.initial_states.shape[0]
    wind = np.zeros((count, 3), dtype=np.float64)
    roles = windows.exogenous_roles or ()
    for column, axis in enumerate(("wind_north", "wind_west", "wind_up")):
        if axis in roles:
            wind[:, column] = windows.initial_exogenous[:, roles.index(axis)]
    return wind


def predict_windows(
    models: Sequence[Any],
    windows: TrajectoryWindows,
    *,
    vertical_wind_fractions: Sequence[float],
    simulation_substeps: int = 10,
) -> np.ndarray:
    """Predict every window under every model variant with Cascade's RK4 dynamics.

    Returns ``(variants, windows, horizon + 1, 13)`` canonical states including the measured
    initial state. Each logged control is held for one sample interval and integrated with
    ``simulation_substeps`` RK4 steps. Actuators and separation start at their equilibria for the
    last control before the window; the typed wind at the window start is held throughout with
    its vertical component scaled by the variant's fraction.
    """

    cascade = _require_cascade()
    from cascade.canonical import (
        nwu_to_ned,
        rigid_body_from_canonical,
        rigid_body_to_canonical,
    )
    from cascade.initialization import (
        control_from_array,
        equilibrate_internal_state,
        standard_environment,
        zero_state,
    )
    from cascade.integration import repeat_control, rollout

    if len(models) != len(vertical_wind_fractions):
        raise ValueError("one vertical-wind fraction is required per model variant")
    if simulation_substeps < 1:
        raise ValueError("simulation_substeps must be positive")
    _check_control_layout(windows)

    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)
    fractions = jnp.asarray(np.asarray(vertical_wind_fractions, dtype=np.float64))
    initial_states = jnp.asarray(windows.initial_states)
    controls = jnp.asarray(windows.controls)
    last_history = jnp.asarray(windows.control_histories[:, -1])
    wind_nwu = jnp.asarray(_initial_wind_nwu(windows))
    dt = windows.dt_s / simulation_substeps
    del cascade

    def predict_one(model, fraction, initial_state, control_sequence, history, wind):
        scaled_wind = wind.at[2].multiply(fraction)
        environment = standard_environment()._replace(wind=nwu_to_ned(scaled_wind))
        state = zero_state(model)._replace(rigid_body=rigid_body_from_canonical(initial_state))
        state = equilibrate_internal_state(
            model, state, control_from_array(model, history), environment
        )

        def hold(carry, control_row):
            control = control_from_array(model, control_row)
            final, _ = rollout(
                model, carry, repeat_control(control, simulation_substeps), environment, dt
            )
            return final, rigid_body_to_canonical(final.rigid_body)

        _, predicted = jax.lax.scan(hold, state, control_sequence)
        return jnp.concatenate((initial_state[None, :], predicted), axis=0)

    over_windows = jax.vmap(predict_one, in_axes=(None, None, 0, 0, 0, 0))
    over_variants = jax.vmap(over_windows, in_axes=(0, 0, None, None, None, None))
    predicted = jax.jit(over_variants)(
        stacked, fractions, initial_states, controls, last_history, wind_nwu
    )
    return np.asarray(predicted, dtype=np.float64)


def evaluate_x8_cascade(
    destination: str | Path,
    *,
    horizons_s: Sequence[float] = X8_EVALUATION_HORIZONS_S,
    vertical_wind_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    cg_shifts_forward_m: Sequence[float] = DEFAULT_CG_SHIFTS_M,
    reference_report: str | Path | None = None,
    simulation_substeps: int = 10,
) -> dict[str, Any]:
    """Score the published Cascade X8, unfitted, on the untouched validation maneuvers.

    Uses exactly the artifact protocol: rolling windows initialized at every admissible sample,
    the four state metrics per horizon, equal weighting over validation maneuvers, and the
    kinematic-persistence baseline. Rows other than the primary one are an environment and
    parameter sensitivity, never a fit: one scalar per variant, applied identically everywhere.
    """

    destination = Path(destination)
    validation_paths = tuple(sorted((destination / "canonical" / "validation").glob("*.npz")))
    paths, trajectories = _validation_trajectories(validation_paths)
    horizon_steps = {
        f"{horizon:g}s": _horizon_steps(trajectories[0], horizon) for horizon in horizons_s
    }
    variants, models = x8_variant_models(
        cg_shifts_forward_m=cg_shifts_forward_m, vertical_wind_fractions=vertical_wind_fractions
    )
    fractions = [variant.vertical_wind_fraction for variant in variants]

    persistence_per_trajectory = []
    per_variant: list[list[dict[str, Any]]] = [[] for _ in variants]
    for path, trajectory in zip(paths, trajectories):
        persistence_per_trajectory.append(
            {
                "path": str(path),
                "horizon_rollouts": {
                    label: kinematic_persistence_windowed_metrics(
                        trajectory, horizon_steps=steps, stride_steps=1
                    )
                    for label, steps in horizon_steps.items()
                },
            }
        )
        rollouts: list[dict[str, dict[str, Any]]] = [{} for _ in variants]
        for label, steps in horizon_steps.items():
            windows = trajectory_windows([trajectory], horizon=steps, stride=1)
            predicted = predict_windows(
                models,
                windows,
                vertical_wind_fractions=fractions,
                simulation_substeps=simulation_substeps,
            )
            for index in range(len(variants)):
                rollouts[index][label] = _state_error_metrics(
                    predicted[index], windows.target_states, duration_s=steps * windows.dt_s
                )
        for index in range(len(variants)):
            per_variant[index].append({"path": str(path), "horizon_rollouts": rollouts[index]})

    persistence_aggregate = _aggregate_horizons(persistence_per_trajectory)
    rows: dict[str, Any] = {}
    for variant, per_trajectory in zip(variants, per_variant):
        aggregate = _aggregate_horizons(per_trajectory)
        rows[variant.label] = {
            "model_type": "cascade_skywalker_x8_published",
            "primary": variant.primary,
            "parameters": {
                "cg_shift_forward_m": variant.cg_shift_forward_m,
                "mass_kg": variant.mass_kg,
                "yaw_damping_cnr": variant.yaw_damping,
                "vertical_wind_fraction": variant.vertical_wind_fraction,
            },
            "aggregate": {"horizon_rollouts": aggregate},
            "per_trajectory": per_trajectory,
            "score_vs_kinematic_persistence": _geometric_ratio(aggregate, persistence_aggregate),
            "all_finite": all(
                np.isfinite(
                    [item["horizon_rollouts"][label]["position_rmse_m"] for item in per_trajectory]
                ).all()
                for label in horizon_steps
            ),
        }

    comparisons: dict[str, Any] = {}
    reference_rows: dict[str, Any] = {}
    if reference_report is not None and Path(reference_report).exists():
        with Path(reference_report).open() as source:
            reference = json.load(source)
        reference_rows = {
            name: model["aggregate"]["horizon_rollouts"]
            for name, model in reference.get("models", {}).items()
        }
    for label, row in rows.items():
        for name, aggregate in reference_rows.items():
            comparisons[f"{label}_vs_{name}"] = {
                "ratio_definition": (
                    "candidate/reference geometric mean over four state metrics and every "
                    "horizon; values below one favor the candidate"
                ),
                "score": _geometric_ratio(row["aggregate"]["horizon_rollouts"], aggregate),
            }

    primary = next(label for label, row in rows.items() if row["primary"])
    best = min(rows, key=lambda label: rows[label]["score_vs_kinematic_persistence"])
    return {
        "format_version": 1,
        "benchmark": {
            "name": X8_REFERENCE_NAME,
            "doi": X8_REFERENCE_DOI,
            "version": X8_REFERENCE_VERSION,
        },
        "protocol": {
            "split": "upstream_validation",
            "initialization": "every_admissible_sample",
            "flight_boundaries_crossed": False,
            "horizons_s": list(horizons_s),
            "aggregation": "equal_validation_maneuver",
            "baseline": "constant_world_velocity_and_constant_body_rate",
            "predictor": "cascade_skywalker_x8_published_model_no_fitting",
            "integration": f"rk4_{simulation_substeps}_substeps_per_sample",
            "actuator_initialization": "equilibrium_at_last_control_before_window",
            "wind": "typed_window_start_wind_held_vertical_component_scaled_per_variant",
            "cg_shift": "lever_arm_transform_of_the_wind_tunnel_pitch_triple",
        },
        "dataset": {
            "validation_trajectory_count": len(trajectories),
            "validation_duration_s": float(sum(t.time_s[-1] for t in trajectories)),
        },
        "kinematic_persistence": {
            "aggregate": {"horizon_rollouts": persistence_aggregate},
            "per_trajectory": persistence_per_trajectory,
        },
        "models": rows,
        "primary_model": primary,
        "best_model": best,
        "comparisons": comparisons,
        "provenance": {
            "cascade_aircraft": "skywalker_x8 (published NTNU model; see the TOML for sources)",
            "cg_note": (
                "Gryte et al. 2018 give the pitch triple about a nominal CG and note a 30 mm "
                "shift moves trim alpha from 11 to 3.25 degrees; the NTNU repository's "
                "flight-tuned triple equals the wind-tunnel triple moved 30 mm forward. The "
                "cg_shift_forward_m axis applies that lever-arm transform; 0 is the published model."
            ),
            "vertical_wind_note": (
                "The campaign's vertical wind (about 2.8 m/s) was inferred upstream from pitot "
                "airspeed minus horizontal relative airspeed. At the measured load factor the "
                "published lift curve is consistent with roughly 0.4 of that estimate; a 1.2% "
                "pitot bias would produce all of it. The vertical_wind_fraction axis reports "
                "that sensitivity; the primary row uses the campaign wind unmodified."
            ),
        },
        "acceptance": {
            "status": "not_scored",
            "passed": None,
            "reason": (
                "an unfitted physics model is characterization evidence for the simulator and "
                "the campaign, not a candidate under the fixed-wing development contract"
            ),
        },
    }


def save_x8_cascade_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")


def trajectory_from_plant_samples(samples: Sequence[Any], spec: Any) -> Trajectory:
    """Assemble canonical plant telemetry into a Glassbox trajectory with zero typed wind."""

    time_s = np.asarray([sample.time_s for sample in samples], dtype=np.float64)
    states = np.stack([sample.state for sample in samples])
    controls = np.stack([sample.commanded_control for sample in samples[1:]])
    exogenous = np.stack([sample.wind_nwu_m_s for sample in samples])
    return Trajectory(
        time_s=time_s,
        states=states,
        controls=controls,
        exogenous=exogenous if spec.exogenous else None,
        spec=spec,
        labels={"source": "cascade_plant"},
        provenance={"source": "cascade_plant"},
    )
