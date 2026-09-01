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
# The bifilar-pendulum inertia (Reinhardt et al. 2022) is for a bare airframe; the older NTNU
# parameter file lists a roll inertia 3.7 times larger, and the 2023 instrumented airframe's
# inertia is undocumented. Scaling the pendulum tensor keeps its ratios and spans that range.
DEFAULT_INERTIA_SCALES = (1.0, 2.0, 3.5)


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
    inertia_scale: float
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


X8_AIRCRAFT = ("skywalker_x8", "skywalker_x8_panels")


def _x8_spec(cascade: Any, aircraft: str) -> Any:
    if aircraft not in X8_AIRCRAFT:
        raise ValueError(f"aircraft must be one of {X8_AIRCRAFT}")
    loader = getattr(cascade, f"{aircraft}_spec", None)
    if loader is None:
        raise CascadeUnavailableError(f"the installed Cascade has no {aircraft}_spec loader")
    return loader()


def shift_center_of_gravity_of_spec(spec: Any, shift_forward_m: float) -> Any:
    """Move the CG forward on either backend.

    A coefficient table gets the lever-arm transform of its pitch coefficients; component
    surfaces and propellers simply sit further aft of the new center of mass.
    """

    if shift_forward_m == 0.0:
        return spec
    body = spec.body
    if body is not None:
        body = shift_center_of_gravity(body, shift_forward_m, spec.reference_chord_m)
    surfaces = tuple(
        replace(surface, position_m=(surface.position_m[0] - shift_forward_m, *surface.position_m[1:]))
        for surface in spec.surfaces
    )
    propellers = tuple(
        replace(propeller, position_m=(propeller.position_m[0] - shift_forward_m, *propeller.position_m[1:]))
        for propeller in spec.propellers
    )
    return replace(spec, body=body, surfaces=surfaces, propellers=propellers)


def x8_variant_models(
    *,
    aircraft: str = "skywalker_x8",
    cg_shifts_forward_m: Sequence[float] = DEFAULT_CG_SHIFTS_M,
    masses_kg: Sequence[float] = (3.364, 4.0),
    yaw_damping: Sequence[float] = (-0.012,),
    inertia_scales: Sequence[float] = DEFAULT_INERTIA_SCALES,
    vertical_wind_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[list[X8Variant], list[Any]]:
    """Build the Cascade X8 variant grid. The primary variant is the published model as-is."""

    cascade = _require_cascade()

    base = _x8_spec(cascade, aircraft)
    variants: list[X8Variant] = []
    models: list[Any] = []
    for shift in cg_shifts_forward_m:
        for mass in masses_kg:
            for cnr in yaw_damping:
                for inertia_scale in inertia_scales:
                    spec = shift_center_of_gravity_of_spec(base, float(shift))
                    if spec.body is not None:
                        spec = replace(
                            spec, body=replace(spec.body, yaw=replace(spec.body.yaw, r=float(cnr)))
                        )
                    inertia = tuple(
                        tuple(float(inertia_scale) * value for value in row)
                        for row in base.inertia_kg_m2
                    )
                    spec = replace(spec, mass_kg=float(mass), inertia_kg_m2=inertia)
                    model = spec.to_model()
                    for fraction in vertical_wind_fractions:
                        primary = (
                            float(shift) == 0.0
                            and float(mass) == base.mass_kg
                            and (base.body is None or float(cnr) == base.body.yaw.r)
                            and float(inertia_scale) == 1.0
                            and float(fraction) == 1.0
                        )
                        variants.append(
                            X8Variant(
                                label=(
                                    f"cg=+{shift:g}m,mass={mass:g}kg,cnr={cnr:g},"
                                    f"inertia=x{inertia_scale:g},vertical_wind={fraction:g}"
                                ),
                                cg_shift_forward_m=float(shift),
                                mass_kg=float(mass),
                                yaw_damping=float(cnr),
                                inertia_scale=float(inertia_scale),
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


def actuator_states_over_controls(
    model: Any, controls: Any, dt_s: float, *, simulation_substeps: int = 10
) -> Any:
    """Integrate the actuator lag over a control sequence, from equilibrium at the first one.

    Returns an ``ActuatorState`` whose leaves carry a leading axis of ``len(controls) + 1``:
    entry ``i`` is the actuator state at the start of interval ``i``, so entry ``0`` is the
    equilibrium for the first control and the last entry is the state after every control was
    applied. Uses the same RK4 sub-stepping as the plant and the physical clips of the core.
    """

    _require_cascade()
    from cascade.actuators import actuator_dynamics, actuator_targets
    from cascade.initialization import control_from_array
    from cascade.state import ActuatorState

    controls = jnp.asarray(controls)
    step = dt_s / simulation_substeps
    limits = model.actuators

    def combine(state, derivative, scale):
        return ActuatorState(
            surface_deflection=state.surface_deflection + scale * derivative.surface_deflection,
            propeller_speed=state.propeller_speed + scale * derivative.propeller_speed,
        )

    def interval(state, control_row):
        control = control_from_array(model, control_row)

        def substep(current, _):
            k1 = actuator_dynamics(model, current, control)
            k2 = actuator_dynamics(model, combine(current, k1, step / 2.0), control)
            k3 = actuator_dynamics(model, combine(current, k2, step / 2.0), control)
            k4 = actuator_dynamics(model, combine(current, k3, step), control)
            weighted = ActuatorState(
                surface_deflection=(
                    k1.surface_deflection
                    + 2.0 * k2.surface_deflection
                    + 2.0 * k3.surface_deflection
                    + k4.surface_deflection
                )
                / 6.0,
                propeller_speed=(
                    k1.propeller_speed
                    + 2.0 * k2.propeller_speed
                    + 2.0 * k3.propeller_speed
                    + k4.propeller_speed
                )
                / 6.0,
            )
            advanced = combine(current, weighted, step)
            clipped = ActuatorState(
                surface_deflection=jnp.clip(
                    advanced.surface_deflection, -limits.surface_limit, limits.surface_limit
                ),
                propeller_speed=jnp.clip(
                    advanced.propeller_speed, limits.propeller_speed_min, limits.propeller_speed_max
                ),
            )
            return clipped, None

        final, _ = jax.lax.scan(substep, state, None, length=simulation_substeps)
        return final, final

    initial = actuator_targets(model, control_from_array(model, controls[0]))
    _, after_each = jax.lax.scan(interval, initial, controls)
    return ActuatorState(
        surface_deflection=jnp.concatenate(
            (initial.surface_deflection[None], after_each.surface_deflection), axis=0
        ),
        propeller_speed=jnp.concatenate(
            (initial.propeller_speed[None], after_each.propeller_speed), axis=0
        ),
    )


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
    histories = jnp.asarray(windows.control_histories)
    wind_nwu = jnp.asarray(_initial_wind_nwu(windows))
    dt = windows.dt_s / simulation_substeps
    del cascade

    def predict_one(model, fraction, initial_state, control_sequence, history, wind):
        scaled_wind = wind.at[2].multiply(fraction)
        environment = standard_environment()._replace(wind=nwu_to_ned(scaled_wind))
        state = zero_state(model)._replace(rigid_body=rigid_body_from_canonical(initial_state))
        state = equilibrate_internal_state(
            model, state, control_from_array(model, history[-1]), environment
        )
        # Actuators carry the lagged response to the whole control history, not just its end.
        lagged = actuator_states_over_controls(
            model, history, windows.dt_s, simulation_substeps=simulation_substeps
        )
        state = state._replace(
            actuators=jax.tree.map(lambda leaf: leaf[-1], lagged)
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
        stacked, fractions, initial_states, controls, histories, wind_nwu
    )
    return np.asarray(predicted, dtype=np.float64)


def evaluate_x8_cascade(
    destination: str | Path,
    *,
    aircraft: str = "skywalker_x8",
    horizons_s: Sequence[float] = X8_EVALUATION_HORIZONS_S,
    vertical_wind_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    cg_shifts_forward_m: Sequence[float] = DEFAULT_CG_SHIFTS_M,
    inertia_scales: Sequence[float] = DEFAULT_INERTIA_SCALES,
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
        aircraft=aircraft,
        cg_shifts_forward_m=cg_shifts_forward_m,
        inertia_scales=inertia_scales,
        vertical_wind_fractions=vertical_wind_fractions,
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
            "model_type": f"cascade_{aircraft}",
            "primary": variant.primary,
            "parameters": {
                "cg_shift_forward_m": variant.cg_shift_forward_m,
                "mass_kg": variant.mass_kg,
                "inertia_scale": variant.inertia_scale,
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
            "actuator_initialization": "lag_integrated_over_the_window_control_history",
            "wind": "typed_window_start_wind_held_vertical_component_scaled_per_variant",
            "cg_shift": "lever_arm_transform_of_the_wind_tunnel_pitch_triple",
            "inertia": "bifilar_pendulum_tensor_scaled_uniformly",
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
            "cascade_aircraft": aircraft,
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


RESIDUAL_FEATURES = (
    "1",
    "alpha",
    "beta",
    "p_hat",
    "q_hat",
    "r_hat",
    "delta_a",
    "delta_e",
    "delta_r",
    "throttle",
)
RESIDUAL_CHANNELS = ("X", "Y", "Z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class ResidualRegression:
    """One-step residual of a force or moment channel regressed on the flight variables.

    ``corrections`` are the regression coefficients expressed as coefficient increments
    (predicted minus measured, per unit feature) after dividing by the mean dynamic pressure and
    the reference area, span, or chord, so they read as corrections to the published derivatives.
    """

    channel: str
    unit: str
    mean: float
    rms: float
    r_squared: float
    sample_count: int
    corrections: dict[str, float]


def residual_regressions(
    model: Any,
    trajectories: Sequence[Trajectory],
    *,
    vertical_wind_fraction: float = 1.0,
) -> dict[str, ResidualRegression]:
    """Regress one-step Cascade residuals on the flight variables, per body axis.

    At every interior sample the model is evaluated at the measured state with actuators
    carrying their lagged response to the logged control history and the typed wind held; its
    body-frame specific
    force and angular acceleration are compared with central differences of the measured
    velocity and body rates. Residuals are in FRD body axes, forces in newtons and moments in
    newton-metres using the model's mass and inertia tensor. This is an equation-error
    diagnostic: the fitted coefficients say which published derivatives disagree with flight and
    by how much, they are not a fit to be written back into the model.
    """

    cascade = _require_cascade()
    from cascade.canonical import (
        flu_to_frd,
        nwu_to_ned,
        rigid_body_from_canonical,
    )
    from cascade.dynamics import evaluate_dynamics
    from cascade.initialization import (
        control_from_array,
        equilibrate_internal_state,
        standard_environment,
        zero_state,
    )
    from cascade.math import quaternion_rotate_inverse
    from cascade.state import ActuatorState

    del cascade
    fraction = float(vertical_wind_fraction)

    def predict(state13, actuators, control_now, wind_nwu):
        environment = standard_environment()._replace(
            wind=nwu_to_ned(wind_nwu.at[2].multiply(fraction))
        )
        state = zero_state(model)._replace(rigid_body=rigid_body_from_canonical(state13))
        state = equilibrate_internal_state(
            model, state, control_from_array(model, control_now), environment
        )
        state = state._replace(actuators=actuators)
        result = evaluate_dynamics(model, state, control_from_array(model, control_now), environment)
        specific_force = quaternion_rotate_inverse(
            state.rigid_body.attitude, result.derivative.rigid_body.velocity - environment.gravity
        )
        body = result.aerodynamics.body
        return (
            specific_force,
            result.derivative.rigid_body.angular_velocity,
            body.angle_of_attack,
            body.sideslip,
            body.airspeed,
        )

    batched = jax.jit(jax.vmap(predict))
    mass = float(np.asarray(model.mass))
    inertia = np.asarray(model.inertia, dtype=np.float64)
    area = float(np.asarray(model.reference_area))
    span = float(np.asarray(model.reference_span))
    chord = float(np.asarray(model.reference_chord))

    force_residuals, moment_residuals, features, pressures = [], [], [], []
    for trajectory in trajectories:
        states = trajectory.states
        controls = trajectory.controls
        dt = trajectory.nominal_dt_s
        index = np.arange(2, len(states) - 2)
        wind = np.zeros((len(index), 3))
        roles = trajectory.spec.exogenous_roles
        for column, axis in enumerate(("wind_north", "wind_west", "wind_up")):
            if axis in roles:
                wind[:, column] = trajectory.exogenous[index, roles.index(axis)]
        lagged = actuator_states_over_controls(model, controls, dt)
        actuators = ActuatorState(
            surface_deflection=lagged.surface_deflection[index],
            propeller_speed=lagged.propeller_speed[index],
        )
        outputs = batched(
            jnp.asarray(states[index]),
            actuators,
            jnp.asarray(controls[index]),
            jnp.asarray(wind),
        )
        sf_pred, alpha_pred, aoa, beta, airspeed = (
            np.asarray(value, dtype=np.float64) for value in outputs
        )
        # Generalized control features come from the logged commands so the regression reads the
        # same way for the coefficient table and the component panels (rudder absent on the X8).
        deflection = np.column_stack(
            (controls[index, 1], controls[index, 2], np.zeros(len(index)))
        )
        # Measured specific force in FRD: rotate (a - g) from NWU into FLU with the canonical
        # attitude, then flip to FRD. Measured angular acceleration: differences of FLU rates.
        acceleration_nwu = (states[index + 1, 3:6] - states[index - 1, 3:6]) / (2.0 * dt)
        w, x, y, z = states[index, 6:10].T
        rotation = np.stack(
            [
                np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
            ],
            -2,
        )
        sf_flu = np.einsum("nji,nj->ni", rotation, acceleration_nwu - np.array([0.0, 0.0, -9.80665]))
        sf_meas = np.asarray(flu_to_frd(jnp.asarray(sf_flu)))
        rate_flu = states[:, 10:13]
        alpha_meas = np.asarray(
            flu_to_frd(jnp.asarray((rate_flu[index + 1] - rate_flu[index - 1]) / (2.0 * dt)))
        )
        rates_frd = np.asarray(flu_to_frd(jnp.asarray(rate_flu[index])))
        force_residuals.append(mass * (sf_pred - sf_meas))
        moment_residuals.append((alpha_pred - alpha_meas) @ inertia.T)
        throttle = controls[index, 0]
        features.append(
            np.column_stack(
                [
                    np.ones(len(index)),
                    aoa,
                    beta,
                    span * rates_frd[:, 0] / (2.0 * airspeed),
                    chord * rates_frd[:, 1] / (2.0 * airspeed),
                    span * rates_frd[:, 2] / (2.0 * airspeed),
                    deflection[:, 0],
                    deflection[:, 1],
                    deflection[:, 2],
                    throttle,
                ]
            )
        )
        pressures.append(0.5 * 1.225 * airspeed**2)

    design = np.concatenate(features)
    forces = np.concatenate(force_residuals)
    moments = np.concatenate(moment_residuals)
    dynamic_pressure = float(np.concatenate(pressures).mean())
    scales = {
        "X": (forces[:, 0], area, "N"),
        "Y": (forces[:, 1], area, "N"),
        "Z": (forces[:, 2], area, "N"),
        "roll": (moments[:, 0], area * span, "N m"),
        "pitch": (moments[:, 1], area * chord, "N m"),
        "yaw": (moments[:, 2], area * span, "N m"),
    }
    results: dict[str, ResidualRegression] = {}
    for channel, (residual, scale, unit) in scales.items():
        coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
        explained = residual - design @ coefficients
        variance = float(np.var(residual))
        r_squared = 0.0 if variance <= 0.0 else 1.0 - float(np.var(explained)) / variance
        results[channel] = ResidualRegression(
            channel=channel,
            unit=unit,
            mean=float(residual.mean()),
            rms=float(np.sqrt(np.mean(residual**2))),
            r_squared=r_squared,
            sample_count=len(residual),
            corrections={
                name: float(value / (dynamic_pressure * scale))
                for name, value in zip(RESIDUAL_FEATURES, coefficients)
            },
        )
    return results


def diagnose_x8_cascade(
    destination: str | Path,
    *,
    aircraft: str = "skywalker_x8",
    split: str = "validation",
    cg_shift_forward_m: float = 0.0,
    mass_kg: float | None = None,
    inertia_scale: float = 1.0,
    vertical_wind_fraction: float = 1.0,
) -> dict[str, Any]:
    """Run the residual regressions for one Cascade X8 configuration on a campaign split."""

    destination = Path(destination)
    splits = ("training", "validation") if split == "all" else (split,)
    paths = []
    for name in splits:
        paths.extend(sorted((destination / "canonical" / name).glob("*.npz")))
    if not paths:
        raise ValueError(f"no canonical trajectories under {destination} for split {split!r}")
    from glassbox.data import load_trajectory_npz

    trajectories = [load_trajectory_npz(path) for path in paths]
    cascade = _require_cascade()
    spec = shift_center_of_gravity_of_spec(_x8_spec(cascade, aircraft), cg_shift_forward_m)
    inertia = tuple(tuple(inertia_scale * value for value in row) for row in spec.inertia_kg_m2)
    spec = replace(spec, inertia_kg_m2=inertia)
    if mass_kg is not None:
        spec = replace(spec, mass_kg=float(mass_kg))
    model = spec.to_model()
    regressions = residual_regressions(
        model, trajectories, vertical_wind_fraction=vertical_wind_fraction
    )
    return {
        "format_version": 1,
        "aircraft": aircraft,
        "configuration": {
            "cg_shift_forward_m": cg_shift_forward_m,
            "mass_kg": spec.mass_kg,
            "inertia_scale": inertia_scale,
            "vertical_wind_fraction": vertical_wind_fraction,
        },
        "split": split,
        "trajectories": [str(path) for path in paths],
        "features": list(RESIDUAL_FEATURES),
        "channels": {
            channel: {
                "unit": item.unit,
                "mean": item.mean,
                "rms": item.rms,
                "r_squared": item.r_squared,
                "sample_count": item.sample_count,
                "corrections": item.corrections,
            }
            for channel, item in regressions.items()
        },
        "reading": (
            "corrections are predicted-minus-measured per unit feature in coefficient units; a "
            "positive p_hat correction on roll means the model's roll damping is too strong by "
            "that amount, a positive constant on Z (FRD down) means too little lift"
        ),
    }


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
