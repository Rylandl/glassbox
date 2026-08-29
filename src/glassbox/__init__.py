"""Core API for telemetry-driven differentiable dynamics identification.

Source adapters, benchmark workflows, and command-line applications live in
their respective modules. Keeping them out of the package root makes the
stable API small and prevents a core import from loading experiment code.
"""

from glassbox.adapter import TrajectoryAdapter
from glassbox.data import (
    RIGID_BODY_STATE_SCHEMA,
    ControlChannel,
    ExogenousChannel,
    ObservationChannel,
    Trajectory,
    TrajectorySpec,
    TrajectoryWindows,
    VehicleConfigurationSpec,
    duration_to_steps,
    load_trajectory_npz,
    make_trajectory_spec,
    save_trajectory_npz,
    split_trajectory,
    trajectory_segment,
    trajectory_windows,
)
from glassbox.dynamics import (
    BaseDynamicsParams,
    DynamicsParams,
    FixedWingDynamicsParams,
    ModelParams,
    ResidualDynamicsParams,
    model_family,
    rollout,
    rollout_with_latent,
    step,
    step_with_latent,
)
from glassbox.evaluation import (
    aggregate_rollout_metrics,
    rollout_divergence_metrics,
    rollout_metrics,
    windowed_rollout_metrics,
)
from glassbox.identification import (
    FitResult,
    RolloutLossConfiguration,
    fit_dynamics,
    fit_dynamics_multi_horizon,
    rollout_loss_configuration,
)
from glassbox.model_io import load_dynamics_model, save_dynamics_model

__all__ = [
    "RIGID_BODY_STATE_SCHEMA",
    "BaseDynamicsParams",
    "ControlChannel",
    "DynamicsParams",
    "ExogenousChannel",
    "FitResult",
    "FixedWingDynamicsParams",
    "ModelParams",
    "ObservationChannel",
    "ResidualDynamicsParams",
    "RolloutLossConfiguration",
    "Trajectory",
    "TrajectoryAdapter",
    "TrajectorySpec",
    "TrajectoryWindows",
    "VehicleConfigurationSpec",
    "aggregate_rollout_metrics",
    "duration_to_steps",
    "fit_dynamics",
    "fit_dynamics_multi_horizon",
    "load_dynamics_model",
    "load_trajectory_npz",
    "make_trajectory_spec",
    "model_family",
    "rollout",
    "rollout_divergence_metrics",
    "rollout_loss_configuration",
    "rollout_metrics",
    "rollout_with_latent",
    "save_dynamics_model",
    "save_trajectory_npz",
    "split_trajectory",
    "step",
    "step_with_latent",
    "trajectory_segment",
    "trajectory_windows",
    "windowed_rollout_metrics",
]
