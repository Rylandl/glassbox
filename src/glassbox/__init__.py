"""Telemetry-driven differentiable dynamics identification."""

from glassbox.adapter import TrajectoryAdapter
from glassbox.arp_reference import (
    ARPReferenceAdapter,
    extract_arp_reference,
    fetch_arp_reference,
)
from glassbox.data import (
    ControlChannel,
    ObservationChannel,
    RIGID_BODY_STATE_SCHEMA,
    Trajectory,
    TrajectorySpec,
    TrajectoryWindows,
    VehicleConfigurationSpec,
    make_trajectory_spec,
    load_trajectory_npz,
    save_trajectory_npz,
    split_trajectory,
    trajectory_segment,
    trajectory_windows,
)
from glassbox.dynamics import (
    BaseDynamicsParams,
    DynamicsParams,
    FixedWingDynamicsParams,
    ResidualDynamicsParams,
    control_state_after_history,
    fixed_wing_trim_control,
    initial_residual_parameters,
    model_family,
    rollout,
    rollout_with_latent,
    step,
    step_with_latent,
    structured_parameters,
)
from glassbox.identification import (
    RolloutLossConfiguration,
    dynamic_envelope_penalty,
    residual_initialization_statistics,
    rollout_loss_configuration,
)
from glassbox.evaluation import (
    aggregate_innovation_diagnostics,
    kinematic_persistence_windowed_metrics,
    one_step_innovation_diagnostics,
    rollout_divergence_metrics,
    state_kinematic_compatibility_diagnostics,
)
from glassbox.idf_reference import (
    IDFFixedWingAdapter,
    extract_idf_reference,
    extract_idf_ulogs,
    fetch_idf_archive,
    idf_corpus_report,
    save_idf_corpus_report,
)
from glassbox.nanodrone_benchmark import (
    NanoDroneBenchmarkAdapter,
    extract_nanodrone_benchmark,
    fetch_nanodrone_benchmark,
    nanodrone_trajectory_spec,
)
from glassbox.nanodrone_evaluation import (
    evaluate_nanodrone_benchmark,
    evaluate_nanodrone_model_artifact,
    save_nanodrone_benchmark_report,
)
from glassbox.observation_identification import (
    actuator_observation_alignment,
    fit_multirotor_observations,
)
from glassbox.source_group_benchmark import benchmark_source_groups
from glassbox.fixedwing_gate import (
    compare_fixedwing_gates,
    evaluate_fixedwing_gate,
    save_fixedwing_gate,
    screen_fixedwing_airframe_candidate,
)
from glassbox.x8_reference import (
    X8ReferenceAdapter,
    extract_x8_reference,
    fetch_x8_reference,
    x8_trajectory_spec,
)
from glassbox.x8_evaluation import (
    evaluate_x8_reference_models,
    save_x8_reference_report,
)

__all__ = [
    "ARPReferenceAdapter",
    "ControlChannel",
    "ObservationChannel",
    "BaseDynamicsParams",
    "DynamicsParams",
    "FixedWingDynamicsParams",
    "IDFFixedWingAdapter",
    "RIGID_BODY_STATE_SCHEMA",
    "RolloutLossConfiguration",
    "ResidualDynamicsParams",
    "NanoDroneBenchmarkAdapter",
    "Trajectory",
    "TrajectoryAdapter",
    "TrajectorySpec",
    "TrajectoryWindows",
    "VehicleConfigurationSpec",
    "X8ReferenceAdapter",
    "control_state_after_history",
    "actuator_observation_alignment",
    "aggregate_innovation_diagnostics",
    "compare_fixedwing_gates",
    "dynamic_envelope_penalty",
    "fixed_wing_trim_control",
    "fetch_arp_reference",
    "fetch_idf_archive",
    "fetch_nanodrone_benchmark",
    "fetch_x8_reference",
    "fit_multirotor_observations",
    "extract_nanodrone_benchmark",
    "extract_arp_reference",
    "extract_idf_reference",
    "extract_idf_ulogs",
    "extract_x8_reference",
    "idf_corpus_report",
    "evaluate_nanodrone_benchmark",
    "evaluate_nanodrone_model_artifact",
    "evaluate_x8_reference_models",
    "evaluate_fixedwing_gate",
    "make_trajectory_spec",
    "nanodrone_trajectory_spec",
    "save_nanodrone_benchmark_report",
    "benchmark_source_groups",
    "save_idf_corpus_report",
    "save_x8_reference_report",
    "save_fixedwing_gate",
    "screen_fixedwing_airframe_candidate",
    "load_trajectory_npz",
    "initial_residual_parameters",
    "kinematic_persistence_windowed_metrics",
    "model_family",
    "one_step_innovation_diagnostics",
    "state_kinematic_compatibility_diagnostics",
    "rollout",
    "rollout_divergence_metrics",
    "rollout_with_latent",
    "residual_initialization_statistics",
    "rollout_loss_configuration",
    "save_trajectory_npz",
    "split_trajectory",
    "step",
    "step_with_latent",
    "structured_parameters",
    "trajectory_segment",
    "trajectory_windows",
    "x8_trajectory_spec",
]
