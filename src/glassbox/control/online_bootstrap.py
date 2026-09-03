"""Recursive, support-aware identification and control from motor I/O alone.

This module is the continuous counterpart to :mod:`bootstrap_identification`.
It keeps a working local belief updated after every measured actuation interval
and gives control authority only to output directions supported by that belief.
There is deliberately no evidence-collection/model-running phase boundary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from glassbox.control._common import (
    ThrustCascade,
    finite_vector,
    immutable_array,
    quaternion_to_rotation,
    thrust_cascade,
)
from glassbox.core.dynamics import GRAVITY_M_S2


@dataclass(frozen=True)
class RecursiveBootstrapConfig:
    """Known I/O contract and evidence thresholds for the working belief."""

    command_minimum: float | tuple[float, float, float, float] = 0.0
    command_maximum: float | tuple[float, float, float, float] = 1.0
    #: Retained for report compatibility and pinned to 1.0.  A value below one
    #: decays the working belief to rank zero once excitation stops, and the
    #: committed excitation cap is far too small to rebuild that rank, so no
    #: later candidate could ever be proposed.  Any other value is rejected.
    forgetting_factor: float = 1.0
    command_rank_relative_tolerance: float = 0.025
    minimum_normalized_command_rms: float = 0.003
    nuisance_rank_relative_tolerance: float = 0.002
    output_rank_relative_tolerance: float = 0.04
    minimum_information_singular_value: float = 0.005
    full_authority_information_singular_value: float = 0.025
    minimum_effect_signal_to_noise: float = 1.0
    full_authority_effect_signal_to_noise: float = 3.0
    collective_residual_std_floor_m_s2: float = 0.05
    angular_residual_std_floor_rad_s2: float = 0.50
    #: Assimilate one sample per this many measured transitions, built from
    #: the window's mean features and mean targets and weighted by the window
    #: length.  Differencing a noisy measurement over one interval multiplies
    #: the noise by the loop rate; the mean over a window telescopes most of
    #: it away, while the weight keeps the sample count, the support
    #: thresholds, and the residual floor exactly per transition, so with a
    #: noise-free measurement the information rate is unchanged.  One means
    #: every transition is its own sample, which is bit-for-bit the identifier
    #: as it was.
    transition_aggregation_steps: int = 1
    #: Floor each residual scale at the belief's own recent prediction error:
    #: the error it makes predicting each new transition before absorbing it,
    #: averaged with exponential forgetting over
    #: ``minimum_certification_interval_count`` transitions.  The in-sample
    #: residual of a fit with as many samples as parameters is nothing, so a
    #: rank-deficient map on a tumbling vehicle reads as certain; the
    #: prequential error is what it actually gets wrong, and it falls as soon
    #: as the map is right.
    prequential_residual: bool = False
    minimum_certification_interval_count: int = 48
    validation_interval_count: int = 16
    minimum_validation_improvement: float = 0.02
    maximum_model_movement_fraction: float = 0.25
    proposal_cooldown_interval_count: int = 16
    #: Which belief the controller flies.  ``"certified"`` hands it the frozen
    #: snapshot the prequential transaction last admitted, so a candidate has to
    #: out-predict the incumbent before it can steer.  ``"working"`` hands it the
    #: continuously updated belief as soon as that belief's own support
    #: conditions hold, and lets authority scaling rather than a quality test
    #: decide how far to trust it.  Working mode never certifies a belief for
    #: control; the transaction it would have run is still scored and reported
    #: under the ``shadow_`` properties.
    control_model: Literal["certified", "working"] = "certified"
    #: Solve the point estimate, ranks, support, and covariances over a staged
    #: column set rather than over every accumulated regressor at once.  The
    #: full Gram and right-hand side are accumulated either way and nothing is
    #: discarded; staging only decides which columns the *solve* runs over.
    #: Stage one is the command block plus the intercept, which is the smallest
    #: system whose residualization is exact (centering) rather than fitted;
    #: stage two admits the nuisance regressors and is bit-for-bit the solve
    #: this identifier has always performed.  Off by default, so the certified
    #: and working control modes are unchanged.
    staged_regressors: bool = False
    #: Effective samples per regressor required before the nuisance block is
    #: admitted, as a pure ratio of counts.  The Schur complement the fit takes
    #: projects the command features onto the orthogonal complement of the
    #: fitted nuisance span, and with ``p`` regressors and ``n`` samples that
    #: projection removes a fraction of order ``p / n`` of the command energy
    #: purely by the fit's own freedom; equivalently the smallest eigenvalue of
    #: a sample Gram sits near ``(1 - sqrt(p / n))**2`` of its population value.
    #: At ``n = 4 p`` both readings bound the damage at a quarter, so the rank
    #: the fit reports is a statement about the design rather than about the
    #: nuisance block's slack.  It is a ratio of counts and refers to nothing
    #: about a vehicle.
    staging_sample_multiple: float = 4.0
    #: Treat the normalized command channel as thrust fraction: more collective
    #: command means more specific force along body z.  The fitted collective
    #: command coefficients are then projected onto the nonnegative orthant, so
    #: a confounded early estimate cannot claim a motor pushes the vehicle down.
    #: The projection is qualitative and carries no magnitude prior: it moves a
    #: negative coefficient to exactly zero and leaves every nonnegative one
    #: untouched.  Off by default.
    enforce_collective_sign: bool = False

    def __post_init__(self) -> None:
        minimum = finite_vector("command_minimum", self.command_minimum, 4)
        maximum = finite_vector("command_maximum", self.command_maximum, 4)
        if np.any(minimum >= maximum):
            raise ValueError("command_minimum must be below command_maximum")
        if self.forgetting_factor != 1.0:
            raise ValueError(
                "forgetting_factor must be exactly 1.0: post-certification "
                "excitation is capped at a small fraction of the command span, "
                "so a decaying working belief loses rank once excitation stops "
                "and can never regain it, leaving the vehicle flying forever on "
                "a belief no later proposal can replace"
            )
        for name in (
            "command_rank_relative_tolerance",
            "minimum_normalized_command_rms",
            "nuisance_rank_relative_tolerance",
            "output_rank_relative_tolerance",
            "minimum_validation_improvement",
            "maximum_model_movement_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if (
            not isinstance(self.transition_aggregation_steps, int)
            or self.transition_aggregation_steps < 1
        ):
            raise ValueError("transition_aggregation_steps must be a positive integer")
        positive_fields = (
            "minimum_information_singular_value",
            "full_authority_information_singular_value",
            "minimum_effect_signal_to_noise",
            "full_authority_effect_signal_to_noise",
            "collective_residual_std_floor_m_s2",
            "angular_residual_std_floor_rad_s2",
        )
        for name in positive_fields:
            if (
                not math.isfinite(float(getattr(self, name)))
                or getattr(self, name) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.full_authority_information_singular_value <= (
            self.minimum_information_singular_value
        ):
            raise ValueError("full information threshold must exceed its minimum")
        if self.full_authority_effect_signal_to_noise <= (
            self.minimum_effect_signal_to_noise
        ):
            raise ValueError("full effect signal-to-noise must exceed its minimum")
        if self.minimum_certification_interval_count < 16:
            raise ValueError("certification needs at least sixteen fitted intervals")
        if self.validation_interval_count < 4:
            raise ValueError("validation_interval_count must be at least four")
        if self.proposal_cooldown_interval_count < 0:
            raise ValueError("proposal_cooldown_interval_count cannot be negative")
        if self.control_model not in ("certified", "working"):
            raise ValueError("control_model must be 'certified' or 'working'")
        if (
            not math.isfinite(float(self.staging_sample_multiple))
            or self.staging_sample_multiple < 1.0
        ):
            raise ValueError(
                "staging_sample_multiple must be finite and at least one: a "
                "staged system solved from fewer samples than it has columns "
                "is not a fit"
            )
        object.__setattr__(self, "command_minimum", tuple(minimum))
        object.__setattr__(self, "command_maximum", tuple(maximum))


@dataclass(frozen=True)
class RecursiveBootstrapBelief:
    """One auditable snapshot of the continuously updated local model."""

    interval_count: int
    effective_interval_count: float
    collective_acceleration_per_command: np.ndarray
    collective_velocity_coefficient: np.ndarray
    collective_intercept_m_s2: float
    angular_acceleration_per_command: np.ndarray
    angular_rate_coefficient: np.ndarray
    angular_rate_product_coefficient: np.ndarray
    angular_intercept_rad_s2: np.ndarray
    normalized_command_support_projector: np.ndarray
    normalized_command_singular_values: np.ndarray
    normalized_command_information: np.ndarray
    supported_collective_effect_covariance: np.ndarray
    supported_angular_effect_covariance: np.ndarray
    #: The raw accumulated, forgetting-weighted Gram of each regression, in the
    #: identifier's own feature order: ``[normalized command (4), body velocity
    #: (3), 1]`` for the collective fit and ``[normalized command (4), body rate
    #: (3), rate products (3), 1]`` for the angular one.  These are the whole
    #: evidence the fits are solved from, before any support rule, Schur
    #: complement, or rescaling to raw command units; the two covariance fields
    #: above are what that evidence implies about the *command* block alone.
    #: A planner that wants the posterior of the full regressor set — because
    #: its plan moves the nuisance regressors as well as the commands — needs
    #: these rather than the reduced summaries.
    collective_information: np.ndarray
    angular_information: np.ndarray
    command_evidence_rank: int
    angular_effect_rank: int
    collective_nuisance_rank: int
    angular_nuisance_rank: int
    angular_output_support_projector: np.ndarray
    collective_support_fraction: float
    minimum_supported_information_singular_value: float
    information_authority: float
    collective_effect_signal_to_noise: float
    angular_effect_signal_to_noise: np.ndarray
    collective_residual_std_m_s2: float
    angular_residual_std_rad_s2: np.ndarray
    exploration_completion: float
    collective_authority: float
    angular_axis_authority: np.ndarray
    hover_command: np.ndarray | None
    update_wall_time_s: float
    #: Whether each regression's nuisance block has been admitted to the solve.
    #: Both are true whenever staging is off, which is the default, so a belief
    #: that never staged is indistinguishable from one that finished staging.
    collective_nuisance_staged: bool = True
    angular_nuisance_staged: bool = True
    #: Interval at which each nuisance block was admitted, ``None`` while the
    #: solve is still running on the command block and the intercept alone.
    collective_staging_interval_count: int | None = None
    angular_staging_interval_count: int | None = None
    #: How many collective command coefficients this update's sign projection
    #: moved, and the norm of what it removed, in specific force per unit
    #: command.  Both are zero when the projection is off or did not fire.
    collective_sign_projection_count: int = 0
    collective_sign_projection_magnitude: float = 0.0

    def __post_init__(self) -> None:
        arrays = {
            "collective_acceleration_per_command": (4,),
            "collective_velocity_coefficient": (3,),
            "angular_acceleration_per_command": (3, 4),
            "angular_rate_coefficient": (3, 3),
            "angular_rate_product_coefficient": (3, 3),
            "angular_intercept_rad_s2": (3,),
            "normalized_command_support_projector": (4, 4),
            "normalized_command_singular_values": (4,),
            "normalized_command_information": (4, 4),
            "supported_collective_effect_covariance": (4, 4),
            "supported_angular_effect_covariance": (3, 4, 4),
            "collective_information": (8, 8),
            "angular_information": (11, 11),
            "angular_output_support_projector": (3, 3),
            "angular_effect_signal_to_noise": (3,),
            "angular_residual_std_rad_s2": (3,),
            "angular_axis_authority": (3,),
        }
        for name, shape in arrays.items():
            object.__setattr__(
                self,
                name,
                immutable_array(getattr(self, name), shape, name),
            )
        if self.hover_command is not None:
            object.__setattr__(
                self,
                "hover_command",
                immutable_array(self.hover_command, (4,), "hover_command"),
            )
        scalars = (
            self.effective_interval_count,
            self.collective_intercept_m_s2,
            self.collective_support_fraction,
            self.minimum_supported_information_singular_value,
            self.information_authority,
            self.collective_effect_signal_to_noise,
            self.collective_residual_std_m_s2,
            self.exploration_completion,
            self.collective_authority,
            self.update_wall_time_s,
        )
        if self.interval_count < 0 or not np.all(np.isfinite(scalars)):
            raise ValueError("recursive belief counts and scalars must be finite")
        if not (
            0 <= self.command_evidence_rank <= 4
            and 0 <= self.angular_effect_rank <= 3
            and 0 <= self.collective_nuisance_rank <= 4
            and 0 <= self.angular_nuisance_rank <= 7
        ):
            raise ValueError("recursive belief ranks lie outside model dimensions")
        if not (
            0.0 <= self.collective_support_fraction <= 1.0 + 1e-9
            and 0.0 <= self.information_authority <= 1.0
            and 0.0 <= self.exploration_completion <= 1.0
            and 0.0 <= self.collective_authority <= 1.0
        ):
            raise ValueError(
                "recursive belief support and authority must lie in [0, 1]"
            )
        if (
            self.collective_effect_signal_to_noise < 0.0
            or np.any(self.angular_effect_signal_to_noise < 0.0)
            or self.collective_residual_std_m_s2 <= 0.0
            or np.any(self.angular_residual_std_rad_s2 <= 0.0)
        ):
            raise ValueError("recursive uncertainty statistics must be positive")
        if np.any(self.angular_axis_authority < 0.0) or np.any(
            self.angular_axis_authority > 1.0
        ):
            raise ValueError("angular authority must lie inside [0, 1]")
        if not 0 <= self.collective_sign_projection_count <= 4:
            raise ValueError("sign projection cannot move more than four commands")
        if (
            not math.isfinite(self.collective_sign_projection_magnitude)
            or self.collective_sign_projection_magnitude < 0.0
        ):
            raise ValueError("sign projection magnitude must be finite and nonnegative")
        for name in (
            "collective_staging_interval_count",
            "angular_staging_interval_count",
        ):
            staged_at = getattr(self, name)
            if staged_at is not None and staged_at < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def has_any_control_authority(self) -> bool:
        return bool(
            self.collective_authority > 0.0 or np.any(self.angular_axis_authority > 0.0)
        )

    def predict_collective_specific_force(
        self,
        command: Sequence[float],
        body_velocity_m_s: Sequence[float],
    ) -> float:
        command_array = finite_vector("command", command, 4)
        body_velocity = finite_vector("body_velocity_m_s", body_velocity_m_s, 3)
        return float(
            self.collective_acceleration_per_command @ command_array
            + self.collective_velocity_coefficient @ body_velocity
            + self.collective_intercept_m_s2
        )

    def predict_angular_acceleration(
        self,
        command: Sequence[float],
        angular_velocity_rad_s: Sequence[float],
    ) -> np.ndarray:
        command_array = finite_vector("command", command, 4)
        angular_velocity = finite_vector(
            "angular_velocity_rad_s", angular_velocity_rad_s, 3
        )
        rate_products = np.asarray(
            (
                angular_velocity[0] * angular_velocity[1],
                angular_velocity[0] * angular_velocity[2],
                angular_velocity[1] * angular_velocity[2],
            )
        )
        return (
            self.angular_acceleration_per_command @ command_array
            + self.angular_rate_coefficient @ angular_velocity
            + self.angular_rate_product_coefficient @ rate_products
            + self.angular_intercept_rad_s2
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "recursive_rank_supported_multirotor_bootstrap_v1",
            "airframe_parameter_prior_used": False,
            "canonical_motor_mixer_assumed": False,
            "interval_count": self.interval_count,
            "effective_interval_count": self.effective_interval_count,
            "collective_acceleration_per_command": (
                self.collective_acceleration_per_command.tolist()
            ),
            "collective_velocity_coefficient": (
                self.collective_velocity_coefficient.tolist()
            ),
            "collective_intercept_m_s2": self.collective_intercept_m_s2,
            "angular_acceleration_per_command": (
                self.angular_acceleration_per_command.tolist()
            ),
            "angular_rate_coefficient": self.angular_rate_coefficient.tolist(),
            "angular_rate_product_coefficient": (
                self.angular_rate_product_coefficient.tolist()
            ),
            "angular_intercept_rad_s2": self.angular_intercept_rad_s2.tolist(),
            "normalized_command_support_projector": (
                self.normalized_command_support_projector.tolist()
            ),
            "normalized_command_singular_values": (
                self.normalized_command_singular_values.tolist()
            ),
            "normalized_command_information": (
                self.normalized_command_information.tolist()
            ),
            "supported_collective_effect_covariance": (
                self.supported_collective_effect_covariance.tolist()
            ),
            "supported_angular_effect_covariance": (
                self.supported_angular_effect_covariance.tolist()
            ),
            "collective_information": self.collective_information.tolist(),
            "angular_information": self.angular_information.tolist(),
            "effect_covariance_scope": "supported_subspace_only",
            "command_evidence_rank": self.command_evidence_rank,
            "angular_effect_rank": self.angular_effect_rank,
            "collective_nuisance_rank": self.collective_nuisance_rank,
            "angular_nuisance_rank": self.angular_nuisance_rank,
            "angular_output_support_projector": (
                self.angular_output_support_projector.tolist()
            ),
            "collective_support_fraction": self.collective_support_fraction,
            "minimum_supported_information_singular_value": (
                self.minimum_supported_information_singular_value
            ),
            "information_authority": self.information_authority,
            "collective_effect_signal_to_noise": (
                self.collective_effect_signal_to_noise
            ),
            "angular_effect_signal_to_noise": (
                self.angular_effect_signal_to_noise.tolist()
            ),
            "collective_residual_std_m_s2": self.collective_residual_std_m_s2,
            "angular_residual_std_rad_s2": (self.angular_residual_std_rad_s2.tolist()),
            "exploration_completion": self.exploration_completion,
            "collective_authority": self.collective_authority,
            "angular_axis_authority": self.angular_axis_authority.tolist(),
            "hover_command": (
                None if self.hover_command is None else self.hover_command.tolist()
            ),
            "update_wall_time_s": self.update_wall_time_s,
            "collective_nuisance_staged": self.collective_nuisance_staged,
            "angular_nuisance_staged": self.angular_nuisance_staged,
            "collective_staging_interval_count": (
                self.collective_staging_interval_count
            ),
            "angular_staging_interval_count": self.angular_staging_interval_count,
            "collective_sign_projection_count": self.collective_sign_projection_count,
            "collective_sign_projection_magnitude": (
                self.collective_sign_projection_magnitude
            ),
        }


@dataclass(frozen=True)
class RecursiveBeliefValidationReport:
    """Prequential evidence for one frozen candidate admission decision."""

    candidate_interval_count: int
    reference_interval_count: int | None
    validation_interval_count: int
    initial_admission: bool
    candidate_collective_rmse_m_s2: float
    reference_collective_rmse_m_s2: float
    collective_improvement: float
    candidate_angular_rmse_rad_s2: np.ndarray
    reference_angular_rmse_rad_s2: np.ndarray
    angular_improvement: np.ndarray
    model_movement_fraction: float
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_angular_rmse_rad_s2",
            immutable_array(
                self.candidate_angular_rmse_rad_s2,
                (3,),
                "candidate_angular_rmse_rad_s2",
            ),
        )
        object.__setattr__(
            self,
            "reference_angular_rmse_rad_s2",
            immutable_array(
                self.reference_angular_rmse_rad_s2,
                (3,),
                "reference_angular_rmse_rad_s2",
            ),
        )
        object.__setattr__(
            self,
            "angular_improvement",
            immutable_array(self.angular_improvement, (3,), "angular_improvement"),
        )
        scalars = (
            self.candidate_collective_rmse_m_s2,
            self.reference_collective_rmse_m_s2,
            self.collective_improvement,
            self.model_movement_fraction,
        )
        if (
            self.candidate_interval_count < 1
            or self.validation_interval_count < 1
            or not np.all(np.isfinite(scalars))
        ):
            raise ValueError("validation counts and scalar metrics must be finite")
        if self.initial_admission != (self.reference_interval_count is None):
            raise ValueError("initial admission must have no reference belief")
        if self.model_movement_fraction < 0.0:
            raise ValueError("model movement cannot be negative")
        if not self.reason:
            raise ValueError("validation reason cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_interval_count": self.candidate_interval_count,
            "reference_interval_count": self.reference_interval_count,
            "validation_interval_count": self.validation_interval_count,
            "initial_admission": self.initial_admission,
            "candidate_collective_rmse_m_s2": (self.candidate_collective_rmse_m_s2),
            "reference_collective_rmse_m_s2": (self.reference_collective_rmse_m_s2),
            "collective_improvement": self.collective_improvement,
            "candidate_angular_rmse_rad_s2": (
                self.candidate_angular_rmse_rad_s2.tolist()
            ),
            "reference_angular_rmse_rad_s2": (
                self.reference_angular_rmse_rad_s2.tolist()
            ),
            "angular_improvement": self.angular_improvement.tolist(),
            "model_movement_fraction": self.model_movement_fraction,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RecursiveBootstrapSampleReport:
    """Whether one offered transition was assimilated, and why not."""

    interval_count: int
    accepted: bool
    reason: str
    update_wall_time_s: float

    def __post_init__(self) -> None:
        if self.interval_count < 0 or not math.isfinite(self.update_wall_time_s):
            raise ValueError("sample report counts and timings must be finite")
        if not self.reason:
            raise ValueError("sample report reason cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_count": self.interval_count,
            "accepted": self.accepted,
            "reason": self.reason,
            "update_wall_time_s": self.update_wall_time_s,
        }


@dataclass
class _PendingBeliefProposal:
    candidate: RecursiveBootstrapBelief
    reference: RecursiveBootstrapBelief | None
    baseline_force_nuisance: np.ndarray
    baseline_angular_nuisance: np.ndarray
    candidate_force_squared_error: float = 0.0
    reference_force_squared_error: float = 0.0
    candidate_angular_squared_error: np.ndarray | None = None
    reference_angular_squared_error: np.ndarray | None = None
    validation_count: int = 0

    def __post_init__(self) -> None:
        self.candidate_angular_squared_error = np.zeros(3, dtype=np.float64)
        self.reference_angular_squared_error = np.zeros(3, dtype=np.float64)


class _UnusableActionInput(Exception):
    """Internal signal naming why one control input cannot be acted on."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _StabilizingFeedback:
    """The fixed cascade evaluated once, with its allocated motor command."""

    rotation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    cascade: ThrustCascade
    command: np.ndarray


@dataclass(frozen=True)
class _InformationAction:
    """The excitation direction, amplitude, and geometry chosen for one step."""

    amplitude: float
    target: np.ndarray
    information: np.ndarray
    floor: float


@dataclass(frozen=True)
class _CandidateScores:
    """Every term of the excitation-scan objective, one entry per candidate."""

    stabilization: np.ndarray
    information_reward: np.ndarray
    uncertainty: np.ndarray
    altitude_risk: np.ndarray
    objective: np.ndarray
    predicted_world_velocity: np.ndarray


@dataclass(frozen=True)
class _SampleFeatures:
    """One measured transition reduced to regression targets and features."""

    command: np.ndarray
    body_specific_force: np.ndarray
    body_velocity: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    rate_products: np.ndarray
    force_features: np.ndarray
    angular_features: np.ndarray


@dataclass(frozen=True)
class _EffectFit:
    """Both support-restricted regressions and their residual scales."""

    force_effect: np.ndarray
    force_nuisance: np.ndarray
    force_support: np.ndarray
    force_rank: int
    force_nuisance_rank: int
    force_residual_inverse: np.ndarray
    force_intercept: float
    force_residual_std: float
    collective_effect_covariance: np.ndarray
    angular_effect: np.ndarray
    angular_nuisance: np.ndarray
    angular_support: np.ndarray
    angular_singular_values: np.ndarray
    angular_command_rank: int
    angular_nuisance_rank: int
    angular_residual_information: np.ndarray
    angular_intercept: np.ndarray
    angular_residual_std: np.ndarray
    angular_effect_covariance: np.ndarray
    force_nuisance_staged: bool
    angular_nuisance_staged: bool
    collective_sign_projection_count: int
    collective_sign_projection_magnitude: float


@dataclass(frozen=True)
class _BeliefAuthority:
    """How much of each output direction one fit is entitled to command."""

    collective_support: float
    angular_output_support: np.ndarray
    angular_effect_rank: int
    minimum_supported_information: float
    information_authority: float
    angular_effect_signal_to_noise: np.ndarray
    angular_axis_authority: np.ndarray
    hover_command: np.ndarray | None
    collective_authority: float
    collective_effect_signal_to_noise: float
    exploration_completion: float


@dataclass(frozen=True)
class ProgressiveBootstrapCommand:
    """One stabilizing cascade command plus its scanned information action."""

    command: np.ndarray
    objective_value: float
    stabilization_cost: float
    information_reward: float
    uncertainty_cost: float
    altitude_risk_cost: float
    estimated_information_gain: float
    information_action_fraction: float
    information_completion: float
    predicted_world_velocity_m_s: np.ndarray
    predicted_angular_velocity_rad_s: np.ndarray
    desired_world_acceleration_m_s2: np.ndarray
    desired_angular_acceleration_rad_s2: np.ndarray
    collective_authority: float
    angular_axis_authority: np.ndarray
    command_usable: bool = True
    reason: str = "stabilizing_cascade_with_scanned_excitation"

    def __post_init__(self) -> None:
        for name, shape in {
            "command": (4,),
            "predicted_world_velocity_m_s": (3,),
            "predicted_angular_velocity_rad_s": (3,),
            "desired_world_acceleration_m_s2": (3,),
            "desired_angular_acceleration_rad_s2": (3,),
            "angular_axis_authority": (3,),
        }.items():
            object.__setattr__(
                self,
                name,
                immutable_array(getattr(self, name), shape, name),
            )
        scalars = (
            self.collective_authority,
            self.objective_value,
            self.stabilization_cost,
            self.information_reward,
            self.uncertainty_cost,
            self.altitude_risk_cost,
            self.estimated_information_gain,
            self.information_action_fraction,
            self.information_completion,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("command objective diagnostics must be finite")
        if not self.reason:
            raise ValueError("command reason cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.tolist(),
            "command_usable": self.command_usable,
            "reason": self.reason,
            "objective_value": self.objective_value,
            "stabilization_cost": self.stabilization_cost,
            "information_reward": self.information_reward,
            "uncertainty_cost": self.uncertainty_cost,
            "altitude_risk_cost": self.altitude_risk_cost,
            "estimated_information_gain": self.estimated_information_gain,
            "information_action_fraction": self.information_action_fraction,
            "information_completion": self.information_completion,
            "collective_authority": self.collective_authority,
            "angular_axis_authority": self.angular_axis_authority.tolist(),
        }


class RecursiveBootstrapIdentifier:
    """Update direct motor effects after every observed actuation interval."""

    _FORCE_NUISANCE_SIZE = 4
    _ANGULAR_NUISANCE_SIZE = 7
    #: Applied commands are accepted this far outside the configured bounds and
    #: then clipped, so a saturated actuator readback cannot end an estimator.
    _BOUND_TOLERANCE_FRACTION = 1e-6

    def __init__(self, config: RecursiveBootstrapConfig | None = None) -> None:
        self.config = RecursiveBootstrapConfig() if config is None else config
        self._minimum = np.asarray(self.config.command_minimum)
        self._maximum = np.asarray(self.config.command_maximum)
        self._span = self._maximum - self._minimum
        self._midpoint = 0.5 * (self._minimum + self._maximum)
        self._interval_count = 0
        self._weight = 0.0
        self._pending_transitions: list[_SampleFeatures] = []
        self._prequential_force_sum = 0.0
        self._prequential_angular_sum = np.zeros(3, dtype=np.float64)
        self._prequential_weight = 0.0
        self._force_gram = np.zeros((8, 8), dtype=np.float64)
        self._force_rhs = np.zeros((8, 1), dtype=np.float64)
        self._force_target_sum_squares = 0.0
        self._angular_gram = np.zeros((11, 11), dtype=np.float64)
        self._angular_rhs = np.zeros((11, 3), dtype=np.float64)
        self._angular_target_sum_squares = np.zeros(3, dtype=np.float64)
        self._force_nuisance_staged = not self.config.staged_regressors
        self._angular_nuisance_staged = not self.config.staged_regressors
        self._force_staged_interval: int | None = None
        self._angular_staged_interval: int | None = None
        self._belief = self._empty_belief()
        self._working_support_reached = False
        self._certified_belief: RecursiveBootstrapBelief | None = None
        self._pending_proposal: _PendingBeliefProposal | None = None
        self._validation_history: list[RecursiveBeliefValidationReport] = []
        self._last_proposal_finished_interval = -(10**9)
        self._last_sample_report = RecursiveBootstrapSampleReport(
            interval_count=0,
            accepted=False,
            reason="no_sample_offered",
            update_wall_time_s=0.0,
        )
        self._rejected_sample_count = 0

    def _empty_belief(self) -> RecursiveBootstrapBelief:
        return RecursiveBootstrapBelief(
            interval_count=0,
            effective_interval_count=0.0,
            collective_acceleration_per_command=np.zeros(4),
            collective_velocity_coefficient=np.zeros(3),
            collective_intercept_m_s2=0.0,
            angular_acceleration_per_command=np.zeros((3, 4)),
            angular_rate_coefficient=np.zeros((3, 3)),
            angular_rate_product_coefficient=np.zeros((3, 3)),
            angular_intercept_rad_s2=np.zeros(3),
            normalized_command_support_projector=np.zeros((4, 4)),
            normalized_command_singular_values=np.zeros(4),
            normalized_command_information=np.zeros((4, 4)),
            supported_collective_effect_covariance=np.zeros((4, 4)),
            supported_angular_effect_covariance=np.zeros((3, 4, 4)),
            collective_information=np.zeros((8, 8)),
            angular_information=np.zeros((11, 11)),
            command_evidence_rank=0,
            angular_effect_rank=0,
            collective_nuisance_rank=0,
            angular_nuisance_rank=0,
            angular_output_support_projector=np.zeros((3, 3)),
            collective_support_fraction=0.0,
            minimum_supported_information_singular_value=0.0,
            information_authority=0.0,
            collective_effect_signal_to_noise=0.0,
            angular_effect_signal_to_noise=np.zeros(3),
            collective_residual_std_m_s2=(
                self.config.collective_residual_std_floor_m_s2
            ),
            angular_residual_std_rad_s2=np.full(
                3,
                self.config.angular_residual_std_floor_rad_s2,
            ),
            exploration_completion=0.0,
            collective_authority=0.0,
            angular_axis_authority=np.zeros(3),
            hover_command=None,
            update_wall_time_s=0.0,
            collective_nuisance_staged=self._force_nuisance_staged,
            angular_nuisance_staged=self._angular_nuisance_staged,
            collective_staging_interval_count=self._force_staged_interval,
            angular_staging_interval_count=self._angular_staged_interval,
        )

    @property
    def belief(self) -> RecursiveBootstrapBelief:
        return self._belief

    @property
    def flies_working_belief(self) -> bool:
        """Whether control tracks the working belief instead of a snapshot."""

        return self.config.control_model == "working"

    @staticmethod
    def _belief_is_supported(belief: RecursiveBootstrapBelief) -> bool:
        """Whether one belief spans everything control has to command.

        These are the same support conditions the certification transaction
        already requires of a candidate, and nothing else: the command evidence
        spans all four motors, the fitted angular effect spans all three body
        axes, and the collective effect implies a hover command inside the
        command box.  Prequential improvement is deliberately not part of it.
        """

        return bool(
            belief.command_evidence_rank == 4
            and belief.angular_effect_rank == 3
            and belief.hover_command is not None
        )

    @property
    def working_belief_supported(self) -> bool:
        """Whether the working belief currently meets the support conditions."""

        return self._belief_is_supported(self._belief)

    @property
    def working_support_reached(self) -> bool:
        """Whether the working belief has ever met the support conditions."""

        return self._working_support_reached

    @property
    def control_model_ready(self) -> bool:
        """Whether an identified model is the one being flown.

        Certified mode reads this from the transaction and working mode from
        the working belief's own support, so a caller can ask one question
        without knowing which mode it is in.  Both latch: an admitted snapshot
        is never withdrawn, and working mode likewise commits once support has
        been demonstrated, rather than reverting to pre-handover behaviour every
        time an interval leaves the fit briefly short of full rank.
        """

        if self.flies_working_belief:
            return self._working_support_reached
        return self._certified_belief is not None

    @property
    def certified_belief(self) -> RecursiveBootstrapBelief | None:
        """Last frozen belief admitted by future predictive validation.

        Working mode never certifies a belief for control, so this is ``None``
        there and the shadow transaction is read through
        :attr:`shadow_certified_belief` instead.
        """

        return None if self.flies_working_belief else self._certified_belief

    @property
    def predictive_belief(self) -> RecursiveBootstrapBelief:
        """The belief supplied to the joint objective as its predictive mean."""

        if self.flies_working_belief:
            return self._belief
        return (
            self._belief if self._certified_belief is None else self._certified_belief
        )

    @property
    def control_belief(self) -> RecursiveBootstrapBelief:
        """Compatibility alias for :attr:`predictive_belief`."""

        return self.predictive_belief

    @property
    def pending_proposal(self) -> bool:
        return not self.flies_working_belief and self._pending_proposal is not None

    @property
    def validation_history(self) -> tuple[RecursiveBeliefValidationReport, ...]:
        return () if self.flies_working_belief else tuple(self._validation_history)

    @property
    def accepted_update_count(self) -> int:
        return sum(report.accepted for report in self.validation_history)

    @property
    def rejected_update_count(self) -> int:
        return sum(not report.accepted for report in self.validation_history)

    @property
    def shadow_certified_belief(self) -> RecursiveBootstrapBelief | None:
        """What the transaction would have been flying in working mode.

        The shadow evaluation runs the ordinary freeze, score, and admit cycle
        against the trajectory working mode actually flew.  Nothing it produces
        reaches the controller, so recording it costs one pending proposal and
        the sixteen predictions that score it.
        """

        return self._certified_belief if self.flies_working_belief else None

    @property
    def shadow_pending_proposal(self) -> bool:
        return self.flies_working_belief and self._pending_proposal is not None

    @property
    def shadow_validation_history(self) -> tuple[RecursiveBeliefValidationReport, ...]:
        return tuple(self._validation_history) if self.flies_working_belief else ()

    @property
    def shadow_accepted_update_count(self) -> int:
        return sum(report.accepted for report in self.shadow_validation_history)

    @property
    def shadow_rejected_update_count(self) -> int:
        return sum(not report.accepted for report in self.shadow_validation_history)

    @property
    def last_sample_report(self) -> RecursiveBootstrapSampleReport:
        """Whether the most recently offered transition was assimilated."""

        return self._last_sample_report

    @property
    def rejected_sample_count(self) -> int:
        """Transitions refused because they were not usable evidence."""

        return self._rejected_sample_count

    @staticmethod
    def _nuisance_inverse(
        nuisance_gram: np.ndarray,
        *,
        relative_tolerance: float,
    ) -> tuple[np.ndarray, int]:
        """Invert only the nuisance directions the window actually excited.

        Nuisance features carry their own physical units, so no normalized span
        is available.  The constant intercept feature supplies the missing unit
        scale: its root-mean-square is exactly one, so the leading nuisance
        direction always has at least unit root-mean-square and a threshold
        relative to it is also an absolute floor in the features' own units.
        Directions below the threshold are dropped rather than inverted, which
        is the same rank-support rule the command directions already follow.
        """

        symmetric = 0.5 * (nuisance_gram + nuisance_gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        singular_values = np.sqrt(eigenvalues)
        leading = float(np.max(singular_values)) if singular_values.size else 0.0
        threshold = max(leading * relative_tolerance, 1e-12)
        supported = singular_values >= threshold
        inverse_eigenvalues = np.where(
            supported,
            1.0 / np.maximum(eigenvalues, threshold * threshold),
            0.0,
        )
        inverse = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
        return inverse, int(np.sum(supported))

    @classmethod
    def _supported_fit(
        cls,
        gram: np.ndarray,
        rhs: np.ndarray,
        *,
        nuisance_size: int,
        effective_count: float,
        relative_tolerance: float,
        minimum_rms: float,
        nuisance_relative_tolerance: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        np.ndarray,
        np.ndarray,
        int,
    ]:
        command_gram = gram[:4, :4]
        cross_gram = gram[:4, 4:]
        nuisance_gram = gram[4:, 4:]
        nuisance_inverse, nuisance_rank = cls._nuisance_inverse(
            nuisance_gram,
            relative_tolerance=nuisance_relative_tolerance,
        )
        residual_gram = command_gram - (cross_gram @ nuisance_inverse @ cross_gram.T)
        residual_gram = 0.5 * (residual_gram + residual_gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(residual_gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        right = eigenvectors[:, order]
        singular_values = np.sqrt(eigenvalues)
        leading = float(singular_values[0]) if len(singular_values) else 0.0
        threshold = max(
            leading * relative_tolerance,
            minimum_rms * math.sqrt(max(effective_count, 1.0)),
        )
        supported = singular_values >= threshold
        inverse_eigenvalues = np.where(
            supported,
            1.0 / np.maximum(eigenvalues, threshold * threshold),
            0.0,
        )
        residual_inverse = (right * inverse_eigenvalues) @ right.T
        residual_rhs = rhs[:4] - cross_gram @ nuisance_inverse @ rhs[4:]
        command_coefficient = residual_inverse @ residual_rhs
        nuisance_coefficient = nuisance_inverse @ (
            rhs[4:] - cross_gram.T @ command_coefficient
        )
        support_projector = (right * supported.astype(np.float64)) @ right.T
        if nuisance_coefficient.shape[0] != nuisance_size:
            raise RuntimeError("recursive nuisance feature shape changed")
        return (
            command_coefficient,
            nuisance_coefficient,
            support_projector,
            singular_values,
            int(np.sum(supported)),
            residual_inverse,
            residual_gram,
            nuisance_rank,
        )

    def _nuisance_admitted(self, nuisance_size: int) -> bool:
        """Whether the accumulated evidence can carry this nuisance block.

        The condition is a ratio of counts and nothing else: effective samples
        against the full column count of the regression, at the declared
        :attr:`RecursiveBootstrapConfig.staging_sample_multiple`.
        """

        if not self.config.staged_regressors:
            return True
        columns = 4 + nuisance_size
        return self._weight >= self.config.staging_sample_multiple * columns

    def _staged_fit(
        self,
        gram: np.ndarray,
        rhs: np.ndarray,
        *,
        nuisance_size: int,
        staged: bool,
        **thresholds: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        np.ndarray,
        np.ndarray,
        int,
    ]:
        """Solve one regression over the staged columns of the same Gram.

        The Gram and right-hand side are always the full accumulated ones; only
        the column set the solve runs over is staged.  Stage one keeps the
        command block and the intercept, so the command features are
        residualized against the intercept alone, which is exact centering
        rather than a fitted projection, and four command directions can be
        resolved from five samples.  Stage two is every column, and the fully
        staged branch hands the accumulated arrays through untouched so the
        solve is bit-for-bit the one this identifier has always run.

        The nuisance coefficient is returned at full size with the unstaged
        entries exactly zero, so every caller downstream sees one shape.
        """

        if staged:
            return self._supported_fit(
                gram,
                rhs,
                nuisance_size=nuisance_size,
                **thresholds,  # type: ignore[arg-type]
            )
        columns = np.concatenate((np.arange(4), np.asarray((3 + nuisance_size,))))
        result = self._supported_fit(
            gram[np.ix_(columns, columns)],
            rhs[columns],
            nuisance_size=1,
            **thresholds,  # type: ignore[arg-type]
        )
        nuisance = np.zeros((nuisance_size, rhs.shape[1]), dtype=np.float64)
        nuisance[-1] = result[1][0]
        return (result[0], nuisance, *result[2:])  # type: ignore[return-value]

    @staticmethod
    def _residual_standard_deviation(
        gram: np.ndarray,
        rhs: np.ndarray,
        target_sum_squares: float | np.ndarray,
        command_coefficient: np.ndarray,
        nuisance_coefficient: np.ndarray,
        *,
        effective_count: float,
        command_rank: int,
        nuisance_rank: int,
        floor: float,
    ) -> np.ndarray:
        coefficient = np.vstack((command_coefficient, nuisance_coefficient))
        squared_error = (
            np.asarray(target_sum_squares, dtype=np.float64)
            - 2.0 * np.sum(coefficient * rhs, axis=0)
            + np.sum(coefficient * (gram @ coefficient), axis=0)
        )
        degrees_of_freedom = max(
            effective_count - command_rank - nuisance_rank,
            1.0,
        )
        variance = np.maximum(squared_error / degrees_of_freedom, floor * floor)
        return np.sqrt(np.maximum(variance, 0.0))

    def _authority_fraction(self, value: float, minimum: float, full: float) -> float:
        return float(np.clip((value - minimum) / (full - minimum), 0.0, 1.0))

    @staticmethod
    def _model_movement_fraction(
        candidate: RecursiveBootstrapBelief,
        reference: RecursiveBootstrapBelief | None,
    ) -> float:
        if reference is None:
            return 0.0
        force_scale = max(
            float(np.linalg.norm(reference.collective_acceleration_per_command)),
            1.0,
        )
        angular_scale = max(
            float(np.linalg.norm(reference.angular_acceleration_per_command)),
            1.0,
        )
        force_movement = float(
            np.linalg.norm(
                candidate.collective_acceleration_per_command
                - reference.collective_acceleration_per_command
            )
            / force_scale
        )
        angular_movement = float(
            np.linalg.norm(
                candidate.angular_acceleration_per_command
                - reference.angular_acceleration_per_command
            )
            / angular_scale
        )
        if candidate.hover_command is None or reference.hover_command is None:
            hover_movement = 1.0
        else:
            hover_movement = float(
                np.linalg.norm(candidate.hover_command - reference.hover_command)
                / max(float(np.linalg.norm(reference.hover_command)), 1e-6)
            )
        return max(force_movement, angular_movement, hover_movement)

    def _start_proposal(self, candidate: RecursiveBootstrapBelief) -> None:
        tolerance = self.config.nuisance_rank_relative_tolerance
        force_inverse, _ = self._nuisance_inverse(
            self._force_gram[4:, 4:],
            relative_tolerance=tolerance,
        )
        angular_inverse, _ = self._nuisance_inverse(
            self._angular_gram[4:, 4:],
            relative_tolerance=tolerance,
        )
        baseline_force_nuisance = force_inverse @ self._force_rhs[4:]
        baseline_angular_nuisance = angular_inverse @ self._angular_rhs[4:]
        self._pending_proposal = _PendingBeliefProposal(
            candidate=candidate,
            reference=self._certified_belief,
            baseline_force_nuisance=baseline_force_nuisance,
            baseline_angular_nuisance=baseline_angular_nuisance,
        )

    def _score_pending_proposal(
        self,
        *,
        command: np.ndarray,
        body_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        rate_products: np.ndarray,
        collective_target: float,
        angular_target: np.ndarray,
    ) -> None:
        pending = self._pending_proposal
        if pending is None:
            return
        candidate_force = pending.candidate.predict_collective_specific_force(
            command,
            body_velocity,
        )
        candidate_angular = pending.candidate.predict_angular_acceleration(
            command,
            angular_velocity,
        )
        if pending.reference is None:
            force_nuisance = np.concatenate((body_velocity, np.ones(1)))
            angular_nuisance = np.concatenate(
                (angular_velocity, rate_products, np.ones(1))
            )
            reference_force = float(
                force_nuisance @ pending.baseline_force_nuisance[:, 0]
            )
            reference_angular = angular_nuisance @ pending.baseline_angular_nuisance
        else:
            reference_force = pending.reference.predict_collective_specific_force(
                command,
                body_velocity,
            )
            reference_angular = pending.reference.predict_angular_acceleration(
                command,
                angular_velocity,
            )
        pending.candidate_force_squared_error += (
            collective_target - candidate_force
        ) ** 2
        pending.reference_force_squared_error += (
            collective_target - reference_force
        ) ** 2
        assert pending.candidate_angular_squared_error is not None
        assert pending.reference_angular_squared_error is not None
        pending.candidate_angular_squared_error += np.square(
            angular_target - candidate_angular
        )
        pending.reference_angular_squared_error += np.square(
            angular_target - reference_angular
        )
        pending.validation_count += 1
        if pending.validation_count >= self.config.validation_interval_count:
            self._finish_pending_proposal()

    def _finish_pending_proposal(self) -> None:
        pending = self._pending_proposal
        if pending is None:
            return
        count = pending.validation_count
        candidate_force_rmse = math.sqrt(pending.candidate_force_squared_error / count)
        reference_force_rmse = math.sqrt(pending.reference_force_squared_error / count)
        assert pending.candidate_angular_squared_error is not None
        assert pending.reference_angular_squared_error is not None
        candidate_angular_rmse = np.sqrt(
            pending.candidate_angular_squared_error / count
        )
        reference_angular_rmse = np.sqrt(
            pending.reference_angular_squared_error / count
        )
        force_improvement = (reference_force_rmse - candidate_force_rmse) / max(
            reference_force_rmse,
            self.config.collective_residual_std_floor_m_s2,
        )
        angular_improvement = (
            reference_angular_rmse - candidate_angular_rmse
        ) / np.maximum(
            reference_angular_rmse,
            self.config.angular_residual_std_floor_rad_s2,
        )
        movement = self._model_movement_fraction(
            pending.candidate,
            pending.reference,
        )
        initial = pending.reference is None
        reference_stale = (
            not initial and pending.reference is not self._certified_belief
        )
        supported = self._belief_is_supported(pending.candidate)
        if initial:
            improved = bool(
                force_improvement >= self.config.minimum_validation_improvement
                and np.all(
                    angular_improvement >= self.config.minimum_validation_improvement
                )
            )
        else:
            improvements = np.concatenate(
                (np.asarray((force_improvement,)), angular_improvement)
            )
            improved = bool(
                np.mean(improvements) >= self.config.minimum_validation_improvement
                and np.all(improvements >= -self.config.minimum_validation_improvement)
            )
        movement_valid = bool(
            initial or movement <= self.config.maximum_model_movement_fraction
        )
        accepted = supported and improved and movement_valid and not reference_stale
        if reference_stale:
            reason = "stale_reference"
        elif not supported:
            reason = "candidate_lost_support"
        elif not movement_valid:
            reason = "model_movement_exceeded"
        elif not improved:
            reason = "prequential_improvement_not_demonstrated"
        elif initial:
            reason = "initial_prequential_admission"
        else:
            reason = "prequential_replacement_committed"
        report = RecursiveBeliefValidationReport(
            candidate_interval_count=pending.candidate.interval_count,
            reference_interval_count=(
                None if pending.reference is None else pending.reference.interval_count
            ),
            validation_interval_count=count,
            initial_admission=initial,
            candidate_collective_rmse_m_s2=candidate_force_rmse,
            reference_collective_rmse_m_s2=reference_force_rmse,
            collective_improvement=force_improvement,
            candidate_angular_rmse_rad_s2=candidate_angular_rmse,
            reference_angular_rmse_rad_s2=reference_angular_rmse,
            angular_improvement=angular_improvement,
            model_movement_fraction=movement,
            accepted=accepted,
            reason=reason,
        )
        self._validation_history.append(report)
        if accepted:
            self._certified_belief = pending.candidate
        self._pending_proposal = None
        self._last_proposal_finished_interval = self._interval_count

    def _maybe_start_proposal(self) -> None:
        if self._pending_proposal is not None:
            return
        if self._interval_count - self._last_proposal_finished_interval < (
            self.config.proposal_cooldown_interval_count
        ):
            return
        candidate = self._belief
        eligible = bool(
            candidate.interval_count >= self.config.minimum_certification_interval_count
            and self._belief_is_supported(candidate)
            and candidate.exploration_completion >= 0.75
        )
        if not eligible:
            return
        self._start_proposal(candidate)

    def _validated_sample(
        self,
        previous_state: Sequence[float],
        current_state: Sequence[float],
        average_applied_motor_command: Sequence[float],
        sample_period_s: float,
    ) -> tuple[str, None] | tuple[None, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Parse one transition, or name why it is not usable evidence.

        Applied commands within :attr:`_BOUND_TOLERANCE_FRACTION` of the span
        outside the bounds are clipped rather than refused, so a saturated
        actuator readback is still evidence.
        """

        parsed: list[np.ndarray] = []
        for name, value, size, reason in (
            ("previous_state", previous_state, 13, "previous_state_not_finite"),
            ("current_state", current_state, 13, "current_state_not_finite"),
            (
                "average_applied_motor_command",
                average_applied_motor_command,
                4,
                "applied_command_not_finite",
            ),
        ):
            try:
                parsed.append(finite_vector(name, value, size))
            except (ValueError, TypeError):
                return reason, None
        previous, current, command = parsed
        if float(np.linalg.norm(previous[6:10])) < 1e-9:
            return "previous_state_quaternion_degenerate", None
        tolerance = self._BOUND_TOLERANCE_FRACTION * self._span
        if np.any(command < self._minimum - tolerance) or np.any(
            command > self._maximum + tolerance
        ):
            return "applied_command_outside_bounds", None
        if not math.isfinite(sample_period_s) or sample_period_s <= 0.0:
            return "sample_period_not_positive", None
        return None, (
            previous,
            current,
            np.clip(command, self._minimum, self._maximum),
        )

    def _refused_sample(
        self,
        rejection: str,
        started_at: float,
    ) -> RecursiveBootstrapBelief:
        """Record why a transition was unusable and keep the belief unchanged."""

        self._rejected_sample_count += 1
        self._last_sample_report = RecursiveBootstrapSampleReport(
            interval_count=self._interval_count,
            accepted=False,
            reason=rejection,
            update_wall_time_s=time.perf_counter() - started_at,
        )
        return self._belief

    def _sample_features(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        command: np.ndarray,
        sample_period_s: float,
    ) -> _SampleFeatures:
        """Reduce one measured transition to regression targets and features.

        The collective target is the body-frame specific force implied by the
        measured world acceleration, and the angular target is the measured
        body-rate derivative.  Both regressions share the same normalized
        command block and differ only in the nuisance terms appended to it.
        """

        rotation = quaternion_to_rotation(previous[6:10])
        world_acceleration = (current[3:6] - previous[3:6]) / sample_period_s
        body_specific_force = rotation.T @ (
            world_acceleration + np.asarray((0.0, 0.0, GRAVITY_M_S2))
        )
        body_velocity = rotation.T @ previous[3:6]
        angular_velocity = previous[10:13]
        angular_acceleration = (current[10:13] - previous[10:13]) / sample_period_s
        rate_products = np.asarray(
            (
                angular_velocity[0] * angular_velocity[1],
                angular_velocity[0] * angular_velocity[2],
                angular_velocity[1] * angular_velocity[2],
            )
        )
        normalized_command = (command - self._midpoint) / self._span
        return _SampleFeatures(
            command=command,
            body_specific_force=body_specific_force,
            body_velocity=body_velocity,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            rate_products=rate_products,
            force_features=np.concatenate(
                (normalized_command, body_velocity, np.ones(1))
            ),
            angular_features=np.concatenate(
                (normalized_command, angular_velocity, rate_products, np.ones(1))
            ),
        )

    def _accumulate_sample(
        self, features: _SampleFeatures, weight: float = 1.0
    ) -> None:
        """Fold one sample into the forgetting-weighted normal equations.

        ``weight`` is the number of transitions the sample stands for: one for
        a plain transition, the window length for an aggregated one.
        """

        forgetting = self.config.forgetting_factor
        self._force_gram *= forgetting
        self._force_rhs *= forgetting
        self._force_target_sum_squares *= forgetting
        self._angular_gram *= forgetting
        self._angular_rhs *= forgetting
        self._angular_target_sum_squares *= forgetting
        self._force_gram += weight * np.outer(
            features.force_features, features.force_features
        )
        self._force_rhs += weight * np.outer(
            features.force_features, features.body_specific_force[2:3]
        )
        self._force_target_sum_squares += weight * float(
            features.body_specific_force[2] ** 2
        )
        self._angular_gram += weight * np.outer(
            features.angular_features, features.angular_features
        )
        self._angular_rhs += weight * np.outer(
            features.angular_features, features.angular_acceleration
        )
        self._angular_target_sum_squares += weight * np.square(
            features.angular_acceleration
        )
        self._weight = forgetting * self._weight + weight
        self._interval_count += round(weight)

    def _record_prequential_error(self, features: _SampleFeatures) -> None:
        """Fold the belief's error on this transition, before absorbing it."""

        belief = self._belief
        force_error = float(features.body_specific_force[2]) - (
            belief.predict_collective_specific_force(
                features.command, features.body_velocity
            )
        )
        angular_error = features.angular_acceleration - np.asarray(
            belief.predict_angular_acceleration(
                features.command, features.angular_velocity
            )
        )
        keep = 1.0 - 1.0 / float(self.config.minimum_certification_interval_count)
        self._prequential_force_sum = (
            keep * self._prequential_force_sum + force_error**2
        )
        self._prequential_angular_sum = (
            keep * self._prequential_angular_sum + np.square(angular_error)
        )
        self._prequential_weight = keep * self._prequential_weight + 1.0

    def _prequential_standard_deviations(self) -> tuple[float, np.ndarray]:
        weight = max(self._prequential_weight, 1e-12)
        return (
            float(np.sqrt(self._prequential_force_sum / weight)),
            np.sqrt(self._prequential_angular_sum / weight),
        )

    def _aggregated_sample(self, window: Sequence[_SampleFeatures]) -> _SampleFeatures:
        """The mean of a window of transitions, as one sample."""

        def mean(name: str) -> np.ndarray:
            return np.mean([getattr(entry, name) for entry in window], axis=0)

        return _SampleFeatures(
            command=mean("command"),
            body_specific_force=mean("body_specific_force"),
            body_velocity=mean("body_velocity"),
            angular_velocity=mean("angular_velocity"),
            angular_acceleration=mean("angular_acceleration"),
            rate_products=mean("rate_products"),
            force_features=mean("force_features"),
            angular_features=mean("angular_features"),
        )

    def _admit_staged_regressors(self) -> None:
        """Admit each nuisance block the moment its staging condition holds.

        Staging only ever moves forwards.  The Gram it is read against is the
        full accumulated one, so nothing about the transition discards
        evidence: the same data is simply solved over more columns from here
        on, and the interval it happened at is recorded on the belief.
        """

        if self._force_nuisance_staged and self._angular_nuisance_staged:
            return
        if not self._force_nuisance_staged and self._nuisance_admitted(
            self._FORCE_NUISANCE_SIZE
        ):
            self._force_nuisance_staged = True
            self._force_staged_interval = self._interval_count
        if not self._angular_nuisance_staged and self._nuisance_admitted(
            self._ANGULAR_NUISANCE_SIZE
        ):
            self._angular_nuisance_staged = True
            self._angular_staged_interval = self._interval_count

    def _fit_supported_effects(self) -> _EffectFit:
        """Solve both regressions and rescale them back to raw command units.

        The regressions run on normalized commands so their rank tests have a
        single scale; the effects, intercepts, and covariances returned here are
        already mapped back to the raw command box the vehicle is flown in.
        """

        force = self._staged_fit(
            self._force_gram,
            self._force_rhs,
            nuisance_size=self._FORCE_NUISANCE_SIZE,
            staged=self._force_nuisance_staged,
            effective_count=self._weight,
            relative_tolerance=self.config.command_rank_relative_tolerance,
            minimum_rms=self.config.minimum_normalized_command_rms,
            nuisance_relative_tolerance=(self.config.nuisance_rank_relative_tolerance),
        )
        angular = self._staged_fit(
            self._angular_gram,
            self._angular_rhs,
            nuisance_size=self._ANGULAR_NUISANCE_SIZE,
            staged=self._angular_nuisance_staged,
            effective_count=self._weight,
            relative_tolerance=self.config.command_rank_relative_tolerance,
            minimum_rms=self.config.minimum_normalized_command_rms,
            nuisance_relative_tolerance=(self.config.nuisance_rank_relative_tolerance),
        )
        (
            normalized_force_effect,
            force_nuisance,
            force_support,
            _,
            force_rank,
            force_residual_inverse,
            _,
            force_nuisance_rank,
        ) = force
        (
            normalized_angular_effect,
            angular_nuisance,
            angular_support,
            angular_singular_values,
            angular_command_rank,
            angular_residual_inverse,
            angular_residual_information,
            angular_nuisance_rank,
        ) = angular
        # The normalized command channel is thrust fraction, so a nonnegative
        # coefficient is a statement about what the channel *means* rather than
        # about how strong this vehicle is.  The span is positive, so clipping
        # in normalized units is the same qualitative constraint as clipping in
        # raw units, and it carries no magnitude: a negative coefficient moves
        # to exactly zero and a nonnegative one does not move at all.
        sign_projection_count = 0
        sign_projection_magnitude = 0.0
        if self.config.enforce_collective_sign:
            projected = np.maximum(normalized_force_effect, 0.0)
            removed = normalized_force_effect - projected
            sign_projection_count = int(np.count_nonzero(removed))
            sign_projection_magnitude = float(
                np.linalg.norm(removed[:, 0] / self._span)
            )
            normalized_force_effect = projected
        force_effect = normalized_force_effect[:, 0] / self._span
        angular_effect = (normalized_angular_effect / self._span[:, None]).T
        force_intercept = float(force_nuisance[-1, 0] - force_effect @ self._midpoint)
        angular_intercept = angular_nuisance[-1] - angular_effect @ self._midpoint
        force_residual_std = float(
            self._residual_standard_deviation(
                self._force_gram,
                self._force_rhs,
                self._force_target_sum_squares,
                normalized_force_effect,
                force_nuisance,
                effective_count=self._weight,
                command_rank=force_rank,
                nuisance_rank=force_nuisance_rank,
                floor=self.config.collective_residual_std_floor_m_s2,
            )[0]
        )
        angular_residual_std = self._residual_standard_deviation(
            self._angular_gram,
            self._angular_rhs,
            self._angular_target_sum_squares,
            normalized_angular_effect,
            angular_nuisance,
            effective_count=self._weight,
            command_rank=angular_command_rank,
            nuisance_rank=angular_nuisance_rank,
            floor=self.config.angular_residual_std_floor_rad_s2,
        )
        if self.config.prequential_residual and self._prequential_weight > 0.0:
            force_prequential, angular_prequential = (
                self._prequential_standard_deviations()
            )
            force_residual_std = max(force_residual_std, force_prequential)
            angular_residual_std = np.maximum(angular_residual_std, angular_prequential)
        raw_scale = np.diag(1.0 / self._span)
        collective_effect_covariance = (
            raw_scale @ (force_residual_std**2 * force_residual_inverse) @ raw_scale
        )
        angular_effect_covariance = np.stack(
            tuple(
                raw_scale
                @ (angular_residual_std[axis] ** 2 * angular_residual_inverse)
                @ raw_scale
                for axis in range(3)
            )
        )
        return _EffectFit(
            force_effect=force_effect,
            force_nuisance=force_nuisance,
            force_support=force_support,
            force_rank=force_rank,
            force_nuisance_rank=force_nuisance_rank,
            force_residual_inverse=force_residual_inverse,
            force_intercept=force_intercept,
            force_residual_std=force_residual_std,
            collective_effect_covariance=collective_effect_covariance,
            angular_effect=angular_effect,
            angular_nuisance=angular_nuisance,
            angular_support=angular_support,
            angular_singular_values=angular_singular_values,
            angular_command_rank=angular_command_rank,
            angular_nuisance_rank=angular_nuisance_rank,
            angular_residual_information=angular_residual_information,
            angular_intercept=angular_intercept,
            angular_residual_std=angular_residual_std,
            angular_effect_covariance=angular_effect_covariance,
            force_nuisance_staged=self._force_nuisance_staged,
            angular_nuisance_staged=self._angular_nuisance_staged,
            collective_sign_projection_count=sign_projection_count,
            collective_sign_projection_magnitude=sign_projection_magnitude,
        )

    def _belief_authority(self, fit: _EffectFit) -> _BeliefAuthority:
        """Decide how much of each output direction this fit may command.

        Authority is the product of what the evidence spans, how strong the
        weakest supported information direction is, and how far each fitted
        effect stands above its own residual noise.  Collective authority
        additionally requires a hover command that lies inside the command box,
        because a collective effect that cannot hold the vehicle up is not
        something to hand control to.
        """

        collective_direction = np.ones(4, dtype=np.float64)
        collective_support = float(
            np.linalg.norm(fit.force_support @ collective_direction) ** 2 / 4.0
        )
        supported_effect = fit.angular_effect @ fit.angular_support
        left, effect_singular_values, _ = np.linalg.svd(
            supported_effect,
            full_matrices=False,
        )
        effect_threshold = max(
            (
                float(effect_singular_values[0])
                * self.config.output_rank_relative_tolerance
            ),
            1e-8,
        )
        effect_supported = effect_singular_values >= effect_threshold
        angular_effect_rank = int(np.sum(effect_supported))
        angular_output_support = (left * effect_supported.astype(np.float64)) @ left.T
        minimum_supported_information = (
            0.0
            if fit.angular_command_rank == 0
            else float(fit.angular_singular_values[fit.angular_command_rank - 1])
        )
        information_strength = self._authority_fraction(
            minimum_supported_information,
            self.config.minimum_information_singular_value,
            self.config.full_authority_information_singular_value,
        )
        command_coverage = fit.angular_command_rank / 4.0
        information_authority = command_coverage * information_strength
        angular_effect_signal_to_noise = np.asarray(
            [
                np.linalg.norm(supported_effect[axis])
                / max(
                    math.sqrt(
                        max(float(np.trace(fit.angular_effect_covariance[axis])), 0.0)
                    ),
                    1e-9,
                )
                for axis in range(3)
            ]
        )
        angular_signal_authority = np.asarray(
            [
                self._authority_fraction(
                    value,
                    self.config.minimum_effect_signal_to_noise,
                    self.config.full_authority_effect_signal_to_noise,
                )
                for value in angular_effect_signal_to_noise
            ]
        )
        angular_axis_authority = np.clip(
            information_authority
            * angular_signal_authority
            * np.diag(angular_output_support),
            0.0,
            1.0,
        )
        hover_command: np.ndarray | None = None
        collective_sum = float(np.sum(fit.force_effect))
        if collective_sum > 1e-8:
            hover_scalar = (GRAVITY_M_S2 - fit.force_intercept) / collective_sum
            candidate = np.full(4, hover_scalar)
            if np.all(candidate >= self._minimum) and np.all(
                candidate <= self._maximum
            ):
                hover_command = candidate
        collective_authority = 0.0
        collective_effect_standard_error = math.sqrt(
            max(
                float(
                    collective_direction
                    @ fit.collective_effect_covariance
                    @ collective_direction
                ),
                0.0,
            )
        )
        collective_effect_signal_to_noise = abs(collective_sum) / max(
            collective_effect_standard_error,
            1e-9,
        )
        collective_signal_authority = self._authority_fraction(
            collective_effect_signal_to_noise,
            self.config.minimum_effect_signal_to_noise,
            self.config.full_authority_effect_signal_to_noise,
        )
        unit_collective = collective_direction / np.linalg.norm(collective_direction)
        collective_directional_information = 1.0 / math.sqrt(
            max(
                float(unit_collective @ fit.force_residual_inverse @ unit_collective),
                1e-12,
            )
        )
        collective_information_authority = self._authority_fraction(
            collective_directional_information,
            self.config.minimum_information_singular_value,
            self.config.full_authority_information_singular_value,
        )
        if fit.force_rank >= 1 and hover_command is not None:
            collective_authority = float(
                np.clip(
                    collective_support
                    * min(
                        collective_signal_authority,
                        collective_information_authority,
                    ),
                    0.0,
                    1.0,
                )
            )
        exploration_completion = float(
            min(
                fit.angular_command_rank / 4.0,
                angular_effect_rank / 3.0,
                information_strength,
                collective_signal_authority,
                float(np.min(angular_signal_authority)),
            )
        )
        return _BeliefAuthority(
            collective_support=collective_support,
            angular_output_support=angular_output_support,
            angular_effect_rank=angular_effect_rank,
            minimum_supported_information=minimum_supported_information,
            information_authority=information_authority,
            angular_effect_signal_to_noise=angular_effect_signal_to_noise,
            angular_axis_authority=angular_axis_authority,
            hover_command=hover_command,
            collective_authority=collective_authority,
            collective_effect_signal_to_noise=collective_effect_signal_to_noise,
            exploration_completion=exploration_completion,
        )

    def _assimilated_belief(
        self,
        fit: _EffectFit,
        authority: _BeliefAuthority,
        started_at: float,
    ) -> RecursiveBootstrapBelief:
        """Assemble the working belief this sample leaves the identifier in."""

        return RecursiveBootstrapBelief(
            interval_count=self._interval_count,
            effective_interval_count=self._weight,
            collective_acceleration_per_command=fit.force_effect,
            collective_velocity_coefficient=fit.force_nuisance[:3, 0],
            collective_intercept_m_s2=fit.force_intercept,
            angular_acceleration_per_command=fit.angular_effect,
            angular_rate_coefficient=fit.angular_nuisance[:3].T,
            angular_rate_product_coefficient=fit.angular_nuisance[3:6].T,
            angular_intercept_rad_s2=fit.angular_intercept,
            normalized_command_support_projector=fit.angular_support,
            normalized_command_singular_values=fit.angular_singular_values,
            normalized_command_information=fit.angular_residual_information,
            supported_collective_effect_covariance=fit.collective_effect_covariance,
            supported_angular_effect_covariance=fit.angular_effect_covariance,
            collective_information=self._force_gram,
            angular_information=self._angular_gram,
            command_evidence_rank=fit.angular_command_rank,
            angular_effect_rank=authority.angular_effect_rank,
            collective_nuisance_rank=fit.force_nuisance_rank,
            angular_nuisance_rank=fit.angular_nuisance_rank,
            angular_output_support_projector=authority.angular_output_support,
            collective_support_fraction=authority.collective_support,
            minimum_supported_information_singular_value=(
                authority.minimum_supported_information
            ),
            information_authority=authority.information_authority,
            collective_effect_signal_to_noise=(
                authority.collective_effect_signal_to_noise
            ),
            angular_effect_signal_to_noise=authority.angular_effect_signal_to_noise,
            collective_residual_std_m_s2=fit.force_residual_std,
            angular_residual_std_rad_s2=fit.angular_residual_std,
            exploration_completion=authority.exploration_completion,
            collective_authority=authority.collective_authority,
            angular_axis_authority=authority.angular_axis_authority,
            hover_command=authority.hover_command,
            update_wall_time_s=time.perf_counter() - started_at,
            collective_nuisance_staged=fit.force_nuisance_staged,
            angular_nuisance_staged=fit.angular_nuisance_staged,
            collective_staging_interval_count=self._force_staged_interval,
            angular_staging_interval_count=self._angular_staged_interval,
            collective_sign_projection_count=fit.collective_sign_projection_count,
            collective_sign_projection_magnitude=(
                fit.collective_sign_projection_magnitude
            ),
        )

    def update(
        self,
        previous_state: Sequence[float],
        current_state: Sequence[float],
        average_applied_motor_command: Sequence[float],
        sample_period_s: float,
    ) -> RecursiveBootstrapBelief:
        """Assimilate one measured transition and return the current belief.

        A transition that is not usable evidence is refused rather than raised
        on, because a single non-finite estimator sample must not end a control
        loop.  The belief is then left exactly as it was and the refusal is
        recorded in :attr:`last_sample_report`.
        """

        started_at = time.perf_counter()
        rejection, sample = self._validated_sample(
            previous_state,
            current_state,
            average_applied_motor_command,
            sample_period_s,
        )
        if sample is None:
            assert rejection is not None
            return self._refused_sample(rejection, started_at)
        features = self._sample_features(*sample, sample_period_s)
        if self.config.prequential_residual:
            self._record_prequential_error(features)
        self._score_pending_proposal(
            command=features.command,
            body_velocity=features.body_velocity,
            angular_velocity=features.angular_velocity,
            rate_products=features.rate_products,
            collective_target=float(features.body_specific_force[2]),
            angular_target=features.angular_acceleration,
        )
        window = self.config.transition_aggregation_steps
        if window > 1:
            self._pending_transitions.append(features)
            if len(self._pending_transitions) < window:
                # The window is not full: the belief stands as it was.
                self._last_sample_report = RecursiveBootstrapSampleReport(
                    interval_count=self._interval_count,
                    accepted=True,
                    reason="sample_buffered",
                    update_wall_time_s=time.perf_counter() - started_at,
                )
                return self._belief
            features = self._aggregated_sample(self._pending_transitions)
            self._pending_transitions = []
            self._accumulate_sample(features, weight=float(window))
        else:
            self._accumulate_sample(features)
        self._admit_staged_regressors()
        fit = self._fit_supported_effects()
        self._belief = self._assimilated_belief(
            fit,
            self._belief_authority(fit),
            started_at,
        )
        self._working_support_reached = (
            self._working_support_reached or self.working_belief_supported
        )
        self._maybe_start_proposal()
        self._last_sample_report = RecursiveBootstrapSampleReport(
            interval_count=self._interval_count,
            accepted=True,
            reason="sample_assimilated",
            update_wall_time_s=self._belief.update_wall_time_s,
        )
        return self._belief


@dataclass(frozen=True)
class ProgressiveBootstrapControllerConfig:
    """Cascade gains, bounds, and the excitation scan this controller uses."""

    velocity_gain: tuple[float, float, float] = (1.5, 1.5, 4.0)
    maximum_world_acceleration_m_s2: tuple[float, float, float] = (2.5, 2.5, 4.0)
    maximum_tilt_rad: float = 0.50
    attitude_gain: tuple[float, float] = (14.0, 14.0)
    angular_rate_gain: tuple[float, float, float] = (6.0, 6.0, 3.0)
    initial_excitation_fraction: float = 0.12
    continuing_excitation_fraction: float = 0.0
    maximum_committed_excitation_fraction: float = 0.0005
    maximum_feedback_delta: float = 0.35
    maximum_motor_step: float = 0.10
    objective_horizon_s: float = 0.10
    minimum_altitude_m: float = 1.0
    altitude_risk_weight: float = 1.0
    #: Inert at its default.  The supported-covariance term it weighs is around
    #: 1e-9 across reachable states, against stabilization and altitude-risk
    #: terms of order 1e-1, so it never changes which candidate the scan picks.
    #: It is kept, and reported, so that a run's recorded configuration stays
    #: comparable; raising it is the only way to make the term matter.
    uncertainty_cost_weight: float = 1e-10
    #: Fractions of the information amplitude the bounded scan may choose from.
    #: The reward is linear in the fraction while the distance from the cascade
    #: is quadratic at the same scale, so the scan takes the largest fraction
    #: unless altitude risk or the bound and slew clipping prefer less.
    objective_information_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        for name, size in (
            ("velocity_gain", 3),
            ("maximum_world_acceleration_m_s2", 3),
            ("attitude_gain", 2),
            ("angular_rate_gain", 3),
        ):
            values = finite_vector(name, getattr(self, name), size)
            if np.any(values <= 0.0):
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.maximum_tilt_rad < 1.0:
            raise ValueError("maximum_tilt_rad must lie inside (0, 1)")
        if (
            not 0.0
            <= self.continuing_excitation_fraction
            <= (self.initial_excitation_fraction)
        ):
            raise ValueError("continuing excitation must not exceed initial excitation")
        if (
            not 0.0
            <= self.maximum_committed_excitation_fraction
            <= (self.initial_excitation_fraction)
        ):
            raise ValueError("committed excitation must not exceed initial excitation")
        if not 0.0 < self.maximum_feedback_delta <= 0.5:
            raise ValueError("maximum_feedback_delta must lie inside (0, 0.5]")
        if not 0.0 < self.maximum_motor_step <= 1.0:
            raise ValueError("maximum_motor_step must lie inside (0, 1]")
        for name in (
            "objective_horizon_s",
            "minimum_altitude_m",
            "altitude_risk_weight",
            "uncertainty_cost_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        scales = np.asarray(self.objective_information_scales, dtype=np.float64)
        if (
            scales.ndim != 1
            or len(scales) < 2
            or not np.all(np.isfinite(scales))
            or scales[0] != 0.0
            or scales[-1] != 1.0
            or np.any(np.diff(scales) <= 0.0)
        ):
            raise ValueError(
                "objective_information_scales must increase from zero to one"
            )


class ProgressiveBootstrapController:
    """Stabilize with a fixed cascade and scan how much to excite on top of it.

    The stabilizing action is a hand-gained cascade from velocity error through
    a thrust direction and tilt error to a pseudo-inverse motor allocation.  The
    belief scales its authority and supplies the allocation; it does not select
    the action.  A bounded scan over
    :attr:`ProgressiveBootstrapControllerConfig.objective_information_scales`
    then chooses how much information excitation rides on top of that cascade.
    """

    def __init__(
        self,
        identifier_config: RecursiveBootstrapConfig | None = None,
        config: ProgressiveBootstrapControllerConfig | None = None,
    ) -> None:
        self.identifier_config = (
            RecursiveBootstrapConfig()
            if identifier_config is None
            else identifier_config
        )
        self.config = (
            ProgressiveBootstrapControllerConfig() if config is None else config
        )
        self._minimum = np.asarray(self.identifier_config.command_minimum)
        self._maximum = np.asarray(self.identifier_config.command_maximum)
        self._span = self._maximum - self._minimum
        self._midpoint = 0.5 * (self._minimum + self._maximum)
        generator = np.random.default_rng(11)
        patterns = generator.standard_normal((4096, 4))
        self._patterns = patterns / np.maximum(
            np.max(np.abs(patterns), axis=1, keepdims=True),
            1e-9,
        )

    def _held_command(self, previous_command: Any) -> np.ndarray:
        """Clip the previous command into bounds, or fall back to the hold."""

        try:
            previous = np.asarray(previous_command, dtype=np.float64)
        except (TypeError, ValueError):
            return self._midpoint.copy()
        if previous.shape != (4,) or not np.all(np.isfinite(previous)):
            return self._midpoint.copy()
        return np.clip(previous, self._minimum, self._maximum)

    def _unusable_command(
        self,
        command: np.ndarray,
        belief: RecursiveBootstrapBelief,
        online: RecursiveBootstrapBelief,
        reason: str,
    ) -> ProgressiveBootstrapCommand:
        """Return a bounded held command that the supervisor can reject."""

        return ProgressiveBootstrapCommand(
            command=command,
            objective_value=0.0,
            stabilization_cost=0.0,
            information_reward=0.0,
            uncertainty_cost=0.0,
            altitude_risk_cost=0.0,
            estimated_information_gain=0.0,
            information_action_fraction=0.0,
            information_completion=online.exploration_completion,
            predicted_world_velocity_m_s=np.zeros(3),
            predicted_angular_velocity_rad_s=np.zeros(3),
            desired_world_acceleration_m_s2=np.zeros(3),
            desired_angular_acceleration_rad_s2=np.zeros(3),
            collective_authority=belief.collective_authority,
            angular_axis_authority=belief.angular_axis_authority,
            command_usable=False,
            reason=reason,
        )

    def _action_inputs(
        self,
        state: Sequence[float],
        previous_command: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse the state and previous command, or name why they are unusable."""

        try:
            state_array = finite_vector("state", state, 13)
        except (ValueError, TypeError) as error:
            raise _UnusableActionInput("state_not_finite") from error
        try:
            previous = finite_vector("previous_command", previous_command, 4)
        except (ValueError, TypeError) as error:
            raise _UnusableActionInput("previous_command_not_finite") from error
        if float(np.linalg.norm(state_array[6:10])) < 1e-9:
            raise _UnusableActionInput("state_quaternion_degenerate")
        return state_array, previous

    def _stabilizing_feedback(
        self,
        state_array: np.ndarray,
        belief: RecursiveBootstrapBelief,
    ) -> _StabilizingFeedback:
        """Run the fixed cascade and allocate it through the identified effect.

        Velocity arrest is scaled by the weakest authority the manoeuvre needs,
        the collective reference is blended from the bounded midpoint toward the
        identified hover command in proportion to collective authority, and the
        angular part is allocated only inside the supported command and output
        subspaces, so an unexcited direction is never commanded.
        """

        rotation = quaternion_to_rotation(state_array[6:10])
        velocity = state_array[3:6]
        angular_velocity = state_array[10:13]
        velocity_authority = float(
            min(
                belief.collective_authority,
                belief.angular_axis_authority[0],
                belief.angular_axis_authority[1],
            )
        )
        cascade = thrust_cascade(
            world_velocity_m_s=velocity,
            rotation=rotation,
            angular_velocity_rad_s=angular_velocity,
            velocity_gain=np.asarray(self.config.velocity_gain),
            maximum_world_acceleration_m_s2=np.asarray(
                self.config.maximum_world_acceleration_m_s2
            ),
            maximum_tilt_rad=self.config.maximum_tilt_rad,
            attitude_gain=np.asarray(self.config.attitude_gain),
            angular_rate_gain=np.asarray(self.config.angular_rate_gain),
            velocity_authority=velocity_authority,
        )

        collective_reference = self._midpoint.copy()
        collective_sum = float(np.sum(belief.collective_acceleration_per_command))
        if belief.hover_command is not None and collective_sum > 1e-8:
            estimated_scalar = (
                cascade.desired_force_magnitude_m_s2 - belief.collective_intercept_m_s2
            ) / collective_sum
            estimated_reference = np.full(4, estimated_scalar)
            collective_reference = (
                1.0 - belief.collective_authority
            ) * self._midpoint + belief.collective_authority * estimated_reference
        collective_reference = np.clip(
            collective_reference,
            self._minimum,
            self._maximum,
        )

        angular_effect = (
            belief.angular_acceleration_per_command
            @ belief.normalized_command_support_projector
        )
        supported_desired = belief.angular_output_support_projector @ (
            belief.angular_axis_authority * cascade.desired_angular_acceleration_rad_s2
        )
        if belief.angular_effect_rank:
            delta = np.linalg.pinv(angular_effect, rcond=1e-5) @ supported_desired
        else:
            delta = np.zeros(4)
        delta_limit = self.config.maximum_feedback_delta * float(
            np.max(belief.angular_axis_authority)
        )
        delta = np.clip(delta, -delta_limit, delta_limit)
        return _StabilizingFeedback(
            rotation=rotation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            cascade=cascade,
            command=np.clip(
                collective_reference + delta,
                self._minimum,
                self._maximum,
            ),
        )

    def _information_action(
        self,
        online: RecursiveBootstrapBelief,
        *,
        committed: bool,
    ) -> _InformationAction:
        """Choose the excitation direction and amplitude to ride on the cascade.

        The amplitude decays from the initial fraction toward the continuing one
        as exploration completes, and is capped further while a candidate belief
        is committed to predictive validation.  The direction is a fixed pseudo
        random pattern half blended with the same pattern whitened by the
        current information geometry, which pushes the probe toward the weakest
        identified directions without ever leaving the bounded pattern family.
        """

        amplitude = self.config.continuing_excitation_fraction + (
            self.config.initial_excitation_fraction
            - self.config.continuing_excitation_fraction
        ) * (1.0 - online.exploration_completion)
        if committed:
            amplitude = min(
                amplitude,
                self.config.maximum_committed_excitation_fraction,
            )
        pattern = self._patterns[online.interval_count % len(self._patterns)]
        information = 0.5 * (
            online.normalized_command_information
            + online.normalized_command_information.T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        floor = self.identifier_config.minimum_information_singular_value**2
        inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))
        inverse_sqrt /= max(float(np.min(inverse_sqrt)), 1e-9)
        inverse_sqrt = np.clip(inverse_sqrt, 1.0, 3.0)
        information_weighted_direction = eigenvectors @ (
            inverse_sqrt * (eigenvectors.T @ pattern)
        )
        information_weighted_direction /= max(
            float(np.max(np.abs(information_weighted_direction))),
            1e-9,
        )
        information_direction = 0.5 * pattern + 0.5 * information_weighted_direction
        information_direction /= max(
            float(np.max(np.abs(information_direction))),
            1e-9,
        )
        return _InformationAction(
            amplitude=amplitude,
            target=amplitude * self._span * information_direction,
            information=information,
            floor=floor,
        )

    def _candidate_commands(
        self,
        feedback: np.ndarray,
        information_target: np.ndarray,
        previous: np.ndarray,
        belief: RecursiveBootstrapBelief,
    ) -> np.ndarray:
        """Return the bounded scan of excitation fractions around the feedback.

        The per-interval motor step limit applies only once the belief has seen
        an interval, so the very first command is free to move from whatever the
        vehicle happened to be holding.
        """

        scales = np.asarray(self.config.objective_information_scales)
        candidates = feedback + scales[:, None] * information_target
        candidates = np.clip(candidates, self._minimum, self._maximum)
        if belief.interval_count:
            candidates = np.clip(
                candidates,
                previous - self.config.maximum_motor_step,
                previous + self.config.maximum_motor_step,
            )
            candidates = np.clip(candidates, self._minimum, self._maximum)
        return candidates

    def _candidate_scores(
        self,
        candidates: np.ndarray,
        feedback: _StabilizingFeedback,
        action: _InformationAction,
        belief: RecursiveBootstrapBelief,
        state_array: np.ndarray,
    ) -> _CandidateScores:
        """Score every candidate on stabilization, information, and risk.

        The objective trades departure from the stabilizing command against the
        information the departure buys, then charges the candidate for the
        predicted-effect variance it rides on and for any predicted breach of
        the minimum altitude.
        """

        normalized_delta = (candidates - feedback.command) / self._span
        normalized_information_target = action.target / self._span
        stabilization = 0.5 * np.sum(np.square(normalized_delta), axis=1)
        information_reward = normalized_delta @ normalized_information_target
        force_variance = np.einsum(
            "ni,ij,nj->n",
            candidates,
            belief.supported_collective_effect_covariance,
            candidates,
        )
        angular_variance = sum(
            np.einsum(
                "ni,ij,nj->n",
                candidates,
                belief.supported_angular_effect_covariance[axis],
                candidates,
            )
            for axis in range(3)
        )
        uncertainty = self.config.uncertainty_cost_weight * (
            force_variance + angular_variance
        )
        body_velocity = feedback.rotation.T @ feedback.velocity
        predicted_force = (
            candidates @ belief.collective_acceleration_per_command
            + belief.collective_velocity_coefficient @ body_velocity
            + belief.collective_intercept_m_s2
        )
        predicted_world_acceleration = predicted_force[:, None] * feedback.rotation[
            :, 2
        ] - np.asarray((0.0, 0.0, GRAVITY_M_S2))
        horizon = self.config.objective_horizon_s
        predicted_velocity = feedback.velocity + horizon * predicted_world_acceleration
        predicted_altitude = (
            state_array[2]
            + horizon * feedback.velocity[2]
            + 0.5 * horizon**2 * predicted_world_acceleration[:, 2]
        )
        altitude_risk = self.config.altitude_risk_weight * np.square(
            np.maximum(self.config.minimum_altitude_m - predicted_altitude, 0.0)
        )
        return _CandidateScores(
            stabilization=stabilization,
            information_reward=information_reward,
            uncertainty=uncertainty,
            altitude_risk=altitude_risk,
            objective=(
                stabilization - information_reward + uncertainty + altitude_risk
            ),
            predicted_world_velocity=predicted_velocity,
        )

    def _selected_decision(
        self,
        selected: int,
        candidates: np.ndarray,
        scores: _CandidateScores,
        feedback: _StabilizingFeedback,
        action: _InformationAction,
        belief: RecursiveBootstrapBelief,
        online: RecursiveBootstrapBelief,
        previous: np.ndarray,
    ) -> ProgressiveBootstrapCommand:
        """Assemble the decision record for the candidate the scan selected."""

        bounded = candidates[selected]
        angular_velocity = feedback.angular_velocity
        rate_products = np.asarray(
            (
                angular_velocity[0] * angular_velocity[1],
                angular_velocity[0] * angular_velocity[2],
                angular_velocity[1] * angular_velocity[2],
            )
        )
        predicted_angular_acceleration = (
            bounded @ belief.angular_acceleration_per_command.T
            + belief.angular_rate_coefficient @ angular_velocity
            + belief.angular_rate_product_coefficient @ rate_products
            + belief.angular_intercept_rad_s2
        )
        predicted_angular_velocity = (
            angular_velocity
            + self.config.objective_horizon_s * predicted_angular_acceleration
        )
        command_innovation = (bounded - previous) / self._span
        regularized_information = action.information + action.floor * np.eye(4)
        information_inverse = np.linalg.pinv(regularized_information, rcond=1e-10)
        estimated_information_gain = math.log1p(
            max(
                float(command_innovation @ information_inverse @ command_innovation),
                0.0,
            )
        )
        scales = np.asarray(self.config.objective_information_scales)
        return ProgressiveBootstrapCommand(
            command=bounded,
            objective_value=float(scores.objective[selected]),
            stabilization_cost=float(scores.stabilization[selected]),
            information_reward=float(scores.information_reward[selected]),
            uncertainty_cost=float(scores.uncertainty[selected]),
            altitude_risk_cost=float(scores.altitude_risk[selected]),
            estimated_information_gain=estimated_information_gain,
            information_action_fraction=float(scales[selected] * action.amplitude),
            information_completion=online.exploration_completion,
            predicted_world_velocity_m_s=scores.predicted_world_velocity[selected],
            predicted_angular_velocity_rad_s=predicted_angular_velocity,
            desired_world_acceleration_m_s2=(
                feedback.cascade.desired_world_acceleration_m_s2
            ),
            desired_angular_acceleration_rad_s2=(
                feedback.cascade.desired_angular_acceleration_rad_s2
            ),
            collective_authority=belief.collective_authority,
            angular_axis_authority=belief.angular_axis_authority,
        )

    def command(
        self,
        state: Sequence[float],
        belief: RecursiveBootstrapBelief,
        *,
        previous_command: Sequence[float],
        online_belief: RecursiveBootstrapBelief | None = None,
    ) -> ProgressiveBootstrapCommand:
        """Return one bounded action from the cascade and the excitation scan.

        ``belief`` is the transactional predictive mean.  ``online_belief`` may
        provide fresher information geometry while a candidate is undergoing
        predictive validation.  Both enter this objective; neither produces a
        separate command.

        A state this controller cannot act on never raises.  The decision comes
        back with ``command_usable`` false, a reason, and the previous command
        clipped into bounds, falling back to the bounded midpoint hold when even
        that is unusable, so the supervisor rather than an exception decides what
        the vehicle does next.
        """

        online = belief if online_belief is None else online_belief
        held = self._held_command(previous_command)
        try:
            state_array, previous = self._action_inputs(state, previous_command)
        except _UnusableActionInput as unusable:
            return self._unusable_command(held, belief, online, unusable.reason)

        feedback = self._stabilizing_feedback(state_array, belief)
        action = self._information_action(
            online,
            committed=online_belief is not None,
        )
        candidates = self._candidate_commands(
            feedback.command,
            action.target,
            previous,
            belief,
        )
        scores = self._candidate_scores(
            candidates,
            feedback,
            action,
            belief,
            state_array,
        )
        return self._selected_decision(
            int(np.argmin(scores.objective)),
            candidates,
            scores,
            feedback,
            action,
            belief,
            online,
            previous,
        )
