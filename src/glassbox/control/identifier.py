"""Recursive, support-aware identification from motor input and output alone.

The identifier updates one :class:`~glassbox.belief.belief.DynamicsBelief`
over the bootstrap parameterization after every measured actuation interval,
and gives control authority only to output directions its evidence supports.
There is deliberately no evidence-collection/model-running phase boundary.

The belief is the same type a fit produces: an executable model, an
information state, and a provenance record. What is specific to this
estimator, the ranks and supports its own thresholds define and the per-axis
authority they imply, is a :class:`BootstrapEvidence` summary alongside it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.information import (
    ParameterInformation,
    default_rank_relative_tolerance,
    innovation_noise_floor,
)
from glassbox.control._common import finite_vector, immutable_array
from glassbox.core.data import (
    RIGID_BODY_STATE_SCHEMA,
    Channel,
    TrajectorySpec,
    VehicleConfigurationSpec,
)
from glassbox.core.dynamics import (
    GRAVITY_M_S2,
    BootstrapMultirotorParams,
    structured_parameter_names,
)
from glassbox.core.families import BOOTSTRAP_MULTIROTOR_FAMILY
from glassbox.core.geometry import TANGENT_GROUP_INDICES, quaternion_to_rotation
from glassbox.core.model import (
    DirectActuationMap,
    ExecutableModel,
    ModelValidityEnvelope,
    RuntimeModelSpec,
)

#: Transitions per assimilated sample.  Differencing a noisy measurement over
#: one interval multiplies the noise by the loop rate; the mean over a window
#: telescopes most of it away, while weighting the sample by the window length
#: keeps the sample count, the support thresholds, and the residual floor
#: exactly per transition, so with a noise-free measurement the information
#: rate is unchanged.  Two is the width measured on the release ensemble.
TRANSITION_AGGREGATION_STEPS = 2

#: The bootstrap parameterization claims no operating envelope.  It is affine
#: in body velocity and body rate with no saturation anywhere, and the
#: identifier measures no supported region, so the envelope it declares is
#: wide enough never to bind and the per-direction authority is what governs
#: how far a plan may trust it.
BOOTSTRAP_VALIDITY_ENVELOPE = ModelValidityEnvelope(
    body_velocity_center_m_s=(0.0, 0.0, 0.0),
    body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
    angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
    angular_velocity_half_width_rad_s=(50.0, 50.0, 50.0),
)


@dataclass(frozen=True)
class RecursiveBootstrapConfig:
    """Known I/O contract and evidence thresholds for the working belief."""

    command_minimum: float | tuple[float, float, float, float] = 0.0
    command_maximum: float | tuple[float, float, float, float] = 1.0
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
    #: The control period the identified model is executed at.  Every
    #: transition is assimilated at its own measured interval; this is the
    #: period the produced belief's runtime contract declares, so a plan model
    #: over that belief has a stable timing contract from the first sample.
    sample_period_s: float = 0.01

    def __post_init__(self) -> None:
        minimum = finite_vector("command_minimum", self.command_minimum, 4)
        maximum = finite_vector("command_maximum", self.command_maximum, 4)
        if np.any(minimum >= maximum):
            raise ValueError("command_minimum must be below command_maximum")
        for name in (
            "command_rank_relative_tolerance",
            "minimum_normalized_command_rms",
            "nuisance_rank_relative_tolerance",
            "output_rank_relative_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        positive_fields = (
            "minimum_information_singular_value",
            "full_authority_information_singular_value",
            "minimum_effect_signal_to_noise",
            "full_authority_effect_signal_to_noise",
            "collective_residual_std_floor_m_s2",
            "angular_residual_std_floor_rad_s2",
            "sample_period_s",
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
        object.__setattr__(self, "command_minimum", tuple(minimum))
        object.__setattr__(self, "command_maximum", tuple(maximum))


@dataclass(frozen=True)
class BootstrapEvidence:
    """What this estimator's own thresholds say about one accumulated fit.

    The belief carries the model and the information state.  This carries the
    part that is specific to how the bootstrap identifier reaches them: the
    two accumulated Grams in its own feature order, the residual scales it
    estimated, the ranks and support projectors its declared tolerances
    define, and the per-direction authority that follows.  It is a pure
    function of the evidence, so two identifiers fed the same transitions
    produce the same summary.

    The two Grams are the whole evidence both fits are solved from, before any
    support rule, Schur complement, or rescaling to raw command units.  The
    belief's ``information.precision`` is that same evidence mapped into the
    parameters' own coordinates and whitened by the residual scales here; a
    planner whose plan moves the nuisance regressors reads either, and a
    planner that wants the identifier's own rank test reads this.
    """

    interval_count: int
    #: Accumulated, per-transition Gram of each regression in the identifier's
    #: own feature order: ``[normalized command (4), body velocity (3), 1]``
    #: for the collective fit and ``[normalized command (4), body rate (3),
    #: rate products (3), 1]`` for the angular one.
    collective_information: np.ndarray
    angular_information: np.ndarray
    collective_residual_std_m_s2: float
    angular_residual_std_rad_s2: np.ndarray
    normalized_command_support_projector: np.ndarray
    normalized_command_singular_values: np.ndarray
    normalized_command_information: np.ndarray
    angular_output_support_projector: np.ndarray
    supported_collective_effect_covariance: np.ndarray
    supported_angular_effect_covariance: np.ndarray
    command_evidence_rank: int
    angular_effect_rank: int
    collective_nuisance_rank: int
    angular_nuisance_rank: int
    collective_support_fraction: float
    minimum_supported_information_singular_value: float
    information_authority: float
    collective_effect_signal_to_noise: float
    angular_effect_signal_to_noise: np.ndarray
    collective_authority: float
    angular_axis_authority: np.ndarray
    exploration_completion: float
    hover_command: np.ndarray | None

    def __post_init__(self) -> None:
        arrays = {
            "collective_information": (8, 8),
            "angular_information": (11, 11),
            "angular_residual_std_rad_s2": (3,),
            "normalized_command_support_projector": (4, 4),
            "normalized_command_singular_values": (4,),
            "normalized_command_information": (4, 4),
            "angular_output_support_projector": (3, 3),
            "supported_collective_effect_covariance": (4, 4),
            "supported_angular_effect_covariance": (3, 4, 4),
            "angular_effect_signal_to_noise": (3,),
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
            self.collective_residual_std_m_s2,
            self.collective_support_fraction,
            self.minimum_supported_information_singular_value,
            self.information_authority,
            self.collective_effect_signal_to_noise,
            self.collective_authority,
            self.exploration_completion,
        )
        if self.interval_count < 0 or not np.all(np.isfinite(scalars)):
            raise ValueError("bootstrap evidence counts and scalars must be finite")
        if not (
            0 <= self.command_evidence_rank <= 4
            and 0 <= self.angular_effect_rank <= 3
            and 0 <= self.collective_nuisance_rank <= 4
            and 0 <= self.angular_nuisance_rank <= 7
        ):
            raise ValueError("bootstrap evidence ranks lie outside model dimensions")
        if not (
            0.0 <= self.collective_support_fraction <= 1.0 + 1e-9
            and 0.0 <= self.information_authority <= 1.0
            and 0.0 <= self.exploration_completion <= 1.0
            and 0.0 <= self.collective_authority <= 1.0
        ):
            raise ValueError("bootstrap support and authority must lie in [0, 1]")
        if (
            self.collective_effect_signal_to_noise < 0.0
            or np.any(self.angular_effect_signal_to_noise < 0.0)
            or self.collective_residual_std_m_s2 <= 0.0
            or np.any(self.angular_residual_std_rad_s2 <= 0.0)
        ):
            raise ValueError("bootstrap uncertainty statistics must be positive")
        if np.any(self.angular_axis_authority < 0.0) or np.any(
            self.angular_axis_authority > 1.0
        ):
            raise ValueError("angular authority must lie inside [0, 1]")

    @property
    def has_any_control_authority(self) -> bool:
        return bool(
            self.collective_authority > 0.0 or np.any(self.angular_axis_authority > 0.0)
        )

    @property
    def supported(self) -> bool:
        """Whether the evidence spans everything control has to command.

        The conditions are what the evidence spans and nothing else: the
        command evidence spans all four motors, the fitted angular effect
        spans all three body axes, and the collective effect implies a hover
        command inside the command box.  How far to trust evidence that meets
        them is a question for the per-direction authority, not for this rule.
        """

        return bool(
            self.command_evidence_rank == 4
            and self.angular_effect_rank == 3
            and self.hover_command is not None
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": "recursive_rank_supported_multirotor_bootstrap_v2",
            "airframe_parameter_prior_used": False,
            "canonical_motor_mixer_assumed": False,
            "transition_aggregation_steps": TRANSITION_AGGREGATION_STEPS,
            "collective_target": "integrated",
            "effect_covariance_scope": "supported_subspace_only",
        }
        for entry in fields(self):
            value = getattr(self, entry.name)
            if value is None or isinstance(value, (int, float)):
                payload[entry.name] = value
            else:
                payload[entry.name] = np.asarray(value).tolist()
        return payload


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


def _bootstrap_input_spec(minimum: np.ndarray, maximum: np.ndarray) -> TrajectorySpec:
    """Declare the command box the identifier was told the vehicle accepts.

    Actuation is the identity on these four normalized motor commands, which
    is what "measured applied motor input" means: the identifier regresses on
    the command the vehicle actually applied, so the model's input is the
    command itself.
    """

    family = BOOTSTRAP_MULTIROTOR_FAMILY
    channels = tuple(
        Channel(
            name=name,
            role=role,
            semantic="normalized_command",
            unit="1",
            kind="control",
            minimum=float(minimum[index]),
            maximum=float(maximum[index]),
        )
        for index, (name, role) in enumerate(
            zip(family.control_names, family.control_roles, strict=True)
        )
    )
    return TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="measured_applied_motor_command",
        channels=channels,
        vehicle=VehicleConfigurationSpec(
            family=family.platform,
            controlled_axes=("roll", "pitch", "yaw"),
        ),
    )


def _structured_index_blocks() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Locate each regression's coefficients in the structured parameter order.

    The collective regression's eight coefficients are one contiguous block;
    each angular axis's eleven are scattered across four fields, because the
    structured order lists every axis of a field together.  Both are looked up
    by name so the layout follows the parameterization rather than restating
    it.
    """

    names = structured_parameter_names(
        BootstrapMultirotorParams(
            collective_acceleration_per_command=np.zeros(4),
            collective_velocity_coefficient=np.zeros(3),
            collective_intercept_m_s2=np.zeros(()),
            angular_acceleration_per_command=np.zeros((3, 4)),
            angular_rate_coefficient=np.zeros((3, 3)),
            angular_rate_product_coefficient=np.zeros((3, 3)),
            angular_intercept_rad_s2=np.zeros(3),
        )
    )
    index_of = {name: index for index, name in enumerate(names)}
    collective = np.asarray(
        [
            index_of[f"collective_acceleration_per_command[{motor}]"]
            for motor in range(4)
        ]
        + [index_of[f"collective_velocity_coefficient[{axis}]"] for axis in range(3)]
        + [index_of["collective_intercept_m_s2"]]
    )
    angular = tuple(
        np.asarray(
            [
                index_of[f"angular_acceleration_per_command[{axis},{motor}]"]
                for motor in range(4)
            ]
            + [
                index_of[f"angular_rate_coefficient[{axis},{other}]"]
                for other in range(3)
            ]
            + [
                index_of[f"angular_rate_product_coefficient[{axis},{pair}]"]
                for pair in range(3)
            ]
            + [index_of[f"angular_intercept_rad_s2[{axis}]"]]
        )
        for axis in range(3)
    )
    return collective, angular


PARAMETER_NAMES = structured_parameter_names(
    BootstrapMultirotorParams(
        collective_acceleration_per_command=np.zeros(4),
        collective_velocity_coefficient=np.zeros(3),
        collective_intercept_m_s2=np.zeros(()),
        angular_acceleration_per_command=np.zeros((3, 4)),
        angular_rate_coefficient=np.zeros((3, 3)),
        angular_rate_product_coefficient=np.zeros((3, 3)),
        angular_intercept_rad_s2=np.zeros(3),
    )
)
_COLLECTIVE_INDICES, _ANGULAR_INDICES = _structured_index_blocks()


def _feature_to_parameter_transform(
    span: np.ndarray,
    midpoint: np.ndarray,
    nuisance_size: int,
) -> np.ndarray:
    """Map the parameters onto the coefficients one regression solves for.

    A regression runs on normalized commands, so its command coefficient is
    ``span`` times the parameter, and its constant column absorbs the
    parameters' intercept plus the command effect at the box midpoint.  Every
    other feature is in its own units and passes through.  Congruence by this
    matrix carries a Gram in the regression's coordinates into precision in
    the parameters'.
    """

    size = 4 + nuisance_size
    transform = np.zeros((size, size), dtype=np.float64)
    transform[np.arange(4), np.arange(4)] = span
    passthrough = np.arange(4, size - 1)
    transform[passthrough, passthrough] = 1.0
    transform[size - 1, :4] = midpoint
    transform[size - 1, size - 1] = 1.0
    return transform


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
        self._sample_period_s: float | None = None
        self._integrated_regressor = np.zeros(8, dtype=np.float64)
        self._integrated_target = 0.0
        self._integrated_gram = np.zeros((9, 9), dtype=np.float64)
        self._integrated_rhs = np.zeros(9, dtype=np.float64)
        self._integrated_target_sum_squares = 0.0
        self._integrated_rows = 0
        self._force_gram_equivalent: np.ndarray | None = None
        self._force_gram = np.zeros((8, 8), dtype=np.float64)
        self._force_rhs = np.zeros((8, 1), dtype=np.float64)
        self._force_target_sum_squares = 0.0
        self._angular_gram = np.zeros((11, 11), dtype=np.float64)
        self._angular_rhs = np.zeros((11, 3), dtype=np.float64)
        self._angular_target_sum_squares = np.zeros(3, dtype=np.float64)
        self._input_spec = _bootstrap_input_spec(self._minimum, self._maximum)
        self._actuation = DirectActuationMap(self._input_spec.controls)
        self._runtime_spec = RuntimeModelSpec(
            sample_period_s=float(self.config.sample_period_s),
            validity_envelope=BOOTSTRAP_VALIDITY_ENVELOPE,
        )
        self._collective_transform = _feature_to_parameter_transform(
            self._span, self._midpoint, self._FORCE_NUISANCE_SIZE
        )
        self._angular_transform = _feature_to_parameter_transform(
            self._span, self._midpoint, self._ANGULAR_NUISANCE_SIZE
        )
        self._evidence = self._empty_evidence()
        self._belief = self._belief_from(
            BootstrapMultirotorParams(
                collective_acceleration_per_command=np.zeros(4),
                collective_velocity_coefficient=np.zeros(3),
                collective_intercept_m_s2=np.zeros(()),
                angular_acceleration_per_command=np.zeros((3, 4)),
                angular_rate_coefficient=np.zeros((3, 3)),
                angular_rate_product_coefficient=np.zeros((3, 3)),
                angular_intercept_rad_s2=np.zeros(3),
            ),
            self._evidence,
        )
        self._last_sample_report = RecursiveBootstrapSampleReport(
            interval_count=0,
            accepted=False,
            reason="no_sample_offered",
            update_wall_time_s=0.0,
        )
        self._rejected_sample_count = 0

    def _empty_evidence(self) -> BootstrapEvidence:
        return BootstrapEvidence(
            interval_count=0,
            collective_information=np.zeros((8, 8)),
            angular_information=np.zeros((11, 11)),
            collective_residual_std_m_s2=(
                self.config.collective_residual_std_floor_m_s2
            ),
            angular_residual_std_rad_s2=np.full(
                3,
                self.config.angular_residual_std_floor_rad_s2,
            ),
            normalized_command_support_projector=np.zeros((4, 4)),
            normalized_command_singular_values=np.zeros(4),
            normalized_command_information=np.zeros((4, 4)),
            angular_output_support_projector=np.zeros((3, 3)),
            supported_collective_effect_covariance=np.zeros((4, 4)),
            supported_angular_effect_covariance=np.zeros((3, 4, 4)),
            command_evidence_rank=0,
            angular_effect_rank=0,
            collective_nuisance_rank=0,
            angular_nuisance_rank=0,
            collective_support_fraction=0.0,
            minimum_supported_information_singular_value=0.0,
            information_authority=0.0,
            collective_effect_signal_to_noise=0.0,
            angular_effect_signal_to_noise=np.zeros(3),
            collective_authority=0.0,
            angular_axis_authority=np.zeros(3),
            exploration_completion=0.0,
            hover_command=None,
        )

    def _parameter_information(
        self, evidence: BootstrapEvidence
    ) -> ParameterInformation:
        """State this evidence as precision over the structured parameters.

        Each regression's Gram is congruence-transformed into the parameters'
        own coordinates and divided by that regression's residual variance, so
        one unit of precision is one transition's worth of information at the
        estimated residual scale.  The angular Gram is shared by the three
        body axes and enters once per axis at that axis's own residual.  Under
        the integrated collective target the exported collective Gram is
        already rescaled so that dividing it by the declared force floor,
        which is then the residual it reports, gives exactly the integrated
        system's precision; so one unit of collective precision is one
        transition's worth of information at that declared floor.

        One normalized unit is the coefficient perturbation that moves that
        regression's prediction by one residual standard deviation at the
        reference excitation, the full command box for a command coefficient
        and a unit feature for a nuisance one.  That is the scale the rank
        test needs: it removes the physical units, so a collective coefficient
        in metres per second squared and an angular one in radians per second
        squared are compared by how well the evidence pins each down rather
        than by how large its units happen to be.

        Nothing is floored into a small variance: an unexcited direction stays
        at zero precision and the rank test reports it as unresolved.
        """

        size = len(PARAMETER_NAMES)
        precision = np.zeros((size, size), dtype=np.float64)
        scale = np.ones(size, dtype=np.float64)
        collective = (
            self._collective_transform.T
            @ (
                evidence.collective_information
                / evidence.collective_residual_std_m_s2**2
            )
            @ self._collective_transform
        )
        precision[np.ix_(_COLLECTIVE_INDICES, _COLLECTIVE_INDICES)] = collective
        scale[_COLLECTIVE_INDICES] = evidence.collective_residual_std_m_s2
        scale[_COLLECTIVE_INDICES[:4]] /= self._span
        for axis, indices in enumerate(_ANGULAR_INDICES):
            block = (
                self._angular_transform.T
                @ (
                    evidence.angular_information
                    / evidence.angular_residual_std_rad_s2[axis] ** 2
                )
                @ self._angular_transform
            )
            precision[np.ix_(indices, indices)] = block
            scale[indices] = evidence.angular_residual_std_rad_s2[axis]
            scale[indices[:4]] /= self._span
        period_s = float(self.config.sample_period_s)
        floor = innovation_noise_floor()
        noise = np.array(floor, dtype=np.float64)
        declared = np.array(floor, dtype=np.float64)
        velocity = list(TANGENT_GROUP_INDICES["velocity"])
        angular = list(TANGENT_GROUP_INDICES["angular_velocity"])
        noise[velocity] = (period_s * evidence.collective_residual_std_m_s2) ** 2
        noise[angular] = (period_s * evidence.angular_residual_std_rad_s2) ** 2
        declared[velocity] = (
            period_s * self.config.collective_residual_std_floor_m_s2
        ) ** 2
        declared[angular] = (
            period_s * self.config.angular_residual_std_floor_rad_s2
        ) ** 2
        return ParameterInformation(
            names=PARAMETER_NAMES,
            precision=0.5 * (precision + precision.T),
            scale=scale,
            estimable=np.ones(size, dtype=bool),
            innovation_noise=noise,
            noise_floor=declared,
            effective_count=self._weight,
            rank_relative_tolerance=default_rank_relative_tolerance(
                evidence.interval_count, size
            ),
            source="recursive_bootstrap_identifier",
        )

    def _belief_from(
        self,
        params: BootstrapMultirotorParams,
        evidence: BootstrapEvidence,
    ) -> DynamicsBelief:
        """Assemble the belief this evidence and these parameters make."""

        return DynamicsBelief(
            model=ExecutableModel(
                params,
                self._input_spec,
                self._runtime_spec,
                self._actuation,
            ),
            information=self._parameter_information(evidence),
            forecast_error=None,
            provenance={
                "source": "recursive_bootstrap_identifier",
                "interval_count": evidence.interval_count,
                "bootstrap_evidence": evidence.to_dict(),
            },
        )

    @property
    def belief(self) -> DynamicsBelief:
        """The belief every measured transition so far supports."""

        return self._belief

    @property
    def evidence(self) -> BootstrapEvidence:
        """This estimator's own account of what that belief rests on."""

        return self._evidence

    @property
    def working_belief_supported(self) -> bool:
        """Whether the evidence currently meets the support conditions."""

        return self._evidence.supported

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
    ) -> DynamicsBelief:
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
        """Fold one sample into the accumulated normal equations.

        ``weight`` is the number of transitions the sample stands for: one for
        a plain transition, the window length for an aggregated one.  Nothing
        already accumulated is ever decayed: the evidence only grows.
        """

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
        self._weight += weight
        self._interval_count += round(weight)

    def _accumulate_integrated_collective(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        command: np.ndarray,
        sample_period_s: float,
        features: _SampleFeatures,
    ) -> None:
        """Fold one transition into the integrated collective system.

        Accumulate cumulative body-z specific force and features, with an
        intercept for the anchor. Temporal error covariance is not whitened.
        """

        del previous, current, command
        self._integrated_regressor += sample_period_s * features.force_features
        self._integrated_target += sample_period_s * float(
            features.body_specific_force[2]
        )
        row = np.concatenate((self._integrated_regressor, np.ones(1)))
        self._integrated_gram += np.outer(row, row)
        self._integrated_rhs += row * self._integrated_target
        self._integrated_target_sum_squares += self._integrated_target**2
        self._integrated_rows += 1

    def _integrated_force_system(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The integrated system as an equivalent per-transition Gram and rhs.

        The anchor column is marginalized by its Schur complement, the
        residual scale of the integrated system is estimated from its own
        residual with the declared force floor over one interval as its
        minimum, and the marginal is rescaled by the declared force floor over
        that scale, so dividing the result by the floor gives exactly the
        integrated precision in the per-transition machinery's units.
        ``None`` until the system is determined.
        """

        unknowns = 9
        if self._integrated_rows <= unknowns or self._sample_period_s is None:
            return None
        gram = self._integrated_gram
        rhs = self._integrated_rhs
        ridge = 1e-12 * max(float(np.trace(gram)) / unknowns, 1e-12)
        solution = np.linalg.solve(gram + ridge * np.eye(unknowns), rhs)
        residual = (
            self._integrated_target_sum_squares
            - 2.0 * solution @ rhs
            + solution @ gram @ solution
        )
        degrees = max(self._integrated_rows - unknowns, 1)
        floor = self.config.collective_residual_std_floor_m_s2 * self._sample_period_s
        velocity_variance = max(residual / degrees, floor * floor)
        theta = gram[:8, :8]
        cross = gram[:8, 8]
        offset = max(float(gram[8, 8]), 1e-12)
        marginal_gram = theta - np.outer(cross, cross) / offset
        marginal_rhs = rhs[:8] - cross * (rhs[8] / offset)
        scale = self.config.collective_residual_std_floor_m_s2**2 / velocity_variance
        return scale * 0.5 * (marginal_gram + marginal_gram.T), scale * marginal_rhs

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

    def _fit_supported_effects(self) -> _EffectFit:
        """Solve both regressions and rescale them back to raw command units.

        The regressions run on normalized commands so their rank tests have a
        single scale; the effects, intercepts, and covariances returned here are
        already mapped back to the raw command box the vehicle is flown in.
        """

        force_gram, force_rhs = self._force_gram, self._force_rhs
        self._force_gram_equivalent = None
        integrated = self._integrated_force_system()
        if integrated is not None:
            force_gram, force_rhs = integrated[0], integrated[1][:, None]
            self._force_gram_equivalent = force_gram
        force = self._supported_fit(
            force_gram,
            force_rhs,
            nuisance_size=self._FORCE_NUISANCE_SIZE,
            effective_count=self._weight,
            relative_tolerance=self.config.command_rank_relative_tolerance,
            minimum_rms=self.config.minimum_normalized_command_rms,
            nuisance_relative_tolerance=(self.config.nuisance_rank_relative_tolerance),
        )
        angular = self._supported_fit(
            self._angular_gram,
            self._angular_rhs,
            nuisance_size=self._ANGULAR_NUISANCE_SIZE,
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
        if self._force_gram_equivalent is not None:
            # The equivalent Gram is scaled so that dividing it by the floor
            # gives the integrated precision; the floor is therefore the scale
            # to report with it.
            force_residual_std = self.config.collective_residual_std_floor_m_s2
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

    def _assimilated_parameters(self, fit: _EffectFit) -> BootstrapMultirotorParams:
        """The bootstrap model this sample leaves the identifier holding."""

        return BootstrapMultirotorParams(
            collective_acceleration_per_command=fit.force_effect,
            collective_velocity_coefficient=fit.force_nuisance[:3, 0],
            collective_intercept_m_s2=np.asarray(fit.force_intercept),
            angular_acceleration_per_command=fit.angular_effect,
            angular_rate_coefficient=fit.angular_nuisance[:3].T,
            angular_rate_product_coefficient=fit.angular_nuisance[3:6].T,
            angular_intercept_rad_s2=fit.angular_intercept,
        )

    def _assimilated_evidence(
        self,
        fit: _EffectFit,
        authority: _BeliefAuthority,
    ) -> BootstrapEvidence:
        """This estimator's own account of the evidence behind that model."""

        return BootstrapEvidence(
            interval_count=self._interval_count,
            collective_information=(
                self._force_gram
                if self._force_gram_equivalent is None
                else self._force_gram_equivalent
            ),
            angular_information=self._angular_gram,
            collective_residual_std_m_s2=fit.force_residual_std,
            angular_residual_std_rad_s2=fit.angular_residual_std,
            normalized_command_support_projector=fit.angular_support,
            normalized_command_singular_values=fit.angular_singular_values,
            normalized_command_information=fit.angular_residual_information,
            angular_output_support_projector=authority.angular_output_support,
            supported_collective_effect_covariance=fit.collective_effect_covariance,
            supported_angular_effect_covariance=fit.angular_effect_covariance,
            command_evidence_rank=fit.angular_command_rank,
            angular_effect_rank=authority.angular_effect_rank,
            collective_nuisance_rank=fit.force_nuisance_rank,
            angular_nuisance_rank=fit.angular_nuisance_rank,
            collective_support_fraction=authority.collective_support,
            minimum_supported_information_singular_value=(
                authority.minimum_supported_information
            ),
            information_authority=authority.information_authority,
            collective_effect_signal_to_noise=(
                authority.collective_effect_signal_to_noise
            ),
            angular_effect_signal_to_noise=authority.angular_effect_signal_to_noise,
            collective_authority=authority.collective_authority,
            angular_axis_authority=authority.angular_axis_authority,
            exploration_completion=authority.exploration_completion,
            hover_command=authority.hover_command,
        )

    def update(
        self,
        previous_state: Sequence[float],
        current_state: Sequence[float],
        average_applied_motor_command: Sequence[float],
        sample_period_s: float,
    ) -> DynamicsBelief:
        """Assimilate one measured transition and return the current belief.

        Transitions are assimilated in windows of
        :data:`TRANSITION_AGGREGATION_STEPS`, as the window's mean features
        and mean targets weighted by its length, so the sample count, the
        support thresholds, and the residual floor all stay per transition.
        The collective map is fit on the integrated target throughout.

        A transition that is not usable evidence is refused rather than raised
        on, because a single non-finite estimator sample must not end a control
        loop.  The belief is then left exactly as it was and the refusal is
        recorded in :attr:`last_sample_report`.
        """

        started_at = time.perf_counter()
        self._sample_period_s = float(sample_period_s)
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
        self._accumulate_integrated_collective(*sample, sample_period_s, features)
        self._pending_transitions.append(features)
        if len(self._pending_transitions) < TRANSITION_AGGREGATION_STEPS:
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
        self._accumulate_sample(features, weight=float(TRANSITION_AGGREGATION_STEPS))
        fit = self._fit_supported_effects()
        self._evidence = self._assimilated_evidence(fit, self._belief_authority(fit))
        self._belief = self._belief_from(
            self._assimilated_parameters(fit),
            self._evidence,
        )
        self._last_sample_report = RecursiveBootstrapSampleReport(
            interval_count=self._interval_count,
            accepted=True,
            reason="sample_assimilated",
            update_wall_time_s=time.perf_counter() - started_at,
        )
        return self._belief
