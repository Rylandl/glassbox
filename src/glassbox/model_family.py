"""Vehicle-family contracts shared by ingestion, fitting, and serialization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsModelFamily:
    """Static schema and capability contract for one dynamics family."""

    key: str
    platform: str
    control_names: tuple[str, ...]
    control_roles: tuple[str, ...]
    latent_state_names: tuple[str, ...]
    required_control_roles: tuple[str, ...]
    optional_control_roles: tuple[str, ...] = ()
    optional_parameter_control_dependencies: tuple[
        tuple[str, str], ...
    ] = ()
    supports_residual: bool = False

    def __post_init__(self) -> None:
        dependencies = dict(self.optional_parameter_control_dependencies)
        if len(dependencies) != len(self.optional_parameter_control_dependencies):
            raise ValueError("optional parameter dependencies must be unique")
        unsupported = set(dependencies.values()) - set(self.optional_control_roles)
        if unsupported:
            raise ValueError(
                "parameter dependencies must name optional control roles"
            )

    @property
    def control_size(self) -> int:
        return len(self.control_names)

    def validate_control_names(
        self,
        control_names: tuple[str, ...],
    ) -> None:
        names = tuple(control_names)
        if len(names) != self.control_size:
            raise ValueError(
                f"{self.key} dynamics require exactly {self.control_size} "
                f"control channels, got {len(names)}"
            )
        if names == self.control_names:
            return
        raise ValueError(
            f"{self.key} requires ordered controls {self.control_names}, got {names}"
        )

    def validate_control_roles(self, control_roles: tuple[str, ...]) -> None:
        """Validate one statically ordered role layout supported by the model."""

        roles = tuple(control_roles)
        if len(set(roles)) != len(roles):
            raise ValueError("control roles must be unique")
        if self.platform == "multirotor" and roles != self.control_roles:
            raise ValueError(
                f"{self.key} requires ordered control roles "
                f"{self.control_roles}, got {roles}"
            )
        missing = [role for role in self.required_control_roles if role not in roles]
        if missing:
            raise ValueError(
                f"{self.key} requires control roles {self.required_control_roles}; "
                f"missing {tuple(missing)} from {roles}"
            )
        supported = set(self.required_control_roles) | set(
            self.optional_control_roles
        )
        unsupported = [role for role in roles if role not in supported]
        if unsupported:
            raise ValueError(
                f"{self.key} does not support control roles {tuple(unsupported)}; "
                f"supported roles are {tuple(sorted(supported))}"
            )

    def validate_control_schema(
        self,
        control_names: tuple[str, ...],
        control_roles: tuple[str, ...] | None,
    ) -> None:
        if control_roles is None:
            self.validate_control_names(control_names)
            return
        if len(control_names) != len(control_roles):
            raise ValueError("control names and roles must have the same length")
        self.validate_control_roles(control_roles)
        if self.platform == "multirotor":
            self.validate_control_names(control_names)

    def parameter_control_dependency(self, parameter_name: str) -> str | None:
        """Return the optional control required to identify one parameter."""

        return dict(self.optional_parameter_control_dependencies).get(
            parameter_name
        )


MULTIROTOR_FAMILY = DynamicsModelFamily(
    key="effective_quadrotor",
    platform="multirotor",
    control_names=(
        "motor_front_left",
        "motor_front_right",
        "motor_rear_right",
        "motor_rear_left",
    ),
    control_roles=(
        "motor_front_left",
        "motor_front_right",
        "motor_rear_right",
        "motor_rear_left",
    ),
    latent_state_names=(
        "applied_motor_front_left",
        "applied_motor_front_right",
        "applied_motor_rear_right",
        "applied_motor_rear_left",
    ),
    required_control_roles=(
        "motor_front_left",
        "motor_front_right",
        "motor_rear_right",
        "motor_rear_left",
    ),
    supports_residual=True,
)


FIXED_WING_FAMILY = DynamicsModelFamily(
    key="effective_fixedwing",
    platform="fixedwing",
    control_names=("throttle", "aileron", "elevator", "rudder"),
    control_roles=("throttle", "roll", "pitch", "yaw"),
    latent_state_names=(
        "applied_throttle",
        "applied_aileron",
        "applied_elevator",
        "applied_rudder",
    ),
    required_control_roles=("throttle", "roll", "pitch"),
    optional_control_roles=("yaw", "flap"),
    optional_parameter_control_dependencies=(
        ("log_surface_angular_accel_per_speed_sq[2]", "yaw"),
        ("lateral_surface_cross_angular_accel_per_speed_sq[0]", "yaw"),
        ("surface_trim_unconstrained[2]", "yaw"),
        ("log_flap_lift_accel_per_speed_sq", "flap"),
        ("log_flap_drag_accel_per_speed_sq", "flap"),
        ("flap_pitch_angular_accel_per_speed_sq", "flap"),
        ("flap_trim_unconstrained", "flap"),
    ),
    supports_residual=True,
)


MODEL_FAMILIES = {
    MULTIROTOR_FAMILY.platform: MULTIROTOR_FAMILY,
    FIXED_WING_FAMILY.platform: FIXED_WING_FAMILY,
}


def family_for_platform(platform: str) -> DynamicsModelFamily:
    """Return the registered family for a canonical platform label."""

    try:
        return MODEL_FAMILIES[platform]
    except KeyError as error:
        supported = ", ".join(sorted(MODEL_FAMILIES))
        raise ValueError(
            f"unsupported dynamics platform {platform!r}; expected one of {supported}"
        ) from error
