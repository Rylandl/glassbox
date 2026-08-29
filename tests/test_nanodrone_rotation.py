import json
from dataclasses import replace

from glassbox.data import save_trajectory_npz
from glassbox.model_io import save_dynamics_model
from glassbox.nanodrone_rotation import (
    select_nanodrone_angular_authority,
    select_nanodrone_rotation_candidate,
)
from glassbox.runtime import runtime_spec_from_trajectory
from glassbox.synthetic import generate_trajectory, true_parameters


def _metrics(value: float) -> dict[str, float]:
    return {
        "position_rmse_m": value,
        "velocity_rmse_m_s": value,
        "attitude_rmse_deg": value,
        "angular_velocity_rmse_rad_s": value,
    }


def _summary(value: float, profiles: tuple[str, ...]) -> dict:
    return {
        "platform": "multirotor",
        "profiles": list(profiles),
        "per_profile": {
            profile: {
                "full_rollout": _metrics(value),
                "horizon_rollouts": {
                    label: _metrics(value)
                    for label in ("0.1s", "0.5s", "1s")
                },
            }
            for profile in profiles
        },
    }


def test_train_only_rotation_selection_requires_material_improvement(tmp_path) -> None:
    profiles = ("chirp", "random", "square")
    reference = tmp_path / "reference.json"
    weak = tmp_path / "weak.json"
    strong = tmp_path / "strong.json"
    output = tmp_path / "selection.json"
    reference.write_text(json.dumps(_summary(1.0, profiles)))
    weak.write_text(json.dumps(_summary(0.995, profiles)))
    strong.write_text(json.dumps(_summary(0.98, profiles)))

    decision = select_nanodrone_rotation_candidate(
        {
            "instantaneous_diagonal": reference,
            "weak": weak,
            "strong": strong,
        },
        output,
    )

    assert decision["selected_candidate"] == "strong"
    assert decision["decision_scope"]["uses_public_melon_test_data"] is False
    assert output.read_text().endswith("\n")


def test_angular_authority_selection_uses_only_held_out_train_profiles(
    tmp_path,
) -> None:
    trajectory_paths = []
    model_paths = {}
    for seed, profile in enumerate(("chirp", "random", "square")):
        trajectory = replace(
            generate_trajectory(seed=seed, duration_s=1.1),
            labels={"profile": profile, "benchmark_split": "train"},
        )
        trajectory_path = tmp_path / f"{profile}.npz"
        model_path = tmp_path / f"{profile}.json"
        save_trajectory_npz(trajectory, trajectory_path)
        save_dynamics_model(
            true_parameters(),
            model_path,
            input_spec=trajectory.spec,
            runtime_spec=runtime_spec_from_trajectory(trajectory),
        )
        trajectory_paths.append(trajectory_path)
        model_paths[profile] = model_path

    decision = select_nanodrone_angular_authority(
        model_paths,
        trajectory_paths,
        tmp_path / "authority.json",
    )

    assert decision["selected_authority"] == 1.0
    assert decision["decision_scope"]["uses_protected_evaluation_data"] is False
    assert decision["decision_scope"]["required_benchmark_split"] == "train"
