from types import SimpleNamespace

import numpy as np

import glassbox.workflows.angular_authority as authority_module
from glassbox.workflows.angular_authority import (
    evaluate_angular_dynamics_candidate,
    select_angular_dynamics_authority,
)


def _metrics(value: float) -> dict[str, object]:
    return {
        "position_rmse_m": value,
        "velocity_rmse_m_s": value,
        "attitude_rmse_deg": value,
        "angular_velocity_rmse_rad_s": value,
        "position_rmse_xyz_m": [value] * 3,
        "velocity_rmse_xyz_m_s": [value] * 3,
        "attitude_rotation_vector_rmse_xyz_deg": [value] * 3,
        "angular_velocity_rmse_xyz_rad_s": [value] * 3,
        "final_position_error_m": value,
        "duration_s": 1.0,
        "sample_count": 10,
        "rollout_count": 1,
    }


def test_source_group_authority_selection_is_train_only(tmp_path, monkeypatch) -> None:
    spec_payload = {"vehicle": {"family": "multirotor"}}
    spec = SimpleNamespace(
        vehicle=SimpleNamespace(family="multirotor"),
        to_dict=lambda: spec_payload,
    )
    trajectories = {}
    models = {}
    for fold in ("flight-a", "flight-b", "flight-c"):
        trajectory_path = (tmp_path / f"{fold}.npz").resolve()
        model_path = (tmp_path / f"{fold}.json").resolve()
        trajectory_path.write_bytes(fold.encode())
        model_path.write_bytes((fold + "-model").encode())
        trajectories[trajectory_path] = SimpleNamespace(
            labels={"source_group": fold},
            spec=spec,
            nominal_dt_s=0.02,
        )
        models[fold] = model_path

    monkeypatch.setattr(
        authority_module,
        "load_trajectory_npz",
        lambda path: trajectories[path.resolve()],
    )
    monkeypatch.setattr(
        authority_module,
        "load_dynamics_model",
        lambda path: (object(), {"platform": "multirotor", "input_spec": spec_payload}),
    )
    monkeypatch.setattr(
        authority_module,
        "with_angular_dynamics_authority",
        lambda params, authority: authority,
    )
    monkeypatch.setattr(
        authority_module,
        "rollout_metrics",
        lambda authority, trajectory: _metrics(1.0 + abs(authority - 0.75)),
    )
    monkeypatch.setattr(
        authority_module,
        "windowed_rollout_metrics",
        lambda authority, trajectory, **_kwargs: _metrics(1.0 + abs(authority - 0.75)),
    )
    monkeypatch.setattr(
        authority_module,
        "kinematic_persistence_windowed_metrics",
        lambda trajectory, **_kwargs: _metrics(1.1),
    )

    decision = select_angular_dynamics_authority(
        models,
        list(trajectories),
        tmp_path / "selection.json",
        fold_axis="source_group",
        dataset_name="second_airframe",
    )

    assert decision["selected_authority"] == 0.75
    assert decision["decision_scope"]["uses_protected_evaluation_data"] is False
    assert decision["folds"] == ["flight-a", "flight-b", "flight-c"]
    assert (
        decision["selected_candidate_vs_kinematic_persistence"]["geometric_ratio"] < 1.0
    )


def test_candidate_gate_requires_reference_and_persistence_improvement(
    monkeypatch,
) -> None:
    trajectory = SimpleNamespace(
        labels={"source_group": "protected"},
        spec=SimpleNamespace(vehicle=SimpleNamespace(family="multirotor")),
        nominal_dt_s=0.02,
        time_s=np.asarray([0.0, 1.0]),
    )

    def metrics(params, trajectory, **_kwargs):
        return _metrics({"reference": 1.0, "candidate": 0.8}[params])

    monkeypatch.setattr(authority_module, "windowed_rollout_metrics", metrics)
    monkeypatch.setattr(
        authority_module,
        "kinematic_persistence_windowed_metrics",
        lambda trajectory, **_kwargs: _metrics(0.9),
    )
    monkeypatch.setattr(authority_module, "rollout_metrics", metrics)
    monkeypatch.setattr(
        authority_module,
        "model_family",
        lambda params: SimpleNamespace(platform="multirotor"),
    )
    monkeypatch.setattr(
        authority_module,
        "rollout_divergence_metrics",
        lambda params, trajectory: {"diverged": False},
    )

    report = evaluate_angular_dynamics_candidate("reference", "candidate", [trajectory])

    assert report["status"] == "promote_complete_flight"
    assert report["gates"]["improves_fitted_reference"]["passed"] is True
    assert report["gates"]["beats_kinematic_persistence"]["passed"] is True


def test_candidate_gate_does_not_promote_when_persistence_is_better(
    monkeypatch,
) -> None:
    trajectory = SimpleNamespace(
        labels={"source_group": "protected"},
        spec=SimpleNamespace(vehicle=SimpleNamespace(family="multirotor")),
        nominal_dt_s=0.02,
        time_s=np.asarray([0.0, 1.0]),
    )

    def metrics(params, trajectory, **_kwargs):
        return _metrics({"reference": 1.0, "candidate": 0.8}[params])

    monkeypatch.setattr(authority_module, "windowed_rollout_metrics", metrics)
    monkeypatch.setattr(
        authority_module,
        "kinematic_persistence_windowed_metrics",
        lambda trajectory, **_kwargs: _metrics(0.7),
    )
    monkeypatch.setattr(authority_module, "rollout_metrics", metrics)
    monkeypatch.setattr(
        authority_module,
        "model_family",
        lambda params: SimpleNamespace(platform="multirotor"),
    )
    monkeypatch.setattr(
        authority_module,
        "rollout_divergence_metrics",
        lambda params, trajectory: {"diverged": False},
    )

    report = evaluate_angular_dynamics_candidate("reference", "candidate", [trajectory])

    assert report["status"] == "improves_reference_only"
    assert report["gates"]["improves_fitted_reference"]["passed"] is True
    assert report["gates"]["beats_kinematic_persistence"]["passed"] is False
