from glassbox.dynamics import ResidualDynamicsParams, initial_residual_parameters
from glassbox.history_selection import (
    HISTORY_RESPONSE_CANDIDATES_S,
    select_history_residual,
)
from glassbox.synthetic import generate_trajectory, true_parameters


def test_history_selection_retains_exact_reference_without_evidence() -> None:
    trajectory = generate_trajectory(seed=21, duration_s=0.3)
    reference = initial_residual_parameters(true_parameters(), hidden_units=2)

    selected, report = select_history_residual(
        reference,
        [trajectory],
        horizons_s=(0.1,),
    )

    assert isinstance(selected, ResidualDynamicsParams)
    assert report["selected"] is False
    assert report["status"] == "retained_instantaneous_reference"
    assert len(report["force"]["candidate_scores"]) == len(
        HISTORY_RESPONSE_CANDIDATES_S
    )
    assert report["production_lockbox_required"] is True
