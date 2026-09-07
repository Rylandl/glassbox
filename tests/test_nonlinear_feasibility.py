"""Host feasibility summaries do not imply optimization or deadline success."""

import numpy as np
import pytest

from glassbox.control.plan import NonlinearFeasibility


@pytest.mark.parametrize(
    "margins,status,violation",
    [
        ([0.1, -1e-7], "feasible", 1e-7),
        ([0.1, -0.01], "infeasible", 0.01),
        ([], "feasible", 0.0),
        ([np.nan], "infeasible", np.inf),
        ([np.inf], "infeasible", np.inf),
    ],
)
def test_signed_margin_assessment(margins, status, violation):
    assessment = NonlinearFeasibility.from_margins(np.asarray(margins), tolerance=1e-6)
    assert assessment.status == status
    assert assessment.constraint_count == len(margins)
    assert assessment.maximum_violation == pytest.approx(violation)


def test_unassessed_is_distinct_from_no_declared_constraints():
    unknown = NonlinearFeasibility()
    assert unknown.status == "not_assessed"
    assert unknown.constraint_count is None
    assert unknown.maximum_violation is None
    assert unknown.tolerance is None


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf])
def test_invalid_tolerances_are_never_feasible(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        NonlinearFeasibility.from_margins(np.ones(1), tolerance=tolerance)


def test_margins_must_preserve_vector_contract():
    with pytest.raises(ValueError, match="one dimensional"):
        NonlinearFeasibility.from_margins(np.ones((1, 2)), tolerance=1e-6)
