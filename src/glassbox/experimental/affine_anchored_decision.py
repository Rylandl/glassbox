"""The pinned residual reducer with this experiment's progress label."""

from copy import deepcopy

from .excited_bilinear_decision import reduce as _pinned_reduce

_OLD_PROGRESS = "bilinear_public_progress"
_PROGRESS = "anchored_public_progress"
_COMPARISONS = {_PROGRESS, "quadratic_gain_retention", "quadratic_public_context"}


def reduce(summaries, protocol, rows=None):
    """Preserve every numerical rule and distinguish external qualification.

    An isolated protocol view supplies the historical reducer's comparison name.
    Its fresh result receives the new label; neither caller data nor historical
    module bindings change. Raw rows, pairwise availability, angular point gates,
    shared bootstrap draws and pending external verification retain their meaning.
    """
    if set(protocol["decision"]["comparisons"]) != _COMPARISONS:
        raise ValueError("frozen anchored comparison roster differs")
    view = deepcopy(protocol)
    policies = view["decision"]["comparisons"]
    policies[_OLD_PROGRESS] = policies.pop(_PROGRESS)
    result = _pinned_reduce(summaries, view, rows)
    for fields in (
        result,
        result["comparisons"],
        result["checks"],
        result["bootstrap_comparison_hashes"],
    ):
        fields[_PROGRESS] = fields.pop(_OLD_PROGRESS)
    return result
