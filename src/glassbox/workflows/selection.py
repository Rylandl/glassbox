"""The shared candidate-selection thresholds every promotion gate applies.

Selection compares a candidate against a reference as a ratio, where values
below one favor the candidate. Three thresholds decide the outcome, and every
gate in the package uses the same numbers so that a result promoted on one
platform means the same thing on another:

* a candidate may not make any single metric much worse, even when its average
  improves, which is what :data:`MAXIMUM_METRIC_REGRESSION` bounds;
* an average improvement may not hide a regression on one platform, which is
  what :data:`MAXIMUM_PLATFORM_REGRESSION` bounds; the fixed-wing gate calls
  its platforms airframes, but applies the same limit;
* an improvement must be large enough to be worth adopting at all, which is
  what :data:`MINIMUM_OVERALL_IMPROVEMENT` requires.

These are decision policy, not tuning knobs. Callers may still override them
for an experiment, but the defaults live here so a change is deliberate and
visible in one place.
"""

from __future__ import annotations

MAXIMUM_METRIC_REGRESSION = 1.5
MAXIMUM_PLATFORM_REGRESSION = 1.05
MINIMUM_OVERALL_IMPROVEMENT = 0.01

# The largest overall score that still counts as a real improvement.
MAXIMUM_SELECTABLE_OVERALL_SCORE = 1.0 - MINIMUM_OVERALL_IMPROVEMENT
