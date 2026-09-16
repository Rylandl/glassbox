# Generic model structure evidence

See [the investigation](../../model-structure-experiments.md) for interpretation.

- `initial-*`: the original frozen plan, environment, source hashes and executed sources.
- `followup-*`: the adaptive explicit-history experiment, preserved separately.
- `case-reports.zip`: all selected-model reports, training traces, source hashes and sequence window usage.
- `summary.json`: all plotted means and sampling-seed ranges, including methods omitted from plots.
- `audit.json`: independent NumPy replay and provenance checks.
- `final-implementation.zip`: final implementations, runner, reporter and regression tests.
- `starting-hashes.json`: workspace file identities before this investigation.
- PNG/SVG files: direct and recursive comparisons.
- `validation.json`: test and build verification.

Large fitted models, training/evaluation windows and saved predictions remain under
`artifacts/model-structures` in the parent workspace. The direct comparison also
references the pinned `artifacts/transition-diagnosis/diagnosis-01` artifacts.
The committed evidence is not a standalone copy of the flight dataset.

The original run and adaptive follow-up have different executed source archives.
Final source adds the delay-MLP option; it preserves the original prediction paths.
Both evaluation groups were used in prior investigations. No untouched holdout,
calibrated uncertainty or second-platform result is claimed.
