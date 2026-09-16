# Cross-platform sequence learning evidence

See [the investigation](../../sequence-transfer.md) for methods, findings and
limitations. The frozen comparison and two adaptive follow-ups are preserved
separately.

- `initial-*`: frozen comparison plan, environment, source identities and
  executed code.
- `regularization-*`: adaptive affine-family/ridge selection plan, summary and
  executed code.
- `refinement-*`: adaptive nonlinear refinement plan, summary and executed code.
- `case-reports.zip`: individual model reports, training traces and exact
  per-recording window origins for all three runs.
- `prepared-metadata.zip` and `inventory.json`: source identities, selected
  intervals, timing conventions and signal-age summaries for X8 and ARP.
- `x8-upstream-readme.txt`: the dataset author's preprocessing description.
- `summary.json`, `selection.json`: means, seed ranges and selected candidates.
  These include methods omitted from the figure.
- `audit.json`: independent NumPy replay of predictions, data extraction and
  selection, with perturbation-based checks of the affine history operators.
- `final-implementation.zip`: final source package, experiment/report scripts,
  targeted tests and package configuration. Executed source archives remain
  unchanged, including the original float64 guard comparison.
- `source-archive-members.json`: hashes of every archived source member.
- `starting-hashes.json`: identities of preexisting workspace files.
- `validation.json`, `source-pytest.xml`, `wheel-pytest.xml`: final source and
  isolated-wheel checks, environment and build hash.
- `sequence-transfer.png` / `.svg`: evaluation errors; error bars are ranges
  across sampling seeds, not independent-flight confidence intervals.
- `manifest.json`: file sizes and SHA-256 hashes for this evidence bundle.

Large model, sample and prediction arrays remain under
`artifacts/sequence-transfer` in the parent workspace. Raw X8 CSVs and ARP ULogs
are in that directory's `corpora` subdirectory. Nano windows and source records
also depend on the earlier real-transition and model-structure investigations.
The bundle is reviewable evidence, not a standalone copy of the flight corpora.
Absolute paths in archived metadata identify the recorded local run; point
reproduction commands at the equivalent downloaded corpora on another machine.

Same recipe means separate fits with the same learning procedure, not shared
learned weights. X8 and Nano retain upstream processing. ARP features respect
recorded publication times but remain onboard estimator observations. All
forecasts receive the future logged input sequence. The regularized ARP model
still loses to hold-current on its one evaluation flight. No new uncertainty,
unsupported-regime accuracy or controller result is established here.
