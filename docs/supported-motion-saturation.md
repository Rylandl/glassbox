# Support-preserving saturation: local response recovered, growth still contained

The fixed representation change passes its predeclared diagnostic hypothesis.
It largely restores the original model's short-horizon command response while
retaining the previous bound's elimination of observed runaway on all 189 known
Dart queries. It gives back some of the previous bound's long-horizon forecast
improvement. This is a useful measured tradeoff, not universal error reduction,
fresh generalization or controller qualification.

The [protocol](harness/supported-motion-saturation-v1.json) was committed before
implementation. The [result record](harness/supported-motion-saturation-v1-result.json)
anchors the source, saved candidate, complete inputs and predictions, replay,
independent audit and evidence inventory. No new fit, training objective,
model gradient, simulator rollout or controller trial was run.

## One representation change

For each of the six normalized body-velocity/angular-rate coordinates, let
`S = max(1, maximum absolute training coordinate)` and retain the previous
asymptotic limit `L = 4*S`. The new map is:

```text
phi(z) = z                                      if |z| <= S
phi(z) = sign(z) * (S + 3*S*tanh((|z|-S)/(3*S))) otherwise
```

This replaces `L*tanh(z/L)`. It is odd, monotone and twice continuously
differentiable. Its value, first derivative and second derivative are exactly
those of the identity within support and at the joins. Outside support it
approaches the same bound as before. Finite-precision saturation can equal the
bound. The map bounds features; it is not a global stability guarantee.

The six supports come only from the existing 1,048 training windows, including
their past states and final future targets, using the original saved body
normalization. Development and evaluation recordings do not set them. All
1,075,248 observed training motion coordinates, counted across the overlapping
cached windows, remain bitwise unchanged under
the map in both float32 and float64. Every parameter and normalization array
is copied bitwise from the previous bounded initialization; only the core
format and feature function change. Mechanics, filters, history, timing and
integration are unchanged. There is no vehicle selector or consumer option.

Exact identity on observed coordinates does not establish identity of a
recursive rollout: predicted and intermediate integration states can leave
the training envelope. No claim is made that the old training objective is
restored; that objective was not evaluated here.

## Population and predeclared readout

The four fixed arms are original shared-v1, the previous tanh-bound
initialization, the new supported initialization and the prior unbounded
refinement. The same complete known query roster is used as in the
[preceding diagnostic](conditioning-attribution.md): six pilot parents provide
12 factual forecasts and 96 response branches; one task parent provides one
forecast and eight branches; two test parents provide 72 forecast queries.
All 189 queries were previously inspected. Old archive `fresh` labels are
provenance only and are reported as **known-pilot**.

Physical component RMSEs retain the previous endpoint/prefix, signed
branch-minus-factual, parent/origin/branch weighting and failure rules. A prefix
requires valid truth at every step through its horizon. Failed predictions
remain visible and make full-cohort errors undefined; paired conditional
comparisons always expose their surviving counts.

The local-response readout was frozen at 50, 150 and 250 ms, covering the
original fitted horizon. Each scope has an equally weighted geometric mean of
18 RMSE ratios: three physical groups, two statistics and three horizons,
using the existing physical floors. Pilot and task means remain separate.
Every declared cell requires full truth eligibility and full paired coverage.

| Known response scope | New / previous tanh | Change | New / original shared-v1 |
| --- | ---: | ---: | ---: |
| Pilot, 96 branches | 0.916469 | −8.35% | 1.001720 |
| Task, 8 branches | 0.987654 | −1.23% | 1.00000018 |

Both scope ratios improve over tanh, and every new trajectory remains finite
and below the declared catastrophic-growth thresholds. The frozen
`local_response_tradeoff_improved` readout is therefore true. It is a
hypothesis result on known evidence, not an adoption or adequacy criterion.
The small task change should be read against its single parent; it is not
evidence of a precisely estimated population effect.

## Growth and physical accuracy

Among the 96 known-pilot response trajectories, original shared-v1 still has
nine nonfinite and four extreme finite predictions; the old unbounded
refinement has eight and four. Both tanh and the new map have zero of either.
Across all 189 new trajectories, the largest predicted speed and body-rate
norms are 23.511 m/s and 23.779 rad/s. The catastrophic threshold is 1,000 m/s
or 1,000 rad/s, not an application tolerance.

The complete pilot response prefix at 250 ms illustrates the recovery:

| Revision | Velocity RMSE (m/s) | Body-rate RMSE (rad/s) | Rotation-entry RMSE |
| --- | ---: | ---: | ---: |
| Original shared-v1 | 0.019188 | 0.136327 | 0.008830 |
| Previous tanh bound | 0.019716 | 0.173212 | 0.010618 |
| New supported map | 0.019162 | 0.137283 | 0.008876 |
| Prior unbounded refinement | 0.021392 | 0.135730 | 0.008761 |

The gain is recovery of local behavior, rather than substantially better
local dynamics than the original model. Pilot response errors at 600 ms also
improve in all three groups and both statistics versus tanh and shared-v1.
Task results at that horizon are mixed. Every measured horizon and forecast
remains in the complete evidence.

![Known-pilot growth and accuracy tradeoff](../artifacts/2026-09-21/supported-motion-saturation-v1-evidence/figures/supported-motion-saturation-summary.png)

The cost is systematic in long forecasts. Every 1.2-second forecast prefix
group worsens versus the previous tanh bound:

| Known forecast scope | Velocity RMSE change | Body-rate RMSE change | Rotation-entry RMSE change |
| --- | ---: | ---: | ---: |
| Pilot, 12 queries | +6.84% | +27.91% | +6.92% |
| Task, 1 query | +10.90% | +14.63% | +18.61% |
| Test, 12 long queries | +6.07% | +36.78% | +10.56% |

Long forecast rate prefixes still improve versus original shared-v1 by about
40.8%, 2.9% and 26.0%, respectively. For example, pilot rate RMSE is
3.320 rad/s originally, 1.536 with tanh and 1.965 with the new map.
There is no claim that the new map dominates tanh.

Full 1.2-second pilot response endpoint errors remain 3.942 m/s and
9.601 rad/s, versus tanh's 3.992 m/s and 8.920 rad/s. New prefix errors are
1.647 m/s and 4.193 rad/s. Finite predictions still leave a substantial task
accuracy gap. The unbounded models' full-cohort errors at this horizon remain
undefined; their conditional ratios exclude nine or eight failed branches
and cannot be presented as complete-population improvements.

## Qualification and next gap

All 124 preflight tests pass, covering map derivatives and numerical extremes,
unchanged integration, training-only preparation, exact copied arrays, reducer
isolation, failure coverage and provenance defects. The 567 historical
reference predictions reproduce exactly on each pass. All 756 predictions,
physical reductions and provisional readouts replay exactly. The independent
implementation verifies all metric rows and both local-response means. Four
distinct actual evidence alterations are rejected, including candidate arrays
and predictions; the unaltered summary is accepted first. Final evidence
qualification and the final hypothesis readout both pass. Preliminary saved
readouts remain explicitly unqualified until those checks finish.

Keep this exact function fixed for the next experiment. The next named gap is
**cross-vehicle preservation of local response under bounded recurrence**:
compare saved-weight original, tanh and supported revisions on the complete
Crazyflow and Cascade condition rosters, deriving each configuration's support
only from its existing training cache. Freeze horizons, precision, metrics,
coverage and decision rules before running. No new fit or Dart-specific
retuning is needed to test whether this behavior transfers. Existing inspected
flight evidence remains known evidence; subsequent fresh evaluation must be
identified separately. No next protocol has been frozen or run yet.

Long-horizon fitting generalization, wind/response-tail regressions, physical
derivatives, calibrated envelopes and Dart controller adequacy remain open.
Public v4 and all prior frozen fit, physical and control verdicts are unchanged.
