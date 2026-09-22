# Why the accumulator lost fixed-wing angular accuracy

The controlled diagnosis points to initialization and limited online solver
convergence, rather than evidence of an unavoidable loss of memory capacity.
Changing only the initial memory projection scale reverses the observed aggregate
angular regression. Giving both architectures more PCG iterations nearly removes
their angular-error gap and greatly improves both. These are causal interventions
on two already inspected recordings, not broad generalization qualification.

## What changed and where the error appears

All shared initial acceleration-head parameters and normalization arrays are
exactly equal. Both architectures make the same first prediction. The regression
develops during learning: pitch contributes 99.65% and 97.42% of excess angular
squared error on the two recordings. Gains and losses alternate across time;
several later overshoots dominate the aggregate. This is not a uniform initial
prediction bias or a rejected/nonfinite-update problem.

The accumulator implementation also changed random projection normalization from
`1 / sqrt(full_feature_count)` to `1 / sqrt(current_feature_count)`. For these
recordings, that changes the denominator from sqrt(53) to sqrt(15), making the
same current-feature projection **1.880 times stronger**. Its direction and random
seed are unchanged. A stronger projection pushes more tanh drive values toward
saturation. On the measured initialization prefix, the fraction with absolute
value above 0.95 rises from 26.0% to 43.5% on fixedwing-80, and from 6.25% to 43.0%
on fixedwing-81. These statistics support a conditioning explanation; they do not
prove that saturation alone mediates every subsequent error.

The initial predictor ignores memory because its output coefficients are zero.
The changed latent features nevertheless affect the first learned parameter step
and all subsequent optimization. Equal initial predictions therefore do not imply
equal learning behavior.

## Controlled results

The [frozen protocol](harness/accumulator-angular-diagnosis-v1.json) contains seven
arms. Each loads the original authenticated initial session and makes all 225
causal predictions per recording before assimilation. Initialization is not rerun;
observations, command order, cache, loss, prior, trust bound and backtracking are
unchanged. The two unmodified arms reproduce every saved prediction, post-update
model fingerprint and report exactly. All **3,150 updates** complete.

| Intervention | Angular RMSE, fixedwing-80 (rad/s) | Angular RMSE, fixedwing-81 (rad/s) | Equal-case angular change vs v8 | Equal-case velocity change vs v8 |
| --- | ---: | ---: | ---: | ---: |
| V8, 16 PCG iterations | 0.6040 | 1.6931 | reference | reference |
| Original accumulator, 16 iterations | 0.6401 | 1.7808 | +5.58% | −11.56% |
| V8: mask hidden feedback into memory | 0.6427 | 1.7712 | +5.51% | −5.69% |
| V8: also mask lag inputs into memory | 0.5980 | 1.5698 | −4.19% | −5.58% |
| Accumulator: restore initial current-projection scale | 0.6021 | 1.4603 | **−7.27%** | −8.42% |
| V8, 64 PCG iterations | 0.3351 | 0.5910 | **−55.99%** | −69.23% |
| Original accumulator, 64 PCG iterations | 0.3494 | 0.5683 | **−55.94%** | −69.49% |

Ratios use equal-case geometric means; lower is better. Removing lag inputs in
the diagnostic affects the memory projection only: the acceleration head retains
all lag inputs. Masks also zero derivatives of the removed memory coefficients.
This decomposition is path-dependent; its changes must not be added as independent
main effects. In particular, feedback removal alone hurts while the simpler
current-only drive improves, so these data do not establish that recurrent
feedback is necessary for these recordings.

The scale intervention preserves the accumulator architecture, parameters count,
learned time constants and 16-iteration budget. It changes only the initial memory
matrix to the exact current-feature rows of the v8 matrix. That alone improves
angular error by 12.17% relative to the original accumulator and makes it better
than v8 on both cases. This is direct evidence that the original regression is
not sufficient evidence of an accumulator capacity limit.

The 64-iteration interventions change only the PCG loop bound. Angular error falls
55.99% for v8 and 58.27% for the accumulator relative to each one's 16-iteration
run. At 64 iterations their equal-case angular results differ by only **0.12%**.
Their errors on already assimilated cache windows also fall substantially: mean
recent-cache rate RMSE changes 0.0736 → 0.0125 and 0.2636 → 0.0470 rad/s for v8,
and 0.0810 → 0.0129 and 0.3244 → 0.0517 for the accumulator. Better cached fit and
better predictions of the next observation support incomplete optimization as a
major limitation of the present fitting budget.

## Implications and limits

Keep the accumulator as the leading candidate. Restore or otherwise control its
initial drive scale before the next full six-stream comparison. Improving solver
conditioning is now a concrete performance target: approach the quality of the
64-iteration solve with less work. Simply increasing the budget is a useful
accuracy reference, not a qualified runtime improvement. The original paired
speed measurement applies to 16 iterations only.

The matched-scale accumulator has not run the quad screen, offline forecast and
command-response checks, or Dart. Combining matched initialization with 64
iterations was not tested. One seed and two known fixed-wing recordings cannot
establish an optimal initialization, a universal solver budget or general system
readiness. V8 remains the maintained implementation pending broader qualification;
no production package files changed.

The independent saved-data verifier recomputes results without importing JAX or
calling a model. Four diagnostic tests pass, including masked derivatives,
residual attribution, progress replacement and independent feature calculation.
The runner stopped once after completing the unchanged arms because its progress
writer used exclusive creation. Coordinator-only correction `f74449f` resumed
both sealed arms without repeating them; the failure record is retained. The
64-iteration arms retain the library's unmodified report schema/counters; their
actual PCG budget is recorded in `arm.json` and the frozen protocol, not inferred
from the production counter that still increments by 16.

The [result index](accumulator-angular-diagnosis.json) records all cases and
source/manifest authorities. Scientific harness `353c44e`, correction `f74449f`
and saved-data explanation `1deac89` live on
`codex/accumulator-angular-diagnosis`. Paired diagnostic authority:
`ff793a65f2ed83c143858ac8e706e16a61dd2ef2cafaeaa5f5a4a77dcc82f43d`.
Independent explanation authority:
`1970d56653bc7925b25f381ddf47743acf1b328d88968fba44f17c676b426965`.

```sh
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python scripts/diagnose_accumulator.py verify \
  --output /Users/ryland/autonomy/glassbox/artifacts/accumulator-angular-diagnosis-v1/evaluation \
  --authority ff793a65f2ed83c143858ac8e706e16a61dd2ef2cafaeaa5f5a4a77dcc82f43d
```
