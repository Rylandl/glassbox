# Causal diagnosis of online fixed-wing failures

The working online fitter has **incorrect learned responses that numerical
integration can amplify**. The four selected worst-error events do not originate
in a harmful immediately preceding update: that update improves each prediction
when evaluated on the same current input. Integration refinement helps two severe
failures, leaves another largely unchanged, and worsens the fourth.

The learner remains unchanged. This investigation identifies the next fitting
problem; it does not qualify a new candidate, controller or unseen vehicle.

## Exact causal reconstruction

The [frozen protocol](harness/online-causal-trace-v1.json) selects the largest
velocity and angular-rate errors on each of the two known fixed-wing tapes,
nearby empirical median-rank controls, and adjacent model states. Selection uses
known errors and is explicitly diagnostic.

Replay starts from each authenticated v4 initial session. All **450 forecasts**,
before/after model fingerprints, full optimizer reports and final session
fingerprints match the original evaluation exactly. The **20 saved snapshots**
are captured before target assimilation. Their retained bootstrap/recent windows,
loss scales, complete state/command tails and observation cursors independently
match the causal input tape. Rejected updates cannot disguise an incorrectly
captured later session.

A separate process reproduces every diagnostic without optimizer updates. An
independent NumPy audit checks **10,440 saved integration-stage states**, exact
filter evolution, rotation updates, support compression, head contributions and
all eight fixed-input update comparisons. The source passed **187 tests**.

## What changes when integration is refined

Each row below evaluates the actual model that produced the original error.
The preceding model is evaluated on identical current history and command, so
that comparison isolates the most recent parameter update. Refined integration
uses 128 substeps instead of two, with the same dynamics, issued command and
observation-grid history/memory. These are single-transition error norms.

| Selected event | Previous model, native | Actual model, native | Actual model, refined | 16× to 64× change |
| --- | ---: | ---: | ---: | ---: |
| FW80 velocity, 8.65 s (m/s) | 1.246 | 1.088 | 1.053 | 0.000061 |
| FW80 rate, 9.35 s (rad/s) | 12.685 | 8.272 | 11.198 | 0.032117 |
| FW81 velocity, 7.90 s (m/s) | 43.516 | 31.139 | 7.646 | 0.005974 |
| FW81 rate, 3.20 s (rad/s) | 29.623 | 17.194 | 8.735 | 0.019575 |

The native columns reproduce deployed float32 inference; refinement diagnostics
use float64. Instrumented native float64 integration matches the unchanged core
within the declared 1e-10 absolute/relative tolerance, and its discrepancy from
float32 is recorded separately. The much smaller 16× to 64× changes support the
interpretation that large residual model errors remain after refinement. This
is numerical evidence at selected points, not a general convergence guarantee.

The FW80 rate failure is especially informative: refinement increases its error.
The coarse integration happens to partially cancel an incorrect learned response.
A finer solver therefore cannot be adopted as a universal accuracy correction
from these results.

[Resolution comparison figure](../artifacts/online-causal-trace-v1/checks/resolution-errors.png)

## Where the large responses arise

At FW81 origins 156 and 157 the model weights are identical, yet velocity error
jumps from 0.635 to 13.915 m/s. The change in incoming state/history/command
exposes a response already present in the model. Before any integration, learned
world vertical acceleration changes from +4.73 to −350.89 m/s² and pitch
acceleration from −29.21 to −550.51 rad/s². The observed next interval averages
at row 157 are +1.96 m/s² and +5.94 rad/s². These secants do not measure the true
instantaneous acceleration, but the refined endpoint forecasts remain inaccurate.

The exact head decomposition locates substantial contributions in the quadratic
and delayed-linear terms. At row 157, quadratic terms contribute −215.30 m/s²
in learned non-gravitational body vertical acceleration and −336.87 rad/s² in pitch; delayed-linear terms
contribute −144.16 and −237.62 respectively. The pure issued-command quadratic
part alone changes from +1.54 to −89.45 m/s² and −3.00 to −194.39 rad/s² between
these same-weight origins. Its dependence on changed commands is explicit in the
algebra; it does not require attributing the change to a new fit update.

The predicted trajectory can then create a second amplification. At row 158's
second native midpoint, one normalized gravity-direction-squared term contributes
about **+1,474 m/s²** body vertical acceleration and **+1,083 rad/s²** pitch
acceleration. The physical vertical gravity-direction component changes from
−0.9995 at the observed origin to −0.9575 on the predicted path. Its fixed startup
normalization scale is only **0.00110**, so the latter maps to a normalized
coordinate near 37 before squaring. Known unit-vector geometry has become a very
large learned feature outside the startup attitude range.

FW80's rate failure has a related quadratic feedback: learned pitch acceleration
rises from +62.61 at the origin to +247.31 rad/s² at the second midpoint, with a
large pitch-rate-squared contribution and rate/command cross terms. Its measured
interval-mean pitch acceleration is −5.72 rad/s², and the refined endpoint rate
remains wrong.

These are exact contributions from the captured model, not proof that deleting
one term is a valid fix. Several heads cancel one another on the fitting cache.
Motion-support saturation and feature novelty also occur at ordinary controls;
neither alone identifies a spike. At FW81 row 158 all issued command components
are within the recent cache's componentwise ranges, while some raw motion
components are outside. Marginal ranges and full command rank do not establish
support for their joint nonlinear response.

## Next iteration

**Constrain unsupported nonlinear response during short online fits.** The next
candidate should prevent narrow startup variation from granting large nonlinear
command, rate and attitude responses that the retained observations do not
support. Known gravity-direction geometry provides a system-independent scale;
any regularization of unsupported curvature must also be defined generically,
without a vehicle label or consumer tuning choice.

First use the captured states for matched-context command comparisons and
retained-cache identifiability checks on the dominant quadratic directions.
Then freeze one correction against working v4, retaining full velocity/rate,
orientation, tail and timing comparisons. The working predictor should remain
unchanged until that candidate is evaluated. A stronger solver-consistency
penalty or a blanket novelty cutoff is not supported by this diagnosis. The
causal snapshots make those future hypotheses testable without another full fit.

Source and artifact identities are in [the result index](online-causal-trace.json).
The prior final-model inspections remain historical evidence; they cannot
substitute for these contemporaneous model states.
