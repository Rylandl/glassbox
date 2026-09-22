# Network structure after temporal compression

The user prioritizes architectural improvements that can carry across hardware.
We now have two completed changes: stable accumulators and compact nonlinear
history. The first greatly reduced online cost; the second reduced parameter
count with useful offline accuracy tradeoffs but **did not improve CPU speed**.
Parameter count is an explanation of model size, not a proxy for complete update
cost or learning efficiency.

## Current parameter inventory

| Component | 4 commands, 10 ms samples | 3 commands, 50 ms samples |
| --- | ---: | ---: |
| Projection into 32 nonlinear units | 2,976 | 1,696 |
| Full linear acceleration head | 1,170 | 318 |
| Quadratic current-feature head | 918 | 720 |
| Nonlinear output projection | 192 | 192 |
| Accumulator input projection | 136 | 120 |
| Biases and learned time constants | 58 | 57 |
| Total | **5,450** | **3,103** |

Current features contain body velocity, angular velocity, gravity direction,
issued commands and filtered commands: `C = 9 + 2U`. The linear head retains
all preceding 100 ms lag differences and eight accumulators: `(D+1)C + 8`
features. Only the nonlinear head uses `(min(4,D)+1)C + 8` features. Histories
of four samples or fewer retain their coordinates; longer histories use fixed
orthonormal constant-through-cubic temporal summaries. There is one formulation,
with dimensions determined by input count and timing.

The learned acceleration passes through shared gravity, rotations and integration
to predict motion. Arbitrary linear lag response is retained. Nonlinear access
to temporal components outside the four-dimensional basis is the explicit
capacity cost. Finite-window polynomial projection is not a learned recurrence
or a claim that generic history can always be represented in four coordinates.

## What the experiments actually establish

The [accumulator migration](accumulator-migration.md) removed nonlinear recurrence
from observed-history processing. It removed 14.0% of quad parameters and reduced
median complete updates by 47.6% on its paired CPU benchmark. Its 4.11% online
accuracy gain belongs to that pre-projection source.

The [projection reuse](history-projection-reuse.md) then centered and reused the
fixed history projection across integration stages. Whole-update snapshot medians
fell 19.17% for quads and 4.85% for fixed wings. It retains the function class,
but later full-stream evidence shows 2.69% higher aggregate error than the
pre-projection accumulator, largely one fixed-wing case. Snapshot numerical
closeness did not imply identical long online learning trajectories.

The [temporal comparison](nonlinear-temporal.md) removes 37.46% more quad
parameters while preserving aggregate online accuracy against the current full
model. Crazyflow command responses improve 4.31%, its forecasts worsen 4.20%,
and Dart 250 ms forecasts improve 14.99%. Fixed-wing two-lag predictions and
fitted arrays match exactly. Updates are 6.10% slower: roughly 31.9 → 33.9 ms.
Separate projections and their derivative work cost time despite fewer weights;
we have not isolated their share with a new component profile. The complete
measurement, rather than a guessed bottleneck, determines the speed claim.

Both online models use one proposal and 16 CG iterations per observation.
The compact model's quad error after 100 updates is 2.80% higher. Offline
work-to-accuracy is mixed too. We adopted the compact architecture for size and
physical response/Dart accuracy, not faster fitting or a universal improvement.
Initialization preserved original draw scales, physical coefficients and initial
predictions; compact normalizers have explicit compensation. The result does
not repeat the earlier initialization-strength error.

## Completed: fast readout fitting without pretraining

The [small cold-start screen](cold-readout.md) retained every output path, using
381 features and 2,286 readout weights for four commands  / 10 ms. Both arms used
identical fresh episode-prefix initialization. No pretrained dynamics, learned
class prior or reused normalization entered either arm.

Float64 RLS over measured midpoint increments achieved complete updates of
0.36–0.41 ms on quads and 0.264–0.266 ms on fixed wings, approximately 91× / 29× faster
than the current full update. Quad one-step error improved 80.4%, but fixed-wing
one-step error nearly doubled, and 250 ms forecasts deteriorated severely on
both families. The candidate was rejected without broad fitting or control trials.

The raw estimator removed recursive loss and its curvature/acceptance safeguards
as well as feature learning. Its local regression objective improves while
recursive forecasts fail, so this is not evidence that pretrained features are
required. Fixed generic features remain a hypothesis; their adequacy has not
been isolated. The first scoring time is 1.25 s after release for these quad tapes,
not immediate cold-start recovery. All initialization data/time remain counted.

## Next: isolate the weights that must adapt

Keep the current recursive forecast objective, reconditioning, curvature prior
and acceptance logic; restrict optimization to the acceleration readout after
fresh episode initialization. Use the same short causal screen and 16-PCG budget.
This changes one factor and tests whether representation learning is necessary
before another estimator is designed. Preserve physical feature functions under
normalizer compensation; do not confuse coefficient rescaling with learning new
features. Freeze the actual protocol before implementation.

All adaptation must use the current episode, including nonlinear representation
learning if required. Fleet training is not a permitted rescue. A promising
trainable-subset result could then motivate faster trajectory-aware fitting;
a clear loss should stop that candidate promptly.

## Deferred: conditioning at the fixed update budget

Earlier 64-PCG experiments reduced fixed-wing angular error by roughly 56–58%
relative to 16 iterations for both architectures. Full-stream sensitivity to
rounding also suggests an optimizer issue. The current preconditioner captures
damping and prior diagonal, not forecast-Jacobian correlations. This remains
useful evidence and a fallback if the no-pretraining linear estimator does not
earn its accuracy/runtime cost; it is no longer the named next experiment.

## Secondary opportunities

The observed command filter is a linear exponential recurrence. A stable parallel
prefix or convolution could remove sequential history work while retaining its
initial condition and gradients through learned time constants. This is a useful
execution optimization, secondary to the user's current architectural priority.

Reproducing the old refined offline fit, improving long-horizon accuracy,
independent calibration and live recovery are separate gaps. Keep one iteration
focused and retain losses alongside gains; neither a small regression nor a
smaller parameter count decides adoption by itself.
