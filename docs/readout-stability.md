# Fast readout: diagnose the angular recurrence

The frozen-feature linear readout has demonstrated its runtime and local-accuracy
value. This investigation supports building on it. The failure is compounding
angular forecast error, with a substantial contribution from **linear delayed
feedback**. It is not explained by unconstrained quadratic terms alone. The next
experiment should control first-order feedback sensitivity while retaining the
direct solve, before adding a retrospective trajectory acceptance gate.

No candidate was fitted in this iteration, no production model changed, and no
pretraining or learned vehicle prior was introduced. The [artifact index](readout-stability.json)
records all attempts, source revisions, protocols and verified payloads.

## What was measured

The [frozen diagnostic protocol](harness/readout-stability-diagnosis-v1.json)
uses all 14 saved conditional origins from each of the two fixed-wing recordings,
with five 50 ms steps per origin. No extreme or initial forecasts are excluded.
Parameters stay fixed at the origin; commands are the recorded issued commands.
These are known development recordings, not independent generalization evidence.
RMSE below is the root mean squared vector norm across those 14 origins. The
50 ms entries therefore differ from one-step scores over every stream update.

The candidate's readout means were saved at every update. Intermediate full-learner
models were missing, so the unchanged reference was replayed from its original
episode prefix. Every original one-step prediction, conditional prediction and
final parameter array reproduced exactly. The recovered models are now saved.
This is explicit reference fitting replay, not a claim that every requested
diagnostic could initially be computed from saved weights alone.

## Error compounds with horizon

Body-rate RMSE, rad/s:

| Recording / model | 50 ms | 100 ms | 150 ms | 200 ms | 250 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 80 / full learner | 0.721 | 1.032 | 0.936 | 1.542 | 2.702 |
| 80 / fast readout | 0.166 | 0.531 | 1.338 | 4.194 | **15.065** |
| 81 / full learner | 2.108 | 3.301 | 1.729 | 4.084 | 14.522 |
| 81 / fast readout | 0.277 | 1.009 | 3.157 | 11.580 | **39.984** |

This is serious recursive failure despite much better local predictions. The
full learner's 14.5 rad/s error in recording 81 is itself poor; matching that
reference would not establish adequate absolute accuracy. Finite five-step errors
show trajectories separating rapidly from truth, not mathematical unboundedness.

![Velocity and angular forecast error at all five horizons](/Users/ryland/autonomy/glassbox/artifacts/readout-stability-diagnosis-v1/horizon-errors.png)

## Output swaps and block ablations

250 ms vector RMSE; velocity in m/s, body rate in rad/s:

| Forecast variant | 80 velocity | 80 rate | 81 velocity | 81 rate |
| --- | ---: | ---: | ---: | ---: |
| Full learner | 2.088 | 2.702 | 4.157 | 14.522 |
| Fast readout | **1.328** | 15.065 | 7.669 | 39.984 |
| Candidate force + full-learner angular acceleration | 1.557 | **2.765** | 6.919 | **18.241** |
| Full-learner force + candidate angular acceleration | 9.521 | 18.115 | 63.373 | 44.394 |
| Candidate, quadratic rate coefficients zeroed | 1.372 | 19.714 | 7.596 | 35.387 |
| Candidate, nonlinear output rate coefficients zeroed | 1.543 | 21.054 | 5.908 | 31.382 |
| Candidate, delayed linear rate coefficients zeroed | 2.892 | **6.936** | **3.639** | **14.762** |

Swaps evaluate physical outputs with each model's own features, normalizers,
command filter and memory, then integrate a common physical state. Coefficients
are not copied between different learned feature maps. The strong benefit from
replacing angular acceleration localizes the main problem to the angular branch;
its coupling into orientation and force also matters.

Removing quadratic rate coefficients **worsens recording 80** and only modestly
helps 81. Removing delayed linear rate terms reduces rate error **54.0% / 63.1%**.
The latter removes only linear lag-difference rows, retaining current features,
memory, quadratic terms and the nonlinear head. Its local rate accuracy worsens
to 0.449 / 0.466 rad/s and recording 80's velocity forecast worsens substantially.
These interventions diagnose the existing fit; they are not refitted candidates
or an argument to delete delayed dynamics.

## Feedback is already high on measured states

The recurrence derivative includes 50 tangent coordinates: world velocity, body
rate and a three-dimensional rotation tangent; filtered commands; lag features;
and latent accumulators. The saved evidence contains full Jacobians, their 9×9
physical partial blocks, singular values and ordered five-step derivative products,
along both measured and free-running trajectories. Reconstructed history is used
at each measured state; free forecasts propagate their own history.

| Median full recurrence spectral radius | 80 measured | 80 rollout | 81 measured | 81 rollout |
| --- | ---: | ---: | ---: | ---: |
| Full learner | 1.271 | 1.275 | 1.421 | 1.463 |
| Fast readout | **3.411** | **3.399** | **3.342** | **3.296** |

Large candidate gain is present before the forecast leaves the measured
trajectory. Eigenvalues of a local map away from equilibrium are diagnostic,
not a nonlinear stability certificate. Ordered Jacobian products matter too;
their singular values depend on the chosen units and state scaling. Physical
partial blocks alone omit the delayed and hidden recurrence.

A [declared posthoc follow-up](harness/readout-stability-feedback-v1.json)
differentiates angular acceleration with respect to current body rate, holding
other state and latent variables fixed, at all 70 measured states per recording.
The median largest real eigenvalue is **+13.04/s / +13.59/s** for the candidate,
versus approximately **+0.011/s / +0.000004/s** for the full learner. All 70 samples
in each candidate case have a positive maximum real part. Zeroing linear lag rate
rows reduces the medians to **+7.49/s / +8.77/s**; removing quadratic rate rows
leaves **+13.13/s / +13.56/s**.

The candidate's most negative real parts are only -12.53/s / -2.99/s. This does
not support stiff negative rate feedback exceeding the explicit-midpoint
negative-real stability interval as the main candidate mechanism. The implemented
50 ms interval already uses **two 25 ms midpoint substeps**. For an isolated scalar
rate derivative a, that substep multiplies perturbations by 1+ha+(ha)²/2; the
negative-real stability interval ends at a=-80/s for h=0.025 s. The full coupled
model need not behave like this isolated block, and the reference itself contains
some much stiffer derivatives. These measurements do not qualify the integrator
for every learned field.

The linear history representation is [current, past−current, memory]. Therefore
its effective current coefficient includes **current coefficient minus the sum
of lag coefficients**. Increment fitting can leave large current/history gains
that cancel on observed motion. The ablations and derivatives support this as a
working mechanism; they do not uniquely identify every offending coefficient.
The current quadratic curvature penalty cannot penalize a purely linear gain.

## Leverage and support

Unregularized leverage uses only assimilated feature rows and the original ridge:
phiᵀG⁻¹phi with G=lambda I+PhiᵀPhi. A QR precision root avoids explicit inversion.
The pack also saves leverage under the actual curvature-regularized precision.

| Candidate leverage diagnostic | 80 | 81 |
| --- | ---: | ---: |
| Median rollout / measured leverage | 1.240 | 1.413 |
| Maximum rollout / measured leverage | 392,141 | 113,364,042 |
| Spearman association: log rollout leverage vs rate-error magnitude | 0.521 | 0.650 |

Large excursions and positive association support extrapolation as an amplifier
of the failure. The scores share horizons and repeated origins; they are descriptive
associations, not independent statistical tests or proof that leverage predicts
failure early. Neither raw nor regularized leverage establishes calibrated
epistemic uncertainty without a noise and model-mismatch evaluation.

Motion inputs are **already smoothly bounded before quadratic products**. They
are unchanged inside prefix support and saturate outside it. This prevents the
particular unrestricted motion polynomial assumed in the initial hypothesis,
but does not control local feedback gain or guarantee useful recursive forecasts.
Issued commands are not clipped by that motion transform. Tighter bounds might
help some queries, but bounded quadratic inputs alone are not the missing feature.

## Evaluation policy and next experiment

Future online architecture comparisons must report per-family 250 ms velocity
and rate ratios separately, with frozen flags in addition to the aggregate.
The diagnostic protocol declares 1.5× per-family component and primary limits,
1.25× aggregate 250 ms, 1.15× aggregate one-step, finite predictions, and a 0.5×
warm-update runtime target. These inform the engineering decision rather than
make every isolated loss a veto. Historical acceptance outcomes are unchanged.

Applied descriptively to the original saved full-roster scores, the candidate's
250 ms velocity/rate ratios are **0.744 / 0.607 for quads**, and **1.083 / 3.918 for
fixed wings**. The proposed fixed-wing rate and primary flags fail, while the
family velocity flag passes. Recording 81's velocity ratio is 1.845 even though
the family aggregate is much smaller; retain individual physical scores too.

The matched quad 10/50 ms probe remains open. It should record one physical
trajectory with issued commands held for each 50 ms interval, then evaluate the
two observation schedules. Simply decimating a tape with changing 10 ms inputs
changes the modeled command history and does not cleanly isolate sample interval.
No claim of a purely fixed-wing structural effect is justified yet.

**Next: a soft first-derivative penalty on the readout's effective instantaneous
and delayed motion feedback.** With frozen features, feature derivatives D are
fixed and an output-Jacobian norm remains quadratic in the readout. A normalized
penalty adds a term proportional to ΣDDᵀ to the feature precision, preserving the
direct solve. It sees the linear current/lag coupling that the present curvature
penalty misses. Freeze generic physical-coordinate scales, probe domain, strength,
data budget and complete-update timing before fitting; test all six acceleration
outputs without a vehicle-family branch. Keep curvature and all other estimator
choices fixed so the comparison identifies this one change.

This is a testable next hypothesis, not a stability guarantee or a promised
sub-millisecond implementation. Soft sensitivity regularization can also suppress
legitimate dynamics; assess its local accuracy, recursive error and derivative
behavior together. Global strict contraction would exclude neutral rigid-body
modes and genuinely unstable plants, while constraining a small partial block
does not control the full delayed recurrence. Local linearization and global
stability claims must remain distinct; see the [MIT Lyapunov notes](https://underactuated.mit.edu/lyapunov.html).
A completed-trajectory acceptance guard remains a possible safety net, not the
primary next architecture change.

The existing n-scaled curvature penalty is constant-strength regularization
relative to mean data error, **not a decaying prior**. With fixed strength it can
bias the asymptotic minimizer; consistency is not established. Nothing in this
plan introduces a fleet-trained prior: all learned values still come from the
current episode, with prefix acquisition and fitting counted.

## Verification and attempt accounting

Eight distinct focused tests pass across recorded runs: output-swap mechanics,
known damping and augmented finite differences, leverage against an independent
solve, ablation isolation, feature reconstruction, snapshot reuse, and continuous
rate derivatives. An initial test failed because its fixture attempted to mutate
a read-only array; copying the fixture arrays fixed it. Eight actual-snapshot
finite-difference directions agree with augmented Jacobians to relative error
at most 1.26e-9. All saved scores and payload hashes verify without fitting.

The first diagnostic stopped after recovering both reference streams because one
float64 diagnostic forecast differed from its saved float32 counterpart by
0.01094%, just beyond a 0.01% diagnostic tolerance. Native float32 replay remained
exact and stepwise float64 forecasts matched the native float64 implementation.
The follow-up records both absolute and relative precision differences and uses
a 0.1% cross-precision tolerance. Maximum absolute canonical-state difference is
0.004409; this is numerical sensitivity, not a changed candidate or acceptance flag.

The second attempt intended to reuse snapshots, but a missing forwarded argument
caused **450 redundant reference updates**. Its reuse/count metadata is therefore
incorrect and is explicitly superseded by this report and index. After fixing and
testing forwarding, the verified run uses only saved snapshots; every one of its
44 diagnostic arrays exactly matches the second attempt. Both replay attempts
matched the original reference predictions and final arrays exactly.

Actual total: **900 unchanged reference updates, versus 450 planned; zero candidate
fits**. The corrected diagnostic and feedback follow-up add zero reference updates.
All partial and superseded attempts remain sealed. These are harness corrections,
not candidate tuning. Experimental implementation is preserved in Git through
`da3123c` and removed from the maintained tree after reporting; production source
and public interfaces are unchanged.
