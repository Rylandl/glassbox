# Public v4 numerical-port design — inspection only

Prepared 2026-09-19 from the expanded-training-cache worktree. This is a prospective
design, not a frozen protocol or qualification result. No fitter, initializer,
forecast, gradient, simulator or runtime experiment was executed for this note.

## Maintained core boundary

Keep the public `fit(recordings)`, `LearnedDynamics.predict(...)`, immutable
`update(recordings)`, save/load and evidence contract. Route these to one maintained
quadratic sequence implementation. Do not ship the research module-import/rebinding
chain as the public implementation, or expose model/optimizer/precision choices.

Exact numerical source ledger:

| Piece | Existing source and extraction |
| --- | --- |
| Observation windows and history | `recordings.py`; `_sequence_model.py:SequenceBatch`, `_features`, `_filter`, shape checks. Memory starts at zero at the first consumed context; explicit-delay differences and memory order stay unchanged. |
| Quadratic mean | `state_input_model.py:interaction_features` and `state_quadratic_model.py:autonomous_features,_rollout`. State-major/input-minor `x*u`, upper-triangular `x*x` including squares once. Preserve floating operation order and scan structure. |
| Initialization | `_sequence_model.py:initialize_sequence_model`, then `affine_anchored_model.py:initialize_candidate`. One affine precursor plus one joint solve; same draws and target subtraction. |
| Fixed objective | `initial_channel_balance_model.py:initial_training_forecast,weighting_metadata`, plus the weight construction in `full_cache_gradient_model.py:fit_candidate_sequence`. |
| Full-cache optimizer/selector | `full_cache_gradient_model.py:fit_candidate_sequence`. Reuse the exact update, checks and strict selector; remove experimental class/global wiring, not numerical steps. |
| Acceptance helpers | `safeguarded_adam_model.py:trial_parameters,make_training_objective,safeguard_metadata`; current full-cache work-accounting helper. |
| Public lifecycle | Existing `learner.py` extraction, automatic holdout, calibration, recording ledger, update-cache merge and `_learner_arrays.py`. Change recipe/archive identities explicitly and require the new parameter/norm roster. |

For state dimension d, input dimension m, explicit delay p, memory 8 and width 32:
base feature count f=(p+1)(d+m)+8, interaction count d*m and autonomous count
d*(d+1)/2. There are nine parameter arrays and eight norm arrays. No 15/4/3
dimension checks, platform channel groups, quaternion projection or simulator state
belong in this numerical implementation.

Initialization must preserve the following sequence, without an extra preparation
solve: derive six base norms, solve the existing affine ridge, make the existing
seed-0 w1/memory draws, derive two product norms, construct the same joint D/Y,
then solve `(D.T@D + penalty) W = D.T@Y + penalty@W_affine`. All non-bias rows
have penalty lambda; bias remains unpenalized. W_affine has the original affine
linear/bias rows and zero product rows. Keep lambda=.01*N*H and the current
subtraction/normalization order. Do not replace this with sequential residual fitting.

Use the unchanged hold scale `max(sqrt(mean((hold-target)^2,axis=0)),.01*state_scale)`.
One recursive training forecast from actual initial parameters gives
`e0=mean(((pred0-target)/hold_scale)^2,axis=(0,1))`; raw weights are
`1/max(e0,.01**2)` and fixed weights are raw/mean(raw). Preserve NumPy float64
reduction and the original eager/scan initial-forecast path. Weighted recursive
loss trains and selects; original loss remains diagnostic. Weights never derive
from development/test data or change during optimization.

Every one of 1,000 proposals sees `arange(N,dtype=int64)` exactly once in cache
order. Globally clip the full gradient once at norm 5, then preserve Adam
(.9/.999, epsilon 1e-8, learning rate .002). Test scales 1 through 1/128 and accept
the first finite strict decrease in the canonical full-training objective. Scale
1 returns proposal bytes; others use `old + alpha*(proposal-old)`. Advance finite
proposal moments even if all trials reject. Nonfinite gradient/norm/moments/proposal
fails; nonfinite trial loss backtracks. Development selection is the first strict
minimum over 0,100,...,1000, not the best training loss. Keep the initial evaluator,
at most 8,001 acceptance calls, separate returned-gradient loss, and truthful work
counts. Preserve the 7,200-second fitter clock and failure semantics; wrapper
preparation/calibration timing stays separate.

## Public data size and sampling interval

Do not port research `prepare` admission literally: its exact 1,536 windows,
72/24 roles, old-384 prefix, reference object and source-seal arguments belong to
that experiment. For v4, proposed 1,536/256 values are upper bounds under the
existing generic automatic split and SHA-priority round-robin extraction. Require
at least two recording identities and at least three complete windows per role;
use actual N for gradient coverage, ridge and report counts. Do not duplicate or
pad windows to satisfy the cap. A short admitted corpus must not inherit a fixed
1,536-sized gather. Record represented parents honestly when a cap limits them.

Retain `steps_for`: delay=max(1,rint(.1/dt)), history=max(rint(.5/dt),delay+1),
horizon=max(1,rint(.25/dt)), with positive finite dt and same dt across a collection.
The effective durations are step counts times dt, including coarse grids and ties
in NumPy rounding. Public prediction consumes aligned C+1 observations/C commands,
where C is the fitted consumed-history length. P is only the explicit-delay span
used within that context; it does not shorten the required public history.
Prediction returns only H future observations, with requested H within the fitted maximum.
No interpolation or resampling is implied. Constant channels retain current scale
fallbacks (standard deviations <=1e-8 become 1; delta floor 1e-4). Small N is
algebraically supported by positive ridge, but extreme conditioning/finiteness
and arbitrary large dimensions remain qualification limits, not proven guarantees.

## One precision/runtime policy

Amended after root review: deterministic float64 fitting and ordinary ambient-JAX
inference, with both default32 and x64 inference mandatory in qualification.
This supersedes the initial suggestion to require a caller-owned x64 scope for
transformed prediction; that would narrow the intended public integration.

Initially qualify on CPU with
Python 3.12, JAX/jaxlib 0.11.1, NumPy 2.5.3 and SciPy 1.18.1 (installed versions
read from package metadata). Current project requirements are Python >=3.11,<3.14,
JAX >=0.10, NumPy >=1.26 and SciPy >=1.12: those ranges are not evidence that
all admitted versions work. Raise the JAX minimum to the inspected public-context
version and pin the complete actual runtime in qualification; do not claim
cross-version bitwise equality or accelerator qualification from CPU evidence.

Use `jax.enable_x64(True)` only around the owned fit/update computation, never
global `jax.config.update`, environment writes, or the removed experimental x64
context. The scope must cover JAX conversion, initialization, initial training
forecast/weight reductions, all loss compilation/execution, AD/Adam/acceptance,
development selection and envelope calibration. Synchronous host reductions and
NumPy result conversion finish before scope exit. No fitter arithmetic changes
are needed; the existing initial-prediction float64 check remains meaningful.

Store and load NumPy float64 parameters, norms and evidence, rather than retaining
fit-created JAX device arrays as inference closure constants. `predict` does not
enter an x64 context or force a dtype: current `jnp.asarray` conversion and ordinary
JAX arithmetic follow the ambient configuration. Under defaultFalse, parameters
and normalizers join the float32 computation; under True they remain float64.
This has no Glassbox consumer knob or environment prerequisite. Ordinary eager,
outer `jax.jit`, JVP, grad, jit-of-grad and grad-of-jit must all qualify under False
and True; transform failures are blocking integration findings, not optional paths.

Compare default32 numerics to an x64 oracle supplied the same values already
quantized at the caller's float32 boundary. This separates input quantization from
arithmetic/parameter-normalization error; it does not hide deployment error against
physical truth. Predeclare fixed scale-normalized forecast and derivative bounds,
with absolute allowances near zero derivatives, and independently evaluate default32
physical errors/response/coverage on the fresh cohort. Envelopes were calibrated
in fit64; their coverage in default32 must be measured. Never silently recalibrate,
relax thresholds, switch precision or clamp outputs after seeing a failure.

The installed context implementation uses `swap_local`/`set_local` with restoration
on exit; x64 participates in trace and JIT keys. Public context documentation is at
https://docs.jax.dev/en/latest/_autosummary/jax.enable_x64.html . JAX's dtype guide
warns about contextual compilation/execution mismatches and treats x64 as a program
configuration: https://docs.jax.dev/en/latest/default_dtypes.html . Keeping inference
ordinary JAX avoids introducing that mismatch inside caller transformations.

## Prospective qualification and independent oracle

Before edits, freeze the hashes below plus the enclosing verified research seal.
Historical experiments import `_sequence_model` and `learner.RECIPE`; modifying
those in place can silently change the purported research oracle. Run oracle
comparison in its source-pinned checkout/process. Never import the historical
oracle against newly rewritten core globals. Public installed-package checks must
work without importing any experimental module or controller code.

Freeze tests for (1) default-x64-false fresh-process import/load/fit/update/predict
without config leakage, including error paths and an independent thread;
(2) mandatory ambientFalse and ambientTrue single/batched/eager/JIT/vmap predictions,
JVP and scalar value-and-grad, including jit-of-grad and grad-of-jit, before/after
load, with float32 quantization reflected in the reference inputs;
(3) short and asymmetric dimensions, constant channels, minimum role windows,
coarse/non-dividing dt and all shape/history/horizon rejections; (4) exact saved
parameter/norm/cache round trips, expected archive-version refusals and immutable
revision behavior; (5) research/public preparation, initialization, weights,
proposal/acceptance/selection and final prediction parity in the same pinned CPU
runtime. For each of the two fresh public flight fits, require byte-identical
ordered training/development cache arrays and sidecars, all actual initial
parameter/norm arrays, hold scales, fixed channel weights, and all selected
parameter/norm arrays against the authenticated expanded research revision and
its saved initialization evidence. Array key rosters, dtypes and shapes must also
match, and the selected development-checkpoint step must be exactly equal.
The ridge scalar and window identities/roles must agree exactly. Public wrapper
format, recipe/version labels, archive metadata, reports, semantic fingerprints
and timing may differ where they describe the new implementation; their equality
is not a substitute for the scientific array/step checks. Do not relax those
checks to floating tolerances. Prediction, AD and independently reduced score
comparisons use their separately frozen tolerances. Never infer equality from
algebra or hash a new source as its own reference. Preserve actual failure
prefixes, not a partially qualified model.

Run the unchanged historical `tests/test_harness.py` with its matching test helpers
under the authenticated old checkout/process. Its NumPy recurrence and loader
describe the old public mean and must not be rewritten or exercised against the
new quadratic core as evidence of v4 parity. Add new public qualification coverage
for the full quadratic mean against the source-pinned oracle, fixed-reference
synthetic scoring and complete rosters, independent physical reductions, and the
new promotion conjunction. Passing historical tests alone cannot satisfy that
new coverage or qualify a new public mean.

Public update can retain its existing immutable cache/refit semantics, but its
accuracy/budget and live adoption are separate unqualified claims. Likewise
finite-difference agreement tests calculus of this model, not physical response
fidelity. A 1,536-window public recipe's synthetic capability, timeout frequency,
small-corpus behavior and consumer portability remain unknown until this protocol
runs. In particular, float32 parameter/norm conversion, cancellation for large
coordinate offsets with small variations, recursive quadratic amplification and
derivative conditioning are real unknowns; finite-difference checks alone do not
bound them. No new numerical trials were used to choose this design.

## Inspected source SHA256

```text
960721cdec0ff83b06f488a22005155481fc4b0972ce81b43a25b9560c60f9b8 src/glassbox/learner.py
a6506d79b3947031db40293e411f51750d7e9d36cf5156e13ee81ba6f8d77e4b src/glassbox/recordings.py
b6492f196c72e0629d1a2c7da446fa4233a74e0dd7a6fd486bfc46414bfe7d9c src/glassbox/_sequence_model.py
c7e33df1421b50b49ea24c65230360a231a9b72772536118a876e986a8e0d4c9 src/glassbox/_learner_arrays.py
9ee1356e37869fed8a7a647a9253543e3e06416ada5fc873f6d81f8c529f42e3 src/glassbox/experimental/state_input_model.py
661779ff16cbc2a92977e460566fffcc590ab49f3ca2847cac4899a9bf4bd969 src/glassbox/experimental/state_quadratic_model.py
fa83abb70f9b906b66f46aa6933a28608341b0c94b33d0da6d7a10615328d714 src/glassbox/experimental/affine_anchored_model.py
9d7362e62746df2c600edfdd24b58c30953aa7c2d92689899eab9d27d2c9389e src/glassbox/experimental/initial_channel_balance_model.py
72d6e0b83554db8a27615bedbedad72f94c5f93b39b318d7abd97112cde6a84a src/glassbox/experimental/safeguarded_adam_model.py
b63e50a7eb996bf31a4e3ddf8ace851dd1c843a5064b0a4fb2c49bc29eecaece src/glassbox/experimental/full_cache_gradient_model.py
a433d9f6b1319acad4ccefd63a844836c6ea1c04da52873996e0c0c84d60f508 src/glassbox/experimental/expanded_training_cache_model.py
ff24ae0e6cd35eecd3ac69e2a9b26117479107a811f2f2de61c9aee1e1e88da0 pyproject.toml
```
