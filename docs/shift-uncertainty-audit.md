# Uncertainty when the horizon shifts

This diagnostic preserves controller behavior. A shifted mean trajectory does not
inherit the old uncertainty margins. The current planner performs a fresh
prediction from fixed physical and actuator states, using unchanged parameter
information and a forecast-error second moment indexed by the new lead time.
That is a coherent *fixed-belief, known-initial-state planning convention*. It is
not the Bayesian conditioning of the previous joint prediction on new telemetry.
No localized controller bug is demonstrated by the 1.000791 shifted-prefix result.

## Derivation and assumptions

Let `z = (x, a)` include physical state and actuator state, and let theta be the
constant structured parameter vector. Locally, in consistent tangent coordinates,

```
dz[j+1] = A[j] dz[j] + B[j] dtheta
S[0] = 0
S[j+1] = A[j] S[j] + B[j]
P_parameter[j] = S[j] C S[j]^T
P_reported[j] = P_parameter[j] + E(j dt)
```

`C` is the covariance on the resolved information subspace; unresolved directions
are separately declared, not known exactly. `E` is the held-out *uncentered*
endpoint error second moment. The equations are a local sensitivity approximation,
not an established probability or safety bound. `fitted.py` computes `S L` by JVP,
with `C = L L^T`, while holding its supplied initial physical/actuator arrays fixed.

For the old endpoint `j+1`, let `Phi[j]` be the derivative of the remaining rollout
with respect to its initial `z[1]`, and `T[j]` its derivative with respect to theta
holding that initial state fixed. The chain rule gives

```
S_old[j+1] = Phi[j] S_old[1] + T[j]
P_old = (Phi S1 + T) C (Phi S1 + T)^T
P_reset = T C T^T
```

There is no ordering between these matrices. In particular, negative cross terms
can make the old variance smaller. The numerical audit carries the first-step
physical and actuator sensitivities separately and checks this identity against
the old covariance at matching endpoints. It also verifies identical nominal
states. This is the falsifiable probe for cancellation rather than mean mismatch.

The exact scalar counterexample is `x1 = theta; x2 = -x1 + theta`, `Var(theta)=1`.
The old `Var(x2)=0`; treating `x1=0` as fixed while retaining the old parameter
variance yields `Var(x2)=1`. This is the planner's reset operation. It is not an
error in differentiating the new forecast, but it is not observational
conditioning. For noiseless observation of `x1`, theta is known and both posterior
variances are zero. With observation noise variance one, the posterior variance of
theta is one half and the joint future variance remains zero.

For a joint Gaussian approximation `q=(z,theta)` and observation `y=Hq+v`, the
actual measurement update is

```
P_plus = P_minus - P_minus H^T (H P_minus H^T + R)^-1 H P_minus
```

Even zero innovation changes covariance. If all current state components are
observed exactly, their posterior variance and parameter cross-covariance vanish,
but generally `C_plus != C_minus`. Retaining the old C is justified as a deliberate
choice to keep identification separate, or if C already conditions on the new
evidence; it cannot generally be justified as an exact posterior update. A future
predictive prior cannot simply be carried forward as though a measurement never
arrived. Conversely, a *predicted*, unobserved next state needs its joint
state-parameter covariance if the aim is to reproduce the old prior prediction.
The scalar probe in the script falsifies conflation of these operations.

The joint-state approach is established in the original [Ait-El-Fquih,
El Gharamti and Hoteit state/parameter estimation paper](https://arxiv.org/abs/1511.02178),
which explicitly distinguishes augmented joint updates from dual update ordering.
The algebra above is independently derived for this audit; no filtering algorithm
from that application is being proposed as a replacement controller.

## Actuator state and empirical covariance

In this synthetic recovery, the true actuator state is supplied each interval.
It is therefore appropriate to hold it fixed in a new conditional forecast.
`DynamicsBelief.rollout` also permits actuator initialization from command history;
that branch differentiates the history reconstruction with respect to theta.
`BeliefPlanModel` instead receives an already constructed latent array. If callers
supply a model-estimated actuator state while claiming it is known, initial latent
uncertainty and its parameter dependence are omitted. That is an unresolved
interface assumption outside this known-actuator fixture, not the explanation of
its failure. Probe it with identical history reconstruction inside versus outside
the differentiated rollout; the nominal means must match and the derivative
can differ whenever actuator lag parameters affect the initializer.

`E((j+1)dt)` becomes `E(j dt)` after reset. This is expected for a newly observed
initial state. Empirical lead-time curves need not be monotone in PSD order; linear
interpolation preserves PSD, but not monotonicity across independently measured
knots. The audit reports eigenvalue extrema of the actual empirical difference and
computes support with empirical covariance alone, parameter covariance alone, and
both. It does not subtract empirical variance from the parameter uncertainty or
change either term to obtain feasibility.

Adding `E + S C S^T` is not automatically a variance decomposition merely because
its components have different names. If endpoint error is `S dtheta + epsilon`,
its second moment includes parameter spread and cross terms with epsilon. If E
already measures that whole error, adding parameter spread again overlaps it.
The scalar probe sets epsilon to zero and obtains 2 instead of 1 when both terms
represent the same error. This establishes a *possible double counting mechanism*,
not that the fitted Glassbox envelope actually counts this exact parameter
uncertainty twice. E is measured using one fitted nominal on held-out trajectories,
whereas C comes from structured information and can later change under adaptation.
The available artifacts do not identify a clean independent residual-noise term
or cross-covariance. A falsifiable calibration study would repeat independent fits
and hold-outs on known synthetic parameters, estimate endpoint squared error,
parameter contribution and their cross term, and compare predicted total with
observed total at matched initial conditions. One fit cannot identify this split.

## Smallest next step

Keep the current behavior and recheck shifted feasibility. Document the solve
boundary as **known current physical and actuator state, fixed parameter belief**;
state explicitly that a caller's estimated actuator array is not thereby an exact
measurement. No new NMPC decision variables, covariance shrinking, support
expansion, or secondary controller are needed for this clarification.

Expose the existing covariance terms separately in diagnostics before changing the
uncertainty model. If a joint estimator is later introduced, its output should
carry current state/actuator covariance, state-parameter cross-covariance and
updated parameter information with an observation timestamp. Using those requires
one augmented prediction equation, not copying the old horizon covariance. This is
larger modeling work and should follow the independent calibration probe above.
The fifth-solve appended-tail failure remains a separate terminal-feasibility
problem; this audit supplies no recursive-feasibility guarantee.

## Reproduction

`scripts/audit_shift_uncertainty.py` accepts a saved belief and NPZ containing
`commands`, `state`, `latent` (the original solved request), and optionally
`actual_state`, `actual_latent` at the next interval. It does no fitting or solving.
The report compares matching retained endpoints and excludes the appended tail.
`test_shift_audit_preserves_old_joint_sensitivity_and_separates_reset` in
`tests/test_nmpc.py` checks the chain-rule identity on a four-step nonlinear
multirotor with nonzero parameter and empirical covariance.

Fresh fixture results (baseline `4f5cddb`, unchanged first solved plan) are in
[the audit report](investigations/shift-uncertainty-audit.json). At shifted stage 15,
roll-rate support at the same physical endpoint is:

| Components | Old lead time | Fresh lead time |
| --- | ---: | ---: |
| Mean | 0.971849322 | 0.971849322 |
| Mean + parameter spread | 0.984554231 | 0.988036275 |
| Mean + empirical spread | 0.996947110 | 0.995841444 |
| Mean + combined spread | 0.999979615 | 1.000791311 |

These rows are independent ablations: standard deviations from covariance sums
are not sums of standard deviations. The empirical contribution decreases here;
the reset parameter contribution increases enough to cross support. The parameter
variance changes from `0.000187990` to `0.000305155`. In the chain identity, the
carried joint variance contributes `+0.000133875`, while the cross term contributes
`-0.000251040`. Dropping both destroys cancellation. The carried actuator
sensitivity is material (`0.000110374` variance at this endpoint); this remains
expected when that actuator state is actually known at the new planning instant.

Nominal matching endpoints are identical. Carrying the old joint sensitivity
reproduces old parameter covariance to `8.01e-11` maximum absolute difference;
physical/actuator/direct sensitivities add to `4.20e-9`. Thus the observed shift
has a demonstrated chain-rule explanation, not a covariance accumulation bug.
The parameter covariance difference has eigenvalues of both signs (extrema
`-0.000146475`, `+0.000469550`), directly falsifying monotone reset uncertainty.
From the actual next state, maximum retained support is `0.999661326`.

Build the shared fixture once (reuse it if already built for the suffix probe):

```bash
uv run python scripts/build_nmpc_research_fixture.py \
  --output /tmp/glassbox-nmpc-fixture
uv run python scripts/audit_shift_uncertainty.py \
  --belief /tmp/glassbox-nmpc-fixture/rich-belief.json \
  --plan-arrays /tmp/glassbox-nmpc-fixture/first-plan-audit.npz \
  --output /tmp/glassbox-shift-uncertainty-audit.json
uv run pytest tests/test_nmpc.py -k shift_audit -q
```

The builder's NPZ is a renaming of `tick0_predicted_commands`, `tick0_state`, `tick0_latent`,
`tick1_state`, and `tick1_latent` from the common `horizon-shift-states.npz` fixture,
in the input-key order described above. The report includes the fresh fixture's
manifest and input hashes. These commands rerun against current code; the artifact
retains its original dependency fingerprints. This pass makes no timing claims.
