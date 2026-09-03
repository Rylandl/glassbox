# Dual-control NMPC for an unseen multirotor

**Status: design under exploration, measured on a 16-release ensemble per
case. Pooled recovery: frozen-snapshot cascade 0.55, working-belief cascade
0.22, dual-control pass 2b 0.06, pass 4 0.02. Within every arm the recovered
releases reach command rank four earlier; the fourth pass's midpoint base
action slowed that and lowered recovery.** This
describes an experimental controller being built in `glassbox.experimental`. Nothing here is part of the stable API,
and the throw diagnostic's default controller is unchanged until the study on
this page reports.

## Goal

Replace the hand-gained cascade and the scan-based excitation in the throw
diagnostic with one optimization policy that starts with zero numerical
knowledge of the vehicle, chooses inputs that are informative while the model
is uncertain, and tracks the recovery goal as the posterior tightens. No hand
gains, no motor geometry, no vehicle numbers.

## What is assumed and what is learned

Retained as structure, not as numbers:

- rigid-body kinematics and gravity are known;
- there are four commands, each bounded in `[0, 1]`;
- accelerations are locally linear in the commands and in low-order state
  features, which is the hypothesis class the recursive identifier already
  fits;
- the state is measured; actuator lag is ignored in this pass.

Everything else is learned from the flight: the collective specific force per
command, the three by four map from commands to angular acceleration, the
nuisance coefficients on body velocity, body rate, and rate products, and the
hover command. The posterior comes from the existing information-form
recursion started from zero information, exposed as `RecursiveBootstrapBelief`
with its supported covariances and command information matrix.

The two action-side choices that no posterior can supply, and that are
declared rather than learned: the command bounds, and a small regularizing
information `epsilon` that makes the information gain finite before the first
observation. There are no other priors.

## Prediction model

With commands `u`, body velocity `v_b`, body rate `omega`, and attitude `R`:

```
f      = c . u + c0 + c_v . v_b                        specific force along body z
alpha  = A u + a_r omega + a_p (omega x omega terms) + a0
a_world = g + R [0, 0, f]^T
p' = v,   q' = 1/2 q (x) omega,   omega' = alpha
```

Every coefficient is the posterior mean from the recursion. First-order
predicted spread is propagated from the command-map covariance: the per-step
acceleration variance `phi_k^T Sigma phi_k` for the command features `phi_k`,
integrated once into rate and tilt spread and twice into altitude spread. This
is deliberately crude and is stated as such; it is the quantity the chance
constraints use.

## Objective

Over a horizon of 30 steps at 10 ms, parameterized as 10 command blocks of 3
steps, the controller minimizes

```
J(u) = sum_k l_k(x_hat_k)                              expected tracking cost
     + w_rate sum_k |u_k - u_{k-1}|^2                    command-rate cost
     - w_info [ log det(I_u + sum_k phi_k phi_k^T / s^2) - log det(I_u) ]
                                                          expected information gain
     + sum_k [ (h_floor + beta s_z,k - z_hat_k)_+ ]^2     altitude chance penalty
     + sum_k [ (tilt_hat_k + beta s_tilt,k - tilt_max)_+ ]^2
```

- `l_k` normalizes velocity, body rate, tilt, and altitude error by task
  tolerances. Those tolerances describe the goal, not the vehicle.
- `I_u` is the current command information matrix (starting at
  `epsilon * I`), `s^2` the identifier's residual variance, so the information
  term is the expected log-determinant gain about the command maps from the
  planned inputs. Its gradient decays on its own as information accrues, so no
  schedule is needed; `w_info` is a fixed constant.
- The chance penalties use predicted spread with a fixed `beta`. When spread
  exceeds a cap, the penalty is treated as saturated and dropped from the
  gradient, so that at zero information the optimizer is driven by the
  information term and the command bounds alone.
- `w_rate` and `w_info` are the only weights besides the task tolerances.

At zero information the tracking term is meaningless and the information term
dominates: the first inputs are bounded and span the command directions as
fast as possible, which is rank-seeking excitation derived from the objective.
As the posterior tightens the tracking term takes over. There is no freeze,
no certification, and no quality test; the controller plans on the live
posterior every interval.

## Solver

Projected gradient on the command blocks with the bounded Armijo search used by
the maintained NMPC, 8 to 12 outer iterations, warm-started by shifting blocks.
Posterior mean, information matrix, and residual variance are dynamic inputs
to one jitted program, so no recompilation happens per interval. A non-finite
or infeasible solve returns the previous bounded command; there is no fallback
controller.

## Study

The controller is a third mode of `glassbox crazyflow throw-study`, run on the
same seven cases as the freeze study: the canonical release, the five campaign
scenarios, the state-noise variant, and the mid-flight arm change. Beyond the
existing metrics it records information gain per step, the log-determinant
trajectory, chance-constraint activity, and solve iterations.

Success criteria for this pass:

1. every command finite and within bounds on every case;
2. the hover envelope is reached on at least the cases the cascade reaches;
3. command information reaches rank four within 0.3 s of control starting,
   measured, so the early actions are demonstrably informative;
4. allocation and command chatter no worse than the frozen-snapshot mode;
5. under state noise and after the mid-flight change, tracking no worse than
   the cascade;
6. no vehicle-specific number anywhere in the controller.

## First pass result (2026-09-02)

The controller as specified does not fly. On all seven cases it commands all
four motors at zero for the first half second after enable and the vehicle
reaches the floor; command information never reaches rank four before floor
contact on any case. The mechanism is structural, not a weight choice:

- At zero information the posterior command maps are exactly zero, so the
  tracking cost and both chance penalties have exactly zero gradient with
  respect to the commands. The only term with a gradient is the information
  gain, and at a motor-uniform command that gradient is parallel to the
  all-ones direction. The diagnostic releases with motors off, so the seed sits
  on the lower bound in that direction and the projected gradient vanishes.
  Uniform commands generate no command information, so the fixed point
  sustains itself. No positive scaling of any term changes this; every knob
  was tried once on the canonical case and none moved the first thirty
  commands.
- Even a global solve would not excite at the dimensional weights: the
  rank-four design earns about 30 nats more information than holding zero but
  costs about 550 units of command-rate penalty, so the information weight
  must exceed roughly 19 before excitation is preferred at all. The claim
  above that the information term dominates at zero information is therefore
  not weight-free, and the two terms have no shared currency.
- A plan seeded constant across blocks stays constant across blocks under
  first-order steps, so even after escaping motor symmetry the optimizer
  cannot build a rank-four design within one horizon.

What was met: every command finite and within bounds on every case, one
jitted program with no recompilation, no vehicle-specific number anywhere in
the controller, and a solve cost of about 3 ms. The implementation lives in
`glassbox.experimental.dual_control` and the study mode `dual_control_nmpc`;
the two existing modes are unchanged.

The next pass has to break the symmetry deliberately (a declared, tiny,
non-uniform seed or a multi-start over bounded orthogonal designs), compute
the information features on the residualized commands the identifier actually
accumulates, and give information a currency in tracking units, for example
by charging the expected tracking cost for predicted spread so that reducing
spread is worth something to the same objective.

## Out of scope for this pass

Actuator lag, forgetting or process noise for tracking configuration changes
(the posterior currently never forgets), integration into the maintained NMPC
model families, and any real-time claim. Each is a follow-up once the
feasibility question is answered.

## Second pass (2026-09-02)

The first pass failed for three reasons that the page above stated as
assumptions: the seed was a fixed point, the information features were not the
features the identifier accumulates, and the information term had no currency
in common with tracking. Each is addressed by one change, and each change is a
config switch, so the passes are three arms of one study rather than three
studies. `pass1` is the configuration above and is kept selectable so its
failure stays reproducible; `pass2a` adds the first two changes; `pass2b` adds
the third.

### Multi-start over declared orthogonal designs

Before the gradient refinement, one jitted `vmap` scores a small candidate set:
the shifted warm start, the previous command held, and both sign polarities of a
bounded orthogonal design at each declared amplitude. The design lays a
Hadamard row of order four on each command block, cycling the rows and flipping
the polarity every fourth block, so a single horizon visits the collective
direction and all three differentials in both polarities. Its
intercept-centered Gram has full rank four, which is the property that matters:
a design whose centered Gram were rank deficient could not buy rank four inside
one horizon at any amplitude.

The best candidate is refined by the existing projected-gradient solver. The
warm start and the held command are both in the set, so the seed is
structurally no worse than either, and the refinement only accepts steps that
decrease the objective, so the returned plan is no worse than the seed.

The declared amplitudes are `0.06`, `0.12`, and `0.25` of the command range: a
geometric ladder whose top rung makes one horizon's Gram outweigh the `epsilon`
prior by three orders of magnitude and whose bottom rung is small enough to
leave a settled hover alone. This tuple and the sign pattern are the
controller's one action-side prior beyond the command box, and they are the
same kind of declaration as the bounds.

### Residualized information features

The identifier accumulates `outer(f, f)` on `f = [(u - mid)/span, omega, omega
products, 1]` and reports as command information the Schur complement of the
nuisance block. The first pass planned on raw commands instead, so it credited
a motor-uniform plan with information the intercept would have absorbed: at the
canonical release it reported 11.7 nats per interval for holding all four
motors at zero. The planned features are now centered over the horizon, which
is exact residualization against the intercept, and then regressed out of the
planned body rates and rate products with a relative ridge. A uniform plan now
earns exactly zero — the same float twice, not a small number — and the
information gradient points along differential inputs.

### A shared currency

`pass2b` deletes the additive information term and the weight `w_info` with it.
The tracking cost is charged for predicted spread,
`E[l(x_k)] = l(x_hat_k) + trace(W Sigma_x,k)`, on the same task tolerances the
tracking error is normalized by. The parameter posterior used over the horizon
is the one the planned inputs would produce,
`Sigma_theta' = (I_u + Phi^T Phi / s^2)^-1` on the residualized features,
applied from the start of the horizon: the standard value-of-information
approximation of the dual effect.

`Sigma_theta'` is turned into an acceleration spread by averaging over the
command box rather than by evaluating it at the planned command. That choice is
load bearing. Every quadratic form `phi^T Sigma phi` has a zero, and an
optimizer handed one parks on it: evaluated at the planned command the charge
is minimized by holding, or by sitting at whatever command the form happens to
vanish at, and no amount of posterior spread changes that. The box average,
`trace(Sigma_theta') / 12` per output axis, asks instead how well the vehicle's
response is known over the whole box it will have to act in, which is the
quantity that decides whether the goal is reachable at all. It is A-optimal
design rather than D-optimal, and unlike a log-determinant it carries units, so
it can be added to a tracking cost without a weight.

Excitation then pays for itself: at zero information `Sigma_theta'` is
`epsilon^-1 I`, and one horizon of the top declared amplitude collapses it by a
factor of about two thousand, while the same design costs between ten and a
hundred task-tolerance units of command rate.
falls on its own and the marginal value of excitation falls with it, so the
controller anneals without a schedule, a freeze, or a quality test.

### One numerical guard

A posterior fitted from a handful of rank-deficient samples routinely carries a
rate-product coefficient of tens, and `omega' = a_p omega^2` blows up in finite
time, so the mean model's own thirty-step prediction reached infinity inside the
horizon and every plan scored the same infinite cost. On the canonical case
that made forty-one solves return no usable command at all, nearly all of them
in the first second after enable.
Predicted accelerations and states are now saturated at `1e6`, six orders of
magnitude above any reachable flight. The effect that matters is not the
finiteness but the flatness: the tracking term goes flat exactly where the mean
model has stopped saying anything, and the spread term decides there instead.

### Result

Seven cases, four arms, in `artifacts/crazyflow_throw_study/report-pass2.json`.
The two cascade arms are byte-identical to the first-pass report.

Symmetry breaking works and is not weight dependent. On the canonical case the
first thirty commands are all exactly zero in pass 1 and none of them are in
pass 2a or 2b; the first interval is won by a declared design in both, the
planned information rank is four at the first interval and at all but two of
the first thirty in pass 2a and all but four in pass 2b, and command evidence
reaches rank four in every case in both passes, worst case 0.98 s after enable
in pass 2a and 0.53 s in pass 2b, against never in five of seven cases in
pass 1. The third success criterion, rank four within 0.3 s of control
starting, is met on five of seven cases in pass 2b (0.25, 0.34, 0.53, 0.25,
0.27, 0.24, 0.25 s) and on two of seven in pass 2a (0.35, 0.38, 0.25, 0.28,
0.40, 0.98, 0.35 s).

The shared currency is what makes the excitation persist for the right length of
time. In pass 2a the log-determinant gain falls from 24.5 nats at the first
interval to 0.18 by the fiftieth while the tracking cost is between `1e5` and
`1e6`, so excitation stops paying long before the model is good enough to fly
on, and the plan wanders at a mean amplitude of 0.014. In pass 2b the spread
charge is 2000 against a tracking cost of `1e5` through the whole arrest, the
same order of magnitude rather than four orders below it, and the controller
trades between them continuously.

Recovery is partial. Pass 2b flies the canonical release, the mid-flight arm
change, and the high release to a hover without touching the floor, at a settled
command step of 0.0003 against 0.0010 for the frozen snapshot and 0.0804 for the
working belief, which is the quietest hover of any arm. It touches the floor on
the other four. Pass 2a touches the floor on six of seven. Both cascade arms
never touch it on any case. The `sustained_hover_duration_s` metric does not
separate these: a vehicle resting on the floor satisfies the hover envelope, so
pass 2a's 5.10 s on the canonical case is a vehicle lying on the ground and
pass 2b's 0.41 s is a real hover at 2.3 m. Read `minimum_altitude_m` alongside
it.

The mechanism of the remaining failures is identification speed against
altitude, and it is a vicious circle rather than a weight. The identifier
residualizes eleven features, so it can report no command information at all
until it has more than eleven samples, and the samples it does get during a
tumble are confounded: body rates of six to eight radians per second make the
rate-product regressors large and correlated with whatever the commands are
doing, so the Schur complement stays small. The canonical release is at 4.8 m
with 1.7 m/s of descent and more than a radian of attitude error when control
is enabled, which is 0.83 s of free fall; the cascade has a supported belief at
0.22 s because its probe rides on a smooth hand-designed base that keeps the
vehicle benign, and the dual controller needs 0.25 s to 0.53 s because its own
early commands make the vehicle less benign. Where that leaves enough altitude, it recovers; where
it does not, it hits the floor and then levels itself on the ground.

### Tuning record

Every value tried, all on the canonical case only, one adjustment per knob. All
six trials start from the same baseline, which is pass 2b at the first-pass
amplitude ladder, so the effect column is measured against that one run
(floor contact, 2.27 s of hover on the ground, rank four 0.44 s after enable,
settled command step 0.20). Only the amplitude row changed anything decisive:
every other trial still put the vehicle on the floor.

| knob | tried | kept | effect at the trial |
| --- | --- | --- | --- |
| `multi_start_amplitudes` | `(0.03, 0.06, 0.12)`, `(0.06, 0.12, 0.25)` | `(0.06, 0.12, 0.25)` | floor contact removed, rank four 0.44 s → 0.25 s, settled command step 0.20 → 0.0003 |
| `w_rate` | `25`, `100` | `25` | still hits the floor, maximum tilt 1.81 → 3.09 |
| `beta` | `2.0`, `3.0` | `2.0` | still hits the floor, hover 2.27 s → 0.84 s |
| `spread_cap` | `3.0`, `30.0` | `3.0` | still hits the floor, terminal rate 0.06 → 0.41 |
| `epsilon` | `1e-3`, `1e-2` | `1e-3` | still hits the floor, information rank four 0.69 s → 3.78 s |
| `altitude_floor_m` | `1.0`, `2.0` | `1.0` | still hits the floor, hover 2.27 s → 1.13 s |

Nothing else was tuned. The horizon, block length, solver budget, and the four
task tolerances are unchanged from the first pass.

The amplitude ladder is the only knob whose value changed, so it is the only one
whose canonical-only choice could be an accident of that case. Re-running all
seven cases at both ladders says it is not: the old ladder touches the floor on
seven of seven, the new one on four, it is at least as fast to rank four on six
of seven cases, and it is quieter in the settled window on every case where the
vehicle is actually hovering. The choice generalizes in the direction it was
made; it does not rescue the four cases that still land.

### What is still open

The three criteria the first pass met still hold: every command finite and in
bounds on every case, one compiled program with no recompilation across
posteriors, and no vehicle-specific number in the controller. The rank-four
timing criterion is met on five cases of seven in pass 2b. The chatter
criterion is met, and beaten, on exactly the cases that do not touch the floor:
settled command steps of 0.0003, 0.0009, and 0.0003 against 0.0010 for the
frozen snapshot, and 0.18 to 0.34 on the four that land, where those numbers
describe a controller working against the ground rather than a hover. The
hover-envelope criterion is met without floor contact on three of seven. Under
state noise pass 2b's terminal speed, rate, and settled angular error are all
better than either cascade arm (0.000, 0.027, 2.47 against 0.429, 0.449, 2.50),
but it reaches them on the floor; after the mid-flight arm change it holds a
real hover with a settled angular error of 0.12 against the snapshot's 0.59.

The honest reading is that this controller is slower to earn attitude authority
than a hand-gained cascade by roughly a factor of two, which on a release with
under a second of altitude budget is the difference between a hover and a
landing.

The next thing to try is not another weight. It is the confounding: the
identifier cannot separate a command effect from a rate-product effect during a
tumble, and the controller's own excitation makes the tumble worse. Either the
objective has to charge for the confounding it is about to create — the planned
Gram already contains the nuisance regressors the plan implies, so the
information it reports is honest, but nothing yet prefers a plan whose nuisance
block is quiet — or the horizon has to be long enough for the arrest to appear
in the tracking term before the altitude is gone.

## Third pass (2026-09-02)

The second pass named identification latency as the remaining mechanism and
proposed two structural answers: stage the identifier's regressors so a
supported model arrives sooner, and constrain the collective command
coefficients to the sign the command channel already means. Both are
implemented, both are opt-in, and both are measured here. Neither helps. What
the measurement does establish is that the premise was wrong and that the
pass-2 outcome split was never a property of the design.

Both switches live in `RecursiveBootstrapConfig` and default to off, so the
certified and working cascade arms are untouched: re-running them on all seven
cases reproduces every certified and working case record in
`report-pass2.json` byte for byte, and their difference block with them. The
new arm `dual_control_nmpc_pass3` turns both on and plans with the pass-2b
objective unchanged, so nothing in the optimization moved between the two arms.
`pass2b` stays selectable and is re-run alongside.

### Staged regressors

The identifier accumulates the full Gram and right-hand side every interval
exactly as before; staging only chooses which columns the point estimate,
ranks, support projectors, authorities, and covariances are solved over. Stage
one is the command block and the intercept, so the command features are
residualized against the intercept alone, which is exact centering rather than
a fitted projection, and four command directions can be resolved from five
samples instead of from more samples than the regression has columns. Stage two
is every column, and the fully staged branch hands the accumulated arrays
through untouched, so once staged the estimate is bit-for-bit the solve this
identifier has always run — asserted on identical data, not approximated.

The admission condition is a ratio of counts: effective samples against the
full column count of that regression, at a declared multiple. The Schur
complement removes a fraction of order `p / n` of the command energy purely by
the nuisance block's own freedom, and the smallest eigenvalue of a sample Gram
sits near `(1 - sqrt(p / n))**2` of its population value; at `n = 4 p` both
readings bound the damage at a quarter. So the multiple is four, and at a
hundred hertz the collective block stages 0.32 s after enable and the angular
block 0.44 s, on every case, deterministically.

Inside the identifier it does what it claims. On the linear hidden-plant
fixture the staged solve is supported at five intervals against nine unstaged.
Replayed on the pass-2b canonical flight — the same measured transitions fed to
both — it is supported at 21 intervals against 24, and in closed loop it
reaches command rank four earlier on three of seven cases, improving the
worst case across the seven from 0.53 s after enable to 0.37 s. Only the
replay is a controlled comparison: from the first interval on, the two arms
are flying different trajectories.

Three intervals. That is what the change buys against a 0.83 s free fall.

### Collective sign as command semantics

The normalized command channel is thrust fraction, so the fitted collective
coefficients are projected onto the nonnegative orthant: a negative coefficient
moves to exactly zero, a nonnegative one does not move at all, and the count
and the norm of what was removed are recorded on the belief and in the study.
The span is positive, so clipping in normalized units is the same qualitative
constraint as clipping in raw units, and it carries no magnitude of its own.

On a healthy trajectory it is almost inert. Replayed on the pass-2b canonical
flight it fires once in nine hundred intervals, at interval three, moving one
coefficient by 0.13 specific force per unit command against a fitted collective
sum of 11.5 — a 1.1 percent change to one number, once.

In the pass-3 flights it fires on 5 to 96 percent of intervals with magnitudes
up to 1123 per unit command, but almost all of that is after floor contact,
where commands produce no acceleration and the collective fit has nothing left
to fit.

### Result

Seven cases, five arms, in `artifacts/crazyflow_throw_study/report-pass3.json`.
The two cascade arms are byte-identical to the pass-2 report.

| case | arm | recovered | speed | rate | tilt | hover s | min alt | supported s | alt m | descent | rank four s | settled step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical | certified | yes | 0.0110 | 0.0119 | 0.0010 | 5.47 | 1.200 | 0.22 | 4.27 | 3.12 | 0.22 | 0.0010 |
| canonical | working | no | 0.1247 | 0.1406 | 0.0543 | 0.00 | 1.200 | 0.22 | 4.27 | 3.12 | 0.22 | 0.0804 |
| canonical | pass2b | yes | 0.0417 | 0.0037 | 0.0004 | 0.41 | 1.200 | 0.25 | 4.16 | 3.25 | 0.25 | 0.0003 |
| canonical | pass3 | floor | 0.0000 | 0.0165 | 0.0060 | 2.62 | -0.001 | 0.58 | 2.46 | 6.08 | 0.35 | 0.0189 |
| shorter_arms_high_release | certified | yes | 0.0119 | 0.0128 | 0.0011 | 5.56 | 2.500 | 0.22 | 5.57 | 3.11 | 0.22 | 0.0010 |
| shorter_arms_high_release | working | no | 0.1086 | 0.1281 | 0.0689 | 0.00 | 2.500 | 0.22 | 5.57 | 3.11 | 0.22 | 0.0807 |
| shorter_arms_high_release | pass2b | yes | 0.0029 | 0.0106 | 0.0006 | 3.37 | 2.500 | 0.34 | 5.09 | 3.85 | 0.34 | 0.0009 |
| shorter_arms_high_release | pass3 | floor | 0.0000 | 0.1598 | 0.0473 | 0.00 | -0.001 | 0.36 | 4.96 | 4.65 | 0.36 | 0.0070 |
| long_arms_cross_axis_tumble | certified | yes | 0.0165 | 0.0111 | 0.0009 | 5.17 | 2.500 | 0.63 | 4.22 | 3.24 | 0.63 | 0.0010 |
| long_arms_cross_axis_tumble | working | no | 0.0830 | 0.2108 | 0.0463 | 0.00 | 2.500 | 0.63 | 4.22 | 3.24 | 0.63 | 0.0803 |
| long_arms_cross_axis_tumble | pass2b | floor | 0.4960 | 0.2152 | 0.0200 | 0.00 | -0.001 | 0.53 | 4.39 | 4.46 | 0.53 | 0.2301 |
| long_arms_cross_axis_tumble | pass3 | floor | 0.0023 | 0.8073 | 0.0453 | 0.00 | -0.001 | 0.28 | 5.30 | 4.06 | 0.28 | 0.4775 |
| milder_low_energy_release | certified | yes | 0.0109 | 0.0118 | 0.0010 | 5.42 | 2.500 | 0.23 | 4.66 | 3.19 | 0.23 | 0.0010 |
| milder_low_energy_release | working | yes | 0.0003 | 0.0002 | 0.0001 | 3.89 | 2.500 | 0.23 | 4.66 | 3.19 | 0.23 | 0.0000 |
| milder_low_energy_release | pass2b | floor | 0.1975 | 0.0577 | 0.0092 | 0.00 | -0.001 | 0.25 | 4.55 | 3.68 | 0.25 | 0.1939 |
| milder_low_energy_release | pass3 | floor | 0.0000 | 0.2317 | 0.0903 | 0.00 | -0.001 | 0.39 | 3.82 | 5.33 | 0.37 | 0.2728 |
| reversed_tumble | certified | yes | 0.0128 | 0.0099 | 0.0008 | 4.83 | 0.845 | 0.23 | 5.53 | 3.30 | 0.23 | 0.0010 |
| reversed_tumble | working | yes | 0.0062 | 0.0030 | 0.0006 | 4.91 | 0.777 | 0.23 | 5.53 | 3.30 | 0.23 | 0.0472 |
| reversed_tumble | pass2b | floor | 0.0075 | 0.2492 | 0.0235 | 0.00 | -0.001 | 0.27 | 5.35 | 3.98 | 0.27 | 0.3358 |
| reversed_tumble | pass3 | floor | 0.0013 | 0.3433 | 0.0583 | 0.00 | -0.001 | 0.17 | 5.70 | 3.18 | 0.17 | 0.3447 |
| canonical_state_noise | certified | no | 0.4291 | 0.4492 | 0.0432 | 0.00 | 1.200 | 0.44 | 3.47 | 3.77 | 0.23 | 0.0755 |
| canonical_state_noise | working | no | 0.4291 | 0.4492 | 0.0432 | 0.00 | 1.200 | 0.44 | 3.47 | 3.77 | 0.23 | 0.0755 |
| canonical_state_noise | pass2b | floor | 0.0000 | 0.0273 | 0.0357 | 1.51 | -0.001 | 0.53 | 2.82 | 5.34 | 0.24 | 0.1764 |
| canonical_state_noise | pass3 | floor | 0.0009 | 0.0382 | 0.0203 | 0.26 | -0.001 | 0.08 | 4.65 | 2.43 | 0.08 | 0.1698 |
| canonical_mid_flight_arm_change | certified | yes | 0.0102 | 0.0114 | 0.0005 | 5.44 | 1.200 | 0.22 | 4.27 | 3.12 | 0.22 | 0.0010 |
| canonical_mid_flight_arm_change | working | yes | 0.0488 | 0.0477 | 0.0065 | 0.09 | 1.200 | 0.22 | 4.27 | 3.12 | 0.22 | 0.0744 |
| canonical_mid_flight_arm_change | pass2b | yes | 0.0506 | 0.0046 | 0.0006 | 0.09 | 1.200 | 0.25 | 4.16 | 3.25 | 0.25 | 0.0003 |
| canonical_mid_flight_arm_change | pass3 | floor | 0.0000 | 0.0231 | 0.0027 | 2.98 | -0.001 | 0.58 | 2.46 | 6.08 | 0.35 | 0.0104 |

`recovered` means the hover envelope was reached without the vehicle touching
the floor. Read `hover s` against `min alt`: a vehicle resting on the ground
satisfies the hover envelope, so pass 3's 2.62 s on the canonical case and
2.98 s after the arm change are a vehicle lying on the ground, and the settled
command steps on those rows describe a controller working against the floor.
The two cascade arms never touch it. Pass 2b recovers three of seven. Pass 3
recovers none. Times are measured from model enable. The report records the
supported-model and rank-four moments only for the dual arms, so the cascade
rows here were measured separately from the same runs.

### The premise does not survive measurement

Identification latency is not what decides these cases.

At the moment each arm first has a supported model, the altitude it has left is
far more than the altitude it needs. Taking the vehicle's own net upward
authority from the hidden hover command — `g (1/h - 1)` = 8.63 m/s² — the
stopping distance implied by the descent rate at first support is 0.34 to
2.14 m, against 2.46 to 5.70 m of altitude actually in hand. Every arm-case in
the table, including every one that lands, has between 0.3 and 5.1 m of spare
altitude at the moment it first knows what its motors do.

The comparison that settles it is `long_arms_cross_axis_tumble`. The certified
cascade reaches a supported model there at 0.63 s after enable, at 4.22 m,
descending at 3.24 m/s. Pass 2b reaches one at 0.53 s, 0.10 s *earlier*, at
4.39 m, a higher altitude. The cascade recovers and pass 2b lands. The pass-2
page's reading — that the cascade's 0.22 s against the dual controller's 0.25
to 0.53 s is the difference between a hover and a landing — is not supported by
its own data.

What differs is not when the model arrives but what is commanded before it
does. Over the first thirty intervals after enable, with nobody holding a model,
the certified cascade commands a mean collective of 0.787 of the command range
on the canonical case, rising from 0.621 in the first ten intervals to 0.950 in
the third ten. This vehicle's hover command is 0.532. The cascade is climbing
hard from the first interval it is allowed to act.

It can do that because its collective reference defaults to the *midpoint of
the command box* whenever collective authority is zero, and the midpoint of
`[0, 1]` is 0.500 against a hover of 0.532. The command box is a declared fact
about the actuator interface, not about the vehicle — but that this particular
vehicle hovers within six percent of that box's midpoint is a fact about the
vehicle, and the cascade collects it for free by defaulting there. The dual
controller collects nothing: its plan starts from the previous command, which
at release is zero, and it has to buy its way up from there.

The two cases read the same way. On the canonical case pass 2b commands a mean
collective of 0.542 over the first thirty intervals — 0.312, then 0.655, then
0.658 — selects a declared excitation design on eleven of the twenty-four
intervals before it has a supported model, and arrests above the release
height. Pass 3 commands 0.035 over the same window, selects a design on eight
of its fifty-seven pre-support intervals, and is still falling at 6.08 m/s when
the model arrives. On `long_arms_cross_axis_tumble`, which every dual arm
loses, pass 2b commands 0.457 and excites on thirty-one of fifty-two
pre-support intervals; pass 3 commands 0.023 and excites on four of
twenty-seven, has a supported model 0.25 s sooner and 0.9 m higher than
pass 2b, and lands anyway.

The mean collective over the same thirty intervals tracks the outcome across
arms far better than any timing does. The three arm-cases that recover sit at
0.436, 0.542 and 0.542; every arm-case below 0.41 lands. `long_arms` under
pass 2b sits at 0.457 and lands anyway, so an early collective near hover is
necessary here, not sufficient.

### Why staging makes it worse

Staging does not merely fail to help; it reproduces the first pass's failure
by another route.

The command information the identifier reports is the Schur complement of the
nuisance block, and the objective divides its planned parameter covariance by
it. Staged against the intercept alone, that complement is much larger on
identical data: replaying the pass-2b canonical flight through both, the
largest command-information direction is 18 to 44 times larger under staging
from the tenth interval to the fortieth, and the box-averaged spread the
objective actually charges — `trace(Sigma) / 12` — falls to between 0.17 and
0.46 of its unstaged value over intervals 25 to 40. From interval 60, where
both are fully staged, the ratio is exactly 1.000.

The spread charge is the only term that pays for moving at zero information.
Shrink it and the command-rate cost wins, and the plan stays where it was — at
the zero command the vehicle was released with. Measured: over the first thirty
intervals pass 3 commands a mean collective of 0.035, 0.042, 0.023, 0.033,
0.066 and 0.035 on six of the seven cases — the state-noise case is the
exception, at 0.269 — against 0.542 for pass 2b on the canonical case. That is
free fall with the motors essentially off, which is the pass-1 fixed point,
arrived at because staging told the objective it already knew the command
map.

Staging alone reproduces this exactly: with the sign projection off, the
canonical case still commands a mean collective of 0.035 over the first thirty
intervals. The sign projection alone leaves it at 0.255.

### The pass-2 outcome split is a coin flip

Enabling the sign projection alone changes exactly one number in the whole run
before the trajectories diverge: at interval three it clips one collective
coefficient from -0.13 to zero, a 1.1 percent change to the fitted collective
sum. The first four commands are bit-identical to pass 2b's. The fifth differs.
Within thirty commands the two plans differ by the full command range, and the
canonical case goes from a recovery at 1.2 m to floor contact.

Running each change separately on all seven cases makes the same point from the
other side. Pass 2b recovers canonical, `shorter_arms_high_release` and the arm
change. Staging alone recovers only `long_arms_cross_axis_tumble` — a case
pass 2b loses. The sign projection alone also recovers only
`long_arms_cross_axis_tumble`, cleanly, at 2.5 m. Both together recover
nothing. Four arms, four different subsets of between zero and three cases —
and the first two of those four are separated by one clipped coefficient at
interval three.

So "pass 2b recovers three of seven" is not a property of pass 2b. It is one
draw from a controller whose early trajectory is chaotically sensitive to a
posterior fitted from three or four samples, and no comparison between arms at
this resolution means anything. That applies to this pass's numbers as much as
to the last one's.

### Tuning record

The staging multiple is the only new knob. It was set to four from the
conditioning argument above before any flight, and two alternatives were tried
on the canonical case only. None of them changes the outcome, so the reasoned
value is kept.

| knob | tried | kept | effect at the trial |
| --- | --- | --- | --- |
| `staging_sample_multiple` | `2.0`, `4.0`, `8.0` | `4.0` | floor contact on all three; supported 0.39 / 0.58 / 0.35 s, rank four 0.39 / 0.35 / 0.35 s, settled command step 0.053 / 0.019 / 0.341 |

Nothing else was tuned. The objective, its weights, the amplitude ladder, the
horizon, the solver budget and the four task tolerances are unchanged from pass
2b, which is what makes pass 3 an ablation of the identifier rather than a new
controller.

### Criteria

1. Every command finite and within bounds on every case: met, on all five arms,
   with no unusable solve in any of the three dual arms.
2. The hover envelope reached on at least the cases the cascade reaches: not
   met. The certified cascade reaches it without floor contact on six of seven
   (it misses under state noise), the working cascade on four of seven, pass 2b
   on three, and pass 3 on none.
3. Command information rank four within 0.3 s of control starting: met on three
   of seven cases in pass 3 (0.35, 0.36, 0.28, 0.37, 0.17, 0.08, 0.35 s) against
   five of seven in pass 2b. The worst case improves, from 0.53 s to 0.37 s.
4. Chatter no worse than the frozen snapshot: not met in pass 3. Settled command
   steps are 0.0070 to 0.4775 against the snapshot's 0.0010, but every pass-3
   row is a vehicle on the floor, so the number describes a controller working
   against the ground rather than a hover.
5. Under state noise and after the mid-flight change, tracking no worse than the
   cascade: pass 3's terminal speed, rate and tilt under state noise (0.0009,
   0.038, 0.020) beat both cascade arms (0.429, 0.449, 0.043), and it reaches
   them on the floor. After the arm change it is better on terminal speed
   (0.0000 against 0.0102) and worse on rate and tilt (0.023 and 0.0027 against
   0.011 and 0.0005), on the floor, where pass 2b held a real if brief hover at
   1.2 m.
6. No vehicle-specific number anywhere in the controller: met. Both new switches
   are unit-free — a ratio of counts, and a sign.

### What this leaves

The next thing to try is not a third identifier change. Two things follow from
the numbers above.

The first is that this study cannot currently distinguish designs. Any
comparison between arms has to survive a perturbation of the size that already
flips it: an ensemble over release states or over first-interval posteriors,
not seven deterministic releases. Until that exists, "recovers *n* of seven" is
a draw, not a result.

The second is that the honest gap is the base action, not the belief. The
cascade is not faster to a model; it climbs at 0.79 of the command range while
it has no model at all, because its zero-information default is the box
midpoint and the box midpoint nearly hovers this vehicle. The dual controller
starts from the released zero command and its objective gives it no reason to
leave, because at zero information the mean model predicts nothing and the only
term with a gradient is a spread charge that any excitation collapses. A
zero-information default that is a statement about the command box rather than
about the vehicle — the midpoint, which the cascade already uses and the
optimizer does not — would be the smallest honest change to test, and it is a
change to where the plan starts rather than to what the objective says.

## Fourth pass (2026-09-02)

The third pass ended with two things to do and this pass does both. The first
is the base action: the controller had no statement about where in the command
box to act while it knew nothing, so it inherited one from the release, and the
throw releases with the motors off. The second is the protocol: seven
deterministic releases cannot tell two designs apart when one clipped
coefficient at interval three flips a case.

Two declared changes, both config switches, both off in every earlier arm, so
`pass2b` is re-run unchanged beside them. On all seven cases the certified and
working records in `report-pass4.json` are byte-identical to `report-pass3.json`,
case block and difference block included, and every `pass2b` number is unchanged
— its record gains three keys and changes none.

The result is that the ensemble works and the base action does not. Both are
below; the second is the more useful of the two.

### The rate cost is a slew cost on the controller's own actions

The command-rate term charges `w_rate |u_k - u_{k-1}|^2` over the horizon, and
its first move is measured from the command the vehicle is carrying when the
solve runs. That is right for every move between two of the controller's own
consecutive actions: it is a statement about how fast this controller is
willing to move. It is not right for the handover. At the first interval after
enable the previous command is the diagnostic's zero, chosen by fiat because
the throw releases with the motors off, and charging the plan for leaving it is
a prior toward zero thrust that nothing in the objective declares.

So the transition out of a command this controller did not issue is not
charged, and every later one is charged exactly as before. The term keeps its
weight and its form; only the set of moves it is summed over changes.
`previous_command_owned` is an argument to `solve`, false exactly once per
flight in this diagnostic, and the study records the count so the claim is
checkable: one uncharged transition per flight on every case.

### The base action at zero information

A multi-start design is a spread of commands around a center. Centering it on
the previous command means that at zero information the only thing the
controller says about where in the box to act is whatever it was handed, and at
a motors-off release that is the lower bound. The declared alternative comes
from the command contract and from nothing else: commands are normalized thrust
fractions on `[0, 1]` and hover is somewhere inside, so with no information
about where, the midpoint is the maximum-entropy choice and the unique point
minimizing the worst-case distance to hover over the box. It is the same class
of statement as the bounds, it is the same number for every vehicle, and it is
not fitted, measured, or tuned. That this particular vehicle then turns out to
hover at 0.532, six percent from that midpoint, is a fact about the vehicle
that the controller neither knows nor uses — it is the same free lunch the
cascade collects by the same route, which is what the third pass measured.

The declaration is not permanent. The moment the posterior can answer the
question the center becomes the posterior's own hover command, on the
identifier's existing support rule and nothing new: command evidence spanning
all four motors, angular effect spanning all three body axes, and a collective
effect implying a hover command inside the box. That is the rule the
certification transaction requires of a candidate and the rule working mode
hands control over on. The center in use is recorded per interval, so the
handover is visible: 0.43, 0.47, 0.54, 0.89, 1.07, 6.20 and 6.55 s after enable
across the seven cases.

It does what it was designed to do at the action level. On the canonical
release `pass2b`'s first four commands average 0.000, 0.026, 0.166 and 0.191 of
the command range — the first is exactly the command it was released with —
and `pass4`'s average 0.681, 0.614, 0.552 and 0.180. Over the ensemble the
first-0.3 s mean collective rises from 0.314 to 0.410, against 0.717 for both
cascade arms.

### The ensemble protocol

Each of the seven study cases is drawn sixteen times. Each world velocity
component is scaled by an independent `U[0.8, 1.2]`, each body rate component
likewise, and the release attitude is rotated by `U[0, 0.1]` radians about a
uniformly random horizontal body axis. The release height, the hidden airframe,
the loop rate and every controller setting are unchanged, so the altitude
budget the arms are spending stays comparable across draws. Each draw is seeded
from the triple `(ensemble seed, case index, replicate index)`, so a single row
reproduces on its own and adding a case or a replicate never moves the others,
and every arm flies exactly the same draws, so every comparison is paired.

Reported per case and arm: the recovery rate with a Wilson 95 percent score
interval, the median and worst terminal speed, rate and tilt, minimum altitude,
time to command rank four, the first-0.3 s mean collective, and the settled
chatter metrics; plus one pooled rate per arm over all 112 releases. Recovery
is the page's own criterion — the hover envelope reached and the floor never
touched — because the envelope alone is satisfied by a vehicle resting on the
ground. The interval is Wilson rather than normal because at sixteen releases
the normal interval for zero recoveries has exactly zero width.

Four arms, 448 trials, 23 minutes on four processes. The report is
`artifacts/crazyflow_throw_study/report-pass4-ensemble.json`.

### Result

| case | arm | recovered | rate | Wilson 95 | early collective | rank four s med/worst | settled step med/worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical | certified | 7/16 | 0.44 | 0.23-0.67 | 0.786 | 0.29/0.95 | 0.0010/0.1000 |
| canonical | working | 2/16 | 0.12 | 0.03-0.36 | 0.786 | 0.29/0.95 | 0.0802/0.1000 |
| canonical | pass2b | 1/16 | 0.06 | 0.01-0.28 | 0.369 | 0.28/0.41 | 0.1697/0.3375 |
| canonical | pass4 | 0/16 | 0.00 | 0.00-0.19 | 0.386 | 0.55/1.38 | 0.2161/0.9869 |
| shorter_arms_high_release | certified | 11/16 | 0.69 | 0.44-0.86 | 0.754 | 0.23/0.76 | 0.0010/0.1000 |
| shorter_arms_high_release | working | 5/16 | 0.31 | 0.14-0.56 | 0.754 | 0.23/0.76 | 0.0742/0.1000 |
| shorter_arms_high_release | pass2b | 1/16 | 0.06 | 0.01-0.28 | 0.328 | 0.34/0.57 | 0.2082/0.4193 |
| shorter_arms_high_release | pass4 | 0/16 | 0.00 | 0.00-0.19 | 0.272 | 0.79/5.24 | 0.1916/0.8728 |
| long_arms_cross_axis_tumble | certified | 9/16 | 0.56 | 0.33-0.77 | 0.726 | 0.23/0.39 | 0.0010/0.1000 |
| long_arms_cross_axis_tumble | working | 2/16 | 0.12 | 0.03-0.36 | 0.726 | 0.23/0.39 | 0.0757/0.1000 |
| long_arms_cross_axis_tumble | pass2b | 2/16 | 0.12 | 0.03-0.36 | 0.421 | 0.29/0.45 | 0.0808/0.4421 |
| long_arms_cross_axis_tumble | pass4 | 2/16 | 0.12 | 0.03-0.36 | 0.531 | 0.51/1.24 | 0.2283/0.4142 |
| milder_low_energy_release | certified | 16/16 | 1.00 | 0.81-1.00 | 0.732 | 0.23/0.31 | 0.0010/0.0010 |
| milder_low_energy_release | working | 9/16 | 0.56 | 0.33-0.77 | 0.732 | 0.23/0.31 | 0.0400/0.0860 |
| milder_low_energy_release | pass2b | 2/16 | 0.12 | 0.03-0.36 | 0.381 | 0.28/0.63 | 0.0549/0.5474 |
| milder_low_energy_release | pass4 | 0/16 | 0.00 | 0.00-0.19 | 0.419 | 0.49/0.98 | 0.2171/0.5275 |
| reversed_tumble | certified | 6/16 | 0.38 | 0.18-0.61 | 0.732 | 0.25/0.38 | 0.0010/0.1000 |
| reversed_tumble | working | 2/16 | 0.12 | 0.03-0.36 | 0.732 | 0.25/0.38 | 0.0733/0.1000 |
| reversed_tumble | pass2b | 0/16 | 0.00 | 0.00-0.19 | 0.278 | 0.34/1.33 | 0.1921/0.4512 |
| reversed_tumble | pass4 | 0/16 | 0.00 | 0.00-0.19 | 0.410 | 0.58/1.36 | 0.2296/0.5776 |
| canonical_state_noise | certified | 2/16 | 0.12 | 0.03-0.36 | 0.511 | 0.23/0.23 | 0.0198/0.1000 |
| canonical_state_noise | working | 1/16 | 0.06 | 0.01-0.28 | 0.511 | 0.23/0.23 | 0.0652/0.1000 |
| canonical_state_noise | pass2b | 0/16 | 0.00 | 0.00-0.19 | 0.069 | 0.58/4.65 | 0.2598/0.4710 |
| canonical_state_noise | pass4 | 0/16 | 0.00 | 0.00-0.19 | 0.450 | 0.36/0.58 | 0.3163/0.8185 |
| canonical_mid_flight_arm_change | certified | 10/16 | 0.62 | 0.39-0.82 | 0.779 | 0.23/0.74 | 0.0010/0.1000 |
| canonical_mid_flight_arm_change | working | 4/16 | 0.25 | 0.10-0.49 | 0.779 | 0.23/0.74 | 0.0757/0.1000 |
| canonical_mid_flight_arm_change | pass2b | 1/16 | 0.06 | 0.01-0.28 | 0.356 | 0.32/0.51 | 0.1609/0.2819 |
| canonical_mid_flight_arm_change | pass4 | 0/16 | 0.00 | 0.00-0.19 | 0.406 | 0.42/1.07 | 0.1991/0.6551 |
| **pooled** | **certified** | **61/112** | **0.545** | **0.452-0.634** | **0.717** | 0.23/0.95 | 0.0010/0.1000 |
| **pooled** | **working** | **25/112** | **0.223** | **0.156-0.309** | **0.717** | 0.23/0.95 | 0.0723/0.1000 |
| **pooled** | **pass2b** | **7/112** | **0.062** | **0.031-0.123** | **0.314** | 0.32/4.65 | 0.1802/0.5474 |
| **pooled** | **pass4** | **2/112** | **0.018** | **0.005-0.063** | **0.410** | 0.48/5.24 | 0.2258/0.9869 |

Median and worst terminal speed, rate, tilt and minimum altitude are in the
report; the pooled medians are 0.011, 0.012, 0.0009 and 1.200 for the certified
cascade, 0.010, 0.115, 0.0202 and -0.001 for `pass2b`, and 0.002, 0.466, 0.0804
and -0.001 for `pass4`. `pass2b` touches the floor on 99 of 112 releases and
`pass4` on 107, so their settled chatter numbers describe a controller working
against the ground, not a hover.

Because every arm flies the same releases, the comparisons are paired, and the
exact sign test on the discordant releases is the honest reading:

| comparison | discordant | exact two-sided p |
| --- | --- | --- |
| certified vs working | 40 / 4 | 1.7e-08 |
| certified vs pass2b | 54 / 0 | 1.1e-16 |
| certified vs pass4 | 59 / 0 | 3.5e-18 |
| working vs pass2b | 23 / 5 | 9.1e-04 |
| working vs pass4 | 25 / 2 | 5.7e-06 |
| pass2b vs pass4 | 6 / 1 | 0.125 |

The protocol does what it was built for. It separates the cascade from the dual
controller at a resolution nothing in the first three passes could reach, and
it separates the certified cascade from the working one. It does not separate
`pass4` from `pass2b`: their intervals overlap, the sign test does not reject,
and the point estimate moves the wrong way.

It also puts a number on how much the earlier reports were worth. Deterministic
against pooled: certified 6/7 against 0.545, working 4/7 against 0.223, `pass2b`
3/7 against 0.062, `pass4` 0/7 against 0.018. Perturbing the release makes every
arm worse, and it costs the dual arms far more than the cascade — `pass2b` keeps
a seventh of its deterministic rate and the certified cascade keeps two thirds.
"Recovers three of seven" was worth even less than the third pass said.

### The base action fails, and it takes the third pass's reading with it

`pass4` raised the pooled first-0.3 s mean collective from 0.314 to 0.410 and
its recovery rate fell from 0.062 to 0.018. That is the intervention the third
pass's closing paragraph asked for, run on 112 releases per arm, and it says
the commanded collective before a model exists is not the causal quantity at
this margin.

The ensemble says why the third pass thought otherwise. The correlation between
the first-0.3 s mean collective and recovery over all 448 trials is 0.381, which
looks like a mechanism. Within an arm it is 0.241 for the certified cascade,
0.083 for the working one, 0.119 for `pass2b` and 0.050 for `pass4`. Almost all
of the pooled association is *between* arms: the arms that recover also happen
to command more collective, and with three arm-cases the third pass could not
tell those apart. Its own caveat — an early collective near hover is necessary,
not sufficient — was closer to right than the reading built on top of it.

What does separate recovered from lost releases, inside every arm, is how
quickly the command evidence reaches rank four. Recovered against lost, median
seconds after enable: 0.245 against 0.351 for the certified cascade, 0.239
against 0.309 for the working one, 0.314 against 0.410 for `pass2b`, and 0.375
against 0.658 for `pass4`. It is the same sign in all four arms, and it is the
quantity the third pass discarded. This is correlational and the obvious
confound is real — a benign trajectory both identifies faster and recovers —
but it is now the reading with within-arm support, and the collective is the
reading without it.

The mechanism by which the base action hurts is visible in the same direction.
`pass4` reaches rank four later than `pass2b` on the pooled median, 0.48 s
against 0.32 s, and its early plan carries markedly less differential: over the
first 0.3 s the mean deviation of a command from its own four-motor mean is
0.039 of the command range against `pass2b`'s 0.113, on the seven deterministic
releases. It buys collective and spends differential. The consequence is a
vehicle that is pushed hard along a body axis it has no authority over: pooled
median terminal rate 0.466 rad/s against 0.115, and the hover envelope reached
on 10 of 112 releases against 37.

### Tuning record

No weight, tolerance, amplitude, horizon or solver setting moved in this pass.
The two switches are the whole change, and neither carries a number.

One choice inside the second switch was made after looking at a flight, and it
is recorded as such. The handover from the declared midpoint to the posterior's
hover estimate was first written on the identifier's collective authority being
positive. On the canonical release that fires at interval five, 0.06 s after
enable, on a posterior fitted from five samples — the design center left a
declaration that is true by construction for an estimate that is not yet true
at all. It was replaced with the identifier's own support rule on that ground:
the observation that decided it is when the handover fires, not what the flight
did afterwards. For the record the discarded version scored better on the one
release that was visible when the change was made — canonical early collective
0.715 against 0.328 — and still put the vehicle on the floor, and no ensemble
was run on it.

| knob | tried | kept | effect |
| --- | --- | --- | --- |
| base-action handover rule | `collective_authority > 0`, identifier support rule | support rule | handover 0.06 s against 6.20 s after enable on the canonical release; floor contact either way |

### Criteria

1. Every command finite and within bounds on every case: met, on all four arms,
   on all 448 ensemble trials and all 28 deterministic ones. The ensemble does
   not record solve status per trial; on the deterministic runs, which do,
   neither dual arm returned an unusable solve on any case.
2. The hover envelope reached on at least the cases the cascade reaches: not
   met, and now not met by a margin that is measured rather than drawn. Pooled
   over 112 releases the certified cascade recovers 0.545 [0.452, 0.634] and
   `pass4` 0.018 [0.005, 0.063], with 59 discordant releases and none of them
   won by `pass4`.
3. Command information rank four within 0.3 s of control starting: not met.
   `pass4`'s pooled median is 0.48 s against `pass2b`'s 0.32 s and the cascade's
   0.23 s; on the deterministic releases `pass4` is slower than `pass2b` on all
   seven.
4. Chatter no worse than the frozen snapshot: not met. Pooled median settled
   command step 0.2258 against the snapshot's 0.0010 — measured on flights that
   are on the floor, so the number describes a controller working against the
   ground.
5. Under state noise and after the mid-flight change, tracking no worse than the
   cascade: not met on either. Under state noise `pass4` recovers 0 of 16 against
   the cascade's 2; after the arm change 0 of 16 against 10.
6. No vehicle-specific number anywhere in the controller: met. The two new
   switches are a boolean and a point in the command box that is the same point
   for every vehicle.

### What this leaves

The base action is answered and the answer is no. Commanding near the box
midpoint from the first interval is defensible, costs nothing to declare, and
does not recover this vehicle; the third pass's reading that it was the decisive
quantity does not survive an ensemble.

The ensemble itself is the durable part of this pass, and it changes what the
next question can be. Three things follow from it.

The gap is not marginal and is not a tuning distance. `pass2b` recovers 6 percent
of releases where the certified cascade recovers 55 percent, and loses all 54
releases they disagree on. No weight moves a controller across that.

The two cascade arms differ from each other by the same protocol — 0.545 against
0.223, p = 1.7e-08 — so the ensemble has the resolution to detect a design change
that matters. A change that does not move it is not being hidden by noise.

And the quantity that separates recovered from lost inside every arm is the time
to command rank four, which is where the second pass started and where the third
pass concluded, on three arm-cases, that the answer was not. That conclusion was
drawn from between-arm comparisons at a resolution that could not support it. It
should be re-opened, and now it can be asked properly: the ensemble is a paired
112-release experiment, and any identifier or excitation change can be run
through it and read against these intervals.

## Fifth pass (2026-09-02)

The fourth pass answered its own question and left the design where the second
pass had it: a thirty-step horizon over bounded command blocks, a spread charge
averaged over the command box, and a declared amplitude ladder to break the
motor symmetry. This pass replaces all three at once, on one stated principle.

There is one goal — stabilize the vehicle: velocity, body rate, tilt, and an
altitude floor — and learning is valued only for what it buys towards that
goal. There is no midpoint, no staging, no cascade, and no amplitude ladder.
The declared quantities are the command box, a per-interval slew limit, the two
outcome limits, and the four task tolerances. Everything else is derived from
the posterior and the state.

### Diagnosis

Four readings from the recorded pass-2b and pass-4 flights, each of which the
formulation below answers directly.

The horizon is too short to prefer the right manoeuvre. Over `0.3 s` a
tumbling vehicle cannot both right itself and arrest its descent, so the
objective compares "thrust now, tilted" against "right first, thrust later"
over a window in which only the first has begun to pay. It picks the first.

The spread charge cannot see the coupling that makes that choice wrong.
Specific force acts along the body `z` axis, so an uncertain attitude turns a
known thrust into an unknown acceleration *direction*. The box-averaged charge
treats the four channels as independent chains of integrators and carries no
term in which tilt uncertainty costs velocity, so thrusting while tilted is
free in exactly the case where it is most expensive.

The amplitude ladder is violent on this vehicle. The identified angular map
carries row norms of `457` and `402 rad/s^2` per unit command on roll and pitch
— the certified arm's own fitted belief on the canonical release — so the
ladder's top rung, a quarter of the command range laid differentially, is a
command for hundreds of radians per second squared. The ladder is declared in
command units and has no way to know that.

And the log-determinant information term is not goal-directed. It scores
directions by how much they are learned, not by what learning them is worth to
the one goal, so it buys evidence about a motor the vehicle does not currently
need and charges the tracking cost for it.

### Formulation

**Moves, not commands.** The decision variables are per-block *moves*, each
bounded by the declared per-interval slew of `0.10` of the command range, and
the plan is their cumulative sum from the command the vehicle is holding,
clipped to the box. The slew stops being a cost to trade against and becomes a
box the bounded solver already projects onto; the executed command cannot leave
the previous one by more than the declared fraction whatever the objective
prefers. The horizon is `100` steps in `20` blocks — one second at the study's
loop rate — so righting and then thrusting fits inside the window.

**The full-regressor planned posterior.** The belief now exposes the two
accumulated Grams the identifier solves from, in the identifier's own feature
order, rather than only the command-block summaries the fits reduce them to.
The spread is the posterior of the whole regressor set that the plan itself
would leave behind: `Sigma' = ((G + Psi^T Psi) / s^2 + eps I)^-1` with `Psi` the
regressors the planned mean trajectory would present. No residualization enters
it, and none should: a plan that moves a nuisance regressor is buying real
information about the coefficient that regressor multiplies.

**Coupled propagation.** The per-step spreads are evaluated at the plan's own
features and integrated along the trajectory, with the coupling the earlier
model could not express:

```
sigma_omega  = cumsum(T sigma_alpha)
sigma_tilt   = cumsum(T sigma_omega)
sigma_a,k    = sigma_f,k + |f_k| sigma_tilt,k
sigma_v      = cumsum(T sigma_a)
sigma_alt    = cumsum(T sigma_v)
```

`|f_k| sigma_tilt,k` is the whole point: thrusting under an uncertain attitude
is charged, in metres per second, for the velocity it might produce sideways.

**Clipped spreads, live penalties.** Each propagated spread is clipped at
`spread_cap` times its matching tolerance and every chance penalty stays in the
objective. The earlier passes dropped a penalty whose spread was saturated,
which lets a plan escape a constraint by making its own prediction worse.
Clipping instead gives "unknown" a bounded, plan-independent charge: no plan
profits from ignorance, and a plan whose own excitation brings a channel back
under the cap is charged strictly less. The gradient goes flat inside a
saturated channel, and the multi-start is what moves the plan there.

**Posterior-derived seeds.** The declared designs are gone. The eight seeds are
the shifted warm start, the held command, four excitation cycles along the
eigenvectors of the *current* command covariance — weakest-known direction
first, in both polarities and in a variant whose polarity flips every cycle so
it is zero-mean — a righting move allocated through the pseudo-inverse of the
posterior's own angular map, and a collective move along the posterior's own
collective map. Each seed moves at most one declared slew per block. At zero
information the covariance is a multiple of the identity, so the excitation
cycle is the four commands one at a time — the symmetry break the ladder used
to declare, now read off the posterior — and the righting and collective seeds
are the held command, because a zero map allocates nothing.

**A second outcome limit.** The maximum body rate joins the maximum tilt as a
declared outcome limit, charged in the same chance form: predicted magnitude
plus a reserved spread, past the declared maximum, squared.

### Declared and derived

| declared | derived |
| --- | --- |
| command box `[0, 1]^4` | every command inside it |
| slew `0.10` of range per interval | which direction to move, and how far inside the slew |
| maximum tilt `0.50 rad`, maximum body rate `5.0 rad/s` | when either constraint is active |
| four task tolerances, altitude floor | the whole trajectory cost |
| horizon, block length, optimizer budget, `epsilon` | the plan and every seed |
| — | the excitation directions, the righting move, the collective move, every spread |

No base action, no design amplitudes, no sign pattern, no staging schedule, and
no vehicle number anywhere. The one thing the fourth pass declared that this
pass does not is the box midpoint.

### Result

The fifth pass is worse than every earlier one, and it fails for a reason that
is visible in one column of the trace.

The same paired ensemble the fourth pass introduced, re-run with all five arms
on the same 112 releases each; the protocol and the full per-case table are on
[the release-ensemble page](../experiments/dual-control-throw-ensemble.md).
The four earlier arms are byte-identical to `report-pass4-ensemble.json`, all
448 trials, so the comparison is the fourth pass's own numbers with a fifth
column added. On the seven deterministic
releases the two cascade arms reproduce `report-pass4.json` byte for byte, and
both dual arms reproduce every number in it; their records gain the six switch
keys this pass declares and one penalty summary, and change nothing.

| arm | recovered | rate | Wilson 95 | early collective | rank four s med | terminal tilt med | on the floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| certified | 61/112 | 0.545 | 0.452-0.634 | 0.717 | 0.23 | 0.0009 | 38 |
| working | 25/112 | 0.223 | 0.156-0.309 | 0.717 | 0.23 | 0.0276 | 37 |
| pass2b | 7/112 | 0.062 | 0.031-0.123 | 0.314 | 0.32 | 0.0202 | 99 |
| pass4 | 2/112 | 0.018 | 0.005-0.063 | 0.410 | 0.48 | 0.0804 | 107 |
| pass5 | 0/112 | 0.000 | 0.000-0.033 | 0.044 | 0.96 | 1.2232 | 109 |

Zero of 112, with a Wilson upper bound of `0.033`, against `pass2b`'s `0.062`
and the certified cascade's `0.545`. On the seven deterministic releases it is
the same picture: no sustained hover on any case and the floor touched on all
seven, against `pass2b`'s three clean recoveries.

Three further readings, none of which any earlier arm produced.

`pass5` is the first arm that fails to reach command information rank four at
all. Thirteen of its 112 ensemble releases never get there, and on the
deterministic `milder_low_energy_release` it never does either, with the solve
stalling on `801` of `900` intervals.

`pass5` is the first arm that drives the simulator to a non-finite state. Three
of its 112 releases end in a position, velocity or quaternion the plant refuses
to return; those trials are recorded as diverged and counted as not recovered
rather than dropped. No other arm has ever done this on this diagnostic.

And `pass5` has the *quietest* settled command of any arm in the study —
pooled median step `0.0006` of the command range against the frozen snapshot's
`0.0010` — measured on flights that are lying still on the floor. It is the
clearest illustration this study has produced of why the chatter criterion is
read after the recovery criterion and not instead of it.

The cause is the first-`0.3 s` mean collective, which pools to `0.044` for
`pass5` against `0.314` for `pass2b` and `0.717` for both cascade arms, and
which on the deterministic releases is `0.000` in every third of that window on
five of the seven cases. The selected candidate over those intervals is `hold`
or the shifted warm start, and nothing else. The vehicle is commanded to keep
its motors off while it falls, and by the time the posterior can say anything
about the collective map it is already low, fast and tilted: pooled median
terminal tilt `1.22 rad`, which is a vehicle on its side.

### Holding is the cheapest plan the spread charge can see

This is not the slew limit. At `0.10` of range per interval a plan reaches this
vehicle's hover command in one interval; the limit never binds on the way up.
It is the spread model, and the mechanism is exact rather than empirical.

The planned posterior is evaluated at the plan's *own* features. A plan that
holds one command visits one direction of the command block for the whole
horizon, and the posterior it leaves behind knows that one direction to
`s^2 / N`. A plan that cycles four directions spends `N / 4` samples on each
and is evaluated at points it knows to `4 s^2 / N`. Spreading the same horizon
over four directions therefore roughly doubles every propagated standard
deviation and quadruples the quadratic charge. At zero information, from any
held command, the four excitation seeds score strictly worse than `hold` — the
measured spread charge on the canonical release is `43.4` for `hold` against
`204.0` for the excitation cycle — and the tracking term cannot break the tie
because a posterior with a zero command map predicts the same free fall for
every plan.

That is the first pass's fixed point, reached by a different route. The first
pass could not earn information from a uniform plan; the fifth pass can, and
declines to, because the charge it pays for the excitation is levied in the
same units as the goal and is larger than what the excitation buys within the
horizon. "Learning is valued only for what it buys towards the goal" is exactly
the principle, faithfully implemented, and on this vehicle at this margin the
answer it returns is *nothing*.

The second pass's box average was not an approximation to be improved on. It
was load-bearing, and its own docstring said so: averaging over the box rather
than evaluating at the planned command is what stops the charge from being
zeroed by parking on whatever command the quadratic form happens to vanish at.
Replacing it with the planned features restores exactly the failure it was
written to prevent.

### Criteria

1. Every command finite and inside the box on every case: met for the
   controller, not met for the flight. `pass5` returned no unusable solve on
   any deterministic case and the slew bound holds elementwise on every
   executed command by construction, but three of 112 ensemble releases ended
   in a simulator state the plant refuses to return, so the arm fails
   `all_values_finite_and_bounded`.
2. The hover envelope reached on at least the cases the cascade reaches: not
   met, by a wider margin than any earlier pass. Zero of 112 ensemble releases
   and zero of seven deterministic ones, against the certified cascade's 61 and
   six and `pass2b`'s 7 and three.
3. Command information rank four within `0.3 s` of control starting: not met,
   and now not met at all on some releases. `pass5`'s pooled median is `0.96 s`
   against `pass2b`'s `0.32 s` and the cascade's `0.23 s`, and 13 of its 112
   releases never reach rank four.
4. Chatter no worse than the frozen snapshot: met on the number and meaningless
   as a reading. Pooled median settled command step `0.0006` against the
   snapshot's `0.0010`, measured on 109 flights that are on the floor.
5. Under state noise and after the mid-flight change, tracking no worse than
   the cascade: not met on either. Zero of 16 under state noise against the
   cascade's 2, and zero of 16 after the arm change against 10.
6. No vehicle-specific number anywhere in the controller: met, and more
   completely than any earlier pass. The midpoint, the amplitude ladder and the
   sign pattern are all gone, and nothing replaced them.

### Tuning record

Nothing was tuned. No weight, tolerance, horizon, block length or solver
setting was moved after a flight was looked at. The horizon and block length
are stated by the design, the slew and the rate limit are declared, and the two
numbers that could have been tuned to rescue the result — `w_rate` and
`spread_cap` — were left at the values every earlier pass used.

| knob | tried | kept | effect |
| --- | --- | --- | --- |
| — | — | — | nothing moved in this pass |

### What this leaves

The spread charge is the open question, and it is now a sharp one. Two forms
have been measured on the same protocol: a box average that rewards excitation
and cannot see the tilt-to-thrust coupling, and a planned-trajectory form that
sees the coupling and rewards standing still. Neither is right, and the reason
neither is right is the same in both cases — the charge is a function of the
plan's own predicted uncertainty, and a plan can always reduce that by
promising to do less.

What the goal actually needs is the spread of the *closed-loop* outcome under
the commands the vehicle will have to issue later, not the spread at the points
this horizon happens to visit. That is a statement about the reachable set, and
neither form approximates it. The box average is a crude bound on it; the
planned-trajectory form is not a bound on it at all.

The coupling term is worth keeping. It is the only part of this pass that
answers a diagnosis rather than restating one, it costs nothing to compute
alongside the box average, and it is the term that makes "right first, then
thrust" cheaper than "thrust while tilted" in the units of the goal. The
obvious next configuration is the second pass's box-averaged charge with the
tilt-to-thrust coupling added and the one-second horizon kept, which separates
the two changes this pass confounded.

The one-second horizon is untested on its own. Every `pass5` flight it appears
in also carries the spread model that stops the vehicle from thrusting, so
nothing here says whether the longer window helps. The same is true of the slew
box, the posterior-derived seeds and the rate limit: this pass moved five things
at once and the ensemble measures their joint effect. That was the wrong shape
for an experiment, and it is only cheap to say so because the answer came back
as a zero rather than as a small improvement that would have had to be
attributed.

The ensemble protocol and the recorded numbers are in
[the release-ensemble page](../experiments/dual-control-throw-ensemble.md).


## Sixth pass (2026-09-02)

The fifth pass took over two hours to reach a conclusion that was visible in
the first thirty intervals of its first flight. The sixth pass was run
differently: one release at a time on a single-trial gate that reports in
seconds, the seven deterministic cases only for a configuration that passed
the gate, and the release ensemble only for one that passed the seven. Every
configuration below was measured that way, and the ones that did not survive
are recorded with the reason, because each rules out a formulation rather
than a number.

### What survived

`pass6` keeps the fifth pass's one-second horizon, slew-bounded moves,
posterior-derived seeds, declared rate limit, and tilt-to-thrust coupling, and
changes four things.

The spread charge is the box-averaged one again, on the full regressor set:
the command block averaged uniformly over the box, the nuisance block fixed at
the *measured* body velocity and rates, the intercept included, under the
posterior the plan would leave behind. It is plan-independent except through
the information the plan buys, so no plan can lower it by visiting fewer
points, which is the fifth pass's failure, and it carries the attitude
coupling, which the second pass's charge could not.

The goal is charged only as far as the *current* posterior can see. Each goal
term, and its chance penalty, counts on the steps where that channel's
predicted spread under the incumbent posterior is still under the cap, and
always on the executed block. The horizon the goal sees is then a consequence
of what has been learned rather than a number, and because it is decided by
the incumbent posterior it is the same for every candidate in a solve: a plan
cannot profit by learning less. A mean rollout that has run past what the data
supports carries no goal cost there, and the clipped spread charge is the
whole statement about those steps. This is what stops a one-second rollout of
a three-sample map from deciding anything.

The seeds gain one descent plan per goal term: the steepest-descent direction
of the velocity, rate, tilt, or floor cost with respect to the whole plan of
moves, taken at the held command by differentiating the objective's own mean
rollout, and rescaled so its largest move is the slew. Nothing is allocated
by hand; at zero information every map is zero and every goal seed is the
held command.

The excitation seeds decompose the command precision in the orthonormal
Hadamard basis rather than the motor basis. Eigenvectors do not depend on the
basis, so once anything is learned the seeds are the posterior's own weakest
directions either way. What the basis decides is the degenerate case: at zero
information the fifth pass probed the four motors one at a time, which buys
the least thrust per unit of torque on any multirotor, and the sixth probes
the collective first and then the three zero-sum patterns. That order treats
the motors as exchangeable and nothing more; it names no vehicle.

### What did not survive

Each of these was measured on the single-release gate, most on all seven
deterministic cases, and none was tuned before being dropped.

- **Goal horizon gated by the planned posterior.** Charging the goal only
  where the plan's *own* posterior can see hands back the fifth pass's fixed
  point: a plan that learns more exposes more goal cost and loses to one that
  stays ignorant. The vehicle held zero command and fell.
- **A sequential action charge.** Recursive least squares inside the rollout,
  so each step's spread is the predictive spread at that step's own command
  under the posterior that has absorbed the plan's earlier samples. Clipped at
  the cap it protects nothing, because the cap is far below what a fall costs;
  unclipped it is decided entirely by the regularizing prior, which is a hidden
  scale for the vehicle's response. Both crashed more often than the box.
- **An action charge under the incumbent posterior.** With the nominal prior
  it paralyses the controller (six of seven floor contacts, commands near
  zero); with the prior scale read from the learned directions, on the
  presumption that unlearned motors are as strong as learned ones, it crashed
  five of seven. The trade-off between paralysis and blind action is set by
  the prior scale either way, and no vehicle-free constant can supply it.
- **A knowledge term within one slew of the held command.** Meant to let the
  information reward shrink at hover. It did, and it starved the early phase
  of the global learning it needs: four of seven floor contacts, and after a
  contact the posterior-gated goal vanished and the vehicle climbed away at
  full thrust. The executed-block floor on the goal horizon dates from this.
- **A knowledge term over the regressors actually visited**, the identifier's
  own second moment with the box as one pseudo-sample. The term shrank at
  hover as intended and the arrest overshot instead: the known horizon
  lengthened into the part of the mean rollout that extrapolates the fitted
  damping and coupling terms.
- **Trusting nuisance regressors only within the data range**, clipping the
  rates and speeds entering the fitted nuisance terms at the half-width of a
  uniform distribution with the accumulated second moment. Alone it did not
  separate from the box; with the goal charged over the full declared horizon
  it was the worst configuration measured, seven of seven floor contacts. The
  short posterior-gated horizon is doing real work during the arrest, and not
  only against nuisance extrapolation: a one-second rollout of a partial
  command map is wrong on its own.

### Result

Three arm-only ensembles on the recorded release distribution, paired with
the fifth pass's report by construction. The first two were measured during
the iteration; the third is the committed `pass6` configuration, re-measured
after the switches that keep the fifth pass bit-identical were added, and it
is the recorded result.

| configuration | recovered | rate | Wilson 95 | floor | airborne, not in envelope |
| --- | --- | --- | --- | --- | --- |
| box charge, motor-basis probes, goal seeds | 14/112 | 0.125 | 0.076-0.199 | 48 | 50 |
| box charge, Hadamard probes, goal seeds | 11/112 | 0.098 | 0.056-0.167 | 50 | 51 |
| `pass6` as committed | 10/112 | 0.089 | 0.049-0.157 | 52 | 48 |

The three are one configuration up to the order of the seeds and last-ulp
arithmetic, and their spread is what chaotic sensitivity to a
three-sample posterior looks like at this sample size. Against the recorded
arms: `pass2b` 0.062, `pass4` 0.018, `pass5` 0.000, the working cascade 0.223,
the certified cascade 0.545. The sixth pass is the first learned arm whose
interval excludes the fifth pass's, and it is not distinguishable from the
second.

The split is the useful part. In the Hadamard run, of the 62 releases that
stay off the floor, 51 end level, within 0.03 rad of upright and under
0.2 rad/s, but drifting laterally at 0.08 to 0.3 m/s, just outside the
0.1 m/s envelope; the committed run splits the same way. The drift alone is
worth about forty points of recovery, which would put the arm level with the
certified cascade. The floor contacts have a lower first-0.3 s
collective than the airborne releases (0.245 against 0.362 of the range) and
the same time to rank four, so they are spending the early window learning
rather than thrusting, not learning more slowly.

### Round two: the neighbourhood the goal needs

The drift is an objective problem, not a solver problem: it survives a
tenfold solver budget. At hover the box-averaged knowledge term was still
near its cap, because "how well is the response known over the whole command
box" never gets small, and the optimizer kept trading a little tracking for a
little information. Every attempt to shrink that term at hover by declaring a
smaller neighbourhood also removed the global learning the early phase needs.

The neighbourhood the goal needs can be read off the posterior. Hover is the
command at which the posterior-mean maps give specific force `g` and zero
angular acceleration: four linear equations in the four normalized commands.
Their information about the hover command is `M^T D^-1 M`, with `M` the
stacked maps and `D` the predictive variance of each output at the held
command, so an unlearned map direction contributes nothing and a learned one
pins the command. Combined with the box as the prior, which is already the
command contract's statement about where hover is (uniform on the box, so
zero mean and `1/12` variance per axis), this gives a Gaussian over the hover
command that is exactly the box at zero information and collapses onto the
hover point as the maps are learned. The knowledge term averages the
predictive variance over that distribution. Nothing between the two limits is
declared.

Used for both the knowledge term and the goal horizon, it removed the drift
and introduced an overshoot: with the hover point pinned, the goal horizon
lengthened into the part of the mean rollout that extrapolates the fitted
damping and coupling terms, the arrest climbed to eight metres, and the
vertical settle took longer than the window. Used for the knowledge term
only, with the goal horizon still decided over the box, the canonical and
reversed-tumble releases settle into a true hover, within 0.02 m/s and level,
with no overshoot. On the arm-only ensemble, same releases, this
configuration with the block warm start recovers 47 of 112 (0.420, Wilson
0.332-0.512), against 10 of 112 for the box-averaged knowledge term. The
floor contacts are unchanged at 51; of the 61 releases that stay airborne,
52 now reach the envelope. The drift is gone, and the arm is above the
working cascade (0.223) and inside the certified cascade's interval
(0.452-0.634). The state-noise case is the exception: 0 of 16, with a median
terminal speed of 0.9 m/s, so under measurement noise the hover itself is
not held. That is the committed `pass6`: the knowledge term over the
posterior's hover distribution, the goal horizon over the box.

### The warm start was executing the plan five times too fast

Found while looking at why the arrests bottom out so low. The plan is
parameterized in blocks of five steps and re-solved every step, and the warm
start dropped the previous plan's first block every solve. The seed the
optimizer refines therefore advanced by one block per interval: an
excitation cycle planned as four fifty-millisecond blocks was executed as
four ten-millisecond ones, and every seed's later blocks arrived five times
sooner than the objective had priced them. The second pass onwards carried
this, at three steps per block. Under `"step"` the phase rides on the
result: the block plan is kept for `block_steps` intervals and shifted only
when its first block has been executed in full, so the seeded plan progresses
in real time. On the gate the two arrests that had bottomed out within
0.2 m of the floor now stop at 1.2 m and 2.5 m, and the reversed tumble,
which had stopped at 2.5 m, now stops at 0.4 m: the same chaotic spread the
ensemble exists to average over. On the ensemble the two warm starts recover
the same 47 of 112 and split the losses differently: the block shift touches
the floor on 51 releases and leaves 9 airborne outside the envelope, the
real-time shift touches the floor on 61 and leaves 1. The compressed
execution was, by accident, a faster excitation cycle, and the early phase
is sensitive to exactly that.

### Composite seeds

The pure seeds offer the optimizer either a goal plan or an excitation plan,
and ten iterations of projected gradient rarely assemble "thrust while
probing" from the two. Each goal seed is therefore also offered laid over the
excitation cycle, clipped to the slew box: four more candidates, no new
numbers. With the real-time warm start this recovers 54 of 112 (0.482,
Wilson 0.392-0.574), floor contacts 50, and 57 of the 62 airborne releases
in the envelope. The interval excludes every earlier learned arm and the
working cascade, and contains the certified cascade's point estimate. That
is the committed `pass6`.

Two more block structures were measured and dropped. Blocks of one step at
the front lengthening to 44 at the back, so the plan probes at the loop rate
and still sees a second, crashed four of four on the gate when each block
could move one slew per step it lasts — the uncharged tail of the plan held
bang-bang commands that the step shift carried forward into execution — and
three of four when every block was bound to one slew. Uniform five-step
blocks stay; `block_lengths` remains available.

### Round three: the maps at the identifier's trust

With trials stopped at floor contact and the ensemble's own perturbed
releases flyable on the gate, the crashed releases could be watched one by
one. They share a signature: within a tenth of a second the goal seeds drive
every command to zero, the vehicle falls for half a second with the motors
off, and the command map never reaches rank four. Exposing every multi-start
candidate's objective on the result showed the trade. At the failing
interval the angular map has rank one and attributes the free tumble to the
collective command; at a 74° tilt the tilt term, normalized by a 0.05 rad
tolerance, dominates every other term; and under that map the way to stop
tilting is to cut thrust. Every candidate's knowledge term is saturated at
the cap, so information cannot outbid it, and with the motors off there is
no excitation to teach the map.

The identifier already publishes how far each fit has been earned: a
collective authority and a per-axis angular authority, ramped on its own
information singular values, which is what the cascade acts through. The
mean rollout now uses the maps at that trust, each angular axis scaled by
its authority and the collective map by its own. A map fitted from a handful
of samples in one direction then predicts almost nothing, the goal has no
lever to cut thrust with, and the collective and excitation seeds decide.
On the ensemble this recovers 57 of 112 (0.509, Wilson 0.418-0.600); the
first-0.3 s collective rises from 0.32 to 0.54 of the range, the level the
cascade thrusts at; and every release that stays off the floor reaches the
envelope. The floor contacts themselves are unchanged at 51: the arrest now
begins earlier, and the releases that still crash spin up on a partial map
before they can right, in flights that last under a second.

### Round four: the identifier's differenced noise

The replay of the state-noise flight settled where the deficit was. The
angular axes reach full authority by 0.45 s under noise, but the collective
residual sits at 2.8 m/s² against a 0.05 floor, its authority stays at zero
for half a second, and the collective map wanders between negative and three
times the truth. The cause is structural: the collective target is the
specific force implied by a velocity change over one 10 ms interval, so
two centimetres per second of velocity noise becomes nearly three metres per
second squared of target noise, against under two from a probe of a tenth of
the range. Every regression form tried offline on the noisy trace converged
to the same estimate by 0.4 s; what differed was the confidence the
identifier could honestly claim on the way there.

The identifier can now assimilate one sample per window of transitions, the
window's mean features and targets weighted by the window length, which
telescopes most of the differenced noise away while keeping the sample
count, the support thresholds, and the residual floor per transition; the
mechanism is on [the identification page](bootstrap-identification.md). The
window trades noise against the variation inside it: a window of five
averaged the differential excitation away across block boundaries and
slowed identification by half, and a window of three measured on the
ensemble recovers 57 of 112, no better than round three, with the
state-noise case still at zero. A window of two recovers 68 of 112 (0.607,
Wilson 0.515-0.693): floor contacts fall from 51
to 39, the state-noise case goes from none to two, the first-0.3 s collective
rises to 0.61, and every airborne release reaches the envelope. The point
estimate is above the certified cascade's for the first time, with the
interval still containing it. That is the committed `pass6`; the cascade
arms keep a window of one and are bit-identical.

### Round five: the remaining contacts, one by one

With the ensemble's own perturbed releases flyable on the gate, the 39
releases round four still loses were characterized from the report and
watched one at a time. They are the low-budget releases: the vertical throw
scale is 0.90 at the median against 1.03 for the recovered ones, so the
apex is lower and the floor closer, while the first-0.3 s collective is the
same 0.6 of the range in both groups. On a representative one, at 3.7 m with
a 77° tilt, two things cost the time it did not have. Rank four arrived at
0.45 s, and the righting oscillated, pitch rate swinging to plus and minus
six radians per second, because the authority-scaled map underestimates the
torque during the brake.

Measuring the excitation that actually reaches the identifier on that
release explained the first: the differential variation in the executed
commands is a tenth of the collective's and is 97 to 99 percent correlated
with the body rates. It is feedback, which the identifier rightly
residualizes against its rate regressors, not a probe. The excitation seeds
win the multi-start only sporadically, because the knowledge term is capped
and a goal that is losing altitude outbids it.

Three responses were measured. Using the maps at face value once the
identifier's support spans them (`face_value_at_full_rank`) addresses the
brake; on the ensemble it recovers 62 of 112 against round four's 68, inside
the noise, with the same floor contacts, so the brake overshoot is not what
decides those releases. It stays as a switch, off. A probe laid over the executed command along the posterior's weakest
direction until the support is complete (`probe_until_supported`) brought
rank four forward to 0.3 s on the crashed draws but recovers 53 of 112
against round four's 68: within the declared slew the probe takes the
budget the arrest needs, and the spin-ups on a wrong-axis map are not
prevented by learning faster. It stays as a switch, off. The identifier's
prequential residual, the error the belief makes predicting each transition
before absorbing it, was meant as the honest scale for a map fitted from as
many samples as it has parameters, and is on
[the identification page](bootstrap-identification.md). As implemented,
averaged with exponential forgetting over the certification window, it is
worse than either: with the probe it recovers 20 of 112 and touches the
floor on 91. The error an empty or rank-one belief makes is the target
itself, and a window of forty-eight transitions remembers it for half a
second, so the authority stays at zero, the maps scale to nothing, and the
vehicle falls. Gating the record on the identifier's own support, so that
only a fitted map's errors count, recovers 29 of 112 with 77 floor contacts:
better than counting ignorance, still far worse than the in-sample scale.
The map's genuine early errors on a tumbling vehicle are large, an honest
scale says so, the authority stays low, and a controller that waits for a
good map loses to one that acts on an imperfect one. It stays as a switch,
off. The reading is that on this task the residual floor's optimism is
load-bearing: it is what lets the controller act at all in the first half
second.

Dissecting the brake itself, interval by interval with every candidate's
value, showed dithering rather than a wrong plan: the winning candidates
alternate among the righting, rate, tilt, and collective seeds, whose values
differ by a few percent, the refinement stalls within a few iterations, and
the pitch rate swings with the alternation. The rate term was charged over
the executed block only, because the goal horizon's spread included the
fitted damping and coupling coefficients evaluated at the measured rates, and
at four radians per second those saturate the rate channel at once. The goal
horizon is now decided by the command maps' box-averaged uncertainty alone
(`horizon_neighbourhood="box_commands"`): how well the maps are known, not
how well the nuisance terms extrapolate to the rates the vehicle has now. On
the ensemble this recovers 69 of 112 (0.616, Wilson 0.524-0.701) with floor
contacts down from 39 to 34; the state-noise case is at zero on this run,
against two on the previous. That is the committed `pass6`.

### Round six: the release distribution

The remaining contacts were the low throws, for the cascade as much as for
the learned arm, and a release thrown too low to be caught by any controller
measures the throw. The ensemble's second distribution never throws weaker
than the case declares and puts its width into the angular impulse: velocity
scale on `[1.0, 1.2]`, angular velocity scale on `[0.5, 1.5]`. Paired on it,
the certified cascade recovers 84 of 112 and the learned arm 79, intervals
0.662-0.821 and 0.615-0.782, with the working cascade at 37. Per case the
learned arm leads on the reversed tumble (13 against 6) and the short arms,
ties on the canonical release, and trails on the state-noise case (1 against
7), which accounts for the pooled gap on its own. The table is on
[the release-ensemble page](../experiments/dual-control-throw-ensemble.md).

### Round seven: the collective map on the integrated target

The one structural deficit left was measurement noise, and its cause was in
the identifier: the collective target is the specific force implied by the
velocity change over one interval, so velocity noise enters divided by the
interval. The identifier can now fit the collective map on the cumulative
target with one constant column for the anchor, which is the exact
least-squares form for white measurement noise; the mechanism, and the
first version that was measured and dropped, are on
[the identification page](bootstrap-identification.md). The rest of the
stack sees it as an equivalent per-transition Gram and does not change.

On the second release distribution the learned arm recovers 90 of 112
(0.804, Wilson 0.720-0.867) against the certified cascade's 84, with the
state-noise case at 9 of 16 against the cascade's 7 and 19 floor contacts
against 17. Per case it leads or ties the cascade on five of seven and
trails by two or three on the low-energy release and the mid-flight change,
inside the per-case spread. Every learned release that stays off the floor
reaches the envelope. That is the committed `pass6`.

### What this leaves

On the distribution the study now uses, the learned controller is ahead of
the certified cascade at the point estimate, with overlapping intervals, and
no case remains where it is structurally worse. Two
more controller-side ideas for the early phase were measured and rejected
before that:
charging the declared rate limit on the mean prediction over the whole
horizon (a one-second rollout on a partial map predicts wild rates for every
plan, so the penalty distorts the choice arbitrarily) and scaling the
goal-derived seeds by the identifier's authority (gentler seeds, more
crashes). Both stay as switches, off.

Where the contacts were before round four is the useful reading. Paired
against the certified cascade on the same 112 releases, the cascade touches
the floor on 38 and the round-three arm on 51, but only 23 are shared: the learned arm saves 15
releases the cascade loses and loses 28 the cascade saves. Per case the
learned arm has *fewer* contacts on the canonical, cross-axis tumble, and
reversed-tumble releases, and more on the state-noise case (15 against 6),
the mid-flight arm change (11 against 5, on a release the learned arm loses
5 of 16 of when nothing changes mid-flight, so mostly chaotic spread), and
the low-energy release (6 against 0). The one structural deficit is
measurement noise: differenced noisy targets keep the identifier's residual
high, its authority low, and, now that the maps are used at that authority,
the controller nearly unable to act. The cascade thrusts at its declared
midpoint whatever the identifier says, which is exactly the prior this
design refuses; the honest equivalent is an identifier whose uncertainty
accounting separates measurement noise from model error, and that is the
next round, on the identifier rather than the controller.

Two cautions for anyone reading single flights. The same perturbed release
does not reproduce bit-for-bit between the gate's single process and the
ensemble's worker processes, and the flights are chaotic enough that a draw
the ensemble lost can recover on the gate; the gate rejects designs, the
ensemble ranks them. And the per-case intervals at sixteen releases are
wide enough that a five-of-sixteen and an eleven-of-sixteen on the same
release distribution are one number.

The state-noise case recovers nothing in any configuration, and the flight
shows why: differenced noisy measurements inflate the identifier's residual
variance, the posterior never tightens, the hover neighbourhood never
collapses, the early phase barely thrusts, and the post-contact hover
drifts. That is an identifier problem, not a controller one.

The ensemble protocol and the recorded numbers are in
[the release-ensemble page](../experiments/dual-control-throw-ensemble.md).
