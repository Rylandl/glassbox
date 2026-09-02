# Dual-control NMPC for an unseen multirotor

**Status: design under exploration; the second pass recovers three of the
seven study cases from zero information with no vehicle numbers.** This
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
