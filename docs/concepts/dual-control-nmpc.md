# Dual-control NMPC for an unseen multirotor

**Status: design under exploration.** This describes an experimental controller
being built in `glassbox.experimental`. Nothing here is part of the stable API,
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

## Out of scope for this pass

Actuator lag, forgetting or process noise for tracking configuration changes
(the posterior currently never forgets), integration into the maintained NMPC
model families, and any real-time claim. Each is a follow-up once the
feasibility question is answered.
