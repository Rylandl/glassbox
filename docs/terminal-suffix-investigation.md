# Terminal suffix feasibility with actuator lag

The chosen six-command suffix repairs the known fifth-request forecast violation
without changing support, uncertainty, cost weights, or the 0.6 s horizon. This is
an offline feasibility result in a changed command parameterization. It does not
establish improved production recovery or recursive feasibility.

## Two approaches and the choice

| Approach | Formulation and assumptions | Decision |
| --- | --- | --- |
| Optimize a suffix inside the existing horizon | Freeze the first 24 exact shifted commands and optimize the remaining six independently, using the complete vehicle/actuator state and existing nonlinear robust constraints. This asks whether earlier commands can repair the appended endpoint. It supplies no invariant terminal set. | Prototype once, six commands fixed before execution. |
| Augmented-state terminal equilibrium | With augmented state `z=(x,a)`, impose `F(z_N,u_s)=z_N`, with feasible terminal state and command. The actuator component requires consistency between applied actuator state and steady command; terminal vehicle state alone is insufficient. This is a specialization of an optimized terminal fixed-point constraint. | Defer. Reachability within 0.6 s and a robust continuation argument for the refreshed covariance have not been established. |

[Fagiano and Teel, *Generalized terminal state constraint for model predictive
control*](https://fagiano.faculty.polimi.it/docs/papers/GenMPC_Automatica_2013.pdf)
formulate an optimized terminal state/input fixed point for a discrete-time
nonlinear model (Section 3). Their nominal shift argument relies on that fixed
point satisfying the same model and constraints. Applying the idea here requires
including actuator state and addressing changing forecast uncertainty; the paper
does not certify this learned stochastic formulation. The suffix experiment is
our diagnostic restriction of the existing NMPC problem, not that paper's
terminal-equilibrium algorithm.

## Exact experiment

At the freshly reproduced failed request `tick4`, let `v` be the previous plan
shifted by one sample with its last command repeated. Solve

```text
minimize_u  J_original(rollout_commands(u, x4, a4), reference, previous_command)
subject to u[k] = v[k],                    k = 0,...,23
           command_min <= u[k] <= command_max, k = 24,...,29
           original_robust_margin[j] >= 0, every original stage and feature.
```

The solver accepts only finite trajectories with minimum margin at least `-1e-6`
and exact command bounds. The objective is the original sum of squared residuals,
including uncertainty, terminal, command-change, and soft safety terms. Safety
limits retain their existing soft meaning. It computes parameter and forecast
covariance through `rollout_commands`; it never resets the actuator state at the
suffix boundary. The fitted actuator time constant is 0.0791904 s, so the chosen
0.12 s suffix spans about 1.52 time constants. No extra forecast stage is added.

There are 24 optimization variables, compared with 40 in the baseline's ten
four-command blocks. These are different spaces: the suffix has finer temporal
freedom but cannot alter the first 24 commands. One initial point (the held-tail
shift), one SLSQP call capped at 100 iterations, analytic JAX derivatives, and
`ftol=1e-9` were prescribed. The script retains the cheapest feasible evaluated
candidate, independently recomputes its physical waveform forecast, and checks
that its frozen prefix is bitwise unchanged. SLSQP success alone is insufficient
for acceptance. Production code and defaults are untouched.

## Results and falsification

The [recorded result](investigations/terminal-suffix.json) reproduces initial
maximum utilization 1.277238607 and returns a feasible maximum 0.980752945.
The unchanged objective falls from 44.851501 to 28.630417. The initial repair
takes 50 SLSQP iterations, 68 distinct value evaluations, and 100 derivative
calls. Commands change by as much as the full normalized physical command range
(1.0); this is a substantial suffix maneuver, not a small final-sample correction.
The final command `[0, 0.33512, 1, 0]` differs substantially from terminal actuator
state `[0.25900, 0.32392, 0.59292, 0.19847]`. Feasibility is established with that
lag present, not by pretending the actuators immediately follow commands.

The initial ten-interval continuation (0.2 s) used exact shifted waveforms,
with no block averaging, and checked the full nonlinear forecast again. All ten
forecasts were feasible; peak actual support utilization was 0.961492419. All ten
applied commands still belonged to the initial frozen prefix, so that initial
observation alone could not show a realized effect of the repair.

The continuation was extended once to 36 intervals (0.72 s), with
identical suffix length, objective and optimizer configuration. The final
artifact supersedes the ten-step artifact and records both run limits. All 36
restricted OCPs and executed intervals passed. The first repaired suffix command
was applied at continuation tick 24 (0.48 s after the repair, absolute simulation
time 0.56 s); the script asserts equality with the initial repaired command at
index 24. Twelve applied intervals originated in the optimized suffix. Peak
actual support remained 0.961492419 and final actual utilization was 0.586114645.
The repaired initial endpoint utilization was 0.498250723; the whole-horizon
peak 0.980752945 remained in the retained prefix. Across all 36 OCPs the SLSQP
iteration range was 22–50. Forecast feasibility and actual-state support thus
both survive this short continuation, including application of suffix commands.
This still establishes neither recovery completion nor production warm1
improvement, terminal invariance, or recursive feasibility under later requests.

Three focused tests pass: a lagged scalar system where even the strongest final
command cannot repair the endpoint but the six-step suffix can; an unrepairable
frozen-prefix violation that must be rejected; and rejection of nonfinite values,
bound violations, and nonlinear margin violations. The first test keeps a
nonzero uncertainty reserve. These tests establish the diagnostic's acceptance
semantics, not a vehicle viability proof.

No suffix-length, objective-weight, start-point, or solver-budget sweep was run.
The initial ten-interval experiment was rerun once after adding report fields
and independent trajectory-finiteness checks; its numerical result was unchanged.
The 36-interval extension then ran once. Fixture and dependency
fingerprints and persistent negative-stop reporting were added afterward without
changing the numerical formulation; the artifact preserves the executed script
hash separately from the final source fingerprints. No latency
qualification was attempted while other agents were active.

## Reproduction and next decision

Build the shared fixture once, then run the suffix probe:

```sh
uv run python scripts/build_nmpc_research_fixture.py \
  --output /tmp/glassbox-nmpc-fixture
uv run python scripts/investigate_terminal_suffix.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/glassbox-terminal-suffix.json
uv run pytest tests/test_terminal_suffix_investigation.py -q
```

The recorded import resolves to the research checkout's `src/glassbox`, and the
fixture manifest identifies baseline `4f5cddb`. The commands rerun against current
code; the artifact retains its original dependency fingerprints. The known extension is repairable
when commands begin changing before its final sample. The next decision is
whether to investigate incorporating such suffix freedom and feasible-candidate
retention into the primary NMPC formulation with a realistic bounded solve
budget. The current result warrants that formulation question; it does not
warrant replacing fixed-grid warm2, claiming terminal invariance, or claiming
that 20 ms deadlines are met.
