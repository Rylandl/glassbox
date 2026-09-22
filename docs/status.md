# Current state

Updated 2026-09-22. Read [the charter](charter.md) first. The shared-physics
learner with compact nonlinear history and stable accumulators is the **single
maintained implementation**. It fits each configuration from that episode's
motion, issued commands and timing; no vehicle family, mixer, mass or inertia
is supplied. Public `fit`, `predict`, immutable `update`, save/load and bounded
`OnlineFit` remain intact. The Throw requirement excludes any pretraining.

| Area | Current evidence and limit |
| --- | --- |
| Maintained model | Generic gravity, rigid-body kinematics and frames; learned command response, accelerations and delayed/hidden dynamics. Four-command/10 ms model has 5,450 parameters; three-command/50 ms has 3,103. [Architecture evidence](nonlinear-temporal.md). |
| Offline and Dart | Matched offline fits changed equal-family forecast error +2.08% and command-response error −2.18% versus the prior full-history model. The compact-model Dart forecast improved at 10–1,200 ms, but no new controller trial established its catch performance. [Results](nonlinear-temporal.md). |
| Maintained online learner | Six recorded streams/3,137 causal updates were effectively equal in aggregate accuracy to the former full-history learner, but the compact model did not speed whole CPU updates. Quad median was 33.9 ms. No demonstrated real-time fitting or held-out vehicle calibration. [Results](nonlinear-temporal.md). |
| Experimental direct readout | A fresh per-episode frozen feature map with a regularized linear readout produces much faster complete updates and better local accuracy. It is **not in production** and does not yet supply the public offline/update workflow. [Earlier screen](readout-sensitivity.md); [latest SO(3) result](readout-physical-so3.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The frozen [physical SO(3) readout experiment](readout-physical-so3.md) added
one generic attitude-feature sensitivity term to the previous fast readout and
completed **4,187 candidate updates** across eight recordings/schedules.
Fixed-wing 250 ms body-rate RMSE improved **12.25 → 2.59 rad/s** and **42.00 →
6.03 rad/s**, with velocity also improving. The latter is still a material
absolute error. Four earlier quad recordings improved at 250 ms; the truncated
quad case remains extremely poor at **62.87 rad/s**. The gentle paired 10/50 ms
quad results changed little. Historical warm complete-update medians are
roughly 0.6–1.5 ms for the candidate on this CPU, somewhat slower than the
previous fast readout but still much faster than the maintained full learner.
All eight candidate fits, direct solves and physical metrics verify from saved
data. No fleet prior, vehicle metadata or system-specific branch was added.
Production source is unchanged.

Plant-referenced tangent data prevent overclaiming the result. The new model's
local fixed-wing attitude-to-rate derivative is **0.85/0.99** versus the
Cascade plant's **2.91/2.90**: the head is now locally too insensitive. Its
median five-step attitude-to-rate gains are **162/266** versus the plant's
**3.9/3.9**. The previous fast readout's gains were **6,912/6,275**. Thus the
new penalty sharply improves the known rollouts but does not align the complete
recurrence with the plant. The initial scored fixed-wing forecasts still have
**8.96/22.06 rad/s** 250 ms rate error before the first post-prefix update.
The [no-fit audit](readout-physical-so3.md) and [artifact index](readout-physical-so3.json)
preserve exact evidence and limitations.

The [paired 10/50 ms quad fixture](paired-quad-sampling.md) established that a
50 ms observation interval alone does not cause the earlier fixed-wing angular
blow-up on its mild quad trajectory. It does not separate interval from update
count/history parameterization or test comparable aerodynamic excitation.

## Next iteration

Target **physical rollout fitting, including the causal prefix**, for the
same generic fast readout. Freeze a bounded short-trajectory objective and
physical 50–250 ms evaluation before fitting; retain the direct solve where
possible and avoid another local-penalty strength sweep. Compare absolute
velocity, body-rate and orientation errors on the eight known streams, plus a
fresh, more demanding held-out condition if available. Record first-origin
cold-start performance, full-recurrence plant mismatch and whole-update cost.
Only then decide whether to integrate the fast architecture into the single
public learner. A new controller claim requires a separate Throw trial.
