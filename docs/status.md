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
| Experimental direct readout | A fresh per-episode frozen feature map with a regularized linear readout produces much faster complete updates and better local accuracy. It is **not in production** and does not yet supply the public offline/update workflow. [SO(3) result](readout-physical-so3.md); [endpoint correction](readout-trajectory-head.md); [path correction](readout-trajectory-path.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The frozen [five-checkpoint path screen](readout-trajectory-path.md) tested
whether fitting an entire recent physical trajectory, rather than only its
endpoint, would constrain the next forecast. It failed: first-origin fixed-wing
81 250 ms error became **123.75 m/s and 153.34 rad/s**, versus **1.90 m/s and
2.14 rad/s** under the endpoint-only correction. The hard quad remained very
poor at **164.62 rad/s**, versus **102.74 rad/s** under the direct readout. The
completed-window objective improved despite these unseen forecast losses.
All eight fits, 126 updates, physical metrics and correction algebra verify
from saved data. The candidate is rejected and production source unchanged;
[the index](readout-trajectory-path.json) preserves the result.

The prior [endpoint correction](readout-trajectory-head.md) achieved large
fixed-wing cold-start gains but made the hard quad worse. The present result
shows that simply adding checkpoints to the same teacher-forced correction is
not enough. The remaining gap is the structure of the recursive model, not
another acceptance guard or strength sweep.

The [paired 10/50 ms quad fixture](paired-quad-sampling.md) established that a
50 ms observation interval alone does not cause the earlier fixed-wing angular
blow-up on its mild quad trajectory. It does not separate interval from update
count/history parameterization or test comparable aerodynamic excitation.

## Next iteration

Target **a lean, reusable benchmark loop** before another architecture idea.
The last model change was small but required copying and editing more than
500 lines of one-off evaluation code; the actual eight-case fit took seconds.
Keep a fixed compact physical-result table and one shared runner. Use a small
diagnostic subset for rapid failure and run the broader frozen suite for a
promising candidate. Preserve causal timing and source identity without
rebuilding the harness each time. Then return to one elegant recursive model,
not a stack of guards or candidate branches. A fresh held-out condition and a
separate Throw controller trial remain necessary for broader claims.
