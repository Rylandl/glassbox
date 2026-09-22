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
| Experimental direct readout | A fresh per-episode frozen feature map with a regularized linear readout produces much faster complete updates and better local accuracy. It is **not in production** and does not yet supply the public offline/update workflow. [SO(3) result](readout-physical-so3.md); [trajectory correction](readout-trajectory-head.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The frozen [trajectory-correction screen](readout-trajectory-head.md) applied
one causal, low-rank physical 250 ms endpoint correction to the fast SO(3)
readout, including the initial prefix. At the first scored origin, fixed-wing
250 ms rate errors fell **8.96 → 0.34** and **22.06 → 2.14 rad/s**, and velocity
improved. But the truncated hard quad worsened **102.74 → 163.72 rad/s**, with
velocity also worse. At the next scored origin, most effects were small or
mixed. This is a real architecture effect and a real extrapolation failure:
improving the recent completed window did not control the next recursion.
The eight fits, 126 causal updates, corrections and physical metrics verify
from saved data. Warm complete-update medians were about 1.6 ms on fixed wing
and 4.1–4.5 ms on 10 ms quads; initial compilation may take seconds. The
[artifact index](readout-trajectory-head.json) preserves both the incomplete
harness-bug attempt and the corrected run. Production source is unchanged.

The prior [SO(3) readout result](readout-physical-so3.md) showed that local
attitude-to-rate sensitivity alone cannot constrain full recursion: its
fixed-wing median five-step attitude-to-rate gains were 162/266 versus the
plant's 3.9/3.9. The trajectory correction cures much of the initial
fixed-wing forecast but does not solve the hard-quad long-horizon instability.

The [paired 10/50 ms quad fixture](paired-quad-sampling.md) established that a
50 ms observation interval alone does not cause the earlier fixed-wing angular
blow-up on its mild quad trajectory. It does not separate interval from update
count/history parameterization or test comparable aerodynamic excitation.

## Next iteration

Target **trajectory-path fitting** in the same generic fast readout. The last
screen corrected only the completed 250 ms endpoint; the hard quad's next
forecast separated after 150 ms despite a better recent 50 ms fit. Freeze one
path-wide physical objective and the same 50–250 ms unseen-forecast comparison
before fitting. Test whether it retains the fixed-wing cold-start gain while
reducing the hard-quad divergence and measuring complete-update cost. Do not
adopt the fast readout into the single public learner from these known-stream
screens alone. A fresh demanding held-out condition and a separate Throw
controller trial remain necessary for broader claims.
