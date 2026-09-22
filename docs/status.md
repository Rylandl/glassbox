# Current state

Updated 2026-09-22. Read [the charter](charter.md) first.

The **shared-physics learner with compact nonlinear history and stable
accumulators** is the single maintained implementation. Each configuration gets
its own fit from motion and issued commands; no vehicle family, layout, mass or
inertia is supplied. The public API remains `fit`, `predict`, immutable `update`,
save/load and bounded `OnlineFit`.

| Area | Evidence and remaining gap |
| --- | --- |
| Architecture | Full 100 ms linear lag path; quadratic current-feature head; 32 nonlinear units receiving up to four fixed orthonormal temporal summaries; eight learned stable accumulators; shared gravity, frames and rigid-body mechanics. No catalog or tuning menu. |
| Model size | Four commands / 10 ms: **8,714 → 5,450 parameters (37.46% fewer)**. Three commands / 50 ms: unchanged **3,103**. Dimensions follow recording timing, not vehicle type. |
| Online accuracy | Six known streams / 3,137 causal updates: effectively equal to the previous full-history model (**0.009% lower** equal-family velocity/rate error). After the first 100 updates, quad error is **2.80% higher**. Fixed-wing predictions match exactly. No demonstrated online learning-efficiency gain. |
| Runtime | Exclusive paired whole updates: quad medians **31.9 → 33.9 ms**, **6.10% slower**; p95 **6.39% slower**. Fixed-wing medians remain about **7.6 ms**. Smaller parameter count did not yield a CPU speedup. Other hardware and real-time fitting remain unqualified. |
| Offline accuracy | Matched fresh 1,000-attempt fits on both architectures. Across the known 8,064-query flight roster, equal-family forecast error is **2.08% higher**, command-response error **2.18% lower**. In the changed Crazyflow model alone: **4.20% higher / 4.31% lower**. Cascade is exactly unchanged. All frozen offline checks pass. |
| Dart | Held-out forecast velocity/rate error falls **7.50%, 14.99%, 13.69%, 9.13%** at 10/250/600/1,200 ms. At 250 ms: **0.0960 m/s**, **1.157 rad/s**, **0.0998 rad** orientation RMSE. No new controller trial. Earlier full-history accumulator **3.669 mm**, fresh v8 **3.597 mm**, and refined historical v8 **0.720 mm** are separate revisions, not compact-model task results. |
| Derivatives and persistence | All **40** offline finite-difference directions pass. All **17** saved prediction/availability arrays replay exactly, also after metadata-only recipe finalization. **488** full-suite cases pass across the initial run and corrected-fixture rerun; **25** installed-wheel checks pass. Temporal archives replace accumulator archives; use historical source for old models/sessions. |
| Calibration and scope | One seed and known configurations; development selects the fit and calibrates envelopes. Independent coverage, broad configuration generalization, long-horizon fidelity and live online controller recovery remain open. At 1.2 s, compact Dart errors are still **8.03 m/s / 13.23 rad/s**. |

## Adoption decision

Adopt nonlinear-only history compression as an overall architectural tradeoff:
substantially fewer learned coefficients, improved physical command responses
and Dart forecasts, essentially unchanged aggregate online accuracy, and a modest
forecast/runtime cost. This is not an unconditional improvement or a portable
speed claim. Preserve the failed online speed target and late-stream loss; an
isolated regression is not a veto. Full linear delayed response remains available;
nonlinear interactions in the discarded temporal components do not.

The [temporal report](nonlinear-temporal.md) and [artifact index](nonlinear-temporal.json)
record protocols, every fit, physical scores, initialization controls, numerical
checks and archive provenance. Only the adopted model remains in the package.

## Correction to earlier evidence

The older **4.11% online improvement versus v8** applied before the history-projection
refactor. A fresh full-stream baseline in this iteration exposes **2.69% higher**
aggregate error after that refactor, mainly one fixed-wing recording; its earlier
snapshot comparison did not establish trajectory equivalence. Small rounding
changes can accumulate through the online optimizer. Against the original v8
predictions, the current compact model improves aggregate error **1.54%**, while
rate error is **0.43% higher**. These are saved-data accuracy comparisons, not new
paired timings or blind generalization tests. Earlier frozen results remain intact.

## Latest experiment: fast readout with physical curvature regularization

The [regularized readout experiment](cold-readout-curvature.md) is promising but
**not adopted**. It starts fresh on every episode with no pretraining, freezes
feature/filter/memory functions, and solves readout coefficients directly from
measured increments plus the existing generic physical curvature penalty.
Production code and public interfaces remain unchanged.

The short screen completed **382 updates per arm** against both the full learner
and raw RLS. One-step primary error fell **79.0%**, 250 ms error **25.1%**; both
controls reproduced exactly. The same unchanged candidate then completed
**3,137 updates per arm** against the full learner on the entire recorded roster:

| Candidate / full learner | Quad | Fixed wing | Equal-family aggregate |
| --- | ---: | ---: | ---: |
| One-step velocity/rate error | 0.1769 | 0.1547 | **0.1654** |
| 250 ms velocity/rate error | 0.6719 | **2.0600** | 1.1765 |
| Warm median update time | 0.0299 | 0.0702 | **0.0458** |

Warm complete updates take **1.006–1.035 ms for quads / 0.535–0.540 ms for fixed
wings**, about **22× faster** overall. After the four-second configuration change,
quad one-step primary error is **40.2% lower**, though that segment is already
near hover with narrow command variation. Both arms' short prefixes reproduce
exactly in the full run. Nine focused tests pass; all saved solves pass the
componentwise equation audit, with maximum backward error **2.91e-15**.

All frozen aggregate flags pass. Nevertheless, full-recording fixed-wing 250 ms
body-rate errors rise from **2.70 to 15.06 rad/s** and **14.52 to 39.98 rad/s**.
A posthoc query view shows losses on 20 of 26 noninitial fixed-wing forecasts.
The local fitting gain persists after 100 updates; it does not yet translate into
reliable recursive trajectories. This is a consistent capability gap alongside
a substantial architectural gain, not a reason to discard the fast approach or
to call it production-ready. Absolute forecast errors in the truncated quad tape
remain large too.

The [index](cold-readout-curvature.json) preserves both protocols, every physical
score, timing, startup cost, source/runtime binding, and the no-fit audits. The
experimental source/tests remain in Git (`be38729`, `6d2b9c4`), not as a second
maintained implementation. The [raw RLS](cold-readout.md) and
[matched recursive-loss](cold-readout-rollout.md) reports establish why curvature
and objective constraints were tested; neither requires a pretrained core.

## Latest diagnosis: excessive linear angular feedback

The [saved-model investigation](readout-stability.md) supports continuing the fast
readout architecture, with a more specific structural target. Across every saved
fixed-wing forecast origin, angular error compounds through 50–250 ms. Replacing
the candidate angular output with the full learner reduces 250 ms rate error to
**2.77 / 18.24 rad/s**, versus **15.06 / 39.98** for the candidate.

Removing quadratic rate coefficients worsens the first case and only modestly
helps the second (**19.71 / 35.39 rad/s**). Removing the **linear lag-difference
rate rows** helps both much more (**6.94 / 14.76 rad/s**), with local accuracy and
velocity tradeoffs. This is a diagnostic ablation, not an adopted replacement.
Motion features already saturate before products; 50 ms integration already uses
two 25 ms midpoint substeps.

The full 50-dimensional recurrence Jacobian has median spectral radius about
**3.3–3.4** on measured states, versus **1.3–1.4** for the reference. Instantaneous
angular-rate feedback has median maximum real eigenvalue **+13.04/s / +13.59/s**;
negative candidate derivatives do not support stiff damping as the main failure.
Leverage spikes correlate with error, but are not calibrated uncertainty. Neither
partial Jacobians nor these local spectral radii constitute a stability proof.

Eight distinct diagnostic tests pass; saved arrays and scores verify without
fitting. Candidate weights were reused throughout. Missing intermediate reference
models required exact replay; a harness reuse bug caused one redundant replay:
**900 reference updates versus 450 planned, zero candidate fits**. All attempts and
the metadata correction are in the [index](readout-stability.json). Production is
unchanged. Completed experimental code is archived through `da3123c` in Git.

## Latest experiment: first-order motion sensitivity

The [single-strength sensitivity experiment](readout-sensitivity.md) completed
**3,137 updates per arm** on the full roster. Analytic feature derivatives add one
quadratic penalty to the shared readout solve. Against the previous fast readout,
aggregate 250 ms error falls **8.65%**, one-step error rises **3.30%**, and complete
updates cost **21.31% more**. Warm medians are **1.292–1.331 ms quad / 0.561–0.565 ms
fixed wing**, still **18.45× faster** than the full learner overall.

Fixed-wing 250 ms angular error changes **15.06 → 12.25 rad/s** and **39.98 → 42.00
rad/s**. All six 250 ms velocity forecasts improve, while angular changes are mixed.
The frozen family primary/rate flags still fail; aggregate flags pass. Production
remains unchanged. This is a measured tradeoff, not a resolved recursive learner.

The decisive diagnostic is that median instantaneous angular feedback falls
**+13.04/s → +4.03/s / +13.59/s → +3.24/s**, but full recurrence radius stays near
**3.4**. The physical-state partial block remains near **3.5–3.7** even with latent
history fixed. Thus the earlier lag ablation does not justify blaming history
alone. The tested derivative perturbs velocity/rate features and omits attitude;
it is an incomplete surrogate for coupled physical-state feedback.

Five focused tests pass. Both controls replay exactly; all 3,137 candidate solves
verify from saved data with maximum componentwise backward error **3.22e-15**.
One fitting run, no repeated attempts or strength sweep. Two saved-model audits
add no fitting. The [index](readout-sensitivity.json) retains protocols, all
physical scores, timing and provenance. Experimental code is archived through
`17e4921` in Git.

## Latest diagnosis: plant-referenced attitude attribution

The [no-fit attribution](readout-attribution.md) reuses every saved fixed-wing
forecast origin and reconstructs Cascade's complete actuator/aerodynamic state
through near-exact replay. Its median local attitude-to-rate derivative is about
**2.9** in the plant, **56/59** in the sensitivity readout and **209/379** in
the maintained full learner (recordings 80/81). The corresponding 250 ms
attitude-to-rate gains are about **3.9**, **6,912/6,275** and **468/1,933**.
These are physical tangent derivatives, not forecast errors or global stability
proofs. Both learned models have a large plant-referenced sensitivity mismatch;
the fast head amplifies it most across five steps.

In the latest readout, replacing only learned attitude-to-rate Jacobian rows by
known-propagation rows lowers median five-step attitude-to-rate gain to
**0.102/0.143** of original on rolled states, at every origin in both cases.
Removing latent-to-physical feedback instead usually raises the gain. An exact
instantaneous-head chain rule finds the omitted body-gravity-direction feature
is the larger attitude-derivative path in both recordings; body-relative
velocity also contributes. This localizes the next architecture test without
deleting history, making a vehicle branch or using simulator truth in the fit.
No new fit or online update occurred. Three authenticated packs verify from
saved arrays. The first no-fit verifier stopped because a stop-gradient
derivative cannot be checked by ordinary finite differences; its corrected
criteria and attempt are retained in the [index](readout-attribution.json).

## Latest experiment: paired quad observation schedules

The [paired 10/50 ms quad experiment](paired-quad-sampling.md) removes the
different-trajectory and command-timing confound: one ten-second controlled
Crazyflow flight supplies exactly shared physical states and commands held for
each 50 ms interval. The unchanged full learner, curvature readout and
sensitivity readout each start fresh and receive the same elapsed prefix; 875
versus 175 causal updates are scored at 35 common origins. A first collection
attempt stopped at 1.95 s after floor contact and remains in the [index](paired-quad-sampling.json).

At 50 ms, the sensitivity readout's 250 ms velocity/rate/orientation RMSE is
**0.00339 m/s / 0.00442 rad/s / 0.000482 rad**; at 10 ms it is **0.00164 /
0.00510 / 0.000528**. No fixed-wing-like angular explosion occurs at 50 ms.
Thus 50 ms sampling alone is insufficient to cause the prior failure in this
regime. The flight is gentle (maximum rate 0.171 rad/s), and sample interval,
update count and history size still change together. One such trajectory does
not establish aircraft-class robustness.

The full learner's recursive velocity error is much larger here despite modest
one-step error: **1.449 m/s** at 250 ms on the 10 ms tape and **0.821 m/s** on
the 50 ms tape. Warm complete updates are **33.47/8.44 ms** versus the
sensitivity readout's **1.324/0.654 ms**. Saved arrays and physical metrics
verify without fitting. Production remains unchanged.

## Next iteration

Freeze **one generic physical SO(3) sensitivity fit** for the fast readout.
An attitude perturbation must change body-relative velocity and gravity
direction together; penalize the six physical acceleration-output derivatives
under these directions while preserving the shared direct solve. Keep both fast
readouts and the full learner as controls. Compare absolute physical velocity,
rate and orientation errors at every 50–250 ms horizon, per-family flags,
initialization and whole-update cost. A local derivative reduction alone will
not qualify it. If realized rollout does not improve, test a bounded short-
trajectory fitting correction, given the previous recursive-loss control.
Neither path introduces a fleet prior, system-specific branch or global
contraction assumption. Cold-start scoring still begins only 1.25 s after
release on the quad tapes; controller recovery and calibrated error envelopes
remain separate unmet criteria.
