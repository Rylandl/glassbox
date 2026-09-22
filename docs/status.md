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

## Next iteration

**First-order state/history sensitivity regularization for the fast readout.**
Penalize the effective instantaneous and delayed motion feedback, including the
linear path that the current curvature penalty cannot constrain. With frozen
features this remains quadratic in the readout and can preserve the shared direct
solve. Freeze generic coordinate scales, probe domain, penalty strength, data
budget and complete-update measurement before implementation. Keep the existing
curvature penalty and other estimator choices fixed. Use one generic formulation
across all acceleration outputs; do not force all plants to be globally contractive.

Compare one-step accuracy and full 250 ms forecasts against both the unchanged
readout and full learner. Report velocity and rate separately, with the future
per-family flags declared in the diagnostic protocol. Include all derivative/
precision construction in update cost. Soft sensitivity regularization can bias
legitimate dynamics and is unproven; a retrospective acceptance guard remains a
possible safety net rather than the primary next change.

No fleet-trained features, class priors, reused normalizers or fitted revisions
may enter any arm. Every learned quantity must come from that episode. Family
and sample interval remain confounded (10/50 ms), as do initialization counts.
A matched quad tape with commands held for 50 ms is still needed to isolate the
observation interval; naive decimation with changing commands does not do that.
Cold-start availability still includes 0.5 s history plus 0.25 s initialization data;
the quad tapes begin scoring 1.25 s after release. Warm-update speed is not live
catch qualification. Counterfactual response, sensor-noise robustness, calibrated
uncertainty and controller recovery remain separate gaps in Glassbox/Dart/Throw.
