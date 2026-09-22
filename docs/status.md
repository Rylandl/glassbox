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

## Latest experiment: fast readout without pretraining

The [bounded cold-start screen](cold-readout.md) completed **382 updates per arm**
on six known recordings. Both arms used identical fresh episode-prefix
initialization, with no pretrained weights. Updating only the acceleration
readout via measured-increment RLS took **0.36–0.41 ms for quads / 0.264–0.266 ms
for fixed wings**, versus **34.1–36.1 ms / 7.71–7.74 ms** for the current learner.
Quad one-step error fell **80.4%**, but fixed-wing one-step error nearly doubled;
250 ms primary error was **1.95× / 278× worse**. **Rejected; production unchanged.**

The local fitting objective improved sharply despite collapsing recursive
forecasts. This screen changed both the optimized weights and the objective;
it does not show that a pretrained representation is necessary. Known quad
scoring begins 1.25 s after release, including 25 actuated initialization samples;
no immediate recovery or successful catch was tested. Full results, every case,
startup costs and limitations remain in the report. No broad offline fitting
or controller campaign followed this clear loss.

## Next iteration

**Isolate readout-only adaptation under the existing recursive objective.** Keep
current reconditioning, curvature prior, trust/backtracking and the 16-PCG budget;
freeze only the feature/filter/accumulator functions after fresh episode
initialization. Compare the full learner with this trainable-subset ablation on
the same bounded causal roster. This separates loss/safeguards from the need to
learn nonlinear features before another estimator design. Count every update
operation and initialization observation; inspect one-step and 250 ms forecasts.
Freeze the actual protocol and implementation before running it.

No fleet-trained features, class priors, reused normalizers or fitted revisions
may enter either arm. Representation changes, if needed, must learn from the
same episode. No pretraining-based rescue. Solver conditioning, earlier usable
predictions, calibrated uncertainty and controller recovery remain separate gaps;
controllers stay in Dart/Throw. User direction is to move quickly and stop
unpromising candidates before broad qualification.
