# Initialized-mean prior: bounded research experiment

This iteration addresses forecast generalization during fitting. Crazyflow training
rate errors fall while development rate errors grow from initialization, despite
the selected updated model outperforming its predecessor. Distribution differences
and objective tradeoffs remain alternative explanations.

Use one fixed prior strength, 0.25, with the exact 144-parent training cache and
original 24-parent development cache. From the existing initial-training prediction Z,
optimize J = mean(w*((F-Y)/s)^2) + 0.25*mean(w*((F-stop_gradient(Z))/s)^2).
The same rollout supplies both terms. Preserve original targets, hold-error scale s,
training-only fixed weights w, initialization, Adam mechanics and data-only development
checkpoint selection. This is an engineering choice, equivalent up to a constant
to a 1.25-scaled 80/20 truth/initial-reference target mixture, not estimated confidence.
The prior may preserve initializer bias or suppress useful command sensitivity.

Use a prepared-cache research fit. Public fit(all168) changes the development split;
update requires new recordings. Do not counterfeit a public operation or relabel a
saved v4 archive. The candidate owns a distinct experimental archive identity and
shares unchanged initialization, rollout, inference and archive primitives. Fit once
per simulator; no baseline refits, sweep or new training data. Both exact terminal candidate
outcomes must be sealed before fresh +13M confirmation collection. Only complete
or fit_failed permits confirmation; timeouts and integrity/preparation failures
abort it. An unavailable numerical-fit arm retains all planned truth and control
results and cannot pass residual criteria.

Four small experimental modules separate responsibilities: stdlib-only common source
and protocol authentication; the private bounded fitter/archive/capture adapter;
confirmation-only historical simulator collection/replay; and process orchestration,
matched physical scoring, decision and replay. Reuse maintained lower operators,
not a copied historical evidence framework. The old 1f46773 source-bound worker and
archives remain immutable. Baseline inference uses that worker; candidate inference
uses the new worker. Current learner/core and public consumer contract stay unchanged.

The new prospective advancement rule is sqrt(R_forecast*R_response)<=0.97 with each
kind<=1.05, retaining existing primary/scope/tail/angular guards. It allows material
gains in either kind or balanced smaller gains in both. It replaces the prior
experiment's response-only gain trigger for this experiment only. Report physical
before/after errors and all losses; no all-cell win or significance veto. A residual
pass is distinct from evidence that angular development drift actually shrank.

First freeze this protocol/roster/plan. Then implement and test the entire harness,
commit it and bind its clean source/runtime before fitting or collecting. Capture
actual initial reference, parameters, weights, objective components,11 checkpoints,
work and failures. Replay native confirmation, selected predictions, cache/checkpoint
predictions and scalar reductions without refitting, and reject deliberate altered
artifacts. A successful research comparison requires separate public qualification
before replacing the adopted recipe. Calibration, derivatives and control remain
separate claims. Record failed experiments and proceed to the next evidenced gap.
